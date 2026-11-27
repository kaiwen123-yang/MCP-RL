// Offline W59 hybrid backend. Import the same Correction::evaluate, FP64 law and
// numerical acceptance predicates as the frozen host implementation. Only the
// interface to its last-tick assimilation differs: before/was_loaded and the
// nominal state/forces are now supplied by the completed CUDA packet.
#include "s1_w57_material_columns.cc"

extern "C" void w59_correct_last(const Model* model, int n, int ticks, const double* dt,
        const double* normal, const double* areas, const int32_t* owners, double gamma,
        double* state, uint8_t* settled, uint8_t* loaded, double* forces, double* stats,
        uint8_t* active_trace, int correct, const double* measured, double* diagnostics,
        const double* before, const uint8_t* was_loaded) {
    const int k = ticks - 1;
    const double packet_dt = std::accumulate(dt, dt + ticks, 0.);
    const double* boundary = normal + k*n;
    const double* area = areas + k*n;
    const int32_t* owner = owners + k*n;
    double* f = forces + 4*k;
    const bool assimilate = correct != 0;
    for (int leg=0;leg<4;++leg) {
        double* d=diagnostics+8*leg;
        d[0]=d[2]=d[3]=d[5]=f[leg];
        d[1]=measured[leg]-f[leg];
        if (!assimilate) continue;
        std::vector<int> group;
        double active_force=0., cover=0.;
        for (int i=0;i<n;++i) {
            if (owner[i]!=leg || owners[(k-1)*n+i]!=leg || !loaded[i] || !was_loaded[i]) continue;
            group.push_back(i);
            active_force-=1e6*gamma*area[i]*state[6*i+4];
            cover+=area[i];
        }
        d[6]=group.size();
        const double resolution=1e-7*1e6*gamma*cover;
        if (group.empty() || std::abs(d[1])<=resolution) continue;
        Correction trial{*model,group,before,boundary,area,dt[k],1e6*gamma,
            std::vector<double>(6*group.size()),std::vector<uint8_t>(group.size())};
        const double unchanged=f[leg]-active_force;
        double best_error=std::abs(d[1]),best_tau=0.;
        auto score=[&](double tau, double& prediction) {
            double updated;
            if (!trial.evaluate(tau,updated)) { d[7]++; return false; }
            prediction=unchanged+updated;
            d[2]=std::min(d[2],prediction);d[3]=std::max(d[3],prediction);
            double error=std::abs(measured[leg]-prediction);
            if (error+resolution<best_error) { best_error=error;best_tau=tau; }
            return true;
        };
        for (double direction : {-1.,1.}) {
            double endpoint;
            if (!score(direction*packet_dt,endpoint)) continue;
            const double span=endpoint-f[leg];
            if (std::abs(span)<=resolution) continue;
            const double fraction=d[1]/span;
            if (fraction>0. && fraction<1.) {
                double projected; score(direction*packet_dt*fraction,projected);
            }
        }
        if (best_tau==0.) continue;
        double updated;
        trial.evaluate(best_tau,updated);
        for (size_t j=0;j<group.size();++j) {
            int i=group[j];
            std::copy_n(trial.state.data()+6*j,6,state+6*i);
            loaded[i]=trial.loaded[j]; settled[i]=0;
            stats[k*4]=std::max({stats[k*4],std::abs(state[i*6]),std::abs(state[i*6+1])});
            stats[k*4+1]=std::max({stats[k*4+1],std::abs(state[i*6+4]),std::abs(state[i*6+5])});
        }
        f[leg]=unchanged+updated;d[4]=best_tau;d[5]=f[leg];
    }
    std::copy_n(loaded, n, active_trace + k*n);
}
