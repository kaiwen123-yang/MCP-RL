// W57 packet observation. The nominal propagation delegates to the exact r60 ABI.
#include "s1_w60_columns.cc"
#include <vector>
#include <numeric>

static bool admissible(const double *row, double residual) {
    return accepted(residual, row, row + 4) && std::abs(row[0]) <= .02 && std::abs(row[1]) <= .02;
}

// Evolve one column under the original unilateral displacement/free rule.
static bool advance_one(const Model& model, const double* prior, double normal, double dt,
                        double* result, bool& loaded) {
    double free[6], contact[6];
    const double rf = mixed(model, prior, 0., dt, false, free, free + 4);
    if (!admissible(free, rf)) return false;
    loaded = false;
    if (normal <= free[0]) {
        const double rc = mixed(model, prior, normal, dt, true, contact, contact + 4);
        if (!admissible(contact, rc)) return false;
        loaded = contact[4] < 0.;
    }
    std::copy(loaded ? contact : free, (loaded ? contact : free) + 6, result);
    return true;
}

struct Correction {
    const Model& model;
    const std::vector<int>& indices;
    const double *prior, *normal, *area;
    double dt, scale;
    std::vector<double> state;
    std::vector<uint8_t> loaded;

    // A shared signed duration parametrizes extra creep at the previous
    // compressive boundary, or zero-traction relaxation. The CURRENT measured
    // boundary is always re-applied with its true dt. No state interpolation.
    bool evaluate(double tau, double& force) {
        force = 0.;
        for (size_t j = 0; j < indices.size(); ++j) {
            const int i = indices[j];
            const double* before = prior + 6*i;
            double aged[6]; bool held, pressed;
            const double bound = tau > 0. ? before[0] : INFINITY;
            if (!advance_one(model, before, bound, std::abs(tau), aged, held)) return false;
            double* result = state.data() + 6*j;
            if (!advance_one(model, aged, normal[i], dt, result, pressed)) return false;
            loaded[j] = pressed;
            if (pressed) force -= scale * area[i] * result[4];
        }
        return true;
    }
};

// corrections[leg] = [prior force, innovation, min/max attainable force among
// evaluated admissible hypotheses, selected signed duration, updated force,
// eligible column count, number of explicitly rejected numerical/domain trials].
extern "C" int w57_packet(const Model* model, int n, int ticks, const double* dt,
        const double* normal, const double* areas, const int32_t* owners, double gamma,
        double* state, uint8_t* settled, uint8_t* loaded, double* forces, double* stats,
        uint8_t* active_trace, int correct, const double* measured, double* diagnostics) {
    const double packet_dt = std::accumulate(dt, dt+ticks, 0.);
    for (int k=0; k<ticks; ++k) {
        const auto* boundary = normal+k*n;
        const auto* area = areas+k*n;
        const auto* owner = owners+k*n;
        double* f = forces+4*k;
        const bool assimilate = correct && k==ticks-1;
        std::vector<double> before;
        std::vector<uint8_t> was_loaded;
        if (assimilate) {
            before.assign(state, state+6*n);
            was_loaded.assign(loaded, loaded+n);
        }
        const int bad=w60_packet(model,n,1,dt+k,boundary,area,owner,gamma,
                                 state,settled,loaded,f,stats+4*k);
        if (bad>=0) return k*n+bad;
        if (k==ticks-1) {
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
                Correction trial{*model,group,before.data(),boundary,area,dt[k],1e6*gamma,
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
        }
        std::copy_n(loaded,n,active_trace+k*n);
    }
    return -1;
}
