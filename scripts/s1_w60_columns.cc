// Production CPU ABI for the frozen W55 FP64 axisymmetric potential and solve.
// One call advances a whole measured packet or an array of inverse-solve rows.
#include "s1_w55_c_column.cu"
#include <cstdint>

static void stress_pair(const Model& m, const double* state, double* stress) {
    double eq[2], over[2], p[2]={state[0]-state[2],state[1]-state[3]};
    elastic(m,m.eq,state,eq);
    elastic(m,m.ov,p,over);
    for(int i=0;i<2;i++) stress[i]=eq[i]+over[i];
}

static double mixed(const Model& m, const double* prev, double normal, double dt,
                    bool fixed, double* next, double* stress) {
    double first;
    double error=solve(m,prev,normal,dt,fixed,next,&first);
    stress_pair(m,next,stress);
    return error;
}

static bool accepted(double r,const double* x,const double* s) {
    return std::isfinite(r) && r<=1e-7 && std::isfinite(s[0]) && std::isfinite(s[1]) &&
        std::all_of(x,x+4,[](double y){return std::isfinite(y);});
}

extern "C" int w60_step(const Model* m,int n,const double* prev,const double* normal,
                        double dt,const uint8_t* fixed,double* out) {
    for(int i=0;i<n;i++){
        double* row=out+7*i;
        row[6]=mixed(*m,prev+4*i,normal[i],dt,fixed[i],row,row+4);
        if(!accepted(row[6],row,row+4)) return i;
    }
    return -1;
}

// row = e11,e22,q11,q22,s11,s22. settled is set only after a free transition
// returns a bit-identical state. Then the Newton residual at zero rate is
// independent of dt. A new contact can always wake that column.
extern "C" int w60_packet(const Model* m,int n,int ticks,const double* dt,
        const double* normal,const double* area,const int32_t* owner,double gamma,
        double* row,uint8_t* settled,uint8_t* loaded,double* foot_force,double* diagnostics) {
    for(int k=0;k<ticks;k++){
        for(int i=0;i<n;i++){
            const int j=k*n+i;
            double* x=row+6*i;
            double free[6],candidate[6],err=0.;
            std::copy(x,x+6,free);
            if(!settled[i]) {
                err=mixed(*m,x,0.,dt[k],false,free,free+4);
                if(!accepted(err,free,free+4)){diagnostics[3]=err;return j;}
            }
            bool active=false;
            double* selected=free;
            if(normal[j]<=free[0]) {
                double rc=mixed(*m,x,normal[j],dt[k],true,candidate,candidate+4);
                if(!accepted(rc,candidate,candidate+4)){diagnostics[3]=rc;return j;}
                err=std::max(err,rc);
                active=candidate[4]<0;
                if(active) selected=candidate;
            }
            settled[i]=!active && std::equal(x,x+4,selected);
            loaded[i]=active;
            if(active) foot_force[4*k+owner[j]]-=selected[4]*1e6*gamma*area[j];
            double* stats=diagnostics+4*k;
            stats[0]=std::max({stats[0],std::abs(selected[0]),std::abs(selected[1])});
            stats[1]=std::max({stats[1],std::abs(selected[4]),std::abs(selected[5])});
            stats[2]=std::max({stats[2],std::abs(selected[0]-x[0])/dt[k],std::abs(selected[1]-x[1])/dt[k]});
            stats[3]=std::max(stats[3],err);
            std::copy(selected,selected+6,x);
        }
    }
    return -1;
}

extern "C" int w60_recovery(const Model* m,int n,const double* initial,double* heights,double* failure) {
    for(int i=0;i<n;i++){
        double x[4],next[4],s[2];std::copy(initial+4*i,initial+4*i+4,x);
        bool stationary=false;
        for(int k=1;k<=250;k++){
            if(!stationary){
                double r=mixed(*m,x,0.,.002,false,next,s);
                if(!accepted(r,next,s)){*failure=r;return i;}
                stationary=std::equal(x,x+4,next);
                std::copy(next,next+4,x);
            }
            if(k==1 || k==100 || k==250) heights[i*3+(k==1?0:k==100?1:2)]=x[0];
        }
    }
    return -1;
}
