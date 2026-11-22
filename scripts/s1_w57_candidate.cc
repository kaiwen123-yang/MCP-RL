// Private force-driven footprints. The r60 model, Newton solve and hard domain are shared.
#include "s1_w60_columns.cc"
#include <array>
#include <numeric>
#include <vector>

namespace {
using State = std::array<double,6>;
struct Column {int id; double area;};
struct Sole {double delta, tangent, residual;};
constexpr double thickness=.05, strain_limit=.02, force_factor=.025e6;
struct MaterialFailure {int reason;};

State advance(const Model& model,const State& state,double value,double dt,bool fixed){
    State next;
    double err=mixed(model,state.data(),value,dt,fixed,next.data(),next.data()+4);
    if(!accepted(err,next.data(),next.data()+4))throw MaterialFailure{1};
    return next;
}
void domain(const State& state){
    if(std::max(std::abs(state[0]),std::abs(state[1]))>strain_limit+1e-14)throw MaterialFailure{2};
}
Sole invert(const Model& model,const std::vector<State>& state,const std::vector<State>& free,
           std::vector<State>& next,const std::vector<Column>& cells,double load,double dt,bool diagnostic){
    double min_delta=std::numeric_limits<double>::infinity(), area=0.;
    const auto& first=state[cells.front().id];
    bool same=true;
    for(auto c:cells){
        min_delta=std::min(min_delta,-thickness*free[c.id][0]);
        area+=c.area;same &= state[c.id]==first;
    }
    auto response=[&](double delta,bool save){
        double total=0.;
        for(auto c:cells){
            if(-delta/thickness>free[c.id][0]){if(save)next[c.id]=free[c.id];continue;}
            auto x=advance(model,state[c.id],-delta/thickness,dt,true);
            if(x[4]<0.){
                total-=c.area*force_factor*x[4];
                if(save){domain(x);next[c.id]=x;}
            }else if(save)next[c.id]=free[c.id];
        }
        return total;
    };
    double d=min_delta, residual=0.;
    if(load>0. && same){
        // Identical columns share a common strain. Prescribed traction gives the exact root.
        auto x=advance(model,first,-load/(area*force_factor),dt,false);
        domain(x);d=-thickness*x[0];residual=-x[4]*area*force_factor-load;
        for(auto c:cells)next[c.id]=x;
    }else if(load>0.){
        double lo=min_delta, hi=thickness*strain_limit;
        // k is a Newton initial guess only. The frozen law sets every accepted load.
        double slope=1.225e9*area;
        d=std::min(hi,lo+load/slope);
        double old_d=min_delta,old_force=0.;
        for(int i=0;i<48;++i){
            double f=response(d,true);
            residual=f-load;
            if(std::abs(residual)<=1e-7)break;
            if(i==47 || (d==hi && residual<0. && hi==thickness*strain_limit))throw MaterialFailure{3};
            if(residual>0.)hi=d;else lo=d;
            if(std::abs(d-old_d)>1e-15 && (f-old_force)/(d-old_d)>0.)slope=(f-old_force)/(d-old_d);
            old_d=d;old_force=f;
            double trial=d-residual/slope;
            if(trial>hi && hi==thickness*strain_limit)d=hi;
            else d=(trial>lo && trial<hi)?trial:.5*(lo+hi);
        }
    }
    if(std::abs(residual)>1e-5)throw MaterialFailure{4};
    double tangent=0.;
    if(diagnostic){
        const double step=d+1e-9<=thickness*strain_limit?1e-9:-1e-9;
        tangent=(response(d+step,false)-response(d,false))/step;
    }
    return {d,tangent,residual};
}
}

extern "C" int w57_candidate(const Model* model,int n,int count,const double* spans,
        const int32_t* subdivisions,const double* areas,const double* loads,
        const double* initial,double* results,double* states,int tangent_requested,double* detail){
    std::vector<State> state(n),free(n),next(n);
    std::vector<uint8_t> settled(n);
    for(int i=0;i<n;++i){
        std::copy(initial+6*i,initial+6*i+6,state[i].begin());
        settled[i]=std::all_of(state[i].begin(),state[i].end(),[](double x){return x==0.;});
    }
    for(int j=0;j<count;++j){
        const auto* aa=areas+j*4*n;
        const auto* force=loads+j*4;
        std::array<std::vector<Column>,4> footprints;
        bool overlap=false;
        for(int i=0;i<n;++i){
            int owners=0;
            for(int leg=0;leg<4;++leg)if(aa[leg*n+i]>0.){
                footprints[leg].push_back({i,aa[leg*n+i]});++owners;
            }
            overlap |= owners>1;
        }
        const double dt=spans[j]/subdivisions[j];
        for(int k=0;k<subdivisions[j];++k){
            try{
                for(int i=0;i<n;++i){
                    free[i]=settled[i]?state[i]:advance(*model,state[i],0.,dt,false);
                    domain(free[i]);
                }
                std::array<int,4> order={0,1,2,3};
                std::array<Sole,4> sole{};
                bool consistent=false;
                int last_reason=5;
                do{
                    next=free;
                    auto owned=footprints;
                    std::vector<int> owner(n,-1);
                    if(overlap){
                        for(auto leg:order){
                            owned[leg].clear();
                            for(auto c:footprints[leg])if(owner[c.id]<0){owned[leg].push_back(c);owner[c.id]=leg;}
                        }
                    }
                    try{
                        for(int leg=0;leg<4;++leg){
                            if(footprints[leg].empty())continue; // External rigid reaction or no contact.
                            if(owned[leg].empty()){
                                if(force[leg]>0.)throw MaterialFailure{5};
                                sole[leg]={-1.,0.,0.};continue;
                            }
                            sole[leg]=invert(*model,state,free,next,owned[leg],force[leg],dt,
                                             tangent_requested && k==subdivisions[j]-1);
                        }
                    }catch(MaterialFailure e){last_reason=e.reason;continue;}
                    consistent=true;
                    if(overlap){
                        for(int leg=0;leg<4;++leg)for(auto c:footprints[leg]){
                            int other=owner[c.id];
                            if(other!=leg && (sole[leg].delta>sole[other].delta ||
                               (sole[leg].delta==sole[other].delta && leg<other)))consistent=false;
                        }
                    }
                    if(consistent)break;
                }while(overlap && std::next_permutation(order.begin(),order.end()));
                if(!consistent)throw MaterialFailure{last_reason};
                for(int i=0;i<n;++i)settled[i]=next[i]==free[i] && std::equal(state[i].begin(),state[i].begin()+4,next[i].begin());
                state.swap(next);
                for(int leg=0;leg<4;++leg)if(!footprints[leg].empty()){
                    auto* out=results+(j*4+leg)*4;
                    out[0]=-sole[leg].delta;
                    out[1]+=-sole[leg].delta/subdivisions[j];
                    out[2]=sole[leg].tangent;
                    out[3]=std::max(out[3],std::abs(sole[leg].residual));
                }
            }catch(MaterialFailure e){detail[0]=j;detail[1]=k;detail[2]=dt;return e.reason;}
        }
        for(int i=0;i<n;++i)std::copy(state[i].begin(),state[i].end(),states+(j*n+i)*6);
    }
    return 0;
}
