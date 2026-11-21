#include "s1_w55_c_column.cu"

// A free cell whose input state was returned bit-for-bit has the same
// solution on the next equal-duration, non-contacting step.
extern "C" int prefix_state(const Model *m, int count, const double *end,
                            const double *normal, double *state) {
    double previous_end = 0, previous_dt = -1;
    bool stationary_free = true;
    for (int k=0; k<count; ++k) {
        const double dt = end[k]-previous_end;
        previous_end = end[k];
        if (stationary_free && dt == previous_dt && normal[k] > state[0]) continue;
        double prior[4], out[6];
        std::copy(state,state+4,prior);
        run_one(*m,normal+k,1,1,dt,state,out);
        if (!std::isfinite(out[5]) || out[5]>1.e-7) return k;
        stationary_free = out[4]==0 && std::equal(state,state+4,prior);
        previous_dt = dt;
    }
    return -1;
}
