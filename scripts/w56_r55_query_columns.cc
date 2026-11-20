// Read-only replay of one world cell at its original observer endpoints.
#include "s1_w55_c_column.cu"

extern "C" int query_states(const Model *model, int count, const double *time,
                            const double *normal, const int *ticks, int queries, double *output) {
    double state[4] = {}, row[6] = {};
    int next = 0;
    for (int k = 0; k < count && next < queries; ++k) {
        if (k) {
            run_one(*model, normal + k, 1, 1, time[k] - time[k-1], state, row);
            if (!std::isfinite(row[5]) || row[5] > 1e-7) return k;
        }
        if (k != ticks[next]) continue;
        double *out = output + 7*next++;
        std::copy(state, state + 4, out);
        double probe[4], unloaded = INFINITY;
        std::copy(state, state + 4, probe);
        for (int j=1; j<=250; ++j) {
            run_one(*model, &unloaded, 1, 1, .002, probe, row);
            if (!std::isfinite(row[5]) || row[5] > 1e-7) return k;
            if (j==1 || j==100 || j==250) out[j==1 ? 4 : j==100 ? 5 : 6] = -.05*probe[0];
        }
    }
    return next == queries ? -1 : count;
}
