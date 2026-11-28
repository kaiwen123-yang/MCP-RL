// Per-packet memoization of the original FP64 mixed boundary solve.
// Area, ownership and the per-foot force reduction remain in the original caller.
#include <array>
#include <cstdint>
#include <cstring>

namespace w61_exact {
constexpr size_t entries = 4096;
struct Entry {
    uint64_t generation;
    std::array<uint64_t,7> key;
    double state[4], stress[2], residual;
};
struct Cache {
    std::array<Entry,entries> table{};
    uint64_t generation = 0, hits = 0, misses = 0;
    bool active = false;
};
static thread_local Cache cache;
struct Scope {
    Scope() { ++cache.generation; cache.hits=0; cache.misses=0; cache.active=true; }
    ~Scope() { cache.active=false; }
};
static inline size_t bucket(const std::array<uint64_t,7>& key) {
    uint64_t h=0x9e3779b97f4a7c15ULL;
    for (auto value:key) {
        h ^= value;
        h = (h << 19 | h >> 45) * 0x9ddfea08eb382d69ULL;
    }
    return (h ^ (h>>32)) & (entries-1);
}
}

static inline double mixed(const Model& m, const double* prior, double normal, double dt,
                           bool fixed, double* next, double* stress) {
    using namespace w61_exact;
    if (!cache.active) return original_mixed(m,prior,normal,dt,fixed,next,stress);
    std::array<uint64_t,7> key;
    std::memcpy(key.data(),prior,4*sizeof(double));
    std::memcpy(&key[4],&normal,sizeof(double));
    std::memcpy(&key[5],&dt,sizeof(double));
    key[6]=fixed;
    auto& entry=cache.table[bucket(key)];
    if (entry.generation==cache.generation && entry.key==key) {
        std::copy_n(entry.state,4,next);
        std::copy_n(entry.stress,2,stress);
        ++cache.hits;
        return entry.residual;
    }
    ++cache.misses;
    const double residual=original_mixed(m,prior,normal,dt,fixed,next,stress);
    if (std::isfinite(residual) && residual<=1e-7) {
        entry.generation=cache.generation;
        entry.key=key;
        std::copy_n(next,4,entry.state);
        std::copy_n(stress,2,entry.stress);
        entry.residual=residual;
    }
    return residual;
}

extern "C" void w61_packet_counts(uint64_t* hits,uint64_t* misses) {
    *hits=w61_exact::cache.hits; *misses=w61_exact::cache.misses;
}
