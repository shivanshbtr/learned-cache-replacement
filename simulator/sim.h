// L1/L2 filter and LLC engine.
#pragma once
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include "config.h"
#include "policies.h"
#include "trace.h"

namespace ac {

inline uint32_t log2u(uint32_t x) { uint32_t s = 0; while ((1u << s) < x) s++; return s; }

// Set-associative LRU cache used for L1 and L2 (non-inclusive, write-allocate, no write-back modelling).
class LruCache {
 public:
  LruCache(uint32_t sets, uint32_t ways) : sets_(sets), ways_(ways), tag_((size_t)sets * ways), stamp_((size_t)sets * ways, 0), valid_((size_t)sets * ways, 0) {}
  // Returns true on hit. On miss the line is filled (evicting the LRU line if the set is full).
  bool access(uint64_t line) {
    size_t b = (size_t)(line & (sets_ - 1)) * ways_;
    ++clk_;
    int inv = -1;
    uint32_t lruw = 0;
    uint64_t lrus = UINT64_MAX;
    for (uint32_t w = 0; w < ways_; w++) {
      if (valid_[b + w]) {
        if (tag_[b + w] == line) { stamp_[b + w] = clk_; return true; }
        if (stamp_[b + w] < lrus) { lrus = stamp_[b + w]; lruw = w; }
      } else if (inv < 0) inv = (int)w;
    }
    uint32_t w = inv >= 0 ? (uint32_t)inv : lruw;
    tag_[b + w] = line; valid_[b + w] = 1; stamp_[b + w] = clk_;
    return false;
  }
 private:
  uint32_t sets_, ways_;
  uint64_t clk_ = 0;
  std::vector<uint64_t> tag_, stamp_;
  std::vector<uint8_t> valid_;
};

struct FilterStats {
  uint64_t raw_records = 0, line_accesses = 0, l1_hits = 0, l1_misses = 0, l2_hits = 0, l2_misses = 0, llc_out = 0;
  uint64_t total_icount = 0;
};

// next(rec) -> bool pulls raw records; emit(rec) receives LLC-bound accesses (L2 misses).
// An access that crosses a line boundary becomes one access per line.
template <class Next, class Emit>
FilterStats filterTrace(const Config& cfg, Next next, Emit emit) {
  cfg.validate();
  const uint32_t shift = log2u(cfg.line_bytes);
  LruCache l1(cfg.l1_sets(), cfg.l1.ways), l2(cfg.l2_sets(), cfg.l2.ways);
  FilterStats st;
  TraceRec r;
  uint64_t pending = 0;
  while (next(r)) {
    st.raw_records++;
    st.total_icount += r.icount;
    pending += r.icount;
    uint64_t first = r.addr >> shift, last = (r.addr + (r.size ? r.size - 1 : 0)) >> shift;
    for (uint64_t line = first; line <= last; ++line) {
      st.line_accesses++;
      if (l1.access(line)) { st.l1_hits++; continue; }
      st.l1_misses++;
      if (l2.access(line)) { st.l2_hits++; continue; }
      st.l2_misses++;
      TraceRec o{};
      o.addr = line << shift;
      o.pc = r.pc;
      uint64_t take = std::min<uint64_t>(pending, 0xFFFFFFFFull);
      o.icount = (uint32_t)take;
      pending -= take;
      o.size = (uint8_t)cfg.line_bytes;
      o.flags = r.flags & FLAG_WRITE;
      emit(o);
      st.llc_out++;
    }
  }
  return st;
}

// next_use[i] = index of the next access to the same line after i, or NO_NEXT_USE. O(N).
inline std::vector<uint32_t> computeNextUse(const std::vector<TraceRec>& t, uint32_t line_shift) {
  if (t.size() >= NO_NEXT_USE) throw std::runtime_error("trace too long for 32-bit next_use");
  std::vector<uint32_t> nu(t.size());
  std::unordered_map<uint64_t, uint32_t> last;
  last.reserve(1 << 16);
  for (size_t i = t.size(); i-- > 0;) {
    uint64_t line = t[i].addr >> line_shift;
    auto it = last.find(line);
    nu[i] = (it == last.end()) ? NO_NEXT_USE : it->second;
    last[line] = (uint32_t)i;
  }
  return nu;
}

struct PolicyResult {
  std::string name;
  uint64_t accesses = 0, warmup_accesses = 0;
  uint64_t hits = 0, misses = 0;          // measured (after warm-up)
  uint64_t measured_icount = 0;
  double wall_s = 0;
  std::string extra;                      // policy-specific statistics (JSON object) or empty
  double hit_rate() const { return (hits + misses) ? (double)hits / (hits + misses) : 0.0; }
  double mpki() const { return measured_icount ? 1000.0 * (double)misses / (double)measured_icount : 0.0; }
  double acc_per_s() const { return wall_s > 0 ? (double)accesses / wall_s : 0.0; }
};

inline PolicyResult runPolicy(Policy& pol, const Config& cfg, const std::vector<TraceRec>& t,
                              const std::vector<uint32_t>* nu, uint64_t warmup) {
  if (pol.needsNextUse() && !nu) throw std::runtime_error(std::string(pol.name()) + " needs next_use");
  const uint32_t sets = cfg.llc_sets(), ways = cfg.llc.ways, shift = log2u(cfg.line_bytes);
  pol.reset(sets, ways);
  std::vector<uint64_t> tags((size_t)sets * ways, 0);
  std::vector<uint8_t> valid((size_t)sets * ways, 0);
  PolicyResult res;
  res.name = pol.name();
  res.accesses = t.size();
  res.warmup_accesses = std::min<uint64_t>(warmup, t.size());
  auto t0 = std::chrono::steady_clock::now();
  for (size_t i = 0; i < t.size(); i++) {
    const uint64_t line = t[i].addr >> shift;
    const uint32_t set = (uint32_t)(line & (sets - 1));
    const size_t b = (size_t)set * ways;
    AccessCtx a{line, t[i].pc, i, set, (t[i].flags & FLAG_WRITE) != 0, nu ? (*nu)[i] : NO_NEXT_USE};
    const bool measured = i >= warmup;
    if (measured) res.measured_icount += t[i].icount;
    int hitw = -1, inv = -1;
    for (uint32_t w = 0; w < ways; w++) {
      if (valid[b + w]) { if (tags[b + w] == line) { hitw = (int)w; break; } }
      else if (inv < 0) inv = (int)w;
    }
    if (hitw >= 0) {
      pol.onHit(set, (uint32_t)hitw, a);
      if (measured) res.hits++;
    } else {
      uint32_t w;
      if (inv >= 0) w = (uint32_t)inv;
      else {
        w = pol.victim(set, a, &tags[b]);
        if (w >= ways) throw std::runtime_error(std::string(pol.name()) + " returned an invalid victim way");
      }
      tags[b + w] = line; valid[b + w] = 1;
      pol.onFill(set, w, a);
      if (measured) res.misses++;
    }
  }
  res.wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  res.extra = pol.statsJson();
  return res;
}

inline uint64_t warmupCount(const Config& cfg, uint64_t n) { return (uint64_t)(cfg.warmup_frac * (double)n); }

}  // namespace ac
