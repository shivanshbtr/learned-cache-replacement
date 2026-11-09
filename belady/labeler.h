// Belady labeler: replays an LLC trace with a DRIVER policy (which decides what is really evicted) and, at every eviction
// decision (miss in a full set), records the state of the set together with Belady's victim set computed from next_use.
// The labels use the future only offline; the recorded state itself contains only past information.
// Record layout (all uint32): [seq, set, label_mask, driver_victim, then for each way: last_idx, insert_idx, count, next_use]
//   seq          index in the LLC trace of the access that missed (the "current" access)
//   label_mask   bit w set = way w holds a Belady-optimal victim (all lines never used again tie)
//   driver_victim way the driver policy actually evicted
//   last_idx     index of the most recent access to the resident line (line address and PC come from trace[last_idx])
//   insert_idx   index of the access that brought the line in; count = accesses to it since insertion (including the fill)
//   next_use     index of the next access to the line (NO_NEXT_USE = never): LABEL INFORMATION, not a model input
#pragma once
#include <vector>
#include "../simulator/sim.h"

namespace ac {

constexpr uint32_t LABEL_MAX_WAYS = 32;

struct LabelStats {
  uint64_t accesses = 0, hits = 0, misses = 0;           // whole trace, warm-up included
  uint64_t measured_hits = 0, measured_misses = 0;       // after warm-up
  uint64_t cold_fills = 0;                               // misses that filled an empty way (no decision)
  uint64_t decisions = 0, recorded = 0, tied = 0;        // tied = decisions with more than one Belady victim
  uint64_t driver_agrees = 0;                            // decisions where the driver's victim is in the Belady set
};

// emit(const uint32_t* rec, size_t words) is called for every `stride`-th decision (stride >= 1).
template <class Emit>
LabelStats labelTrace(Policy& driver, const Config& cfg, const std::vector<TraceRec>& t, const std::vector<uint32_t>& nu,
                      uint64_t warmup, uint32_t stride, Emit emit) {
  const uint32_t sets = cfg.llc_sets(), ways = cfg.llc.ways, shift = log2u(cfg.line_bytes);
  if (ways > LABEL_MAX_WAYS) throw std::runtime_error("labeler supports at most 32 ways");
  if (stride < 1) stride = 1;
  if (nu.size() != t.size()) throw std::runtime_error("next_use size mismatch");
  driver.reset(sets, ways);
  const size_t n = (size_t)sets * ways;
  std::vector<uint64_t> tags(n, 0);
  std::vector<uint8_t> valid(n, 0);
  std::vector<uint32_t> last(n, 0), ins(n, 0), cnt(n, 0), nxt(n, NO_NEXT_USE);
  std::vector<uint32_t> rec(4 + 4 * (size_t)ways);
  LabelStats st;
  for (size_t i = 0; i < t.size(); i++) {
    const uint64_t line = t[i].addr >> shift;
    const uint32_t set = (uint32_t)(line & (sets - 1));
    const size_t b = (size_t)set * ways;
    AccessCtx a{line, t[i].pc, i, set, (t[i].flags & FLAG_WRITE) != 0, nu[i]};
    const bool measured = i >= warmup;
    st.accesses++;
    int hitw = -1, inv = -1;
    for (uint32_t w = 0; w < ways; w++) {
      if (valid[b + w]) { if (tags[b + w] == line) { hitw = (int)w; break; } }
      else if (inv < 0) inv = (int)w;
    }
    if (hitw >= 0) {
      driver.onHit(set, (uint32_t)hitw, a);
      last[b + hitw] = (uint32_t)i; nxt[b + hitw] = nu[i]; if (cnt[b + hitw] != 0xFFFFFFFFu) cnt[b + hitw]++;
      st.hits++; if (measured) st.measured_hits++;
      continue;
    }
    st.misses++; if (measured) st.measured_misses++;
    uint32_t w;
    if (inv >= 0) { w = (uint32_t)inv; st.cold_fills++; }
    else {
      uint32_t best = 0;                                      // Belady victim set: farthest next use
      for (uint32_t x = 0; x < ways; x++) if (nxt[b + x] > best) best = nxt[b + x];
      uint32_t mask = 0;
      for (uint32_t x = 0; x < ways; x++) if (nxt[b + x] == best) mask |= 1u << x;
      w = driver.victim(set, a, &tags[b]);
      if (w >= ways) throw std::runtime_error(std::string(driver.name()) + " returned an invalid victim way");
      st.decisions++;
      if (mask & (mask - 1)) st.tied++;
      if (mask & (1u << w)) st.driver_agrees++;
      if ((st.decisions - 1) % stride == 0) {
        rec[0] = (uint32_t)i; rec[1] = set; rec[2] = mask; rec[3] = w;
        for (uint32_t x = 0; x < ways; x++) {
          rec[4 + 4 * x] = last[b + x]; rec[5 + 4 * x] = ins[b + x]; rec[6 + 4 * x] = cnt[b + x]; rec[7 + 4 * x] = nxt[b + x];
        }
        emit(rec.data(), rec.size());
        st.recorded++;
      }
    }
    tags[b + w] = line; valid[b + w] = 1;
    last[b + w] = ins[b + w] = (uint32_t)i; cnt[b + w] = 1; nxt[b + w] = nu[i];
    driver.onFill(set, w, a);
  }
  return st;
}

#pragma pack(push, 1)
struct DecHeader {
  char magic[8];          // "ACDEC\0\0\0"
  uint32_t version;       // 1
  uint32_t ways;
  uint32_t rec_words;     // uint32 words per record = 4 + 4*ways
  uint32_t stride;
  uint32_t sets;
  uint32_t line_bytes;
  uint64_t num_records;
  uint64_t trace_len;
  uint64_t warmup;
  char driver[8];
};
#pragma pack(pop)
static_assert(sizeof(DecHeader) == 64, "DecHeader must be 64 bytes");

}  // namespace ac
