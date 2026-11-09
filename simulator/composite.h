// Composite driver policy for MERGED multi-application LLC traces (mixed-stream DAgger, see docs/mixed_dagger.md).
// A merged trace (multisim merge) carries the application id in the high bits of every record: line address bits 48+ (so the tag differs per
// application but the set index does not) and PC bits 56+. CompositePolicy decides like the multi-application engine (simulator/multi.h):
//   req=P0+P1+..  the requester's own policy chooses among ALL lines of the set        (rule Req)
//   own=P0+P1+..  ownership rule: quota = ways / apps, a policy only evicts lines of its own application   (rule Own)
//   gen=P         one policy for the whole cache; it sees plain line and PC                (rule Gen)
// Every instance sees every hit and fill. For req/own an instance of application X sees the PC of other applications' accesses as 0.
// Given the same trace, the same models and the same warm-up, the misses equal those of multisim (tests/test_mixed.py checks this).
#pragma once
#include <memory>
#include <stdexcept>
#include "policies.h"

namespace ac {

constexpr uint32_t MIX_APP_SHIFT = 48;                       // same value as APP_SHIFT in multi.h
constexpr uint32_t MIX_PC_SHIFT = 56;
constexpr uint64_t MIX_LINE_MASK = (1ull << MIX_APP_SHIFT) - 1;
constexpr uint64_t MIX_PC_MASK = (1ull << MIX_PC_SHIFT) - 1;

class CompositePolicy : public Policy {
 public:
  enum Mode { REQ, OWN, GEN };
  CompositePolicy(Mode m, std::vector<std::unique_ptr<Policy>> p, const std::string& label) : mode_(m), pols_(std::move(p)), name_(label) {
    if (pols_.empty() || pols_.size() > 16) throw std::runtime_error("composite policy needs 1..16 member policies");
    if (m == GEN && pols_.size() != 1) throw std::runtime_error("gen= takes exactly one policy");
  }
  const char* name() const override { return name_.c_str(); }
  bool needsNextUse() const override { for (auto& p : pols_) if (p->needsNextUse()) return true; return false; }
  void reset(uint32_t sets, uint32_t ways) override {
    ways_ = ways; allowed_.assign(ways, 0);
    for (auto& p : pols_) p->reset(sets, ways);
  }
  void onHit(uint32_t set, uint32_t way, const AccessCtx& a) override { for (size_t j = 0; j < pols_.size(); j++) pols_[j]->onHit(set, way, ctxFor(j, a)); }
  void onFill(uint32_t set, uint32_t way, const AccessCtx& a) override { for (size_t j = 0; j < pols_.size(); j++) pols_[j]->onFill(set, way, ctxFor(j, a)); }
  uint32_t victim(uint32_t set, const AccessCtx& a, const uint64_t* tags) override {
    const uint32_t app = (uint32_t)(a.line >> MIX_APP_SHIFT), n = (uint32_t)pols_.size();
    if (mode_ == GEN) return pols_[0]->victim(set, ctxFor(0, a), tags);
    if (app >= n) throw std::runtime_error("composite policy: record of application " + std::to_string(app) + " but only " + std::to_string(n) + " policies");
    if (mode_ == REQ) return pols_[app]->victim(set, ctxFor(app, a), tags);
    uint32_t cnt[16] = {0};
    for (uint32_t q = 0; q < ways_; q++) cnt[(tags[q] >> MIX_APP_SHIFT) & 15]++;
    const uint32_t quota = std::max<uint32_t>(1, ways_ / n);
    uint32_t decider = app;
    if (cnt[app] < quota) {
      uint32_t best = 0; bool have = false;
      for (uint32_t o = 0; o < n; o++) if (o != app && (!have || cnt[o] > cnt[best])) { best = o; have = true; }
      if (have && cnt[best] > 0) decider = best;
    }
    for (uint32_t q = 0; q < ways_; q++) allowed_[q] = ((tags[q] >> MIX_APP_SHIFT) & 15) == decider;
    const uint32_t w = pols_[decider]->victimAllowed(set, ctxFor(decider, a), tags, allowed_.data(), ways_);
    if (w >= ways_ || !allowed_[w]) throw std::runtime_error("ownership rule: model chose a way it may not evict");
    return w;
  }
  std::string statsJson() const override { return ""; }

 private:
  AccessCtx ctxFor(size_t j, const AccessCtx& a) const {
    AccessCtx c = a;
    c.line = a.line & MIX_LINE_MASK;
    const uint32_t app = (uint32_t)(a.pc >> MIX_PC_SHIFT);
    c.pc = a.pc & MIX_PC_MASK;
    if (mode_ != GEN && app != j) c.pc = 0;
    return c;
  }
  Mode mode_;
  std::vector<std::unique_ptr<Policy>> pols_;
  std::string name_;
  uint32_t ways_ = 0;
  std::vector<uint8_t> allowed_;
};

}  // namespace ac
