// LLC replacement policies behind one plug-in interface. Learned models will implement the same interface.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "config.h"

namespace ac {

constexpr uint32_t NO_NEXT_USE = 0xFFFFFFFFu;  // line is never used again (or next_use unknown)

struct AccessCtx {
  uint64_t line;      // line address (addr >> log2(line_bytes))
  uint64_t pc;
  uint64_t seq;       // index of this access in the LLC trace
  uint32_t set;
  bool is_write;
  uint32_t next_use;  // index of the next access to this line; only filled when Belady is running
};

class Policy {
 public:
  virtual ~Policy() {}
  virtual const char* name() const = 0;
  virtual void reset(uint32_t sets, uint32_t ways) = 0;
  virtual void onHit(uint32_t set, uint32_t way, const AccessCtx& a) = 0;
  // Called after a line was placed into `way` (either an empty way or the way returned by victim()).
  virtual void onFill(uint32_t set, uint32_t way, const AccessCtx& a) = 0;
  // Called only when all ways of the set are valid. `tags` holds the resident line addresses of the set.
  virtual uint32_t victim(uint32_t set, const AccessCtx& a, const uint64_t* tags) = 0;
  // Like victim(), but only ways with allowed[w] != 0 may be chosen (used by the shared-LLC ownership rule). Default: ask victim() and,
  // if it is not allowed, take the first allowed way. LRU, MRU and the learned policies override it with a real restricted choice.
  virtual uint32_t victimAllowed(uint32_t set, const AccessCtx& a, const uint64_t* tags, const uint8_t* allowed, uint32_t ways) {
    uint32_t v = victim(set, a, tags);
    if (allowed[v]) return v;
    for (uint32_t w = 0; w < ways; w++) if (allowed[w]) return w;
    return v;
  }
  virtual bool needsNextUse() const { return false; }
  // Optional extra statistics as a JSON object (learned policies: model size, ops per decision, decision latency).
  virtual std::string statsJson() const { return ""; }
};

class LruPolicy : public Policy {
 public:
  const char* name() const override { return "lru"; }
  void reset(uint32_t sets, uint32_t ways) override { ways_ = ways; stamp_.assign((size_t)sets * ways, 0); clk_ = 0; }
  void onHit(uint32_t set, uint32_t way, const AccessCtx&) override { stamp_[(size_t)set * ways_ + way] = ++clk_; }
  void onFill(uint32_t set, uint32_t way, const AccessCtx&) override { stamp_[(size_t)set * ways_ + way] = ++clk_; }
  uint32_t victim(uint32_t set, const AccessCtx&, const uint64_t*) override {
    size_t b = (size_t)set * ways_;
    uint32_t v = 0;
    for (uint32_t w = 1; w < ways_; w++) if (stamp_[b + w] < stamp_[b + v]) v = w;
    return v;
  }
  uint32_t victimAllowed(uint32_t set, const AccessCtx&, const uint64_t*, const uint8_t* allowed, uint32_t) override {
    size_t b = (size_t)set * ways_;
    int v = -1;
    for (uint32_t w = 0; w < ways_; w++) if (allowed[w] && (v < 0 || stamp_[b + w] < stamp_[b + (uint32_t)v])) v = (int)w;
    return (uint32_t)(v < 0 ? 0 : v);
  }
 private:
  uint32_t ways_ = 0;
  uint64_t clk_ = 0;
  std::vector<uint64_t> stamp_;
};

class MruPolicy : public Policy {   // evicts the most recently used line (the right answer on cyclic thrashing patterns)
 public:
  const char* name() const override { return "mru"; }
  void reset(uint32_t sets, uint32_t ways) override { ways_ = ways; stamp_.assign((size_t)sets * ways, 0); clk_ = 0; }
  void onHit(uint32_t set, uint32_t way, const AccessCtx&) override { stamp_[(size_t)set * ways_ + way] = ++clk_; }
  void onFill(uint32_t set, uint32_t way, const AccessCtx&) override { stamp_[(size_t)set * ways_ + way] = ++clk_; }
  uint32_t victim(uint32_t set, const AccessCtx&, const uint64_t*) override {
    size_t b = (size_t)set * ways_;
    uint32_t v = 0;
    for (uint32_t w = 1; w < ways_; w++) if (stamp_[b + w] > stamp_[b + v]) v = w;
    return v;
  }
  uint32_t victimAllowed(uint32_t set, const AccessCtx&, const uint64_t*, const uint8_t* allowed, uint32_t) override {
    size_t b = (size_t)set * ways_;
    int v = -1;
    for (uint32_t w = 0; w < ways_; w++) if (allowed[w] && (v < 0 || stamp_[b + w] > stamp_[b + (uint32_t)v])) v = (int)w;
    return (uint32_t)(v < 0 ? 0 : v);
  }
 private:
  uint32_t ways_ = 0;
  uint64_t clk_ = 0;
  std::vector<uint64_t> stamp_;
};

class FifoPolicy : public Policy {
 public:
  const char* name() const override { return "fifo"; }
  void reset(uint32_t sets, uint32_t ways) override { ways_ = ways; stamp_.assign((size_t)sets * ways, 0); clk_ = 0; }
  void onHit(uint32_t, uint32_t, const AccessCtx&) override {}  // hits do not change FIFO order
  void onFill(uint32_t set, uint32_t way, const AccessCtx&) override { stamp_[(size_t)set * ways_ + way] = ++clk_; }
  uint32_t victim(uint32_t set, const AccessCtx&, const uint64_t*) override {
    size_t b = (size_t)set * ways_;
    uint32_t v = 0;
    for (uint32_t w = 1; w < ways_; w++) if (stamp_[b + w] < stamp_[b + v]) v = w;
    return v;
  }
 private:
  uint32_t ways_ = 0;
  uint64_t clk_ = 0;
  std::vector<uint64_t> stamp_;
};

class RandomPolicy : public Policy {
 public:
  explicit RandomPolicy(uint64_t seed) : seed_(seed) {}
  const char* name() const override { return "random"; }
  void reset(uint32_t, uint32_t ways) override { ways_ = ways; s_ = seed_ * 0x9E3779B97F4A7C15ULL + 88172645463325252ULL; if (!s_) s_ = 1; }
  void onHit(uint32_t, uint32_t, const AccessCtx&) override {}
  void onFill(uint32_t, uint32_t, const AccessCtx&) override {}
  uint32_t victim(uint32_t, const AccessCtx&, const uint64_t*) override {
    s_ ^= s_ << 13; s_ ^= s_ >> 7; s_ ^= s_ << 17;
    return (uint32_t)(s_ % ways_);
  }
 private:
  uint64_t seed_, s_ = 1;
  uint32_t ways_ = 0;
};

// SRRIP-HP: M-bit re-reference prediction values. Insert at max-1, hit -> 0, evict first way at max (aging if none).
class SrripPolicy : public Policy {
 public:
  explicit SrripPolicy(uint32_t bits) : max_((uint8_t)((1u << bits) - 1)) {}
  const char* name() const override { return "srrip"; }
  void reset(uint32_t sets, uint32_t ways) override { ways_ = ways; rrpv_.assign((size_t)sets * ways, max_); }
  void onHit(uint32_t set, uint32_t way, const AccessCtx&) override { rrpv_[(size_t)set * ways_ + way] = 0; }
  void onFill(uint32_t set, uint32_t way, const AccessCtx&) override { rrpv_[(size_t)set * ways_ + way] = (uint8_t)(max_ - 1); }
  uint32_t victim(uint32_t set, const AccessCtx&, const uint64_t*) override {
    size_t b = (size_t)set * ways_;
    for (;;) {
      for (uint32_t w = 0; w < ways_; w++) if (rrpv_[b + w] == max_) return w;
      for (uint32_t w = 0; w < ways_; w++) rrpv_[b + w]++;
    }
  }
 private:
  uint8_t max_;
  uint32_t ways_ = 0;
  std::vector<uint8_t> rrpv_;
};

// Small deterministic generator shared by the bimodal policies (xorshift64).
struct SmallRng {
  uint64_t s = 1;
  void seed(uint64_t x) { s = x * 0x9E3779B97F4A7C15ULL + 88172645463325252ULL; if (!s) s = 1; }
  uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
};

// BIP (bimodal insertion, Qureshi et al. 2007): LRU stack, but a new line goes to the LRU end, except with
// probability 1/eps_den it goes to the MRU end. Hits move the line to MRU. Thrash resistant.
class BipPolicy : public Policy {
 public:
  BipPolicy(uint64_t seed, uint32_t eps_den) : seed_(seed), eps_den_(eps_den) {}
  const char* name() const override { return "bip"; }
  void reset(uint32_t sets, uint32_t ways) override {
    ways_ = ways; stamp_.assign((size_t)sets * ways, 0); valid_.assign((size_t)sets * ways, 0); clk_ = 0; rng_.seed(seed_);
  }
  void onHit(uint32_t set, uint32_t way, const AccessCtx&) override { stamp_[(size_t)set * ways_ + way] = ++clk_; }
  void onFill(uint32_t set, uint32_t way, const AccessCtx&) override {
    size_t b = (size_t)set * ways_;
    if (rng_.next() % eps_den_ == 0) stamp_[b + way] = ++clk_;          // MRU insertion
    else {                                                              // LRU insertion: just below every other valid line
      int64_t lo = 0; bool any = false;
      for (uint32_t w = 0; w < ways_; w++) if (w != way && valid_[b + w]) { if (!any || stamp_[b + w] < lo) lo = stamp_[b + w]; any = true; }
      stamp_[b + way] = any ? lo - 1 : 0;
    }
    valid_[b + way] = 1;
  }
  uint32_t victim(uint32_t set, const AccessCtx&, const uint64_t*) override {
    size_t b = (size_t)set * ways_;
    uint32_t v = 0;
    for (uint32_t w = 1; w < ways_; w++) if (stamp_[b + w] < stamp_[b + v]) v = w;
    return v;
  }
 private:
  uint64_t seed_;
  uint32_t eps_den_, ways_ = 0;
  int64_t clk_ = 0;
  SmallRng rng_;
  std::vector<int64_t> stamp_;
  std::vector<uint8_t> valid_;
};

// RRIP family with the same RRPV mechanics as SRRIP. mode: 0 = BRRIP (bimodal insertion: max, or max-1 with prob 1/eps_den),
// 1 = DRRIP (set dueling, Jaleel et al. 2010: SRRIP leader sets vs BRRIP leader sets, 10-bit PSEL picks the follower sets' insertion).
// Leader sets: in every group of G = min(64, sets) sets, set%G==0 is an SRRIP leader and set%G==G/2 a BRRIP leader.
class RripDuelPolicy : public Policy {
 public:
  RripDuelPolicy(uint32_t bits, uint64_t seed, uint32_t eps_den, bool duel)
      : max_((uint8_t)((1u << bits) - 1)), seed_(seed), eps_den_(eps_den), duel_(duel) {}
  const char* name() const override { return duel_ ? "drrip" : "brrip"; }
  void reset(uint32_t sets, uint32_t ways) override {
    ways_ = ways; rrpv_.assign((size_t)sets * ways, max_); rng_.seed(seed_); psel_ = 512;
    group_ = sets < 64 ? sets : 64;
  }
  void onHit(uint32_t set, uint32_t way, const AccessCtx&) override { rrpv_[(size_t)set * ways_ + way] = 0; }
  void onFill(uint32_t set, uint32_t way, const AccessCtx&) override {
    bool bimodal = true;
    if (duel_) {
      int kind = leaderKind(set);                 // 1 = SRRIP leader, 2 = BRRIP leader, 0 = follower
      if (kind == 1) { if (psel_ < 1023) psel_++; bimodal = false; }    // a miss in an SRRIP leader votes for BRRIP
      else if (kind == 2) { if (psel_ > 0) psel_--; bimodal = true; }
      else bimodal = psel_ >= 512;
    }
    uint8_t v = (uint8_t)(max_ - 1);
    if (bimodal && rng_.next() % eps_den_ != 0) v = max_;
    rrpv_[(size_t)set * ways_ + way] = v;
  }
  uint32_t victim(uint32_t set, const AccessCtx&, const uint64_t*) override {
    size_t b = (size_t)set * ways_;
    for (;;) {
      for (uint32_t w = 0; w < ways_; w++) if (rrpv_[b + w] == max_) return w;
      for (uint32_t w = 0; w < ways_; w++) rrpv_[b + w]++;
    }
  }
  uint32_t psel() const { return psel_; }
 private:
  int leaderKind(uint32_t set) const {
    if (group_ < 2) return 0;
    uint32_t r = set % group_;
    return r == 0 ? 1 : (r == group_ / 2 ? 2 : 0);
  }
  uint8_t max_;
  uint64_t seed_;
  uint32_t eps_den_, ways_ = 0, group_ = 1, psel_ = 512;
  bool duel_;
  SmallRng rng_;
  std::vector<uint8_t> rrpv_;
};

// Belady / MIN: evict the resident line whose next use is farthest in the future (offline; never used online).
// No bypass: the missing line is always inserted. Ties (several lines never used again) go to the lowest way.
class BeladyPolicy : public Policy {
 public:
  const char* name() const override { return "belady"; }
  bool needsNextUse() const override { return true; }
  void reset(uint32_t sets, uint32_t ways) override { ways_ = ways; nu_.assign((size_t)sets * ways, NO_NEXT_USE); }
  void onHit(uint32_t set, uint32_t way, const AccessCtx& a) override { nu_[(size_t)set * ways_ + way] = a.next_use; }
  void onFill(uint32_t set, uint32_t way, const AccessCtx& a) override { nu_[(size_t)set * ways_ + way] = a.next_use; }
  uint32_t victim(uint32_t set, const AccessCtx&, const uint64_t*) override {
    size_t b = (size_t)set * ways_;
    uint32_t v = 0;
    for (uint32_t w = 1; w < ways_; w++) if (nu_[b + w] > nu_[b + v]) v = w;
    return v;
  }
 private:
  uint32_t ways_ = 0;
  std::vector<uint32_t> nu_;
};

}  // namespace ac
