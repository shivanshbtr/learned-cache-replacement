// Unit tests for the multi-application engine (simulator/multi.h): schedule, isolation of applications, ownership rule (hand-checked),
// equivalence with the single-application simulator, Belady optimality on the merged stream.
#include <cstdio>
#include "../simulator/factory.h"
#include "../simulator/multi.h"

using namespace ac;

static int g_fail = 0, g_checks = 0;
#define CHECK(c) do { g_checks++; if (!(c)) { g_fail++; fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)
#define CHECK_EQ(a, b) do { g_checks++; long long _a=(long long)(a), _b=(long long)(b); if (_a != _b) { g_fail++; fprintf(stderr, "FAIL %s:%d  %s == %s  (%lld vs %lld)\n", __FILE__, __LINE__, #a, #b, _a, _b); } } while (0)

static Config cfgFor(uint32_t sets, uint32_t ways) {
  Config c;
  c.llc.ways = ways;
  c.llc.size = (uint64_t)sets * ways * c.line_bytes;
  c.warmup_frac = 0.0;
  c.validate();
  return c;
}
static AppStream mkApp(const std::string& name, const std::vector<uint64_t>& lines, const std::vector<uint32_t>& ic = {}) {
  AppStream a; a.name = name;
  for (size_t i = 0; i < lines.size(); i++) {
    TraceRec r{}; r.addr = lines[i] << 6; r.pc = 0x400000 + (i % 5); r.icount = ic.empty() ? 1 : ic[i]; r.size = 64;
    a.t.push_back(r); a.total_icount += r.icount;
  }
  return a;
}
static uint64_t rngState = 0xABCDEF12345ULL;
static uint64_t rnd() { rngState ^= rngState << 13; rngState ^= rngState >> 7; rngState ^= rngState << 17; return rngState; }

static MultiSpec global(const std::string& n, const Config& c) {
  MultiSpec m; m.name = n; m.rule = Rule::Global; m.make = [n, c](uint32_t) { return makePolicy(n, c); }; return m;
}
static MultiSpec perApp(Rule r, const std::string& n) {   // every application's "model" is the baseline policy n
  MultiSpec m; m.name = n; m.rule = r; Config c; m.make = [n, c](uint32_t) { return makePolicy(n, c); }; return m;
}
static MultiResult go(const MultiSpec& s, const Config& c, const std::vector<AppStream>& apps, const Schedule& sch, uint64_t warm = 0, uint64_t coldK = 0, bool slices = false) {
  std::vector<uint32_t> ids(apps.size());
  for (uint32_t i = 0; i < ids.size(); i++) ids[i] = i;
  std::vector<uint32_t> nu;
  const bool needNu = s.name == "belady";
  if (needNu) nu = mergedNextUse(apps, sch, 6);
  return runMulti(s, c, apps, ids, sch, needNu ? &nu : nullptr, warm, coldK, slices);
}

static void testSchedule() {
  // A: 5 records of 1 instruction. B: 3 records of 3 instructions. Slice = 2 instructions.
  std::vector<AppStream> apps{mkApp("A", {0, 1, 2, 3, 4}), mkApp("B", {10, 11, 12}, {3, 3, 3})};
  Schedule s = buildSchedule(apps, 2, true);
  std::vector<int> wantApp{0, 0, 1, 0, 0, 1, 0, 1}, wantIdx{0, 1, 0, 2, 3, 1, 4, 2};
  CHECK_EQ(s.app.size(), wantApp.size());
  for (size_t i = 0; i < wantApp.size() && i < s.app.size(); i++) { CHECK_EQ(s.app[i], wantApp[i]); CHECK_EQ(s.idx[i], wantIdx[i]); }
  CHECK_EQ(s.slice_begin.size(), 6);   // A B A B A B
  CHECK_EQ(s.switches, 5);
  // until first: stops as soon as A has consumed its last record
  Schedule f = buildSchedule(apps, 2, false);
  std::vector<int> wantApp2{0, 0, 1, 0, 0, 1, 0};
  CHECK_EQ(f.app.size(), wantApp2.size());
  for (size_t i = 0; i < wantApp2.size() && i < f.app.size(); i++) CHECK_EQ(f.app[i], wantApp2[i]);
  // one application only: no switches
  std::vector<AppStream> one{mkApp("A", {0, 1, 2})};
  Schedule o = buildSchedule(one, 100, true);
  CHECK_EQ(o.switches, 0); CHECK_EQ(o.app.size(), 3);
  // a finished application is skipped, the remaining one keeps running
  std::vector<AppStream> two{mkApp("A", {0}), mkApp("B", {5, 6, 7, 8})};
  Schedule t = buildSchedule(two, 1, true);
  std::vector<int> wantApp3{0, 1, 1, 1, 1};
  CHECK_EQ(t.app.size(), 5);
  for (size_t i = 0; i < wantApp3.size() && i < t.app.size(); i++) CHECK_EQ(t.app[i], wantApp3[i]);
  CHECK_EQ(t.switches, 1);
}

static void testIsolation() {
  // Same plain line numbers in two applications must never hit on each other. 1 set, 2 ways.
  Config c = cfgFor(1, 2);
  std::vector<AppStream> apps{mkApp("A", {0, 1, 0, 1}), mkApp("B", {0, 1})};
  Schedule s = buildSchedule(apps, 2, true);   // A0 A1 | B0 B1 | A0 A1
  MultiResult r = go(global("lru", c), c, apps, s);
  CHECK_EQ(r.misses, 6); CHECK_EQ(r.hits, 0);
  CHECK_EQ(r.app[0].misses, 4); CHECK_EQ(r.app[1].misses, 2);
}

static void testOwnership() {
  // 1 set, 4 ways, 2 applications (quota 2). A: a0 a1 a2 a3 a2 a3   B: b0 b1 b2 b3.  slice = 4 instructions: A | B | A.
  Config c = cfgFor(1, 4);
  std::vector<AppStream> apps{mkApp("A", {0, 1, 2, 3, 2, 3}), mkApp("B", {0, 1, 2, 3})};
  Schedule s = buildSchedule(apps, 4, true);
  CHECK_EQ(s.app.size(), 10);
  MultiResult g = go(global("lru", c), c, apps, s);   // plain LRU: B's four lines push out all of A's lines, A's second visit misses
  CHECK_EQ(g.misses, 10); CHECK_EQ(g.hits, 0);
  MultiResult rq = go(perApp(Rule::Req, "lru"), c, apps, s);   // requester rule with LRU as the model = plain LRU
  CHECK_EQ(rq.misses, 10);
  CHECK_EQ(rq.app[0].misses, g.app[0].misses); CHECK_EQ(rq.app[1].misses, g.app[1].misses);
  // Ownership, worked out by hand:
  //   b0: B owns 0 (< quota 2) -> A (4 lines) gives up its LRU line a0.   b1: B owns 1 -> A (3 lines) gives up a1.
  //   b2: B owns 2 (= quota) -> B evicts its own LRU b0.                  b3: B evicts its own LRU b1.   Left: b2 b3 a2 a3.
  //   A's second visit a2 a3 hits twice.
  MultiResult ow = go(perApp(Rule::Own, "lru"), c, apps, s);
  CHECK_EQ(ow.misses, 8); CHECK_EQ(ow.hits, 2);
  CHECK_EQ(ow.app[0].misses, 4); CHECK_EQ(ow.app[0].hits, 2);
  CHECK_EQ(ow.app[1].misses, 4); CHECK_EQ(ow.app[1].hits, 0);
  CHECK_EQ(ow.evict[1][0], 2); CHECK_EQ(ow.evict[1][1], 2); CHECK_EQ(ow.evict[0][0], 0); CHECK_EQ(ow.evict[0][1], 0);
  CHECK_EQ(ow.app[1].borrow, 2); CHECK_EQ(ow.app[0].borrow, 0);
  // Under plain LRU the same evictions all hit A: b0..b3 evict a0 a1 a2 a3.
  CHECK_EQ(g.evict[1][0], 4);
  // Warm-up: the first 4 merged accesses are not counted.
  MultiResult w = go(perApp(Rule::Own, "lru"), c, apps, s, 4);
  CHECK_EQ(w.misses, 4); CHECK_EQ(w.hits, 2); CHECK_EQ(w.app[0].accesses, 2); CHECK_EQ(w.app[1].accesses, 4);
}


static void testOwnershipDifferentModels() {
  // Application A's model is LRU, application B's model is MRU. Same set-up as above but B has a 5th access (b0) after A's second visit.
  Config c = cfgFor(1, 4);
  std::vector<AppStream> apps{mkApp("A", {0, 1, 2, 3, 2, 3}), mkApp("B", {0, 1, 2, 3, 0})};
  Schedule s = buildSchedule(apps, 4, true);   // A a0..a3 | B b0..b3 | A a2 a3 | B b0
  MultiSpec m; m.name = "mix"; m.rule = Rule::Own;
  m.make = [c](uint32_t gid) { return makePolicy(gid == 0 ? "lru" : "mru", c); };
  MultiResult ow = go(m, c, apps, s);
  // b0, b1: A's model (LRU) gives up a0, a1.  b2: B's model (MRU) evicts its most recent own line b1.  b3: MRU evicts b2.  Left: b0 b3 a2 a3.
  // A's a2 a3 hit; B's final b0 HITS (with LRU as B's model it would have been evicted: b0 is B's least recent line).
  CHECK_EQ(ow.app[0].hits, 2); CHECK_EQ(ow.app[0].misses, 4);
  CHECK_EQ(ow.app[1].hits, 1); CHECK_EQ(ow.app[1].misses, 4);
  CHECK_EQ(ow.evict[1][0], 2); CHECK_EQ(ow.evict[1][1], 2);
  // Requester rule with the same models: B's MRU chooses among ALL lines. b0: evicts the most recent line (a3), b1: b0, b2: b1, b3: b2 -> A loses a3; A's own miss on a3 then evicts a0 (LRU); B's last access b0 misses and MRU evicts the fresh a3.
  MultiSpec r = m; r.rule = Rule::Req;
  MultiResult rq = go(r, c, apps, s);
  CHECK_EQ(rq.evict[1][0], 2); CHECK_EQ(rq.evict[1][1], 3);
  CHECK_EQ(rq.app[0].hits, 1);   // a2 survives, a3 does not (A's own a3 miss evicts a0)
}

// Records what each policy instance is shown.
struct SpyPolicy : public Policy {
  std::vector<uint64_t>* seen; uint64_t* nHit; uint64_t* nFill; uint64_t* nVict;
  SpyPolicy(std::vector<uint64_t>* s, uint64_t* h, uint64_t* f, uint64_t* v) : seen(s), nHit(h), nFill(f), nVict(v) {}
  const char* name() const override { return "spy"; }
  void reset(uint32_t, uint32_t ways) override { ways_ = ways; clk_ = 0; stamp_.assign(1 << 12, 0); }
  void onHit(uint32_t set, uint32_t way, const AccessCtx& a) override { seen->push_back(a.pc); (*nHit)++; stamp_[set * ways_ + way] = ++clk_; }
  void onFill(uint32_t set, uint32_t way, const AccessCtx& a) override { seen->push_back(a.pc); (*nFill)++; stamp_[set * ways_ + way] = ++clk_; }
  uint32_t victim(uint32_t set, const AccessCtx& a, const uint64_t*) override {
    (void)a; (*nVict)++; uint32_t v = 0;
    for (uint32_t w = 1; w < ways_; w++) if (stamp_[set * ways_ + w] < stamp_[set * ways_ + v]) v = w;
    return v;
  }
  uint32_t ways_ = 0; uint64_t clk_ = 0; std::vector<uint64_t> stamp_;
};

static void testPcMasking() {
  Config c = cfgFor(2, 2);
  AppStream A = mkApp("A", {0, 1, 2, 3, 4, 5}), B = mkApp("B", {8, 9, 10, 11, 12, 13});
  for (auto& r : A.t) r.pc = 0x1111;
  for (auto& r : B.t) r.pc = 0x2222;
  std::vector<AppStream> apps{A, B};
  Schedule s = buildSchedule(apps, 2, true);
  std::vector<uint64_t> seen[2]; uint64_t h[2] = {0, 0}, f[2] = {0, 0}, v[2] = {0, 0};
  MultiSpec m; m.name = "spy"; m.rule = Rule::Req;
  m.make = [&](uint32_t gid) { return std::unique_ptr<Policy>(new SpyPolicy(&seen[gid], &h[gid], &f[gid], &v[gid])); };
  MultiResult r = go(m, c, apps, s);
  // every instance sees EVERY hit and fill of the merged stream, in order; only its own application's PCs are visible
  for (int g = 0; g < 2; g++) {
    CHECK_EQ(h[g] + f[g], s.app.size());
    CHECK_EQ(seen[g].size(), s.app.size());
    for (size_t i = 0; i < seen[g].size(); i++) CHECK_EQ(seen[g][i], s.app[i] == g ? (g == 0 ? 0x1111 : 0x2222) : 0);
  }
  // decisions are asked from the REQUESTING application's instance only
  CHECK_EQ(v[0] + v[1], r.misses - 4);   // 4 cold fills (2 sets x 2 ways) need no decision
  // generic rule: one instance, sees all PCs unmasked
  std::vector<uint64_t> sg; uint64_t gh = 0, gf = 0, gv = 0;
  MultiSpec gm; gm.name = "spy"; gm.rule = Rule::Gen;
  gm.make = [&](uint32_t) { return std::unique_ptr<Policy>(new SpyPolicy(&sg, &gh, &gf, &gv)); };
  go(gm, c, apps, s);
  CHECK_EQ(sg.size(), s.app.size());
  for (size_t i = 0; i < sg.size(); i++) CHECK_EQ(sg[i], s.app[i] == 0 ? 0x1111 : 0x2222);
}

static std::vector<AppStream> randomApps(int n, size_t len, uint64_t range) {
  std::vector<AppStream> apps;
  for (int a = 0; a < n; a++) {
    std::vector<uint64_t> lines; std::vector<uint32_t> ic;
    size_t L = len / 2 + rnd() % len;
    for (size_t i = 0; i < L; i++) { lines.push_back(rnd() % range); ic.push_back(1 + (uint32_t)(rnd() % 4)); }
    apps.push_back(mkApp(std::string(1, (char)('A' + a)), lines, ic));
  }
  return apps;
}

static void testSingleAppEquivalence() {
  const char* pols[] = {"lru", "fifo", "random", "srrip", "brrip", "drrip", "bip", "mru", "belady"};
  for (int trial = 0; trial < 40; trial++) {
    uint32_t sets = 1u << (rnd() % 4), ways = 1 + (uint32_t)(rnd() % 6);
    Config c = cfgFor(sets, ways);
    c.warmup_frac = (trial % 2) ? 0.1 : 0.0;
    std::vector<AppStream> apps = randomApps(1, 300, 40);
    std::vector<TraceRec>& t = apps[0].t;
    Schedule s = buildSchedule(apps, 7, true);
    CHECK_EQ(s.app.size(), t.size());
    for (const char* pn : pols) {
      auto p = makePolicy(pn, c);
      std::vector<uint32_t> nu;
      if (p->needsNextUse()) nu = computeNextUse(t, 6);
      PolicyResult ref = runPolicy(*p, c, t, p->needsNextUse() ? &nu : nullptr, warmupCount(c, t.size()));
      MultiResult m = go(global(pn, c), c, apps, s, warmupCount(c, t.size()));
      CHECK_EQ(m.misses, ref.misses); CHECK_EQ(m.hits, ref.hits); CHECK_EQ(m.icount, ref.measured_icount);
    }
    // ownership with one application = plain behavior of the model
    MultiResult ow = go(perApp(Rule::Own, "lru"), c, apps, s, warmupCount(c, t.size()));
    MultiResult lr = go(global("lru", c), c, apps, s, warmupCount(c, t.size()));
    CHECK_EQ(ow.misses, lr.misses);
  }
}

static void testMergedProperties() {
  for (int trial = 0; trial < 60; trial++) {
    uint32_t sets = 1u << (rnd() % 3), ways = 2 + (uint32_t)(rnd() % 5);
    Config c = cfgFor(sets, ways);
    int n = 2 + (int)(rnd() % 3);
    std::vector<AppStream> apps = randomApps(n, 200, 24);
    uint64_t slice = 1 + rnd() % 20;
    Schedule s = buildSchedule(apps, slice, (trial % 2) == 0);
    uint64_t warm = (trial % 3) ? s.app.size() / 10 : 0;
    MultiResult bel = go(global("belady", c), c, apps, s, warm);
    // Belady on the merged stream equals Belady on one hand-merged single trace with application-tagged addresses
    std::vector<TraceRec> merged;
    for (size_t i = 0; i < s.app.size(); i++) {
      TraceRec r = apps[s.app[i]].t[s.idx[i]];
      r.addr = ((r.addr >> 6) | ((uint64_t)s.app[i] << APP_SHIFT)) << 6;
      merged.push_back(r);
    }
    std::vector<uint32_t> nu = computeNextUse(merged, 6);
    BeladyPolicy bp;
    PolicyResult ref = runPolicy(bp, c, merged, &nu, warm);
    CHECK_EQ(bel.misses, ref.misses);
    for (const char* pn : {"lru", "fifo", "srrip", "mru", "bip", "random"}) {
      MultiResult m = go(global(pn, c), c, apps, s, warm);
      CHECK(bel.misses <= m.misses);
      uint64_t am = 0, ah = 0, aa = 0;
      for (auto& x : m.app) { am += x.misses; ah += x.hits; aa += x.accesses; CHECK_EQ(x.hits + x.misses, x.accesses); }
      CHECK_EQ(am, m.misses); CHECK_EQ(ah, m.hits); CHECK_EQ(aa, m.hits + m.misses);
      uint64_t ev = 0;
      for (auto& row : m.evict) for (uint64_t v : row) ev += v;
      CHECK(ev <= m.misses);
    }
    // requester rule with LRU as the model == plain LRU; ownership never throws and keeps the books balanced
    MultiResult lru = go(global("lru", c), c, apps, s, warm), rq = go(perApp(Rule::Req, "lru"), c, apps, s, warm);
    CHECK_EQ(rq.misses, lru.misses);
    for (const char* pn : {"lru", "mru"}) {
      MultiResult ow = go(perApp(Rule::Own, pn), c, apps, s, warm);
      uint64_t am = 0;
      for (auto& x : ow.app) { am += x.misses; CHECK_EQ(x.hits + x.misses, x.accesses); }
      CHECK_EQ(am, ow.misses);
      CHECK(ow.misses >= bel.misses);
    }
  }
}

static void testColdAndSlices() {
  // One application, slices of 3 instructions, coldK = 2: the first two accesses of every slice are "cold".
  Config c = cfgFor(1, 2);
  std::vector<AppStream> apps{mkApp("A", {0, 1, 2, 3, 4, 5, 6})};
  Schedule s = buildSchedule(apps, 3, true);   // slices of 3, 3, 1 accesses
  MultiResult r = go(global("lru", c), c, apps, s, 0, 2, true);
  CHECK_EQ(r.slices.size(), 3);
  CHECK_EQ(r.slices[0][1], 3); CHECK_EQ(r.slices[2][1], 1);
  CHECK_EQ(r.app[0].cold_acc, 5);    // 2 + 2 + 1
  CHECK_EQ(r.app[0].cold_miss, 5);   // all lines are new
  uint64_t m = 0;
  for (auto& x : r.slices) m += x[2];
  CHECK_EQ(m, r.misses);
}

int main() {
  testSchedule();
  testIsolation();
  testOwnership();
  testOwnershipDifferentModels();
  testPcMasking();
  testSingleAppEquivalence();
  testMergedProperties();
  testColdAndSlices();
  printf("multi tests: %d checks, %d failed\n", g_checks, g_fail);
  return g_fail ? 1 : 0;
}
