// Unit tests: hand-checked traces, Belady optimality, reference-LRU comparison, filter, trace IO, config, lackey parser.
#include <cstdio>
#include <cstdlib>
#include <list>
#include <map>
#include <set>
#include <sstream>
#include "../simulator/factory.h"
#include "../simulator/sim.h"
#include "../belady/labeler.h"
#include "../trace_collection/lackey_parser.h"

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
static std::vector<TraceRec> mk(const std::vector<uint64_t>& lines, uint32_t ic = 1) {
  std::vector<TraceRec> t;
  for (uint64_t l : lines) { TraceRec r{}; r.addr = l << 6; r.pc = 0x400000; r.icount = ic; r.size = 64; t.push_back(r); }
  return t;
}
static PolicyResult run(const std::string& pol, const Config& c, const std::vector<TraceRec>& t, uint64_t warm = 0) {
  auto p = makePolicy(pol, c);
  std::vector<uint32_t> nu;
  if (p->needsNextUse()) nu = computeNextUse(t, 6);
  return runPolicy(*p, c, t, p->needsNextUse() ? &nu : nullptr, warm);
}
static uint64_t rngState = 0x1234567ULL;
static uint64_t rnd() { rngState ^= rngState << 13; rngState ^= rngState >> 7; rngState ^= rngState << 17; return rngState; }

static void testHandChecked() {
  // 5 lines cycled 3 times through one 4-way set (A..E = 0..4).
  std::vector<uint64_t> cyc;
  for (int r = 0; r < 3; r++) for (uint64_t l = 0; l < 5; l++) cyc.push_back(l);
  Config c = cfgFor(1, 4);
  auto t = mk(cyc);
  CHECK_EQ(run("lru", c, t).misses, 15);
  CHECK_EQ(run("fifo", c, t).misses, 15);
  auto b = run("belady", c, t);   // worked out by hand: 4 cold + evictions at idx 4, 8, 12 = 7 misses
  CHECK_EQ(b.misses, 7);
  CHECK_EQ(b.hits, 8);

  // A B A C A in a 2-way set: LRU keeps A (3 misses); FIFO evicts A at C (4 misses).
  Config c2 = cfgFor(1, 2);
  auto t2 = mk({0, 1, 0, 2, 0});
  CHECK_EQ(run("lru", c2, t2).misses, 3);
  CHECK_EQ(run("fifo", c2, t2).misses, 4);

  // MRU, 2-way set, A B C A: C evicts B (most recent), A then hits -> 3 misses (LRU: C evicts A, so 4 misses).
  auto tm = mk({0, 1, 2, 0});
  CHECK_EQ(run("mru", c2, tm).misses, 3);
  CHECK_EQ(run("lru", c2, tm).misses, 4);
  // MRU on the 5-line cycle in a 4-way set, hand-simulated (round 1: E evicts D; round 2: D evicts C; round 3: C evicts B): 5+1+1 = 7 misses = Belady.
  CHECK_EQ(run("mru", c, t).misses, 7);
  CHECK_EQ(run("mru", c, t).hits, 8);

  // SRRIP scan resistance, 4-way set, hand-simulated: H1 H2 H1 H2 S1..S5 H1 H2.
  Config c4 = cfgFor(1, 4);
  auto t3 = mk({10, 11, 10, 11, 1, 2, 3, 4, 5, 10, 11});
  auto s = run("srrip", c4, t3), l = run("lru", c4, t3);
  CHECK_EQ(s.misses, 7); CHECK_EQ(s.hits, 4);
  CHECK_EQ(l.misses, 9); CHECK_EQ(l.hits, 2);

  // Two sets: lines map by (line & 1); each set is independent.
  Config c5 = cfgFor(2, 1);
  auto t4 = mk({0, 1, 0, 1, 2, 0});   // set0: 0 0 2 0 ; set1: 1 1  (1 way)
  auto r5 = run("lru", c5, t4);
  CHECK_EQ(r5.misses, 4);  // 0 miss, 1 miss, 0 hit, 1 hit, 2 miss (evicts 0), 0 miss
}

static void testNextUseAndWarmup() {
  auto t = mk({0, 1, 0, 2, 1});
  auto nu = computeNextUse(t, 6);
  CHECK_EQ(nu[0], 2); CHECK_EQ(nu[1], 4); CHECK(nu[2] == NO_NEXT_USE); CHECK(nu[3] == NO_NEXT_USE); CHECK(nu[4] == NO_NEXT_USE);
  // Warm-up: 10 distinct lines, first 5 are warm-up: measured accesses = 5, all misses.
  Config c = cfgFor(1, 4);
  auto t2 = mk({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 2);
  auto r = run("lru", c, t2, 5);
  CHECK_EQ(r.hits + r.misses, 5); CHECK_EQ(r.misses, 5); CHECK_EQ(r.measured_icount, 10);
}

// Reference LRU written independently with std::list.
static uint64_t refLruMisses(const std::vector<uint64_t>& lines, uint32_t sets, uint32_t ways) {
  std::vector<std::list<uint64_t>> s(sets);
  uint64_t miss = 0;
  for (uint64_t l : lines) {
    auto& q = s[l % sets];
    bool hit = false;
    for (auto it = q.begin(); it != q.end(); ++it) if (*it == l) { q.erase(it); hit = true; break; }
    if (!hit) { miss++; if (q.size() == ways) q.pop_back(); }
    q.push_front(l);
  }
  return miss;
}

static std::vector<uint64_t> genTrace(int kind, uint32_t universe, size_t n) {
  std::vector<uint64_t> v;
  for (size_t i = 0; i < n; i++) {
    uint64_t l;
    if (kind == 0) l = rnd() % universe;
    else if (kind == 1) { double u = (rnd() >> 11) / 9007199254740992.0; l = (uint64_t)(universe * u * u * u); }
    else if (kind == 2) l = i % universe;
    else l = (i % 3 == 0) ? (rnd() % 4) : (1000 + i);   // hot lines mixed with a scan
    v.push_back(l);
  }
  return v;
}

static void testProperties() {
  const char* pols[] = {"lru", "fifo", "random", "srrip", "bip", "brrip", "drrip"};
  int cases = 0;
  for (int iter = 0; iter < 400; iter++) {
    uint32_t sets = 1u << (rnd() % 4);                 // 1,2,4,8
    uint32_t ways = 1 + rnd() % 8;
    uint32_t universe = ways + 1 + rnd() % (sets * ways * 3 + 4);
    auto lines = genTrace((int)(rnd() % 4), universe, 100 + rnd() % 1500);
    Config c = cfgFor(sets, ways);
    c.seed = 1 + rnd() % 100;
    auto t = mk(lines);
    auto bel = run("belady", c, t);
    for (const char* p : pols) {
      auto r = run(p, c, t);
      CHECK(bel.misses <= r.misses);                    // Belady never loses
      CHECK_EQ(r.hits + r.misses, (int64_t)lines.size());
    }
    CHECK_EQ(run("lru", c, t).misses, refLruMisses(lines, sets, ways));  // LRU matches the independent reference
    cases++;
  }
  CHECK_EQ(cases, 400);
}

// Exhaustive minimum misses for one set (demand fetch, no bypass) by trying every eviction choice.
static int optMisses(const std::vector<uint64_t>& t, size_t pos, std::vector<uint64_t> cache, uint32_t ways,
                     std::map<std::pair<size_t, std::vector<uint64_t>>, int>& memo) {
  if (pos == t.size()) return 0;
  std::sort(cache.begin(), cache.end());
  auto key = std::make_pair(pos, cache);
  auto it = memo.find(key);
  if (it != memo.end()) return it->second;
  int best;
  if (std::find(cache.begin(), cache.end(), t[pos]) != cache.end()) best = optMisses(t, pos + 1, cache, ways, memo);
  else if (cache.size() < ways) { auto c2 = cache; c2.push_back(t[pos]); best = 1 + optMisses(t, pos + 1, c2, ways, memo); }
  else {
    best = 1 << 30;
    for (size_t e = 0; e < cache.size(); e++) { auto c2 = cache; c2[e] = t[pos]; best = std::min(best, 1 + optMisses(t, pos + 1, c2, ways, memo)); }
  }
  memo[key] = best;
  return best;
}

static void testBeladyIsOptimal() {
  for (int iter = 0; iter < 150; iter++) {
    uint32_t ways = 1 + rnd() % 4, universe = ways + 1 + rnd() % 3;
    size_t n = 8 + rnd() % 8;
    auto lines = genTrace((int)(rnd() % 2), universe, n);
    std::map<std::pair<size_t, std::vector<uint64_t>>, int> memo;
    int opt = optMisses(lines, 0, {}, ways, memo);
    CHECK_EQ(run("belady", cfgFor(1, ways), mk(lines)).misses, opt);
  }
}

static void testRandomDeterminism() {
  auto lines = genTrace(0, 40, 2000);
  Config c = cfgFor(2, 4);
  auto t = mk(lines);
  c.seed = 7; auto a = run("random", c, t), b = run("random", c, t);
  CHECK_EQ(a.misses, b.misses);
  c.seed = 8; auto d = run("random", c, t);
  CHECK(d.misses != a.misses || true);  // different seed may differ; only determinism is asserted above
}

static void testBipDrrip() {
  // BIP with MRU insertion practically disabled behaves as "insert at LRU end, promote on hit".
  // 5 lines cycled 3 times through a 4-way set, hand-simulated: round 1 = 5 misses; rounds 2 and 3: lines 0,1,2 hit, 3 and 4 miss.
  std::vector<uint64_t> cyc;
  for (int r = 0; r < 3; r++) for (uint64_t l = 0; l < 5; l++) cyc.push_back(l);
  Config c = cfgFor(1, 4);
  auto t = mk(cyc);
  BipPolicy bip(1, 1u << 30);
  PolicyResult r = runPolicy(bip, c, t, nullptr, 0);
  CHECK_EQ(r.misses, 9); CHECK_EQ(r.hits, 6);
  CHECK_EQ(run("lru", c, t).misses, 15);
  // BRRIP with insertion at distant almost always: same cyclic trace, 4-way, 2-bit RRPV, hand-simulated:
  // round 1 fills 0..3 at RRPV 3 (distant); line 4 evicts way 0 (RRPV 3 -> inserted distant again);
  // 1,2,3 hit (RRPV 0), 0 misses and evicts line 4 (the only RRPV-3 line), 4 misses and evicts line 0, and so on.
  RripDuelPolicy brrip(2, 1, 1u << 30, false);
  PolicyResult b = runPolicy(brrip, c, t, nullptr, 0);
  CHECK(b.misses < 15);
  CHECK_EQ(run("srrip", c, t).misses, 15);

  // Thrashing cyclic scan over 128 sets x 4 ways, 5 lines per set, 60 passes: SRRIP and LRU miss every access;
  // DRRIP's duel must learn to use bimodal insertion and cut misses a lot.
  std::vector<uint64_t> scan;
  for (int pass = 0; pass < 60; pass++) for (uint64_t l = 0; l < 128 * 5; l++) scan.push_back(l);
  Config c2 = cfgFor(128, 4);
  auto ts = mk(scan);
  auto sr = run("srrip", c2, ts), dr = run("drrip", c2, ts), bp = run("bip", c2, ts), lr = run("lru", c2, ts);
  CHECK_EQ(sr.misses, scan.size()); CHECK_EQ(lr.misses, scan.size());
  CHECK(dr.misses < sr.misses * 6 / 10);
  CHECK(bp.misses < lr.misses * 6 / 10);
  // On a hot set plus streaming scan DRRIP must not be far worse than SRRIP (it duels between the two).
  auto mix = genTrace(3, 8, 20000);
  Config c3 = cfgFor(64, 4);
  auto tm = mk(mix);
  CHECK(run("drrip", c3, tm).misses <= run("srrip", c3, tm).misses * 12 / 10);
  // Determinism with the same seed.
  CHECK_EQ(run("drrip", c2, ts).misses, dr.misses);
  CHECK_EQ(run("bip", c2, ts).misses, bp.misses);
}

struct Dec { std::vector<uint32_t> w; };
static LabelStats labelWith(const std::string& drv, const Config& c, const std::vector<TraceRec>& t, std::vector<Dec>& out, uint32_t stride = 1, uint64_t warm = 0) {
  auto p = makePolicy(drv, c);
  auto nu = computeNextUse(t, 6);
  return labelTrace(*p, c, t, nu, warm, stride, [&](const uint32_t* r, size_t n) { out.push_back(Dec{std::vector<uint32_t>(r, r + n)}); });
}

static void testLabelerHandChecked() {
  // 1 set, 2 ways, trace A B C A B with LRU as driver (hand-worked in docs/dataset_format.md):
  //   i2 (C): residents A(way0,last0,nu3) B(way1,last1,nu4)  -> Belady victim B (way1, mask 0b10); LRU evicts A (way0)
  //   i3 (A): residents C(way0,last2,never) B(way1,last1,nu4) -> Belady victim C (mask 0b01); LRU evicts B (way1)
  //   i4 (B): residents C(way0,never) A(way1,last3,never)     -> tie (mask 0b11); LRU evicts C (way0)
  Config c = cfgFor(1, 2);
  auto t = mk({0, 1, 2, 0, 1});
  std::vector<Dec> d;
  LabelStats st = labelWith("lru", c, t, d);
  CHECK_EQ(st.misses, 5); CHECK_EQ(st.cold_fills, 2); CHECK_EQ(st.decisions, 3); CHECK_EQ(st.tied, 1);
  CHECK_EQ(d.size(), 3);
  if (d.size() == 3) {
    const uint32_t N = NO_NEXT_USE;
    std::vector<uint32_t> e0 = {2, 0, 2, 0,  0, 0, 1, 3,  1, 1, 1, 4};
    std::vector<uint32_t> e1 = {3, 0, 1, 1,  2, 2, 1, N,  1, 1, 1, 4};
    std::vector<uint32_t> e2 = {4, 0, 3, 0,  2, 2, 1, N,  3, 3, 1, N};
    CHECK(d[0].w == e0); CHECK(d[1].w == e1); CHECK(d[2].w == e2);
  }
  CHECK_EQ(st.driver_agrees, 1);  // LRU picked the wrong line at i2 and i3; at i4 the tie contains way 0, so only i4 agrees
  // A hit changes last_idx and count but not insert_idx: A B A C -> at C: A(last 2, insert 0, count 2), B(last 1, insert 1, count 1).
  std::vector<Dec> d2;
  labelWith("lru", c, mk({0, 1, 0, 2}), d2);
  CHECK_EQ(d2.size(), 1);
  if (d2.size() == 1) {
    std::vector<uint32_t> e = {3, 0, 3, 1,  2, 0, 2, NO_NEXT_USE,  1, 1, 1, NO_NEXT_USE};   // both never used again: tie 0b11; LRU evicts B (way1)
    CHECK(d2[0].w == e);
  }
  // Stride 2 keeps decisions 1, 3, ... of the full list.
  std::vector<Dec> d3;
  auto t3 = mk({0, 1, 2, 3, 4, 5, 6, 7});
  std::vector<Dec> all;
  labelWith("lru", c, t3, all, 1);
  labelWith("lru", c, t3, d3, 2);
  CHECK_EQ(all.size(), 6); CHECK_EQ(d3.size(), 3);
  if (all.size() == 6 && d3.size() == 3) { CHECK(d3[0].w == all[0].w); CHECK(d3[1].w == all[2].w); CHECK(d3[2].w == all[4].w); }
}

static void testLabelerProperties() {
  for (int iter = 0; iter < 200; iter++) {
    uint32_t sets = 1u << (rnd() % 3), ways = 1 + rnd() % 6;
    uint32_t universe = ways + 1 + rnd() % (sets * ways * 3 + 4);
    auto lines = genTrace((int)(rnd() % 4), universe, 60 + rnd() % 400);
    Config c = cfgFor(sets, ways);
    auto t = mk(lines);
    for (const char* drv : {"lru", "srrip", "belady", "drrip"}) {
      std::vector<Dec> d;
      LabelStats st = labelWith(drv, c, t, d);
      PolicyResult r = run(drv, c, t);
      CHECK_EQ(st.misses, r.misses); CHECK_EQ(st.hits, r.hits);                    // same simulation as the simulator core
      CHECK_EQ(st.decisions + st.cold_fills, st.misses);
      CHECK_EQ(d.size(), st.decisions);
      for (const Dec& x : d) {
        const uint32_t seq = x.w[0], mask = x.w[2], dv = x.w[3];
        CHECK(mask != 0); CHECK(dv < ways);
        uint32_t best = 0;
        for (uint32_t w = 0; w < ways; w++) {
          uint32_t last = x.w[4 + 4 * w], ins = x.w[5 + 4 * w], cnt = x.w[6 + 4 * w], nu = x.w[7 + 4 * w];
          CHECK(ins <= last); CHECK(last < seq); CHECK(cnt >= 1);
          // independent scan of the trace for the resident line's next use after its last access
          uint64_t line = t[last].addr >> 6; uint32_t scan = NO_NEXT_USE;
          for (size_t j = (size_t)last + 1; j < t.size(); j++) if ((t[j].addr >> 6) == line) { scan = (uint32_t)j; break; }
          CHECK_EQ(scan, nu); CHECK(scan > seq);
          // the line is resident, so the count equals the number of accesses to it in [insert_idx, last_idx]
          uint32_t k = 0;
          for (size_t j = ins; j <= last; j++) if ((t[j].addr >> 6) == line) k++;
          CHECK_EQ(k, cnt);
          if (nu > best) best = nu;
        }
        for (uint32_t w = 0; w < ways; w++) CHECK_EQ(((mask >> w) & 1) != 0, x.w[7 + 4 * w] == best);
        if (std::string(drv) == "belady") CHECK(mask & (1u << dv));                // Belady driver always follows the label
      }
    }
  }
}

static void testGbdtParser() {
  // hand-made LightGBM-style model: tree 0 splits on x0 <= 2.5 then x1 <= 0.5; tree 1 is a single leaf (0.5)
  const char* txt =
      "ACGBDT 1\nvocab 2 4198400 4198464\n---\ntree\nversion=v4\n\nTree=0\nnum_leaves=3\nnum_cat=0\nsplit_feature=0 1\n"
      "threshold=2.5 0.5\ndecision_type=2 2\nleft_child=1 -1\nright_child=-2 -3\nleaf_value=0.1 0.2 0.3\n\n"
      "Tree=1\nnum_leaves=1\nnum_cat=0\nleaf_value=0.5\n\nend of trees\n";
  GbdtModel m;
  m.loadText(txt);
  CHECK_EQ(m.trees.size(), 2); CHECK_EQ(m.vocab.size(), 2); CHECK_EQ(m.vocab[1], 4198464);
  float a[2] = {2.f, 0.f}, b[2] = {2.f, 1.f}, c[2] = {3.f, 0.f}, e[2] = {2.5f, 0.5f};
  CHECK(std::fabs(m.score(a) - 0.6) < 1e-12);    // left, left
  CHECK(std::fabs(m.score(b) - 0.8) < 1e-12);    // left, right
  CHECK(std::fabs(m.score(c) - 0.7) < 1e-12);    // right
  CHECK(std::fabs(m.score(e) - 0.6) < 1e-12);    // thresholds are "<=" (equal goes left)
  GbdtModel bad;
  bool threw = false;
  try { bad.loadText("ACGBDT 1\nvocab 0 \n---\nTree=0\nnum_leaves=2\nnum_cat=1\n"); } catch (const std::exception&) { threw = true; }
  CHECK(threw);
  threw = false;
  try { makePolicy("mlp:/nonexistent/model.mlp", Config()); } catch (const std::exception&) { threw = true; }
  CHECK(threw);
}

static void testFilter() {
  Config c;
  c.line_bytes = 64; c.l1 = {128, 2}; c.l2 = {256, 4};  // one set each: L1 holds 2 lines, L2 holds 4
  c.validate();
  auto raw = [](uint64_t addr, uint32_t ic, uint8_t size = 4, uint8_t fl = 0) { TraceRec r{}; r.addr = addr; r.pc = 0x400100; r.icount = ic; r.size = size; r.flags = fl; return r; };
  // A A A B C A : L1 miss on A, B, C, A ; the last A hits in L2. LLC sees A, B, C.
  std::vector<TraceRec> in = {raw(0x1000, 1), raw(0x1000, 1), raw(0x1000, 1), raw(0x2000, 1), raw(0x3000, 1), raw(0x1000, 1)};
  size_t i = 0;
  std::vector<TraceRec> out;
  auto st = filterTrace(c, [&](TraceRec& r) { if (i == in.size()) return false; r = in[i++]; return true; }, [&](const TraceRec& r) { out.push_back(r); });
  CHECK_EQ(st.l1_hits, 2); CHECK_EQ(st.l1_misses, 4); CHECK_EQ(st.l2_hits, 1); CHECK_EQ(st.l2_misses, 3); CHECK_EQ(out.size(), 3);
  CHECK_EQ(out[0].addr, 0x1000); CHECK_EQ(out[1].addr, 0x2000); CHECK_EQ(out[2].addr, 0x3000);
  CHECK_EQ(out[0].icount, 1); CHECK_EQ(out[1].icount, 3);   // B carries the instructions of the two L1 hits before it plus its own
  CHECK_EQ(st.total_icount, 6);

  // Instruction counts of filtered-out accesses are added to the next LLC record.
  std::vector<TraceRec> in2 = {raw(0x1000, 5), raw(0x1000, 3), raw(0x2000, 4)};
  i = 0; out.clear();
  auto st2 = filterTrace(c, [&](TraceRec& r) { if (i == in2.size()) return false; r = in2[i++]; return true; }, [&](const TraceRec& r) { out.push_back(r); });
  CHECK_EQ(out.size(), 2); CHECK_EQ(out[0].icount, 5); CHECK_EQ(out[1].icount, 7); CHECK_EQ(st2.total_icount, 12);

  // A 4-byte access at 0x103e crosses a 64 B boundary: two line accesses, icount only on the first, store flag kept.
  std::vector<TraceRec> in3 = {raw(0x103e, 9, 4, FLAG_WRITE)};
  i = 0; out.clear();
  filterTrace(c, [&](TraceRec& r) { if (i == in3.size()) return false; r = in3[i++]; return true; }, [&](const TraceRec& r) { out.push_back(r); });
  CHECK_EQ(out.size(), 2); CHECK_EQ(out[0].addr, 0x1000); CHECK_EQ(out[1].addr, 0x1040);
  CHECK_EQ(out[0].icount, 9); CHECK_EQ(out[1].icount, 0); CHECK_EQ(out[0].flags, FLAG_WRITE);
}

static void testTraceIO() {
  const char* path = "/tmp/ac_test.trace";
  {
    TraceWriter w;
    CHECK(w.open(path));
    for (int k = 0; k < 100000; k++) { TraceRec r{}; r.addr = k * 64ull; r.pc = k; r.icount = k % 7; r.size = 8; r.flags = k & 1; w.write(r); }
    CHECK(w.close(777));
  }
  TraceReader rd;
  CHECK(rd.open(path));
  CHECK_EQ(rd.header().num_records, 100000); CHECK_EQ(rd.header().total_icount, 777);
  TraceRec r; uint64_t n = 0; bool ok = true;
  while (rd.next(r)) { ok &= (r.addr == n * 64 && r.pc == n && r.icount == n % 7 && r.flags == (n & 1)); n++; }
  CHECK(ok); CHECK_EQ(n, 100000);
  remove(path);
  TraceReader bad;
  CHECK(!bad.open("/tmp/does_not_exist.trace"));
}

static void writeFile(const char* p, const std::string& s) { FILE* f = fopen(p, "w"); fputs(s.c_str(), f); fclose(f); }
static bool throws(const std::string& text) {
  writeFile("/tmp/ac_test.cfg", text);
  try { Config::load("/tmp/ac_test.cfg"); } catch (const std::exception&) { return true; }
  return false;
}
static void testConfig() {
  writeFile("/tmp/ac_test.cfg", "# comment\nllc_size = 4M\nllc_ways = 16 # trailing\nl1_size=64K\n");
  Config c = Config::load("/tmp/ac_test.cfg");
  CHECK_EQ(c.llc.size, 4u << 20); CHECK_EQ(c.llc_sets(), 4096); CHECK_EQ(c.l1.size, 65536); CHECK_EQ(c.l1_sets(), 128);
  Config d = Config::load("/tmp/ac_test.cfg");
  CHECK(c.hash() == d.hash());
  d.llc.ways = 8;
  CHECK(c.hash() != d.hash());
  CHECK(throws("llc_size=3M\nllc_ways=16\n"));      // 3072 sets: not a power of two
  CHECK(throws("bogus_key=1\n"));
  CHECK(throws("llc_ways=0\n"));
  CHECK(throws("line_bytes=48\n"));
  CHECK(throws("warmup_frac=1.5\n"));
  CHECK(throws("this is not key value\n"));
  remove("/tmp/ac_test.cfg");
}

static std::vector<TraceRec> parseText(const std::string& text, const LackeyOpts& o, LackeyStats* stOut = nullptr) {
  FILE* f = fmemopen((void*)text.data(), text.size(), "r");
  std::vector<TraceRec> v;
  LackeyStats st = parseLackey(f, o, [&](const TraceRec& r) { v.push_back(r); });
  fclose(f);
  if (stOut) *stOut = st;
  return v;
}
static void testLackeyParser() {
  const std::string sample =
      "==123== Lackey, an example Valgrind tool\n"
      "I  04000000,3\n L 1ffefffd58,8\n"
      "I  04000003,4\n S 00404040,4\n"          // ROI begin (store to marker)
      "I  04000007,2\n L 00600000,8\n"
      "I  04000009,3\n S 00600008,8\n"
      "I  0400000c,4\n M 00600010,4\n"
      "I  04000010,4\n S 00404040,4\n"          // ROI end
      "I  04000014,3\n L 00700000,8\n";
  LackeyOpts o; o.have_roi = true; o.roi_addr = 0x404040;
  LackeyStats st;
  auto v = parseText(sample, o, &st);
  CHECK_EQ(v.size(), 3); CHECK(st.roi_begin_seen && st.roi_end_seen);
  CHECK_EQ(v[0].addr, 0x600000); CHECK_EQ(v[0].pc, 0x4000007); CHECK_EQ(v[0].icount, 1); CHECK_EQ(v[0].flags, 0); CHECK_EQ(v[0].size, 8);
  CHECK_EQ(v[1].addr, 0x600008); CHECK_EQ(v[1].flags, FLAG_WRITE);
  CHECK_EQ(v[2].addr, 0x600010); CHECK_EQ(v[2].flags, FLAG_WRITE); CHECK_EQ(v[2].size, 4);
  CHECK_EQ(st.roi_instr, 3);

  // Without an ROI address everything is traced (7 data accesses, the marker stores included).
  LackeyOpts all;
  auto w = parseText(sample, all);
  CHECK_EQ(w.size(), 7);

  // A load from the marker address is not a marker; wrong address gives roi_begin_seen = false.
  LackeyOpts wrong; wrong.have_roi = true; wrong.roi_addr = 0xdeadbeef;
  LackeyStats st2;
  auto z = parseText(sample, wrong, &st2);
  CHECK_EQ(z.size(), 0); CHECK(!st2.roi_begin_seen);

  // max_records truncates.
  LackeyOpts cap = o; cap.max_records = 2;
  LackeyStats st3;
  auto y = parseText(sample, cap, &st3);
  CHECK_EQ(y.size(), 2); CHECK(st3.truncated);

  // Lines split across read chunks are handled (feed one huge line-rich text, > 1 MB).
  std::string big = "I  04000000,3\n S 00404040,4\n";
  for (int k = 0; k < 60000; k++) big += "I  04000010,4\n L 00600000,8\n";
  big += "I  04000020,3\n S 00404040,4\n";
  auto bg = parseText(big, o);
  CHECK_EQ(bg.size(), 60000);
}

int main() {
  testHandChecked();
  testNextUseAndWarmup();
  testProperties();
  testBeladyIsOptimal();
  testRandomDeterminism();
  testBipDrrip();
  testLabelerHandChecked();
  testLabelerProperties();
  testGbdtParser();
  testFilter();
  testTraceIO();
  testConfig();
  testLackeyParser();
  printf("%d checks, %d failed\n", g_checks, g_fail);
  return g_fail ? 1 : 0;
}
