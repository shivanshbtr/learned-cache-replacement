// Multi-application LLC engine (experiments E7 switching and E8 shared LLC). See docs/multi_app.md.
//
// Model of the system (all of it is a simulation choice, stated in the report):
//  * Each application has its own LLC trace (already filtered by L1/L2). A schedule interleaves them: round robin, each turn an application
//    runs until it has consumed `slice_instr` instructions (at least one record), then the next unfinished application runs.
//    The schedule does not depend on hits or misses (there is no timing model), so Belady can be computed on the merged stream.
//  * Every line carries the id of the application that owns it: tag = line | app << 48. The SET is taken from the plain line address, so
//    two applications never hit on each other's lines but do compete for the same sets.
//  * Time is the global access counter of the merged stream. Policy metadata of a line therefore counts accesses of ALL applications.
//  * Arbitration rules (who chooses the victim on a miss by application R in a full set):
//      Global : one policy instance for the whole cache (LRU, SRRIP, Belady, ... the baselines).
//      Req    : R's own model scores every line of the set and may evict any line (also another application's).
//      Own    : ownership. Quota = ways / applications. If R owns at least `quota` lines in the set, R's model evicts one of ITS OWN lines;
//               otherwise the model of the application that owns the most lines in the set gives up one of ITS OWN lines. A model never
//               chooses among lines of another application.
//      Gen    : one generic model for the whole cache.
//  * Per-application model instances see every hit and fill of the merged stream (so their per-line metadata is complete). An instance of
//    application X is shown the PC of an access only if X issued it; PCs of other applications are shown as 0 (unknown code).
#pragma once
#include <array>
#include <functional>
#include "sim.h"

namespace ac {

constexpr uint32_t APP_SHIFT = 48;   // plain line addresses must be < 2^48 (user-space virtual addresses are < 2^47 / 64)
constexpr uint32_t MAX_APPS = 16;

struct AppStream {
  std::string name;
  std::vector<TraceRec> t;
  uint64_t total_icount = 0;
};

struct Schedule {
  std::vector<uint8_t> app;          // application of merged position i
  std::vector<uint32_t> idx;         // record index inside that application's trace
  std::vector<uint32_t> slice_begin; // merged position where each slice starts
  std::vector<uint8_t> slice_app;
  uint64_t switches = 0;             // number of times the active application changed
};

inline Schedule buildSchedule(const std::vector<AppStream>& apps, uint64_t slice_instr, bool until_all) {
  if (apps.empty() || apps.size() > MAX_APPS) throw std::runtime_error("need 1..16 applications");
  if (slice_instr == 0) throw std::runtime_error("slice length must be > 0 instructions");
  Schedule s;
  const size_t n = apps.size();
  std::vector<size_t> cur(n, 0);
  size_t total = 0;
  for (auto& a : apps) total += a.t.size();
  s.app.reserve(total); s.idx.reserve(total);
  int last = -1;
  bool stop = false;
  while (!stop) {
    bool any = false;
    for (size_t a = 0; a < n && !stop; a++) {
      if (cur[a] >= apps[a].t.size()) continue;
      any = true;
      s.slice_begin.push_back((uint32_t)s.app.size());
      s.slice_app.push_back((uint8_t)a);
      if (last >= 0 && last != (int)a) s.switches++;
      last = (int)a;
      uint64_t used = 0;
      while (cur[a] < apps[a].t.size() && used < slice_instr) {
        used += apps[a].t[cur[a]].icount;
        s.app.push_back((uint8_t)a);
        s.idx.push_back((uint32_t)cur[a]);
        cur[a]++;
      }
      if (!until_all && cur[a] >= apps[a].t.size()) stop = true;   // stop as soon as the first application has finished
    }
    if (!any) break;
  }
  return s;
}

inline std::vector<uint32_t> mergedNextUse(const std::vector<AppStream>& apps, const Schedule& s, uint32_t shift) {
  const size_t n = s.app.size();
  if (n >= NO_NEXT_USE) throw std::runtime_error("merged trace too long for 32-bit next_use");
  std::vector<uint32_t> nu(n);
  std::unordered_map<uint64_t, uint32_t> last;
  last.reserve(1 << 18);
  for (size_t i = n; i-- > 0;) {
    const uint64_t key = (apps[s.app[i]].t[s.idx[i]].addr >> shift) | ((uint64_t)s.app[i] << APP_SHIFT);
    auto it = last.find(key);
    nu[i] = (it == last.end()) ? NO_NEXT_USE : it->second;
    last[key] = (uint32_t)i;
  }
  return nu;
}

enum class Rule { Global, Req, Own, Gen };

struct MultiSpec {
  std::string name;
  Rule rule = Rule::Global;
  // Global/Gen: called once with app 0. Req/Own: called once per application with the application's GLOBAL index.
  std::function<std::unique_ptr<Policy>(uint32_t)> make;
};

struct AppStats {
  uint64_t accesses = 0, hits = 0, misses = 0, icount = 0;   // measured (after warm-up)
  uint64_t cold_acc = 0, cold_miss = 0;                       // measured accesses within the first coldK accesses after a switch-in
  uint64_t borrow = 0;                                        // Own rule: misses of this application served by another owner's model
};

struct MultiResult {
  std::string name;
  std::vector<AppStats> app;
  std::vector<std::vector<uint64_t>> evict;   // [requester][owner of the evicted line], measured
  uint64_t hits = 0, misses = 0, icount = 0, accesses = 0, warmup = 0, decisions = 0;
  double wall_s = 0;
  std::vector<std::string> model_stats;       // statsJson of every policy instance
  std::vector<std::array<uint64_t, 3>> slices;// per slice: app, accesses, misses (whole run, warm-up included), only if requested
};

inline MultiResult runMulti(const MultiSpec& spec, const Config& cfg, const std::vector<AppStream>& apps, const std::vector<uint32_t>& appIds,
                            const Schedule& sch, const std::vector<uint32_t>* nu, uint64_t warmup, uint64_t coldK, bool keepSlices) {
  const uint32_t sets = cfg.llc_sets(), ways = cfg.llc.ways, shift = log2u(cfg.line_bytes);
  const uint32_t n = (uint32_t)apps.size();
  std::vector<std::unique_ptr<Policy>> pols;
  const bool perApp = spec.rule == Rule::Req || spec.rule == Rule::Own;
  if (perApp) for (uint32_t a = 0; a < n; a++) pols.push_back(spec.make(appIds[a]));
  else pols.push_back(spec.make(0));
  for (auto& p : pols) {
    if (p->needsNextUse() && !nu) throw std::runtime_error(std::string(p->name()) + " needs next_use");
    p->reset(sets, ways);
  }
  std::vector<uint64_t> tags((size_t)sets * ways, 0);
  std::vector<uint8_t> valid((size_t)sets * ways, 0);
  MultiResult R;
  R.name = spec.name;
  R.app.assign(n, AppStats());
  R.evict.assign(n, std::vector<uint64_t>(n, 0));
  R.accesses = sch.app.size();
  R.warmup = std::min<uint64_t>(warmup, R.accesses);
  const uint32_t quota = std::max<uint32_t>(1, ways / n);
  std::vector<uint8_t> allowed(ways);
  size_t nextSlice = 0;
  uint64_t posInSlice = 0, sliceAcc = 0, sliceMiss = 0;
  uint8_t sliceApp = 0;
  auto t0 = std::chrono::steady_clock::now();
  for (size_t i = 0; i < sch.app.size(); i++) {
    if (nextSlice < sch.slice_begin.size() && sch.slice_begin[nextSlice] == i) {
      if (keepSlices && i > 0) R.slices.push_back({sliceApp, sliceAcc, sliceMiss});
      sliceApp = sch.slice_app[nextSlice]; sliceAcc = sliceMiss = 0; posInSlice = 0; nextSlice++;
    }
    const uint32_t a = sch.app[i];
    const TraceRec& r = apps[a].t[sch.idx[i]];
    const uint64_t raw = r.addr >> shift;
    if ((raw >> APP_SHIFT) != 0) throw std::runtime_error("line address does not fit below 2^48");
    const uint64_t tag = raw | ((uint64_t)a << APP_SHIFT);
    const uint32_t set = (uint32_t)(raw & (sets - 1));
    const size_t b = (size_t)set * ways;
    const AccessCtx ctx{raw, r.pc, i, set, (r.flags & FLAG_WRITE) != 0, nu ? (*nu)[i] : NO_NEXT_USE};
    AccessCtx other = ctx; other.pc = 0;   // what an instance of another application is shown
    auto ctxFor = [&](size_t j) -> const AccessCtx& { return (perApp && j != a) ? other : ctx; };
    const bool measured = i >= warmup;
    const bool cold = posInSlice < coldK;
    int hitw = -1, inv = -1;
    for (uint32_t w = 0; w < ways; w++) {
      if (valid[b + w]) { if (tags[b + w] == tag) { hitw = (int)w; break; } }
      else if (inv < 0) inv = (int)w;
    }
    if (measured) { R.app[a].accesses++; R.app[a].icount += r.icount; R.icount += r.icount; if (cold) R.app[a].cold_acc++; }
    sliceAcc++;
    if (hitw >= 0) {
      for (size_t j = 0; j < pols.size(); j++) pols[j]->onHit(set, (uint32_t)hitw, ctxFor(j));
      if (measured) { R.app[a].hits++; R.hits++; }
    } else {
      uint32_t w;
      if (inv >= 0) w = (uint32_t)inv;
      else {
        if (spec.rule == Rule::Own) {
          uint32_t cnt[MAX_APPS] = {0};
          for (uint32_t q = 0; q < ways; q++) cnt[tags[b + q] >> APP_SHIFT]++;
          uint32_t decider = a;
          if (cnt[a] < quota) {   // under quota: another owner gives up a line (the one with the most lines in the set; lowest id on ties)
            uint32_t best = 0; bool have = false;
            for (uint32_t o = 0; o < n; o++) if (o != a && (!have || cnt[o] > cnt[best])) { best = o; have = true; }
            if (have && cnt[best] > 0) decider = best;
          }
          for (uint32_t q = 0; q < ways; q++) allowed[q] = (tags[b + q] >> APP_SHIFT) == decider;
          w = pols[decider]->victimAllowed(set, ctxFor(decider), &tags[b], allowed.data(), ways);
          if (w >= ways || !allowed[w]) throw std::runtime_error("ownership rule: model chose a way it may not evict");
          if (measured && decider != a) R.app[a].borrow++;
        } else if (spec.rule == Rule::Req) {
          w = pols[a]->victim(set, ctx, &tags[b]);
        } else {
          w = pols[0]->victim(set, ctx, &tags[b]);
        }
        if (w >= ways) throw std::runtime_error(spec.name + " returned an invalid victim way");
        if (measured) R.evict[a][tags[b + w] >> APP_SHIFT]++;
      }
      tags[b + w] = tag; valid[b + w] = 1;
      for (size_t j = 0; j < pols.size(); j++) pols[j]->onFill(set, w, ctxFor(j));
      if (measured) { R.app[a].misses++; R.misses++; if (cold) R.app[a].cold_miss++; }
      sliceMiss++;
    }
    posInSlice++;
  }
  if (keepSlices && !sch.app.empty()) R.slices.push_back({sliceApp, sliceAcc, sliceMiss});
  R.wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  for (auto& p : pols) R.model_stats.push_back(p->statsJson());
  return R;
}

}  // namespace ac
