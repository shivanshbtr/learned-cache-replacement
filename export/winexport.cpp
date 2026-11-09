// winexport: sample-window exporter for the web simulation (no new experiment: it only replays an existing LLC trace).
//   winexport --config C --trace LLC.trace --policies p1,p2,.. [--labels l1,l2,..] [--apps hash,graph,..]
//             [--start-frac 0.5 | --start N] [--len 4000] [--sets 32] [--inspect p1,p2] [--max-decisions 150] --out W.json
// Every policy is replayed from the START of the trace (so its cache state is the true state), and the window [start, start+len) is recorded:
//   per access: set, line id, application, PC id, store flag;  per policy: way and hit flag of every access;
//   initial content of the `sets` busiest sets of the window (per policy);
//   for the policies listed in --inspect: the first --max-decisions eviction decisions of the window with all candidate lines, their
//   features, the model scores (learned policies), the victim chosen, the LRU victim and the Belady victim(s) of the SAME state.
// Policy names are the ones of cachesim/labeler (lru, drrip, belady, mlp:F, gbdt:F, lstm:F, req=.., own=.., gen=..).
// The window never starts inside the warm-up. The numbers of a window are a SAMPLE; the web page labels them so.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <sstream>
#include <unordered_map>
#include "../simulator/factory.h"
#include "../simulator/sim.h"

using namespace ac;

static std::vector<std::string> splitc(const std::string& s) {
  std::vector<std::string> v; std::stringstream ss(s); std::string x;
  while (std::getline(ss, x, ',')) if (!x.empty()) v.push_back(x);
  return v;
}
static std::string jstr(const std::string& s) {
  std::string o = "\"";
  for (char c : s) { if (c == '"' || c == '\\') o += '\\'; o += c; }
  return o + "\"";
}
static std::string hex(uint64_t v) { char b[24]; snprintf(b, sizeof b, "%llx", (unsigned long long)v); return b; }

struct PolOut {
  std::string name, label;
  std::vector<uint8_t> o;                       // way*2 + hit
  std::vector<int> init;                        // sets_shown * ways line ids (-1 empty)
  uint64_t w_hits = 0, w_misses = 0, measured_misses = 0;
  std::string decisions;                        // JSON array body
  uint32_t ndec = 0;
  bool inspected = false;
};

int main(int argc, char** argv) {
  try {
    std::map<std::string, std::string> a;
    for (int i = 1; i < argc; i += 2) {
      std::string k = argv[i];
      if (k.rfind("--", 0) != 0 || i + 1 >= argc) throw std::runtime_error("bad argument: " + k);
      a[k.substr(2)] = argv[i + 1];
    }
    if (!a.count("config") || !a.count("trace") || !a.count("policies") || !a.count("out"))
      throw std::runtime_error("usage: winexport --config C --trace T --policies p1,p2 [--labels ..] [--apps ..] [--start-frac F|--start N] [--len L] [--sets S] [--inspect p1,..] [--max-decisions D] --out W.json");
    Config cfg = Config::load(a.at("config"));
    auto names = splitc(a.at("policies"));
    auto labels = a.count("labels") ? splitc(a.at("labels")) : names;
    if (labels.size() != names.size()) throw std::runtime_error("--labels must have one entry per policy");
    auto apps = a.count("apps") ? splitc(a.at("apps")) : std::vector<std::string>{};
    auto insp = a.count("inspect") ? splitc(a.at("inspect")) : std::vector<std::string>{};
    const size_t L = a.count("len") ? std::stoull(a.at("len")) : 4000;
    const size_t NS = a.count("sets") ? std::stoul(a.at("sets")) : 32;
    const uint32_t MAXD = a.count("max-decisions") ? (uint32_t)std::stoul(a.at("max-decisions")) : 150;

    TraceReader in;
    if (!in.open(a.at("trace"))) throw std::runtime_error("cannot open LLC trace: " + a.at("trace"));
    std::vector<TraceRec> t;
    { TraceRec r; while (in.next(r)) t.push_back(r); }
    if (t.empty()) throw std::runtime_error("LLC trace is empty");
    const uint32_t shift = log2u(cfg.line_bytes), sets = cfg.llc_sets(), ways = cfg.llc.ways;
    if (ways > 32) throw std::runtime_error("at most 32 ways");
    const size_t N = t.size();
    const uint64_t warm = warmupCount(cfg, N);
    auto nu = computeNextUse(t, shift);

    size_t len = std::min(L, N);
    size_t start = a.count("start") ? std::stoull(a.at("start")) : (size_t)((a.count("start-frac") ? std::stod(a.at("start-frac")) : 0.5) * (double)N);
    if (start < warm) start = warm;
    if (start + len > N) start = N - len;
    if (start < warm && !a.count("allow-warmup")) throw std::runtime_error("trace too short for a window after the warm-up");
    const size_t end = start + len;

    // busiest sets of the window
    std::map<uint32_t, uint32_t> cnt;
    for (size_t i = start; i < end; i++) cnt[(uint32_t)((t[i].addr >> shift) & (sets - 1))]++;
    std::vector<std::pair<uint32_t, uint32_t>> cs(cnt.begin(), cnt.end());   // (set, count)
    std::stable_sort(cs.begin(), cs.end(), [](auto& x, auto& y) { return x.second > y.second; });
    if (cs.size() > NS) cs.resize(NS);
    std::vector<uint32_t> shown; for (auto& c : cs) shown.push_back(c.first);
    std::unordered_map<uint32_t, int> shownIdx; for (size_t i = 0; i < shown.size(); i++) shownIdx[shown[i]] = (int)i;

    // line ids (local to this file) and PC ids
    std::unordered_map<uint64_t, int> lid; std::vector<uint64_t> lines;
    auto LID = [&](uint64_t line) { auto it = lid.find(line); if (it != lid.end()) return it->second; int id = (int)lines.size(); lid[line] = id; lines.push_back(line); return id; };
    std::unordered_map<uint64_t, int> pid; std::vector<uint64_t> pcs;
    auto PID = [&](uint64_t pc) { auto it = pid.find(pc); if (it != pid.end()) return it->second; int id = (int)pcs.size(); pid[pc] = id; pcs.push_back(pc); return id; };

    std::vector<int> acc_set(len), acc_line(len), acc_pc(len); std::vector<uint8_t> acc_wr(len);
    for (size_t i = start; i < end; i++) {
      const uint64_t line = t[i].addr >> shift;
      acc_set[i - start] = (int)(line & (sets - 1)); acc_line[i - start] = LID(line);
      acc_pc[i - start] = PID(t[i].pc & MIX_PC_MASK); acc_wr[i - start] = (t[i].flags & FLAG_WRITE) ? 1 : 0;
    }

    std::vector<PolOut> outs(names.size());
    for (size_t pi = 0; pi < names.size(); pi++) {
      PolOut& po = outs[pi]; po.name = names[pi]; po.label = labels[pi]; po.o.assign(len, 0);
      po.inspected = std::find(insp.begin(), insp.end(), names[pi]) != insp.end();
      auto pol = makePolicy(names[pi], cfg);
      if (!pol) throw std::runtime_error("unknown policy: " + names[pi]);
      pol->reset(sets, ways);
      const size_t n = (size_t)sets * ways;
      std::vector<uint64_t> tags(n, 0); std::vector<uint8_t> valid(n, 0);
      std::vector<uint32_t> last, ins, cn, nxt;
      if (po.inspected) { last.assign(n, 0); ins.assign(n, 0); cn.assign(n, 0); nxt.assign(n, NO_NEXT_USE); }
      std::ostringstream dec;
      for (size_t i = 0; i < end; i++) {
        if (i == start) {                                  // snapshot of the shown sets before the first window access
          po.init.assign(shown.size() * ways, -1);
          for (size_t s = 0; s < shown.size(); s++) for (uint32_t w = 0; w < ways; w++) {
            size_t b = (size_t)shown[s] * ways + w;
            if (valid[b]) po.init[s * ways + w] = LID(tags[b]);
          }
        }
        const uint64_t line = t[i].addr >> shift;
        const uint32_t set = (uint32_t)(line & (sets - 1));
        const size_t b = (size_t)set * ways;
        AccessCtx ctx{line, t[i].pc, i, set, (t[i].flags & FLAG_WRITE) != 0, nu[i]};
        int hitw = -1, inv = -1;
        for (uint32_t w = 0; w < ways; w++) {
          if (valid[b + w]) { if (tags[b + w] == line) { hitw = (int)w; break; } }
          else if (inv < 0) inv = (int)w;
        }
        const bool inwin = i >= start;
        const bool measured = i >= warm;
        if (hitw >= 0) {
          pol->onHit(set, (uint32_t)hitw, ctx);
          if (po.inspected) { last[b + hitw] = (uint32_t)i; nxt[b + hitw] = nu[i]; if (cn[b + hitw] != 0xFFFFFFFFu) cn[b + hitw]++; }
          if (inwin) { po.o[i - start] = (uint8_t)(hitw * 2 + 1); po.w_hits++; }
          continue;
        }
        if (measured) po.measured_misses++;
        uint32_t w;
        if (inv >= 0) w = (uint32_t)inv;
        else {
          DbgSink& dbg = dbgSink();
          const bool rec = po.inspected && inwin && po.ndec < MAXD;
          if (rec) { dbg.on = true; for (uint32_t x = 0; x < 32; x++) dbg.has[x] = false; }
          w = pol->victim(set, ctx, &tags[b]);
          dbg.on = false;
          if (w >= ways) throw std::runtime_error(names[pi] + " returned an invalid victim way");
          if (rec) {
            uint32_t lruw = 0, bestn = 0;
            for (uint32_t x = 1; x < ways; x++) if (last[b + x] < last[b + lruw]) lruw = x;
            for (uint32_t x = 0; x < ways; x++) if (nxt[b + x] > bestn) bestn = nxt[b + x];
            if (po.ndec) dec << ",";
            dec << "{\"i\":" << (i - start) << ",\"s\":" << set << ",\"c\":" << LID(line) << ",\"pc\":" << PID(t[i].pc & MIX_PC_MASK)
                << ",\"v\":" << w << ",\"lru\":" << lruw << ",\"bel\":[";
            bool first = true;
            for (uint32_t x = 0; x < ways; x++) if (nxt[b + x] == bestn) { dec << (first ? "" : ",") << x; first = false; }
            dec << "],\"cand\":[";
            for (uint32_t x = 0; x < ways; x++) {
              const uint64_t tl = tags[b + x];
              char buf[512]; int k = 0;
              long nx = nxt[b + x] == NO_NEXT_USE ? -1 : (long)(nxt[b + x] - i);
              k += snprintf(buf + k, sizeof buf - k, "%s[%d,%llu,%llu,%u,%ld", x ? "," : "", LID(tl), (unsigned long long)(tl >> MIX_APP_SHIFT),
                            (unsigned long long)(i - last[b + x]), cn[b + x], nx);
              if (dbg.has[x]) {
                k += snprintf(buf + k, sizeof buf - k, ",%u,%.5g", dbg.lpc[x], dbg.s[x]);
                for (int q = 0; q < LP_NUM; q++) k += snprintf(buf + k, sizeof buf - k, ",%.3g", (double)dbg.f[x][q]);
              }
              k += snprintf(buf + k, sizeof buf - k, "]");
              dec << buf;
            }
            dec << "]}";
            po.ndec++;
          }
        }
        tags[b + w] = line; valid[b + w] = 1;
        if (po.inspected) { last[b + w] = ins[b + w] = (uint32_t)i; cn[b + w] = 1; nxt[b + w] = nu[i]; }
        pol->onFill(set, w, ctx);
        if (inwin) { po.o[i - start] = (uint8_t)(w * 2); po.w_misses++; }
      }
      po.decisions = dec.str();
    }

    FILE* f = fopen(a.at("out").c_str(), "w");
    if (!f) throw std::runtime_error("cannot write " + a.at("out"));
    fprintf(f, "{\"schema\":1,\"tool\":\"winexport\",\"note\":\"SAMPLE WINDOW of an LLC trace; not a full run\",\"trace\":%s,\"config_hash\":\"%s\","
               "\"llc_size\":%llu,\"ways\":%u,\"sets\":%u,\"trace_len\":%zu,\"warmup\":%llu,\"start\":%zu,\"len\":%zu,\n",
            jstr(a.at("trace")).c_str(), cfg.hash().c_str(), (unsigned long long)cfg.llc.size, ways, sets, N, (unsigned long long)warm, start, len);
    fprintf(f, "\"apps\":[");
    for (size_t i = 0; i < apps.size(); i++) fprintf(f, "%s%s", i ? "," : "", jstr(apps[i]).c_str());
    fprintf(f, "],\"feature_names\":[\"log2 age since last access\",\"log2 age since insert\",\"log2 access count\",\"reused\",\"log2 last reuse gap\",\"recency rank\",\"age minus set mean\",\"last access was store\",\"current access is store\"],"
               "\"cand_fields\":[\"line\",\"app\",\"age\",\"count\",\"next_use_distance\",\"pc_id\",\"score\",\"f0\",\"f1\",\"f2\",\"f3\",\"f4\",\"f5\",\"f6\",\"f7\",\"f8\"],\n");
    fprintf(f, "\"shown_sets\":[");
    for (size_t i = 0; i < shown.size(); i++) fprintf(f, "%s%u", i ? "," : "", shown[i]);
    fprintf(f, "],\n\"lines\":[");
    for (size_t i = 0; i < lines.size(); i++) fprintf(f, "%s\"%s\"", i ? "," : "", hex(lines[i] & MIX_LINE_MASK).c_str());
    fprintf(f, "],\"line_app\":[");
    for (size_t i = 0; i < lines.size(); i++) fprintf(f, "%s%llu", i ? "," : "", (unsigned long long)(lines[i] >> MIX_APP_SHIFT));
    fprintf(f, "],\"pcs\":[");
    for (size_t i = 0; i < pcs.size(); i++) fprintf(f, "%s\"%s\"", i ? "," : "", hex(pcs[i]).c_str());
    fprintf(f, "],\n\"acc\":{\"set\":[");
    for (size_t i = 0; i < len; i++) fprintf(f, "%s%d", i ? "," : "", acc_set[i]);
    fprintf(f, "],\"line\":[");
    for (size_t i = 0; i < len; i++) fprintf(f, "%s%d", i ? "," : "", acc_line[i]);
    fprintf(f, "],\"pc\":[");
    for (size_t i = 0; i < len; i++) fprintf(f, "%s%d", i ? "," : "", acc_pc[i]);
    fprintf(f, "],\"wr\":[");
    for (size_t i = 0; i < len; i++) fprintf(f, "%s%d", i ? "," : "", (int)acc_wr[i]);
    fprintf(f, "]},\n\"policies\":[\n");
    for (size_t pi = 0; pi < outs.size(); pi++) {
      PolOut& po = outs[pi];
      fprintf(f, "%s{\"name\":%s,\"label\":%s,\"window_hits\":%llu,\"window_misses\":%llu,\"measured_misses_to_window_end\":%llu,\"o\":[",
              pi ? ",\n" : "", jstr(po.name).c_str(), jstr(po.label).c_str(), (unsigned long long)po.w_hits, (unsigned long long)po.w_misses,
              (unsigned long long)po.measured_misses);
      for (size_t i = 0; i < len; i++) fprintf(f, "%s%d", i ? "," : "", (int)po.o[i]);
      fprintf(f, "],\"init\":[");
      for (size_t i = 0; i < po.init.size(); i++) fprintf(f, "%s%d", i ? "," : "", po.init[i]);
      fprintf(f, "]");
      if (po.inspected) fprintf(f, ",\"decisions\":[%s]", po.decisions.c_str());
      fprintf(f, "}");
    }
    fprintf(f, "\n]}\n");
    if (fclose(f) != 0) throw std::runtime_error("failed to finish output");
    printf("winexport: trace=%zu accesses, warm-up %llu, window [%zu,%zu), %zu shown sets, %zu local lines, %zu policies\n", N,
           (unsigned long long)warm, start, end, shown.size(), lines.size(), outs.size());
    for (auto& po : outs) printf("  %-40s window hits %llu misses %llu%s\n", po.name.c_str(), (unsigned long long)po.w_hits, (unsigned long long)po.w_misses,
                                 po.inspected ? ("  (inspector: " + std::to_string(po.ndec) + " decisions)").c_str() : "");
    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "winexport: %s\n", e.what());
    return 1;
  }
}
