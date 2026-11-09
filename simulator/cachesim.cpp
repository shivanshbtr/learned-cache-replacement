// cachesim: L1/L2 filter and LLC policy comparison.
//   cachesim filter --config C --in RAW|- --out LLC [--stats S.json]
//   cachesim run    --config C --trace LLC [--policies lru,fifo,random,srrip,belady] [--out R.json] [--max-accesses N]
//   policies: lru fifo random srrip brrip drrip bip belady  mlp:<file.mlp>  gbdt:<file.gbdt>
#include <cstdio>
#include <cstring>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "factory.h"
#include "sim.h"

using namespace ac;

static std::string esc(const std::string& s) {
  std::string o;
  for (char c : s) { if (c == '"' || c == '\\') o += '\\'; o += c; }
  return o;
}

static std::map<std::string, std::string> parseArgs(int argc, char** argv, int from) {
  std::map<std::string, std::string> m;
  for (int i = from; i < argc; i++) {
    std::string k = argv[i];
    if (k.rfind("--", 0) != 0 || i + 1 >= argc) throw std::runtime_error("bad argument: " + k);
    m[k.substr(2)] = argv[++i];
  }
  return m;
}

static std::string configJson(const Config& c) {
  std::ostringstream o;
  o << "{\"hash\":\"" << c.hash() << "\",\"line_bytes\":" << c.line_bytes << ",\"l1_size\":" << c.l1.size
    << ",\"l1_ways\":" << c.l1.ways << ",\"l2_size\":" << c.l2.size << ",\"l2_ways\":" << c.l2.ways
    << ",\"llc_size\":" << c.llc.size << ",\"llc_ways\":" << c.llc.ways << ",\"warmup_frac\":" << c.warmup_frac
    << ",\"seed\":" << c.seed << ",\"srrip_bits\":" << c.srrip_bits << "}";
  return o.str();
}

static int cmdFilter(const std::map<std::string, std::string>& a) {
  if (!a.count("config") || !a.count("in") || !a.count("out")) throw std::runtime_error("filter needs --config --in --out");
  Config cfg = Config::load(a.at("config"));
  TraceReader in;
  if (!in.open(a.at("in"))) throw std::runtime_error("cannot open raw trace (bad file or bad header): " + a.at("in"));
  TraceWriter out;
  if (!out.open(a.at("out"))) throw std::runtime_error("cannot open output: " + a.at("out"));
  FilterStats st = filterTrace(cfg, [&](TraceRec& r) { return in.next(r); }, [&](const TraceRec& r) { out.write(r); });
  if (!out.close(st.total_icount)) throw std::runtime_error("failed to finish output trace");
  fprintf(stderr, "filter: raw=%llu line_accesses=%llu L1 miss=%llu L2 miss=%llu -> LLC accesses=%llu, instructions=%llu\n",
          (unsigned long long)st.raw_records, (unsigned long long)st.line_accesses, (unsigned long long)st.l1_misses,
          (unsigned long long)st.l2_misses, (unsigned long long)st.llc_out, (unsigned long long)st.total_icount);
  if (a.count("stats")) {
    FILE* f = fopen(a.at("stats").c_str(), "w");
    if (!f) throw std::runtime_error("cannot write stats file");
    fprintf(f, "{\"config\":%s,\"raw_records\":%llu,\"line_accesses\":%llu,\"l1_hits\":%llu,\"l1_misses\":%llu,"
               "\"l2_hits\":%llu,\"l2_misses\":%llu,\"llc_accesses\":%llu,\"total_icount\":%llu}\n",
            configJson(cfg).c_str(), (unsigned long long)st.raw_records, (unsigned long long)st.line_accesses,
            (unsigned long long)st.l1_hits, (unsigned long long)st.l1_misses, (unsigned long long)st.l2_hits,
            (unsigned long long)st.l2_misses, (unsigned long long)st.llc_out, (unsigned long long)st.total_icount);
    fclose(f);
  }
  return 0;
}

static int cmdRun(const std::map<std::string, std::string>& a) {
  if (!a.count("config") || !a.count("trace")) throw std::runtime_error("run needs --config --trace");
  Config cfg = Config::load(a.at("config"));
  if (a.count("seed")) cfg.seed = std::stoull(a.at("seed"));
  std::string plist = a.count("policies") ? a.at("policies") : "lru,fifo,random,srrip,drrip,bip,belady";
  std::vector<std::string> names;
  { std::stringstream ss(plist); std::string n; while (std::getline(ss, n, ',')) if (!n.empty()) names.push_back(n); }
  TraceReader in;
  if (!in.open(a.at("trace"))) throw std::runtime_error("cannot open LLC trace: " + a.at("trace"));
  std::vector<TraceRec> t;
  { TraceRec r; while (in.next(r)) t.push_back(r); }
  if (t.empty()) throw std::runtime_error("LLC trace is empty (ROI markers not found, or nothing reached the LLC)");
  if (a.count("max-accesses")) {   // prefix of the trace (used for latency measurements only; never for reported miss counts)
    uint64_t m = std::stoull(a.at("max-accesses"));
    if (m > 0 && m < t.size()) t.resize((size_t)m);
  }
  const uint32_t shift = log2u(cfg.line_bytes);
  std::unordered_set<uint64_t> uniq;
  for (auto& r : t) uniq.insert(r.addr >> shift);
  const uint64_t warm = warmupCount(cfg, t.size());
  std::vector<uint32_t> nu;
  bool needNu = false;
  for (auto& n : names) if (n == "belady") needNu = true;
  if (needNu) nu = computeNextUse(t, shift);

  std::vector<PolicyResult> res;
  for (auto& n : names) {
    auto p = makePolicy(n, cfg);
    if (!p) throw std::runtime_error("unknown policy: " + n);
    res.push_back(runPolicy(*p, cfg, t, needNu ? &nu : nullptr, warm));
  }
  const PolicyResult *lru = nullptr, *bel = nullptr;
  for (auto& r : res) { if (r.name == "lru") lru = &r; if (r.name == "belady") bel = &r; }
  auto gap = [&](const PolicyResult& r, bool& ok) {
    ok = lru && bel && lru->misses > bel->misses;
    return ok ? ((double)lru->misses - (double)r.misses) / ((double)lru->misses - (double)bel->misses) : 0.0;
  };

  uint64_t meas = t.size() - std::min<uint64_t>(warm, t.size());
  printf("trace: %s\nLLC accesses=%zu (warm-up %llu, measured %llu), unique lines=%zu (%.1f KB), config hash=%s\n",
         a.at("trace").c_str(), t.size(), (unsigned long long)warm, (unsigned long long)meas, uniq.size(),
         uniq.size() * cfg.line_bytes / 1024.0, cfg.hash().c_str());
  printf("%-8s %12s %10s %9s %11s %12s\n", "policy", "misses", "hit_rate", "MPKI", "gap_closed", "acc/s");
  for (auto& r : res) {
    bool ok; double g = gap(r, ok);
    if (ok) printf("%-8s %12llu %10.4f %9.3f %10.1f%% %12.0f\n", r.name.c_str(), (unsigned long long)r.misses, r.hit_rate(), r.mpki(), 100 * g, r.acc_per_s());
    else printf("%-8s %12llu %10.4f %9.3f %11s %12.0f\n", r.name.c_str(), (unsigned long long)r.misses, r.hit_rate(), r.mpki(), "n/a", r.acc_per_s());
  }
  for (auto& r : res) if (!r.extra.empty()) printf("  model %s: %s\n", r.name.c_str(), r.extra.c_str());
  if (a.count("out")) {
    FILE* f = fopen(a.at("out").c_str(), "w");
    if (!f) throw std::runtime_error("cannot write result file");
    fprintf(f, "{\"tool\":\"cachesim\",\"version\":\"0.1\",\"config\":%s,\"trace\":{\"path\":\"%s\",\"llc_accesses\":%zu,"
               "\"warmup_accesses\":%llu,\"measured_accesses\":%llu,\"unique_lines\":%zu,\"header_total_icount\":%llu},\"policies\":[",
            configJson(cfg).c_str(), esc(a.at("trace")).c_str(), t.size(), (unsigned long long)warm, (unsigned long long)meas,
            uniq.size(), (unsigned long long)in.header().total_icount);
    for (size_t i = 0; i < res.size(); i++) {
      bool ok; double g = gap(res[i], ok);
      fprintf(f, "%s{\"name\":\"%s\",\"hits\":%llu,\"misses\":%llu,\"hit_rate\":%.6f,\"mpki\":%.6f,\"measured_icount\":%llu,"
                 "\"gap_closed\":", i ? "," : "", res[i].name.c_str(), (unsigned long long)res[i].hits, (unsigned long long)res[i].misses,
              res[i].hit_rate(), res[i].mpki(), (unsigned long long)res[i].measured_icount);
      if (ok) fprintf(f, "%.6f", g); else fprintf(f, "null");
      fprintf(f, ",\"wall_s\":%.6f,\"acc_per_s\":%.1f", res[i].wall_s, res[i].acc_per_s());
      if (!res[i].extra.empty()) fprintf(f, ",\"model\":%s", res[i].extra.c_str());
      fprintf(f, "}");
    }
    fprintf(f, "]}\n");
    fclose(f);
  }
  return 0;
}

int main(int argc, char** argv) {
  try {
    if (argc < 2) throw std::runtime_error("usage: cachesim filter|run ... (see docs/)");
    std::string cmd = argv[1];
    if (cmd == "version") { puts("cachesim 0.1"); return 0; }
    auto a = parseArgs(argc, argv, 2);
    if (cmd == "filter") return cmdFilter(a);
    if (cmd == "run") return cmdRun(a);
    throw std::runtime_error("unknown command: " + cmd);
  } catch (const std::exception& e) {
    fprintf(stderr, "cachesim: %s\n", e.what());
    return 1;
  }
}
