// multisim: several applications on one LLC (E7 switching, E8 shared LLC). See docs/multi_app.md.
//   multisim run --config C --app NAME=LLC.trace [--app ...] --slice-instr N [--until first|all] [--policies P1,P2,...]
//                [--set NAME=app:file,app:file,...] [--gen NAME=file] [--alone 1] [--cold-k K] [--slices 1] [--out R.json]
//   policy tokens: baselines (lru fifo random srrip brrip drrip bip mru belady),
//                  req:SET   each application's own model of SET chooses among ALL lines of the set
//                  own:SET   ownership rule: a model only ever evicts lines of its own application (quota = ways / apps)
//                  gen:NAME  one generic model (--gen NAME=file) for the whole cache
//   multisim merge --config C --app NAME=LLC.trace [--app ...] --slice-instr N [--until first|all] --out M.llc.trace [--meta M.json]
//                  writes the merged stream of `run` as ONE LLC trace: application id a in line bits 48+ and PC bits 56+ (for the labeler and
//                  the composite drivers req=/own=/gen= in simulator/composite.h; used by mixed-stream DAgger, docs/mixed_dagger.md)
//   model files: .mlp .gbdt .lstm (kind by extension).      multisim info --trace T ...  prints records and instructions per trace.
#include <cstdio>
#include <map>
#include <sstream>
#include "factory.h"
#include "multi.h"

using namespace ac;

static std::string esc(const std::string& s) {
  std::string o;
  for (char c : s) { if (c == '"' || c == '\\') o += '\\'; o += c; }
  return o;
}

static std::unique_ptr<Policy> fromFile(const std::string& f) {
  auto ends = [&](const char* e) { size_t n = strlen(e); return f.size() >= n && f.compare(f.size() - n, n, e) == 0; };
  std::string spec = ends(".mlp") ? "mlp:" + f : ends(".gbdt") ? "gbdt:" + f : ends(".lstm") ? "lstm:" + f : "";
  if (spec.empty()) throw std::runtime_error("unknown model file type (need .mlp .gbdt .lstm): " + f);
  Config c;
  auto p = makePolicy(spec, c);
  if (!p) throw std::runtime_error("cannot load model: " + f);
  return p;
}

static void loadTrace(const std::string& path, AppStream& s) {
  TraceReader in;
  if (!in.open(path)) throw std::runtime_error("cannot open LLC trace: " + path);
  TraceRec r;
  while (in.next(r)) s.t.push_back(r);
  s.total_icount = in.header().total_icount;
  if (s.t.empty()) throw std::runtime_error("LLC trace is empty: " + path);
}

static std::string resJson(const MultiResult& r, const std::vector<AppStream>& apps, const std::vector<uint32_t>& ids, bool slices) {
  std::ostringstream o;
  auto mpki = [](uint64_t m, uint64_t ic) { return ic ? 1000.0 * (double)m / (double)ic : 0.0; };
  o << "{\"policy\":\"" << esc(r.name) << "\",\"hits\":" << r.hits << ",\"misses\":" << r.misses << ",\"measured_icount\":" << r.icount
    << ",\"mpki\":" << mpki(r.misses, r.icount) << ",\"wall_s\":" << r.wall_s << ",\"apps\":[";
  for (size_t a = 0; a < r.app.size(); a++) {
    const AppStats& s = r.app[a];
    o << (a ? "," : "") << "{\"app\":\"" << esc(apps[a].name) << "\",\"accesses\":" << s.accesses << ",\"hits\":" << s.hits << ",\"misses\":" << s.misses
      << ",\"icount\":" << s.icount << ",\"mpki\":" << mpki(s.misses, s.icount) << ",\"cold_acc\":" << s.cold_acc << ",\"cold_miss\":" << s.cold_miss
      << ",\"borrow\":" << s.borrow << "}";
  }
  o << "],\"evict\":[";
  for (size_t a = 0; a < r.evict.size(); a++) {
    o << (a ? "," : "") << "[";
    for (size_t b = 0; b < r.evict[a].size(); b++) o << (b ? "," : "") << r.evict[a][b];
    o << "]";
  }
  o << "],\"models\":[";
  bool first = true;
  for (auto& m : r.model_stats) if (!m.empty()) { o << (first ? "" : ",") << m; first = false; }
  o << "]";
  if (slices) {
    o << ",\"slices\":[";
    for (size_t i = 0; i < r.slices.size(); i++) o << (i ? "," : "") << "[" << r.slices[i][0] << "," << r.slices[i][1] << "," << r.slices[i][2] << "]";
    o << "]";
  }
  (void)ids;
  o << "}";
  return o.str();
}

int main(int argc, char** argv) {
  try {
    if (argc < 2) throw std::runtime_error("usage: multisim run|info ... (see the header of simulator/multisim.cpp)");
    const std::string cmd = argv[1];
    std::multimap<std::string, std::string> A;
    for (int i = 2; i < argc; i++) {
      std::string k = argv[i];
      if (k.rfind("--", 0) != 0 || i + 1 >= argc) throw std::runtime_error("bad argument: " + k);
      A.insert({k.substr(2), argv[++i]});
    }
    auto get = [&](const std::string& k, const std::string& d = "") { auto it = A.find(k); return it == A.end() ? d : it->second; };
    if (cmd == "info") {
      auto r = A.equal_range("trace");
      for (auto it = r.first; it != r.second; ++it) {
        AppStream s; loadTrace(it->second, s);
        printf("%s records=%zu total_icount=%llu\n", it->second.c_str(), s.t.size(), (unsigned long long)s.total_icount);
      }
      return 0;
    }
    if (cmd != "run" && cmd != "merge") throw std::runtime_error("unknown command: " + cmd);
    if (cmd == "merge") {
      if (get("config").empty() || !A.count("app") || get("slice-instr").empty() || get("out").empty()) throw std::runtime_error("merge needs --config, --app, --slice-instr and --out");
      Config cfg = Config::load(get("config"));
      std::vector<AppStream> apps;
      auto r = A.equal_range("app");
      for (auto it = r.first; it != r.second; ++it) {
        size_t eq = it->second.find('=');
        if (eq == std::string::npos) throw std::runtime_error("--app needs NAME=TRACE");
        AppStream s; s.name = it->second.substr(0, eq);
        loadTrace(it->second.substr(eq + 1), s);
        apps.push_back(std::move(s));
      }
      const uint32_t shift = log2u(cfg.line_bytes);
      Schedule sch = buildSchedule(apps, std::stoull(get("slice-instr")), get("until", "first") == "all");
      TraceWriter w;
      if (!w.open(get("out"))) throw std::runtime_error("cannot write " + get("out"));
      uint64_t ic = 0;
      for (size_t i = 0; i < sch.app.size(); i++) {
        TraceRec rec = apps[sch.app[i]].t[sch.idx[i]];
        if ((rec.addr >> shift) >> APP_SHIFT) throw std::runtime_error("line address does not fit below 2^48");
        if (rec.pc >> 56) throw std::runtime_error("pc does not fit below 2^56");
        rec.addr |= (uint64_t)sch.app[i] << (APP_SHIFT + shift);
        rec.pc |= (uint64_t)sch.app[i] << 56;
        ic += rec.icount;
        w.write(rec);
      }
      if (!w.close(ic)) throw std::runtime_error("failed to finish " + get("out"));
      if (!get("meta").empty()) {
        FILE* f = fopen(get("meta").c_str(), "w");
        if (!f) throw std::runtime_error("cannot write meta file");
        fprintf(f, "{\"apps\":[");
        for (size_t a = 0; a < apps.size(); a++) fprintf(f, "%s\"%s\"", a ? "," : "", esc(apps[a].name).c_str());
        fprintf(f, "],\"slice_instr\":%s,\"merged_accesses\":%zu,\"switches\":%llu,\"until\":\"%s\",\"config_hash\":\"%s\"}\n", get("slice-instr").c_str(), sch.app.size(),
                (unsigned long long)sch.switches, get("until", "first").c_str(), cfg.hash().c_str());
        fclose(f);
      }
      printf("merged %zu apps -> %zu accesses, %llu switches\n", apps.size(), sch.app.size(), (unsigned long long)sch.switches);
      return 0;
    }
    if (get("config").empty() || !A.count("app") || get("slice-instr").empty()) throw std::runtime_error("run needs --config, at least one --app and --slice-instr");
    Config cfg = Config::load(get("config"));
    std::vector<AppStream> apps;
    {
      auto r = A.equal_range("app");
      for (auto it = r.first; it != r.second; ++it) {
        size_t eq = it->second.find('=');
        if (eq == std::string::npos) throw std::runtime_error("--app needs NAME=TRACE");
        AppStream s; s.name = it->second.substr(0, eq);
        loadTrace(it->second.substr(eq + 1), s);
        apps.push_back(std::move(s));
      }
    }
    std::map<std::string, std::map<std::string, std::string>> sets;   // set name -> app name -> file
    {
      auto r = A.equal_range("set");
      for (auto it = r.first; it != r.second; ++it) {
        size_t eq = it->second.find('=');
        if (eq == std::string::npos) throw std::runtime_error("--set needs NAME=app:file,app:file");
        std::stringstream ss(it->second.substr(eq + 1)); std::string item;
        while (std::getline(ss, item, ',')) {
          size_t c = item.find(':');
          if (c == std::string::npos) throw std::runtime_error("--set item needs app:file");
          sets[it->second.substr(0, eq)][item.substr(0, c)] = item.substr(c + 1);
        }
      }
    }
    std::map<std::string, std::string> gens;
    {
      auto r = A.equal_range("gen");
      for (auto it = r.first; it != r.second; ++it) {
        size_t eq = it->second.find('=');
        if (eq == std::string::npos) throw std::runtime_error("--gen needs NAME=file");
        gens[it->second.substr(0, eq)] = it->second.substr(eq + 1);
      }
    }
    std::vector<MultiSpec> specs;
    {
      std::stringstream ss(get("policies", "lru,belady")); std::string tok;
      while (std::getline(ss, tok, ',')) {
        if (tok.empty()) continue;
        MultiSpec m; m.name = tok;
        if (tok.rfind("req:", 0) == 0 || tok.rfind("own:", 0) == 0) {
          m.rule = tok[0] == 'r' ? Rule::Req : Rule::Own;
          const std::string sn = tok.substr(4);
          if (!sets.count(sn)) throw std::runtime_error("unknown model set in " + tok + " (use --set)");
          auto files = sets[sn];
          auto names = std::make_shared<std::vector<std::string>>();
          for (auto& a : apps) names->push_back(a.name);
          m.make = [files, names, tok](uint32_t gid) {
            auto it = files.find((*names)[gid]);
            if (it == files.end()) throw std::runtime_error(tok + ": no model for application " + (*names)[gid]);
            return fromFile(it->second);
          };
        } else if (tok.rfind("gen:", 0) == 0) {
          m.rule = Rule::Gen;
          const std::string gn = tok.substr(4);
          if (!gens.count(gn)) throw std::runtime_error("unknown generic model in " + tok + " (use --gen)");
          const std::string f = gens[gn];
          m.make = [f](uint32_t) { return fromFile(f); };
        } else {
          m.rule = Rule::Global;
          if (!makePolicy(tok, cfg)) throw std::runtime_error("unknown policy: " + tok);
          m.make = [tok, cfg](uint32_t) { return makePolicy(tok, cfg); };
        }
        specs.push_back(m);
      }
    }
    const bool untilAll = get("until", "first") == "all";
    const uint64_t slice = std::stoull(get("slice-instr"));
    const uint64_t coldK = std::stoull(get("cold-k", "2000"));
    bool keepSlices = get("slices", "0") == "1";
    const bool alone = get("alone", "0") == "1";
    const uint32_t shift = log2u(cfg.line_bytes);
    Schedule sch = buildSchedule(apps, slice, untilAll);
    bool needNu = false;
    for (auto& s : specs) if (s.name == "belady") needNu = true;
    std::vector<uint32_t> nu;
    if (needNu) nu = mergedNextUse(apps, sch, shift);
    const uint64_t warm = warmupCount(cfg, sch.app.size());
    std::vector<uint32_t> ids(apps.size());
    for (uint32_t a = 0; a < apps.size(); a++) ids[a] = a;
    uint64_t slicesN = sch.slice_begin.size();
    if (slicesN > 50000) keepSlices = false;   // too many to store; the timeline is only for long slices

    printf("apps:"); for (auto& a : apps) printf(" %s(%zu acc, %llu instr)", a.name.c_str(), a.t.size(), (unsigned long long)a.total_icount);
    printf("\nmerged accesses=%zu (warm-up %llu), slice=%llu instr, slices=%llu, switches=%llu, until=%s, config hash=%s\n", sch.app.size(),
           (unsigned long long)warm, (unsigned long long)slice, (unsigned long long)slicesN, (unsigned long long)sch.switches, untilAll ? "all" : "first", cfg.hash().c_str());
    std::vector<MultiResult> res;
    std::vector<std::vector<MultiResult>> aloneRes;
    std::map<std::string, std::vector<MultiResult>> aloneCache;
    for (auto& sp : specs) {
      res.push_back(runMulti(sp, cfg, apps, ids, sch, needNu ? &nu : nullptr, warm, coldK, keepSlices));
      std::vector<MultiResult> al;
      std::string akey = sp.name;   // one application alone: req:X, own:X and the model X itself are the same run
      if (akey.rfind("own:", 0) == 0) akey = "req:" + akey.substr(4);
      if (alone && aloneCache.count(akey)) al = aloneCache[akey];
      else if (alone) {
        for (uint32_t a = 0; a < apps.size(); a++) {
          std::vector<AppStream> one{apps[a]};
          Schedule s1 = buildSchedule(one, ~0ull, true);   // one slice = the whole trace
          std::vector<uint32_t> nu1;
          if (needNu) nu1 = mergedNextUse(one, s1, shift);
          al.push_back(runMulti(sp, cfg, one, std::vector<uint32_t>{a}, s1, needNu ? &nu1 : nullptr, warmupCount(cfg, s1.app.size()), coldK, false));
        }
      }
      if (alone) aloneCache[akey] = al;
      aloneRes.push_back(al);
      const MultiResult& r = res.back();
      printf("%-26s misses=%-9llu MPKI=%-7.3f |", r.name.c_str(), (unsigned long long)r.misses, r.icount ? 1000.0 * (double)r.misses / (double)r.icount : 0.0);
      for (size_t a = 0; a < apps.size(); a++) {
        printf(" %s %llu (%.2f)", apps[a].name.c_str(), (unsigned long long)r.app[a].misses, r.app[a].icount ? 1000.0 * (double)r.app[a].misses / (double)r.app[a].icount : 0.0);
      }
      printf("  [%.1fs]\n", r.wall_s);
    }
    if (!get("out").empty()) {
      FILE* f = fopen(get("out").c_str(), "w");
      if (!f) throw std::runtime_error("cannot write result file");
      fprintf(f, "{\"tool\":\"multisim\",\"version\":\"0.1\",\"config_hash\":\"%s\",\"llc_size\":%llu,\"llc_ways\":%u,\"warmup_frac\":%g,"
                 "\"schedule\":{\"slice_instr\":%llu,\"until\":\"%s\",\"merged_accesses\":%zu,\"warmup_accesses\":%llu,\"slices\":%llu,\"switches\":%llu,\"cold_k\":%llu},\"apps\":[",
              cfg.hash().c_str(), (unsigned long long)cfg.llc.size, cfg.llc.ways, cfg.warmup_frac, (unsigned long long)slice, untilAll ? "all" : "first",
              sch.app.size(), (unsigned long long)warm, (unsigned long long)slicesN, (unsigned long long)sch.switches, (unsigned long long)coldK);
      for (size_t a = 0; a < apps.size(); a++)
        fprintf(f, "%s{\"name\":\"%s\",\"llc_accesses\":%zu,\"total_icount\":%llu}", a ? "," : "", esc(apps[a].name).c_str(), apps[a].t.size(), (unsigned long long)apps[a].total_icount);
      fprintf(f, "],\"runs\":[");
      for (size_t i = 0; i < res.size(); i++) {
        std::string js = resJson(res[i], apps, ids, keepSlices);
        js.pop_back();   // reopen the object to append the alone runs
        fprintf(f, "%s%s", i ? "," : "", js.c_str());
        if (alone) {
          fprintf(f, ",\"alone\":[");
          for (size_t a = 0; a < aloneRes[i].size(); a++) {
            const MultiResult& r = aloneRes[i][a];
            fprintf(f, "%s{\"app\":\"%s\",\"hits\":%llu,\"misses\":%llu,\"measured_icount\":%llu,\"mpki\":%.6f}", a ? "," : "", esc(apps[a].name).c_str(),
                    (unsigned long long)r.hits, (unsigned long long)r.misses, (unsigned long long)r.icount, r.icount ? 1000.0 * (double)r.misses / (double)r.icount : 0.0);
          }
          fprintf(f, "]");
        }
        fprintf(f, "}");
      }
      fprintf(f, "]}\n");
      fclose(f);
    }
    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "multisim: %s\n", e.what());
    return 1;
  }
}
