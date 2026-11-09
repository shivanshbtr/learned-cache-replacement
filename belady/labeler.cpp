// labeler: Belady labels for training. See belady/labeler.h for the record layout and docs/dataset_format.md.
//   labeler --config C --trace LLC.trace --driver lru|srrip|belady|... --out X.dec [--stride N] [--stats S.json]
#include <cstdio>
#include <map>
#include "../simulator/factory.h"
#include "labeler.h"

using namespace ac;

int main(int argc, char** argv) {
  try {
    std::map<std::string, std::string> a;
    for (int i = 1; i < argc; i += 2) {
      std::string k = argv[i];
      if (k.rfind("--", 0) != 0 || i + 1 >= argc) throw std::runtime_error("bad argument: " + k);
      a[k.substr(2)] = argv[i + 1];
    }
    if (!a.count("config") || !a.count("trace") || !a.count("out")) throw std::runtime_error("usage: labeler --config C --trace T --out X.dec [--driver lru] [--stride 1] [--stats S.json]");
    Config cfg = Config::load(a.at("config"));
    std::string drv = a.count("driver") ? a.at("driver") : "lru";
    uint32_t stride = a.count("stride") ? (uint32_t)std::stoul(a.at("stride")) : 1;
    auto pol = makePolicy(drv, cfg);
    if (!pol) throw std::runtime_error("unknown driver policy: " + drv);
    TraceReader in;
    if (!in.open(a.at("trace"))) throw std::runtime_error("cannot open LLC trace: " + a.at("trace"));
    std::vector<TraceRec> t;
    { TraceRec r; while (in.next(r)) t.push_back(r); }
    if (t.empty()) throw std::runtime_error("LLC trace is empty");
    auto nu = computeNextUse(t, log2u(cfg.line_bytes));
    const uint64_t warm = warmupCount(cfg, t.size());
    FILE* f = fopen(a.at("out").c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + a.at("out"));
    setvbuf(f, nullptr, _IOFBF, 1 << 20);
    DecHeader h{};
    memcpy(h.magic, "ACDEC", 6);
    h.version = 1; h.ways = cfg.llc.ways; h.rec_words = 4 + 4 * cfg.llc.ways; h.stride = stride; h.sets = cfg.llc_sets();
    h.line_bytes = cfg.line_bytes; h.trace_len = t.size(); h.warmup = warm;
    strncpy(h.driver, drv.c_str(), 7);
    fwrite(&h, sizeof h, 1, f);
    LabelStats st = labelTrace(*pol, cfg, t, nu, warm, stride, [&](const uint32_t* r, size_t w) { fwrite(r, 4, w, f); });
    h.num_records = st.recorded;
    fseek(f, 0, SEEK_SET);
    fwrite(&h, sizeof h, 1, f);
    if (fclose(f) != 0) throw std::runtime_error("failed to finish " + a.at("out"));
    printf("labeler: driver=%s accesses=%llu misses=%llu decisions=%llu recorded=%llu (stride %u) tied=%.1f%% driver_agrees_with_belady=%.1f%%\n",
           drv.c_str(), (unsigned long long)st.accesses, (unsigned long long)st.misses, (unsigned long long)st.decisions,
           (unsigned long long)st.recorded, stride, st.decisions ? 100.0 * st.tied / st.decisions : 0.0,
           st.decisions ? 100.0 * st.driver_agrees / st.decisions : 0.0);
    if (a.count("stats")) {
      FILE* s = fopen(a.at("stats").c_str(), "w");
      if (!s) throw std::runtime_error("cannot write stats");
      fprintf(s, "{\"driver\":\"%s\",\"config_hash\":\"%s\",\"trace\":\"%s\",\"accesses\":%llu,\"warmup\":%llu,\"misses\":%llu,\"measured_misses\":%llu,"
                 "\"measured_hits\":%llu,\"cold_fills\":%llu,\"decisions\":%llu,\"recorded\":%llu,\"stride\":%u,\"tied\":%llu,\"driver_agrees\":%llu}\n",
              drv.c_str(), cfg.hash().c_str(), a.at("trace").c_str(), (unsigned long long)st.accesses, (unsigned long long)warm,
              (unsigned long long)st.misses, (unsigned long long)st.measured_misses, (unsigned long long)st.measured_hits,
              (unsigned long long)st.cold_fills, (unsigned long long)st.decisions, (unsigned long long)st.recorded, stride,
              (unsigned long long)st.tied, (unsigned long long)st.driver_agrees);
      fclose(s);
    }
    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "labeler: %s\n", e.what());
    return 1;
  }
}
