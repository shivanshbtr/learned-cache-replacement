// Cache configuration: plain key=value file (see docs/cache_config.md).
#pragma once
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ac {

struct CacheGeom {
  uint64_t size = 0;
  uint32_t ways = 0;
};

struct Config {
  uint32_t line_bytes = 64;
  CacheGeom l1{32 * 1024, 8};
  CacheGeom l2{256 * 1024, 8};
  CacheGeom llc{2 * 1024 * 1024, 16};
  double warmup_frac = 0.1;  // fraction of LLC accesses used only to warm the cache
  uint64_t seed = 1;         // seed for the Random policy
  uint32_t srrip_bits = 2;

  uint32_t sets(const CacheGeom& g) const { return (uint32_t)(g.size / ((uint64_t)g.ways * line_bytes)); }
  uint32_t l1_sets() const { return sets(l1); }
  uint32_t l2_sets() const { return sets(l2); }
  uint32_t llc_sets() const { return sets(llc); }

  static bool pow2(uint64_t x) { return x && !(x & (x - 1)); }

  void validate() const {
    auto bad = [](const std::string& m) { throw std::runtime_error("config error: " + m); };
    if (!pow2(line_bytes) || line_bytes < 8 || line_bytes > 128) bad("line_bytes must be a power of two in [8,128]");
    const CacheGeom* gs[3] = {&l1, &l2, &llc};
    const char* names[3] = {"l1", "l2", "llc"};
    for (int i = 0; i < 3; i++) {
      const CacheGeom& g = *gs[i];
      std::string n = names[i];
      if (g.ways == 0 || g.size == 0) bad(n + ": size and ways must be > 0");
      if (g.size % ((uint64_t)g.ways * line_bytes) != 0) bad(n + ": size must be a multiple of ways*line_bytes");
      if (!pow2(sets(g))) bad(n + ": number of sets must be a power of two");
    }
    if (warmup_frac < 0.0 || warmup_frac >= 1.0) bad("warmup_frac must be in [0,1)");
    if (srrip_bits < 1 || srrip_bits > 4) bad("srrip_bits must be in [1,4]");
  }

  // Canonical text of every setting; the hash of it stamps result files.
  std::string canonical() const {
    std::ostringstream o;
    o << "line_bytes=" << line_bytes << ";l1=" << l1.size << "/" << l1.ways << ";l2=" << l2.size << "/" << l2.ways
      << ";llc=" << llc.size << "/" << llc.ways << ";warmup_frac=" << warmup_frac << ";seed=" << seed
      << ";srrip_bits=" << srrip_bits;
    return o.str();
  }
  std::string hash() const {
    uint64_t h = 1469598103934665603ULL;  // FNV-1a 64
    for (unsigned char c : canonical()) { h ^= c; h *= 1099511628211ULL; }
    char b[32];
    snprintf(b, sizeof b, "%016llx", (unsigned long long)h);
    return b;
  }

  static uint64_t parseNum(const std::string& v, const std::string& key) {
    if (v.empty()) throw std::runtime_error("config error: empty value for " + key);
    size_t i = 0;
    while (i < v.size() && isdigit((unsigned char)v[i])) i++;
    if (i == 0) throw std::runtime_error("config error: bad number for " + key + ": " + v);
    uint64_t n = std::stoull(v.substr(0, i));
    std::string suf = v.substr(i);
    if (suf == "") return n;
    if (suf == "K" || suf == "k") return n << 10;
    if (suf == "M" || suf == "m") return n << 20;
    if (suf == "G" || suf == "g") return n << 30;
    throw std::runtime_error("config error: bad suffix for " + key + ": " + v);
  }

  static Config load(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open config file: " + path);
    Config c;
    std::string line;
    auto trim = [](std::string s) {
      size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
      return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
    };
    while (std::getline(f, line)) {
      size_t h = line.find('#');
      if (h != std::string::npos) line = line.substr(0, h);
      line = trim(line);
      if (line.empty()) continue;
      size_t eq = line.find('=');
      if (eq == std::string::npos) throw std::runtime_error("config error: expected key=value: " + line);
      std::string k = trim(line.substr(0, eq)), v = trim(line.substr(eq + 1));
      if (k == "line_bytes") c.line_bytes = (uint32_t)parseNum(v, k);
      else if (k == "l1_size") c.l1.size = parseNum(v, k);
      else if (k == "l1_ways") c.l1.ways = (uint32_t)parseNum(v, k);
      else if (k == "l2_size") c.l2.size = parseNum(v, k);
      else if (k == "l2_ways") c.l2.ways = (uint32_t)parseNum(v, k);
      else if (k == "llc_size") c.llc.size = parseNum(v, k);
      else if (k == "llc_ways") c.llc.ways = (uint32_t)parseNum(v, k);
      else if (k == "warmup_frac") c.warmup_frac = std::stod(v);
      else if (k == "seed") c.seed = parseNum(v, k);
      else if (k == "srrip_bits") c.srrip_bits = (uint32_t)parseNum(v, k);
      else throw std::runtime_error("config error: unknown key: " + k);
    }
    c.validate();
    return c;
  }
};

}  // namespace ac
