// Parser for `valgrind --tool=lackey --trace-mem=yes` text output.
//   "I  04016b30,3"   instruction fetch (address,size)
//   " L 04016b30,4"   load    " S ..." store    " M ..." modify (load+store)
// Lines starting with "==" are Valgrind messages and are skipped.
#pragma once
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include "../simulator/trace.h"

namespace ac {

struct LackeyOpts {
  bool have_roi = false;
  uint64_t roi_addr = 0;     // address of the global `roi_marker`
  uint64_t max_records = 0;  // 0 = unlimited
};

struct LackeyStats {
  uint64_t records = 0, loads = 0, stores = 0, roi_instr = 0, lines = 0;
  bool roi_begin_seen = false, roi_end_seen = false, truncated = false;
};

// Emits records via sink(const TraceRec&). Without an ROI address the whole run is traced.
// With one: the first store to it starts the ROI, the second store ends it (parsing stops there).
template <class Sink>
LackeyStats parseLackey(FILE* in, const LackeyOpts& o, Sink sink) {
  LackeyStats st;
  bool inside = !o.have_roi;
  if (!o.have_roi) st.roi_begin_seen = true;
  uint64_t cur_pc = 0, pending = 0;
  std::vector<char> buf(1 << 20);
  std::string carry;

  auto hexval = [](const char*& p, const char* e, uint64_t& v) {
    v = 0; bool any = false;
    while (p < e) {
      char c = *p; int d;
      if (c >= '0' && c <= '9') d = c - '0';
      else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
      else if (c >= 'A' && c <= 'F') d = c - 'A' + 10;
      else break;
      v = (v << 4) | (uint64_t)d; any = true; p++;
    }
    return any;
  };
  auto decval = [](const char*& p, const char* e, uint64_t& v) {
    v = 0; bool any = false;
    while (p < e && *p >= '0' && *p <= '9') { v = v * 10 + (uint64_t)(*p - '0'); any = true; p++; }
    return any;
  };

  // returns false when parsing should stop
  auto handle = [&](const char* s, size_t n) -> bool {
    if (n < 3) return true;
    st.lines++;
    char kind;
    const char* p = s;
    const char* e = s + n;
    if (s[0] == 'I') { kind = 'I'; p++; }
    else if (s[0] == ' ' && (s[1] == 'L' || s[1] == 'S' || s[1] == 'M')) { kind = s[1]; p += 2; }
    else return true;  // "==pid==" messages, blank or unknown lines
    while (p < e && *p == ' ') p++;
    uint64_t addr, size;
    if (!hexval(p, e, addr) || p >= e || *p != ',') return true;
    p++;
    if (!decval(p, e, size)) return true;
    if (kind == 'I') { cur_pc = addr; if (inside) pending++; return true; }
    bool wr = (kind == 'S' || kind == 'M');
    if (o.have_roi && wr && addr == o.roi_addr) {
      if (!inside && !st.roi_begin_seen) { inside = true; st.roi_begin_seen = true; pending = 0; return true; }
      if (inside) { st.roi_end_seen = true; return false; }
      return true;
    }
    if (!inside) return true;
    TraceRec r{};
    r.addr = addr; r.pc = cur_pc;
    r.icount = (uint32_t)(pending > 0xFFFFFFFFull ? 0xFFFFFFFFull : pending);
    st.roi_instr += pending;
    pending = 0;
    r.size = (uint8_t)(size > 255 ? 255 : size);
    r.flags = wr ? FLAG_WRITE : 0;
    sink(r);
    st.records++;
    if (wr) st.stores++; else st.loads++;
    if (o.max_records && st.records >= o.max_records) { st.truncated = true; return false; }
    return true;
  };

  for (;;) {
    size_t got = fread(buf.data(), 1, buf.size(), in);
    if (got == 0) break;
    size_t start = 0;
    for (size_t i = 0; i < got; i++) {
      if (buf[i] != '\n') continue;
      bool cont;
      if (!carry.empty()) { carry.append(buf.data() + start, i - start); cont = handle(carry.data(), carry.size()); carry.clear(); }
      else cont = handle(buf.data() + start, i - start);
      if (!cont) return st;
      start = i + 1;
    }
    if (start < got) carry.append(buf.data() + start, got - start);
  }
  if (!carry.empty()) handle(carry.data(), carry.size());
  return st;
}

}  // namespace ac
