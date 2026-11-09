// lackey2trace: reads Valgrind lackey text on stdin, writes a binary raw trace (ROI only).
//   lackey2trace --roi-addr 0x404040 --out raw.trace|-  [--max-records N]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include "lackey_parser.h"

using namespace ac;

int main(int argc, char** argv) {
  LackeyOpts o;
  std::string out = "-";
  for (int i = 1; i < argc; i++) {
    std::string k = argv[i];
    if (k == "--roi-addr" && i + 1 < argc) { o.have_roi = true; o.roi_addr = strtoull(argv[++i], nullptr, 16); }
    else if (k == "--out" && i + 1 < argc) out = argv[++i];
    else if (k == "--max-records" && i + 1 < argc) o.max_records = strtoull(argv[++i], nullptr, 10);
    else { fprintf(stderr, "usage: lackey2trace [--roi-addr HEX] --out FILE|- [--max-records N]\n"); return 2; }
  }
  TraceWriter w;
  if (!w.open(out)) { fprintf(stderr, "lackey2trace: cannot open output %s\n", out.c_str()); return 1; }
  LackeyStats st = parseLackey(stdin, o, [&](const TraceRec& r) { w.write(r); });
  w.close(st.roi_instr);
  fprintf(stderr, "lackey2trace: records=%llu (loads=%llu stores=%llu) roi_instructions=%llu roi_begin=%d roi_end=%d%s\n",
          (unsigned long long)st.records, (unsigned long long)st.loads, (unsigned long long)st.stores,
          (unsigned long long)st.roi_instr, st.roi_begin_seen, st.roi_end_seen, st.truncated ? " TRUNCATED(max-records)" : "");
  if (o.have_roi && !st.roi_begin_seen) { fprintf(stderr, "lackey2trace: ERROR ROI begin marker never seen (wrong --roi-addr, or PIE binary?)\n"); return 3; }
  if (o.have_roi && !st.roi_end_seen && !st.truncated) fprintf(stderr, "lackey2trace: WARNING ROI end marker not seen (trace ends at program exit)\n");
  return 0;
}
