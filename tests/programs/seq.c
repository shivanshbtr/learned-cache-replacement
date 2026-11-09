// Test workload: reads one int per 64 B line, `passes` times over `bytes` of memory, inside the ROI.
// Expected LLC-visible accesses: (bytes/64)*passes when bytes is larger than L2.
#include "common.h"
int main(int argc, char** argv) {
  size_t bytes = argc > 1 ? strtoull(argv[1], 0, 10) : (1u << 20);
  int passes = argc > 2 ? atoi(argv[2]) : 2;
  size_t lines = bytes / 64;
  volatile int* a = (volatile int*)aligned_alloc(64, lines * 64);
  for (size_t i = 0; i < lines * 16; i++) a[i] = (int)i;   // touch memory outside the ROI
  long sum = 0;
  ROI_BEGIN();
  for (int p = 0; p < passes; p++)
    for (size_t i = 0; i < lines; i++) sum += a[i * 16];
  ROI_END();
  printf("%ld\n", sum);
  return 0;
}
