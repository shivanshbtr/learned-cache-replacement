#pragma once
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
volatile int roi_marker;
static uint64_t rng_state = 88172645463325252ULL;
static inline uint64_t rnd(void){ rng_state ^= rng_state<<13; rng_state ^= rng_state>>7; rng_state ^= rng_state<<17; return rng_state; }
#define COMPILER_BARRIER() __asm__ __volatile__("" ::: "memory")
#define ROI_BEGIN() do { COMPILER_BARRIER(); roi_marker = 1; COMPILER_BARRIER(); } while(0)
#define ROI_END()   do { COMPILER_BARRIER(); roi_marker = 2; COMPILER_BARRIER(); } while(0)
static inline void seed_rng(uint64_t s){ rng_state ^= s * 0x9E3779B97F4A7C15ULL; for(int i=0;i<8;i++) rnd(); }
