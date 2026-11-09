#include "common.h"
typedef struct { uint64_t key, val; } Slot;
int main(int argc,char**argv){
  int lg = argc>1?atoi(argv[1]):15; int nkeys = argc>2?atoi(argv[2]):20000; int nops = argc>3?atoi(argv[3]):60000; int skew = argc>4?atoi(argv[4]):1; seed_rng(argc>5?strtoull(argv[5],0,10):1);
  size_t S=(size_t)1<<lg, mask=S-1; Slot *tab=calloc(S,sizeof(Slot)); uint64_t *keys=malloc(nkeys*8);
  for(int i=0;i<nkeys;i++){ uint64_t k=(rnd()|1); keys[i]=k; size_t h=(k*0x9E3779B97F4A7C15ULL)>>(64-lg); while(tab[h].key) h=(h+1)&mask; tab[h].key=k; tab[h].val=i; }
  uint64_t acc=0;
  ROI_BEGIN();
  for(int o=0;o<nops;o++){
    double u=(rnd()>>11)/9007199254740992.0; int idx = skew ? (int)(nkeys*u*u*u) : (int)(nkeys*u);
    uint64_t k=keys[idx]; size_t h=(k*0x9E3779B97F4A7C15ULL)>>(64-lg);
    while(tab[h].key && tab[h].key!=k) h=(h+1)&mask; acc+=tab[h].val;
  }
  ROI_END();
  printf("%llu\n",(unsigned long long)acc); return 0; }
