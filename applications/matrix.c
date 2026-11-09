#include "common.h"
int main(int argc,char**argv){
  int N = argc>1?atoi(argv[1]):96; int variant = argc>2?atoi(argv[2]):0; seed_rng(argc>3?strtoull(argv[3],0,10):1);
  double *A=malloc(N*N*8),*B=malloc(N*N*8),*C=calloc(N*N,8);
  for(int i=0;i<N*N;i++){A[i]=(rnd()&255)/64.0;B[i]=(rnd()&255)/64.0;}
  ROI_BEGIN();
  if(variant==0){
    for(int i=0;i<N;i++) for(int j=0;j<N;j++){ double s=0; for(int k=0;k<N;k++) s+=A[i*N+k]*B[k*N+j]; C[i*N+j]=s; }
  } else {
    int T=16;
    for(int ii=0;ii<N;ii+=T) for(int jj=0;jj<N;jj+=T) for(int kk=0;kk<N;kk+=T)
      for(int i=ii;i<ii+T&&i<N;i++) for(int j=jj;j<jj+T&&j<N;j++){ double s=C[i*N+j]; for(int k=kk;k<kk+T&&k<N;k++) s+=A[i*N+k]*B[k*N+j]; C[i*N+j]=s; }
  }
  ROI_END();
  printf("%f\n",C[N*N/2]); return 0; }
