#include "common.h"
int main(int argc,char**argv){
  int V = argc>1?atoi(argv[1]):20000; int deg = argc>2?atoi(argv[2]):8; int kind = argc>3?atoi(argv[3]):0; seed_rng(argc>4?strtoull(argv[4],0,10):1);
  int nb = argc>5?atoi(argv[5]):3;
  int *off=malloc((V+1)*4); off[0]=0;
  for(int i=0;i<V;i++) off[i+1]=off[i]+1+rnd()%(2*deg-1);
  int E=off[V]; int *adj=malloc(E*4);
  for(int e=0;e<E;e++){ uint64_t a=rnd()%V; if(kind==1){ a=(a*(rnd()%V))/V; } adj[e]=(int)a; }
  int *dist=malloc(V*4),*q=malloc(V*4); long tot=0;
  ROI_BEGIN();
  for(int b=0;b<nb;b++){
    for(int i=0;i<V;i++) dist[i]=-1;
    int src=(int)(rnd()%V), h=0,t=0; q[t++]=src; dist[src]=0;
    while(h<t){ int u=q[h++]; for(int e=off[u];e<off[u+1];e++){ int v=adj[e]; if(dist[v]<0){dist[v]=dist[u]+1;q[t++]=v;} } }
    tot+=t;
  }
  ROI_END();
  printf("%ld\n",tot); return 0; }
