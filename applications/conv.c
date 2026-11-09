// Multi-output-channel 2D convolution layer (one input channel, OC output channels, KxK kernels, "valid" borders).
// usage: conv W K OC seed      (defaults 256 3 4 1)
// Loop order is output-channel outermost: the whole input image is swept once per output channel, and each sweep also
// writes a fresh W*W output plane. Reuse distance of the image between sweeps = image + one output plane, so it exceeds
// the LLC when 2*W*W*4 bytes > LLC size (W >= 600 for a 2 MB LLC). The seed only changes data values, not addresses.
#include "common.h"
int main(int argc,char**argv){
  int W = argc>1?atoi(argv[1]):256; int K = argc>2?atoi(argv[2]):3; int OC = argc>3?atoi(argv[3]):4;
  seed_rng(argc>4?strtoull(argv[4],0,10):1);
  size_t plane=(size_t)W*W;
  float *img=malloc(plane*4),*out=calloc(plane*OC,4),*ker=malloc((size_t)OC*K*K*4);
  for(size_t i=0;i<plane;i++) img[i]=(rnd()&255)/255.0f;
  for(int i=0;i<OC*K*K;i++) ker[i]=((rnd()&255)/255.0f)/(K*K);
  int r=K/2;
  ROI_BEGIN();
  for(int oc=0;oc<OC;oc++){
    float *kk=ker+(size_t)oc*K*K, *o=out+(size_t)oc*plane;
    for(int y=r;y<W-r;y++) for(int x=r;x<W-r;x++){
      float s=0;
      for(int ky=-r;ky<=r;ky++) for(int kx=-r;kx<=r;kx++) s+=img[(size_t)(y+ky)*W+(x+kx)]*kk[(ky+r)*K+(kx+r)];
      o[(size_t)y*W+x]=s;
    }
  }
  ROI_END();
  printf("%f\n",out[plane*(OC-1)+(size_t)(W/2)*W+W/2]); return 0; }
