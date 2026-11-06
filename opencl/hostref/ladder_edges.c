// ProxiChorizo License 1.0; see LICENSE.
// Run the actual ladder and batched preparation as C, with fe29 limb assertions.
#include "shim.h"
#include "proxichorizo.cl"
#include "../merged_scalar.h"
static void decode_hex(const char *s,u8 out[32]) {for(int i=0;i<32;i++){unsigned v;if(sscanf(s+2*i,"%2x",&v)!=1)abort();out[i]=(u8)v;}}
static int check(const u8 scalar[32],HPoint *points,u8 expected[][32],int *valid,uint n) {
 Consts K;memset(&K,0,sizeof K);u8 pk[32];int ok=0;init_kernel(scalar,pk,&ok,&K);if(ok!=1)return 1;
#if PX_LAD_COFACTOR
 v_xonly_scalar(scalar,K.ladder_scalar);
#else
 v_merged_scalar(scalar,K.ladder_scalar);
#endif
 for(int b=255;b>=0;b--)if((K.ladder_scalar[b>>3]>>(b&7))&1){K.ladder_top=b;break;}
 ProjectivePoint output[1024];prep_kernel(&n,points,&K);ladder_kernel(points,&n,output,&K);
 for(uint i=0;i<n;i++) {
  if(output[i].valid!=(uint)valid[i])return 2;
  if(valid[i]) {fe zi,y;u8 bytes[32];fe_invert(&zi,&output[i].Z);fe_mul(&y,&output[i].Y,&zi);fe_tobytes(bytes,&y);if(memcmp(bytes,expected[i],32))return 3;}
 }
 return 0;
}
int main(int argc,char **argv) {
 if(argc!=2)return 2;FILE*f=fopen(argv[1],"r");if(!f)return 2;
 char sc[65],x[65],y[65],ex[65],ey[65],last[65]={0};u8 scalar[32],expected[1024][32];int valid[1024],v,total=0;uint n=0;HPoint points[1024];
 while(fscanf(f,"%64s %64s %64s %64s %64s %d",sc,x,y,ex,ey,&v)==6) {
  if(n&&strcmp(sc,last)) {int rc=check(scalar,points,expected,valid,n);if(rc){printf("FAIL %d\n",rc);return 1;}total+=n;n=0;}
  if(n>=1024)return 2;strcpy(last,sc);decode_hex(sc,scalar);u8 xb[32],yb[32];decode_hex(x,xb);decode_hex(y,yb);
  fe_frombytes(&points[n].X,xb);fe_frombytes(&points[n].Y,yb);points[n].nonce=n;decode_hex(ey,expected[n]);valid[n++]=v;
 }
 if(n){int rc=check(scalar,points,expected,valid,n);if(rc){printf("FAIL %d\n",rc);return 1;}total+=n;}
 fclose(f);printf("PASS host ladder: %d independent point outputs with limb assertions\n",total);return total<560 || total%56!=0;
}
