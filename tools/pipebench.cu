// pipebench.cu — integer pipe throughput per SM-cycle on the card at hand (development tool).
// Each block measures its own clock64 span; ops per SM-cycle = threads*ops / cycles (median over blocks).
// nvcc -O3 -arch=sm_120 -o pipebench pipebench.cu
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s line %d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)

#define ITERS 4096
// T selects the op mix inside the loop; every variant executes 8 "ops" per iteration (OPS below says how many instructions that is)
template<int T> __global__ void __launch_bounds__(256) kern(uint32_t *out, long long *cyc, uint32_t seed) {
    uint32_t a0=seed+threadIdx.x, a1=a0*3, a2=a0*5, a3=a0*7, a4=a0*11, a5=a0*13, a6=a0*17, a7=a0*19;
    uint32_t b=seed|1, c=seed*2654435761u|1;
    uint64_t w0=a0,w1=a1,w2=a2,w3=a3,w4=a4,w5=a5,w6=a6,w7=a7;
    float f0=a0,f1=a1,f2=a2,f3=a3,f4=a4,f5=a5,f6=a6,f7=a7, fb=1.0001f, fc=0.999f;
    __syncthreads();
    long long t0=clock64();
    #pragma unroll 1
    for(int i=0;i<ITERS;i++){
        if(T==0){ asm volatile("add.u32 %0,%0,%8;\n\tadd.u32 %1,%1,%8;\n\tadd.u32 %2,%2,%8;\n\tadd.u32 %3,%3,%8;\n\tadd.u32 %4,%4,%8;\n\tadd.u32 %5,%5,%8;\n\tadd.u32 %6,%6,%8;\n\tadd.u32 %7,%7,%8;"
            :"+r"(a0),"+r"(a1),"+r"(a2),"+r"(a3),"+r"(a4),"+r"(a5),"+r"(a6),"+r"(a7):"r"(b)); }
        if(T==1){ asm volatile("xor.b32 %0,%0,%8;\n\txor.b32 %1,%1,%8;\n\txor.b32 %2,%2,%8;\n\txor.b32 %3,%3,%8;\n\txor.b32 %4,%4,%8;\n\txor.b32 %5,%5,%8;\n\txor.b32 %6,%6,%8;\n\txor.b32 %7,%7,%8;"
            :"+r"(a0),"+r"(a1),"+r"(a2),"+r"(a3),"+r"(a4),"+r"(a5),"+r"(a6),"+r"(a7):"r"(b)); }
        if(T==2){ asm volatile("mad.lo.u32 %0,%0,%8,%9;\n\tmad.lo.u32 %1,%1,%8,%9;\n\tmad.lo.u32 %2,%2,%8,%9;\n\tmad.lo.u32 %3,%3,%8,%9;\n\tmad.lo.u32 %4,%4,%8,%9;\n\tmad.lo.u32 %5,%5,%8,%9;\n\tmad.lo.u32 %6,%6,%8,%9;\n\tmad.lo.u32 %7,%7,%8,%9;"
            :"+r"(a0),"+r"(a1),"+r"(a2),"+r"(a3),"+r"(a4),"+r"(a5),"+r"(a6),"+r"(a7):"r"(b),"r"(c)); }
        if(T==3){ asm volatile("mad.hi.u32 %0,%0,%8,%9;\n\tmad.hi.u32 %1,%1,%8,%9;\n\tmad.hi.u32 %2,%2,%8,%9;\n\tmad.hi.u32 %3,%3,%8,%9;\n\tmad.hi.u32 %4,%4,%8,%9;\n\tmad.hi.u32 %5,%5,%8,%9;\n\tmad.hi.u32 %6,%6,%8,%9;\n\tmad.hi.u32 %7,%7,%8,%9;"
            :"+r"(a0),"+r"(a1),"+r"(a2),"+r"(a3),"+r"(a4),"+r"(a5),"+r"(a6),"+r"(a7):"r"(b),"r"(c)); }
        if(T==4){ asm volatile("mad.wide.u32 %0,%8,%9,%0;\n\tmad.wide.u32 %1,%8,%9,%1;\n\tmad.wide.u32 %2,%8,%9,%2;\n\tmad.wide.u32 %3,%8,%9,%3;\n\tmad.wide.u32 %4,%8,%9,%4;\n\tmad.wide.u32 %5,%8,%9,%5;\n\tmad.wide.u32 %6,%8,%9,%6;\n\tmad.wide.u32 %7,%8,%9,%7;"
            :"+l"(w0),"+l"(w1),"+l"(w2),"+l"(w3),"+l"(w4),"+l"(w5),"+l"(w6),"+l"(w7):"r"(b),"r"(c)); b=(uint32_t)w7|1; }
        if(T==5){ // dependent carry chain: 8 madc through CC (the current fe_mul pattern)
            asm volatile("mad.lo.cc.u32 %0,%0,%8,%9;\n\tmadc.lo.cc.u32 %1,%1,%8,%9;\n\tmadc.lo.cc.u32 %2,%2,%8,%9;\n\tmadc.lo.cc.u32 %3,%3,%8,%9;\n\tmadc.lo.cc.u32 %4,%4,%8,%9;\n\tmadc.lo.cc.u32 %5,%5,%8,%9;\n\tmadc.lo.cc.u32 %6,%6,%8,%9;\n\tmadc.lo.u32 %7,%7,%8,%9;"
            :"+r"(a0),"+r"(a1),"+r"(a2),"+r"(a3),"+r"(a4),"+r"(a5),"+r"(a6),"+r"(a7):"r"(b),"r"(c)); }
        if(T==6){ // 1:1 mix IMAD + IADD on independent registers
            asm volatile("mad.lo.u32 %0,%0,%8,%9;\n\tadd.u32 %1,%1,%8;\n\tmad.lo.u32 %2,%2,%8,%9;\n\tadd.u32 %3,%3,%8;\n\tmad.lo.u32 %4,%4,%8,%9;\n\tadd.u32 %5,%5,%8;\n\tmad.lo.u32 %6,%6,%8,%9;\n\tadd.u32 %7,%7,%8;"
            :"+r"(a0),"+r"(a1),"+r"(a2),"+r"(a3),"+r"(a4),"+r"(a5),"+r"(a6),"+r"(a7):"r"(b),"r"(c)); }
        if(T==7){ // 1:1 mix mad.wide + add
            asm volatile("mad.wide.u32 %0,%8,%9,%0;\n\tadd.u32 %4,%4,%8;\n\tmad.wide.u32 %1,%8,%9,%1;\n\tadd.u32 %5,%5,%8;\n\tmad.wide.u32 %2,%8,%9,%2;\n\tadd.u32 %6,%6,%8;\n\tmad.wide.u32 %3,%8,%9,%3;\n\tadd.u32 %7,%7,%8;"
            :"+l"(w0),"+l"(w1),"+l"(w2),"+l"(w3),"+r"(a4),"+r"(a5),"+r"(a6),"+r"(a7):"r"(b),"r"(c)); b=(uint32_t)w3|1; }
        if(T==8){ // dependent IMAD chain (latency bound within a thread): all 8 on one register
            asm volatile("mad.lo.u32 %0,%0,%8,%9;\n\tmad.lo.u32 %0,%0,%8,%9;\n\tmad.lo.u32 %0,%0,%8,%9;\n\tmad.lo.u32 %0,%0,%8,%9;\n\tmad.lo.u32 %0,%0,%8,%9;\n\tmad.lo.u32 %0,%0,%8,%9;\n\tmad.lo.u32 %0,%0,%8,%9;\n\tmad.lo.u32 %0,%0,%8,%9;"
            :"+r"(a0),"+r"(a1),"+r"(a2),"+r"(a3),"+r"(a4),"+r"(a5),"+r"(a6),"+r"(a7):"r"(b),"r"(c)); }
        if(T==9){ asm volatile("fma.rn.f32 %0,%0,%8,%9;\n\tfma.rn.f32 %1,%1,%8,%9;\n\tfma.rn.f32 %2,%2,%8,%9;\n\tfma.rn.f32 %3,%3,%8,%9;\n\tfma.rn.f32 %4,%4,%8,%9;\n\tfma.rn.f32 %5,%5,%8,%9;\n\tfma.rn.f32 %6,%6,%8,%9;\n\tfma.rn.f32 %7,%7,%8,%9;"
            :"+f"(f0),"+f"(f1),"+f"(f2),"+f"(f3),"+f"(f4),"+f"(f5),"+f"(f6),"+f"(f7):"f"(fb),"f"(fc)); }
        if(T==10){ // comba step: mad.lo.cc + madc.hi.cc + addc  (3 PTX ops -> hopefully IMAD.WIDE.X + IADD.X), 2 steps per 'iteration' of 8 ops counted as 6 PTX
            asm volatile("mad.lo.cc.u32 %0,%8,%9,%0;\n\tmadc.hi.cc.u32 %1,%8,%9,%1;\n\taddc.u32 %2,%2,0;\n\tmad.lo.cc.u32 %3,%8,%9,%3;\n\tmadc.hi.cc.u32 %4,%8,%9,%4;\n\taddc.u32 %5,%5,0;\n\tadd.u32 %6,%6,%8;\n\tadd.u32 %7,%7,%9;"
            :"+r"(a0),"+r"(a1),"+r"(a2),"+r"(a3),"+r"(a4),"+r"(a5),"+r"(a6),"+r"(a7):"r"(b),"r"(c)); }
        if(T==11){ // dependent mad.wide chain (latency of IMAD.WIDE): multiplier = low word of the accumulator
            asm volatile("cvt.u32.u64 %1,%0;\n\tmad.wide.u32 %0,%1,%2,%0;\n\tcvt.u32.u64 %1,%0;\n\tmad.wide.u32 %0,%1,%2,%0;\n\tcvt.u32.u64 %1,%0;\n\tmad.wide.u32 %0,%1,%2,%0;\n\tcvt.u32.u64 %1,%0;\n\tmad.wide.u32 %0,%1,%2,%0;\n\tcvt.u32.u64 %1,%0;\n\tmad.wide.u32 %0,%1,%2,%0;\n\tcvt.u32.u64 %1,%0;\n\tmad.wide.u32 %0,%1,%2,%0;\n\tcvt.u32.u64 %1,%0;\n\tmad.wide.u32 %0,%1,%2,%0;\n\tcvt.u32.u64 %1,%0;\n\tmad.wide.u32 %0,%1,%2,%0;"
            :"+l"(w0),"+r"(b):"r"(c)); }
        if(T==12){ // shf (funnel shift) rate
            asm volatile("shf.r.clamp.b32 %0,%0,%8,26;\n\tshf.r.clamp.b32 %1,%1,%8,26;\n\tshf.r.clamp.b32 %2,%2,%8,26;\n\tshf.r.clamp.b32 %3,%3,%8,26;\n\tshf.r.clamp.b32 %4,%4,%8,26;\n\tshf.r.clamp.b32 %5,%5,%8,26;\n\tshf.r.clamp.b32 %6,%6,%8,26;\n\tshf.r.clamp.b32 %7,%7,%8,26;"
            :"+r"(a0),"+r"(a1),"+r"(a2),"+r"(a3),"+r"(a4),"+r"(a5),"+r"(a6),"+r"(a7):"r"(b)); }
        if(T==13){ // 2:1 mix mad.wide : add  (the no-carry 10x26 profile)
            asm volatile("mad.wide.u32 %0,%8,%9,%0;\n\tmad.wide.u32 %1,%8,%9,%1;\n\tadd.u32 %4,%4,%8;\n\tmad.wide.u32 %2,%8,%9,%2;\n\tmad.wide.u32 %3,%8,%9,%3;\n\tadd.u32 %5,%5,%8;\n\tadd.u32 %6,%6,%8;\n\tadd.u32 %7,%7,%9;"
            :"+l"(w0),"+l"(w1),"+l"(w2),"+l"(w3),"+r"(a4),"+r"(a5),"+r"(a6),"+r"(a7):"r"(b),"r"(c)); b=(uint32_t)w3|1; }
    }
    long long t1=clock64();
    if(threadIdx.x==0) cyc[blockIdx.x]=t1-t0;
    uint32_t s=a0^a1^a2^a3^a4^a5^a6^a7^(uint32_t)(w0^w1^w2^w3^w4^w5^w6^w7)^__float_as_uint(f0+f1+f2+f3+f4+f5+f6+f7);
    if(s==0x12345678u) out[0]=s;
}
static const char *NAMES[]={"IADD (add.u32)","LOP3 (xor)","IMAD (mad.lo)","IMAD.HI (mad.hi)","IMAD.WIDE.U32 (mad.wide)","madc.lo.cc chain (CC-dependent)","mix IMAD:IADD 1:1","mix IMAD.WIDE:IADD 1:1","dependent IMAD chain (1 acc)","FFMA","comba step lo.cc/hi.cc/addc (+2 add)","dependent IMAD.WIDE chain (1 acc)","SHF funnel","mix IMAD.WIDE:IADD 2:1"};
template<int T> void run(int sms,int bps){
    uint32_t *out; long long *cyc; int blocks=sms*bps;
    CK(cudaMalloc(&out,4)); CK(cudaMalloc(&cyc,blocks*8));
    kern<T><<<blocks,256>>>(out,cyc,1); CK(cudaDeviceSynchronize());
    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0); kern<T><<<blocks,256>>>(out,cyc,2); cudaEventRecord(e1); CK(cudaDeviceSynchronize());
    float ms; cudaEventElapsedTime(&ms,e0,e1);
    std::vector<long long> c(blocks); CK(cudaMemcpy(c.data(),cyc,blocks*8,cudaMemcpyDeviceToHost));
    std::sort(c.begin(),c.end()); long long med=c[blocks/2];
    // per block: 256 threads * ITERS * 8 ops; bps blocks share one SM concurrently (if resident together) -> per-SM-cycle = bps*256*ITERS*8/med
    double ops=(double)blocks*256.0*ITERS*8.0;
    printf("%-40s blocks/SM %d: %7.2f lane-ops/SM-cycle (median block span), %8.1f Gop/s wall\n",NAMES[T],bps,bps*256.0*ITERS*8.0/med,ops/ms/1e6);
    cudaFree(out); cudaFree(cyc);
}
int main(int argc,char**argv){
    int dev=0; cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,dev));
    printf("%s sm_%d%d %d SMs\n",p.name,p.major,p.minor,p.multiProcessorCount);
    int sms=p.multiProcessorCount;
    for(int bps: {2,8}){
        run<0>(sms,bps); run<1>(sms,bps); run<2>(sms,bps); run<3>(sms,bps); run<4>(sms,bps); run<5>(sms,bps);
        run<6>(sms,bps); run<7>(sms,bps); run<13>(sms,bps); run<8>(sms,bps); run<11>(sms,bps); run<9>(sms,bps); run<10>(sms,bps); run<12>(sms,bps);
    }
    return 0;
}
