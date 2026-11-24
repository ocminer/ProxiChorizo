// ProxiChorizo - Copyright (c) 2026 suprnova.cc. ProxiChorizo License 1.0; see LICENSE.
// Compile this translation unit as compute_120 PTX only. Its driver-compiled
// kernels coexist with the unchanged native kernels in proxichorizo.cu.
#include <cuda_runtime.h>
#define TXID_BW_DEVICE 1
#include "txid_blake2b.h"
#include "txid_bw_tail.cuh"
#include <atomic>
#include <mutex>
#include <cstdio>
#include <cstdlib>

namespace txid_bw {
using namespace txid;
template <int W, int HIGH, int LAST>
__global__ void __launch_bounds__(512) search(const ParamsCore p, uint32_t low, uint64_t fixed,
        uint32_t full, uint32_t tail, HitRec *hits, unsigned *nhit, uint32_t nshift, const Plan01 plan) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
    uint32_t n = low + gid, limit = full + (gid < tail);
    for (uint32_t i = 0; i < limit; ++i, n += stride) {
        uint64_t m[16], v[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) m[j] = j <= LAST ? p.m[j] : 0;
        m[W] |= (uint64_t)n << (32 * HIGH);
        rounds01<W, false>(v, p, plan, m); rounds<2>(v, m);
        uint64_t h3 = p.h[3] ^ v[3] ^ v[11];
        if ((h3 & p.mask) == 0) {
            uint64_t nonce = fixed | ((uint64_t)__byte_perm(n, 0, 0x0123) << nshift);
            unsigned slot = atomicAdd(nhit, 1u);
            if (slot < TXID_MAX_HITS) hits[slot] = HitRec{nonce, h3};
        }
    }
#endif
}
template <int KEY> void dispatch(int key, unsigned grid, unsigned block, cudaStream_t stream,
        const ParamsCore &p, uint32_t low, uint64_t fixed, uint32_t full, uint32_t tail,
        HitRec *hits, unsigned *nhit, uint32_t nshift, const Plan01 &plan) {
    constexpr int W = KEY / 4, HIGH = (KEY / 2) & 1, LAST = W + (KEY & 1);
    if constexpr (LAST < 16) if (key == KEY) {
        search<W, HIGH, LAST><<<grid, block, 0, stream>>>(p, low, fixed, full, tail, hits, nhit, nshift, plan);
        return;
    }
    if constexpr (KEY < 63) dispatch<KEY + 1>(key, grid, block, stream, p, low, fixed, full, tail, hits, nhit, nshift, plan);
}
} // namespace txid_bw

namespace txid_bw {
// Preserve the existing module for ordinary layouts and for cards on which
// the zero-tail specialization is slower. Each device makes its own choice.
static void tail_launch(int mode,unsigned grid,cudaStream_t stream,const ParamsCore &p,
        uint32_t low,uint64_t fixed,uint32_t full,uint32_t tail,HitRec *hits,unsigned *nhit,const Plan01 &plan) {
    if (mode == 3) {
        constexpr unsigned block = 768;
        const uint64_t old_stride = (uint64_t)grid * 512;
        const uint64_t count = old_stride * full + tail;
        // Preserve the full-SM grid. Only the number of nonces per thread changes.
        const uint64_t stride = (uint64_t)grid * block;
        zero_tail_search<2,block><<<grid,block,0,stream>>>(p, low, fixed,
            (uint32_t)(count / stride), (uint32_t)(count % stride), hits, nhit);
    }
    else if(mode==1)zero_tail_search<2><<<grid,512,0,stream>>>(p,low,fixed,full,tail,hits,nhit);
    else if(mode==2)zero_tail_search<3><<<grid,512,0,stream>>>(p,low,fixed,full,tail,hits,nhit);
    else search<15,1,15><<<grid,512,0,stream>>>(p,low,fixed,full,tail,hits,nhit,0,plan);
}
static bool tail_shape(int key,const ParamsCore &p,uint32_t nshift) {
    if(key!=62||nshift||p.kernel!=30||p.shift||p.tfin<128||p.tfin%128)return false;
    uint64_t nonzero=p.m[15]>>32;
    for(int i=0;i<15;++i)nonzero|=p.m[i];
    return nonzero==0;
}
static void checked(cudaError_t error) {
    if(error!=cudaSuccess){fprintf(stderr,"Blackwell zero-tail check: %s\n",cudaGetErrorString(error));std::exit(2);}
}
struct TailChoices {std::atomic<int> mode[128];std::mutex mutex[128];TailChoices(){for(auto &m:mode)m.store(-1);}};
static int tail_choice(unsigned grid,const ParamsCore &input,uint64_t fixed,const Plan01 &plan) {
    const char *force=getenv("PX_TXID_BW_ZERO");
    if(force&&!strcmp(force,"0"))return 0;
    int dev=-1;checked(cudaGetDevice(&dev));if(dev<0||dev>=128)return 0;
    static TailChoices choices;
    int chosen=choices.mode[dev].load();if(chosen>=0)return chosen;
    std::lock_guard<std::mutex> lock(choices.mutex[dev]);
    chosen=choices.mode[dev].load();if(chosen>=0)return chosen;
    HitRec *hits=nullptr;unsigned *nhit=nullptr;
    checked(cudaMalloc(&hits,TXID_MAX_HITS*sizeof(HitRec)));checked(cudaMalloc(&nhit,sizeof(unsigned)));
    // Check both candidates against the CPU's complete twelve-round evaluation
    // of this already-prepared block. These attempts never count as mining.
    ParamsCore p=input;p.mask=0;
    for(int mode:{1,2,3})for(uint32_t low:{0u,0x89abc123u,0xfffffbffu}) {
        checked(cudaMemset(nhit,0,4));tail_launch(mode,3,0,p,low,fixed,0,1025,hits,nhit,plan);
        checked(cudaGetLastError());checked(cudaDeviceSynchronize());
        unsigned got=0;checked(cudaMemcpy(&got,nhit,4,cudaMemcpyDeviceToHost));
        std::vector<HitRec> result(std::min<unsigned>(got,TXID_MAX_HITS));
        if(got!=1025){fprintf(stderr,"Blackwell zero-tail hit-count self-check failed\n");std::exit(2);}
        checked(cudaMemcpy(result.data(),hits,result.size()*sizeof(HitRec),cudaMemcpyDeviceToHost));
        std::sort(result.begin(),result.end(),[](const HitRec&a,const HitRec&b){return a.n<b.n;});
        for(size_t i=0;i<result.size();++i){
            auto h=result[i];uint32_t n=(uint32_t)(bswap64h((uint32_t)h.n)>>32);
            const uint64_t expected=txid_h3<15,false>(p,p.m[15]|((uint64_t)n<<32));
            if((uint64_t)(n-low)>=1025||h.n!=(fixed|(uint64_t)(uint32_t)(bswap64h(n)>>32))||h.h3!=expected||(i&&result[i-1].n==h.n)) {
                fprintf(stderr,"Blackwell zero-tail CPU hash/nonce self-check failed\n");std::exit(2);
            }
        }
    }
    p.mask=~0ULL;
    cudaEvent_t begin,end;checked(cudaEventCreate(&begin));checked(cudaEventCreateWithFlags(&end,cudaEventBlockingSync));
    const uint64_t stride=(uint64_t)grid*512;
    // Use a mining-sized launch when comparing close geometries.
    const uint32_t loops=(uint32_t)std::max<uint64_t>(1,std::min<uint64_t>(1024,(1ULL<<30)/std::max<uint64_t>(stride,1)));
    auto timed_launch = [&](int mode) {
        checked(cudaMemset(nhit, 0, sizeof(unsigned)));
        checked(cudaEventRecord(begin));
        tail_launch(mode, grid, 0, p, 0x10000000u, fixed, loops, 0, hits, nhit, plan);
        checked(cudaGetLastError());
        checked(cudaEventRecord(end));
        checked(cudaEventSynchronize(end));
        float ms = 0;
        checked(cudaEventElapsedTime(&ms, begin, end));
        if (!(ms > 0)) { fprintf(stderr, "Invalid zero-tail timing\n"); std::exit(2); }
        return ms;
    };
    // Warm the card once. Each arm then has a settling launch and at least
    // one mining-sized timed launch. The second pass reverses the clock drift.
    float warm = 0;
    while (warm < 75.f) warm += timed_launch(0);
    float times[4] = {};
    for (int mode : {0,1,2,3,3,2,1,0}) {
        timed_launch(mode);
        float total = 0;
        unsigned samples = 0;
        do { total += timed_launch(mode); ++samples; } while (total < 24.f);
        times[mode] += total / samples;
    }
    chosen=times[2]<times[1]?2:1;
    if (times[3] < times[chosen] * .99f) chosen = 3;
    if(!(times[chosen]<times[0]*.98f))chosen=0;
    if(force&&!strcmp(force,"1"))chosen=1;
    if(force&&!strcmp(force,"2"))chosen=2;
    if(force&&!strcmp(force,"3"))chosen=3;
    fprintf(stderr,"GPU %d: Blackwell long-job tuning selects %s (existing %.3f ms, zero-tail LB2 %.3f ms, LB3 %.3f ms, 768-thread %.3f ms)\n",dev,
        chosen==1?"zero-tail LB2":chosen==2?"zero-tail LB3":chosen==3?"zero-tail 768-thread":"existing kernel",times[0]*.5f,times[1]*.5f,times[2]*.5f,times[3]*.5f);
    checked(cudaEventDestroy(begin));checked(cudaEventDestroy(end));checked(cudaFree(hits));checked(cudaFree(nhit));
    choices.mode[dev].store(chosen);return chosen;
}
} // namespace txid_bw

extern "C" void txid_bw_launch(int key, unsigned grid, unsigned block, cudaStream_t stream,
        const txid::ParamsCore &p, uint32_t low, uint64_t fixed, uint32_t full, uint32_t tail,
        txid::HitRec *hits, unsigned *nhit, uint32_t nshift, const txid::Plan01 &plan, bool allow_tail) {
    // Retain the caller's exact nonce count, including partial grids, while
    // grouping the same work into the measured 512-thread Blackwell blocks.
    const uint64_t threads = (uint64_t)grid * block;
    const uint64_t count = threads * full + tail;
    grid = (unsigned)((threads + 511) / 512); block = 512;
    const uint64_t stride = (uint64_t)grid * block;
    full = (uint32_t)(count / stride); tail = (uint32_t)(count % stride);
    if(allow_tail && txid_bw::tail_shape(key,p,nshift)) {
        const int mode=txid_bw::tail_choice(grid,p,fixed,plan);
        txid_bw::tail_launch(mode,grid,stream,p,low,fixed,full,tail,hits,nhit,plan);
        return;
    }
    txid_bw::dispatch<0>(key, grid, block, stream, p, low, fixed, full, tail, hits, nhit, nshift, plan);
}
