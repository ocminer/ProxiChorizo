// ProxiChorizo - Copyright (c) 2026 suprnova.cc. ProxiChorizo License 1.0; see LICENSE.
#pragma once
#include "txid_cuda.cuh"
#include "txid_zero_tail.h"
namespace txid {
// Set once during per-device calibration, before that device starts mining.
// Zero means the original 256-thread launch, including on unmeasured devices.
inline unsigned zero_tail_blocks[128] = {};
template <int BLOCK> __global__ void __launch_bounds__(BLOCK)
zero_tail_kernel(const ParamsCore p, uint64_t nbase, uint64_t count,
                 uint32_t iters, HitRec *hits, unsigned *nhit) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (uint32_t i = 0; i < iters; ++i) {
        const uint64_t idx = gid + (uint64_t)i * stride;
        if (idx >= count) break;
        const uint64_t n = nbase + idx;
        const uint64_t h3 = zero_tail_h3(p, bswap64d(n));
        if (is_hit(p, h3)) record(n, h3, hits, nhit);
    }
}
static inline void launch_zero_tail(unsigned grid, unsigned block, cudaStream_t stream,
        const Params &p, uint64_t first, uint64_t count, uint32_t iters, HitRec *hits, unsigned *nhit,
        unsigned forced_block = 0) {
    // A wider Pascal candidate is compared with BOTH previous choices. Never
    // replace a card's faster old zero-tail kernel merely because it shares SM61.
    int device=-1;cudaGetDevice(&device);
    const unsigned selected_block=forced_block?forced_block:device>=0&&device<128?zero_tail_blocks[device]:256;
    if(selected_block==512) {
        const uint64_t threads=(uint64_t)grid*block;
        grid=(unsigned)((threads+511)/512);
        const uint64_t stride=(uint64_t)grid*512;
        const uint32_t rounds=stride?(uint32_t)((count+stride-1)/stride):0;
        zero_tail_kernel<512><<<grid,512,0,stream>>>(static_cast<const ParamsCore &>(p),first,count,rounds,hits,nhit);
        return;
    }
    zero_tail_kernel<256><<<grid, block, 0, stream>>>(static_cast<const ParamsCore &>(p), first, count, iters, hits, nhit);
}
struct CudaZeroTailSearch : CudaSearch {
    unsigned chosen_block=256;
    std::vector<HitRec> operator()(const Params &p, uint64_t first, uint64_t count) {
        std::vector<HitRec> out;
        if (!is_zero_tail(p)) return out;
        cudaMemset(d_n, 0, sizeof(unsigned));
        const unsigned block = 256, iters = 4;
        const unsigned grid = (unsigned)((count + block * iters - 1) / (block * iters));
        if (count) launch_zero_tail(grid, block, 0, p, first, count, iters, d_hits, d_n,chosen_block);
        if (cudaDeviceSynchronize() != cudaSuccess) return {{~0ULL, 0}};
        unsigned n = 0;
        if (cudaMemcpy(&n, d_n, sizeof n, cudaMemcpyDeviceToHost) != cudaSuccess) return {{~0ULL, 0}};
        out.resize(std::min<unsigned>(n, TXID_MAX_HITS));
        if (!out.empty() && cudaMemcpy(out.data(), d_hits, out.size() * sizeof(HitRec), cudaMemcpyDeviceToHost) != cudaSuccess) return {{~0ULL, 0}};
        return out;
    }
};
static inline int cuda_zero_tail_selfcheck(std::string &why) {
    CudaZeroTailSearch search;int rc=zero_tail_selfcheck(std::ref(search),why);
    int device=-1;cudaDeviceProp prop{};cudaGetDevice(&device);cudaGetDeviceProperties(&prop,device);
    if(!rc&&prop.major==6&&prop.minor==1){search.chosen_block=512;std::string wide;rc=zero_tail_selfcheck(std::ref(search),wide);why+="; 512-thread "+wide;}
    return rc;
}
} // namespace txid
