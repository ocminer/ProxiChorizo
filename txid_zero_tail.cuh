// ProxiChorizo - Copyright (c) 2026 suprnova.cc. ProxiChorizo License 1.0; see LICENSE.
#pragma once
#include "txid_cuda.cuh"
#include "txid_zero_tail.h"
namespace txid {
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
        const Params &p, uint64_t first, uint64_t count, uint32_t iters, HitRec *hits, unsigned *nhit) {
    zero_tail_kernel<256><<<grid, block, 0, stream>>>(static_cast<const ParamsCore &>(p), first, count, iters, hits, nhit);
}
struct CudaZeroTailSearch : CudaSearch {
    std::vector<HitRec> operator()(const Params &p, uint64_t first, uint64_t count) {
        std::vector<HitRec> out;
        if (!is_zero_tail(p)) return out;
        cudaMemset(d_n, 0, sizeof(unsigned));
        const unsigned block = 256, iters = 4;
        const unsigned grid = (unsigned)((count + block * iters - 1) / (block * iters));
        if (count) launch_zero_tail(grid, block, 0, p, first, count, iters, d_hits, d_n);
        if (cudaDeviceSynchronize() != cudaSuccess) return {{~0ULL, 0}};
        unsigned n = 0;
        if (cudaMemcpy(&n, d_n, sizeof n, cudaMemcpyDeviceToHost) != cudaSuccess) return {{~0ULL, 0}};
        out.resize(std::min<unsigned>(n, TXID_MAX_HITS));
        if (!out.empty() && cudaMemcpy(out.data(), d_hits, out.size() * sizeof(HitRec), cudaMemcpyDeviceToHost) != cudaSuccess) return {{~0ULL, 0}};
        return out;
    }
};
static inline int cuda_zero_tail_selfcheck(std::string &why) {
    CudaZeroTailSearch search; return zero_tail_selfcheck(std::ref(search), why);
}
} // namespace txid
