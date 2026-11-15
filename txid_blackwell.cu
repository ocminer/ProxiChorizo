// ProxiChorizo - Copyright (c) 2026 suprnova.cc. ProxiChorizo License 1.0; see LICENSE.
// Compile this translation unit as compute_120 PTX only. Its driver-compiled
// kernels coexist with the unchanged native kernels in proxichorizo.cu.
#include <cuda_runtime.h>
#define TXID_BW_DEVICE 1
#include "txid_blake2b.h"

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

extern "C" void txid_bw_launch(int key, unsigned grid, unsigned block, cudaStream_t stream,
        const txid::ParamsCore &p, uint32_t low, uint64_t fixed, uint32_t full, uint32_t tail,
        txid::HitRec *hits, unsigned *nhit, uint32_t nshift, const txid::Plan01 &plan) {
    // Retain the caller's exact nonce count, including partial grids, while
    // grouping the same work into the measured 512-thread Blackwell blocks.
    const uint64_t threads = (uint64_t)grid * block;
    const uint64_t count = threads * full + tail;
    grid = (unsigned)((threads + 511) / 512); block = 512;
    const uint64_t stride = (uint64_t)grid * block;
    full = (uint32_t)(count / stride); tail = (uint32_t)(count % stride);
    txid_bw::dispatch<0>(key, grid, block, stream, p, low, fixed, full, tail, hits, nhit, nshift, plan);
}
