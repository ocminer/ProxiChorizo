// ProxiChorizo - Copyright (c) 2026 suprnova.cc. ProxiChorizo License 1.0: free to run for mining;
// reuse in commercial or closed-source miners needs written permission. See LICENSE.
//
// txid_cuda.cuh — the CUDA search kernels for the txid-blake2b work (library upgrade 6).
// One kernel per message shape (32: the word that holds the first nonce byte, aligned or not, and the
// block-straddling nonce), so the nonce word, the zero padding after it and the precomputed part of
// round 0 are compile-time facts. A thread hashes `iters` nonces nbase + gid + i*stride; a hit is a
// nonce whose output word 3 has the share's k zero bits (hash bytes 24..31 read big-endian).
#pragma once
#include "txid_blake2b.h"
namespace txid {
#ifndef TXID_TPB
#define TXID_TPB 256
#endif
__device__ __forceinline__ uint64_t bswap64d(uint64_t x) {
    return ((uint64_t)__byte_perm((uint32_t)x, 0, 0x0123) << 32) | __byte_perm((uint32_t)(x >> 32), 0, 0x0123);
}
template <int ID>
__global__ void __launch_bounds__(TXID_TPB) search_kernel(const Params p, uint64_t nbase, uint64_t count, uint32_t iters, HitRec *hits, unsigned *nhit) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (uint32_t i = 0; i < iters; i++) {
        const uint64_t idx = gid + (uint64_t)i * stride;
        if (idx >= count) break;
        const uint64_t n = nbase + idx;
        const uint64_t X = bswap64d(n);
        uint64_t h3;
        if constexpr (ID == 31) h3 = txid_h3_straddle(p, X);
        else h3 = txid_h3<ID / 2, (ID & 1) != 0>(p, X);
        if ((h3 & p.mask) == 0) {
            unsigned s = atomicAdd(nhit, 1u);
            if (s < TXID_MAX_HITS) hits[s] = HitRec{n, h3};
        }
    }
}
template <int ID> static void launch_id(int id, unsigned grid, unsigned block, cudaStream_t st, const Params &p, uint64_t nbase, uint64_t count, uint32_t iters, HitRec *hits, unsigned *nhit) {
    if (id == ID) { search_kernel<ID><<<grid, block, 0, st>>>(p, nbase, count, iters, hits, nhit); return; }
    if constexpr (ID < 31) launch_id<ID + 1>(id, grid, block, st, p, nbase, count, iters, hits, nhit);
}
// one search launch over [nbase, nbase + count); grid*block*iters must cover count
static inline void launch(unsigned grid, unsigned block, cudaStream_t st, const Params &p, uint64_t nbase, uint64_t count, uint32_t iters, HitRec *hits, unsigned *nhit) {
    launch_id<0>((int)p.kernel, grid, block, st, p, nbase, count, iters, hits, nhit);
}

// a synchronous search for the self-check: every hit of [nbase, nbase+count)
struct CudaSearch {
    HitRec *d_hits = nullptr; unsigned *d_n = nullptr;
    CudaSearch() { cudaMalloc(&d_hits, sizeof(HitRec) * TXID_MAX_HITS); cudaMalloc(&d_n, sizeof(unsigned)); }
    ~CudaSearch() { cudaFree(d_hits); cudaFree(d_n); }
    std::vector<HitRec> operator()(const Params &p, uint64_t nbase, uint64_t count) {
        std::vector<HitRec> out;
        cudaMemset(d_n, 0, sizeof(unsigned));
        unsigned block = 128, iters = 4; uint64_t per = (uint64_t)block * iters;
        unsigned grid = (unsigned)((count + per - 1) / per);
        launch(grid, block, 0, p, nbase, count, iters, d_hits, d_n);
        if (cudaDeviceSynchronize() != cudaSuccess) { out.push_back(HitRec{~0ULL, 0}); return out; }
        unsigned n = 0; cudaMemcpy(&n, d_n, sizeof n, cudaMemcpyDeviceToHost);
        if (n > TXID_MAX_HITS) n = TXID_MAX_HITS;
        out.resize(n);
        if (n) cudaMemcpy(out.data(), d_hits, sizeof(HitRec) * n, cudaMemcpyDeviceToHost);
        return out;
    }
};
// the self-check on the current device; 0 = pass
static inline int cuda_selfcheck(std::string &why) { CudaSearch s; return selfcheck(std::ref(s), why); }
} // namespace txid
