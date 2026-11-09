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
// TXID_TRIM (bits; per measured architecture, see txid_blake2b.h; 0 elsewhere = the original loop):
//  1: when every thread runs all `iters` nonces and the launch stays inside one 2^32 window of the nonce's low word
//     (a launch-uniform test), a loop over the low word only: the high word of X is one byte permute per nonce and
//     the bounds test is the loop counter;
//  2: when the share mask has no bits in the low word of h3 (k <= 32), only the high word is tested first.
template <int ID> __device__ __forceinline__ uint64_t h3_of(const ParamsCore &p, uint64_t X, const Plan01 &q) {
    if constexpr (ID == 31) return txid_h3_straddle(p, X, &q);
    else return txid_h3<ID / 2, (ID & 1) != 0>(p, X, &q);
}
__device__ __forceinline__ bool is_hit(const ParamsCore &p, uint64_t h3) {
    if (TXID_TRIM & 2) {
        if (((uint32_t)(h3 >> 32) & (uint32_t)(p.mask >> 32)) != 0) return false;
        return ((uint32_t)h3 & (uint32_t)p.mask) == 0;
    }
    return (h3 & p.mask) == 0;
}
__device__ __forceinline__ void record(uint64_t n, uint64_t h3, HitRec *hits, unsigned *nhit) {
    unsigned s = atomicAdd(nhit, 1u);
    if (s < TXID_MAX_HITS) hits[s] = HitRec{n, h3};
}
template <int ID>
__global__ void __launch_bounds__(TXID_TPB) search_kernel(const ParamsCore p, uint64_t nbase, uint64_t count, uint32_t iters, HitRec *hits, unsigned *nhit, const Plan01 q) {
#if TXID_TRIM & 1
    {
        const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x, stride32 = gridDim.x * blockDim.x;
        if ((uint64_t)stride32 * iters == count && (uint64_t)(uint32_t)nbase + count <= (1ULL << 32)) {
            const uint64_t xl = __byte_perm((uint32_t)(nbase >> 32), 0, 0x0123);
            uint32_t nlo = (uint32_t)nbase + tid;
            for (uint32_t i = 0; i < iters; i++, nlo += stride32) {
                const uint64_t X = ((uint64_t)__byte_perm(nlo, 0, 0x0123) << 32) | xl;
                const uint64_t h3 = h3_of<ID>(p, X, q);
                if (is_hit(p, h3)) record((nbase & 0xffffffff00000000ULL) | nlo, h3, hits, nhit);
            }
            return;
        }
    }
#endif
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (uint32_t i = 0; i < iters; i++) {
        const uint64_t idx = gid + (uint64_t)i * stride;
        if (idx >= count) break;
        const uint64_t n = nbase + idx;
        const uint64_t X = bswap64d(n);
#if TXID_TRIM
        const uint64_t h3 = h3_of<ID>(p, X, q);
        if (is_hit(p, h3)) record(n, h3, hits, nhit);
#else
        uint64_t h3;
        if constexpr (ID == 31) h3 = txid_h3_straddle(p, X);
        else h3 = txid_h3<ID / 2, (ID & 1) != 0>(p, X);
        if ((h3 & p.mask) == 0) {
            unsigned s = atomicAdd(nhit, 1u);
            if (s < TXID_MAX_HITS) hits[s] = HitRec{n, h3};
        }
#endif
    }
}
template <int ID> static void launch_id(int id, unsigned grid, unsigned block, cudaStream_t st, const Params &p, uint64_t nbase, uint64_t count, uint32_t iters, HitRec *hits, unsigned *nhit) {
    if (id == ID) { search_kernel<ID><<<grid, block, 0, st>>>(static_cast<const ParamsCore &>(p), nbase, count, iters, hits, nhit, p.q); return; }
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
