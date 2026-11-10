// ProxiChorizo - Copyright (c) 2026 suprnova.cc. ProxiChorizo License 1.0; see LICENSE.
#pragma once
#include "txid_cuda.cuh"
#include <cstdlib>

namespace txid {
// Only measured devices and message layouts use this search. Other devices and
// all unlisted layouts retain their existing kernels, including block straddles.
static inline bool aligned_device(const cudaDeviceProp &p) {
    if (const char *e = getenv("PX_TXID_ALIGNED")) if (!strcmp(e, "0")) return false;
    return p.major == 12 && p.minor == 0 &&
        (!strcmp(p.name, "NVIDIA GeForce RTX 5090") || !strcmp(p.name, "GeForce RTX 5090"));
}
static inline bool aligned_shape(int plen) {
    int off = plen % 128;
    return off == 52 || off == 97 || off == 108 || off == 113;
}
static inline int aligned_offset(int plen) { return ((plen + 5) / 4) * 4 - plen; }
// A bijection of the 48-bit extranonce2 space. The low 32 logical bits become
// four little-endian bytes aligned in the message; the remaining two bytes
// hold the high 16 bits. Extranonce1 is unchanged. GPU ranges stay disjoint.
static inline uint64_t aligned_nonce(int off, uint64_t logical) {
    uint64_t n = logical & 0xffff000000000000ULL;
    int j = 0;
    for (int i = 2; i < 8; i++) {
        uint8_t b = i >= off && i < off + 4 ? (uint8_t)(logical >> (8 * (i - off))) : (uint8_t)(logical >> (32 + 8 * j++));
        n |= (uint64_t)b << (56 - 8 * i);
    }
    return n;
}
static inline uint64_t aligned_count(uint64_t nbase, uint64_t count) {
    return std::min<uint64_t>(count, (1ULL << 32) - (uint32_t)nbase);
}
struct AlignedSearch {
    ParamsCore p{};
    uint64_t window = ~0ULL, fixed = 0;
    void reset() { window = ~0ULL; }
    void prepare(const Params &base, int plen, uint64_t nbase) {
        if (window == nbase >> 32) { p.mask = base.mask; return; }
        window = nbase >> 32; p = static_cast<const ParamsCore &>(base);
        fixed = aligned_nonce(aligned_offset(plen), nbase & 0xffffffff00000000ULL);
        int off = plen % 128;
        for (int i = 0; i < 8; i++) p.m[(off + i) / 8] |= (uint64_t)(uint8_t)(fixed >> (56 - 8 * i)) << (8 * ((off + i) % 8));
        // Recompute just the first round with the fixed nonce bytes. The prefix
        // chaining value is retained; all 12 Blake2b rounds remain exact.
        uint64_t c[16]; memcpy(c, p.vi, sizeof c); memcpy(p.vhc, c, sizeof c);
        for (int j = 0; j < 4; j++) {
            g1(p.vhc[j], p.vhc[4+j], p.vhc[8+j], p.vhc[12+j], p.m[2*j]);
            gf(c[j], c[4+j], c[8+j], c[12+j], p.m[2*j], p.m[2*j+1]);
        }
        memcpy(p.vc, c, sizeof c); memcpy(p.vhd, c, sizeof c); memcpy(p.vr, c, sizeof c);
        for (int j = 0; j < 4; j++) {
            int a=j, b=4+(j+1)%4, cc=8+(j+2)%4, d=12+(j+3)%4;
            g1(p.vhd[a], p.vhd[b], p.vhd[cc], p.vhd[d], p.m[8+2*j]);
            gf(p.vr[a], p.vr[b], p.vr[cc], p.vr[d], p.m[8+2*j], p.m[9+2*j]);
        }
    }
};

template <int OFFSET> __global__ void __launch_bounds__(TXID_TPB)
aligned_kernel(const ParamsCore p, uint32_t low, uint64_t fixed, uint32_t full, uint32_t tail, HitRec *hits, unsigned *nhit) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200
    constexpr int OFF=((OFFSET+5)/4)*4-OFFSET, W=(OFFSET+OFF)/8, HIGH=((OFFSET+OFF)%8)/4, LAST=(OFFSET+7)/8;
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
    uint32_t n = low + gid, limit = full + (gid < tail);
    for (uint32_t i = 0; i < limit; i++, n += stride) {
        uint64_t m[16], v[16];
#pragma unroll
        for (int j = 0; j < 16; j++) m[j] = j <= LAST ? p.m[j] : 0;
        m[W] |= (uint64_t)n << (32 * HIGH);
        round0<W, false>(v, p, m); rounds<1>(v, m);
        uint64_t h3 = p.h[3] ^ v[3] ^ v[11];
        if ((h3 & p.mask) == 0) {
            uint64_t nonce = fixed | ((uint64_t)__byte_perm(n, 0, 0x0123) << (8 * (4 - OFF)));
            record(nonce, h3, hits, nhit);
        }
    }
#endif
}
// Caller bounds count at a 2^32 logical-counter boundary and resets cache on a
// new job. Tail threads run one additional nonce: no padded or repeated work.
static inline void launch_aligned(unsigned grid, unsigned block, cudaStream_t st, const Params &base, int plen,
                                  uint64_t nbase, uint64_t count, HitRec *hits, unsigned *nhit, AlignedSearch &cache) {
    cache.prepare(base, plen, nbase);
    uint64_t stride = (uint64_t)grid * block;
    uint32_t full = (uint32_t)(count / stride), tail = (uint32_t)(count % stride);
#define TX_ALIGNED_CASE(O) case O: aligned_kernel<O><<<grid, block, 0, st>>>(cache.p, (uint32_t)nbase, cache.fixed, full, tail, hits, nhit); break
    switch (plen % 128) { TX_ALIGNED_CASE(52); TX_ALIGNED_CASE(97); TX_ALIGNED_CASE(108); TX_ALIGNED_CASE(113); }
#undef TX_ALIGNED_CASE
}

// Same independent full CPU hash used for ordinary-kernel checks. Exercise
// fixed-byte changes, partial grids, zero/low masks and 2^32 carry boundaries.
static inline int aligned_selfcheck(std::string &why) {
    HitRec *dh=nullptr; unsigned *dn=nullptr;
    cudaError_t e = cudaMalloc(&dh, TXID_MAX_HITS * sizeof(HitRec));
    if (e == cudaSuccess) e = cudaMalloc(&dn, sizeof(unsigned));
    if (e != cudaSuccess) { cudaFree(dh); why=cudaGetErrorString(e); return 1; }
    int cases=0; uint64_t checked=0;
    for (int off : {52,97,108,113}) for (int blocks : {0,2,5}) for (int test=0; test<8; test++) {
        int plen=blocks*128+off, k=test%3==0 ? 0 : test%3==1 ? 3 : 12;
        std::vector<uint8_t> prefix(plen); for (int i=0; i<plen; i++) prefix[i]=(uint8_t)(i*31+test*19);
        Params p; prepare(prefix.data(), plen, k, p); AlignedSearch cache;
        uint64_t first=0xa1b27856ffffff00ULL+(uint64_t)test*0x123450011ULL;
        if (test==7) first=0xfffffffffffffeffULL;
        uint64_t count=513+test*17; std::vector<HitRec> got,want;
        for (uint64_t done=0; done<count;) {
            uint64_t nbase=first+done, batch=aligned_count(nbase,count-done);
            e=cudaMemset(dn,0,sizeof(unsigned));
            if (e==cudaSuccess) { launch_aligned(3,128,0,p,plen,nbase,batch,dh,dn,cache); e=cudaDeviceSynchronize(); }
            unsigned n=0; if(e==cudaSuccess)e=cudaMemcpy(&n,dn,sizeof n,cudaMemcpyDeviceToHost);
            if(e!=cudaSuccess||n>TXID_MAX_HITS) { why=e==cudaSuccess?"aligned hit overflow":cudaGetErrorString(e);cudaFree(dh);cudaFree(dn);return 1; }
            size_t at=got.size();got.resize(at+n);if(n)e=cudaMemcpy(got.data()+at,dh,n*sizeof(HitRec),cudaMemcpyDeviceToHost);
            if(e!=cudaSuccess){why=cudaGetErrorString(e);cudaFree(dh);cudaFree(dn);return 1;} done+=batch;
        }
        for(uint64_t i=0;i<count;i++) {
            uint64_t n=aligned_nonce(aligned_offset(plen),first+i);uint8_t h[32];hash_nonce(prefix,n,h);
            if(work_bits(h)>=k)want.push_back({n,load64le(h+24)});
        }
        auto less=[](const HitRec&a,const HitRec&b){return a.n<b.n;};std::sort(got.begin(),got.end(),less);std::sort(want.begin(),want.end(),less);
        bool same=got.size()==want.size();for(size_t i=0;same&&i<got.size();i++)same=got[i].n==want[i].n&&got[i].h3==want[i].h3;
        if(!same){why="aligned search differs from CPU at prefix "+std::to_string(plen);cudaFree(dh);cudaFree(dn);return 1;}
        cases++;checked+=count;
    }
    cudaFree(dh);cudaFree(dn);why="aligned search: "+std::to_string(cases)+" cases, "+std::to_string(checked)+" CPU/GPU nonces";return 0;
}
} // namespace txid
