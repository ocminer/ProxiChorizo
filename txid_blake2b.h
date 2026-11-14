// ProxiChorizo - Copyright (c) 2026 suprnova.cc. ProxiChorizo License 1.0: free to run for mining;
// reuse in commercial or closed-source miners needs written permission. See LICENSE.
//
// txid_blake2b.h — the "txid-blake2b" proof of work of Proxima library upgrade 6 (from slot 197000).
//
//   message = prefix (from mining.notify) || extranonce1 (2 bytes) || extranonce2 (6 bytes)
//   hash    = BLAKE2b-256 (RFC 7693, unkeyed, 32-byte digest)
//   work    = trailing zero bits of uint64_big_endian(hash[24..31]); a share when work >= k
//
// This header is shared by the CUDA miner (host and device), the OpenCL miner's host and the tests:
//  - a plain CPU BLAKE2b-256 (the reference every GPU share is checked against before it is sent),
//  - the per-job precomputation for the GPU search: the chaining value over every full 128-byte block
//    before the block that holds the first nonce byte, and the nonce-independent part of round 0,
//  - txid_h3<W, SH>() / txid_h3_straddle(): output word 3 for one nonce from that precomputation,
//    written once for both the CPU (to test the precomputation) and the GPU (the search kernel).
//
// The nonce is the last 8 bytes of the message, so it ends in the final block. Two shapes:
//  (a) all 8 nonce bytes in the final block at byte offset o: word W = o/8 holds the first nonce byte;
//      when o is not a multiple of 8 (SH), the nonce spills into word W+1. Words after it are zero.
//      One compression per attempt, with counter = message length and the final flag.
//  (b) "straddle": the nonce starts in the second-to-last block (offset 121..127) and ends in the final
//      block (1..7 bytes long). Two compressions per attempt; the first one is not final.
#pragma once
#include <stdint.h>
#include <string.h>
#include <string>
#include <vector>
#include <algorithm>
#include "txid_vectors.h"

#ifdef __CUDACC__
#define TXQ __host__ __device__ __forceinline__
#else
#define TXQ inline
#endif

#define TXID_MAX_HITS 4096   // hits kept per launch
namespace txid {

static const uint64_t B2B_IV[8] = {0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
                                   0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL, 0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL};
#define TXID_PARAM_WORD 0x01010020ULL   // digest length 32, no key, fanout 1, depth 1
// the IV as a constant expression (usable in device code, folded into immediates)
TXQ constexpr uint64_t iv(int i) {
    return i == 0 ? 0x6a09e667f3bcc908ULL : i == 1 ? 0xbb67ae8584caa73bULL : i == 2 ? 0x3c6ef372fe94f82bULL : i == 3 ? 0xa54ff53a5f1d36f1ULL :
           i == 4 ? 0x510e527fade682d1ULL : i == 5 ? 0x9b05688c2b3e6c1fULL : i == 6 ? 0x1f83d9abfb41bd6bULL : 0x5be0cd19137e2179ULL;
}

// the per-job input of a search kernel (passed by value: it lives in the kernel's constant bank)
struct ParamsCore {
    uint64_t m[16];   // the block holding the first nonce byte, nonce bytes zero
    uint64_t h[8];    // chaining value before that block
    uint64_t vi[16];  // compression state before round 0
    uint64_t vc[16];  // after the column step of round 0 (valid for the columns without nonce words)
    uint64_t vhc[16]; // each column G after its first half (from vi)
    uint64_t vhd[16]; // each diagonal G after its first half (from vc)
    uint64_t vr[16];  // after round 0 (valid for the diagonal Gs without nonce words, when no column has one)
    uint64_t mask;    // byte-swapped low-k mask: a share when (h3 & mask) == 0
    uint64_t tfin;    // straddle: the final block's counter (the message length)
    uint32_t shift;   // 8 * (nonce offset % 8)
    uint32_t kernel;  // the shape: 2*W + SH for (a), 31 for the straddle
};
// TXID_TRIM 4 (rounds 0 and 1 from a per-step plan, see Plan): every G-step result of rounds 0..1 with the nonce bytes
// zero (snap; the true value for every step that does not depend on the nonce) and per G half the sum of the two
// nonce-free operands of a + b + m when exactly one depends on the nonce (pre). A separate, last kernel argument, so
// the ParamsCore layout (read word by word by the OpenCL kernels) and the other architectures' code stay as in 2.0.0.
struct Plan01 {
    uint64_t snap[16 * 8];
    uint64_t pre[16 * 2];
};
struct Params : ParamsCore { Plan01 q; };

static inline uint64_t bswap64h(uint64_t x) {
    x = ((x & 0x00ff00ff00ff00ffULL) << 8) | ((x >> 8) & 0x00ff00ff00ff00ffULL);
    x = ((x & 0x0000ffff0000ffffULL) << 16) | ((x >> 16) & 0x0000ffff0000ffffULL);
    return (x << 32) | (x >> 32);
}

// ---- the compression, shared by CPU and GPU ---------------------------------------------
template <int R> TXQ uint64_t rotr(uint64_t x) {
#ifdef __CUDA_ARCH__
    uint32_t lo = (uint32_t)x, hi = (uint32_t)(x >> 32), a, b;
    if (R == 32) return ((uint64_t)lo << 32) | hi;
#ifndef TXID_ROT_IMAD
#define TXID_ROT_IMAD 0   // bits: 1 rot24, 2 rot16, 4 rot63 computed on the multiply-add pipe (wide multiplies by 2^s)
#endif
    if (R == 63 && (TXID_ROT_IMAD & 32)) {
        // rotl by 1 without the shifter: y = x + x (64-bit add); s = hi's sign bit replicated by a byte permute
        // (0 or 0xffffffff); y.lo - s adds the wrapped bit (y.lo is even, so no carry). The subtraction is a
        // multiply-add s * 0xffffffff + y.lo so it stays on the multiply-add pipe.
        const uint64_t y = x + x;
        uint32_t sgn; asm("prmt.b32 %0, %1, 0, 0xBBBB;" : "=r"(sgn) : "r"(hi));   // __byte_perm drops the sign-replicate bit
        uint32_t m1, l; asm("mov.b32 %0, -1;" : "=r"(m1));
        asm("mad.lo.u32 %0, %1, %2, %3;" : "=r"(l) : "r"(sgn), "r"(m1), "r"((uint32_t)y));
        return (y & 0xffffffff00000000ULL) | l;
    }
    if (R == 63 && (TXID_ROT_IMAD & 16)) {
        // rotl by 1: x + x (64-bit add) | hi >> 31 taken as the high half of hi * 2 (multiply pipe, no shifter)
        uint32_t two, top; asm("mov.b32 %0, 2;" : "=r"(two));
        asm("mul.hi.u32 %0, %1, %2;" : "=r"(top) : "r"(hi), "r"(two));
        return (x + x) | top;
    }
    if (R == 63 && (TXID_ROT_IMAD & 8)) {
        // rotl by 1 as add-with-carry: lo+lo carries lo's top bit into hi+hi, whose carry is hi's top bit for lo
        uint32_t l, h;
        asm("add.cc.u32 %0, %2, %2;\n\taddc.cc.u32 %1, %3, %3;\n\taddc.u32 %0, %0, 0;" : "=&r"(l), "=&r"(h) : "r"(lo), "r"(hi));
        return ((uint64_t)h << 32) | l;
    }
    if ((R == 24 && (TXID_ROT_IMAD & 1)) || (R == 16 && (TXID_ROT_IMAD & 2)) || (R == 63 && (TXID_ROT_IMAD & 4))) {
        // rotr(x, R) == rotl(x, s), s = 64 - R; per 32-bit half: lo*2^s and hi*2^s (s mod 32) as 64-bit products.
        // R < 32: result.lo = (lo >> R) | (hi << (32-R)), result.hi = (hi >> R) | (lo << (32-R)).
        // R > 32 (63): rotl by 1 on the swapped halves.  The pieces never overlap, so + equals |.
        constexpr uint32_t K = R < 32 ? (1u << (32 - R)) : (1u << (64 - R));
        uint32_t x0 = R < 32 ? lo : hi, x1 = R < 32 ? hi : lo;     // R > 32: rotr(x,R) = rotr(swap(x), R-32)
        uint64_t t; asm("mul.wide.u32 %0, %1, %2;" : "=l"(t) : "r"(x1), "r"(K));
        const uint32_t tlo = (uint32_t)t, thi = (uint32_t)(t >> 32);
        uint32_t ulo, uhi; uint64_t u; asm("mad.wide.u32 %0, %1, %2, %3;" : "=l"(u) : "r"(x0), "r"(K), "l"((uint64_t)tlo << 32));
        ulo = (uint32_t)u; uhi = (uint32_t)(u >> 32);
        uint32_t r1; asm("mad.lo.u32 %0, %1, 1, %2;" : "=r"(r1) : "r"(ulo), "r"(thi));
        // result low word = (x0 >> (32-s')) | (x1 << s') = uhi; result high word = r1
        return ((uint64_t)r1 << 32) | uhi;
    }
#ifndef TXID_ROT_PRMT
#define TXID_ROT_PRMT 1
#endif
    if (TXID_ROT_PRMT && (R == 16 || R == 24)) {
        // byte rotations as byte permutes (PRMT): 1 instruction per half, like SHF, but a different pipe mix
        const uint32_t sel = R == 16 ? 0x5432 : 0x6543;
        a = __byte_perm(lo, hi, sel); b = __byte_perm(hi, lo, sel);
        return ((uint64_t)b << 32) | a;
    }
    if (R < 32) {
        asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(a) : "r"(lo), "r"(hi), "n"(R & 31));
        asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(b) : "r"(hi), "r"(lo), "n"(R & 31));
    } else {
        asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(a) : "r"(hi), "r"(lo), "n"(R & 31));
        asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(b) : "r"(lo), "r"(hi), "n"(R & 31));
    }
    return ((uint64_t)b << 32) | a;
#else
    return (x >> R) | (x << (64 - R));
#endif
}
TXQ void g1(uint64_t &a, uint64_t &b, uint64_t &c, uint64_t &d, uint64_t x) { a += b + x; d = rotr<32>(d ^ a); c += d; b = rotr<24>(b ^ c); }
TXQ void g2(uint64_t &a, uint64_t &b, uint64_t &c, uint64_t &d, uint64_t y) { a += b + y; d = rotr<16>(d ^ a); c += d; b = rotr<63>(b ^ c); }
TXQ void gf(uint64_t &a, uint64_t &b, uint64_t &c, uint64_t &d, uint64_t x, uint64_t y) { g1(a, b, c, d, x); g2(a, b, c, d, y); }

// ---- per-architecture search options ---------------------------------------------------------
// TXID_TRIM (bits) selects loop/precomputation trims of the search kernels (txid_cuda.cuh, rounds01 below) for the
// architectures they were measured on; every other architecture compiles the v2.0.0 code (TXID_TRIM 0).
#ifdef __CUDA_ARCH__
#ifndef TXID_TRIM_SM86
#define TXID_TRIM_SM86 5   // RTX 3070: fast loop + rounds 0..1 plan, +0.6..1.3% (bit 2 measured slower there)
#endif
#ifndef TXID_TRIM
#if __CUDA_ARCH__ == 860
#define TXID_TRIM TXID_TRIM_SM86
#else
#define TXID_TRIM 0
#endif
#endif
#else
#undef TXID_TRIM
#define TXID_TRIM 0
#endif

#define TXID_SIGMA_ROWS {{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},{14,10,4,8,9,15,13,6,1,12,0,2,11,7,5,3},\
    {11,8,12,0,5,2,15,13,10,14,3,6,7,1,9,4},{7,9,3,1,13,12,11,14,2,6,5,10,4,0,15,8},{9,0,5,7,2,4,10,15,14,1,11,12,6,8,3,13},\
    {2,12,6,10,0,11,8,3,4,13,7,5,15,14,1,9},{12,5,1,15,14,13,4,10,0,7,6,3,9,2,8,11},{13,11,7,14,12,1,3,9,5,0,15,4,8,6,2,10},\
    {6,15,14,9,11,3,0,8,12,2,13,7,1,4,10,5},{10,2,8,4,7,6,1,5,15,11,9,14,3,12,13,0},{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},\
    {14,10,4,8,9,15,13,6,1,12,0,2,11,7,5,3}}

// rounds FROM..11 on v with message m (fully unrolled: the sigma indices become register names)
template <int FROM> TXQ void rounds(uint64_t v[16], const uint64_t m[16]) {
    constexpr int S[12][16] = TXID_SIGMA_ROWS;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (int r = FROM; r < 12; r++) {
        gf(v[0], v[4], v[8], v[12], m[S[r][0]], m[S[r][1]]);
        gf(v[1], v[5], v[9], v[13], m[S[r][2]], m[S[r][3]]);
        gf(v[2], v[6], v[10], v[14], m[S[r][4]], m[S[r][5]]);
        gf(v[3], v[7], v[11], v[15], m[S[r][6]], m[S[r][7]]);
        gf(v[0], v[5], v[10], v[15], m[S[r][8]], m[S[r][9]]);
        gf(v[1], v[6], v[11], v[12], m[S[r][10]], m[S[r][11]]);
        gf(v[2], v[7], v[8], v[13], m[S[r][12]], m[S[r][13]]);
        gf(v[3], v[4], v[9], v[14], m[S[r][14]], m[S[r][15]]);
    }
}

// round 0 of the nonce block from the per-job precomputation: only the Gs that read a nonce word run.
// DEP(i): word i holds nonce bytes. Every condition below is a compile-time constant after unrolling.
template <int W, bool SH> TXQ constexpr bool dep(int i) { return i == W || (SH && i == W + 1 && W + 1 < 16); }
template <int W, bool SH> TXQ void round0(uint64_t v[16], const ParamsCore &p, const uint64_t m[16]) {
    const bool anyCol = W < 8;   // the first nonce word is in a column G
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (int j = 0; j < 4; j++) {
        const int a = j, b = 4 + j, c = 8 + j, d = 12 + j;
        const bool dx = dep<W, SH>(2 * j), dy = dep<W, SH>(2 * j + 1);
        if (!dx && !dy) { v[a] = p.vc[a]; v[b] = p.vc[b]; v[c] = p.vc[c]; v[d] = p.vc[d]; }
        else if (!dx) { v[a] = p.vhc[a]; v[b] = p.vhc[b]; v[c] = p.vhc[c]; v[d] = p.vhc[d]; g2(v[a], v[b], v[c], v[d], m[2 * j + 1]); }
        else { v[a] = p.vi[a]; v[b] = p.vi[b]; v[c] = p.vi[c]; v[d] = p.vi[d]; gf(v[a], v[b], v[c], v[d], m[2 * j], m[2 * j + 1]); }
    }
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (int j = 0; j < 4; j++) {
        const int a = j, b = 4 + (j + 1) % 4, c = 8 + (j + 2) % 4, d = 12 + (j + 3) % 4;
        const bool dx = dep<W, SH>(8 + 2 * j), dy = dep<W, SH>(9 + 2 * j);
        if (anyCol) gf(v[a], v[b], v[c], v[d], m[8 + 2 * j], m[9 + 2 * j]);
        else if (!dx && !dy) { v[a] = p.vr[a]; v[b] = p.vr[b]; v[c] = p.vr[c]; v[d] = p.vr[d]; }
        else if (!dx) { v[a] = p.vhd[a]; v[b] = p.vhd[b]; v[c] = p.vhd[c]; v[d] = p.vhd[d]; g2(v[a], v[b], v[c], v[d], m[9 + 2 * j]); }
        else gf(v[a], v[b], v[c], v[d], m[8 + 2 * j], m[9 + 2 * j]);   // v already holds vc here
    }
}

// ---- rounds 0 and 1 from a compile-time plan (TXID_TRIM 4) ---------------------------------
// The nonce is in message word W (and W+1 when SH). Plan follows which state words depend on it through the 16 Gs of
// rounds 0 and 1 (8 steps each: a += b + m; d = rotr32(d ^ a); c += d; b = rotr24(b ^ c); a += b + m'; d = rotr16(d ^ a);
// c += d; b = rotr63(b ^ c)). A step whose inputs are nonce-free is a per-job constant (Plan01::snap); a + b + m with one
// nonce-dependent operand is one add with the precomputed sum of the other two (Plan01::pre).
struct Plan {
    bool dep[16][8] = {};   // the step's result depends on the nonce
    int pre[16][2] = {};    // a + b + m: 0 = plain, 1 = a + (b + m), 2 = b + (a + m), 3 = m + (a + b)
    int vi[16][4] = {}, mi[16][2] = {};
    TXQ constexpr Plan(int W, bool SH) {
        constexpr int S[2][16] = {{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},{14,10,4,8,9,15,13,6,1,12,0,2,11,7,5,3}};
        bool vd[16] = {};
        for (int g = 0; g < 16; g++) {
            const int r = g / 8, j = g % 8, q = j % 4;
            const int a = q, b = j < 4 ? 4 + q : 4 + (q + 1) % 4, c = j < 4 ? 8 + q : 8 + (q + 2) % 4, d = j < 4 ? 12 + q : 12 + (q + 3) % 4;
            vi[g][0] = a; vi[g][1] = b; vi[g][2] = c; vi[g][3] = d; mi[g][0] = S[r][2 * j]; mi[g][1] = S[r][2 * j + 1];
            for (int h = 0; h < 2; h++) {
                const int mw = mi[g][h];
                const bool md = mw == W || (SH && mw == W + 1);
                const int nd = (vd[a] ? 1 : 0) + (vd[b] ? 1 : 0) + (md ? 1 : 0);
                pre[g][h] = nd != 1 ? 0 : vd[a] ? 1 : vd[b] ? 2 : 3;
                vd[a] = vd[a] || vd[b] || md; dep[g][4 * h + 0] = vd[a];
                vd[d] = vd[d] || vd[a];       dep[g][4 * h + 1] = vd[d];
                vd[c] = vd[c] || vd[d];       dep[g][4 * h + 2] = vd[c];
                vd[b] = vd[b] || vd[c];       dep[g][4 * h + 3] = vd[b];
            }
        }
    }
};
// the snap/pre precomputation for one job (host): rounds 0..1 over the nonce-free message
static inline void plan_snap(Params &p, int W, bool SH) {
    Plan01 &q = p.q;
    Plan P(W, SH); uint64_t v[16];
    for (int i = 0; i < 16; i++) v[i] = p.vi[i];
    for (int g = 0; g < 16; g++) {
        uint64_t &a = v[P.vi[g][0]], &b = v[P.vi[g][1]], &c = v[P.vi[g][2]], &d = v[P.vi[g][3]];
        for (int h = 0; h < 2; h++) {
            const uint64_t mv = p.m[P.mi[g][h]];
            q.pre[2 * g + h] = P.pre[g][h] == 1 ? b + mv : P.pre[g][h] == 2 ? a + mv : a + b;
            a = a + b + mv;                                   q.snap[8 * g + 4 * h + 0] = a;
            d = h ? rotr<16>(d ^ a) : rotr<32>(d ^ a);         q.snap[8 * g + 4 * h + 1] = d;
            c = c + d;                                        q.snap[8 * g + 4 * h + 2] = c;
            b = h ? rotr<63>(b ^ c) : rotr<24>(b ^ c);         q.snap[8 * g + 4 * h + 3] = b;
        }
    }
}
#ifdef __CUDA_ARCH__
template <int W, bool SH> __device__ __forceinline__ void rounds01(uint64_t v[16], const ParamsCore &p, const Plan01 &q, const uint64_t m[16]) {
    constexpr Plan P(W, SH);
#pragma unroll
    for (int i = 0; i < 16; i++) v[i] = p.vi[i];
#pragma unroll
    for (int g = 0; g < 16; g++) {
        uint64_t &a = v[P.vi[g][0]], &b = v[P.vi[g][1]], &c = v[P.vi[g][2]], &d = v[P.vi[g][3]];
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int k = 8 * g + 4 * h;
            const uint64_t mv = m[P.mi[g][h]];
            if (!P.dep[g][4 * h + 0]) a = q.snap[k + 0];
            else if (P.pre[g][h] == 1) a = a + q.pre[2 * g + h];
            else if (P.pre[g][h] == 2) a = b + q.pre[2 * g + h];
            else if (P.pre[g][h] == 3) a = mv + q.pre[2 * g + h];
            else a = a + b + mv;
            if (!P.dep[g][4 * h + 1]) d = q.snap[k + 1]; else d = h ? rotr<16>(d ^ a) : rotr<32>(d ^ a);
            if (!P.dep[g][4 * h + 2]) c = q.snap[k + 2]; else c = c + d;
            if (!P.dep[g][4 * h + 3]) b = q.snap[k + 3]; else b = h ? rotr<63>(b ^ c) : rotr<24>(b ^ c);
        }
    }
}
#endif

// shape (a): X = the 8 nonce bytes as a little-endian word (bswap of en1<<48|en2); returns output word 3
template <int W, bool SH> TXQ uint64_t txid_h3(const ParamsCore &p, uint64_t X, const Plan01 *q = nullptr) {
    (void)q;
    uint64_t m[16], v[16];
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (int i = 0; i < 16; i++) {
        if (i < W) m[i] = p.m[i];
        else if (i == W) m[i] = SH ? (p.m[i] | (X << p.shift)) : X;
        else if (SH && i == W + 1) m[i] = X >> (64 - p.shift);
        else m[i] = 0;   // padding after the nonce
    }
#if TXID_TRIM & 4
    rounds01<W, SH>(v, p, *q, m);
    rounds<2>(v, m);
#else
    round0<W, SH>(v, p, m);
    rounds<1>(v, m);
#endif
    return p.h[3] ^ v[3] ^ v[11];
}

// shape (b): the nonce starts in word 15 of the second-to-last block and ends in word 0 of the final block
TXQ uint64_t txid_h3_straddle(const ParamsCore &p, uint64_t X, const Plan01 *q = nullptr) {
    (void)q;
    uint64_t m[16], v[16], h[8];
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (int i = 0; i < 15; i++) m[i] = p.m[i];
    m[15] = p.m[15] | (X << p.shift);
#if TXID_TRIM & 4
    rounds01<15, false>(v, p, *q, m);
    rounds<2>(v, m);
#else
    round0<15, false>(v, p, m);
    rounds<1>(v, m);
#endif
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (int i = 0; i < 8; i++) h[i] = p.h[i] ^ v[i] ^ v[i + 8];
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (int i = 0; i < 16; i++) m[i] = 0;
    m[0] = X >> (64 - p.shift);
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (int i = 0; i < 8; i++) { v[i] = h[i]; v[i + 8] = iv(i); }
    v[12] ^= p.tfin; v[14] = ~v[14];
    rounds<0>(v, m);
    return h[3] ^ v[3] ^ v[11];
}


// ---- host side: reference hash, per-job precomputation, CPU search ------------------------
static inline uint64_t load64le(const uint8_t *p) { uint64_t w = 0; for (int i = 7; i >= 0; i--) w = (w << 8) | p[i]; return w; }
// one full compression on the CPU (h updated in place)
static inline void compress(uint64_t h[8], const uint64_t m[16], uint64_t t, bool last) {
    uint64_t v[16];
    for (int i = 0; i < 8; i++) { v[i] = h[i]; v[i + 8] = iv(i); }
    v[12] ^= t; if (last) v[14] = ~v[14];
    rounds<0>(v, m);
    for (int i = 0; i < 8; i++) h[i] ^= v[i] ^ v[i + 8];
}
// BLAKE2b-256 of msg (RFC 7693, unkeyed): the reference
static inline void blake2b256(const uint8_t *msg, size_t len, uint8_t out[32]) {
    uint64_t h[8]; for (int i = 0; i < 8; i++) h[i] = iv(i);
    h[0] ^= TXID_PARAM_WORD;
    size_t off = 0; uint64_t m[16]; uint8_t blk[128];
    while (len - off > 128) {
        for (int i = 0; i < 16; i++) m[i] = load64le(msg + off + 8 * i);
        off += 128; compress(h, m, (uint64_t)off, false);
    }
    memset(blk, 0, sizeof blk); if (len > off) memcpy(blk, msg + off, len - off);
    for (int i = 0; i < 16; i++) m[i] = load64le(blk + 8 * i);
    compress(h, m, (uint64_t)len, true);
    for (int i = 0; i < 32; i++) out[i] = (uint8_t)(h[i / 8] >> (8 * (i % 8)));
}
// the work of a hash: trailing zero bits of hash[24..31] read big-endian (64 for all zero)
static inline int work_bits(const uint8_t hash[32]) {
    uint64_t w = 0; for (int i = 24; i < 32; i++) w = (w << 8) | hash[i];
    if (!w) return 64;
    int n = 0; while (!(w & 1)) { w >>= 1; n++; }
    return n;
}
// work from output word 3 (hash bytes 24..31 are its little-endian bytes)
static inline int work_bits_h3(uint64_t h3) { uint64_t w = bswap64h(h3); if (!w) return 64; int n = 0; while (!(w & 1)) { w >>= 1; n++; } return n; }
static inline uint64_t mask_for_k(int k) { if (k <= 0) return 0; uint64_t low = k >= 64 ? ~0ULL : ((1ULL << k) - 1); return bswap64h(low); }
// the nonce (en1 || en2, 8 bytes in message order) as the little-endian word the kernels insert
static inline uint64_t nonce_word(uint64_t n) { return bswap64h(n); }

// the shape of a job's message and the kernel that searches it
struct Shape { int prefix_len = 0, len = 0, nblocks = 0, block = 0, offset = 0, W = 0, shift = 0; bool straddle = false; int kernel = -1; };
static inline Shape shape_of(int prefix_len) {
    Shape s; s.prefix_len = prefix_len; s.len = prefix_len + 8;
    s.nblocks = (s.len + 127) / 128;
    s.block = prefix_len / 128;                 // the block holding the first nonce byte
    s.offset = prefix_len % 128;
    s.straddle = s.block != s.nblocks - 1;      // first nonce byte not in the final block
    s.W = s.offset / 8; s.shift = 8 * (s.offset % 8);
    s.kernel = s.straddle ? 31 : 2 * s.W + (s.shift ? 1 : 0);
    return s;
}
// The per-job precomputation: chaining value before the nonce block and the nonce-free parts of
// its round 0. The nonce bytes are zero in p.m; k sets the share mask.
static inline Shape prepare(const uint8_t *prefix, int prefix_len, int k, Params &p) {
    Shape s = shape_of(prefix_len);
    memset(&p, 0, sizeof p);
    std::vector<uint8_t> msg((size_t)s.nblocks * 128, 0);
    memcpy(msg.data(), prefix, (size_t)prefix_len);
    uint64_t h[8]; for (int i = 0; i < 8; i++) h[i] = iv(i);
    h[0] ^= TXID_PARAM_WORD;
    uint64_t m[16];
    for (int b = 0; b < s.block; b++) {
        for (int i = 0; i < 16; i++) m[i] = load64le(msg.data() + 128 * b + 8 * i);
        compress(h, m, (uint64_t)128 * (b + 1), false);
    }
    for (int i = 0; i < 8; i++) p.h[i] = h[i];
    for (int i = 0; i < 16; i++) p.m[i] = load64le(msg.data() + 128 * s.block + 8 * i);
    bool last = !s.straddle;
    uint64_t t = last ? (uint64_t)s.len : (uint64_t)128 * (s.block + 1);
    uint64_t v[16];
    for (int i = 0; i < 8; i++) { v[i] = h[i]; v[i + 8] = iv(i); }
    v[12] ^= t; if (last) v[14] = ~v[14];
    memcpy(p.vi, v, sizeof v);
    // columns: halves from vi, full from vi
    uint64_t c[16]; memcpy(c, v, sizeof c); memcpy(p.vhc, v, sizeof v);
    for (int j = 0; j < 4; j++) {
        g1(p.vhc[j], p.vhc[4 + j], p.vhc[8 + j], p.vhc[12 + j], p.m[2 * j]);
        gf(c[j], c[4 + j], c[8 + j], c[12 + j], p.m[2 * j], p.m[2 * j + 1]);
    }
    memcpy(p.vc, c, sizeof c); memcpy(p.vhd, c, sizeof c); memcpy(p.vr, c, sizeof c);
    for (int j = 0; j < 4; j++) {
        int a = j, b = 4 + (j + 1) % 4, cc = 8 + (j + 2) % 4, d = 12 + (j + 3) % 4;
        g1(p.vhd[a], p.vhd[b], p.vhd[cc], p.vhd[d], p.m[8 + 2 * j]);
        gf(p.vr[a], p.vr[b], p.vr[cc], p.vr[d], p.m[8 + 2 * j], p.m[9 + 2 * j]);
    }
    p.mask = mask_for_k(k);
    p.tfin = (uint64_t)s.len;
    p.shift = (uint32_t)s.shift;
    p.kernel = (uint32_t)s.kernel;
    plan_snap(p, s.straddle ? 15 : s.W, s.straddle ? false : s.shift != 0);
    return s;
}
// output word 3 for nonce n through the GPU algorithm, on the CPU (tests the precomputation)
template <int ID> static inline uint64_t emu_one(const Params &p, uint64_t X) { return ID == 31 ? txid_h3_straddle(p, X) : txid_h3<ID / 2, (ID & 1) != 0>(p, X); }
template <int ID> static inline uint64_t emu_dispatch(const Params &p, uint64_t X) { if ((int)p.kernel == ID) return emu_one<ID>(p, X); if constexpr (ID < 31) return emu_dispatch<ID + 1>(p, X); return 0; }
static inline uint64_t emulate_h3(const Params &p, uint64_t n) { return emu_dispatch<0>(p, nonce_word(n)); }

// the full message and its hash for a nonce
static inline void hash_nonce(const std::vector<uint8_t> &prefix, uint64_t n, uint8_t out[32]) {
    std::vector<uint8_t> msg(prefix); for (int i = 0; i < 8; i++) msg.push_back((uint8_t)(n >> (56 - 8 * i)));
    blake2b256(msg.data(), msg.size(), out);
}

// ---- the self-check every GPU passes before it mines a hash job ----------------------------
// search(params, first nonce, count) runs the backend's kernel over [first, first+count) and returns
// every hit {nonce, output word 3}. Checks (all exact):
//  1. the upgrade-6 vectors: output word 3 of the given nonce, and the first extranonce2 from 0 with
//     >= 12 zero bits (lengths 120..640, before/at/after block boundaries);
//  2. every kernel shape (prefix lengths 256..383: all 16 word positions, aligned and shifted, and the
//     straddle with every shift): the GPU hit set at k=3 over 1024 nonces equals the CPU reference.
struct HitRec { uint64_t n, h3; };
template <class Search> static inline int selfcheck(Search search, std::string &why) {
    char b[256];
    for (int v = 0; v < NVECTORS; v++) {
        const Vector &x = VECTORS[v];
        std::vector<uint8_t> prefix((size_t)x.prefix_len); for (int i = 0; i < x.prefix_len; i++) prefix[i] = (uint8_t)((i * 7 + 3) & 0xff);
        uint64_t N = (uint64_t)x.en1 << 48 | x.en2;
        uint8_t ref[32]; hash_nonce(prefix, N, ref);
        static const char *hx = "0123456789abcdef"; std::string rh; for (int i = 0; i < 32; i++) { rh += hx[ref[i] >> 4]; rh += hx[ref[i] & 15]; }
        if (rh != x.hash || work_bits(ref) != x.tz) { snprintf(b, sizeof b, "vector %d (length %d): CPU reference hash mismatch", v, x.prefix_len + 8); why = b; return 1; }
        Params p; prepare(prefix.data(), x.prefix_len, 0, p);
        std::vector<HitRec> h = search(p, N, 1);
        if (h.size() != 1 || h[0].n != N || h[0].h3 != load64le(ref + 24)) { snprintf(b, sizeof b, "vector %d (length %d): GPU hash word differs", v, x.prefix_len + 8); why = b; return 1; }
        prepare(prefix.data(), x.prefix_len, 12, p);
        uint64_t base = (uint64_t)x.en1 << 48;
        h = search(p, base, 1u << 16);
        uint64_t first = ~0ULL;
        for (auto &r : h) {
            uint8_t hh[32]; hash_nonce(prefix, r.n, hh);
            if (r.n < base || r.n >= base + (1u << 16) || work_bits(hh) < 12 || load64le(hh + 24) != r.h3) { snprintf(b, sizeof b, "vector %d (length %d): GPU reports a wrong share", v, x.prefix_len + 8); why = b; return 1; }
            first = r.n - base < first ? r.n - base : first;
        }
        if (first != x.first_en2) { snprintf(b, sizeof b, "vector %d (length %d): first extranonce2 with >= 12 zero bits %llx, expected %llx", v, x.prefix_len + 8, (unsigned long long)first, x.first_en2); why = b; return 1; }
    }
    for (int v = 0; v < NNODE_VECTORS; v++) {
        const NodeVector &x = NODE_VECTORS[v];
        std::vector<uint8_t> prefix((size_t)x.prefix_len); for (int i = 0; i < x.prefix_len; i++) prefix[i] = (uint8_t)i;
        const uint64_t N = 0x0102030405060708ULL;
        uint8_t ref[32]; hash_nonce(prefix, N, ref);
        static const char *hx = "0123456789abcdef"; std::string rh; for (int i = 0; i < 32; i++) { rh += hx[ref[i] >> 4]; rh += hx[ref[i] & 15]; }
        if (rh != x.hash) { snprintf(b, sizeof b, "node vector %d (length %d): CPU reference hash mismatch", v, x.prefix_len + 8); why = b; return 1; }
        Params p; prepare(prefix.data(), x.prefix_len, 0, p);
        std::vector<HitRec> h = search(p, N, 1);
        if (h.size() != 1 || h[0].n != N || h[0].h3 != load64le(ref + 24)) { snprintf(b, sizeof b, "node vector %d (length %d): GPU hash word differs", v, x.prefix_len + 8); why = b; return 1; }
    }
    uint64_t seed = 0x9E3779B97F4A7C15ULL;
    for (int plen = 256; plen < 384; plen++) {
        std::vector<uint8_t> prefix((size_t)plen);
        for (auto &c : prefix) { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; c = (uint8_t)seed; }
        Params p; Shape s = prepare(prefix.data(), plen, 3, p);
        uint64_t base = 0xA5C3000000000000ULL | ((seed >> 16) & 0xffff00000000ULL);
        std::vector<HitRec> want;
        for (uint64_t i = 0; i < 1024; i++) { uint8_t hh[32]; hash_nonce(prefix, base + i, hh); if (work_bits(hh) >= 3) want.push_back({base + i, load64le(hh + 24)}); }
        std::vector<HitRec> got = search(p, base, 1024);
        auto lt = [](const HitRec &a, const HitRec &c) { return a.n < c.n; };
        std::sort(got.begin(), got.end(), lt);
        bool same = got.size() == want.size();
        for (size_t i = 0; same && i < got.size(); i++) same = got[i].n == want[i].n && got[i].h3 == want[i].h3;
        if (!same) { snprintf(b, sizeof b, "kernel shape %d (message length %d): %zu GPU hits, %zu expected or different", s.kernel, plen + 8, got.size(), want.size()); why = b; return 1; }
    }
    why = "9 pool + 11 node vectors and all 32 kernel shapes exact";
    return 0;
}
} // namespace txid
