// ProxiChorizo - Copyright (c) 2026 suprnova.cc. ProxiChorizo License 1.0: free to run for mining;
// reuse in commercial or closed-source miners needs written permission. See LICENSE.
//
// txid_blake2b.cl — the OpenCL search kernels for the txid-blake2b work (Proxima library upgrade 6).
// The same algorithm as ../txid_blake2b.h (txid_h3 / txid_h3_straddle), one kernel per message shape:
// txid_k<ID>, ID = 2*W + SH for a nonce in the final block starting in word W (SH: not word-aligned),
// 31 for the nonce straddling the last two blocks. P is the raw txid::Params of the job (107 words):
// m[16] h[8] vi[16] vc[16] vhc[16] vhd[16] vr[16] mask tfin shift|kernel.
#define P_M 0
#define P_H 16
#define P_VI 24
#define P_VC 40
#define P_VHC 56
#define P_VHD 72
#define P_VR 88
#define P_MASK 104
#define P_TFIN 105
#define P_SHIFT 106

#ifdef cl_amd_media_ops
#pragma OPENCL EXTENSION cl_amd_media_ops : enable
#define TX_ROR(x, n) tx_ror((x), (n))
inline ulong tx_ror(ulong x, uint n) {
    uint2 v = as_uint2(x);
    if (n == 32) return as_ulong(v.s10);
    if (n < 32) return as_ulong((uint2)(amd_bitalign(v.s1, v.s0, n), amd_bitalign(v.s0, v.s1, n)));
    return as_ulong((uint2)(amd_bitalign(v.s0, v.s1, n - 32), amd_bitalign(v.s1, v.s0, n - 32)));
}
#else
#define TX_ROR(x, n) rotate((x), (ulong)(64 - (n)))
#endif
#define G1(a, b, c, d, x) { a += b + (x); d = TX_ROR(d ^ a, 32); c += d; b = TX_ROR(b ^ c, 24); }
#define G2(a, b, c, d, y) { a += b + (y); d = TX_ROR(d ^ a, 16); c += d; b = TX_ROR(b ^ c, 63); }
#define GF(a, b, c, d, x, y) { G1(a, b, c, d, x) G2(a, b, c, d, y) }

__constant uchar TX_SIGMA[12][16] = {
    {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},{14,10,4,8,9,15,13,6,1,12,0,2,11,7,5,3},
    {11,8,12,0,5,2,15,13,10,14,3,6,7,1,9,4},{7,9,3,1,13,12,11,14,2,6,5,10,4,0,15,8},{9,0,5,7,2,4,10,15,14,1,11,12,6,8,3,13},
    {2,12,6,10,0,11,8,3,4,13,7,5,15,14,1,9},{12,5,1,15,14,13,4,10,0,7,6,3,9,2,8,11},{13,11,7,14,12,1,3,9,5,0,15,4,8,6,2,10},
    {6,15,14,9,11,3,0,8,12,2,13,7,1,4,10,5},{10,2,8,4,7,6,1,5,15,11,9,14,3,12,13,0},{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},
    {14,10,4,8,9,15,13,6,1,12,0,2,11,7,5,3}};
__constant ulong TX_IV[8] = {0x6a09e667f3bcc908UL, 0xbb67ae8584caa73bUL, 0x3c6ef372fe94f82bUL, 0xa54ff53a5f1d36f1UL,
                             0x510e527fade682d1UL, 0x9b05688c2b3e6c1fUL, 0x1f83d9abfb41bd6bUL, 0x5be0cd19137e2179UL};

#define TX_ROUNDS(FROM) \
    _Pragma("unroll") for (int r = FROM; r < 12; r++) { \
        GF(v[0], v[4], v[8], v[12], m[TX_SIGMA[r][0]], m[TX_SIGMA[r][1]]) \
        GF(v[1], v[5], v[9], v[13], m[TX_SIGMA[r][2]], m[TX_SIGMA[r][3]]) \
        GF(v[2], v[6], v[10], v[14], m[TX_SIGMA[r][4]], m[TX_SIGMA[r][5]]) \
        GF(v[3], v[7], v[11], v[15], m[TX_SIGMA[r][6]], m[TX_SIGMA[r][7]]) \
        GF(v[0], v[5], v[10], v[15], m[TX_SIGMA[r][8]], m[TX_SIGMA[r][9]]) \
        GF(v[1], v[6], v[11], v[12], m[TX_SIGMA[r][10]], m[TX_SIGMA[r][11]]) \
        GF(v[2], v[7], v[8], v[13], m[TX_SIGMA[r][12]], m[TX_SIGMA[r][13]]) \
        GF(v[3], v[4], v[9], v[14], m[TX_SIGMA[r][14]], m[TX_SIGMA[r][15]]) }

#define TX_DEP(W, SH, i) ((i) == (W) || ((SH) && (i) == (W) + 1 && (W) + 1 < 16))
// round 0 from the precomputation: only the Gs that read a nonce word run (W, SH are literals)
#define TX_ROUND0(W, SH) { \
    _Pragma("unroll") for (int j = 0; j < 4; j++) { \
        const int a = j, b = 4 + j, c = 8 + j, d = 12 + j; \
        const bool dx = TX_DEP(W, SH, 2 * j), dy = TX_DEP(W, SH, 2 * j + 1); \
        if (!dx && !dy) { v[a] = P[P_VC + a]; v[b] = P[P_VC + b]; v[c] = P[P_VC + c]; v[d] = P[P_VC + d]; } \
        else if (!dx) { v[a] = P[P_VHC + a]; v[b] = P[P_VHC + b]; v[c] = P[P_VHC + c]; v[d] = P[P_VHC + d]; G2(v[a], v[b], v[c], v[d], m[2 * j + 1]) } \
        else { v[a] = P[P_VI + a]; v[b] = P[P_VI + b]; v[c] = P[P_VI + c]; v[d] = P[P_VI + d]; GF(v[a], v[b], v[c], v[d], m[2 * j], m[2 * j + 1]) } \
    } \
    _Pragma("unroll") for (int j = 0; j < 4; j++) { \
        const int a = j, b = 4 + (j + 1) % 4, c = 8 + (j + 2) % 4, d = 12 + (j + 3) % 4; \
        const bool dx = TX_DEP(W, SH, 8 + 2 * j), dy = TX_DEP(W, SH, 9 + 2 * j); \
        if ((W) < 8) GF(v[a], v[b], v[c], v[d], m[8 + 2 * j], m[9 + 2 * j]) \
        else if (!dx && !dy) { v[a] = P[P_VR + a]; v[b] = P[P_VR + b]; v[c] = P[P_VR + c]; v[d] = P[P_VR + d]; } \
        else if (!dx) { v[a] = P[P_VHD + a]; v[b] = P[P_VHD + b]; v[c] = P[P_VHD + c]; v[d] = P[P_VHD + d]; G2(v[a], v[b], v[c], v[d], m[9 + 2 * j]) } \
        else GF(v[a], v[b], v[c], v[d], m[8 + 2 * j], m[9 + 2 * j]) \
    } }

inline ulong tx_bswap(ulong x) { return as_ulong(as_uchar8(x).s76543210); }

// shape (a): output word 3 for nonce word X
#define TX_H3_A(W, SH, X, out) { \
    ulong m[16], v[16]; const uint sh = (uint)P[P_SHIFT] & 63u; \
    _Pragma("unroll") for (int i = 0; i < 16; i++) { \
        if (i < (W)) m[i] = P[P_M + i]; \
        else if (i == (W)) m[i] = (SH) ? (P[P_M + i] | ((X) << sh)) : (X); \
        else if ((SH) && i == (W) + 1) m[i] = (X) >> (64u - sh); \
        else m[i] = 0; \
    } \
    TX_ROUND0(W, SH) \
    TX_ROUNDS(1) \
    out = P[P_H + 3] ^ v[3] ^ v[11]; }

// shape (b): the nonce straddles the last two blocks
#define TX_H3_B(X, out) { \
    ulong m[16], v[16], h[8]; const uint sh = (uint)P[P_SHIFT] & 63u; \
    _Pragma("unroll") for (int i = 0; i < 15; i++) m[i] = P[P_M + i]; \
    m[15] = P[P_M + 15] | ((X) << sh); \
    TX_ROUND0(15, 0) \
    TX_ROUNDS(1) \
    _Pragma("unroll") for (int i = 0; i < 8; i++) h[i] = P[P_H + i] ^ v[i] ^ v[i + 8]; \
    _Pragma("unroll") for (int i = 0; i < 16; i++) m[i] = 0; \
    m[0] = (X) >> (64u - sh); \
    _Pragma("unroll") for (int i = 0; i < 8; i++) { v[i] = h[i]; v[i + 8] = TX_IV[i]; } \
    v[12] ^= P[P_TFIN]; v[14] = ~v[14]; \
    TX_ROUNDS(0) \
    out = h[3] ^ v[3] ^ v[11]; }

#define TX_KERNEL(ID, BODY) \
__kernel __attribute__((reqd_work_group_size(TX_WG, 1, 1))) void txid_k##ID(__constant ulong *P, ulong nbase, ulong count, uint iters, __global ulong2 *hits, __global uint *nhit) { \
    const ulong gid = get_global_id(0), stride = get_global_size(0), mask = P[P_MASK]; \
    for (uint it = 0; it < iters; it++) { \
        const ulong idx = gid + (ulong)it * stride; \
        if (idx >= count) break; \
        const ulong n = nbase + idx, X = tx_bswap(n); ulong h3; \
        BODY \
        if ((h3 & mask) == 0) { uint s = atomic_inc(nhit); if (s < TX_MAX_HITS) hits[s] = (ulong2)(n, h3); } \
    } }
#ifndef TX_WG
#define TX_WG 256
#endif
#ifndef TX_MAX_HITS
#define TX_MAX_HITS 4096
#endif
#define TXA(ID, W, SH) TX_KERNEL(ID, TX_H3_A(W, SH, X, h3))
TXA(0, 0, 0) TXA(1, 0, 1) TXA(2, 1, 0) TXA(3, 1, 1) TXA(4, 2, 0) TXA(5, 2, 1) TXA(6, 3, 0) TXA(7, 3, 1)
TXA(8, 4, 0) TXA(9, 4, 1) TXA(10, 5, 0) TXA(11, 5, 1) TXA(12, 6, 0) TXA(13, 6, 1) TXA(14, 7, 0) TXA(15, 7, 1)
TXA(16, 8, 0) TXA(17, 8, 1) TXA(18, 9, 0) TXA(19, 9, 1) TXA(20, 10, 0) TXA(21, 10, 1) TXA(22, 11, 0) TXA(23, 11, 1)
TXA(24, 12, 0) TXA(25, 12, 1) TXA(26, 13, 0) TXA(27, 13, 1) TXA(28, 14, 0) TXA(29, 14, 1) TXA(30, 15, 0)
TX_KERNEL(31, TX_H3_B(X, h3))


// The caller proves that the final block is 120 zero bytes followed by the
// nonce. Literal zeros let the OpenCL compiler delete the message additions.
#define TX_H3_ZERO(X, out) { \
    ulong m[16], v[16]; \
    _Pragma("unroll") for (int i = 0; i < 15; ++i) m[i] = 0; \
    m[15] = (X); \
    TX_ROUND0(15, 0) \
    TX_ROUNDS(1) \
    out = P[P_H + 3] ^ v[3] ^ v[11]; }
TX_KERNEL(32, TX_H3_ZERO(X, h3))

#ifdef TX_RDNA3_SPECIALIZED
// For an input split across two BLAKE2b blocks, sh is the bit length of
// the second block. A consecutive interval with the same (nonce >> sh)
// has one common first-block chaining value.
static inline __attribute__((always_inline))
void tx_prepare_straddle(__constant ulong *P, ulong nonce, uint sh, __local ulong *saved) {
    ulong v[16], m[16];
    #pragma unroll
    for (int i=0;i<15;++i) m[i]=P[P_M+i];
    m[15]=P[P_M+15] | (tx_bswap(nonce)<<sh);
    TX_ROUND0(15,0)
    TX_ROUNDS(1)
    ulong h[8];
    #pragma unroll
    for (int i=0;i<8;++i) h[i]=P[P_H+i]^v[i]^v[i+8];
    #pragma unroll
    for (int i=0;i<8;++i) { v[i]=h[i]; v[i+8]=TX_IV[i]; }
    v[12]^=P[P_TFIN]; v[14]=~v[14];
    // Only column zero of the final block depends on its nonce bytes.
    GF(v[1],v[5],v[9],v[13],0UL,0UL)
    GF(v[2],v[6],v[10],v[14],0UL,0UL)
    GF(v[3],v[7],v[11],v[15],0UL,0UL)
    #pragma unroll
    for (int i=0;i<16;++i) saved[i]=v[i];
    saved[16]=h[3];
}
static inline __attribute__((always_inline))
ulong tx_finish_straddle(__local ulong *saved, ulong suffix) {
    ulong v[16], m[16] = {0}; m[0]=suffix;
    #pragma unroll
    for(int i=0;i<16;++i) v[i]=saved[i];
    GF(v[0],v[4],v[8],v[12],suffix,0UL)
    GF(v[0],v[5],v[10],v[15],0UL,0UL)
    GF(v[1],v[6],v[11],v[12],0UL,0UL)
    GF(v[2],v[7],v[8],v[13],0UL,0UL)
    GF(v[3],v[4],v[9],v[14],0UL,0UL)
    TX_ROUNDS(1)
    return saved[16]^v[3]^v[11];
}
// One group prepares contiguous 256-nonce chunks. Sixteen uses 2,176 B of LDS.
#define TX_STRADDLE_CHUNKS 16
__kernel __attribute__((reqd_work_group_size(TX_WG,1,1)))
void txid_straddle(__constant ulong *P, ulong nbase, ulong count, uint iters,
                   __global ulong2 *hits, __global uint *nhit) {
    if (!count) return;
    const uint lid=get_local_id(0);
    const uint sh=(uint)P[P_SHIFT]&63;
    const ulong suffix_mask=(1UL<<sh)-1;
    const ulong aligned=nbase & ~((ulong)TX_WG-1);
    const ulong skip=nbase-aligned;
    const ulong nchunks=1+(count-1)/TX_WG + ((count-1)%TX_WG+skip>=TX_WG);
    __local ulong snapshots[TX_STRADDLE_CHUNKS][17];
    for (ulong chunk=(ulong)get_group_id(0)*TX_STRADDLE_CHUNKS; chunk<nchunks;
         chunk+=(ulong)get_num_groups(0)*TX_STRADDLE_CHUNKS) {
        const uint used=(uint)min((ulong)TX_STRADDLE_CHUNKS,nchunks-chunk);
        const ulong start=aligned+chunk*TX_WG;
        const ulong initial_offset=start & suffix_mask;
        const uint last_key=(uint)((initial_offset+(ulong)used*TX_WG-1)>>sh);
        if (lid<=last_key)
            tx_prepare_straddle(P,(start & ~suffix_mask)+((ulong)lid<<sh),sh,snapshots[lid]);
        barrier(CLK_LOCAL_MEM_FENCE);
        for(uint j=0;j<used;++j) {
            const ulong advance=(ulong)j*TX_WG;
            const ulong n=start+advance+lid;
            if (n-nbase>=count) continue;
            // The aligned chunk spans 256 nonces. Its low eight bits
            // cannot carry into the key on any of the seven split shapes.
            const uint key=(uint)((initial_offset+advance)>>sh);
            const ulong h=tx_finish_straddle(snapshots[key],tx_bswap(n)>>(64-sh));
            if ((h & P[P_MASK])==0) {
                uint slot=atomic_inc(nhit);
                if(slot<TX_MAX_HITS) hits[slot]=(ulong2)(n,h);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

// Run one elementary BLAKE2b step across the four independent columns
// before the next dependent step. All message words except word 15 are zero.
#define TX_ALIGNED_FOUR(H,EXPR) { \
    _Pragma("unroll") for (uint lane=0;lane<4;++lane) { \
        const uint a=lane,b=4+(lane+(H))%4,c=8+(lane+2*(H))%4,d=12+(lane+3*(H))%4; \
        EXPR; \
    } \
}
#define TX_ALIGNED_HALF(H) { \
    TX_ALIGNED_FOUR(H,v[a]+=v[b]+(TX_SIGMA[r][8*(H)+2*lane]==15?X:0UL)) \
    TX_ALIGNED_FOUR(H,v[d]=TX_ROR(v[d]^v[a],32)) \
    TX_ALIGNED_FOUR(H,v[c]+=v[d]) \
    TX_ALIGNED_FOUR(H,v[b]=TX_ROR(v[b]^v[c],24)) \
    TX_ALIGNED_FOUR(H,v[a]+=v[b]+(TX_SIGMA[r][8*(H)+2*lane+1]==15?X:0UL)) \
    TX_ALIGNED_FOUR(H,v[d]=TX_ROR(v[d]^v[a],16)) \
    TX_ALIGNED_FOUR(H,v[c]+=v[d]) \
    TX_ALIGNED_FOUR(H,v[b]=TX_ROR(v[b]^v[c],63)) \
}
static inline __attribute__((always_inline))
ulong tx_sched_h3(__constant ulong *P,ulong X) {
    ulong v[16],m[16]={0};m[15]=X;
    TX_ROUND0(15,0)
    #pragma unroll
    for(uint r=1;r<12;++r) { TX_ALIGNED_HALF(0) TX_ALIGNED_HALF(1) }
    return P[P_H+3]^v[3]^v[11];
}
TX_KERNEL(33,h3=tx_sched_h3(P,X);)
#endif
