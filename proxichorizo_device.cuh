// ProxiChorizo - Copyright (c) 2026 suprnova.cc. ProxiChorizo License 1.0: free to run for mining;
// reuse in commercial or closed-source miners needs written permission. See LICENSE.
// proxichorizo_device.cuh — the device side of ProxiChorizo: field arithmetic, Ed25519 points,
// SHA-512, the VRF attempt and the kernels. Included once per variant from
// ProxiChorizo.cu inside a namespace, with PX_FE32 / PX_WINDOW / PX_LB_BLOCKS /
// PX_VARIANT_NAME set by the includer; see PLATFORMS.md for what each costs.
#if defined(PX_FE26)
#define fe_sq_pow fe_sq
// ----------------------------------------------------------------------------
// field element, 10 signed 32-bit limbs in radix 2^25.5 (26, 25, 26, ... bits),
// the ref10 layout: products are accumulated in 64-bit columns, one IMAD.WIDE
// each and independent of one another, so there is no carry chain for the
// scheduler to wait on (the 8x32 schoolbook below serialises on the carry
// flag). Columns past limb 9 are kept apart and folded once with 2^255 = 19,
// so no operand is ever scaled by more than 4 and lazily reduced inputs with
// limbs up to ~2^27 stay within int64. Measured 1.4x the multiply rate of the
// 8x32 version on Blackwell (PLATFORMS.md, round 6).
// ----------------------------------------------------------------------------
struct fe { int32_t v[10]; };

__device__ __forceinline__ void fe_0(fe &h) { for (int i = 0; i < 10; i++) h.v[i] = 0; }
__device__ __forceinline__ void fe_1(fe &h) { fe_0(h); h.v[0] = 1; }
__device__ __forceinline__ void fe_copy(fe &h, const fe &f) { for (int i = 0; i < 10; i++) h.v[i] = f.v[i]; }
__device__ __forceinline__ void fe_add(fe &h, const fe &f, const fe &g) { for (int i = 0; i < 10; i++) h.v[i] = f.v[i] + g.v[i]; }
__device__ __forceinline__ void fe_sub(fe &h, const fe &f, const fe &g) { for (int i = 0; i < 10; i++) h.v[i] = f.v[i] - g.v[i]; }
__device__ __forceinline__ void fe_neg(fe &h, const fe &f) { for (int i = 0; i < 10; i++) h.v[i] = -f.v[i]; }

// the ref10 carry pass over 64-bit columns: limbs back to 26/25 bits, signed
__device__ __forceinline__ void fe_carry64(fe &h, int64_t t[10]) {
    int64_t c;
    c = (t[0] + (1 << 25)) >> 26; t[1] += c; t[0] -= c << 26;
    c = (t[4] + (1 << 25)) >> 26; t[5] += c; t[4] -= c << 26;
    c = (t[1] + (1 << 24)) >> 25; t[2] += c; t[1] -= c << 25;
    c = (t[5] + (1 << 24)) >> 25; t[6] += c; t[5] -= c << 25;
    c = (t[2] + (1 << 25)) >> 26; t[3] += c; t[2] -= c << 26;
    c = (t[6] + (1 << 25)) >> 26; t[7] += c; t[6] -= c << 26;
    c = (t[3] + (1 << 24)) >> 25; t[4] += c; t[3] -= c << 25;
    c = (t[7] + (1 << 24)) >> 25; t[8] += c; t[7] -= c << 25;
    c = (t[4] + (1 << 25)) >> 26; t[5] += c; t[4] -= c << 26;
    c = (t[8] + (1 << 25)) >> 26; t[9] += c; t[8] -= c << 26;
    c = (t[9] + (1 << 24)) >> 25; t[0] += c * 19; t[9] -= c << 25;
    c = (t[0] + (1 << 25)) >> 26; t[1] += c; t[0] -= c << 26;
    for (int i = 0; i < 10; i++) h.v[i] = (int32_t)t[i];
}
__device__ __forceinline__ void fe_reduce(fe &h) { int64_t t[10]; for (int i = 0; i < 10; i++) t[i] = h.v[i]; fe_carry64(h, t); }

// h = f * g: 100 products into 64-bit columns (odd x odd limbs weigh one bit
// more, hence x2), the wrapped columns folded with 19 at the end
__device__ void fe_mul(fe &h, const fe &f, const fe &g) {
    int64_t t[10], w[10];
    #pragma unroll
    for (int k = 0; k < 10; k++) { t[k] = 0; w[k] = 0; }
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        const int32_t fi = f.v[i], fi2 = fi * 2;
        #pragma unroll
        for (int j = 0; j < 10; j++) {
            const int32_t a = ((i & 1) && (j & 1)) ? fi2 : fi;
            const int64_t p = (int64_t)a * g.v[j];
            if (i + j < 10) t[i + j] += p; else w[i + j - 10] += p;
        }
    }
    #pragma unroll
    for (int k = 0; k < 10; k++) t[k] += w[k] * 19;
    fe_carry64(h, t);
}
// h = f^2: the 55 distinct products, cross terms doubled
__device__ void fe_sq(fe &h, const fe &f) {
    int64_t t[10], w[10];
    #pragma unroll
    for (int k = 0; k < 10; k++) { t[k] = 0; w[k] = 0; }
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        #pragma unroll
        for (int j = i; j < 10; j++) {
            const int32_t m = (i == j ? 1 : 2) * (((i & 1) && (j & 1)) ? 2 : 1);
            const int64_t p = (int64_t)(f.v[i] * m) * f.v[j];
            if (i + j < 10) t[i + j] += p; else w[i + j - 10] += p;
        }
    }
    #pragma unroll
    for (int k = 0; k < 10; k++) t[k] += w[k] * 19;
    fe_carry64(h, t);
}
__device__ __forceinline__ void fe_pow2k(fe &h, const fe &f, int k) { fe_copy(h, f); for (int i = 0; i < k; i++) fe_sq_pow(h, h); }

// canonical little-endian bytes (ref10 fe_tobytes after a carry pass)
__device__ void fe_tobytes(uint8_t s[32], const fe &f) {
    fe r; fe_copy(r, f); fe_reduce(r);
    int64_t h[10];
    for (int i = 0; i < 10; i++) h[i] = r.v[i];
    int64_t q = (19 * h[9] + (1 << 24)) >> 25;
    q = (h[0] + q) >> 26; q = (h[1] + q) >> 25; q = (h[2] + q) >> 26; q = (h[3] + q) >> 25; q = (h[4] + q) >> 26;
    q = (h[5] + q) >> 25; q = (h[6] + q) >> 26; q = (h[7] + q) >> 25; q = (h[8] + q) >> 26; q = (h[9] + q) >> 25;
    h[0] += 19 * q;
    int64_t c;
    c = h[0] >> 26; h[1] += c; h[0] -= c << 26;
    c = h[1] >> 25; h[2] += c; h[1] -= c << 25;
    c = h[2] >> 26; h[3] += c; h[2] -= c << 26;
    c = h[3] >> 25; h[4] += c; h[3] -= c << 25;
    c = h[4] >> 26; h[5] += c; h[4] -= c << 26;
    c = h[5] >> 25; h[6] += c; h[5] -= c << 25;
    c = h[6] >> 26; h[7] += c; h[6] -= c << 26;
    c = h[7] >> 25; h[8] += c; h[7] -= c << 25;
    c = h[8] >> 26; h[9] += c; h[8] -= c << 26;
    c = h[9] >> 25; h[9] -= c << 25;
    const uint32_t h0 = (uint32_t)h[0], h1 = (uint32_t)h[1], h2 = (uint32_t)h[2], h3 = (uint32_t)h[3], h4 = (uint32_t)h[4];
    const uint32_t h5 = (uint32_t)h[5], h6 = (uint32_t)h[6], h7 = (uint32_t)h[7], h8 = (uint32_t)h[8], h9 = (uint32_t)h[9];
    s[0] = h0; s[1] = h0 >> 8; s[2] = h0 >> 16; s[3] = (h0 >> 24) | (h1 << 2);
    s[4] = h1 >> 6; s[5] = h1 >> 14; s[6] = (h1 >> 22) | (h2 << 3);
    s[7] = h2 >> 5; s[8] = h2 >> 13; s[9] = (h2 >> 21) | (h3 << 5);
    s[10] = h3 >> 3; s[11] = h3 >> 11; s[12] = (h3 >> 19) | (h4 << 6);
    s[13] = h4 >> 2; s[14] = h4 >> 10; s[15] = h4 >> 18;
    s[16] = h5; s[17] = h5 >> 8; s[18] = h5 >> 16; s[19] = (h5 >> 24) | (h6 << 1);
    s[20] = h6 >> 7; s[21] = h6 >> 15; s[22] = (h6 >> 23) | (h7 << 3);
    s[23] = h7 >> 5; s[24] = h7 >> 13; s[25] = (h7 >> 21) | (h8 << 4);
    s[26] = h8 >> 4; s[27] = h8 >> 12; s[28] = (h8 >> 20) | (h9 << 6);
    s[29] = h9 >> 2; s[30] = h9 >> 10; s[31] = h9 >> 18;
}
__device__ __forceinline__ int64_t fe26_load3(const uint8_t *p) { return (int64_t)p[0] | ((int64_t)p[1] << 8) | ((int64_t)p[2] << 16); }
__device__ __forceinline__ int64_t fe26_load4(const uint8_t *p) { return fe26_load3(p) | ((int64_t)p[3] << 24); }
__device__ void fe_frombytes(fe &h, const uint8_t s[32]) {
    int64_t t[10];
    t[0] = fe26_load4(s); t[1] = fe26_load3(s + 4) << 6; t[2] = fe26_load3(s + 7) << 5; t[3] = fe26_load3(s + 10) << 3; t[4] = fe26_load3(s + 13) << 2;
    t[5] = fe26_load4(s + 16); t[6] = fe26_load3(s + 20) << 7; t[7] = fe26_load3(s + 23) << 5; t[8] = fe26_load3(s + 26) << 4; t[9] = (fe26_load3(s + 29) & 8388607) << 2;
    int64_t c;
    c = (t[9] + (1 << 24)) >> 25; t[0] += c * 19; t[9] -= c << 25;
    c = (t[1] + (1 << 24)) >> 25; t[2] += c; t[1] -= c << 25;
    c = (t[3] + (1 << 24)) >> 25; t[4] += c; t[3] -= c << 25;
    c = (t[5] + (1 << 24)) >> 25; t[6] += c; t[5] -= c << 25;
    c = (t[7] + (1 << 24)) >> 25; t[8] += c; t[7] -= c << 25;
    c = (t[0] + (1 << 25)) >> 26; t[1] += c; t[0] -= c << 26;
    c = (t[2] + (1 << 25)) >> 26; t[3] += c; t[2] -= c << 26;
    c = (t[4] + (1 << 25)) >> 26; t[5] += c; t[4] -= c << 26;
    c = (t[6] + (1 << 25)) >> 26; t[7] += c; t[6] -= c << 26;
    c = (t[8] + (1 << 25)) >> 26; t[9] += c; t[8] -= c << 26;
    for (int i = 0; i < 10; i++) h.v[i] = (int32_t)t[i];
}
__device__ __forceinline__ bool fe_isnegative(const fe &f) { uint8_t s[32]; fe_tobytes(s, f); return s[0] & 1; }
__device__ __forceinline__ bool fe_iszero(const fe &f) { uint8_t s[32]; fe_tobytes(s, f); uint8_t o = 0; for (int i = 0; i < 32; i++) o |= s[i]; return o == 0; }
__device__ __forceinline__ bool fe_eq(const fe &f, const fe &g) { fe t; fe_sub(t, f, g); return fe_iszero(t); }
#elif defined(PX_FE32)
// ----------------------------------------------------------------------------
// field element, 8 x 32-bit limbs (little-endian), lazily reduced: a value is
// any 256-bit representative of its residue, and only tobytes / eq /
// isnegative reduce fully. Products are schoolbook rows over PTX carry chains
// (mad.lo.cc / madc.hi.cc), the fast path on NVIDIA, where a 64-bit multiply
// is emulated. Folding uses 2^256 = 38 (mod p).
// ----------------------------------------------------------------------------
struct fe { uint32_t v[8]; };

__device__ __forceinline__ void fe_0(fe &h) { for (int i = 0; i < 8; i++) h.v[i] = 0; }
__device__ __forceinline__ void fe_1(fe &h) { fe_0(h); h.v[0] = 1; }
__device__ __forceinline__ void fe_copy(fe &h, const fe &f) { for (int i = 0; i < 8; i++) h.v[i] = f.v[i]; }
__device__ __forceinline__ void fe_reduce(fe &h) { (void)h; }

// h += a with the carry propagated through all limbs; returns the carry out
__device__ __forceinline__ uint32_t fe_addsmall(fe &h, uint32_t a) {
    uint32_t c;
    asm(
        "add.cc.u32 %0, %0, %9;\n\t"
        "addc.cc.u32 %1, %1, 0;\n\t" "addc.cc.u32 %2, %2, 0;\n\t" "addc.cc.u32 %3, %3, 0;\n\t"
        "addc.cc.u32 %4, %4, 0;\n\t" "addc.cc.u32 %5, %5, 0;\n\t" "addc.cc.u32 %6, %6, 0;\n\t"
        "addc.cc.u32 %7, %7, 0;\n\t" "addc.u32 %8, 0, 0;"
        : "+r"(h.v[0]), "+r"(h.v[1]), "+r"(h.v[2]), "+r"(h.v[3]), "+r"(h.v[4]), "+r"(h.v[5]), "+r"(h.v[6]), "+r"(h.v[7]), "=&r"(c)
        : "r"(a));
    return c;
}
// h -= a with the borrow propagated; returns 1 on borrow out
__device__ __forceinline__ uint32_t fe_subsmall(fe &h, uint32_t a) {
    uint32_t b;
    asm(
        "sub.cc.u32 %0, %0, %9;\n\t"
        "subc.cc.u32 %1, %1, 0;\n\t" "subc.cc.u32 %2, %2, 0;\n\t" "subc.cc.u32 %3, %3, 0;\n\t"
        "subc.cc.u32 %4, %4, 0;\n\t" "subc.cc.u32 %5, %5, 0;\n\t" "subc.cc.u32 %6, %6, 0;\n\t"
        "subc.cc.u32 %7, %7, 0;\n\t" "subc.u32 %8, 0, 0;"
        : "+r"(h.v[0]), "+r"(h.v[1]), "+r"(h.v[2]), "+r"(h.v[3]), "+r"(h.v[4]), "+r"(h.v[5]), "+r"(h.v[6]), "+r"(h.v[7]), "=&r"(b)
        : "r"(a));
    return b & 1u;
}

// carry (borrow) out of limb 1 propagated upward, with a wrap past 2^256 folded as 38: the fold of c*38 into
// limb 0 overflows limb 1 only when limb 0 is within 38 of 2^32, so callers branch here about once in 10^8
__device__ __forceinline__ void fe_inc2(fe &h) {
    uint32_t c;
    asm("add.cc.u32 %0, %0, 1;\n\taddc.cc.u32 %1, %1, 0;\n\taddc.cc.u32 %2, %2, 0;\n\taddc.cc.u32 %3, %3, 0;\n\taddc.cc.u32 %4, %4, 0;\n\taddc.cc.u32 %5, %5, 0;\n\taddc.u32 %6, 0, 0;"
        : "+r"(h.v[2]), "+r"(h.v[3]), "+r"(h.v[4]), "+r"(h.v[5]), "+r"(h.v[6]), "+r"(h.v[7]), "=&r"(c));
    if (c) fe_addsmall(h, 38u);
}
__device__ __forceinline__ void fe_dec2(fe &h) {
    uint32_t b;
    asm("sub.cc.u32 %0, %0, 1;\n\tsubc.cc.u32 %1, %1, 0;\n\tsubc.cc.u32 %2, %2, 0;\n\tsubc.cc.u32 %3, %3, 0;\n\tsubc.cc.u32 %4, %4, 0;\n\tsubc.cc.u32 %5, %5, 0;\n\tsubc.u32 %6, 0, 0;"
        : "+r"(h.v[2]), "+r"(h.v[3]), "+r"(h.v[4]), "+r"(h.v[5]), "+r"(h.v[6]), "+r"(h.v[7]), "=&r"(b));
    if (b & 1u) fe_subsmall(h, 38u);
}
// h = f + g
__device__ __forceinline__ void fe_add(fe &h, const fe &f, const fe &g) {
    uint32_t c;
    asm(
        "add.cc.u32 %0, %9, %17;\n\t"
        "addc.cc.u32 %1, %10, %18;\n\t" "addc.cc.u32 %2, %11, %19;\n\t" "addc.cc.u32 %3, %12, %20;\n\t"
        "addc.cc.u32 %4, %13, %21;\n\t" "addc.cc.u32 %5, %14, %22;\n\t" "addc.cc.u32 %6, %15, %23;\n\t"
        "addc.cc.u32 %7, %16, %24;\n\t" "addc.u32 %8, 0, 0;"
        : "=&r"(h.v[0]), "=&r"(h.v[1]), "=&r"(h.v[2]), "=&r"(h.v[3]), "=&r"(h.v[4]), "=&r"(h.v[5]), "=&r"(h.v[6]), "=&r"(h.v[7]), "=&r"(c)
        : "r"(f.v[0]), "r"(f.v[1]), "r"(f.v[2]), "r"(f.v[3]), "r"(f.v[4]), "r"(f.v[5]), "r"(f.v[6]), "r"(f.v[7]),
          "r"(g.v[0]), "r"(g.v[1]), "r"(g.v[2]), "r"(g.v[3]), "r"(g.v[4]), "r"(g.v[5]), "r"(g.v[6]), "r"(g.v[7]));
    // the carry out is set in about half the lanes, so a branch around the
    // fix-up diverges nearly every time and the body runs anyway: fold c*38 in
    // unconditionally (H100 +7 %, RTX PRO 6000 +3.4 %, RTX A6000 +3.5 %)
#ifdef PX_COMBA
    // the carry out of limb 1 is rarer than one in 10^8: propagate it off the hot path
    { uint32_t c2;
      asm("mad.lo.cc.u32 %0, %3, 38, %0;\n\taddc.cc.u32 %1, %1, 0;\n\taddc.u32 %2, 0, 0;" : "+r"(h.v[0]), "+r"(h.v[1]), "=&r"(c2) : "r"(c));
#if !PX_NOBRANCH
      if (c2) fe_inc2(h);
#else
      (void)c2;
#endif
      }
#else
    { uint32_t k = 38u, c2;
      asm("mad.lo.cc.u32 %0, %9, %10, %0;\n\t"
          "addc.cc.u32 %1, %1, 0;\n\t" "addc.cc.u32 %2, %2, 0;\n\t" "addc.cc.u32 %3, %3, 0;\n\t"
          "addc.cc.u32 %4, %4, 0;\n\t" "addc.cc.u32 %5, %5, 0;\n\t" "addc.cc.u32 %6, %6, 0;\n\t"
          "addc.cc.u32 %7, %7, 0;\n\t" "addc.u32 %8, 0, 0;"
          : "+r"(h.v[0]), "+r"(h.v[1]), "+r"(h.v[2]), "+r"(h.v[3]), "+r"(h.v[4]), "+r"(h.v[5]), "+r"(h.v[6]), "+r"(h.v[7]), "=&r"(c2)
          : "r"(c), "r"(k));
      if (c2) fe_addsmall(h, 38u); }
#endif
}
// h = f - g
__device__ __forceinline__ void fe_sub(fe &h, const fe &f, const fe &g) {
    uint32_t b;
    asm(
        "sub.cc.u32 %0, %9, %17;\n\t"
        "subc.cc.u32 %1, %10, %18;\n\t" "subc.cc.u32 %2, %11, %19;\n\t" "subc.cc.u32 %3, %12, %20;\n\t"
        "subc.cc.u32 %4, %13, %21;\n\t" "subc.cc.u32 %5, %14, %22;\n\t" "subc.cc.u32 %6, %15, %23;\n\t"
        "subc.cc.u32 %7, %16, %24;\n\t" "subc.u32 %8, 0, 0;"
        : "=&r"(h.v[0]), "=&r"(h.v[1]), "=&r"(h.v[2]), "=&r"(h.v[3]), "=&r"(h.v[4]), "=&r"(h.v[5]), "=&r"(h.v[6]), "=&r"(h.v[7]), "=&r"(b)
        : "r"(f.v[0]), "r"(f.v[1]), "r"(f.v[2]), "r"(f.v[3]), "r"(f.v[4]), "r"(f.v[5]), "r"(f.v[6]), "r"(f.v[7]),
          "r"(g.v[0]), "r"(g.v[1]), "r"(g.v[2]), "r"(g.v[3]), "r"(g.v[4]), "r"(g.v[5]), "r"(g.v[6]), "r"(g.v[7]));
    // a wrap past zero added 2^256 = 38 too much; subtract it unconditionally
    // (same reasoning as in fe_add: the borrow is set in about half the lanes)
#ifdef PX_COMBA
    { uint32_t t = (b & 1u) * 38u, b2;
      asm("sub.cc.u32 %0, %0, %3;\n\tsubc.cc.u32 %1, %1, 0;\n\tsubc.u32 %2, 0, 0;" : "+r"(h.v[0]), "+r"(h.v[1]), "=&r"(b2) : "r"(t));
#if !PX_NOBRANCH
      if (b2 & 1u) fe_dec2(h);
#else
      (void)b2;
#endif
      }
#else
    { uint32_t t = (b & 1u) * 38u, b2;
      asm("sub.cc.u32 %0, %0, %9;\n\t"
          "subc.cc.u32 %1, %1, 0;\n\t" "subc.cc.u32 %2, %2, 0;\n\t" "subc.cc.u32 %3, %3, 0;\n\t"
          "subc.cc.u32 %4, %4, 0;\n\t" "subc.cc.u32 %5, %5, 0;\n\t" "subc.cc.u32 %6, %6, 0;\n\t"
          "subc.cc.u32 %7, %7, 0;\n\t" "subc.u32 %8, 0, 0;"
          : "+r"(h.v[0]), "+r"(h.v[1]), "+r"(h.v[2]), "+r"(h.v[3]), "+r"(h.v[4]), "+r"(h.v[5]), "+r"(h.v[6]), "+r"(h.v[7]), "=&r"(b2)
          : "r"(t));
      if (b2 & 1u) fe_subsmall(h, 38u); }
#endif
}
__device__ __forceinline__ void fe_neg(fe &h, const fe &f) { fe z; fe_0(z); fe_sub(h, z, f); }

// R[0..9] += a * b[0..7], one schoolbook row: 8 low halves, 8 high halves, two carries
#define PX_MUL_ROW(R, a, b) asm( \
    "mad.lo.cc.u32 %0, %10, %11, %0;\n\t" \
    "madc.lo.cc.u32 %1, %10, %12, %1;\n\t" "madc.lo.cc.u32 %2, %10, %13, %2;\n\t" \
    "madc.lo.cc.u32 %3, %10, %14, %3;\n\t" "madc.lo.cc.u32 %4, %10, %15, %4;\n\t" \
    "madc.lo.cc.u32 %5, %10, %16, %5;\n\t" "madc.lo.cc.u32 %6, %10, %17, %6;\n\t" \
    "madc.lo.cc.u32 %7, %10, %18, %7;\n\t" "addc.u32 %8, %8, 0;\n\t" \
    "mad.hi.cc.u32 %1, %10, %11, %1;\n\t" \
    "madc.hi.cc.u32 %2, %10, %12, %2;\n\t" "madc.hi.cc.u32 %3, %10, %13, %3;\n\t" \
    "madc.hi.cc.u32 %4, %10, %14, %4;\n\t" "madc.hi.cc.u32 %5, %10, %15, %5;\n\t" \
    "madc.hi.cc.u32 %6, %10, %16, %6;\n\t" "madc.hi.cc.u32 %7, %10, %17, %7;\n\t" \
    "madc.hi.cc.u32 %8, %10, %18, %8;\n\t" "addc.u32 %9, %9, 0;" \
    : "+r"((R)[0]), "+r"((R)[1]), "+r"((R)[2]), "+r"((R)[3]), "+r"((R)[4]), "+r"((R)[5]), "+r"((R)[6]), "+r"((R)[7]), "+r"((R)[8]), "+r"((R)[9]) \
    : "r"(a), "r"((b)[0]), "r"((b)[1]), "r"((b)[2]), "r"((b)[3]), "r"((b)[4]), "r"((b)[5]), "r"((b)[6]), "r"((b)[7]))

#define PX_ROW1(R, a, b) asm( \
    "mad.lo.cc.u32 %0, %3, %4, %0;\n\t" \
    "addc.u32 %1, %1, 0;\n\t" \
    "mad.hi.cc.u32 %1, %3, %4, %1;\n\t" \
    "addc.u32 %2, %2, 0;" \
    : "+r"((R)[0]), "+r"((R)[1]), "+r"((R)[2]) \
    : "r"(a), "r"((b)[0]))
#define PX_ROW2(R, a, b) asm( \
    "mad.lo.cc.u32 %0, %4, %5, %0;\n\t" \
    "madc.lo.cc.u32 %1, %4, %6, %1;\n\t" \
    "addc.u32 %2, %2, 0;\n\t" \
    "mad.hi.cc.u32 %1, %4, %5, %1;\n\t" \
    "madc.hi.cc.u32 %2, %4, %6, %2;\n\t" \
    "addc.u32 %3, %3, 0;" \
    : "+r"((R)[0]), "+r"((R)[1]), "+r"((R)[2]), "+r"((R)[3]) \
    : "r"(a), "r"((b)[0]), "r"((b)[1]))
#define PX_ROW3(R, a, b) asm( \
    "mad.lo.cc.u32 %0, %5, %6, %0;\n\t" \
    "madc.lo.cc.u32 %1, %5, %7, %1;\n\t" \
    "madc.lo.cc.u32 %2, %5, %8, %2;\n\t" \
    "addc.u32 %3, %3, 0;\n\t" \
    "mad.hi.cc.u32 %1, %5, %6, %1;\n\t" \
    "madc.hi.cc.u32 %2, %5, %7, %2;\n\t" \
    "madc.hi.cc.u32 %3, %5, %8, %3;\n\t" \
    "addc.u32 %4, %4, 0;" \
    : "+r"((R)[0]), "+r"((R)[1]), "+r"((R)[2]), "+r"((R)[3]), "+r"((R)[4]) \
    : "r"(a), "r"((b)[0]), "r"((b)[1]), "r"((b)[2]))
#define PX_ROW4(R, a, b) asm( \
    "mad.lo.cc.u32 %0, %6, %7, %0;\n\t" \
    "madc.lo.cc.u32 %1, %6, %8, %1;\n\t" \
    "madc.lo.cc.u32 %2, %6, %9, %2;\n\t" \
    "madc.lo.cc.u32 %3, %6, %10, %3;\n\t" \
    "addc.u32 %4, %4, 0;\n\t" \
    "mad.hi.cc.u32 %1, %6, %7, %1;\n\t" \
    "madc.hi.cc.u32 %2, %6, %8, %2;\n\t" \
    "madc.hi.cc.u32 %3, %6, %9, %3;\n\t" \
    "madc.hi.cc.u32 %4, %6, %10, %4;\n\t" \
    "addc.u32 %5, %5, 0;" \
    : "+r"((R)[0]), "+r"((R)[1]), "+r"((R)[2]), "+r"((R)[3]), "+r"((R)[4]), "+r"((R)[5]) \
    : "r"(a), "r"((b)[0]), "r"((b)[1]), "r"((b)[2]), "r"((b)[3]))
#define PX_ROW5(R, a, b) asm( \
    "mad.lo.cc.u32 %0, %7, %8, %0;\n\t" \
    "madc.lo.cc.u32 %1, %7, %9, %1;\n\t" \
    "madc.lo.cc.u32 %2, %7, %10, %2;\n\t" \
    "madc.lo.cc.u32 %3, %7, %11, %3;\n\t" \
    "madc.lo.cc.u32 %4, %7, %12, %4;\n\t" \
    "addc.u32 %5, %5, 0;\n\t" \
    "mad.hi.cc.u32 %1, %7, %8, %1;\n\t" \
    "madc.hi.cc.u32 %2, %7, %9, %2;\n\t" \
    "madc.hi.cc.u32 %3, %7, %10, %3;\n\t" \
    "madc.hi.cc.u32 %4, %7, %11, %4;\n\t" \
    "madc.hi.cc.u32 %5, %7, %12, %5;\n\t" \
    "addc.u32 %6, %6, 0;" \
    : "+r"((R)[0]), "+r"((R)[1]), "+r"((R)[2]), "+r"((R)[3]), "+r"((R)[4]), "+r"((R)[5]), "+r"((R)[6]) \
    : "r"(a), "r"((b)[0]), "r"((b)[1]), "r"((b)[2]), "r"((b)[3]), "r"((b)[4]))
#define PX_ROW6(R, a, b) asm( \
    "mad.lo.cc.u32 %0, %8, %9, %0;\n\t" \
    "madc.lo.cc.u32 %1, %8, %10, %1;\n\t" \
    "madc.lo.cc.u32 %2, %8, %11, %2;\n\t" \
    "madc.lo.cc.u32 %3, %8, %12, %3;\n\t" \
    "madc.lo.cc.u32 %4, %8, %13, %4;\n\t" \
    "madc.lo.cc.u32 %5, %8, %14, %5;\n\t" \
    "addc.u32 %6, %6, 0;\n\t" \
    "mad.hi.cc.u32 %1, %8, %9, %1;\n\t" \
    "madc.hi.cc.u32 %2, %8, %10, %2;\n\t" \
    "madc.hi.cc.u32 %3, %8, %11, %3;\n\t" \
    "madc.hi.cc.u32 %4, %8, %12, %4;\n\t" \
    "madc.hi.cc.u32 %5, %8, %13, %5;\n\t" \
    "madc.hi.cc.u32 %6, %8, %14, %6;\n\t" \
    "addc.u32 %7, %7, 0;" \
    : "+r"((R)[0]), "+r"((R)[1]), "+r"((R)[2]), "+r"((R)[3]), "+r"((R)[4]), "+r"((R)[5]), "+r"((R)[6]), "+r"((R)[7]) \
    : "r"(a), "r"((b)[0]), "r"((b)[1]), "r"((b)[2]), "r"((b)[3]), "r"((b)[4]), "r"((b)[5]))
#define PX_ROW7(R, a, b) asm( \
    "mad.lo.cc.u32 %0, %9, %10, %0;\n\t" \
    "madc.lo.cc.u32 %1, %9, %11, %1;\n\t" \
    "madc.lo.cc.u32 %2, %9, %12, %2;\n\t" \
    "madc.lo.cc.u32 %3, %9, %13, %3;\n\t" \
    "madc.lo.cc.u32 %4, %9, %14, %4;\n\t" \
    "madc.lo.cc.u32 %5, %9, %15, %5;\n\t" \
    "madc.lo.cc.u32 %6, %9, %16, %6;\n\t" \
    "addc.u32 %7, %7, 0;\n\t" \
    "mad.hi.cc.u32 %1, %9, %10, %1;\n\t" \
    "madc.hi.cc.u32 %2, %9, %11, %2;\n\t" \
    "madc.hi.cc.u32 %3, %9, %12, %3;\n\t" \
    "madc.hi.cc.u32 %4, %9, %13, %4;\n\t" \
    "madc.hi.cc.u32 %5, %9, %14, %5;\n\t" \
    "madc.hi.cc.u32 %6, %9, %15, %6;\n\t" \
    "madc.hi.cc.u32 %7, %9, %16, %7;\n\t" \
    "addc.u32 %8, %8, 0;" \
    : "+r"((R)[0]), "+r"((R)[1]), "+r"((R)[2]), "+r"((R)[3]), "+r"((R)[4]), "+r"((R)[5]), "+r"((R)[6]), "+r"((R)[7]), "+r"((R)[8]) \
    : "r"(a), "r"((b)[0]), "r"((b)[1]), "r"((b)[2]), "r"((b)[3]), "r"((b)[4]), "r"((b)[5]), "r"((b)[6]))

// shared tail of mul and sq: fold the 512-bit product r[0..15] into h
__device__ __forceinline__ void fe_fold512(fe &h, uint32_t r[18]) {
    uint32_t hi[8];
    for (int i = 0; i < 8; i++) { hi[i] = r[8 + i]; r[8 + i] = 0; }
    PX_MUL_ROW(r + 0, 38u, hi);        // r[0..8] = lo + 38*hi, r[8] <= 38
    for (int i = 0; i < 8; i++) h.v[i] = r[i];
    // r[8] is non-zero in about 97 % of the lanes: fold it without a branch
    uint32_t top = r[8];
    {
        uint32_t c, k = 38u;
        asm(
            "mad.lo.cc.u32 %0, %9, %10, %0;\n\t"
            "addc.cc.u32 %1, %1, 0;\n\t" "addc.cc.u32 %2, %2, 0;\n\t" "addc.cc.u32 %3, %3, 0;\n\t"
            "addc.cc.u32 %4, %4, 0;\n\t" "addc.cc.u32 %5, %5, 0;\n\t" "addc.cc.u32 %6, %6, 0;\n\t"
            "addc.cc.u32 %7, %7, 0;\n\t" "addc.u32 %8, 0, 0;"
            : "+r"(h.v[0]), "+r"(h.v[1]), "+r"(h.v[2]), "+r"(h.v[3]), "+r"(h.v[4]), "+r"(h.v[5]), "+r"(h.v[6]), "+r"(h.v[7]), "=&r"(c)
            : "r"(top), "r"(k));
        if (c) fe_addsmall(h, 38u);
    }
}

#ifdef PX_COMBA
// ----------------------------------------------------------------------------
// Product scanning (comba) over 32-bit limbs. Each 32x32 product is accumulated
// into a three-word column with "mad.lo.cc / madc.hi.cc / addc", which ptxas
// fuses into one IMAD.WIDE.U32 with a predicate carry-out plus one IADD.X; no
// mad.hi (which runs at a third of the IMAD rate on sm_120) and no long chain
// through a single carry flag. The 2^256 = 38 fold is done inside the low
// columns: the high words are computed first and each low column k is seeded
// with 38 * word(8+k). Measured 1.4-1.6x the shipped row/carry-chain version
// per field operation on an RTX 5090 (PLATFORMS.md).
// ----------------------------------------------------------------------------
// (c2:c1:c0) += a*b
#define PX_CSTEP(c0, c1, c2, a, b) asm("mad.lo.cc.u32 %0, %3, %4, %0;\n\tmadc.hi.cc.u32 %1, %3, %4, %1;\n\taddc.u32 %2, %2, 0;" : "+r"(c0), "+r"(c1), "+r"(c2) : "r"(a), "r"(b))
// (c1:c0) += a*b, no third word (a column that cannot overflow two words)
#define PX_CSTEP2(c0, c1, a, b) asm("mad.lo.cc.u32 %0, %2, %3, %0;\n\tmadc.hi.u32 %1, %2, %3, %1;" : "+r"(c0), "+r"(c1) : "r"(a), "r"(b))
#define PX_CSHIFT(c0, c1, c2) do { c0 = c1; c1 = c2; c2 = 0; } while (0)
// h += (t1:t0) at words 0,1 with the carry propagated; a wrap past 2^256 is folded as 38
__device__ __forceinline__ void fe_addtop(fe &h, uint32_t t0, uint32_t t1) {
    uint32_t c;   // t1 < 2^11: the carry out of limb 1 is rare, propagate it off the hot path
    asm("add.cc.u32 %0, %0, %3;\n\taddc.cc.u32 %1, %1, %4;\n\taddc.u32 %2, 0, 0;" : "+r"(h.v[0]), "+r"(h.v[1]), "=&r"(c) : "r"(t0), "r"(t1));
#if !PX_NOBRANCH
    if (c) fe_inc2(h);
#else
    (void)c;
#endif
}
// NB = 1: the carry out of limb 1 is dropped instead of propagated (ladder hot path only, see mlad_step_nb)
template<int NB> __device__ __forceinline__ void fe_addtop_t(fe &h, uint32_t t0, uint32_t t1) {
    if (NB) { asm("add.cc.u32 %0, %0, %2;\n\taddc.u32 %1, %1, %3;" : "+r"(h.v[0]), "+r"(h.v[1]) : "r"(t0), "r"(t1)); }
    else fe_addtop(h, t0, t1);
}
#ifdef PX_WIDECHAIN
// ----------------------------------------------------------------------------
// Row-ordered schoolbook with wide-multiply carry chains (0.5.10). The products
// of one row are issued two columns apart: the 2^64 carry of one IMAD.WIDE.U32
// is the carry-in of the next (IMAD.WIDE.U32.X), so no per-product add is
// needed; one addc per row captures the chain's last carry, the other chain
// of the row ends at column i+7 whose carry is provably zero (the partial
// product has no word there). Even columns accumulate into E on aligned
// register pairs, odd columns into Od (window (c,c+1) stored at Od[c-1]),
// and the two are added once at the end. Same multiply count as comba, about
// half its integer-add count: on GA10x (RTX 30xx) the adds share the issue
// slots and the integer pipe with the carries, and the multiplier pipe is the
// bottleneck, so the freed slots go to the remaining work.
// ----------------------------------------------------------------------------
// Row-ordered schoolbook. Products at even columns accumulate into E (64-bit windows (c,c+1) on aligned
// register pairs), products at odd columns into Od (window (c,c+1) stored at Od[c-1], Od[c], also aligned).
// Within a row the products of one parity are chained: the 2^64 carry of one IMAD.WIDE.U32 is the carry-in of
// the next one two columns up (IMAD.WIDE.U32.X), so there is no per-product IADD.X; one addc per row captures
// the chain's last carry (the other chain ends at column i+7 whose carry is provably zero).
// t * 38 for a small t (< 2^26) as shifts and adds on the integer pipe, never a multiplier-pipe IMAD
__device__ __forceinline__ uint32_t fe_mul38small(uint32_t t) {
    uint32_t r; asm("shl.b32 %0, %1, 5;\n\tshl.b32 %1, %1, 1;\n\tadd.u32 %0, %0, %1;\n\tshl.b32 %1, %1, 1;\n\tadd.u32 %0, %0, %1;" : "=&r"(r), "+r"(t)); return r;
}
#ifndef PX_FOLDSHIFT
#define PX_FOLDSHIFT 0
#endif
#if PX_ZFREE && !PX_ROW0C
#error PX_ZFREE needs PX_ROW0C
#endif
template<int NB, int FS = PX_FOLDSHIFT> __device__ void fe_mul_t(fe &h, const fe &f, const fe &g) {
    const uint32_t *a = f.v, *b = g.v;
    uint32_t E[16], Od[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) { E[i] = 0; Od[i] = 0; }
#if PX_ROW0C
    // row 0 into zero accumulators: eight independent 32x32->64 products, no carries (one IMAD.WIDE each)
    #pragma unroll
    for (int c = 0; c < 4; c++) {
        uint64_t pe = (uint64_t)a[0] * b[2 * c], po = (uint64_t)a[0] * b[2 * c + 1];
        E[2 * c] = (uint32_t)pe; E[2 * c + 1] = (uint32_t)(pe >> 32);
        Od[2 * c] = (uint32_t)po; Od[2 * c + 1] = (uint32_t)(po >> 32);
    }
#else
    // row 0
    asm("mad.lo.cc.u32 %0, %17, %18, %0;\n\tmadc.hi.cc.u32 %1, %17, %18, %1;\n\tmadc.lo.cc.u32 %2, %17, %20, %2;\n\tmadc.hi.cc.u32 %3, %17, %20, %3;\n\tmadc.lo.cc.u32 %4, %17, %22, %4;\n\tmadc.hi.cc.u32 %5, %17, %22, %5;\n\tmadc.lo.cc.u32 %6, %17, %24, %6;\n\tmadc.hi.cc.u32 %7, %17, %24, %7;\n\taddc.u32 %8, %8, 0;\n\tmad.lo.cc.u32 %9, %17, %19, %9;\n\tmadc.hi.cc.u32 %10, %17, %19, %10;\n\tmadc.lo.cc.u32 %11, %17, %21, %11;\n\tmadc.hi.cc.u32 %12, %17, %21, %12;\n\tmadc.lo.cc.u32 %13, %17, %23, %13;\n\tmadc.hi.cc.u32 %14, %17, %23, %14;\n\tmadc.lo.cc.u32 %15, %17, %25, %15;\n\tmadc.hi.u32 %16, %17, %25, %16;"
        : "+r"(E[0]), "+r"(E[1]), "+r"(E[2]), "+r"(E[3]), "+r"(E[4]), "+r"(E[5]), "+r"(E[6]), "+r"(E[7]), "+r"(E[8]), "+r"(Od[0]), "+r"(Od[1]), "+r"(Od[2]), "+r"(Od[3]), "+r"(Od[4]), "+r"(Od[5]), "+r"(Od[6]), "+r"(Od[7])
        : "r"(a[0]), "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), "r"(b[4]), "r"(b[5]), "r"(b[6]), "r"(b[7]));
#endif
    // row 1
#if PX_ZFREE
    asm("mad.lo.cc.u32 %0, %17, %19, %0;\n\tmadc.hi.cc.u32 %1, %17, %19, %1;\n\tmadc.lo.cc.u32 %2, %17, %21, %2;\n\tmadc.hi.cc.u32 %3, %17, %21, %3;\n\tmadc.lo.cc.u32 %4, %17, %23, %4;\n\tmadc.hi.cc.u32 %5, %17, %23, %5;\n\tmadc.lo.cc.u32 %6, %17, %25, 0;\n\tmadc.hi.u32 %7, %17, %25, 0;\n\tmad.lo.cc.u32 %8, %17, %18, %8;\n\tmadc.hi.cc.u32 %9, %17, %18, %9;\n\tmadc.lo.cc.u32 %10, %17, %20, %10;\n\tmadc.hi.cc.u32 %11, %17, %20, %11;\n\tmadc.lo.cc.u32 %12, %17, %22, %12;\n\tmadc.hi.cc.u32 %13, %17, %22, %13;\n\tmadc.lo.cc.u32 %14, %17, %24, %14;\n\tmadc.hi.cc.u32 %15, %17, %24, %15;\n\taddc.u32 %16, 0, 0;"
        : "+r"(E[2]), "+r"(E[3]), "+r"(E[4]), "+r"(E[5]), "+r"(E[6]), "+r"(E[7]), "=&r"(E[8]), "=&r"(E[9]), "+r"(Od[0]), "+r"(Od[1]), "+r"(Od[2]), "+r"(Od[3]), "+r"(Od[4]), "+r"(Od[5]), "+r"(Od[6]), "+r"(Od[7]), "=&r"(Od[8])
        : "r"(a[1]), "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), "r"(b[4]), "r"(b[5]), "r"(b[6]), "r"(b[7]));
#else
    asm("mad.lo.cc.u32 %0, %17, %19, %0;\n\tmadc.hi.cc.u32 %1, %17, %19, %1;\n\tmadc.lo.cc.u32 %2, %17, %21, %2;\n\tmadc.hi.cc.u32 %3, %17, %21, %3;\n\tmadc.lo.cc.u32 %4, %17, %23, %4;\n\tmadc.hi.cc.u32 %5, %17, %23, %5;\n\tmadc.lo.cc.u32 %6, %17, %25, %6;\n\tmadc.hi.u32 %7, %17, %25, %7;\n\tmad.lo.cc.u32 %8, %17, %18, %8;\n\tmadc.hi.cc.u32 %9, %17, %18, %9;\n\tmadc.lo.cc.u32 %10, %17, %20, %10;\n\tmadc.hi.cc.u32 %11, %17, %20, %11;\n\tmadc.lo.cc.u32 %12, %17, %22, %12;\n\tmadc.hi.cc.u32 %13, %17, %22, %13;\n\tmadc.lo.cc.u32 %14, %17, %24, %14;\n\tmadc.hi.cc.u32 %15, %17, %24, %15;\n\taddc.u32 %16, %16, 0;"
        : "+r"(E[2]), "+r"(E[3]), "+r"(E[4]), "+r"(E[5]), "+r"(E[6]), "+r"(E[7]), "+r"(E[8]), "+r"(E[9]), "+r"(Od[0]), "+r"(Od[1]), "+r"(Od[2]), "+r"(Od[3]), "+r"(Od[4]), "+r"(Od[5]), "+r"(Od[6]), "+r"(Od[7]), "+r"(Od[8])
        : "r"(a[1]), "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), "r"(b[4]), "r"(b[5]), "r"(b[6]), "r"(b[7]));
#endif
    // row 2
#if PX_ZFREE
    asm("mad.lo.cc.u32 %0, %17, %18, %0;\n\tmadc.hi.cc.u32 %1, %17, %18, %1;\n\tmadc.lo.cc.u32 %2, %17, %20, %2;\n\tmadc.hi.cc.u32 %3, %17, %20, %3;\n\tmadc.lo.cc.u32 %4, %17, %22, %4;\n\tmadc.hi.cc.u32 %5, %17, %22, %5;\n\tmadc.lo.cc.u32 %6, %17, %24, %6;\n\tmadc.hi.cc.u32 %7, %17, %24, %7;\n\taddc.u32 %8, 0, 0;\n\tmad.lo.cc.u32 %9, %17, %19, %9;\n\tmadc.hi.cc.u32 %10, %17, %19, %10;\n\tmadc.lo.cc.u32 %11, %17, %21, %11;\n\tmadc.hi.cc.u32 %12, %17, %21, %12;\n\tmadc.lo.cc.u32 %13, %17, %23, %13;\n\tmadc.hi.cc.u32 %14, %17, %23, %14;\n\tmadc.lo.cc.u32 %15, %17, %25, %15;\n\tmadc.hi.u32 %16, %17, %25, 0;"
        : "+r"(E[2]), "+r"(E[3]), "+r"(E[4]), "+r"(E[5]), "+r"(E[6]), "+r"(E[7]), "+r"(E[8]), "+r"(E[9]), "=&r"(E[10]), "+r"(Od[2]), "+r"(Od[3]), "+r"(Od[4]), "+r"(Od[5]), "+r"(Od[6]), "+r"(Od[7]), "+r"(Od[8]), "=&r"(Od[9])
        : "r"(a[2]), "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), "r"(b[4]), "r"(b[5]), "r"(b[6]), "r"(b[7]));
#else
    asm("mad.lo.cc.u32 %0, %17, %18, %0;\n\tmadc.hi.cc.u32 %1, %17, %18, %1;\n\tmadc.lo.cc.u32 %2, %17, %20, %2;\n\tmadc.hi.cc.u32 %3, %17, %20, %3;\n\tmadc.lo.cc.u32 %4, %17, %22, %4;\n\tmadc.hi.cc.u32 %5, %17, %22, %5;\n\tmadc.lo.cc.u32 %6, %17, %24, %6;\n\tmadc.hi.cc.u32 %7, %17, %24, %7;\n\taddc.u32 %8, %8, 0;\n\tmad.lo.cc.u32 %9, %17, %19, %9;\n\tmadc.hi.cc.u32 %10, %17, %19, %10;\n\tmadc.lo.cc.u32 %11, %17, %21, %11;\n\tmadc.hi.cc.u32 %12, %17, %21, %12;\n\tmadc.lo.cc.u32 %13, %17, %23, %13;\n\tmadc.hi.cc.u32 %14, %17, %23, %14;\n\tmadc.lo.cc.u32 %15, %17, %25, %15;\n\tmadc.hi.u32 %16, %17, %25, %16;"
        : "+r"(E[2]), "+r"(E[3]), "+r"(E[4]), "+r"(E[5]), "+r"(E[6]), "+r"(E[7]), "+r"(E[8]), "+r"(E[9]), "+r"(E[10]), "+r"(Od[2]), "+r"(Od[3]), "+r"(Od[4]), "+r"(Od[5]), "+r"(Od[6]), "+r"(Od[7]), "+r"(Od[8]), "+r"(Od[9])
        : "r"(a[2]), "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), "r"(b[4]), "r"(b[5]), "r"(b[6]), "r"(b[7]));
#endif
    // row 3
#if PX_ZFREE
    asm("mad.lo.cc.u32 %0, %17, %19, %0;\n\tmadc.hi.cc.u32 %1, %17, %19, %1;\n\tmadc.lo.cc.u32 %2, %17, %21, %2;\n\tmadc.hi.cc.u32 %3, %17, %21, %3;\n\tmadc.lo.cc.u32 %4, %17, %23, %4;\n\tmadc.hi.cc.u32 %5, %17, %23, %5;\n\tmadc.lo.cc.u32 %6, %17, %25, %6;\n\tmadc.hi.u32 %7, %17, %25, 0;\n\tmad.lo.cc.u32 %8, %17, %18, %8;\n\tmadc.hi.cc.u32 %9, %17, %18, %9;\n\tmadc.lo.cc.u32 %10, %17, %20, %10;\n\tmadc.hi.cc.u32 %11, %17, %20, %11;\n\tmadc.lo.cc.u32 %12, %17, %22, %12;\n\tmadc.hi.cc.u32 %13, %17, %22, %13;\n\tmadc.lo.cc.u32 %14, %17, %24, %14;\n\tmadc.hi.cc.u32 %15, %17, %24, %15;\n\taddc.u32 %16, 0, 0;"
        : "+r"(E[4]), "+r"(E[5]), "+r"(E[6]), "+r"(E[7]), "+r"(E[8]), "+r"(E[9]), "+r"(E[10]), "=&r"(E[11]), "+r"(Od[2]), "+r"(Od[3]), "+r"(Od[4]), "+r"(Od[5]), "+r"(Od[6]), "+r"(Od[7]), "+r"(Od[8]), "+r"(Od[9]), "=&r"(Od[10])
        : "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), "r"(b[4]), "r"(b[5]), "r"(b[6]), "r"(b[7]));
#else
    asm("mad.lo.cc.u32 %0, %17, %19, %0;\n\tmadc.hi.cc.u32 %1, %17, %19, %1;\n\tmadc.lo.cc.u32 %2, %17, %21, %2;\n\tmadc.hi.cc.u32 %3, %17, %21, %3;\n\tmadc.lo.cc.u32 %4, %17, %23, %4;\n\tmadc.hi.cc.u32 %5, %17, %23, %5;\n\tmadc.lo.cc.u32 %6, %17, %25, %6;\n\tmadc.hi.u32 %7, %17, %25, %7;\n\tmad.lo.cc.u32 %8, %17, %18, %8;\n\tmadc.hi.cc.u32 %9, %17, %18, %9;\n\tmadc.lo.cc.u32 %10, %17, %20, %10;\n\tmadc.hi.cc.u32 %11, %17, %20, %11;\n\tmadc.lo.cc.u32 %12, %17, %22, %12;\n\tmadc.hi.cc.u32 %13, %17, %22, %13;\n\tmadc.lo.cc.u32 %14, %17, %24, %14;\n\tmadc.hi.cc.u32 %15, %17, %24, %15;\n\taddc.u32 %16, %16, 0;"
        : "+r"(E[4]), "+r"(E[5]), "+r"(E[6]), "+r"(E[7]), "+r"(E[8]), "+r"(E[9]), "+r"(E[10]), "+r"(E[11]), "+r"(Od[2]), "+r"(Od[3]), "+r"(Od[4]), "+r"(Od[5]), "+r"(Od[6]), "+r"(Od[7]), "+r"(Od[8]), "+r"(Od[9]), "+r"(Od[10])
        : "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), "r"(b[4]), "r"(b[5]), "r"(b[6]), "r"(b[7]));
#endif
    // row 4
#if PX_ZFREE
    asm("mad.lo.cc.u32 %0, %17, %18, %0;\n\tmadc.hi.cc.u32 %1, %17, %18, %1;\n\tmadc.lo.cc.u32 %2, %17, %20, %2;\n\tmadc.hi.cc.u32 %3, %17, %20, %3;\n\tmadc.lo.cc.u32 %4, %17, %22, %4;\n\tmadc.hi.cc.u32 %5, %17, %22, %5;\n\tmadc.lo.cc.u32 %6, %17, %24, %6;\n\tmadc.hi.cc.u32 %7, %17, %24, %7;\n\taddc.u32 %8, 0, 0;\n\tmad.lo.cc.u32 %9, %17, %19, %9;\n\tmadc.hi.cc.u32 %10, %17, %19, %10;\n\tmadc.lo.cc.u32 %11, %17, %21, %11;\n\tmadc.hi.cc.u32 %12, %17, %21, %12;\n\tmadc.lo.cc.u32 %13, %17, %23, %13;\n\tmadc.hi.cc.u32 %14, %17, %23, %14;\n\tmadc.lo.cc.u32 %15, %17, %25, %15;\n\tmadc.hi.u32 %16, %17, %25, 0;"
        : "+r"(E[4]), "+r"(E[5]), "+r"(E[6]), "+r"(E[7]), "+r"(E[8]), "+r"(E[9]), "+r"(E[10]), "+r"(E[11]), "=&r"(E[12]), "+r"(Od[4]), "+r"(Od[5]), "+r"(Od[6]), "+r"(Od[7]), "+r"(Od[8]), "+r"(Od[9]), "+r"(Od[10]), "=&r"(Od[11])
        : "r"(a[4]), "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), "r"(b[4]), "r"(b[5]), "r"(b[6]), "r"(b[7]));
#else
    asm("mad.lo.cc.u32 %0, %17, %18, %0;\n\tmadc.hi.cc.u32 %1, %17, %18, %1;\n\tmadc.lo.cc.u32 %2, %17, %20, %2;\n\tmadc.hi.cc.u32 %3, %17, %20, %3;\n\tmadc.lo.cc.u32 %4, %17, %22, %4;\n\tmadc.hi.cc.u32 %5, %17, %22, %5;\n\tmadc.lo.cc.u32 %6, %17, %24, %6;\n\tmadc.hi.cc.u32 %7, %17, %24, %7;\n\taddc.u32 %8, %8, 0;\n\tmad.lo.cc.u32 %9, %17, %19, %9;\n\tmadc.hi.cc.u32 %10, %17, %19, %10;\n\tmadc.lo.cc.u32 %11, %17, %21, %11;\n\tmadc.hi.cc.u32 %12, %17, %21, %12;\n\tmadc.lo.cc.u32 %13, %17, %23, %13;\n\tmadc.hi.cc.u32 %14, %17, %23, %14;\n\tmadc.lo.cc.u32 %15, %17, %25, %15;\n\tmadc.hi.u32 %16, %17, %25, %16;"
        : "+r"(E[4]), "+r"(E[5]), "+r"(E[6]), "+r"(E[7]), "+r"(E[8]), "+r"(E[9]), "+r"(E[10]), "+r"(E[11]), "+r"(E[12]), "+r"(Od[4]), "+r"(Od[5]), "+r"(Od[6]), "+r"(Od[7]), "+r"(Od[8]), "+r"(Od[9]), "+r"(Od[10]), "+r"(Od[11])
        : "r"(a[4]), "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), "r"(b[4]), "r"(b[5]), "r"(b[6]), "r"(b[7]));
#endif
    // row 5
#if PX_ZFREE
    asm("mad.lo.cc.u32 %0, %17, %19, %0;\n\tmadc.hi.cc.u32 %1, %17, %19, %1;\n\tmadc.lo.cc.u32 %2, %17, %21, %2;\n\tmadc.hi.cc.u32 %3, %17, %21, %3;\n\tmadc.lo.cc.u32 %4, %17, %23, %4;\n\tmadc.hi.cc.u32 %5, %17, %23, %5;\n\tmadc.lo.cc.u32 %6, %17, %25, %6;\n\tmadc.hi.u32 %7, %17, %25, 0;\n\tmad.lo.cc.u32 %8, %17, %18, %8;\n\tmadc.hi.cc.u32 %9, %17, %18, %9;\n\tmadc.lo.cc.u32 %10, %17, %20, %10;\n\tmadc.hi.cc.u32 %11, %17, %20, %11;\n\tmadc.lo.cc.u32 %12, %17, %22, %12;\n\tmadc.hi.cc.u32 %13, %17, %22, %13;\n\tmadc.lo.cc.u32 %14, %17, %24, %14;\n\tmadc.hi.cc.u32 %15, %17, %24, %15;\n\taddc.u32 %16, 0, 0;"
        : "+r"(E[6]), "+r"(E[7]), "+r"(E[8]), "+r"(E[9]), "+r"(E[10]), "+r"(E[11]), "+r"(E[12]), "=&r"(E[13]), "+r"(Od[4]), "+r"(Od[5]), "+r"(Od[6]), "+r"(Od[7]), "+r"(Od[8]), "+r"(Od[9]), "+r"(Od[10]), "+r"(Od[11]), "=&r"(Od[12])
        : "r"(a[5]), "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), "r"(b[4]), "r"(b[5]), "r"(b[6]), "r"(b[7]));
#else
    asm("mad.lo.cc.u32 %0, %17, %19, %0;\n\tmadc.hi.cc.u32 %1, %17, %19, %1;\n\tmadc.lo.cc.u32 %2, %17, %21, %2;\n\tmadc.hi.cc.u32 %3, %17, %21, %3;\n\tmadc.lo.cc.u32 %4, %17, %23, %4;\n\tmadc.hi.cc.u32 %5, %17, %23, %5;\n\tmadc.lo.cc.u32 %6, %17, %25, %6;\n\tmadc.hi.u32 %7, %17, %25, %7;\n\tmad.lo.cc.u32 %8, %17, %18, %8;\n\tmadc.hi.cc.u32 %9, %17, %18, %9;\n\tmadc.lo.cc.u32 %10, %17, %20, %10;\n\tmadc.hi.cc.u32 %11, %17, %20, %11;\n\tmadc.lo.cc.u32 %12, %17, %22, %12;\n\tmadc.hi.cc.u32 %13, %17, %22, %13;\n\tmadc.lo.cc.u32 %14, %17, %24, %14;\n\tmadc.hi.cc.u32 %15, %17, %24, %15;\n\taddc.u32 %16, %16, 0;"
        : "+r"(E[6]), "+r"(E[7]), "+r"(E[8]), "+r"(E[9]), "+r"(E[10]), "+r"(E[11]), "+r"(E[12]), "+r"(E[13]), "+r"(Od[4]), "+r"(Od[5]), "+r"(Od[6]), "+r"(Od[7]), "+r"(Od[8]), "+r"(Od[9]), "+r"(Od[10]), "+r"(Od[11]), "+r"(Od[12])
        : "r"(a[5]), "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), "r"(b[4]), "r"(b[5]), "r"(b[6]), "r"(b[7]));
#endif
    // row 6
#if PX_ZFREE
    asm("mad.lo.cc.u32 %0, %17, %18, %0;\n\tmadc.hi.cc.u32 %1, %17, %18, %1;\n\tmadc.lo.cc.u32 %2, %17, %20, %2;\n\tmadc.hi.cc.u32 %3, %17, %20, %3;\n\tmadc.lo.cc.u32 %4, %17, %22, %4;\n\tmadc.hi.cc.u32 %5, %17, %22, %5;\n\tmadc.lo.cc.u32 %6, %17, %24, %6;\n\tmadc.hi.cc.u32 %7, %17, %24, %7;\n\taddc.u32 %8, 0, 0;\n\tmad.lo.cc.u32 %9, %17, %19, %9;\n\tmadc.hi.cc.u32 %10, %17, %19, %10;\n\tmadc.lo.cc.u32 %11, %17, %21, %11;\n\tmadc.hi.cc.u32 %12, %17, %21, %12;\n\tmadc.lo.cc.u32 %13, %17, %23, %13;\n\tmadc.hi.cc.u32 %14, %17, %23, %14;\n\tmadc.lo.cc.u32 %15, %17, %25, %15;\n\tmadc.hi.u32 %16, %17, %25, 0;"
        : "+r"(E[6]), "+r"(E[7]), "+r"(E[8]), "+r"(E[9]), "+r"(E[10]), "+r"(E[11]), "+r"(E[12]), "+r"(E[13]), "=&r"(E[14]), "+r"(Od[6]), "+r"(Od[7]), "+r"(Od[8]), "+r"(Od[9]), "+r"(Od[10]), "+r"(Od[11]), "+r"(Od[12]), "=&r"(Od[13])
        : "r"(a[6]), "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), "r"(b[4]), "r"(b[5]), "r"(b[6]), "r"(b[7]));
#else
    asm("mad.lo.cc.u32 %0, %17, %18, %0;\n\tmadc.hi.cc.u32 %1, %17, %18, %1;\n\tmadc.lo.cc.u32 %2, %17, %20, %2;\n\tmadc.hi.cc.u32 %3, %17, %20, %3;\n\tmadc.lo.cc.u32 %4, %17, %22, %4;\n\tmadc.hi.cc.u32 %5, %17, %22, %5;\n\tmadc.lo.cc.u32 %6, %17, %24, %6;\n\tmadc.hi.cc.u32 %7, %17, %24, %7;\n\taddc.u32 %8, %8, 0;\n\tmad.lo.cc.u32 %9, %17, %19, %9;\n\tmadc.hi.cc.u32 %10, %17, %19, %10;\n\tmadc.lo.cc.u32 %11, %17, %21, %11;\n\tmadc.hi.cc.u32 %12, %17, %21, %12;\n\tmadc.lo.cc.u32 %13, %17, %23, %13;\n\tmadc.hi.cc.u32 %14, %17, %23, %14;\n\tmadc.lo.cc.u32 %15, %17, %25, %15;\n\tmadc.hi.u32 %16, %17, %25, %16;"
        : "+r"(E[6]), "+r"(E[7]), "+r"(E[8]), "+r"(E[9]), "+r"(E[10]), "+r"(E[11]), "+r"(E[12]), "+r"(E[13]), "+r"(E[14]), "+r"(Od[6]), "+r"(Od[7]), "+r"(Od[8]), "+r"(Od[9]), "+r"(Od[10]), "+r"(Od[11]), "+r"(Od[12]), "+r"(Od[13])
        : "r"(a[6]), "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), "r"(b[4]), "r"(b[5]), "r"(b[6]), "r"(b[7]));
#endif
    // row 7
#if PX_ZFREE
    asm("mad.lo.cc.u32 %0, %17, %19, %0;\n\tmadc.hi.cc.u32 %1, %17, %19, %1;\n\tmadc.lo.cc.u32 %2, %17, %21, %2;\n\tmadc.hi.cc.u32 %3, %17, %21, %3;\n\tmadc.lo.cc.u32 %4, %17, %23, %4;\n\tmadc.hi.cc.u32 %5, %17, %23, %5;\n\tmadc.lo.cc.u32 %6, %17, %25, %6;\n\tmadc.hi.u32 %7, %17, %25, 0;\n\tmad.lo.cc.u32 %8, %17, %18, %8;\n\tmadc.hi.cc.u32 %9, %17, %18, %9;\n\tmadc.lo.cc.u32 %10, %17, %20, %10;\n\tmadc.hi.cc.u32 %11, %17, %20, %11;\n\tmadc.lo.cc.u32 %12, %17, %22, %12;\n\tmadc.hi.cc.u32 %13, %17, %22, %13;\n\tmadc.lo.cc.u32 %14, %17, %24, %14;\n\tmadc.hi.cc.u32 %15, %17, %24, %15;\n\taddc.u32 %16, 0, 0;"
        : "+r"(E[8]), "+r"(E[9]), "+r"(E[10]), "+r"(E[11]), "+r"(E[12]), "+r"(E[13]), "+r"(E[14]), "=&r"(E[15]), "+r"(Od[6]), "+r"(Od[7]), "+r"(Od[8]), "+r"(Od[9]), "+r"(Od[10]), "+r"(Od[11]), "+r"(Od[12]), "+r"(Od[13]), "=&r"(Od[14])
        : "r"(a[7]), "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), "r"(b[4]), "r"(b[5]), "r"(b[6]), "r"(b[7]));
#else
    asm("mad.lo.cc.u32 %0, %17, %19, %0;\n\tmadc.hi.cc.u32 %1, %17, %19, %1;\n\tmadc.lo.cc.u32 %2, %17, %21, %2;\n\tmadc.hi.cc.u32 %3, %17, %21, %3;\n\tmadc.lo.cc.u32 %4, %17, %23, %4;\n\tmadc.hi.cc.u32 %5, %17, %23, %5;\n\tmadc.lo.cc.u32 %6, %17, %25, %6;\n\tmadc.hi.u32 %7, %17, %25, %7;\n\tmad.lo.cc.u32 %8, %17, %18, %8;\n\tmadc.hi.cc.u32 %9, %17, %18, %9;\n\tmadc.lo.cc.u32 %10, %17, %20, %10;\n\tmadc.hi.cc.u32 %11, %17, %20, %11;\n\tmadc.lo.cc.u32 %12, %17, %22, %12;\n\tmadc.hi.cc.u32 %13, %17, %22, %13;\n\tmadc.lo.cc.u32 %14, %17, %24, %14;\n\tmadc.hi.cc.u32 %15, %17, %24, %15;\n\taddc.u32 %16, %16, 0;"
        : "+r"(E[8]), "+r"(E[9]), "+r"(E[10]), "+r"(E[11]), "+r"(E[12]), "+r"(E[13]), "+r"(E[14]), "+r"(E[15]), "+r"(Od[6]), "+r"(Od[7]), "+r"(Od[8]), "+r"(Od[9]), "+r"(Od[10]), "+r"(Od[11]), "+r"(Od[12]), "+r"(Od[13]), "+r"(Od[14])
        : "r"(a[7]), "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), "r"(b[4]), "r"(b[5]), "r"(b[6]), "r"(b[7]));
#endif
    // hi = words 8..15 of the product = E[8..15] + Od[7..14]. No carry out: the sum is at most the
    // product's high half (it lacks only the carry from the low half), which is below 2^256.
    uint32_t hi[8];
    asm("add.cc.u32 %0, %8, %16;\n\taddc.cc.u32 %1, %9, %17;\n\taddc.cc.u32 %2, %10, %18;\n\taddc.cc.u32 %3, %11, %19;\n\t"
        "addc.cc.u32 %4, %12, %20;\n\taddc.cc.u32 %5, %13, %21;\n\taddc.cc.u32 %6, %14, %22;\n\taddc.u32 %7, %15, %23;"
        : "=&r"(hi[0]), "=&r"(hi[1]), "=&r"(hi[2]), "=&r"(hi[3]), "=&r"(hi[4]), "=&r"(hi[5]), "=&r"(hi[6]), "=&r"(hi[7])
        : "r"(E[8]), "r"(E[9]), "r"(E[10]), "r"(E[11]), "r"(E[12]), "r"(E[13]), "r"(E[14]), "r"(E[15]),
          "r"(Od[7]), "r"(Od[8]), "r"(Od[9]), "r"(Od[10]), "r"(Od[11]), "r"(Od[12]), "r"(Od[13]), "r"(Od[14]));
    if (FS) {
    // 2^256 = 38: h = lo + 38 * hi with 38 * hi = hi<<5 + hi<<2 + hi<<1 built from funnel shifts and
    // added on the integer pipe, so the fold costs no wide multiplies at all (the multiplier pipe is
    // the bottleneck on GA10x). lo = E[0..8] + Od[0..7]<<32 first, nine words.
    // lo = E[0..7] + Od[0..6]<<32 (E[8] and Od[7] are word 8: they belong to hi), L[8] = its carry
    uint32_t L[9];
    L[0] = E[0];
    asm("add.cc.u32 %0, %8, %15;\n\taddc.cc.u32 %1, %9, %16;\n\taddc.cc.u32 %2, %10, %17;\n\taddc.cc.u32 %3, %11, %18;\n\t"
        "addc.cc.u32 %4, %12, %19;\n\taddc.cc.u32 %5, %13, %20;\n\taddc.cc.u32 %6, %14, %21;\n\taddc.u32 %7, 0, 0;"
        : "=&r"(L[1]), "=&r"(L[2]), "=&r"(L[3]), "=&r"(L[4]), "=&r"(L[5]), "=&r"(L[6]), "=&r"(L[7]), "=&r"(L[8])
        : "r"(E[1]), "r"(E[2]), "r"(E[3]), "r"(E[4]), "r"(E[5]), "r"(E[6]), "r"(E[7]),
          "r"(Od[0]), "r"(Od[1]), "r"(Od[2]), "r"(Od[3]), "r"(Od[4]), "r"(Od[5]), "r"(Od[6]));
#define PX_ADDSHIFTED(sh) do { \
        uint32_t t[9]; \
        t[0] = hi[0] << (sh); \
        _Pragma("unroll") for (int k = 1; k < 8; k++) t[k] = __funnelshift_l(hi[k - 1], hi[k], (sh)); \
        t[8] = hi[7] >> (32 - (sh)); \
        asm("add.cc.u32 %0, %0, %9;\n\taddc.cc.u32 %1, %1, %10;\n\taddc.cc.u32 %2, %2, %11;\n\taddc.cc.u32 %3, %3, %12;\n\t" \
            "addc.cc.u32 %4, %4, %13;\n\taddc.cc.u32 %5, %5, %14;\n\taddc.cc.u32 %6, %6, %15;\n\taddc.cc.u32 %7, %7, %16;\n\taddc.u32 %8, %8, %17;" \
            : "+r"(L[0]), "+r"(L[1]), "+r"(L[2]), "+r"(L[3]), "+r"(L[4]), "+r"(L[5]), "+r"(L[6]), "+r"(L[7]), "+r"(L[8]) \
            : "r"(t[0]), "r"(t[1]), "r"(t[2]), "r"(t[3]), "r"(t[4]), "r"(t[5]), "r"(t[6]), "r"(t[7]), "r"(t[8])); \
    } while (0)
    PX_ADDSHIFTED(5); PX_ADDSHIFTED(2); PX_ADDSHIFTED(1);
#undef PX_ADDSHIFTED
    for (int i = 0; i < 8; i++) h.v[i] = L[i];
    fe_addtop_t<NB>(h, fe_mul38small(L[8]), 0u);   // L[8] <= 40
    } else {
    // fold row: 38 * hi into words 0..8
    asm("mad.lo.cc.u32 %0, %17, 38, %0;\n\tmadc.hi.cc.u32 %1, %17, 38, %1;\n\tmadc.lo.cc.u32 %2, %19, 38, %2;\n\tmadc.hi.cc.u32 %3, %19, 38, %3;\n\tmadc.lo.cc.u32 %4, %21, 38, %4;\n\tmadc.hi.cc.u32 %5, %21, 38, %5;\n\tmadc.lo.cc.u32 %6, %23, 38, %6;\n\tmadc.hi.cc.u32 %7, %23, 38, %7;\n\taddc.u32 %8, 0, 0;\n\tmad.lo.cc.u32 %9, %18, 38, %9;\n\tmadc.hi.cc.u32 %10, %18, 38, %10;\n\tmadc.lo.cc.u32 %11, %20, 38, %11;\n\tmadc.hi.cc.u32 %12, %20, 38, %12;\n\tmadc.lo.cc.u32 %13, %22, 38, %13;\n\tmadc.hi.cc.u32 %14, %22, 38, %14;\n\tmadc.lo.cc.u32 %15, %24, 38, %15;\n\tmadc.hi.u32 %16, %24, 38, 0;"
        : "+r"(E[0]), "+r"(E[1]), "+r"(E[2]), "+r"(E[3]), "+r"(E[4]), "+r"(E[5]), "+r"(E[6]), "+r"(E[7]), "+r"(E[8]), "+r"(Od[0]), "+r"(Od[1]), "+r"(Od[2]), "+r"(Od[3]), "+r"(Od[4]), "+r"(Od[5]), "+r"(Od[6]), "+r"(Od[7])
        : "r"(hi[0]), "r"(hi[1]), "r"(hi[2]), "r"(hi[3]), "r"(hi[4]), "r"(hi[5]), "r"(hi[6]), "r"(hi[7]));
    // combine: h[w] = E[w] + Od[w-1]; the top word t8 (< 41) and the 2^512 carry fold into word 0
    uint32_t o[8], t8;
    asm("add.cc.u32 %0, %10, %17;\n\taddc.cc.u32 %1, %11, %18;\n\taddc.cc.u32 %2, %12, %19;\n\taddc.cc.u32 %3, %13, %20;\n\t"
        "addc.cc.u32 %4, %14, %21;\n\taddc.cc.u32 %5, %15, %22;\n\taddc.cc.u32 %6, %16, %23;\n\taddc.u32 %7, %9, %24;"
        : "=&r"(o[1]), "=&r"(o[2]), "=&r"(o[3]), "=&r"(o[4]), "=&r"(o[5]), "=&r"(o[6]), "=&r"(o[7]), "=&r"(t8)
        : "r"(E[0]), "r"(E[8]), "r"(E[1]), "r"(E[2]), "r"(E[3]), "r"(E[4]), "r"(E[5]), "r"(E[6]), "r"(E[7]),
          "r"(Od[0]), "r"(Od[1]), "r"(Od[2]), "r"(Od[3]), "r"(Od[4]), "r"(Od[5]), "r"(Od[6]), "r"(Od[7]));
    o[0] = E[0];
    for (int i = 0; i < 8; i++) h.v[i] = o[i];
    fe_addtop_t<NB>(h, fe_mul38small(t8), 0u);
    }
}
__device__ __forceinline__ void fe_mul(fe &h, const fe &f, const fe &g) { fe_mul_t<0>(h, f, g); }
#if PX_LAD_KARA
// ----------------------------------------------------------------------------
// PX_LAD_KARA: one level of (subtractive) Karatsuba for the ladder multiplications. The ladder is bound by the
// multiplier pipe (IMAD.WIDE) while the integer pipe is mostly idle; three 4x4 products (48 wide multiplies)
// instead of one 8x8 (64) trade 16 multiplies for about 60 integer additions per field multiplication.
//   a = aL + 2^128 aH, b likewise; m = |aL - aH| |bL - bH|;
//   a b = zL + (zL + zH -+ m) 2^128 + zH 2^256, zL = aL bL, zH = aH bH (minus when the differences have equal signs)
// ----------------------------------------------------------------------------
__device__ __forceinline__ void mul4w(uint32_t r[8], const uint32_t *a, const uint32_t *b) {
    uint32_t E[8], Od[8];
    { uint64_t p;
      p = (uint64_t)a[0] * b[0]; E[0] = (uint32_t)p; E[1] = (uint32_t)(p >> 32);
      p = (uint64_t)a[0] * b[2]; E[2] = (uint32_t)p; E[3] = (uint32_t)(p >> 32);
      p = (uint64_t)a[0] * b[1]; Od[0] = (uint32_t)p; Od[1] = (uint32_t)(p >> 32);
      p = (uint64_t)a[0] * b[3]; Od[2] = (uint32_t)p; Od[3] = (uint32_t)(p >> 32); }
    E[4] = E[5] = E[6] = E[7] = 0; Od[4] = Od[5] = Od[6] = Od[7] = 0;
    // row 1: even columns a1*b1 (2), a1*b3 (4); odd columns a1*b0 (1), a1*b2 (3)
    asm("mad.lo.cc.u32 %0, %10, %12, %0;\n\tmadc.hi.cc.u32 %1, %10, %12, %1;\n\tmadc.lo.cc.u32 %2, %10, %14, %2;\n\tmadc.hi.cc.u32 %3, %10, %14, %3;\n\taddc.u32 %4, %4, 0;\n\t"
        "mad.lo.cc.u32 %5, %10, %11, %5;\n\tmadc.hi.cc.u32 %6, %10, %11, %6;\n\tmadc.lo.cc.u32 %7, %10, %13, %7;\n\tmadc.hi.cc.u32 %8, %10, %13, %8;\n\taddc.u32 %9, %9, 0;"
        : "+r"(E[2]), "+r"(E[3]), "+r"(E[4]), "+r"(E[5]), "+r"(E[6]), "+r"(Od[0]), "+r"(Od[1]), "+r"(Od[2]), "+r"(Od[3]), "+r"(Od[4])
        : "r"(a[1]), "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]));
    // row 2: even a2*b0 (2), a2*b2 (4); odd a2*b1 (3), a2*b3 (5)
    asm("mad.lo.cc.u32 %0, %10, %11, %0;\n\tmadc.hi.cc.u32 %1, %10, %11, %1;\n\tmadc.lo.cc.u32 %2, %10, %13, %2;\n\tmadc.hi.cc.u32 %3, %10, %13, %3;\n\taddc.u32 %4, %4, 0;\n\t"
        "mad.lo.cc.u32 %5, %10, %12, %5;\n\tmadc.hi.cc.u32 %6, %10, %12, %6;\n\tmadc.lo.cc.u32 %7, %10, %14, %7;\n\tmadc.hi.cc.u32 %8, %10, %14, %8;\n\taddc.u32 %9, %9, 0;"
        : "+r"(E[2]), "+r"(E[3]), "+r"(E[4]), "+r"(E[5]), "+r"(E[6]), "+r"(Od[2]), "+r"(Od[3]), "+r"(Od[4]), "+r"(Od[5]), "+r"(Od[6])
        : "r"(a[2]), "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]));
    // row 3: even a3*b1 (4), a3*b3 (6); odd a3*b0 (3), a3*b2 (5)
    asm("mad.lo.cc.u32 %0, %9, %11, %0;\n\tmadc.hi.cc.u32 %1, %9, %11, %1;\n\tmadc.lo.cc.u32 %2, %9, %13, %2;\n\tmadc.hi.u32 %3, %9, %13, %3;\n\t"
        "mad.lo.cc.u32 %4, %9, %10, %4;\n\tmadc.hi.cc.u32 %5, %9, %10, %5;\n\tmadc.lo.cc.u32 %6, %9, %12, %6;\n\tmadc.hi.cc.u32 %7, %9, %12, %7;\n\taddc.u32 %8, %8, 0;"
        : "+r"(E[4]), "+r"(E[5]), "+r"(E[6]), "+r"(E[7]), "+r"(Od[2]), "+r"(Od[3]), "+r"(Od[4]), "+r"(Od[5]), "+r"(Od[6])
        : "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]));
    r[0] = E[0];
    asm("add.cc.u32 %0, %7, %14;\n\taddc.cc.u32 %1, %8, %15;\n\taddc.cc.u32 %2, %9, %16;\n\taddc.cc.u32 %3, %10, %17;\n\t"
        "addc.cc.u32 %4, %11, %18;\n\taddc.cc.u32 %5, %12, %19;\n\taddc.u32 %6, %13, %20;"
        : "=&r"(r[1]), "=&r"(r[2]), "=&r"(r[3]), "=&r"(r[4]), "=&r"(r[5]), "=&r"(r[6]), "=&r"(r[7])
        : "r"(E[1]), "r"(E[2]), "r"(E[3]), "r"(E[4]), "r"(E[5]), "r"(E[6]), "r"(E[7]),
          "r"(Od[0]), "r"(Od[1]), "r"(Od[2]), "r"(Od[3]), "r"(Od[4]), "r"(Od[5]), "r"(Od[6]));
}
// d = |x - y| over four words; returns all ones when x < y
__device__ __forceinline__ uint32_t absdiff4(uint32_t d[4], const uint32_t *x, const uint32_t *y) {
    uint32_t s;
    asm("sub.cc.u32 %0, %5, %9;\n\tsubc.cc.u32 %1, %6, %10;\n\tsubc.cc.u32 %2, %7, %11;\n\tsubc.cc.u32 %3, %8, %12;\n\tsubc.u32 %4, 0, 0;"
        : "=&r"(d[0]), "=&r"(d[1]), "=&r"(d[2]), "=&r"(d[3]), "=&r"(s)
        : "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]), "r"(y[0]), "r"(y[1]), "r"(y[2]), "r"(y[3]));
    #pragma unroll
    for (int i = 0; i < 4; i++) d[i] ^= s;
    asm("sub.cc.u32 %0, %0, %4;\n\tsubc.cc.u32 %1, %1, %4;\n\tsubc.cc.u32 %2, %2, %4;\n\tsubc.u32 %3, %3, %4;"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : "r"(s));
    return s;
}
template<int NB> __device__ __forceinline__ void fe_mul_kara(fe &h, const fe &f, const fe &g) {
    const uint32_t *a = f.v, *b = g.v;
    uint32_t zL[8], zH[8], m[8], dA[4], dB[4];
    mul4w(zL, a, b);
    mul4w(zH, a + 4, b + 4);
    uint32_t sA = absdiff4(dA, a, a + 4), sB = absdiff4(dB, b, b + 4);
    mul4w(m, dA, dB);
    // n = all ones when the middle term is zL + zH - m (equal signs), zero when it is zL + zH + m
    uint32_t n = ~(sA ^ sB);
    uint32_t M[9];
    asm("add.cc.u32 %0, %9, %17;\n\taddc.cc.u32 %1, %10, %18;\n\taddc.cc.u32 %2, %11, %19;\n\taddc.cc.u32 %3, %12, %20;\n\t"
        "addc.cc.u32 %4, %13, %21;\n\taddc.cc.u32 %5, %14, %22;\n\taddc.cc.u32 %6, %15, %23;\n\taddc.cc.u32 %7, %16, %24;\n\taddc.u32 %8, 0, 0;"
        : "=&r"(M[0]), "=&r"(M[1]), "=&r"(M[2]), "=&r"(M[3]), "=&r"(M[4]), "=&r"(M[5]), "=&r"(M[6]), "=&r"(M[7]), "=&r"(M[8])
        : "r"(zL[0]), "r"(zL[1]), "r"(zL[2]), "r"(zL[3]), "r"(zL[4]), "r"(zL[5]), "r"(zL[6]), "r"(zL[7]),
          "r"(zH[0]), "r"(zH[1]), "r"(zH[2]), "r"(zH[3]), "r"(zH[4]), "r"(zH[5]), "r"(zH[6]), "r"(zH[7]));
    // M += (m ^ n) - n over nine words (m[8] = 0): + m or - m, exact (the middle term is non-negative)
    uint32_t t[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) t[i] = m[i] ^ n;
    asm("sub.cc.u32 %0, %0, %8;\n\tsubc.cc.u32 %1, %1, %8;\n\tsubc.cc.u32 %2, %2, %8;\n\tsubc.cc.u32 %3, %3, %8;\n\t"
        "subc.cc.u32 %4, %4, %8;\n\tsubc.cc.u32 %5, %5, %8;\n\tsubc.cc.u32 %6, %6, %8;\n\tsubc.u32 %7, %7, %8;"
        : "+r"(t[0]), "+r"(t[1]), "+r"(t[2]), "+r"(t[3]), "+r"(t[4]), "+r"(t[5]), "+r"(t[6]), "+r"(t[7]) : "r"(n));
    // the ninth word of (m ^ n) - n is n - n - borrow; folded as: M[8] += n + carry chain below handles it exactly
    {   uint32_t t8;
        // recompute the borrow-out of the subtraction above: (m ^ n) - n borrows iff n = all ones and m != 0 ... handled
        // directly: ninth word of -(m) is all ones unless m = 0; of +m is zero
        uint32_t mz = m[0] | m[1] | m[2] | m[3] | m[4] | m[5] | m[6] | m[7];
        t8 = n & (mz ? 0xFFFFFFFFu : 0u);
        asm("add.cc.u32 %0, %0, %9;\n\taddc.cc.u32 %1, %1, %10;\n\taddc.cc.u32 %2, %2, %11;\n\taddc.cc.u32 %3, %3, %12;\n\t"
            "addc.cc.u32 %4, %4, %13;\n\taddc.cc.u32 %5, %5, %14;\n\taddc.cc.u32 %6, %6, %15;\n\taddc.cc.u32 %7, %7, %16;\n\taddc.u32 %8, %8, %17;"
            : "+r"(M[0]), "+r"(M[1]), "+r"(M[2]), "+r"(M[3]), "+r"(M[4]), "+r"(M[5]), "+r"(M[6]), "+r"(M[7]), "+r"(M[8])
            : "r"(t[0]), "r"(t[1]), "r"(t[2]), "r"(t[3]), "r"(t[4]), "r"(t[5]), "r"(t[6]), "r"(t[7]), "r"(t8)); }
    // P = zL + M 2^128 + zH 2^256 (16 words)
    uint32_t P[16];
    P[0] = zL[0]; P[1] = zL[1]; P[2] = zL[2]; P[3] = zL[3];
    asm("add.cc.u32 %0, %12, %20;\n\taddc.cc.u32 %1, %13, %21;\n\taddc.cc.u32 %2, %14, %22;\n\taddc.cc.u32 %3, %15, %23;\n\t"
        "addc.cc.u32 %4, %16, %24;\n\taddc.cc.u32 %5, %17, %25;\n\taddc.cc.u32 %6, %18, %26;\n\taddc.cc.u32 %7, %19, %27;\n\t"
        "addc.cc.u32 %8, %28, %32;\n\taddc.cc.u32 %9, %29, 0;\n\taddc.cc.u32 %10, %30, 0;\n\taddc.u32 %11, %31, 0;"
        : "=&r"(P[4]), "=&r"(P[5]), "=&r"(P[6]), "=&r"(P[7]), "=&r"(P[8]), "=&r"(P[9]), "=&r"(P[10]), "=&r"(P[11]),
          "=&r"(P[12]), "=&r"(P[13]), "=&r"(P[14]), "=&r"(P[15])
        : "r"(zL[4]), "r"(zL[5]), "r"(zL[6]), "r"(zL[7]), "r"(zH[0]), "r"(zH[1]), "r"(zH[2]), "r"(zH[3]),
          "r"(M[0]), "r"(M[1]), "r"(M[2]), "r"(M[3]), "r"(M[4]), "r"(M[5]), "r"(M[6]), "r"(M[7]),
          "r"(zH[4]), "r"(zH[5]), "r"(zH[6]), "r"(zH[7]), "r"(M[8]));
    // fold 2^256 = 38: even columns as one chain into P[0..7], odd products independent, top word t8
    uint32_t Od2[8], t8;
    asm("mad.lo.cc.u32 %0, %9, 38, %0;\n\tmadc.hi.cc.u32 %1, %9, 38, %1;\n\tmadc.lo.cc.u32 %2, %10, 38, %2;\n\tmadc.hi.cc.u32 %3, %10, 38, %3;\n\t"
        "madc.lo.cc.u32 %4, %11, 38, %4;\n\tmadc.hi.cc.u32 %5, %11, 38, %5;\n\tmadc.lo.cc.u32 %6, %12, 38, %6;\n\tmadc.hi.cc.u32 %7, %12, 38, %7;\n\taddc.u32 %8, 0, 0;"
        : "+r"(P[0]), "+r"(P[1]), "+r"(P[2]), "+r"(P[3]), "+r"(P[4]), "+r"(P[5]), "+r"(P[6]), "+r"(P[7]), "=&r"(t8)
        : "r"(P[8]), "r"(P[10]), "r"(P[12]), "r"(P[14]));
    #pragma unroll
    for (int c = 0; c < 4; c++) { uint64_t q = (uint64_t)P[9 + 2 * c] * 38u; Od2[2 * c] = (uint32_t)q; Od2[2 * c + 1] = (uint32_t)(q >> 32); }
    uint32_t o[8];
    asm("add.cc.u32 %0, %9, %16;\n\taddc.cc.u32 %1, %10, %17;\n\taddc.cc.u32 %2, %11, %18;\n\taddc.cc.u32 %3, %12, %19;\n\t"
        "addc.cc.u32 %4, %13, %20;\n\taddc.cc.u32 %5, %14, %21;\n\taddc.cc.u32 %6, %15, %22;\n\taddc.u32 %7, %8, %23;"
        : "=&r"(o[1]), "=&r"(o[2]), "=&r"(o[3]), "=&r"(o[4]), "=&r"(o[5]), "=&r"(o[6]), "=&r"(o[7]), "=&r"(t8)
        : "r"(t8), "r"(P[1]), "r"(P[2]), "r"(P[3]), "r"(P[4]), "r"(P[5]), "r"(P[6]), "r"(P[7]),
          "r"(Od2[0]), "r"(Od2[1]), "r"(Od2[2]), "r"(Od2[3]), "r"(Od2[4]), "r"(Od2[5]), "r"(Od2[6]), "r"(Od2[7]));
    o[0] = P[0];
    #pragma unroll
    for (int i = 0; i < 8; i++) h.v[i] = o[i];
    fe_addtop_t<NB>(h, fe_mul38small(t8), 0u);
}
#endif
template<int FS, int NB = 0> __device__ void fe_sq_impl(fe &h, const fe &f) {
    const uint32_t *a = f.v;
    uint32_t E[16], Od[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) { E[i] = 0; Od[i] = 0; }
#if PX_ROW0C
    // sq row 0 into zero accumulators: a0*a1..a7 as independent wide products
    {   uint64_t p;
        p = (uint64_t)a[0] * a[2]; E[2] = (uint32_t)p; E[3] = (uint32_t)(p >> 32);
        p = (uint64_t)a[0] * a[4]; E[4] = (uint32_t)p; E[5] = (uint32_t)(p >> 32);
        p = (uint64_t)a[0] * a[6]; E[6] = (uint32_t)p; E[7] = (uint32_t)(p >> 32);
        p = (uint64_t)a[0] * a[1]; Od[0] = (uint32_t)p; Od[1] = (uint32_t)(p >> 32);
        p = (uint64_t)a[0] * a[3]; Od[2] = (uint32_t)p; Od[3] = (uint32_t)(p >> 32);
        p = (uint64_t)a[0] * a[5]; Od[4] = (uint32_t)p; Od[5] = (uint32_t)(p >> 32);
        p = (uint64_t)a[0] * a[7]; Od[6] = (uint32_t)p; Od[7] = (uint32_t)(p >> 32);
    }
#else
    // sq row 0
    asm("mad.lo.cc.u32 %0, %15, %17, %0;\n\tmadc.hi.cc.u32 %1, %15, %17, %1;\n\tmadc.lo.cc.u32 %2, %15, %19, %2;\n\tmadc.hi.cc.u32 %3, %15, %19, %3;\n\tmadc.lo.cc.u32 %4, %15, %21, %4;\n\tmadc.hi.cc.u32 %5, %15, %21, %5;\n\taddc.u32 %6, %6, 0;\n\tmad.lo.cc.u32 %7, %15, %16, %7;\n\tmadc.hi.cc.u32 %8, %15, %16, %8;\n\tmadc.lo.cc.u32 %9, %15, %18, %9;\n\tmadc.hi.cc.u32 %10, %15, %18, %10;\n\tmadc.lo.cc.u32 %11, %15, %20, %11;\n\tmadc.hi.cc.u32 %12, %15, %20, %12;\n\tmadc.lo.cc.u32 %13, %15, %22, %13;\n\tmadc.hi.u32 %14, %15, %22, %14;"
        : "+r"(E[2]), "+r"(E[3]), "+r"(E[4]), "+r"(E[5]), "+r"(E[6]), "+r"(E[7]), "+r"(E[8]), "+r"(Od[0]), "+r"(Od[1]), "+r"(Od[2]), "+r"(Od[3]), "+r"(Od[4]), "+r"(Od[5]), "+r"(Od[6]), "+r"(Od[7])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]));
#endif
    // sq row 1
#if PX_ZFREE
    asm("mad.lo.cc.u32 %0, %13, %15, %0;\n\tmadc.hi.cc.u32 %1, %13, %15, %1;\n\tmadc.lo.cc.u32 %2, %13, %17, %2;\n\tmadc.hi.cc.u32 %3, %13, %17, %3;\n\tmadc.lo.cc.u32 %4, %13, %19, 0;\n\tmadc.hi.u32 %5, %13, %19, 0;\n\tmad.lo.cc.u32 %6, %13, %14, %6;\n\tmadc.hi.cc.u32 %7, %13, %14, %7;\n\tmadc.lo.cc.u32 %8, %13, %16, %8;\n\tmadc.hi.cc.u32 %9, %13, %16, %9;\n\tmadc.lo.cc.u32 %10, %13, %18, %10;\n\tmadc.hi.cc.u32 %11, %13, %18, %11;\n\taddc.u32 %12, 0, 0;"
        : "+r"(E[4]), "+r"(E[5]), "+r"(E[6]), "+r"(E[7]), "=&r"(E[8]), "=&r"(E[9]), "+r"(Od[2]), "+r"(Od[3]), "+r"(Od[4]), "+r"(Od[5]), "+r"(Od[6]), "+r"(Od[7]), "=&r"(Od[8])
        : "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]));
#else
    asm("mad.lo.cc.u32 %0, %13, %15, %0;\n\tmadc.hi.cc.u32 %1, %13, %15, %1;\n\tmadc.lo.cc.u32 %2, %13, %17, %2;\n\tmadc.hi.cc.u32 %3, %13, %17, %3;\n\tmadc.lo.cc.u32 %4, %13, %19, %4;\n\tmadc.hi.u32 %5, %13, %19, %5;\n\tmad.lo.cc.u32 %6, %13, %14, %6;\n\tmadc.hi.cc.u32 %7, %13, %14, %7;\n\tmadc.lo.cc.u32 %8, %13, %16, %8;\n\tmadc.hi.cc.u32 %9, %13, %16, %9;\n\tmadc.lo.cc.u32 %10, %13, %18, %10;\n\tmadc.hi.cc.u32 %11, %13, %18, %11;\n\taddc.u32 %12, %12, 0;"
        : "+r"(E[4]), "+r"(E[5]), "+r"(E[6]), "+r"(E[7]), "+r"(E[8]), "+r"(E[9]), "+r"(Od[2]), "+r"(Od[3]), "+r"(Od[4]), "+r"(Od[5]), "+r"(Od[6]), "+r"(Od[7]), "+r"(Od[8])
        : "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]));
#endif
    // sq row 2
#if PX_ZFREE
    asm("mad.lo.cc.u32 %0, %11, %13, %0;\n\tmadc.hi.cc.u32 %1, %11, %13, %1;\n\tmadc.lo.cc.u32 %2, %11, %15, %2;\n\tmadc.hi.cc.u32 %3, %11, %15, %3;\n\taddc.u32 %4, 0, 0;\n\tmad.lo.cc.u32 %5, %11, %12, %5;\n\tmadc.hi.cc.u32 %6, %11, %12, %6;\n\tmadc.lo.cc.u32 %7, %11, %14, %7;\n\tmadc.hi.cc.u32 %8, %11, %14, %8;\n\tmadc.lo.cc.u32 %9, %11, %16, %9;\n\tmadc.hi.u32 %10, %11, %16, 0;"
        : "+r"(E[6]), "+r"(E[7]), "+r"(E[8]), "+r"(E[9]), "=&r"(E[10]), "+r"(Od[4]), "+r"(Od[5]), "+r"(Od[6]), "+r"(Od[7]), "+r"(Od[8]), "=&r"(Od[9])
        : "r"(a[2]), "r"(a[3]), "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]));
#else
    asm("mad.lo.cc.u32 %0, %11, %13, %0;\n\tmadc.hi.cc.u32 %1, %11, %13, %1;\n\tmadc.lo.cc.u32 %2, %11, %15, %2;\n\tmadc.hi.cc.u32 %3, %11, %15, %3;\n\taddc.u32 %4, %4, 0;\n\tmad.lo.cc.u32 %5, %11, %12, %5;\n\tmadc.hi.cc.u32 %6, %11, %12, %6;\n\tmadc.lo.cc.u32 %7, %11, %14, %7;\n\tmadc.hi.cc.u32 %8, %11, %14, %8;\n\tmadc.lo.cc.u32 %9, %11, %16, %9;\n\tmadc.hi.u32 %10, %11, %16, %10;"
        : "+r"(E[6]), "+r"(E[7]), "+r"(E[8]), "+r"(E[9]), "+r"(E[10]), "+r"(Od[4]), "+r"(Od[5]), "+r"(Od[6]), "+r"(Od[7]), "+r"(Od[8]), "+r"(Od[9])
        : "r"(a[2]), "r"(a[3]), "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]));
#endif
    // sq row 3
#if PX_ZFREE
    asm("mad.lo.cc.u32 %0, %9, %11, %0;\n\tmadc.hi.cc.u32 %1, %9, %11, %1;\n\tmadc.lo.cc.u32 %2, %9, %13, %2;\n\tmadc.hi.u32 %3, %9, %13, 0;\n\tmad.lo.cc.u32 %4, %9, %10, %4;\n\tmadc.hi.cc.u32 %5, %9, %10, %5;\n\tmadc.lo.cc.u32 %6, %9, %12, %6;\n\tmadc.hi.cc.u32 %7, %9, %12, %7;\n\taddc.u32 %8, 0, 0;"
        : "+r"(E[8]), "+r"(E[9]), "+r"(E[10]), "=&r"(E[11]), "+r"(Od[6]), "+r"(Od[7]), "+r"(Od[8]), "+r"(Od[9]), "=&r"(Od[10])
        : "r"(a[3]), "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]));
#else
    asm("mad.lo.cc.u32 %0, %9, %11, %0;\n\tmadc.hi.cc.u32 %1, %9, %11, %1;\n\tmadc.lo.cc.u32 %2, %9, %13, %2;\n\tmadc.hi.u32 %3, %9, %13, %3;\n\tmad.lo.cc.u32 %4, %9, %10, %4;\n\tmadc.hi.cc.u32 %5, %9, %10, %5;\n\tmadc.lo.cc.u32 %6, %9, %12, %6;\n\tmadc.hi.cc.u32 %7, %9, %12, %7;\n\taddc.u32 %8, %8, 0;"
        : "+r"(E[8]), "+r"(E[9]), "+r"(E[10]), "+r"(E[11]), "+r"(Od[6]), "+r"(Od[7]), "+r"(Od[8]), "+r"(Od[9]), "+r"(Od[10])
        : "r"(a[3]), "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]));
#endif
    // sq row 4
#if PX_ZFREE
    asm("mad.lo.cc.u32 %0, %7, %9, %0;\n\tmadc.hi.cc.u32 %1, %7, %9, %1;\n\taddc.u32 %2, 0, 0;\n\tmad.lo.cc.u32 %3, %7, %8, %3;\n\tmadc.hi.cc.u32 %4, %7, %8, %4;\n\tmadc.lo.cc.u32 %5, %7, %10, %5;\n\tmadc.hi.u32 %6, %7, %10, 0;"
        : "+r"(E[10]), "+r"(E[11]), "=&r"(E[12]), "+r"(Od[8]), "+r"(Od[9]), "+r"(Od[10]), "=&r"(Od[11])
        : "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]));
#else
    asm("mad.lo.cc.u32 %0, %7, %9, %0;\n\tmadc.hi.cc.u32 %1, %7, %9, %1;\n\taddc.u32 %2, %2, 0;\n\tmad.lo.cc.u32 %3, %7, %8, %3;\n\tmadc.hi.cc.u32 %4, %7, %8, %4;\n\tmadc.lo.cc.u32 %5, %7, %10, %5;\n\tmadc.hi.u32 %6, %7, %10, %6;"
        : "+r"(E[10]), "+r"(E[11]), "+r"(E[12]), "+r"(Od[8]), "+r"(Od[9]), "+r"(Od[10]), "+r"(Od[11])
        : "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]));
#endif
    // sq row 5
#if PX_ZFREE
    asm("mad.lo.cc.u32 %0, %5, %7, %0;\n\tmadc.hi.u32 %1, %5, %7, 0;\n\tmad.lo.cc.u32 %2, %5, %6, %2;\n\tmadc.hi.cc.u32 %3, %5, %6, %3;\n\taddc.u32 %4, 0, 0;"
        : "+r"(E[12]), "=&r"(E[13]), "+r"(Od[10]), "+r"(Od[11]), "=&r"(Od[12])
        : "r"(a[5]), "r"(a[6]), "r"(a[7]));
#else
    asm("mad.lo.cc.u32 %0, %5, %7, %0;\n\tmadc.hi.u32 %1, %5, %7, %1;\n\tmad.lo.cc.u32 %2, %5, %6, %2;\n\tmadc.hi.cc.u32 %3, %5, %6, %3;\n\taddc.u32 %4, %4, 0;"
        : "+r"(E[12]), "+r"(E[13]), "+r"(Od[10]), "+r"(Od[11]), "+r"(Od[12])
        : "r"(a[5]), "r"(a[6]), "r"(a[7]));
#endif
    // sq row 6
#if PX_ZFREE
    asm("mad.lo.cc.u32 %0, %2, %3, %0;\n\tmadc.hi.u32 %1, %2, %3, 0;"
        : "+r"(Od[12]), "=&r"(Od[13])
        : "r"(a[6]), "r"(a[7]));
#else
    asm("mad.lo.cc.u32 %0, %2, %3, %0;\n\tmadc.hi.u32 %1, %2, %3, %1;"
        : "+r"(Od[12]), "+r"(Od[13])
        : "r"(a[6]), "r"(a[7]));
#endif
    // cross sum s = E + Od<<32 (< 2^480), doubled by a funnel shift
    uint32_t s[16];
    s[0] = E[0];
    asm("add.cc.u32 %0, %15, %29;\n\taddc.cc.u32 %1, %16, %30;\n\taddc.cc.u32 %2, %17, %31;\n\taddc.cc.u32 %3, %18, %32;\n\t"
        "addc.cc.u32 %4, %19, %33;\n\taddc.cc.u32 %5, %20, %34;\n\taddc.cc.u32 %6, %21, %35;\n\taddc.cc.u32 %7, %22, %36;\n\t"
        "addc.cc.u32 %8, %23, %37;\n\taddc.cc.u32 %9, %24, %38;\n\taddc.cc.u32 %10, %25, %39;\n\taddc.cc.u32 %11, %26, %40;\n\t"
        "addc.cc.u32 %12, %27, %41;\n\taddc.cc.u32 %13, %28, %42;\n\taddc.u32 %14, 0, %43;"
        : "=&r"(s[1]), "=&r"(s[2]), "=&r"(s[3]), "=&r"(s[4]), "=&r"(s[5]), "=&r"(s[6]), "=&r"(s[7]), "=&r"(s[8]), "=&r"(s[9]), "=&r"(s[10]), "=&r"(s[11]), "=&r"(s[12]), "=&r"(s[13]), "=&r"(s[14]), "=&r"(s[15])
        : "r"(E[1]), "r"(E[2]), "r"(E[3]), "r"(E[4]), "r"(E[5]), "r"(E[6]), "r"(E[7]), "r"(E[8]), "r"(E[9]), "r"(E[10]), "r"(E[11]), "r"(E[12]), "r"(E[13]), "r"(E[14]),
          "r"(Od[0]), "r"(Od[1]), "r"(Od[2]), "r"(Od[3]), "r"(Od[4]), "r"(Od[5]), "r"(Od[6]), "r"(Od[7]), "r"(Od[8]), "r"(Od[9]), "r"(Od[10]), "r"(Od[11]), "r"(Od[12]), "r"(Od[13]), "r"(Od[14]));
    #pragma unroll
    for (int k = 15; k > 0; k--) s[k] = (s[k] << 1) | (s[k - 1] >> 31);
    s[0] <<= 1;
    // the 8 squares at columns 0,2,..,14: one chain of aligned windows
    asm("mad.lo.cc.u32 %0, %16, %16, 0;\n\tmadc.hi.cc.u32 %1, %16, %16, %1;\n\t"
        "madc.lo.cc.u32 %2, %17, %17, %2;\n\tmadc.hi.cc.u32 %3, %17, %17, %3;\n\t"
        "madc.lo.cc.u32 %4, %18, %18, %4;\n\tmadc.hi.cc.u32 %5, %18, %18, %5;\n\t"
        "madc.lo.cc.u32 %6, %19, %19, %6;\n\tmadc.hi.cc.u32 %7, %19, %19, %7;\n\t"
        "madc.lo.cc.u32 %8, %20, %20, %8;\n\tmadc.hi.cc.u32 %9, %20, %20, %9;\n\t"
        "madc.lo.cc.u32 %10, %21, %21, %10;\n\tmadc.hi.cc.u32 %11, %21, %21, %11;\n\t"
        "madc.lo.cc.u32 %12, %22, %22, %12;\n\tmadc.hi.cc.u32 %13, %22, %22, %13;\n\t"
        "madc.lo.cc.u32 %14, %23, %23, %14;\n\tmadc.hi.u32 %15, %23, %23, %15;"
        : "=&r"(s[0]), "+r"(s[1]), "+r"(s[2]), "+r"(s[3]), "+r"(s[4]), "+r"(s[5]), "+r"(s[6]), "+r"(s[7]), "+r"(s[8]), "+r"(s[9]), "+r"(s[10]), "+r"(s[11]), "+r"(s[12]), "+r"(s[13]), "+r"(s[14]), "+r"(s[15])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]));
    if (FS) {
    // 2^256 = 38 by funnel shifts on the integer pipe (see fe_mul): h = s[0..7] + s[8..15] * (32 + 4 + 2)
    { uint32_t L[9]; const uint32_t *hi = s + 8;
      #pragma unroll
      for (int i = 0; i < 8; i++) L[i] = s[i];
      L[8] = 0;
#define PX_ADDSHIFTED(sh) do { \
        uint32_t t[9]; \
        t[0] = hi[0] << (sh); \
        _Pragma("unroll") for (int k = 1; k < 8; k++) t[k] = __funnelshift_l(hi[k - 1], hi[k], (sh)); \
        t[8] = hi[7] >> (32 - (sh)); \
        asm("add.cc.u32 %0, %0, %9;\n\taddc.cc.u32 %1, %1, %10;\n\taddc.cc.u32 %2, %2, %11;\n\taddc.cc.u32 %3, %3, %12;\n\t" \
            "addc.cc.u32 %4, %4, %13;\n\taddc.cc.u32 %5, %5, %14;\n\taddc.cc.u32 %6, %6, %15;\n\taddc.cc.u32 %7, %7, %16;\n\taddc.u32 %8, %8, %17;" \
            : "+r"(L[0]), "+r"(L[1]), "+r"(L[2]), "+r"(L[3]), "+r"(L[4]), "+r"(L[5]), "+r"(L[6]), "+r"(L[7]), "+r"(L[8]) \
            : "r"(t[0]), "r"(t[1]), "r"(t[2]), "r"(t[3]), "r"(t[4]), "r"(t[5]), "r"(t[6]), "r"(t[7]), "r"(t[8])); \
    } while (0)
      PX_ADDSHIFTED(5); PX_ADDSHIFTED(2); PX_ADDSHIFTED(1);
#undef PX_ADDSHIFTED
      for (int i = 0; i < 8; i++) h.v[i] = L[i];
      fe_addtop_t<NB>(h, fe_mul38small(L[8]), 0u); }
    } else {
    // fold 38 * s[8..15] into s[0..7]: even columns into s itself (aligned), odd columns into Od2, top word t8
    uint32_t Od2[8], t8;
#if PX_ROW0C
    // even columns as one carry chain into s; the odd products have zero addends, so their chain never carries:
    // four independent wide products
    asm("mad.lo.cc.u32 %0, %9, 38, %0;\n\tmadc.hi.cc.u32 %1, %9, 38, %1;\n\tmadc.lo.cc.u32 %2, %10, 38, %2;\n\tmadc.hi.cc.u32 %3, %10, 38, %3;\n\t"
        "madc.lo.cc.u32 %4, %11, 38, %4;\n\tmadc.hi.cc.u32 %5, %11, 38, %5;\n\tmadc.lo.cc.u32 %6, %12, 38, %6;\n\tmadc.hi.cc.u32 %7, %12, 38, %7;\n\taddc.u32 %8, 0, 0;"
        : "+r"(s[0]), "+r"(s[1]), "+r"(s[2]), "+r"(s[3]), "+r"(s[4]), "+r"(s[5]), "+r"(s[6]), "+r"(s[7]), "=&r"(t8)
        : "r"(s[8]), "r"(s[10]), "r"(s[12]), "r"(s[14]));
    #pragma unroll
    for (int c = 0; c < 4; c++) { uint64_t q = (uint64_t)s[9 + 2 * c] * 38u; Od2[2 * c] = (uint32_t)q; Od2[2 * c + 1] = (uint32_t)(q >> 32); }
#else
    asm("mad.lo.cc.u32 %0, %17, 38, %0;\n\tmadc.hi.cc.u32 %1, %17, 38, %1;\n\tmadc.lo.cc.u32 %2, %19, 38, %2;\n\tmadc.hi.cc.u32 %3, %19, 38, %3;\n\t"
        "madc.lo.cc.u32 %4, %21, 38, %4;\n\tmadc.hi.cc.u32 %5, %21, 38, %5;\n\tmadc.lo.cc.u32 %6, %23, 38, %6;\n\tmadc.hi.cc.u32 %7, %23, 38, %7;\n\taddc.u32 %8, 0, 0;\n\t"
        "mad.lo.cc.u32 %9, %18, 38, 0;\n\tmadc.hi.cc.u32 %10, %18, 38, 0;\n\tmadc.lo.cc.u32 %11, %20, 38, 0;\n\tmadc.hi.cc.u32 %12, %20, 38, 0;\n\t"
        "madc.lo.cc.u32 %13, %22, 38, 0;\n\tmadc.hi.cc.u32 %14, %22, 38, 0;\n\tmadc.lo.cc.u32 %15, %24, 38, 0;\n\tmadc.hi.u32 %16, %24, 38, 0;"
        : "+r"(s[0]), "+r"(s[1]), "+r"(s[2]), "+r"(s[3]), "+r"(s[4]), "+r"(s[5]), "+r"(s[6]), "+r"(s[7]), "=&r"(t8),
          "=&r"(Od2[0]), "=&r"(Od2[1]), "=&r"(Od2[2]), "=&r"(Od2[3]), "=&r"(Od2[4]), "=&r"(Od2[5]), "=&r"(Od2[6]), "=&r"(Od2[7])
        : "r"(s[8]), "r"(s[9]), "r"(s[10]), "r"(s[11]), "r"(s[12]), "r"(s[13]), "r"(s[14]), "r"(s[15]));
#endif
    uint32_t o[8];
    asm("add.cc.u32 %0, %9, %16;\n\taddc.cc.u32 %1, %10, %17;\n\taddc.cc.u32 %2, %11, %18;\n\taddc.cc.u32 %3, %12, %19;\n\t"
        "addc.cc.u32 %4, %13, %20;\n\taddc.cc.u32 %5, %14, %21;\n\taddc.cc.u32 %6, %15, %22;\n\taddc.u32 %7, %8, %23;"
        : "=&r"(o[1]), "=&r"(o[2]), "=&r"(o[3]), "=&r"(o[4]), "=&r"(o[5]), "=&r"(o[6]), "=&r"(o[7]), "=&r"(t8)
        : "r"(t8), "r"(s[1]), "r"(s[2]), "r"(s[3]), "r"(s[4]), "r"(s[5]), "r"(s[6]), "r"(s[7]),
          "r"(Od2[0]), "r"(Od2[1]), "r"(Od2[2]), "r"(Od2[3]), "r"(Od2[4]), "r"(Od2[5]), "r"(Od2[6]), "r"(Od2[7]));
    o[0] = s[0];
    for (int i = 0; i < 8; i++) h.v[i] = o[i];
    fe_addtop_t<NB>(h, fe_mul38small(t8), 0u);
    }
}
__device__ __forceinline__ void fe_sq(fe &h, const fe &f) { fe_sq_impl<(PX_FOLDSHIFT >= 2)>(h, f); }
#ifndef PX_FOLDSHIFT_POW
#define PX_FOLDSHIFT_POW 0
#endif
#if PX_FOLDSHIFT_POW
// the exponentiation chains (square roots in hash-to-curve, inversions) square with the shift fold: there the
// integer pipe has room while the multiplier pipe is the limit
__device__ __forceinline__ void fe_sq_pow(fe &h, const fe &f) { fe_sq_impl<1>(h, f); }
#define PX_HAVE_SQ_POW 1
#endif

#else // PX_WIDECHAIN
// h = f * g
__device__ void fe_mul(fe &h, const fe &f, const fe &g) {
    const uint32_t *a = f.v, *b = g.v;
    uint32_t r[8];                      // words 8..15 of the product
    uint32_t o[8];                      // output kept apart: h may alias f or g (fe_mul(x, x, y))
    uint32_t c0 = 0, c1 = 0, c2 = 0;
    // high columns k = 8..14
    PX_CSTEP(c0, c1, c2, a[1], b[7]); PX_CSTEP(c0, c1, c2, a[2], b[6]); PX_CSTEP(c0, c1, c2, a[3], b[5]); PX_CSTEP(c0, c1, c2, a[4], b[4]); PX_CSTEP(c0, c1, c2, a[5], b[3]); PX_CSTEP(c0, c1, c2, a[6], b[2]); PX_CSTEP(c0, c1, c2, a[7], b[1]);
    r[0] = c0; PX_CSHIFT(c0, c1, c2);
    PX_CSTEP(c0, c1, c2, a[2], b[7]); PX_CSTEP(c0, c1, c2, a[3], b[6]); PX_CSTEP(c0, c1, c2, a[4], b[5]); PX_CSTEP(c0, c1, c2, a[5], b[4]); PX_CSTEP(c0, c1, c2, a[6], b[3]); PX_CSTEP(c0, c1, c2, a[7], b[2]);
    r[1] = c0; PX_CSHIFT(c0, c1, c2);
    PX_CSTEP(c0, c1, c2, a[3], b[7]); PX_CSTEP(c0, c1, c2, a[4], b[6]); PX_CSTEP(c0, c1, c2, a[5], b[5]); PX_CSTEP(c0, c1, c2, a[6], b[4]); PX_CSTEP(c0, c1, c2, a[7], b[3]);
    r[2] = c0; PX_CSHIFT(c0, c1, c2);
    PX_CSTEP(c0, c1, c2, a[4], b[7]); PX_CSTEP(c0, c1, c2, a[5], b[6]); PX_CSTEP(c0, c1, c2, a[6], b[5]); PX_CSTEP(c0, c1, c2, a[7], b[4]);
    r[3] = c0; PX_CSHIFT(c0, c1, c2);
    PX_CSTEP(c0, c1, c2, a[5], b[7]); PX_CSTEP(c0, c1, c2, a[6], b[6]); PX_CSTEP(c0, c1, c2, a[7], b[5]);
    r[4] = c0; PX_CSHIFT(c0, c1, c2);
    PX_CSTEP(c0, c1, c2, a[6], b[7]); PX_CSTEP(c0, c1, c2, a[7], b[6]);
    r[5] = c0; PX_CSHIFT(c0, c1, c2);
    PX_CSTEP2(c0, c1, a[7], b[7]);
    r[6] = c0; r[7] = c1;
    // low columns k = 0..7, each seeded with 38 * word(8+k)
    c0 = c1 = c2 = 0;
    PX_CSTEP(c0, c1, c2, r[0], 38u); PX_CSTEP(c0, c1, c2, a[0], b[0]);
    o[0] = c0; PX_CSHIFT(c0, c1, c2);
    PX_CSTEP(c0, c1, c2, r[1], 38u); PX_CSTEP(c0, c1, c2, a[0], b[1]); PX_CSTEP(c0, c1, c2, a[1], b[0]);
    o[1] = c0; PX_CSHIFT(c0, c1, c2);
    PX_CSTEP(c0, c1, c2, r[2], 38u); PX_CSTEP(c0, c1, c2, a[0], b[2]); PX_CSTEP(c0, c1, c2, a[1], b[1]); PX_CSTEP(c0, c1, c2, a[2], b[0]);
    o[2] = c0; PX_CSHIFT(c0, c1, c2);
    PX_CSTEP(c0, c1, c2, r[3], 38u); PX_CSTEP(c0, c1, c2, a[0], b[3]); PX_CSTEP(c0, c1, c2, a[1], b[2]); PX_CSTEP(c0, c1, c2, a[2], b[1]); PX_CSTEP(c0, c1, c2, a[3], b[0]);
    o[3] = c0; PX_CSHIFT(c0, c1, c2);
    PX_CSTEP(c0, c1, c2, r[4], 38u); PX_CSTEP(c0, c1, c2, a[0], b[4]); PX_CSTEP(c0, c1, c2, a[1], b[3]); PX_CSTEP(c0, c1, c2, a[2], b[2]); PX_CSTEP(c0, c1, c2, a[3], b[1]); PX_CSTEP(c0, c1, c2, a[4], b[0]);
    o[4] = c0; PX_CSHIFT(c0, c1, c2);
    PX_CSTEP(c0, c1, c2, r[5], 38u); PX_CSTEP(c0, c1, c2, a[0], b[5]); PX_CSTEP(c0, c1, c2, a[1], b[4]); PX_CSTEP(c0, c1, c2, a[2], b[3]); PX_CSTEP(c0, c1, c2, a[3], b[2]); PX_CSTEP(c0, c1, c2, a[4], b[1]); PX_CSTEP(c0, c1, c2, a[5], b[0]);
    o[5] = c0; PX_CSHIFT(c0, c1, c2);
    PX_CSTEP(c0, c1, c2, r[6], 38u); PX_CSTEP(c0, c1, c2, a[0], b[6]); PX_CSTEP(c0, c1, c2, a[1], b[5]); PX_CSTEP(c0, c1, c2, a[2], b[4]); PX_CSTEP(c0, c1, c2, a[3], b[3]); PX_CSTEP(c0, c1, c2, a[4], b[2]); PX_CSTEP(c0, c1, c2, a[5], b[1]); PX_CSTEP(c0, c1, c2, a[6], b[0]);
    o[6] = c0; PX_CSHIFT(c0, c1, c2);
    PX_CSTEP(c0, c1, c2, r[7], 38u); PX_CSTEP(c0, c1, c2, a[0], b[7]); PX_CSTEP(c0, c1, c2, a[1], b[6]); PX_CSTEP(c0, c1, c2, a[2], b[5]); PX_CSTEP(c0, c1, c2, a[3], b[4]); PX_CSTEP(c0, c1, c2, a[4], b[3]); PX_CSTEP(c0, c1, c2, a[5], b[2]); PX_CSTEP(c0, c1, c2, a[6], b[1]); PX_CSTEP(c0, c1, c2, a[7], b[0]);
    o[7] = c0;
    // (c2:c1) is the part above 2^256 (< 2^36: the low columns carry up to 2^35 past 2^256); times 38 it spans words 0 and 1
    uint64_t t; asm("mul.wide.u32 %0, %1, 38;" : "=l"(t) : "r"(c1));
    for (int i = 0; i < 8; i++) h.v[i] = o[i];
    fe_addtop(h, (uint32_t)t, (uint32_t)(t >> 32) + c2 * 38u);
}
// h = f^2: the 28 cross products once (comba), doubled by a 16-word shift, the 8 squares added in one
// carry chain of fused wide multiplies, then the 38-fold over exact words
__device__ void fe_sq(fe &h, const fe &f) {
    const uint32_t *a = f.v;
    uint32_t s[16], o[8];
    uint32_t c0 = 0, c1 = 0, c2 = 0;
    s[0] = 0;
    PX_CSTEP(c0, c1, c2, a[0], a[1]); s[1] = c0; PX_CSHIFT(c0, c1, c2);
    PX_CSTEP(c0, c1, c2, a[0], a[2]); s[2] = c0; PX_CSHIFT(c0, c1, c2);
    PX_CSTEP(c0, c1, c2, a[0], a[3]); PX_CSTEP(c0, c1, c2, a[1], a[2]); s[3] = c0; PX_CSHIFT(c0, c1, c2);
    PX_CSTEP(c0, c1, c2, a[0], a[4]); PX_CSTEP(c0, c1, c2, a[1], a[3]); s[4] = c0; PX_CSHIFT(c0, c1, c2);
    PX_CSTEP(c0, c1, c2, a[0], a[5]); PX_CSTEP(c0, c1, c2, a[1], a[4]); PX_CSTEP(c0, c1, c2, a[2], a[3]); s[5] = c0; PX_CSHIFT(c0, c1, c2);
    PX_CSTEP(c0, c1, c2, a[0], a[6]); PX_CSTEP(c0, c1, c2, a[1], a[5]); PX_CSTEP(c0, c1, c2, a[2], a[4]); s[6] = c0; PX_CSHIFT(c0, c1, c2);
    PX_CSTEP(c0, c1, c2, a[0], a[7]); PX_CSTEP(c0, c1, c2, a[1], a[6]); PX_CSTEP(c0, c1, c2, a[2], a[5]); PX_CSTEP(c0, c1, c2, a[3], a[4]); s[7] = c0; PX_CSHIFT(c0, c1, c2);
    PX_CSTEP(c0, c1, c2, a[1], a[7]); PX_CSTEP(c0, c1, c2, a[2], a[6]); PX_CSTEP(c0, c1, c2, a[3], a[5]); s[8] = c0; PX_CSHIFT(c0, c1, c2);
    PX_CSTEP(c0, c1, c2, a[2], a[7]); PX_CSTEP(c0, c1, c2, a[3], a[6]); PX_CSTEP(c0, c1, c2, a[4], a[5]); s[9] = c0; PX_CSHIFT(c0, c1, c2);
    PX_CSTEP(c0, c1, c2, a[3], a[7]); PX_CSTEP(c0, c1, c2, a[4], a[6]); s[10] = c0; PX_CSHIFT(c0, c1, c2);
    PX_CSTEP(c0, c1, c2, a[4], a[7]); PX_CSTEP(c0, c1, c2, a[5], a[6]); s[11] = c0; PX_CSHIFT(c0, c1, c2);
    PX_CSTEP(c0, c1, c2, a[5], a[7]); s[12] = c0; PX_CSHIFT(c0, c1, c2);
    PX_CSTEP2(c0, c1, a[6], a[7]); s[13] = c0; s[14] = c1; s[15] = 0;
    #pragma unroll
    for (int k = 15; k > 0; k--) s[k] = (s[k] << 1) | (s[k - 1] >> 31);   // the cross sum is < 2^480: nothing lost
    s[0] <<= 1;
#ifndef PX_SQCHAIN
#define PX_SQCHAIN 1
#endif
#if PX_SQCHAIN
    // the 8 squares in one carry chain through the flags. s[0] is written, not read (it is zero here), so no
    // two operands of the statement hold a provably equal value: the round-1 operand-coalescing hazard needs
    // equal-valued read-write operands. Checked by --fetest (x = 121666 and the pow22501 replay) and the vectors.
    asm("mad.lo.cc.u32 %0, %16, %16, 0;\n\tmadc.hi.cc.u32 %1, %16, %16, %1;\n\t"
        "madc.lo.cc.u32 %2, %17, %17, %2;\n\tmadc.hi.cc.u32 %3, %17, %17, %3;\n\t"
        "madc.lo.cc.u32 %4, %18, %18, %4;\n\tmadc.hi.cc.u32 %5, %18, %18, %5;\n\t"
        "madc.lo.cc.u32 %6, %19, %19, %6;\n\tmadc.hi.cc.u32 %7, %19, %19, %7;\n\t"
        "madc.lo.cc.u32 %8, %20, %20, %8;\n\tmadc.hi.cc.u32 %9, %20, %20, %9;\n\t"
        "madc.lo.cc.u32 %10, %21, %21, %10;\n\tmadc.hi.cc.u32 %11, %21, %21, %11;\n\t"
        "madc.lo.cc.u32 %12, %22, %22, %12;\n\tmadc.hi.cc.u32 %13, %22, %22, %13;\n\t"
        "madc.lo.cc.u32 %14, %23, %23, %14;\n\tmadc.hi.u32 %15, %23, %23, %15;"
        : "=&r"(s[0]), "+r"(s[1]), "+r"(s[2]), "+r"(s[3]), "+r"(s[4]), "+r"(s[5]), "+r"(s[6]), "+r"(s[7]), "+r"(s[8]), "+r"(s[9]), "+r"(s[10]), "+r"(s[11]), "+r"(s[12]), "+r"(s[13]), "+r"(s[14]), "+r"(s[15])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]));
#else
    // one statement per limb pair with the carry in a register: a single 16-operand statement let the
    // compiler coalesce equal-valued (zero) operands and the writes collided (PLATFORMS.md, round 1 lesson)
    { uint32_t cin = 0;
      #pragma unroll
      for (int i = 0; i < 8; i++) {
          uint32_t cout;
          asm("add.cc.u32 %0, %0, %4;\n\tmadc.lo.cc.u32 %0, %3, %3, %0;\n\tmadc.hi.cc.u32 %1, %3, %3, %1;\n\taddc.u32 %2, 0, 0;"
              : "+r"(s[2 * i]), "+r"(s[2 * i + 1]), "=&r"(cout) : "r"(a[i]), "r"(cin));
          cin = cout;
      } }
#endif
    // fold: h = s_lo + 38 * s_hi over the exact words: column k = 38*s[8+k] + s[k] + carry, which never
    // exceeds two words, so one wide multiply with s[k] as its addend and a two-limb carry add per column
    uint32_t cin = 0;
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        asm("mad.lo.cc.u32 %0, %2, 38, %3;\n\tmadc.hi.u32 %1, %2, 38, 0;\n\tadd.cc.u32 %0, %0, %4;\n\taddc.u32 %1, %1, 0;"
            : "=&r"(c0), "=&r"(c1) : "r"(s[8 + k]), "r"(s[k]), "r"(cin));
        o[k] = c0; cin = c1;
    }
    for (int i = 0; i < 8; i++) h.v[i] = o[i];
    fe_addtop(h, cin * 38u, 0u);   // cin < 39
}
#endif // PX_WIDECHAIN
#else
// h = f * g: 512-bit schoolbook product, then folded
__device__ void fe_mul(fe &h, const fe &f, const fe &g) {
    uint32_t r[18];
    for (int i = 0; i < 18; i++) r[i] = 0;
    PX_MUL_ROW(r + 0, f.v[0], g.v); PX_MUL_ROW(r + 1, f.v[1], g.v);
    PX_MUL_ROW(r + 2, f.v[2], g.v); PX_MUL_ROW(r + 3, f.v[3], g.v);
    PX_MUL_ROW(r + 4, f.v[4], g.v); PX_MUL_ROW(r + 5, f.v[5], g.v);
    PX_MUL_ROW(r + 6, f.v[6], g.v); PX_MUL_ROW(r + 7, f.v[7], g.v);
    fe_fold512(h, r);
}

#ifdef PX_SQ_MUL
__device__ __forceinline__ void fe_sq(fe &h, const fe &f) { fe_mul(h, f, f); }
#else
// h = f^2: the 28 cross products once, doubled by a shift, plus the 8 squares
__device__ void fe_sq(fe &h, const fe &f) {
    uint32_t r[18];
    for (int i = 0; i < 18; i++) r[i] = 0;
    const uint32_t *a = f.v;
    PX_ROW7(r + 1, a[0], a + 1); PX_ROW6(r + 3, a[1], a + 2); PX_ROW5(r + 5, a[2], a + 3);
    PX_ROW4(r + 7, a[3], a + 4); PX_ROW3(r + 9, a[4], a + 5); PX_ROW2(r + 11, a[5], a + 6);
    PX_ROW1(r + 13, a[6], a + 7);
    for (int k = 15; k > 0; k--) r[k] = (r[k] << 1) | (r[k - 1] >> 31);
    r[0] <<= 1;
    // the squares a_i^2 at limbs 2i, 2i+1, one statement per pair with the carry carried
    // explicitly: a single statement over all 16 limbs let the compiler coalesce
    // equal-valued operands into one register and the writes collided
    uint32_t cin = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint32_t cout;
        asm("add.cc.u32 %0, %0, %4;\n\t"
            "madc.lo.cc.u32 %0, %3, %3, %0;\n\t"
            "madc.hi.cc.u32 %1, %3, %3, %1;\n\t"
            "addc.u32 %2, 0, 0;"
            : "+r"(r[2 * i]), "+r"(r[2 * i + 1]), "=&r"(cout) : "r"(a[i]), "r"(cin));
        cin = cout;
    }
    fe_fold512(h, r);
}
#endif
#endif // PX_COMBA
#ifndef PX_HAVE_SQ_POW
#define fe_sq_pow fe_sq
#endif

__device__ __forceinline__ void fe_pow2k(fe &h, const fe &f, int k) { fe_copy(h, f); for (int i = 0; i < k; i++) fe_sq_pow(h, h); }

// canonical representative in [0, p)
__device__ __forceinline__ void fe_canon(uint32_t out[8], const fe &f) {
    fe t; fe_copy(t, f);
    for (int rep = 0; rep < 2; rep++) {
        uint32_t s[8], b;
        asm(
            "sub.cc.u32 %0, %9, 0xFFFFFFED;\n\t"
            "subc.cc.u32 %1, %10, 0xFFFFFFFF;\n\t" "subc.cc.u32 %2, %11, 0xFFFFFFFF;\n\t" "subc.cc.u32 %3, %12, 0xFFFFFFFF;\n\t"
            "subc.cc.u32 %4, %13, 0xFFFFFFFF;\n\t" "subc.cc.u32 %5, %14, 0xFFFFFFFF;\n\t" "subc.cc.u32 %6, %15, 0xFFFFFFFF;\n\t"
            "subc.cc.u32 %7, %16, 0x7FFFFFFF;\n\t" "subc.u32 %8, 0, 0;"
            : "=&r"(s[0]), "=&r"(s[1]), "=&r"(s[2]), "=&r"(s[3]), "=&r"(s[4]), "=&r"(s[5]), "=&r"(s[6]), "=&r"(s[7]), "=&r"(b)
            : "r"(t.v[0]), "r"(t.v[1]), "r"(t.v[2]), "r"(t.v[3]), "r"(t.v[4]), "r"(t.v[5]), "r"(t.v[6]), "r"(t.v[7]));
        if (!(b & 1u)) for (int i = 0; i < 8; i++) t.v[i] = s[i];
    }
    for (int i = 0; i < 8; i++) out[i] = t.v[i];
}
__device__ void fe_tobytes(uint8_t s[32], const fe &f) {
    uint32_t c[8]; fe_canon(c, f);
    for (int i = 0; i < 8; i++) { s[4 * i] = c[i]; s[4 * i + 1] = c[i] >> 8; s[4 * i + 2] = c[i] >> 16; s[4 * i + 3] = c[i] >> 24; }
}
__device__ void fe_frombytes(fe &h, const uint8_t s[32]) {
    for (int i = 0; i < 8; i++) h.v[i] = (uint32_t)s[4 * i] | ((uint32_t)s[4 * i + 1] << 8) | ((uint32_t)s[4 * i + 2] << 16) | ((uint32_t)s[4 * i + 3] << 24);
    h.v[7] &= 0x7FFFFFFFu;
}
__device__ __forceinline__ bool fe_isnegative(const fe &f) { uint32_t c[8]; fe_canon(c, f); return c[0] & 1u; }
__device__ __forceinline__ bool fe_iszero(const fe &f) { uint32_t c[8]; fe_canon(c, f); uint32_t o = 0; for (int i = 0; i < 8; i++) o |= c[i]; return o == 0; }
__device__ __forceinline__ bool fe_eq(const fe &f, const fe &g) { fe t; fe_sub(t, f, g); return fe_iszero(t); }
#else
#define fe_sq_pow fe_sq
struct fe { uint64_t v[5]; };
#define MASK51 0x7FFFFFFFFFFFFULL

struct u128 { uint64_t lo, hi; };
__device__ __forceinline__ void mac(u128 &c, uint64_t a, uint64_t b) {
    uint64_t lo = a * b, hi = __umul64hi(a, b);
    uint64_t nlo = c.lo + lo;
    c.hi += hi + (nlo < c.lo);
    c.lo = nlo;
}
__device__ __forceinline__ void add64(u128 &c, uint64_t a) { uint64_t n = c.lo + a; c.hi += (n < c.lo); c.lo = n; }
__device__ __forceinline__ uint64_t shr51(const u128 &c) { return (c.lo >> 51) | (c.hi << 13); }

__device__ __forceinline__ void fe_0(fe &h) { h.v[0] = h.v[1] = h.v[2] = h.v[3] = h.v[4] = 0; }
__device__ __forceinline__ void fe_1(fe &h) { h.v[0] = 1; h.v[1] = h.v[2] = h.v[3] = h.v[4] = 0; }
__device__ __forceinline__ void fe_copy(fe &h, const fe &f) { for (int i = 0; i < 5; i++) h.v[i] = f.v[i]; }

__device__ __forceinline__ void fe_reduce(fe &h) {
    uint64_t c0 = h.v[0] >> 51, c1 = h.v[1] >> 51, c2 = h.v[2] >> 51, c3 = h.v[3] >> 51, c4 = h.v[4] >> 51;
    h.v[0] &= MASK51; h.v[1] &= MASK51; h.v[2] &= MASK51; h.v[3] &= MASK51; h.v[4] &= MASK51;
    h.v[0] += c4 * 19; h.v[1] += c0; h.v[2] += c1; h.v[3] += c2; h.v[4] += c3;
}
__device__ __forceinline__ void fe_add(fe &h, const fe &f, const fe &g) {
    for (int i = 0; i < 5; i++) h.v[i] = f.v[i] + g.v[i];
}
// h = f - g, inputs with limbs < 2^54
__device__ __forceinline__ void fe_sub(fe &h, const fe &f, const fe &g) {
    h.v[0] = f.v[0] + 36028797018963664ULL - g.v[0];
    h.v[1] = f.v[1] + 36028797018963952ULL - g.v[1];
    h.v[2] = f.v[2] + 36028797018963952ULL - g.v[2];
    h.v[3] = f.v[3] + 36028797018963952ULL - g.v[3];
    h.v[4] = f.v[4] + 36028797018963952ULL - g.v[4];
    fe_reduce(h);
}
__device__ __forceinline__ void fe_neg(fe &h, const fe &f) { fe z; fe_0(z); fe_sub(h, z, f); }

__device__ void fe_mul(fe &h, const fe &f, const fe &g) {
    uint64_t a0 = f.v[0], a1 = f.v[1], a2 = f.v[2], a3 = f.v[3], a4 = f.v[4];
    uint64_t b0 = g.v[0], b1 = g.v[1], b2 = g.v[2], b3 = g.v[3], b4 = g.v[4];
    uint64_t b1_19 = b1 * 19, b2_19 = b2 * 19, b3_19 = b3 * 19, b4_19 = b4 * 19;
    u128 c0 = {0, 0}, c1 = {0, 0}, c2 = {0, 0}, c3 = {0, 0}, c4 = {0, 0};
    mac(c0, a0, b0); mac(c0, a4, b1_19); mac(c0, a3, b2_19); mac(c0, a2, b3_19); mac(c0, a1, b4_19);
    mac(c1, a1, b0); mac(c1, a0, b1); mac(c1, a4, b2_19); mac(c1, a3, b3_19); mac(c1, a2, b4_19);
    mac(c2, a2, b0); mac(c2, a1, b1); mac(c2, a0, b2); mac(c2, a4, b3_19); mac(c2, a3, b4_19);
    mac(c3, a3, b0); mac(c3, a2, b1); mac(c3, a1, b2); mac(c3, a0, b3); mac(c3, a4, b4_19);
    mac(c4, a4, b0); mac(c4, a3, b1); mac(c4, a2, b2); mac(c4, a1, b3); mac(c4, a0, b4);
    uint64_t o0, o1, o2, o3, o4;
    add64(c1, shr51(c0)); o0 = c0.lo & MASK51;
    add64(c2, shr51(c1)); o1 = c1.lo & MASK51;
    add64(c3, shr51(c2)); o2 = c2.lo & MASK51;
    add64(c4, shr51(c3)); o3 = c3.lo & MASK51;
    uint64_t carry = shr51(c4); o4 = c4.lo & MASK51;
    o0 += carry * 19; carry = o0 >> 51; o0 &= MASK51; o1 += carry;
    h.v[0] = o0; h.v[1] = o1; h.v[2] = o2; h.v[3] = o3; h.v[4] = o4;
}

__device__ void fe_sq(fe &h, const fe &f) {
    uint64_t a0 = f.v[0], a1 = f.v[1], a2 = f.v[2], a3 = f.v[3], a4 = f.v[4];
    uint64_t a3_19 = a3 * 19, a4_19 = a4 * 19;
    uint64_t a0_2 = a0 * 2, a1_2 = a1 * 2;
    u128 c0 = {0, 0}, c1 = {0, 0}, c2 = {0, 0}, c3 = {0, 0}, c4 = {0, 0};
    mac(c0, a0, a0);   mac(c0, a1_2, a4_19); mac(c0, a2 * 2, a3_19);
    mac(c1, a3, a3_19); mac(c1, a0_2, a1);   mac(c1, a2 * 2, a4_19);
    mac(c2, a1, a1);   mac(c2, a0_2, a2);    mac(c2, a4 * 2, a3_19);
    mac(c3, a4, a4_19); mac(c3, a0_2, a3);   mac(c3, a1_2, a2);
    mac(c4, a2, a2);   mac(c4, a0_2, a4);    mac(c4, a1_2, a3);
    uint64_t o0, o1, o2, o3, o4;
    add64(c1, shr51(c0)); o0 = c0.lo & MASK51;
    add64(c2, shr51(c1)); o1 = c1.lo & MASK51;
    add64(c3, shr51(c2)); o2 = c2.lo & MASK51;
    add64(c4, shr51(c3)); o3 = c3.lo & MASK51;
    uint64_t carry = shr51(c4); o4 = c4.lo & MASK51;
    o0 += carry * 19; carry = o0 >> 51; o0 &= MASK51; o1 += carry;
    h.v[0] = o0; h.v[1] = o1; h.v[2] = o2; h.v[3] = o3; h.v[4] = o4;
}
__device__ __forceinline__ void fe_pow2k(fe &h, const fe &f, int k) { fe_copy(h, f); for (int i = 0; i < k; i++) fe_sq_pow(h, h); }

__device__ void fe_tobytes(uint8_t s[32], const fe &f) {
    fe h; fe_copy(h, f); fe_reduce(h);
    uint64_t q = (h.v[0] + 19) >> 51;
    q = (h.v[1] + q) >> 51; q = (h.v[2] + q) >> 51; q = (h.v[3] + q) >> 51; q = (h.v[4] + q) >> 51;
    h.v[0] += 19 * q;
    h.v[1] += h.v[0] >> 51; h.v[0] &= MASK51;
    h.v[2] += h.v[1] >> 51; h.v[1] &= MASK51;
    h.v[3] += h.v[2] >> 51; h.v[2] &= MASK51;
    h.v[4] += h.v[3] >> 51; h.v[3] &= MASK51;
    h.v[4] &= MASK51;
    uint64_t l0 = h.v[0], l1 = h.v[1], l2 = h.v[2], l3 = h.v[3], l4 = h.v[4];
    s[0] = l0; s[1] = l0 >> 8; s[2] = l0 >> 16; s[3] = l0 >> 24; s[4] = l0 >> 32; s[5] = l0 >> 40;
    s[6] = (l0 >> 48) | (l1 << 3); s[7] = l1 >> 5; s[8] = l1 >> 13; s[9] = l1 >> 21; s[10] = l1 >> 29; s[11] = l1 >> 37;
    s[12] = (l1 >> 45) | (l2 << 6); s[13] = l2 >> 2; s[14] = l2 >> 10; s[15] = l2 >> 18; s[16] = l2 >> 26; s[17] = l2 >> 34; s[18] = l2 >> 42;
    s[19] = (l2 >> 50) | (l3 << 1); s[20] = l3 >> 7; s[21] = l3 >> 15; s[22] = l3 >> 23; s[23] = l3 >> 31; s[24] = l3 >> 39;
    s[25] = (l3 >> 47) | (l4 << 4); s[26] = l4 >> 4; s[27] = l4 >> 12; s[28] = l4 >> 20; s[29] = l4 >> 28; s[30] = l4 >> 36; s[31] = l4 >> 44;
}
__device__ __forceinline__ uint64_t load8(const uint8_t *p) {
    uint64_t r = 0; for (int i = 7; i >= 0; i--) r = (r << 8) | p[i]; return r;
}
__device__ void fe_frombytes(fe &h, const uint8_t s[32]) {
    h.v[0] = load8(s + 0) & MASK51;
    h.v[1] = (load8(s + 6) >> 3) & MASK51;
    h.v[2] = (load8(s + 12) >> 6) & MASK51;
    h.v[3] = (load8(s + 19) >> 1) & MASK51;
    h.v[4] = (load8(s + 24) >> 12) & MASK51;
}
__device__ __forceinline__ bool fe_isnegative(const fe &f) { uint8_t s[32]; fe_tobytes(s, f); return s[0] & 1; }
__device__ __forceinline__ bool fe_iszero(const fe &f) { uint8_t s[32]; fe_tobytes(s, f); uint8_t o = 0; for (int i = 0; i < 32; i++) o |= s[i]; return o == 0; }
__device__ __forceinline__ bool fe_eq(const fe &f, const fe &g) { fe t; fe_sub(t, f, g); return fe_iszero(t); }

#endif // PX_FE32

// (x^(2^250-1), x^11)
__device__ void fe_pow22501(fe &t19, fe &t3, const fe &x) {
    fe t0, t1, t2, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18;
    fe_sq(t0, x);
    fe_sq(t1, t0); fe_sq(t1, t1);
    fe_mul(t2, x, t1);
    fe_mul(t3, t0, t2);
    fe_sq(t4, t3);
    fe_mul(t5, t2, t4);
    fe_pow2k(t6, t5, 5);
    fe_mul(t7, t6, t5);
    fe_pow2k(t8, t7, 10);
    fe_mul(t9, t8, t7);
    fe_pow2k(t10, t9, 20);
    fe_mul(t11, t10, t9);
    fe_pow2k(t12, t11, 10);
    fe_mul(t13, t12, t7);
    fe_pow2k(t14, t13, 50);
    fe_mul(t15, t14, t13);
    fe_pow2k(t16, t15, 100);
    fe_mul(t17, t16, t15);
    fe_pow2k(t18, t17, 50);
    fe_mul(t19, t18, t13);
}
__device__ void fe_invert(fe &h, const fe &x) {
    fe t19, t3, t20;
    fe_pow22501(t19, t3, x);
    fe_pow2k(t20, t19, 5);
    fe_mul(h, t20, t3);
}
// x^((p-5)/8)
__device__ void fe_pow_p58(fe &h, const fe &x) {
    fe t19, t3, t20;
    fe_pow22501(t19, t3, x);
    fe_pow2k(t20, t19, 2);
    fe_mul(h, x, t20);
}

__device__ fe d_SQRT_M1, d_EDWARDS_D, d_EDWARDS_D2;

// r = sqrt(u/v) non-negative; returns whether u/v was a (nonzero or zero) square
__device__ bool fe_sqrt_ratio_i(fe &r, const fe &u, const fe &v) {
    fe v3, v7, t, uv3, uv7, check, nu, nui, rp;
    fe_sq(v3, v); fe_mul(v3, v3, v);
    fe_sq(v7, v3); fe_mul(v7, v7, v);
    fe_mul(uv3, u, v3);
    fe_mul(uv7, u, v7);
    fe_pow_p58(t, uv7);
    fe_mul(r, uv3, t);
    fe_sq(check, r); fe_mul(check, check, v);
    fe_neg(nu, u);
    bool correct = fe_eq(check, u);
    bool flipped = fe_eq(check, nu);
    fe_mul(rp, r, d_SQRT_M1);
#if PX_SQRT_DEC
    // only point decoding calls this, and it discards r unless u/v is a square (correct or flipped):
    // the check against -u*sqrt(-1) can only change r in the rejected case
    (void)nui;
    if (flipped) fe_copy(r, rp);
#else
    fe_mul(nui, nu, d_SQRT_M1);
    bool flipped_i = fe_eq(check, nui);
    if (flipped || flipped_i) fe_copy(r, rp);
#endif
    if (fe_isnegative(r)) fe_neg(r, r);
    return correct || flipped;
}

// ----------------------------------------------------------------------------
// points, extended coordinates (X:Y:Z:T), a = -1
// ----------------------------------------------------------------------------
struct ge { fe X, Y, Z, T; };

__device__ __forceinline__ void ge_identity(ge &p) { fe_0(p.X); fe_1(p.Y); fe_1(p.Z); fe_0(p.T); }

__device__ bool ge_frombytes(ge &p, const uint8_t s[32]) {
    fe Y, YY, u, v, X;
    fe_frombytes(Y, s);
    fe_1(p.Z);
    fe_sq(YY, Y);
    fe_sub(u, YY, p.Z);
    fe_mul(v, YY, d_EDWARDS_D); fe_add(v, v, p.Z);
    if (!fe_sqrt_ratio_i(X, u, v)) return false;
    if (s[31] >> 7) fe_neg(X, X);
    fe_copy(p.X, X); fe_copy(p.Y, Y);
    fe_mul(p.T, X, Y);
    return true;
}
__device__ void ge_tobytes(uint8_t s[32], const ge &p) {
    fe zi, x, y;
    fe_invert(zi, p.Z);
    fe_mul(x, p.X, zi); fe_mul(y, p.Y, zi);
    fe_tobytes(s, y);
    s[31] ^= (fe_isnegative(x) ? 0x80 : 0);
}
__device__ bool ge_is_identity(const ge &p) { return fe_iszero(p.X) && fe_eq(p.Y, p.Z); }

// Doubling reads X, Y, Z only (4M + 4S). The T coordinate is needed solely as an
// input to the next addition, so a doubling that is followed by another doubling
// may skip it (ge_dbl_noT, 3M + 4S): in the scalar multiplication most doublings
// are, which saves about a tenth of its multiplications.
__device__ void ge_dbl(ge &r, const ge &p) {
    fe A, B, C, D, E, F, G, H, t;
    fe_sq(A, p.X);
    fe_sq(B, p.Y);
    fe_sq(C, p.Z); fe_add(C, C, C);
    fe_neg(D, A);
    fe_add(t, p.X, p.Y); fe_sq(t, t); fe_sub(t, t, A); fe_sub(E, t, B);
    fe_add(G, D, B);
    fe_sub(F, G, C);
    fe_sub(H, D, B);
    fe_mul(r.X, E, F); fe_mul(r.Y, G, H); fe_mul(r.T, E, H); fe_mul(r.Z, F, G);
}
__device__ void ge_dbl_noT(ge &r, const ge &p) {
    fe A, B, C, D, E, F, G, H, t;
    fe_sq(A, p.X);
    fe_sq(B, p.Y);
    fe_sq(C, p.Z); fe_add(C, C, C);
    fe_neg(D, A);
    fe_add(t, p.X, p.Y); fe_sq(t, t); fe_sub(t, t, A); fe_sub(E, t, B);
    fe_add(G, D, B);
    fe_sub(F, G, C);
    fe_sub(H, D, B);
    fe_mul(r.X, E, F); fe_mul(r.Y, G, H); fe_mul(r.Z, F, G);
}
// one doubling for both cases of the scalar multiplication loop: T only when
// an addition follows (wantT is uniform across the grid). A single call site
// keeps one copy of the doubling in the instruction cache instead of two.
__device__ void ge_dbl_opt(ge &r, const ge &p, bool wantT) {
    fe A, B, C, D, E, F, G, H, t;
    fe_sq(A, p.X);
    fe_sq(B, p.Y);
    fe_sq(C, p.Z); fe_add(C, C, C);
    fe_neg(D, A);
    fe_add(t, p.X, p.Y); fe_sq(t, t); fe_sub(t, t, A); fe_sub(E, t, B);
    fe_add(G, D, B);
    fe_sub(F, G, C);
    fe_sub(H, D, B);
    fe_mul(r.X, E, F); fe_mul(r.Y, G, H); fe_mul(r.Z, F, G);
    if (wantT) fe_mul(r.T, E, H);
}
#if PX_DBL6
// The same doubling with two linear operations fewer: with s = A + B the formulas give
// H = -s, and F' = C - G = -F. (X3:Y3:Z3:T3) = (E F', G s, F' G, E s) is the negated
// projective representative of (E F : G H : F G : E H), i.e. the same point.
__device__ void ge_dbl_opt6(ge &r, const ge &p, bool wantT) {
    fe A, B, C, E, F, G, s, t;
    fe_sq(A, p.X);
    fe_sq(B, p.Y);
    fe_sq(C, p.Z); fe_add(C, C, C);
    fe_add(t, p.X, p.Y); fe_sq(t, t);
    fe_add(s, A, B);
    fe_sub(E, t, s);
    fe_sub(G, B, A);
    fe_sub(F, C, G);
    fe_mul(r.X, E, F); fe_mul(r.Y, G, s); fe_mul(r.Z, F, G);
    if (wantT) fe_mul(r.T, E, s);
}
#define ge_dbl_opt ge_dbl_opt6
#endif
__device__ void ge_add(ge &r, const ge &p, const ge &q) {
    fe A, B, C, D, E, F, G, H, t1, t2;
    fe_sub(t1, p.Y, p.X); fe_sub(t2, q.Y, q.X); fe_mul(A, t1, t2);
    fe_add(t1, p.Y, p.X); fe_add(t2, q.Y, q.X); fe_mul(B, t1, t2);
    fe_mul(C, p.T, d_EDWARDS_D2); fe_mul(C, C, q.T);
    fe_mul(D, p.Z, q.Z); fe_add(D, D, D);
    fe_sub(E, B, A); fe_sub(F, D, C); fe_add(G, D, C); fe_add(H, B, A);
    fe_mul(r.X, E, F); fe_mul(r.Y, G, H); fe_mul(r.T, E, H); fe_mul(r.Z, F, G);
}
// only the last doubling needs T (the result may feed an addition)
__device__ __forceinline__ void ge_mul8(ge &r, const ge &p) { ge_dbl_noT(r, p); ge_dbl_noT(r, r); ge_dbl(r, r); }

// r = scalar * p, scalar bits in constant memory (uniform across the grid)
__constant__ uint8_t c_scalar[32];
__device__ __forceinline__ void ge_copy(ge &r, const ge &p) { fe_copy(r.X, p.X); fe_copy(r.Y, p.Y); fe_copy(r.Z, p.Z); fe_copy(r.T, p.T); }
#ifndef PX_WINDOW
#define PX_WINDOW 4
#endif
#ifndef PX_LB_THREADS
#define PX_LB_THREADS 256
#endif
#ifndef PX_COMPACT
#define PX_COMPACT 0
#endif
#ifndef PX_LB_BLOCKS
#define PX_LB_BLOCKS 1
#endif
// wNAF digits of the constant scalar (host computes them, see v_set_scalar);
// odd values in [-(2^(w-1)-1), 2^(w-1)-1], uniform across the grid
__constant__ int8_t c_wnaf[260];
__constant__ int c_wnaf_len;
// c_wnaf_next[i]: table index of the next non-zero digit below i (-1: none), so
// the entry can be prefetched into L1 during the doublings that precede its use
__constant__ int8_t c_wnaf_next[260];
#ifndef PX_PREFETCH
#define PX_PREFETCH 0
#endif
// PX_MERGE8: the two cofactor multiplications (x8 before and after the scalar multiplication) are
// folded into the fixed scalar: 8*(x*(8H)) = (64x mod L)*H for every decompressed point H, because the
// cofactor kills the torsion part of H either way. Saves six doublings and one scalar bit per nonce.
// A torsion point H (8H = identity) gives the identity after the multiplication and is rejected there.
// wNAF only: the plain and 4-bit windows assume a clamped scalar with bit 254 set.
#ifndef PX_MERGE8
#define PX_MERGE8 0
#endif
#ifndef PX_DBL6
#define PX_DBL6 0
#endif
#ifndef PX_ADD_NOT
#define PX_ADD_NOT 0
#endif
#ifndef PX_TBL_CACHED
#define PX_TBL_CACHED 0
#endif
#if PX_MERGE8 && PX_WINDOW != 5
#error PX_MERGE8 requires the wNAF scalar multiplication
#endif

// a point cached for mixed addition: Y+X, Y-X, 2d*T, 2Z
struct geCached { fe YpX, YmX, T2d, Z2; };
__device__ __forceinline__ void ge_cache(geCached &c, const ge &p) {
    fe_add(c.YpX, p.Y, p.X); fe_sub(c.YmX, p.Y, p.X); fe_mul(c.T2d, p.T, d_EDWARDS_D2); fe_add(c.Z2, p.Z, p.Z);
}
// r = p + q (or p - q when neg); the sign only swaps the two sums and negates 2dT
__device__ void ge_add_cached(ge &r, const ge &p, const geCached &q, bool neg) {
    fe A, B, C, D, E, F, G, H, t1, t2;
    fe_sub(t1, p.Y, p.X); fe_add(t2, p.Y, p.X);
    fe_mul(A, t1, neg ? q.YpX : q.YmX);
    fe_mul(B, t2, neg ? q.YmX : q.YpX);
    fe_mul(C, p.T, q.T2d); if (neg) fe_neg(C, C);
    fe_mul(D, p.Z, q.Z2);
    fe_sub(E, B, A); fe_sub(F, D, C); fe_add(G, D, C); fe_add(H, B, A);
    fe_mul(r.X, E, F); fe_mul(r.Y, G, H); fe_mul(r.T, E, H); fe_mul(r.Z, F, G);
}
#if PX_ADD_NOT
// The addition inside the wNAF loop: the next operation is always a doubling (wNAF digits are
// never adjacent), which does not read T, and the result after the last digit is only compared
// and compressed. So the T product is never needed there (one multiplication per addition).
__device__ void ge_add_cached_noT(ge &r, const ge &p, const geCached &q, bool neg) {
    fe A, B, C, D, E, F, G, H, t1, t2;
    fe_sub(t1, p.Y, p.X); fe_add(t2, p.Y, p.X);
    fe_mul(A, t1, neg ? q.YpX : q.YmX);
    fe_mul(B, t2, neg ? q.YmX : q.YpX);
    fe_mul(C, p.T, q.T2d); if (neg) fe_neg(C, C);
    fe_mul(D, p.Z, q.Z2);
    fe_sub(E, B, A); fe_sub(F, D, C); fe_add(G, D, C); fe_add(H, B, A);
    fe_mul(r.X, E, F); fe_mul(r.Y, G, H); fe_mul(r.Z, F, G);
}
#define PX_LOOP_ADD ge_add_cached_noT
#else
#define PX_LOOP_ADD ge_add_cached
#endif

#if PX_WINDOW == 5
// width-5 wNAF over the constant scalar: 8 cached odd multiples P, 3P, ..., 15P,
// about 255 doublings and 43 mixed additions, no branch divergence (digits uniform)
__device__ void ge_scalarmult_const(ge &r, const ge &p) {
    geCached T[8];
    ge P2, acc;
    ge_dbl(P2, p);
    ge_copy(acc, p);
    ge_cache(T[0], acc);
#if PX_TBL_CACHED
    // 2P cached once: each table step is then a cached addition (8 instead of 9 multiplications)
    { geCached C2; ge_cache(C2, P2);
      #pragma unroll 1
      for (int i = 1; i < 8; i++) { ge_add_cached(acc, acc, C2, false); ge_cache(T[i], acc); } }
#else
    #pragma unroll 1
    for (int i = 1; i < 8; i++) { ge_add(acc, acc, P2); ge_cache(T[i], acc); }
#endif
    ge_identity(r);
    // The loop body is the hot code of the whole miner; with one doubling and
    // one addition site it is about half the size of the two-plus-two version
    // and stays in the instruction cache (ncu: "no instruction" stalls).
    for (int i = c_wnaf_len - 1; i >= 0; i--) {
        int d = c_wnaf[i];
        // the digit is uniform across the grid: no divergence
        ge_dbl_opt(r, r, d != 0);
        if (d) PX_LOOP_ADD(r, r, T[(d < 0 ? -d : d) >> 1], d < 0);
#if PX_PREFETCH
        // The table lives in local memory and, at 1 KB per thread, misses L1
        // (30 % hit rate measured): fetch the next entry while the doublings
        // before its use run. Local memory is interleaved per word across the
        // warp, so every word of the entry is on its own cache line.
        if (d) {
            int n = c_wnaf_next[i];
            if (n >= 0) {
                const char *p = (const char *)&T[n];
                #pragma unroll
                for (int k = 0; k < (int)sizeof(geCached); k += 4) asm volatile("prefetch.L1 [%0];" :: "l"(p + k));
            }
        }
#endif
    }
}
#elif PX_WINDOW == 0
// plain double-and-add over the constant scalar: no table, no local memory
__device__ void ge_scalarmult_const(ge &r, const ge &p) {
    ge_copy(r, p);
    for (int i = 253; i >= 0; i--) {
        if ((c_scalar[i >> 3] >> (i & 7)) & 1) { ge_dbl(r, r); ge_add(r, r, p); } else ge_dbl_noT(r, r);
    }
}
#elif PX_WINDOW == 2
// 2-bit windows over the constant scalar: table P, 2P, 3P
__device__ void ge_scalarmult_const(ge &r, const ge &p) {
    ge T2, T3;
    ge_dbl(T2, p); ge_add(T3, T2, p);
    int w = (c_scalar[31] >> 6) & 3; // bits 254..255 = 1 for a clamped scalar
    if (w == 2) ge_copy(r, T2); else if (w == 3) ge_copy(r, T3); else ge_copy(r, p);
    for (int i = 126; i >= 0; i--) {
        ge_dbl(r, r); ge_dbl(r, r);
        w = (c_scalar[i >> 2] >> ((i & 3) * 2)) & 3;
        if (w == 1) ge_add(r, r, p); else if (w == 2) ge_add(r, r, T2); else if (w == 3) ge_add(r, r, T3);
    }
}
#else
// fixed 4-bit windows over the constant scalar: 7 dbl + 7 add for the table,
// then 63 x (4 dbl + <=1 add). The window values are uniform across the grid.
__device__ void ge_scalarmult_const(ge &r, const ge &p) {
#ifdef PX_LINEAR_W4
    // Construct cached multiples directly; the uncached 16-point table is
    // unnecessary. One doubling and thirteen cached additions replace seven
    // doublings and seven uncached additions while halving table storage.
    geCached C[16];
    ge acc; ge_copy(acc, p);
    ge_cache(C[1], acc);
    int w = (c_scalar[31] >> 4) & 15;
    #pragma unroll 1
    for (int i = 2; i < 16; ++i) {
        if (i == 2) ge_dbl(acc, acc);
        else ge_add_cached(acc, acc, C[1], false);
        ge_cache(C[i], acc);
        if (i == w) ge_copy(r, acc);
    }
#else
    ge T[16];
    ge_copy(T[1], p);
    for (int i = 2; i < 16; i++) {
        if (i & 1) ge_add(T[i], T[i - 1], p); else ge_dbl(T[i], T[i >> 1]);
    }
    // top window (bits 252..255) is 4..7 for a clamped scalar, never zero
    int w = (c_scalar[31] >> 4) & 0xF;
    ge_copy(r, T[w]);
    // the table in cached form: 8 instead of 9 multiplications per addition (H100 +1.7 %)
    geCached C[16];
    for (int i = 1; i < 16; i++) ge_cache(C[i], T[i]);
#endif
    for (int i = 62; i >= 0; i--) {
        w = (c_scalar[i >> 1] >> ((i & 1) * 4)) & 0xF;
        ge_dbl_noT(r, r); ge_dbl_noT(r, r); ge_dbl_noT(r, r);
        if (w) { ge_dbl(r, r); ge_add_cached(r, r, C[w], false); } else ge_dbl_noT(r, r);
    }
}
#endif
// generic scalar mult with a per-call scalar (used for key derivation)
__device__ void ge_scalarmult(ge &r, const uint8_t *sc, const ge &p) {
    ge_identity(r);
    for (int i = 255; i >= 0; i--) {
        ge_dbl(r, r);
        if ((sc[i >> 3] >> (i & 7)) & 1) ge_add(r, r, p);
    }
}

// ----------------------------------------------------------------------------
// SHA-512, single block (message <= 111 bytes)
// ----------------------------------------------------------------------------
__constant__ uint64_t c_K512[80] = {
    0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL, 0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL,
    0xd807aa98a3030242ULL, 0x12835b0145706fbeULL, 0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL, 0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL, 0xc19bf174cf692694ULL,
    0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL, 0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL, 0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL,
    0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL, 0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL, 0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL, 0x06ca6351e003826fULL, 0x142929670a0e6e70ULL,
    0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL, 0x53380d139d95b3dfULL, 0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL, 0x92722c851482353bULL,
    0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL, 0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL, 0xd192e819d6ef5218ULL, 0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL,
    0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL, 0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL, 0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL, 0x5b9cca4f7763e373ULL, 0x682e6ff3d6b2b8a3ULL,
    0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL, 0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL, 0xc67178f2e372532bULL,
    0xca273eceea26619cULL, 0xd186b8c721c0c207ULL, 0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL, 0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL, 0x113f9804bef90daeULL, 0x1b710b35131c471bULL,
    0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL, 0x431d67c49c100d4cULL, 0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL, 0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL
};
__device__ __forceinline__ uint64_t rotr64(uint64_t x, int n) { return (x >> n) | (x << (64 - n)); }

#ifdef PX_SHA_OLD
__device__ void sha512_single(const uint8_t *msg, int len, uint8_t out[64]) {
    uint8_t blk[128];
    for (int i = 0; i < 128; i++) blk[i] = 0;
    for (int i = 0; i < len; i++) blk[i] = msg[i];
    blk[len] = 0x80;
    uint64_t bits = (uint64_t)len * 8;
    blk[126] = bits >> 8; blk[127] = bits & 0xff;
    uint64_t W[80];
    for (int i = 0; i < 16; i++) {
        uint64_t w = 0; for (int j = 0; j < 8; j++) w = (w << 8) | blk[i * 8 + j]; W[i] = w;
    }
    for (int i = 16; i < 80; i++) {
        uint64_t s0 = rotr64(W[i - 15], 1) ^ rotr64(W[i - 15], 8) ^ (W[i - 15] >> 7);
        uint64_t s1 = rotr64(W[i - 2], 19) ^ rotr64(W[i - 2], 61) ^ (W[i - 2] >> 6);
        W[i] = W[i - 16] + s0 + W[i - 7] + s1;
    }
    uint64_t a = 0x6a09e667f3bcc908ULL, b = 0xbb67ae8584caa73bULL, c = 0x3c6ef372fe94f82bULL, d = 0xa54ff53a5f1d36f1ULL;
    uint64_t e = 0x510e527fade682d1ULL, f = 0x9b05688c2b3e6c1fULL, g = 0x1f83d9abfb41bd6bULL, h = 0x5be0cd19137e2179ULL;
    for (int i = 0; i < 80; i++) {
        uint64_t S1 = rotr64(e, 14) ^ rotr64(e, 18) ^ rotr64(e, 41);
        uint64_t ch = (e & f) ^ (~e & g);
        uint64_t t1 = h + S1 + ch + c_K512[i] + W[i];
        uint64_t S0 = rotr64(a, 28) ^ rotr64(a, 34) ^ rotr64(a, 39);
        uint64_t mj = (a & b) ^ (a & c) ^ (b & c);
        uint64_t t2 = S0 + mj;
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    uint64_t st[8] = { a + 0x6a09e667f3bcc908ULL, b + 0xbb67ae8584caa73bULL, c + 0x3c6ef372fe94f82bULL, d + 0xa54ff53a5f1d36f1ULL,
                       e + 0x510e527fade682d1ULL, f + 0x9b05688c2b3e6c1fULL, g + 0x1f83d9abfb41bd6bULL, h + 0x5be0cd19137e2179ULL };
    for (int i = 0; i < 8; i++) for (int j = 0; j < 8; j++) out[i * 8 + j] = st[i] >> (56 - 8 * j);
}

#else
__device__ void sha512_single(const uint8_t *msg, int len, uint8_t out[64]) {
    uint64_t W[16];
    // the padded single block straight into the schedule: message, 0x80, zeros, 128-bit length
    for (int i = 0; i < 16; i++) {
        uint64_t w = 0;
        for (int j = 0; j < 8; j++) {
            int k = i * 8 + j;
            uint8_t b = k < len ? msg[k] : (k == len ? 0x80 : 0);
            w = (w << 8) | b;
        }
        W[i] = w;
    }
    W[15] = (uint64_t)len * 8;
    uint64_t a = 0x6a09e667f3bcc908ULL, b = 0xbb67ae8584caa73bULL, c = 0x3c6ef372fe94f82bULL, d = 0xa54ff53a5f1d36f1ULL;
    uint64_t e = 0x510e527fade682d1ULL, f = 0x9b05688c2b3e6c1fULL, g = 0x1f83d9abfb41bd6bULL, h = 0x5be0cd19137e2179ULL;
    #pragma unroll
    for (int i = 0; i < 80; i++) {
        uint64_t wi;
        if (i < 16) wi = W[i];
        else {
            uint64_t w15 = W[(i - 15) & 15], w2 = W[(i - 2) & 15];
            uint64_t s0 = rotr64(w15, 1) ^ rotr64(w15, 8) ^ (w15 >> 7);
            uint64_t s1 = rotr64(w2, 19) ^ rotr64(w2, 61) ^ (w2 >> 6);
            wi = W[i & 15] + s0 + W[(i - 7) & 15] + s1;
            W[i & 15] = wi;
        }
        uint64_t S1 = rotr64(e, 14) ^ rotr64(e, 18) ^ rotr64(e, 41);
        uint64_t ch = (e & f) ^ (~e & g);
        uint64_t t1 = h + S1 + ch + c_K512[i] + wi;
        uint64_t S0 = rotr64(a, 28) ^ rotr64(a, 34) ^ rotr64(a, 39);
        uint64_t mj = (a & b) ^ (a & c) ^ (b & c);
        uint64_t t2 = S0 + mj;
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    uint64_t st[8] = { a + 0x6a09e667f3bcc908ULL, b + 0xbb67ae8584caa73bULL, c + 0x3c6ef372fe94f82bULL, d + 0xa54ff53a5f1d36f1ULL,
                       e + 0x510e527fade682d1ULL, f + 0x9b05688c2b3e6c1fULL, g + 0x1f83d9abfb41bd6bULL, h + 0x5be0cd19137e2179ULL };
    for (int i = 0; i < 8; i++) for (int j = 0; j < 8; j++) out[i * 8 + j] = st[i] >> (56 - 8 * j);
}

#endif

// ----------------------------------------------------------------------------
// the VRF output for one nonce
// ----------------------------------------------------------------------------

__device__ bool vrf_beta(const JobDev &job, uint64_t nonce, uint8_t beta[64], uint8_t *dbgH, uint8_t *dbgCG) {
    uint8_t msg[128];
    int n = 0;
    msg[n++] = 0x03; msg[n++] = 0x01;
    for (int i = 0; i < 32; i++) msg[n++] = job.pk[i];
    for (int i = 0; i < job.prefix_len; i++) msg[n++] = job.prefix[i];
    for (int i = 0; i < 8; i++) msg[n++] = (uint8_t)(nonce >> (56 - 8 * i));
    int ctrpos = n; n += 2;
    ge H;
    bool found = false;
    ge G;
#if PX_MERGE8
    for (int ctr = 0; ctr < 256 && !found; ctr++) {
        msg[ctrpos] = (uint8_t)ctr; msg[ctrpos + 1] = 0;
        uint8_t dg[64];
        sha512_single(msg, n, dg);
        if (!ge_frombytes(H, dg)) continue;
        ge_scalarmult_const(G, H);           // (64x mod L) * H = 8 * (x * (8H))
        if (ge_is_identity(G)) continue;     // H was a torsion point
        found = true;
    }
    if (!found) return false;
    if (dbgH) { ge t; ge_mul8(t, H); ge_tobytes(dbgH, t); }
#else
    for (int ctr = 0; ctr < 256 && !found; ctr++) {
        msg[ctrpos] = (uint8_t)ctr; msg[ctrpos + 1] = 0;
        uint8_t dg[64];
        sha512_single(msg, n, dg);
        if (!ge_frombytes(H, dg)) continue;
        ge_mul8(H, H);
        if (ge_is_identity(H)) continue;
        found = true;
    }
    if (!found) return false;
    if (dbgH) ge_tobytes(dbgH, H);
    ge_scalarmult_const(G, H);
    ge_mul8(G, G);
#endif
    uint8_t m2[35];
    m2[0] = 0x03; m2[1] = 0x03;
    ge_tobytes(m2 + 2, G);
    m2[34] = 0x00;
    if (dbgCG) for (int i = 0; i < 32; i++) dbgCG[i] = m2[2 + i];
    sha512_single(m2, 35, beta);
    return true;
}
__device__ __forceinline__ int trailing_zero_bits(const uint8_t beta[64]) {
    int n = 0;
    for (int i = 63; i >= 0; i--) {
        if (beta[i] == 0) { n += 8; continue; }
        return n + __ffs(beta[i]) - 1;
    }
    return n;
}
__device__ __forceinline__ bool below(const uint8_t a[64], const uint8_t b[64]) {
    for (int i = 0; i < 64; i++) { if (a[i] < b[i]) return true; if (a[i] > b[i]) return false; }
    return false;
}

__global__ void __launch_bounds__(PX_LB_THREADS, PX_LB_BLOCKS) mine_kernel(const JobDev *job, uint64_t nonce_base, uint64_t stride, int iters, Hit *hits, unsigned *nhits) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t nonce = nonce_base + gid;
    JobDev j = *job;
    for (int it = 0; it < iters; it++, nonce += stride) {
        uint8_t beta[64];
        if (!vrf_beta(j, nonce, beta, nullptr, nullptr)) continue;
        if (trailing_zero_bits(beta) >= j.k && below(beta, j.beat)) {
            unsigned idx = atomicAdd(nhits, 1u);
            if (idx < MAX_HITS) { hits[idx].nonce = nonce; for (int i = 0; i < 64; i++) hits[idx].beta[i] = beta[i]; }
        }
    }
}


// ----------------------------------------------------------------------------
// compacted pipeline (PX_COMPACT): the try-and-increment loop is where warps
// diverge — each lane needs a geometric number of decompressions and the warp
// waits for the slowest. So the work is split: h2c_kernel does exactly one
// decompression per item and queues the failures for the next pass, then
// mult_kernel runs the scalar multiplication over the found points with no
// divergence at all. The passes are launched blindly; an empty pass costs a
// few microseconds.
// ----------------------------------------------------------------------------
struct HPoint { fe X, Y; uint64_t nonce; };
#ifdef PX_SPLIT_FINISH
struct ProjectivePoint { fe X,Y,Z; uint64_t nonce; unsigned valid; };
#endif
struct Scratch {
#ifdef PX_SPLIT_FINISH
    ProjectivePoint *projective;
#endif
    HPoint *pts; uint64_t *nonce[2]; uint8_t *ctr[2]; unsigned *cnt; size_t n;
#if PX_TAIL_SPLIT
    cudaStream_t sa, sb; cudaEvent_t ev0, evk, evda, evdb; int sms;
#endif
#if PX_H2C_MID
    uint64_t *mid;   // per launch: SHA-512 state after the job-constant rounds 0..7 + message words 0..8 (see sha_prep_kernel)
#endif
};
#define PX_MAX_PASSES 12
#if PX_H2C_MID
// PX_H2C_MID: the hash-to-curve message is 0x03 0x01 | pk (32) | prefix | nonce (8, big endian) | ctr | 0. With the usual
// 37-byte prefix the first 71 bytes are the same for every nonce of a job, so message words 0..7 and with them rounds 0..7
// of SHA-512 are job constants: one thread computes that state once per launch and every hash starts at round 8. Word 8
// carries the nonce's top byte, word 9 the other seven and the counter. Any other prefix length takes the generic hash.
#define PX_MID_PREFIX 37
__global__ void sha_prep_kernel(const JobDev *job, uint64_t *mid) {
    if (threadIdx.x || blockIdx.x || job->prefix_len != PX_MID_PREFIX) return;
    uint8_t m[72];
    int n = 0;
    m[n++] = 0x03; m[n++] = 0x01;
    for (int k = 0; k < 32; k++) m[n++] = job->pk[k];
    for (int k = 0; k < PX_MID_PREFIX; k++) m[n++] = job->prefix[k];
    m[n++] = 0;                                   // byte 71 = nonce top byte, ORed in per nonce
    uint64_t W[9];
    for (int i = 0; i < 9; i++) { uint64_t w = 0; for (int j = 0; j < 8; j++) w = (w << 8) | m[i * 8 + j]; W[i] = w; }
    uint64_t a = 0x6a09e667f3bcc908ULL, b = 0xbb67ae8584caa73bULL, c = 0x3c6ef372fe94f82bULL, d = 0xa54ff53a5f1d36f1ULL;
    uint64_t e = 0x510e527fade682d1ULL, f = 0x9b05688c2b3e6c1fULL, g = 0x1f83d9abfb41bd6bULL, h = 0x5be0cd19137e2179ULL;
    for (int i = 0; i < 8; i++) {
        uint64_t S1 = rotr64(e, 14) ^ rotr64(e, 18) ^ rotr64(e, 41);
        uint64_t ch = (e & f) ^ (~e & g);
        uint64_t t1 = h + S1 + ch + c_K512[i] + W[i];
        uint64_t S0 = rotr64(a, 28) ^ rotr64(a, 34) ^ rotr64(a, 39);
        uint64_t mj = (a & b) ^ (a & c) ^ (b & c);
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + S0 + mj;
    }
    mid[0] = a; mid[1] = b; mid[2] = c; mid[3] = d; mid[4] = e; mid[5] = f; mid[6] = g; mid[7] = h;
    for (int i = 0; i < 9; i++) mid[8 + i] = W[i];
}
// the hash-to-curve SHA-512 from round 8 on; identical arithmetic to sha512_single for the 81-byte message
__device__ __forceinline__ void sha512_h2c_mid_st(const uint64_t *mid, uint64_t nonce, uint8_t ctr, uint64_t st_out[8]) {
    uint64_t W[16];
    #pragma unroll
    for (int i = 0; i < 8; i++) W[i] = mid[8 + i];
    W[8] = mid[16] | (nonce >> 56);
    W[9] = (nonce << 8) | ctr;
    W[10] = 0x0080000000000000ULL;                // byte 80 = 0, byte 81 = 0x80 (padding)
    W[11] = 0; W[12] = 0; W[13] = 0; W[14] = 0;
    W[15] = (uint64_t)(2 + 32 + PX_MID_PREFIX + 8 + 2) * 8;
    uint64_t a = mid[0], b = mid[1], c = mid[2], d = mid[3], e = mid[4], f = mid[5], g = mid[6], h = mid[7];
    #pragma unroll
    for (int i = 8; i < 80; i++) {
        uint64_t wi;
        if (i < 16) wi = W[i];
        else {
            uint64_t w15 = W[(i - 15) & 15], w2 = W[(i - 2) & 15];
            uint64_t s0 = rotr64(w15, 1) ^ rotr64(w15, 8) ^ (w15 >> 7);
            uint64_t s1 = rotr64(w2, 19) ^ rotr64(w2, 61) ^ (w2 >> 6);
            wi = W[i & 15] + s0 + W[(i - 7) & 15] + s1;
            W[i & 15] = wi;
        }
        uint64_t S1 = rotr64(e, 14) ^ rotr64(e, 18) ^ rotr64(e, 41);
        uint64_t ch = (e & f) ^ (~e & g);
        uint64_t t1 = h + S1 + ch + c_K512[i] + wi;
        uint64_t S0 = rotr64(a, 28) ^ rotr64(a, 34) ^ rotr64(a, 39);
        uint64_t mj = (a & b) ^ (a & c) ^ (b & c);
        uint64_t t2 = S0 + mj;
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    uint64_t st[8] = { a + 0x6a09e667f3bcc908ULL, b + 0xbb67ae8584caa73bULL, c + 0x3c6ef372fe94f82bULL, d + 0xa54ff53a5f1d36f1ULL,
                       e + 0x510e527fade682d1ULL, f + 0x9b05688c2b3e6c1fULL, g + 0x1f83d9abfb41bd6bULL, h + 0x5be0cd19137e2179ULL };
    for (int i = 0; i < 8; i++) st_out[i] = st[i];
}
__device__ void sha512_h2c_mid(const uint64_t *mid, uint64_t nonce, uint8_t ctr, uint8_t out[64]) {
    uint64_t st[8]; sha512_h2c_mid_st(mid, nonce, ctr, st);
    for (int i = 0; i < 8; i++) for (int j = 0; j < 8; j++) out[i * 8 + j] = st[i] >> (56 - 8 * j);
}
#endif

#ifndef PX_LB_H2C
#define PX_LB_H2C 1
#endif
// one hash-to-curve attempt for (nonce, ctr): true and H decoded when the hash is a valid encoding
__device__ __forceinline__ bool h2c_try(const JobDev *job, const uint64_t *mid, uint64_t nonce, uint8_t ctr, ge &H) {
    uint8_t dg[64];
#if PX_H2C_MID
    if (job->prefix_len == PX_MID_PREFIX) sha512_h2c_mid(mid, nonce, ctr, dg);
    else
#else
    (void)mid;
#endif
    {
        uint8_t msg[128];
        int n = 0;
        msg[n++] = 0x03; msg[n++] = 0x01;
        for (int k = 0; k < 32; k++) msg[n++] = job->pk[k];
        for (int k = 0; k < job->prefix_len; k++) msg[n++] = job->prefix[k];
        for (int k = 0; k < 8; k++) msg[n++] = (uint8_t)(nonce >> (56 - 8 * k));
        msg[n++] = ctr; msg[n++] = 0;
        sha512_single(msg, n, dg);
    }
    return ge_frombytes(H, dg);
}
__global__ void __launch_bounds__(256, PX_LB_H2C) h2c_kernel(const JobDev *job, const uint64_t *mid, uint64_t nonce_base, const uint64_t *in_nonce, const uint8_t *in_ctr, const unsigned *n_in_ptr,
                                                  HPoint *pts, unsigned *n_pts, uint64_t *out_nonce, uint8_t *out_ctr, unsigned *n_out) {
    unsigned n_in = *n_in_ptr;
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x, total = gridDim.x * blockDim.x;
    if (gid >= n_in) return;
    for (unsigned i = gid; i < n_in; i += total) {
        uint64_t nonce = in_nonce ? in_nonce[i] : nonce_base + i;
        uint8_t ctr = in_nonce ? in_ctr[i] : 0;
        ge H;
        if (h2c_try(job, mid, nonce, ctr, H)) {
            unsigned o = atomicAdd(n_pts, 1u);
            fe_copy(pts[o].X, H.X); fe_copy(pts[o].Y, H.Y); pts[o].nonce = nonce;
        } else if (ctr < 255) {
            unsigned o = atomicAdd(n_out, 1u);
            out_nonce[o] = nonce; out_ctr[o] = ctr + 1;
        }
    }
}
#if PX_H2C_TAIL
// PX_H2C_TAIL = K: passes K..PX_MAX_PASSES-1 merged into one launch. Each queued item retries its counters in a loop up
// to the same bound as the separate passes (counters < PX_MAX_PASSES), so the same points come out; the few items left
// after K compacted passes no longer pay one launch per counter.
__global__ void __launch_bounds__(256, PX_LB_H2C) h2c_tail_kernel(const JobDev *job, const uint64_t *mid, const uint64_t *in_nonce, const uint8_t *in_ctr, const unsigned *n_in_ptr,
                                                       HPoint *pts, unsigned *n_pts) {
    unsigned n_in = *n_in_ptr;
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x, total = gridDim.x * blockDim.x;
    for (unsigned i = gid; i < n_in; i += total) {
        uint64_t nonce = in_nonce[i];
        for (unsigned c = in_ctr[i]; c < PX_MAX_PASSES; c++) {
            ge H;
            if (h2c_try(job, mid, nonce, (uint8_t)c, H)) {
                unsigned o = atomicAdd(n_pts, 1u);
                fe_copy(pts[o].X, H.X); fe_copy(pts[o].Y, H.Y); pts[o].nonce = nonce;
                break;
            }
        }
    }
}
#endif

// one point: cofactor, scalar multiplication, cofactor; false for the identity
__device__ __forceinline__ bool mult_point(ge &G, const HPoint &pt) {
    ge H;
#if PX_MERGE8
    fe_copy(H.X, pt.X); fe_copy(H.Y, pt.Y); fe_1(H.Z); fe_mul(H.T, pt.X, pt.Y); // affine: T = X*Y
    ge_scalarmult_const(G, H);           // (64x mod L) * H
    return !ge_is_identity(G);           // identity only for a torsion H
#else
    fe_copy(H.X, pt.X); fe_copy(H.Y, pt.Y); fe_1(H.Z); // T is not read by the doublings
    ge_mul8(H, H);
    if (ge_is_identity(H)) return false;
    ge_scalarmult_const(G, H);
    ge_mul8(G, G);
    return true;
#endif
}
// compress with a precomputed 1/Z, hash, test
__device__ __forceinline__ void finish_point(const JobDev *job, const ge &G, const fe &zi, uint64_t nonce, Hit *hits, unsigned *nhits) {
    fe x, y;
    fe_mul(x, G.X, zi); fe_mul(y, G.Y, zi);
    uint8_t m2[35];
    m2[0] = 0x03; m2[1] = 0x03;
    fe_tobytes(m2 + 2, y);
    m2[33] ^= (fe_isnegative(x) ? 0x80 : 0);
    m2[34] = 0x00;
    uint8_t beta[64];
    sha512_single(m2, 35, beta);
    if (trailing_zero_bits(beta) >= job->k && below(beta, job->beat)) {
        unsigned idx = atomicAdd(nhits, 1u);
        if (idx < MAX_HITS) { hits[idx].nonce = nonce; for (int k = 0; k < 64; k++) hits[idx].beta[k] = beta[k]; }
    }
}
// One point per step. Sharing one inversion between two points (Montgomery's
// trick) was measured at no gain: the point parked in local memory while the
// second is multiplied costs what the saved inversion brings (RTX 3070, 2026-10-19).
#ifndef PX_LB_MULT_BLOCKS
#define PX_LB_MULT_BLOCKS PX_LB_BLOCKS
#endif
// two points per thread share one inversion by default: +1.4 % (RTX 3070) to
// +2.9 % (RTX PRO 6000) once the loop above fit the instruction cache; larger
// batches park too much (PLATFORMS.md)
#ifndef PX_INV_BATCH
#define PX_INV_BATCH 2
#endif
__global__ void __launch_bounds__(PX_LB_THREADS, PX_LB_MULT_BLOCKS) mult_kernel(const JobDev *job, const HPoint *pts, const unsigned *n_ptr, Hit *hits, unsigned *nhits) {
    unsigned n = *n_ptr;
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x, total = gridDim.x * blockDim.x;
#if PX_INV_BATCH > 1
    // PX_INV_BATCH points per thread share one inversion (Montgomery's trick):
    // the multiplied points are parked, the prefix products of their Z give
    // every 1/Z from a single fe_invert at 3 multiplications per point.
    for (unsigned base = gid; base < n; base += total * PX_INV_BATCH) {
        fe X[PX_INV_BATCH], Y[PX_INV_BATCH], Z[PX_INV_BATCH], P[PX_INV_BATCH];
        uint64_t nonce[PX_INV_BATCH]; bool ok[PX_INV_BATCH];
        for (int k = 0; k < PX_INV_BATCH; k++) {
            unsigned i = base + (unsigned)k * total;
            ge G; ok[k] = i < n && mult_point(G, pts[i]);
            if (ok[k]) { fe_copy(X[k], G.X); fe_copy(Y[k], G.Y); fe_copy(Z[k], G.Z); nonce[k] = pts[i].nonce; } else fe_1(Z[k]);
            if (k == 0) fe_copy(P[0], Z[0]); else fe_mul(P[k], P[k - 1], Z[k]);
        }
        fe inv; fe_invert(inv, P[PX_INV_BATCH - 1]);
        for (int k = PX_INV_BATCH - 1; k > 0; k--) {
            fe zi; fe_mul(zi, inv, P[k - 1]);
            fe_mul(inv, inv, Z[k]);
            if (ok[k]) { ge G; fe_copy(G.X, X[k]); fe_copy(G.Y, Y[k]); finish_point(job, G, zi, nonce[k], hits, nhits); }
        }
        if (ok[0]) { ge G; fe_copy(G.X, X[0]); fe_copy(G.Y, Y[0]); finish_point(job, G, inv, nonce[0], hits, nhits); }
    }
#else
    for (unsigned i = gid; i < n; i += total) {
        ge G;
        if (!mult_point(G, pts[i])) continue;
        fe zi;
        fe_invert(zi, G.Z);
        finish_point(job, G, zi, pts[i].nonce, hits, nhits);
    }
#endif
}
#ifdef PX_SPLIT_FINISH
// Keep only one point live in the expensive scalar stage. Normalization is
// a separate stage so a larger inversion batch needs no parked scalar tables.
#ifndef PX_LB_SCALAR
#define PX_LB_SCALAR 2
#endif
#ifndef PX_SCALAR_THREADS
#define PX_SCALAR_THREADS 256
#endif
__global__ void __launch_bounds__(PX_SCALAR_THREADS, PX_LB_SCALAR) scalar_kernel(const HPoint *pts,const unsigned *n_ptr,ProjectivePoint *out) {
    unsigned n=*n_ptr;
    for(unsigned i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x) {
        ge G; bool ok=mult_point(G,pts[i]);
        out[i].valid=ok;out[i].nonce=pts[i].nonce;
        if(ok){fe_copy(out[i].X,G.X);fe_copy(out[i].Y,G.Y);fe_copy(out[i].Z,G.Z);}
        else fe_1(out[i].Z);
    }
}
__global__ void __launch_bounds__(256, 1) finish_batch_kernel(const JobDev *job,const ProjectivePoint *pts,const unsigned *n_ptr,Hit *hits,unsigned *nhits) {
    unsigned n=*n_ptr,total=gridDim.x*blockDim.x;
    for(unsigned base=blockIdx.x*blockDim.x+threadIdx.x;base<n;base+=total*PX_SPLIT_FINISH) {
        fe prefix[PX_SPLIT_FINISH],prod;fe_1(prod);
        #pragma unroll
        for(int k=0;k<PX_SPLIT_FINISH;k++) {
            unsigned i=base+(unsigned)k*total;
            fe_copy(prefix[k],prod);
            if(i<n) fe_mul(prod,prod,pts[i].Z);
        }
        fe inv;fe_invert(inv,prod);
        #pragma unroll
        for(int k=PX_SPLIT_FINISH-1;k>=0;k--) {
            unsigned i=base+(unsigned)k*total;
            if(i<n) {
                fe zi;fe_mul(zi,inv,prefix[k]);fe_mul(inv,inv,pts[i].Z);
                if(pts[i].valid) {
                    ge G;fe_copy(G.X,pts[i].X);fe_copy(G.Y,pts[i].Y);
                    finish_point(job,G,zi,pts[i].nonce,hits,nhits);
                }
            }
        }
    }
}
#endif

#if PX_LADDER
// ----------------------------------------------------------------------------
// PX_LADDER: the scalar stage as an x-only Montgomery ladder on Curve25519
// (RFC 7748 step: 5M + 4S + one small multiplication per scalar bit) instead of
// the Edwards wNAF. The ladder keeps no table: two projective points and the
// base u fit in registers, so the stage has no local-memory traffic at all (the
// wNAF table and its spills cost about 15 GB of DRAM traffic per launch on an
// RTX 5090 and with it board power and clock under a power cap).
//   prep_kernel   Edwards (x, y) of H -> Montgomery (u, v) = ((1+y)/(1-y), c*u/x),
//                 c = sqrt(-486664), one shared inversion per PX_PREP_BATCH points
//   ladder_kernel u -> (X:Z) of s*H and (s+1)*H for the job scalar s; then the
//                 v-coordinate of s*H is recovered exactly from v of H
//                 (Okeya-Sakurai, as in Costello-Smith "Montgomery curves and
//                 their arithmetic", Alg. 5), and the point is written back in
//                 projective Edwards form for the unchanged finish stage:
//                 x = c*U/V, y = (U-Z)/(U+Z) over the common denominator V*(U+Z).
// The result is the exact point, with the sign of x, so every output is the same
// beta as the reference pipeline (no candidate sign, no host-side re-check).
// ----------------------------------------------------------------------------
__device__ fe d_MONT_C;   // sqrt(-486664), the scale of the Edwards <-> Montgomery map
__constant__ int c_ladder_top;   // index of the scalar's top set bit
#if PX_XONLY
// x-only scalar: t = 8x mod L or L - t, whichever is shorter (u and y are the same for a point and its negative); the
// ladder runs over t and three doublings follow: 8*(t*H) = 64x*H on the prime-order part, the torsion part of H is killed
__constant__ uint8_t c_xscalar[32];
__constant__ int c_xladder_top;
#endif
#define PX_MONT_A 486662u
#define PX_MONT_A24 121665u
// branch-free add / sub for the ladder hot path: identical to fe_add / fe_sub except that a carry (borrow) out of
// limb 1 after the 38-fold is dropped. That needs limb 0 within 38 of 2^32 and limb 1 = 2^32 - 1 at once, about
// 2^-58 per operation on the uniformly distributed values of a ladder; a corrupted point only produces a wrong
// candidate, which the exact resolve stage discards (PX_XONLY). Never used where values are structured.
__device__ __forceinline__ void fe_add_nb(fe &h, const fe &f, const fe &g) {
    uint32_t c;
    asm("add.cc.u32 %0, %9, %17;\n\t"
        "addc.cc.u32 %1, %10, %18;\n\t" "addc.cc.u32 %2, %11, %19;\n\t" "addc.cc.u32 %3, %12, %20;\n\t"
        "addc.cc.u32 %4, %13, %21;\n\t" "addc.cc.u32 %5, %14, %22;\n\t" "addc.cc.u32 %6, %15, %23;\n\t"
        "addc.cc.u32 %7, %16, %24;\n\t" "addc.u32 %8, 0, 0;"
        : "=&r"(h.v[0]), "=&r"(h.v[1]), "=&r"(h.v[2]), "=&r"(h.v[3]), "=&r"(h.v[4]), "=&r"(h.v[5]), "=&r"(h.v[6]), "=&r"(h.v[7]), "=&r"(c)
        : "r"(f.v[0]), "r"(f.v[1]), "r"(f.v[2]), "r"(f.v[3]), "r"(f.v[4]), "r"(f.v[5]), "r"(f.v[6]), "r"(f.v[7]),
          "r"(g.v[0]), "r"(g.v[1]), "r"(g.v[2]), "r"(g.v[3]), "r"(g.v[4]), "r"(g.v[5]), "r"(g.v[6]), "r"(g.v[7]));
#if PX_ADDFOLD_ALU
    {   uint32_t t = (0u - c) & 38u;   // select on the integer pipe instead of a multiply
        asm("add.cc.u32 %0, %0, %2;\n\taddc.u32 %1, %1, 0;" : "+r"(h.v[0]), "+r"(h.v[1]) : "r"(t)); }
#else
    asm("mad.lo.cc.u32 %0, %2, 38, %0;\n\taddc.u32 %1, %1, 0;" : "+r"(h.v[0]), "+r"(h.v[1]) : "r"(c));
#endif
}
__device__ __forceinline__ void fe_sub_nb(fe &h, const fe &f, const fe &g) {
    uint32_t b;
    asm("sub.cc.u32 %0, %9, %17;\n\t"
        "subc.cc.u32 %1, %10, %18;\n\t" "subc.cc.u32 %2, %11, %19;\n\t" "subc.cc.u32 %3, %12, %20;\n\t"
        "subc.cc.u32 %4, %13, %21;\n\t" "subc.cc.u32 %5, %14, %22;\n\t" "subc.cc.u32 %6, %15, %23;\n\t"
        "subc.cc.u32 %7, %16, %24;\n\t" "subc.u32 %8, 0, 0;"
        : "=&r"(h.v[0]), "=&r"(h.v[1]), "=&r"(h.v[2]), "=&r"(h.v[3]), "=&r"(h.v[4]), "=&r"(h.v[5]), "=&r"(h.v[6]), "=&r"(h.v[7]), "=&r"(b)
        : "r"(f.v[0]), "r"(f.v[1]), "r"(f.v[2]), "r"(f.v[3]), "r"(f.v[4]), "r"(f.v[5]), "r"(f.v[6]), "r"(f.v[7]),
          "r"(g.v[0]), "r"(g.v[1]), "r"(g.v[2]), "r"(g.v[3]), "r"(g.v[4]), "r"(g.v[5]), "r"(g.v[6]), "r"(g.v[7]));
#if PX_ADDFOLD_ALU
    uint32_t t = b & 38u;              // b is 0 or all ones
#else
    uint32_t t = (b & 1u) * 38u;
#endif
    asm("sub.cc.u32 %0, %0, %2;\n\tsubc.u32 %1, %1, 0;" : "+r"(h.v[0]), "+r"(h.v[1]) : "r"(t));
}
// h = f * k for a small constant k < 2^20: eight 32x32->64 products, one carry chain, the top word folded as 38
template<int NB> __device__ __forceinline__ void fe_mul_small_t(fe &h, const fe &f, uint32_t k) {
    uint32_t lo[8], hi[8], top;
#if PX_SMALLMUL_WIDE
    // one 32x32->64 product per limb (IMAD.WIDE), then the same carry chain
    #pragma unroll
    for (int i = 0; i < 8; i++) { uint64_t t = (uint64_t)f.v[i] * k; lo[i] = (uint32_t)t; hi[i] = (uint32_t)(t >> 32); }
#else
    #pragma unroll
    for (int i = 0; i < 8; i++) { lo[i] = f.v[i] * k; hi[i] = __umulhi(f.v[i], k); }
#endif
    asm("add.cc.u32 %1, %9, %17;\n\t"
        "addc.cc.u32 %2, %10, %18;\n\t" "addc.cc.u32 %3, %11, %19;\n\t" "addc.cc.u32 %4, %12, %20;\n\t"
        "addc.cc.u32 %5, %13, %21;\n\t" "addc.cc.u32 %6, %14, %22;\n\t" "addc.cc.u32 %7, %15, %23;\n\t"
        "addc.u32 %8, %16, 0;\n\t" "mov.u32 %0, %24;"
        : "=&r"(h.v[0]), "=&r"(h.v[1]), "=&r"(h.v[2]), "=&r"(h.v[3]), "=&r"(h.v[4]), "=&r"(h.v[5]), "=&r"(h.v[6]), "=&r"(h.v[7]), "=&r"(top)
        : "r"(lo[1]), "r"(lo[2]), "r"(lo[3]), "r"(lo[4]), "r"(lo[5]), "r"(lo[6]), "r"(lo[7]), "r"(hi[7]),
          "r"(hi[0]), "r"(hi[1]), "r"(hi[2]), "r"(hi[3]), "r"(hi[4]), "r"(hi[5]), "r"(hi[6]), "r"(lo[0]));
    fe_addtop_t<NB>(h, top * 38u, 0);   // top < 2^20: top*38 < 2^26
}
__device__ __forceinline__ void fe_mul_small(fe &h, const fe &f, uint32_t k) { fe_mul_small_t<0>(h, f, k); }
// one ladder step: (x2:z2) <- 2*(x2:z2), (x3:z3) <- (x2:z2) + (x3:z3), whose difference has affine u = x1
__device__ __forceinline__ void mlad_step(fe &x2, fe &z2, fe &x3, fe &z3, const fe &x1) {
    fe A, B, C, D, AA, BB, E, DA, CB;
    fe_add(A, x2, z2); fe_sub(B, x2, z2);
    fe_add(C, x3, z3); fe_sub(D, x3, z3);
    fe_mul(DA, D, A); fe_mul(CB, C, B);
    fe_sq(AA, A); fe_sq(BB, B);
    fe_add(x3, DA, CB); fe_sq(x3, x3);
    fe_sub(z3, DA, CB); fe_sq(z3, z3); fe_mul(z3, z3, x1);
    fe_mul(x2, AA, BB);
    fe_sub(E, AA, BB);
    fe_mul_small(C, E, PX_MONT_A24); fe_add(C, C, AA); fe_mul(z2, E, C);
}
#ifndef PX_LAD_FSS
#define PX_LAD_FSS 0
#endif
#ifndef PX_LAD_FSM
#define PX_LAD_FSM 0
#endif
#if PX_LADDER_NB
// the same step on the branch-free arithmetic (no rare-carry branches; see fe_add_nb)
__device__ __forceinline__ void mlad_step_nb(fe &x2, fe &z2, fe &x3, fe &z3, const fe &x1) {
    fe A, B, C, D, AA, BB, E, DA, CB;
    fe_add_nb(A, x2, z2); fe_sub_nb(B, x2, z2);
    fe_add_nb(C, x3, z3); fe_sub_nb(D, x3, z3);
#if PX_LAD_KARA
#define PX_LMUL(h, f, g) fe_mul_kara<1>(h, f, g)
#else
#define PX_LMUL(h, f, g) fe_mul_t<1, PX_LAD_FSM>(h, f, g)
#endif
    PX_LMUL(DA, D, A); PX_LMUL(CB, C, B);
    fe_sq_impl<PX_LAD_FSS, 1>(AA, A); fe_sq_impl<PX_LAD_FSS, 1>(BB, B);
    fe_add_nb(x3, DA, CB); fe_sq_impl<PX_LAD_FSS, 1>(x3, x3);
    fe_sub_nb(z3, DA, CB); fe_sq_impl<PX_LAD_FSS, 1>(z3, z3); PX_LMUL(z3, z3, x1);
    PX_LMUL(x2, AA, BB);
    fe_sub_nb(E, AA, BB);
    fe_mul_small_t<1>(C, E, PX_MONT_A24); fe_add_nb(C, C, AA); PX_LMUL(z2, E, C);
#undef PX_LMUL
}
#endif
#if PX_LADDER_NB
#define PX_XSTEP mlad_step_nb
#else
#define PX_XSTEP mlad_step
#endif
#ifndef PX_PREP_BATCH
#define PX_PREP_BATCH 16
#endif
// in place: pts[i].X = u, pts[i].Y = v (Montgomery, affine); u = 0 marks a point the ladder skips (x = 0: y = +-1, torsion)
__global__ void __launch_bounds__(256, 1) prep_kernel(const unsigned *n_ptr, HPoint *pts) {
    unsigned n = *n_ptr, total = gridDim.x * blockDim.x;
    for (unsigned base = blockIdx.x * blockDim.x + threadIdx.x; base < n; base += total * PX_PREP_BATCH) {
        fe prefix[PX_PREP_BATCH], prod; fe_1(prod);
        #pragma unroll
        for (int k = 0; k < PX_PREP_BATCH; k++) {
            unsigned i = base + (unsigned)k * total;
            fe_copy(prefix[k], prod);
            if (i < n) {
                fe one, d; fe_1(one);
                fe_sub(d, one, pts[i].Y); fe_mul(d, d, pts[i].X);       // (1 - y) * x
                if (!fe_iszero(d)) fe_mul(prod, prod, d);
            }
        }
        fe inv; fe_invert(inv, prod);
        #pragma unroll
        for (int k = PX_PREP_BATCH - 1; k >= 0; k--) {
            unsigned i = base + (unsigned)k * total;
            if (i < n) {
                fe x, y, one, d, di, p1, u, v;
                fe_copy(x, pts[i].X); fe_copy(y, pts[i].Y); fe_1(one);
                fe_sub(d, one, y); fe_mul(d, d, x);
                if (fe_iszero(d)) { fe_0(pts[i].X); fe_0(pts[i].Y); continue; }
                fe_mul(di, inv, prefix[k]); fe_mul(inv, inv, d);        // di = 1 / ((1 - y) x)
                fe_add(p1, one, y); fe_mul(v, p1, di);                  // (1 + y) / ((1 - y) x)
                fe_mul(u, v, x);                                        // u = (1 + y) / (1 - y)
                fe_mul(v, v, d_MONT_C);                                 // v = c u / x
                fe_copy(pts[i].X, u); fe_copy(pts[i].Y, v);
            }
        }
    }
}
#ifndef PX_LB_LADDER
#define PX_LB_LADDER 2
#endif
__global__ void __launch_bounds__(256, PX_LB_LADDER) ladder_kernel(const HPoint *pts, const unsigned *n_ptr, ProjectivePoint *out) {
    unsigned n = *n_ptr;
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        fe xP, yP; fe_copy(xP, pts[i].X); fe_copy(yP, pts[i].Y);
        out[i].nonce = pts[i].nonce;
        if (fe_iszero(xP)) { out[i].valid = 0; fe_1(out[i].Z); continue; }
        fe x2, z2, x3, z3;
        fe_1(x2); fe_0(z2); fe_copy(x3, xP); fe_1(z3);              // R0 = O, R1 = P
        // the scalar is the same for every thread: the branch below is warp-uniform
        for (int b = c_ladder_top; b >= 0; b--) {
            if ((c_scalar[b >> 3] >> (b & 7)) & 1) mlad_step(x3, z3, x2, z2, xP);
            else mlad_step(x2, z2, x3, z3, xP);
        }
        // Q = (x2:z2) = s*P, R = (x3:z3) = (s+1)*P: recover v of Q (B = 1)
        fe v1, v2, v3, v4, X, Y, Z;
        fe_mul(v1, xP, z2);
        fe_add(v2, x2, v1); fe_sub(v3, x2, v1);
        fe_sq(v3, v3); fe_mul(v3, v3, x3);
        fe_mul_small(v1, z2, 2u * PX_MONT_A);
        fe_add(v2, v2, v1);
        fe_mul(v4, xP, x2); fe_add(v4, v4, z2);
        fe_mul(v2, v2, v4);
        fe_mul(v1, v1, z2);
        fe_sub(v2, v2, v1);
        fe_mul(v2, v2, z3);
        fe_sub(Y, v2, v3);
        fe_add(v1, yP, yP);
        fe_mul(v1, v1, z2); fe_mul(v1, v1, z3);
        fe_mul(X, v1, x2); fe_mul(Z, v1, z2);
        // Montgomery (X:Y:Z) -> Edwards over the denominator Y (X + Z)
        fe s, t;
        fe_add(s, X, Z); fe_sub(t, X, Z);
        fe_mul(out[i].Z, Y, s);
        fe_mul(out[i].Y, t, Y);
        fe_mul(X, X, s); fe_mul(out[i].X, X, d_MONT_C);
        // s*P = O (z2 = 0) only for a torsion P; the denominator is then zero, and must not enter the batch inversion
        out[i].valid = !fe_iszero(out[i].Z);
        if (!out[i].valid) fe_1(out[i].Z);
    }
}
#if PX_XONLY
// ----------------------------------------------------------------------------
// PX_XONLY: hash-to-curve without a square root. A hash is a valid encoding iff
// w = (y^2 - 1)(d y^2 + 1) is a square or zero (the same condition as the square
// root in ge_frombytes, since d y^2 + 1 != 0); that is decided by a Legendre
// symbol computed with a binary GCD on 64-bit approximations (divsteps, 29 per
// round, the 2x2 transition matrix then applied to the full values). The x-only
// ladder needs only y of H. Without x the sign of x(s*H) is unknown, so the
// finish hashes both encodings of y(s*H); every candidate that meets the target
// is then recomputed exactly (vrf_beta, one thread per candidate) and only the
// true output is kept (resolve_kernel): the reported hits equal the reference.
// ----------------------------------------------------------------------------
#ifndef PX_JAC_STEPS
#define PX_JAC_STEPS 29
#endif
#ifndef PX_JAC_MAXR
#define PX_JAC_MAXR 24
#endif
// r = (f*a + g*b) / 2^PX_JAC_STEPS for |f|, |g| <= 2^PX_JAC_STEPS (exact division); returns 1 if the result was
// negative (r is then its absolute value)
__device__ __forceinline__ uint32_t jac_lin(uint32_t r[8], int32_t f, int32_t g, const uint32_t a[8], const uint32_t b[8], int L = 8) {
    // positive and negative parts accumulated separately as unsigned 9-word sums
    uint32_t fa = (uint32_t)(f < 0 ? -f : f), ga = (uint32_t)(g < 0 ? -g : g);
    uint32_t P[9], N[9];
    uint64_t cp = 0, cn = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        if (i >= L) { P[i] = (uint32_t)cp; N[i] = (uint32_t)cn; cp = cn = 0; continue; }   // warp-uniform: a, b < 2^(32 L)
        uint64_t x = (uint64_t)fa * a[i], y = (uint64_t)ga * b[i];
        uint64_t px = f < 0 ? 0 : x, nx = f < 0 ? x : 0, py = g < 0 ? 0 : y, ny = g < 0 ? y : 0;
        // each sum of two 64-bit products plus a carry < 2^34 stays below 2^64 (products < 2^61)
        uint64_t sp = px + py + cp, sn = nx + ny + cn;
        P[i] = (uint32_t)sp; cp = sp >> 32;
        N[i] = (uint32_t)sn; cn = sn >> 32;
    }
    P[8] = (uint32_t)cp; N[8] = (uint32_t)cn;
    uint32_t D[9], bw;
    asm("sub.cc.u32 %0, %10, %19;\n\tsubc.cc.u32 %1, %11, %20;\n\tsubc.cc.u32 %2, %12, %21;\n\tsubc.cc.u32 %3, %13, %22;\n\t"
        "subc.cc.u32 %4, %14, %23;\n\tsubc.cc.u32 %5, %15, %24;\n\tsubc.cc.u32 %6, %16, %25;\n\tsubc.cc.u32 %7, %17, %26;\n\t"
        "subc.cc.u32 %8, %18, %27;\n\tsubc.u32 %9, 0, 0;"
        : "=&r"(D[0]), "=&r"(D[1]), "=&r"(D[2]), "=&r"(D[3]), "=&r"(D[4]), "=&r"(D[5]), "=&r"(D[6]), "=&r"(D[7]), "=&r"(D[8]), "=&r"(bw)
        : "r"(P[0]), "r"(P[1]), "r"(P[2]), "r"(P[3]), "r"(P[4]), "r"(P[5]), "r"(P[6]), "r"(P[7]), "r"(P[8]),
          "r"(N[0]), "r"(N[1]), "r"(N[2]), "r"(N[3]), "r"(N[4]), "r"(N[5]), "r"(N[6]), "r"(N[7]), "r"(N[8]));
    uint32_t neg = bw & 1u;
    // absolute value: (D ^ m) - m with m = all ones when negative
    uint32_t m = 0u - neg;
    #pragma unroll
    for (int i = 0; i < 9; i++) D[i] ^= m;
    asm("sub.cc.u32 %0, %0, %9;\n\tsubc.cc.u32 %1, %1, %9;\n\tsubc.cc.u32 %2, %2, %9;\n\tsubc.cc.u32 %3, %3, %9;\n\t"
        "subc.cc.u32 %4, %4, %9;\n\tsubc.cc.u32 %5, %5, %9;\n\tsubc.cc.u32 %6, %6, %9;\n\tsubc.cc.u32 %7, %7, %9;\n\tsubc.u32 %8, %8, %9;"
        : "+r"(D[0]), "+r"(D[1]), "+r"(D[2]), "+r"(D[3]), "+r"(D[4]), "+r"(D[5]), "+r"(D[6]), "+r"(D[7]), "+r"(D[8]) : "r"(m));
    #pragma unroll
    for (int i = 0; i < 8; i++) r[i] = __funnelshift_r(D[i], D[i + 1], PX_JAC_STEPS);
    return neg;
}
// true when w is a square or zero modulo p (exact: a value the binary GCD does not finish within PX_JAC_MAXR
// rounds, never seen in 2*10^4 random trials beyond 14, is decided by Euler's criterion instead)
__device__ bool fe_is_square_or_zero(const fe &w) {
    uint32_t a[8], b[8];
    fe_canon(a, w);
    b[0] = 0xFFFFFFEDu; b[1] = b[2] = b[3] = b[4] = b[5] = b[6] = 0xFFFFFFFFu; b[7] = 0x7FFFFFFFu;
    uint32_t ls = 0;
    int r = 0;
    for (; r < PX_JAC_MAXR; r++) {
        uint32_t anz = a[0] | a[1] | a[2] | a[3] | a[4] | a[5] | a[6] | a[7];
        if (!anz) break;
#if PX_JAC_B1
        // b = 1: every later step keeps the sign ((2/1) = 1, no swap below 1, (-1/1) = 1), so the symbol is final
        if (b[0] == 1u && (b[1] | b[2] | b[3] | b[4] | b[5] | b[6] | b[7]) == 0u) break;
#endif
        // top word index of max(a, b) (at least 1), then the 64-bit approximations: low 32 bits exact,
        // high 32 bits = the 32 bits of a (b) just below the common bit length n
        uint32_t ah = a[7], al = a[6], bh = b[7], bl = b[6];
        #pragma unroll
        for (int i = 6; i >= 1; i--) {
            bool z = (ah | bh) == 0;
            ah = z ? a[i] : ah; al = z ? a[i - 1] : al; bh = z ? b[i] : bh; bl = z ? b[i - 1] : bl;
        }
        int s = __clz(ah | bh);
        uint32_t ta = __funnelshift_l(al, ah, s), tb = __funnelshift_l(bl, bh, s);
        // n <= 64 (a, b both in words 0..1): use the exact values
        bool small = ((a[2] | a[3] | a[4] | a[5] | a[6] | a[7] | b[2] | b[3] | b[4] | b[5] | b[6] | b[7]) == 0);
#if PX_JAC_TRIM
        if (small) {   // both below 2^64: finish on the exact values, no transition matrix (at most ~128 divsteps)
            uint64_t ea = ((uint64_t)a[1] << 32) | a[0], eb = ((uint64_t)b[1] << 32) | b[0];
            while (ea) {
                uint32_t odd = (uint32_t)ea & 1u;
                if (odd && ea < eb) { ls ^= (((uint32_t)ea & (uint32_t)eb) >> 1) & 1u; uint64_t t = ea; ea = eb; eb = t; }
                if (odd) ea -= eb;
                ea >>= 1;
                ls ^= (((uint32_t)eb + 2u) >> 2) & 1u;
            }
            return eb != 1 || (ls & 1u) == 0u;
        }
        int Lt = 8;
        {   int tl = 7;
            #pragma unroll
            for (int i = 7; i >= 1; i--) if ((a[i] | b[i]) == 0 && tl == i) tl = i - 1;
            Lt = __reduce_max_sync(__activemask(), tl + 1); }
#endif
        uint64_t xa = ((uint64_t)(small ? a[1] : ta) << 32) | a[0];
        uint64_t xb = ((uint64_t)(small ? b[1] : tb) << 32) | b[0];
        int32_t f0 = 1, g0 = 0, f1 = 0, g1 = 1;
#if PX_JAC_MASK == 2
        // select form, sign bits kept at bit 1 of ls: (2/b) = -1 iff bit1(b) ^ bit2(b), reciprocity iff bit1(a) & bit1(b)
        #pragma unroll
        for (int j = 0; j < PX_JAC_STEPS; j++) {
            uint64_t d = xa - xb;
            bool lt = xa < xb, odd = (uint32_t)xa & 1u, sw = odd && lt;
            uint32_t swm = sw ? 0xFFFFFFFFu : 0u;
            ls ^= (uint32_t)xa & (uint32_t)xb & swm;
            uint64_t nb = sw ? xa : xb;
            uint64_t ad = lt ? (0 - d) : d;
            uint64_t na = odd ? ad : xa;
            int32_t df = f0 - f1, dg = g0 - g1;
            int32_t af = lt ? -df : df, ag = lt ? -dg : dg;
            int32_t nf1 = sw ? f0 : f1, ng1 = sw ? g0 : g1;
            f0 = odd ? af : f0; g0 = odd ? ag : g0;
            f1 = nf1 << 1; g1 = ng1 << 1;
            xa = na >> 1; xb = nb;
            uint32_t b32 = (uint32_t)nb;
            ls ^= b32 ^ (b32 >> 1);
        }
#elif PX_JAC_MASK == 3
        // packed form: the transition pairs travel as F = f + 2^32 g in one 64-bit word (only linear updates: select,
        // subtract, double), so each update is one 64-bit operation instead of two; sign bits at bit 1 of ls as above
        {
        uint64_t F0 = 1u, F1 = (uint64_t)1u << 32;
        #pragma unroll
        for (int j = 0; j < PX_JAC_STEPS; j++) {
            bool odd = ((uint32_t)xa & 1u) != 0u;
            bool sw = odd && xa < xb;
            if (sw) ls ^= (uint32_t)xa & (uint32_t)xb;          // both = 3 mod 4: reciprocity
            uint64_t na = sw ? xb : xa, nb = sw ? xa : xb;
            uint64_t nF0 = sw ? F1 : F0, nF1 = sw ? F0 : F1;
            if (odd) { na -= nb; nF0 -= nF1; }
            xa = na >> 1; xb = nb; F0 = nF0; F1 = nF1 << 1;
            uint32_t b32 = (uint32_t)nb;
            ls ^= b32 ^ (b32 >> 1);                              // (2/b) = -1 for b = 3, 5 mod 8
        }
        f0 = (int32_t)(uint32_t)F0; f1 = (int32_t)(uint32_t)F1;
        g0 = (int32_t)((int64_t)(F0 - (uint64_t)(int64_t)f0) >> 32);
        g1 = (int32_t)((int64_t)(F1 - (uint64_t)(int64_t)f1) >> 32);
        }
#elif PX_JAC_MASK == 4
        // packed form as in 3, written as predicated PTX so that every update is one 64-bit select / subtract
        {
        uint64_t F0 = 1u, F1 = (uint64_t)1u << 32;
        #pragma unroll
        for (int j = 0; j < PX_JAC_STEPS; j++) {
            asm("{\n\t.reg .pred po, ps;\n\t.reg .b32 al, bl, t, nbl;\n\t.reg .b64 na, nb, g0, g1;\n\t"
                "cvt.u32.u64 al, %0;\n\tcvt.u32.u64 bl, %1;\n\t"
                "and.b32 t, al, 1;\n\tsetp.ne.u32 po, t, 0;\n\t"
                "setp.lt.and.u64 ps, %0, %1, po;\n\t"
                "and.b32 t, al, bl;\n\t@ps xor.b32 %4, %4, t;\n\t"
                "selp.b64 na, %1, %0, ps;\n\tselp.b64 nb, %0, %1, ps;\n\t"
                "selp.b64 g0, %3, %2, ps;\n\tselp.b64 g1, %2, %3, ps;\n\t"
                "@po sub.s64 na, na, nb;\n\t@po sub.s64 g0, g0, g1;\n\t"
                "shr.b64 %0, na, 1;\n\tmov.b64 %1, nb;\n\tmov.b64 %2, g0;\n\tadd.s64 %3, g1, g1;\n\t"
                "cvt.u32.u64 nbl, nb;\n\tshr.b32 t, nbl, 1;\n\txor.b32 t, t, nbl;\n\txor.b32 %4, %4, t;\n\t}"
                : "+l"(xa), "+l"(xb), "+l"(F0), "+l"(F1), "+r"(ls));
        }
        f0 = (int32_t)(uint32_t)F0; f1 = (int32_t)(uint32_t)F1;
        g0 = (int32_t)((int64_t)(F0 - (uint64_t)(int64_t)f0) >> 32);
        g1 = (int32_t)((int64_t)(F1 - (uint64_t)(int64_t)f1) >> 32);
        }
#elif PX_JAC_MASK
        // mask form: swaps by XOR masks, sign bits accumulated unmasked in ls (only bit 0 is read)
        #pragma unroll
        for (int j = 0; j < PX_JAC_STEPS; j++) {
            uint32_t alo = (uint32_t)xa, odd = alo & 1u;
            uint32_t sw = odd & (uint32_t)(xa < xb);
            uint32_t m32 = 0u - sw; uint64_t m64 = ((uint64_t)m32 << 32) | m32;
            ls ^= (alo & (uint32_t)xb & m32) >> 1;                 // both = 3 mod 4 on a swap
            uint64_t t = (xa ^ xb) & m64; xa ^= t; xb ^= t;
            int32_t tf = (f0 ^ f1) & (int32_t)m32, tg = (g0 ^ g1) & (int32_t)m32;
            f0 ^= tf; f1 ^= tf; g0 ^= tg; g1 ^= tg;
            uint32_t o32 = 0u - odd; uint64_t o64 = ((uint64_t)o32 << 32) | o32;
            xa -= xb & o64; f0 -= f1 & (int32_t)o32; g0 -= g1 & (int32_t)o32;
            xa >>= 1; f1 <<= 1; g1 <<= 1;
            ls ^= ((uint32_t)xb + 2u) >> 2;                        // (2/b) = -1 for b = 3, 5 mod 8
        }
#else
        #pragma unroll
        for (int j = 0; j < PX_JAC_STEPS; j++) {
            uint32_t odd = (uint32_t)xa & 1u;
            bool sw = odd && xa < xb;
            // swap when a odd and a < b, with the reciprocity sign: both = 3 mod 4
            ls ^= sw ? (((uint32_t)xa & (uint32_t)xb) >> 1) & 1u : 0u;
            uint64_t na = sw ? xb : xa, nb = sw ? xa : xb;
            int32_t nf0 = sw ? f1 : f0, ng0 = sw ? g1 : g0, nf1 = sw ? f0 : f1, ng1 = sw ? g0 : g1;
            // a odd: a -= b (even afterwards); then a /= 2 with the (2/b) sign: b = 3, 5 mod 8
            na -= odd ? nb : 0; nf0 -= odd ? nf1 : 0; ng0 -= odd ? ng1 : 0;
            xa = na >> 1; xb = nb; f0 = nf0; g0 = ng0; f1 = nf1 << 1; g1 = ng1 << 1;
            ls ^= (((uint32_t)xb + 2u) >> 2) & 1u;
        }
#endif
        uint32_t na[8], nb[8];
#if PX_JAC_TRIM
        uint32_t nega = jac_lin(na, f0, g0, a, b, Lt);
        (void)jac_lin(nb, f1, g1, a, b, Lt);
#else
        uint32_t nega = jac_lin(na, f0, g0, a, b);
        (void)jac_lin(nb, f1, g1, a, b);
#endif
                              // b stays positive (it is always an earlier a or b)
#if PX_JAC_MASK >= 2
        ls ^= nega ? nb[0] : 0u;                               // (-1/b) = -1 for b = 3 mod 4 (bit 1 of ls)
#else
        ls ^= nega & (nb[0] >> 1);                             // (-1/b) = -1 for b = 3 mod 4 (bit 0 of ls)
#endif
        #pragma unroll
        for (int i = 0; i < 8; i++) { a[i] = na[i]; b[i] = nb[i]; }
    }
    if (r == PX_JAC_MAXR) {    // not finished: Euler's criterion w^((p-1)/2), from w^(2^250-1) and w^11
        // (p-1)/2 = 2^254 - 10 = (2^250 - 1) * 2^4 + 6
        fe t19, t3, t, one, m1; fe_pow22501(t19, t3, w);
        fe w2, w4, w6; fe_sq(w2, w); fe_sq(w4, w2); fe_mul(w6, w4, w2);
        fe_pow2k(t, t19, 4); fe_mul(t, t, w6);
        fe_1(one); fe_neg(m1, one);
        return !fe_eq(t, m1);
    }
    uint32_t bo = b[1] | b[2] | b[3] | b[4] | b[5] | b[6] | b[7];
    if (b[0] != 1u || bo) return true;                        // gcd = p: w = 0, which is a valid encoding (x = 0)
#if PX_JAC_MASK >= 2
    return ((ls >> 1) & 1u) == 0u;
#else
    return (ls & 1u) == 0u;
#endif
}
// one attempt: true and y stored when the hash is a valid encoding
__device__ __forceinline__ bool h2c_try_y(const JobDev *job, const uint64_t *mid, uint64_t nonce, uint8_t ctr, fe &Y) {
#if PX_H2C_MID
    if (job->prefix_len == PX_MID_PREFIX) {
        // y straight from the state words (digest bytes 0..31 little-endian = byte-swapped big-endian words): no byte array
        uint64_t st[8]; sha512_h2c_mid_st(mid, nonce, ctr, st);
        #pragma unroll
        for (int i = 0; i < 4; i++) { Y.v[2 * i] = __byte_perm((uint32_t)(st[i] >> 32), 0, 0x0123); Y.v[2 * i + 1] = __byte_perm((uint32_t)st[i], 0, 0x0123); }
        Y.v[7] &= 0x7FFFFFFFu;
    } else
#else
    (void)mid;
#endif
    {
        uint8_t dg[64];
        uint8_t msg[128];
        int n = 0;
        msg[n++] = 0x03; msg[n++] = 0x01;
        for (int k = 0; k < 32; k++) msg[n++] = job->pk[k];
        for (int k = 0; k < job->prefix_len; k++) msg[n++] = job->prefix[k];
        for (int k = 0; k < 8; k++) msg[n++] = (uint8_t)(nonce >> (56 - 8 * k));
        msg[n++] = ctr; msg[n++] = 0;
        sha512_single(msg, n, dg);
        fe_frombytes(Y, dg);
    }
    fe YY, one, u, v;
    fe_1(one);
    fe_sq(YY, Y); fe_sub(u, YY, one);
    fe_mul(v, YY, d_EDWARDS_D); fe_add(v, v, one);
    fe_mul(u, u, v);
    return fe_is_square_or_zero(u);
}
__global__ void __launch_bounds__(256, PX_LB_H2C) h2cx_kernel(const JobDev *job, const uint64_t *mid, uint64_t nonce_base, const uint64_t *in_nonce, const uint8_t *in_ctr, const unsigned *n_in_ptr,
                                                   HPoint *pts, unsigned *n_pts, uint64_t *out_nonce, uint8_t *out_ctr, unsigned *n_out) {
    unsigned n_in = *n_in_ptr;
    unsigned total = gridDim.x * blockDim.x, lane = threadIdx.x & 31u;
    // warp-uniform loop; one atomic per warp and queue (ballot + prefix count) instead of one per point
    for (unsigned base = blockIdx.x * blockDim.x + (threadIdx.x & ~31u); base < n_in; base += total) {
        unsigned i = base + lane;
        bool act = i < n_in, ok = false;
        uint64_t nonce = 0; uint8_t ctr = 0; fe Y;
        if (act) {
            nonce = in_nonce ? in_nonce[i] : nonce_base + i;
            ctr = in_nonce ? in_ctr[i] : 0;
            ok = h2c_try_y(job, mid, nonce, ctr, Y);
        }
        bool rt = act && !ok && ctr < 255;
        unsigned mo = __ballot_sync(0xffffffffu, ok), mr = __ballot_sync(0xffffffffu, rt);
        unsigned bo = 0, br = 0;
        if (lane == 0) { if (mo) bo = atomicAdd(n_pts, __popc(mo)); if (mr) br = atomicAdd(n_out, __popc(mr)); }
        bo = __shfl_sync(0xffffffffu, bo, 0); br = __shfl_sync(0xffffffffu, br, 0);
        unsigned lt = (1u << lane) - 1u;
        if (ok) { unsigned o = bo + __popc(mo & lt); fe_copy(pts[o].Y, Y); pts[o].nonce = nonce; }
        if (rt) { unsigned o = br + __popc(mr & lt); out_nonce[o] = nonce; out_ctr[o] = ctr + 1; }
    }
}
#if PX_H2C_PERSIST
// PX_H2C_PERSIST: one persistent launch instead of PX_MAX_PASSES passes. Every lane keeps its nonce and retries the
// next counter itself until the hash is a valid encoding (or the same counter bound as the passes is reached), then
// takes the next nonce; the warp refills its free lanes with one atomic. Same points as the passes (first valid
// counter below PX_MAX_PASSES), without the poorly occupied tail passes and the retry queues.
__global__ void __launch_bounds__(256, PX_LB_H2C) h2cr_kernel(const JobDev *job, const uint64_t *mid, uint64_t nonce_base, const unsigned *n_ptr,
                                                    unsigned *next, HPoint *pts, unsigned *n_pts) {
    const unsigned N = *n_ptr, lane = threadIdx.x & 31u, lt = (1u << lane) - 1u;
    unsigned idx = 0xFFFFFFFFu, ctr = 0; bool need = true;
    for (;;) {
        unsigned mn = __ballot_sync(0xffffffffu, need);
        if (mn) {
            unsigned b = 0;
            if (lane == 0) b = atomicAdd(next, (unsigned)__popc(mn));
            b = __shfl_sync(0xffffffffu, b, 0);
            if (need) { unsigned j = b + __popc(mn & lt); idx = j < N ? j : 0xFFFFFFFFu; ctr = 0; need = false; }
        }
        bool act = idx != 0xFFFFFFFFu;
        if (!__any_sync(0xffffffffu, act)) break;
        bool ok = false; fe Y; uint64_t nonce = nonce_base + idx;
        if (act) ok = h2c_try_y(job, mid, nonce, (uint8_t)ctr, Y);
        unsigned mo = __ballot_sync(0xffffffffu, ok), bo = 0;
        if (lane == 0 && mo) bo = atomicAdd(n_pts, __popc(mo));
        bo = __shfl_sync(0xffffffffu, bo, 0);
        if (ok) { unsigned o = bo + __popc(mo & lt); fe_copy(pts[o].Y, Y); pts[o].nonce = nonce; need = true; }
        else if (act) { if (++ctr >= PX_MAX_PASSES) need = true; }
    }
}
#endif
#if PX_CORUN
// one persistent 384-thread block per multiprocessor at <= 128 registers (49152 of 65536): the rest of the register
// file stays free for one 128-thread block of the hash-to-curve / prep / finish kernels of the next slice, which then
// issue on the integer pipe beside the ladder's multiplies
#ifndef PX_LADDER_MAXREG
#define PX_LADDER_MAXREG 128
#define PX_LADDER_TPB 384
#define PX_LADDER_BPS 1
#endif
#define PX_LADDER_ATTR __maxnreg__(PX_LADDER_MAXREG)
#define PX_SMALL_ATTR __launch_bounds__(128, 4)
#else
#define PX_LADDER_ATTR __launch_bounds__(256, PX_LB_LADDER)
#if PX_PFX_MEM
#ifndef PX_LB_SMALL
#define PX_LB_SMALL 3
#endif
#define PX_SMALL_ATTR __launch_bounds__(256, PX_LB_SMALL)
#else
#define PX_SMALL_ATTR __launch_bounds__(256, 1)
#endif
#endif
// separate occupancy for prep and finish (dev knobs); default as PX_SMALL_ATTR
#ifdef PX_LB_PREP
#define PX_PREP_ATTR __launch_bounds__(256, PX_LB_PREP)
#else
#define PX_PREP_ATTR PX_SMALL_ATTR
#endif
#ifdef PX_LB_FIN
#define PX_FIN_ATTR __launch_bounds__(256, PX_LB_FIN)
#else
#define PX_FIN_ATTR PX_SMALL_ATTR
#endif
// in place: pts[i].X = u = (1 + y) / (1 - y); u = 0 marks a point the ladder skips (y = 1: identity, y = -1: order 2)
#if PX_PFX_MEM
// PX_PFX_MEM: the batch inversion keeps its prefix products in the point record itself (X is free until u is written)
// instead of a register array, so the kernel needs few registers and runs at a useful occupancy
__global__ void PX_PREP_ATTR prepx_kernel(const unsigned *n_ptr, HPoint *pts) {
    unsigned n = *n_ptr, total = gridDim.x * blockDim.x;
    for (unsigned base = blockIdx.x * blockDim.x + threadIdx.x; base < n; base += total * PX_PREP_BATCH) {
        fe prod; fe_1(prod);
        #pragma unroll 4
        for (int k = 0; k < PX_PREP_BATCH; k++) {
            unsigned i = base + (unsigned)k * total;
            if (i < n) { fe one, d; fe_1(one); fe_copy(pts[i].X, prod); fe_sub(d, one, pts[i].Y); if (!fe_iszero(d)) fe_mul(prod, prod, d); }
        }
        fe inv; fe_invert(inv, prod);
        #pragma unroll 4
        for (int k = PX_PREP_BATCH - 1; k >= 0; k--) {
            unsigned i = base + (unsigned)k * total;
            if (i < n) {
                fe y, one, d, di, u, pre; fe_copy(y, pts[i].Y); fe_1(one);
                fe_sub(d, one, y);
                if (fe_iszero(d)) { fe_0(pts[i].X); continue; }
                fe_copy(pre, pts[i].X);
                fe_mul(di, inv, pre); fe_mul(inv, inv, d);
                fe_add(u, one, y); fe_mul(u, u, di);
                fe_copy(pts[i].X, u);
            }
        }
    }
}
#else
__global__ void PX_PREP_ATTR prepx_kernel(const unsigned *n_ptr, HPoint *pts, const unsigned *lo_ptr = nullptr) {
    unsigned n = *n_ptr, total = gridDim.x * blockDim.x, lo = lo_ptr ? *lo_ptr : 0u;
    for (unsigned base = lo + blockIdx.x * blockDim.x + threadIdx.x; base < n; base += total * PX_PREP_BATCH) {
        fe prefix[PX_PREP_BATCH], prod; fe_1(prod);
        #pragma unroll
        for (int k = 0; k < PX_PREP_BATCH; k++) {
            unsigned i = base + (unsigned)k * total;
            fe_copy(prefix[k], prod);
            if (i < n) { fe one, d; fe_1(one); fe_sub(d, one, pts[i].Y); if (!fe_iszero(d)) fe_mul(prod, prod, d); }
        }
        fe inv; fe_invert(inv, prod);
        #pragma unroll
        for (int k = PX_PREP_BATCH - 1; k >= 0; k--) {
            unsigned i = base + (unsigned)k * total;
            if (i < n) {
                fe y, one, d, di, u; fe_copy(y, pts[i].Y); fe_1(one);
                fe_sub(d, one, y);
                if (fe_iszero(d)) { fe_0(pts[i].X); continue; }
                fe_mul(di, inv, prefix[k]); fe_mul(inv, inv, d);
                fe_add(u, one, y); fe_mul(u, u, di);
                fe_copy(pts[i].X, u);
            }
        }
    }
}
#endif
// x-only: out.Y = X - Z, out.Z = X + Z of s*P, so y(s*P) = out.Y / out.Z
__global__ void PX_LADDER_ATTR ladderx_kernel(const HPoint *pts, const unsigned *n_ptr, ProjectivePoint *out, const unsigned *lo_ptr = nullptr) {
    unsigned n = *n_ptr, lo = lo_ptr ? *lo_ptr : 0u;
    for (unsigned i = lo + blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        fe xP; fe_copy(xP, pts[i].X);
        out[i].nonce = pts[i].nonce;
        if (fe_iszero(xP)) { out[i].valid = 0; fe_1(out[i].Z); continue; }
        fe x2, z2, x3, z3;
        fe_1(x2); fe_0(z2); fe_copy(x3, xP); fe_1(z3);
#if PX_LADDER_CSWAP
        // one code path: conditional swap by a (warp-uniform) mask before each step, RFC 7748 order
        uint32_t swp = 0;
        for (int b = c_xladder_top; b >= 0; b--) {
            uint32_t kt = (c_xscalar[b >> 3] >> (b & 7)) & 1u;
            uint32_t m = 0u - (swp ^ kt);
            #pragma unroll
            for (int l = 0; l < 8; l++) {
                uint32_t t = (x2.v[l] ^ x3.v[l]) & m; x2.v[l] ^= t; x3.v[l] ^= t;
                uint32_t u = (z2.v[l] ^ z3.v[l]) & m; z2.v[l] ^= u; z3.v[l] ^= u;
            }
            swp = kt;
            PX_XSTEP(x2, z2, x3, z3, xP);
        }
        {   uint32_t m = 0u - swp;
            #pragma unroll
            for (int l = 0; l < 8; l++) {
                uint32_t t = (x2.v[l] ^ x3.v[l]) & m; x2.v[l] ^= t; x3.v[l] ^= t;
                uint32_t u = (z2.v[l] ^ z3.v[l]) & m; z2.v[l] ^= u; z3.v[l] ^= u;
            } }
#else
        for (int b = c_xladder_top; b >= 0; b--) {
            if ((c_xscalar[b >> 3] >> (b & 7)) & 1) PX_XSTEP(x3, z3, x2, z2, xP);
            else PX_XSTEP(x2, z2, x3, z3, xP);
        }
#endif
        #pragma unroll 1
        for (int d = 0; d < 3; d++) {           // x-only doubling: (X+Z)^2 (X-Z)^2 : E ((X+Z)^2 + a24 E), E = 4XZ
            fe A, B, AA, BB, E, C;
            fe_add(A, x2, z2); fe_sub(B, x2, z2); fe_sq(AA, A); fe_sq(BB, B);
            fe_mul(x2, AA, BB); fe_sub(E, AA, BB);
            fe_mul_small(C, E, PX_MONT_A24); fe_add(C, C, AA); fe_mul(z2, E, C);
        }
        fe_sub(out[i].Y, x2, z2); fe_add(out[i].Z, x2, z2);
        // s*P = O (z2 = 0) only for a torsion P; X + Z = 0 cannot occur on the curve. Neither enters the inversion.
        out[i].valid = !fe_iszero(z2) && !fe_iszero(out[i].Z);
        if (!out[i].valid) fe_1(out[i].Z);
    }
}
__device__ __forceinline__ void finishx_point(const JobDev *job, const fe &y, uint64_t nonce, Hit *hits, unsigned *nhits) {
    uint8_t m2[35];
    m2[0] = 0x03; m2[1] = 0x03;
    fe_tobytes(m2 + 2, y);
    m2[34] = 0x00;
    #pragma unroll 1
    for (int sgn = 0; sgn < 2; sgn++) {
        m2[33] = (m2[33] & 0x7f) | (uint8_t)(sgn << 7);
        uint8_t beta[64];
        sha512_single(m2, 35, beta);
        if (trailing_zero_bits(beta) >= job->k && below(beta, job->beat)) {
            unsigned idx = atomicAdd(nhits, 1u);
            if (idx < MAX_HITS) { hits[idx].nonce = nonce; for (int k = 0; k < 64; k++) hits[idx].beta[k] = beta[k]; }
        }
    }
}
#if PX_PFX_MEM
__global__ void PX_FIN_ATTR finishx_kernel(const JobDev *job, ProjectivePoint *pts, const unsigned *n_ptr, Hit *hits, unsigned *nhits) {
    unsigned n = *n_ptr, total = gridDim.x * blockDim.x;
    for (unsigned base = blockIdx.x * blockDim.x + threadIdx.x; base < n; base += total * PX_SPLIT_FINISH) {
        fe prod; fe_1(prod);
        #pragma unroll 4
        for (int k = 0; k < PX_SPLIT_FINISH; k++) {
            unsigned i = base + (unsigned)k * total;
            if (i < n) { fe_copy(pts[i].X, prod); fe_mul(prod, prod, pts[i].Z); }
        }
        fe inv; fe_invert(inv, prod);
        #pragma unroll 1
        for (int k = PX_SPLIT_FINISH - 1; k >= 0; k--) {
            unsigned i = base + (unsigned)k * total;
            if (i < n) {
                fe zi, pre; fe_copy(pre, pts[i].X); fe_mul(zi, inv, pre); fe_mul(inv, inv, pts[i].Z);
                if (pts[i].valid) { fe y; fe_mul(y, pts[i].Y, zi); finishx_point(job, y, pts[i].nonce, hits, nhits); }
            }
        }
    }
}
#elif PX_FIN_SPLIT
// PX_FIN_SPLIT: the batch inversion writes affine y into the record (Y), and a separate low-register kernel hashes
// both encodings at full occupancy (the hashing is most of the finish)
__global__ void PX_FIN_ATTR finishx_kernel(const JobDev *job, ProjectivePoint *pts, const unsigned *n_ptr, Hit *hits, unsigned *nhits) {
    (void)job; (void)hits; (void)nhits;
    unsigned n = *n_ptr, total = gridDim.x * blockDim.x;
    for (unsigned base = blockIdx.x * blockDim.x + threadIdx.x; base < n; base += total * PX_SPLIT_FINISH) {
        fe prefix[PX_SPLIT_FINISH], prod; fe_1(prod);
        #pragma unroll
        for (int k = 0; k < PX_SPLIT_FINISH; k++) {
            unsigned i = base + (unsigned)k * total;
            fe_copy(prefix[k], prod);
            if (i < n) fe_mul(prod, prod, pts[i].Z);
        }
        fe inv; fe_invert(inv, prod);
        #pragma unroll
        for (int k = PX_SPLIT_FINISH - 1; k >= 0; k--) {
            unsigned i = base + (unsigned)k * total;
            if (i < n) {
                fe zi, y; fe_mul(zi, inv, prefix[k]); fe_mul(inv, inv, pts[i].Z);
                fe_mul(y, pts[i].Y, zi); fe_copy(pts[i].Y, y);
            }
        }
    }
}
#ifndef PX_LB_FSHA
#define PX_LB_FSHA 2
#endif
__global__ void __launch_bounds__(256, PX_LB_FSHA) fsha_kernel(const JobDev *job, const ProjectivePoint *pts, const unsigned *n_ptr, Hit *hits, unsigned *nhits) {
    unsigned n = *n_ptr;
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        if (pts[i].valid) { fe y; fe_copy(y, pts[i].Y); finishx_point(job, y, pts[i].nonce, hits, nhits); }
}
#else
__global__ void PX_FIN_ATTR finishx_kernel(const JobDev *job, const ProjectivePoint *pts, const unsigned *n_ptr, Hit *hits, unsigned *nhits, const unsigned *lo_ptr = nullptr) {
    unsigned n = *n_ptr, total = gridDim.x * blockDim.x, lo = lo_ptr ? *lo_ptr : 0u;
    for (unsigned base = lo + blockIdx.x * blockDim.x + threadIdx.x; base < n; base += total * PX_SPLIT_FINISH) {
        fe prefix[PX_SPLIT_FINISH], prod; fe_1(prod);
        #pragma unroll
        for (int k = 0; k < PX_SPLIT_FINISH; k++) {
            unsigned i = base + (unsigned)k * total;
            fe_copy(prefix[k], prod);
            if (i < n) fe_mul(prod, prod, pts[i].Z);
        }
        fe inv; fe_invert(inv, prod);
        #pragma unroll
        for (int k = PX_SPLIT_FINISH - 1; k >= 0; k--) {
            unsigned i = base + (unsigned)k * total;
            if (i < n) {
                fe zi; fe_mul(zi, inv, prefix[k]); fe_mul(inv, inv, pts[i].Z);
                if (pts[i].valid) { fe y; fe_mul(y, pts[i].Y, zi); finishx_point(job, y, pts[i].nonce, hits, nhits); }
            }
        }
    }
}
#endif
// keep only candidates whose beta is the exact output of their nonce; one block, any hit count up to MAX_HITS
__global__ void __launch_bounds__(128, 1) resolve_kernel(const JobDev *job, Hit *hits, unsigned *nhits) {
    __shared__ uint8_t keep[MAX_HITS];
    unsigned n = *nhits; if (n > MAX_HITS) n = MAX_HITS;
    JobDev j = *job;
    for (unsigned i = threadIdx.x; i < n; i += blockDim.x) {
        uint8_t beta[64];
        bool ok = vrf_beta(j, hits[i].nonce, beta, nullptr, nullptr);
        bool same = ok;
        for (int k = 0; k < 64 && same; k++) same = beta[k] == hits[i].beta[k];
        keep[i] = same;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned o = 0;
        for (unsigned i = 0; i < n; i++) if (keep[i]) { if (o != i) hits[o] = hits[i]; o++; }
        *nhits = o;
    }
}
#endif
#endif
// self-test: beta for a complete alpha (= prefix 37 + nonce 8) plus intermediates
__global__ void test_kernel(const JobDev *job, uint64_t nonce, uint8_t *out_beta, uint8_t *out_h, uint8_t *out_cg, int *ok) {
    *ok = vrf_beta(*job, nonce, out_beta, out_h, out_cg) ? 1 : 0;
}


// randomized field-arithmetic test: sq vs mul, (x+y)^2 = x^2 + 2xy + y^2, (x-y)(x+y) = x^2 - y^2
__global__ void fe_test_kernel(uint32_t seed, int n, int *fail, uint8_t *dump) {
    uint64_t st = seed * 0x9E3779B97F4A7C15ULL + 1;
    for (int it = 0; it < n; it++) {
        uint8_t xb[32], yb[32];
        for (int i = 0; i < 32; i++) { st = st * 6364136223846793005ULL + 1442695040888963407ULL; xb[i] = st >> 56; st = st * 6364136223846793005ULL + 1442695040888963407ULL; yb[i] = st >> 56; }
        fe x, y, s1, s2, t, u, v, w;
        fe_frombytes(x, xb); fe_frombytes(y, yb);
        // lazily reduced operands up to 2^256, as the chain produces them
        if (it & 1) { fe_add(x, x, x); fe_add(y, y, x); }
#ifdef PX_FE26
        fe_reduce(x); fe_reduce(y);
#endif
        fe_sq(s1, x); fe_mul(s2, x, x);
        int code = 0;
        if (!fe_eq(s1, s2)) code = 1;
        fe_add(t, x, y); fe_sq(t, t);                       // (x+y)^2
        fe_mul(u, x, y); fe_add(u, u, u); fe_sq(v, y); fe_add(u, u, v); fe_add(u, u, s2); // x^2+2xy+y^2
        if (!code && !fe_eq(t, u)) code = 2;
        fe_sub(w, x, y); fe_add(t, x, y); fe_mul(w, w, t);  // (x-y)(x+y)
        fe_sub(v, s2, v);                                   // x^2 - y^2
        if (!code && !fe_eq(w, v)) code = 3;
        fe_copy(t, x); fe_sq(t, t);                          // aliasing
        if (!code && !fe_eq(t, s2)) code = 4;
#if PX_LAD_KARA
        {   // Karatsuba product against the schoolbook one, on the value and on structured variants (equal halves,
            // a zero half, all-ones words, differences of either sign)
            fe xs[4], ys[4];
            fe_copy(xs[0], x); fe_copy(ys[0], y);
            fe_copy(xs[1], x); for (int i = 0; i < 4; i++) xs[1].v[4 + i] = x.v[i];
            fe_copy(ys[1], y); for (int i = 0; i < 4; i++) ys[1].v[i] = 0;
            for (int i = 0; i < 8; i++) { xs[2].v[i] = (it >> (i & 3)) & 1 ? 0xFFFFFFFFu : x.v[i]; ys[2].v[i] = 0xFFFFFFFFu; }
            ys[2].v[7] = (it & 4) ? 0xFFFFFFFFu : 0u;
            fe_copy(xs[3], y); fe_copy(ys[3], x); for (int i = 0; i < 4; i++) ys[3].v[4 + i] = ys[3].v[i] + (uint32_t)(it & 1);
            for (int c = 0; c < 4 && !code; c++) {
                fe k1, k2; fe_mul_kara<0>(k1, xs[c], ys[c]); fe_mul(k2, xs[c], ys[c]);
                if (!fe_eq(k1, k2)) code = 9;
            }
        }
#endif
        if (code) { *fail = code * 1000000 + it; for (int i = 0; i < 32; i++) { dump[i] = xb[i]; dump[32 + i] = yb[i]; } fe_tobytes(dump + 64, s1); fe_tobytes(dump + 96, s2); return; }
    }
    // the sqrt(-1) exponentiation step by step, squaring vs multiplying
    fe two, acc1, acc2; fe_0(two); two.v[0] = 2; fe_1(acc1); fe_1(acc2);
    for (int i = 253; i >= 0; i--) {
        fe before; fe_copy(before, acc1);
        fe_sq(acc1, acc1); fe_mul(acc2, acc2, acc2);
        if (!fe_eq(acc1, acc2)) { *fail = 5 * 1000000 + i; fe_tobytes(dump, before); fe_tobytes(dump + 64, acc1); fe_tobytes(dump + 96, acc2); for (int k = 0; k < 32; k++) dump[32 + k] = 0; return; }
        int bit = (i >= 3 && i <= 252) || i == 1 || i == 0;
        if (bit) { fe_mul(acc1, acc1, two); fe_mul(acc2, acc2, two); }
    }
    // inversion of a small constant, replayed step by step (pow22501 chain) with sq vs mul
    {
        fe x; fe_0(x); x.v[0] = 121666;
        fe a1, a2; fe_copy(a1, x); fe_copy(a2, x);
        // t0 = x^2, then 249 more squarings interleaved with multiplies by x, enough to hit the pattern
        for (int step = 0; step < 300; step++) {
            fe before; fe_copy(before, a1);
            if (step % 7 == 3) { fe_mul(a1, a1, x); fe_mul(a2, a2, x); }
            else { fe_sq(a1, a1); fe_mul(a2, a2, a2); }
            if (!fe_eq(a1, a2)) { *fail = 6 * 1000000 + step; fe_tobytes(dump, before); fe_tobytes(dump + 64, a1); fe_tobytes(dump + 96, a2); for (int k = 0; k < 32; k++) dump[32 + k] = 0; return; }
        }
        // pow22501 replayed: every intermediate with sq vs with mul
        #define SQM(h, f) fe_mul(h, f, f)
        fe u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13;
        fe m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13;
        int st = 0;
        #define CHK(A, B) if (!fe_eq(A, B)) { *fail = 8 * 1000000 + st; fe_tobytes(dump, A); fe_tobytes(dump + 64, B); return; } st++;
        fe_sq(u0, x); SQM(m0, x); CHK(u0, m0)
        fe_sq(u1, u0); fe_sq(u1, u1); SQM(m1, m0); SQM(m1, m1); CHK(u1, m1)
        fe_mul(u2, x, u1); fe_mul(m2, x, m1); CHK(u2, m2)
        fe_mul(u3, u0, u2); fe_mul(m3, m0, m2); CHK(u3, m3)
        fe_sq(u4, u3); SQM(m4, m3); CHK(u4, m4)
        fe_mul(u5, u2, u4); fe_mul(m5, m2, m4); CHK(u5, m5)
        fe_pow2k(u6, u5, 5); fe_copy(m6, m5); for (int i = 0; i < 5; i++) SQM(m6, m6); CHK(u6, m6)
        fe_mul(u7, u6, u5); fe_mul(m7, m6, m5); CHK(u7, m7)
        fe_pow2k(u8, u7, 10); fe_copy(m8, m7); for (int i = 0; i < 10; i++) SQM(m8, m8); CHK(u8, m8)
        fe_mul(u9, u8, u7); fe_mul(m9, m8, m7); CHK(u9, m9)
        fe_pow2k(u10, u9, 20); fe_copy(m10, m9); for (int i = 0; i < 20; i++) SQM(m10, m10); CHK(u10, m10)
        fe_mul(u11, u10, u9); fe_mul(m11, m10, m9); CHK(u11, m11)
        fe_pow2k(u12, u11, 10); fe_copy(m12, m11); for (int i = 0; i < 10; i++) SQM(m12, m12); CHK(u12, m12)
        fe_mul(u13, u12, u7); fe_mul(m13, m12, m7); CHK(u13, m13)
        fe inv, one; fe_invert(inv, x); fe_mul(one, inv, x);
        fe o; fe_1(o);
        if (!fe_eq(one, o)) { *fail = 7 * 1000000; fe_tobytes(dump, inv); fe_tobytes(dump + 64, one); return; }
    }
    *fail = 0;
}

// constants: d = -121665/121666, sqrt(-1) = 2^((p-1)/4), base point B (y = 4/5, x even); pk = x*B
__global__ void init_kernel(const uint8_t *scalar, uint8_t *pk_out, int *ok) {
    fe a, b, t;
    // d
    fe_0(a); a.v[0] = 121665; fe_neg(a, a);
    fe_0(b); b.v[0] = 121666; fe_invert(t, b);
    fe_mul(d_EDWARDS_D, a, t);
    fe_add(d_EDWARDS_D2, d_EDWARDS_D, d_EDWARDS_D); fe_reduce(d_EDWARDS_D2);
    // arithmetic self-check: squaring must agree with multiplication
    { fe s1, s2; fe_sq(s1, d_EDWARDS_D); fe_mul(s2, d_EDWARDS_D, d_EDWARDS_D); if (!fe_eq(s1, s2)) { *ok = -2; return; } }
    // sqrt(-1) = 2^((p-1)/4), exponent 2^253 - 5 : bits 252..3 set, bit 2 clear, bits 1,0 set
    fe r; fe_0(r); r.v[0] = 2;
    fe acc; fe_1(acc);
    for (int i = 253; i >= 0; i--) {
        fe_sq(acc, acc);
        int bit = (i >= 3 && i <= 252) || i == 1 || i == 0;
        if (bit) fe_mul(acc, acc, r);
    }
    fe_copy(d_SQRT_M1, acc);
    fe m1; fe_1(m1); fe_neg(m1, m1);
    fe chk; fe_sq(chk, d_SQRT_M1);
    *ok = fe_eq(chk, m1) ? 1 : 0;
#if PX_LADDER
    { fe a2, one, c, c2; fe_0(a2); a2.v[0] = PX_MONT_A + 2u; fe_neg(a2, a2); fe_1(one);
      if (!fe_sqrt_ratio_i(c, a2, one)) { *ok = -3; return; }
      fe_sq(c2, c); if (!fe_eq(c2, a2)) { *ok = -3; return; }       // c^2 = -486664 exactly
      fe_copy(d_MONT_C, c); }
#endif
    // base point: y = 4/5
    fe four, five, y; fe_0(four); four.v[0] = 4; fe_0(five); five.v[0] = 5;
    fe_invert(t, five); fe_mul(y, four, t);
    uint8_t yb[32]; fe_tobytes(yb, y);
    ge B;
    if (!ge_frombytes(B, yb)) { *ok = 0; return; }
    ge P; ge_scalarmult(P, scalar, B);
    ge_tobytes(pk_out, P);
}


// ---- host-side adapter for this variant --------------------------------------
// width-5 wNAF of the little-endian 256-bit scalar: digits are odd and at most one
// in any 5 consecutive positions, so ~1/6 of the positions carry an addition
#if PX_MERGE8
// (64 * scalar) mod 8L, L = 2^252 + 27742317777372353535851937790883648493 (the group order), as 32 little-endian
// bytes. Reducing modulo 8L (not L) keeps the result divisible by 8, so the torsion part of H is still killed.
static void v_merged_scalar(const uint8_t *scalar, uint8_t out[32]) {
    static const uint8_t Lb[32] = { 0xed, 0xd3, 0xf5, 0x5c, 0x1a, 0x63, 0x12, 0x58, 0xd6, 0x9c, 0xf7, 0xa2, 0xde, 0xf9, 0xde, 0x14,
                                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10 };
    // t = scalar << 6 in 40 bytes
    uint8_t t[40]; memset(t, 0, sizeof t);
    for (int i = 0; i < 32; i++) { unsigned v = (unsigned)scalar[i] << 6; t[i] |= v & 0xff; t[i + 1] |= (v >> 8) & 0xff; }
    // subtract 8L << sh while t >= 8L << sh, for sh = 16..0 bits (64x < 2^262 = 2^7 * 8L, so 16 shifts are plenty)
    for (int sh = 16; sh >= 0; sh--) {
        uint8_t m[40]; memset(m, 0, sizeof m);
        for (int i = 0; i < 32; i++) { unsigned v = (unsigned)Lb[i] << ((sh + 3) & 7); int o = i + ((sh + 3) >> 3); m[o] |= v & 0xff; m[o + 1] |= (v >> 8) & 0xff; }
        for (;;) {
            int ge = 0; for (int i = 39; i >= 0; i--) { if (t[i] != m[i]) { ge = t[i] > m[i]; break; } if (i == 0) ge = 1; }
            if (!ge) break;
            int b = 0; for (int i = 0; i < 40; i++) { int d = (int)t[i] - (int)m[i] - b; b = d < 0; t[i] = (uint8_t)(d + (b ? 256 : 0)); }
        }
    }
    memcpy(out, t, 32);
}
#endif
static void v_set_scalar(const uint8_t *scalar_in) {
#if PX_MERGE8
    uint8_t merged[32]; v_merged_scalar(scalar_in, merged);
    const uint8_t *scalar = merged;
#else
    const uint8_t *scalar = scalar_in;
#endif
    CUDA_CHECK(cudaMemcpyToSymbol(c_scalar, scalar, 32));
#if PX_LADDER
    { int top = 0; for (int b = 255; b >= 0; b--) if ((scalar[b >> 3] >> (b & 7)) & 1) { top = b; break; }
      CUDA_CHECK(cudaMemcpyToSymbol(c_ladder_top, &top, sizeof top)); }
#endif
#if PX_XONLY
    {   // t = 8x mod L, then min(t, L - t), as little-endian 32-bit words
        static const uint32_t Lw[8] = { 0x5cf5d3edu, 0x5812631au, 0xa2f79cd6u, 0x14def9deu, 0, 0, 0, 0x10000000u };
        uint32_t t[9] = {0};
        for (int i = 0; i < 8; i++) t[i] = (uint32_t)scalar_in[4 * i] | ((uint32_t)scalar_in[4 * i + 1] << 8) | ((uint32_t)scalar_in[4 * i + 2] << 16) | ((uint32_t)scalar_in[4 * i + 3] << 24);
        // t <<= 3 (x < 2^256: 8x < 2^259 fits nine words)
        for (int i = 8; i > 0; i--) t[i] = (t[i] << 3) | (t[i - 1] >> 29);
        t[0] <<= 3;
        auto geq = [](const uint32_t *a, const uint32_t *b) { for (int i = 8; i >= 0; i--) { if (a[i] != b[i]) return a[i] > b[i]; } return true; };
        auto sub = [](uint32_t *a, const uint32_t *b) { uint64_t br = 0; for (int i = 0; i < 9; i++) { uint64_t d = (uint64_t)a[i] - b[i] - br; a[i] = (uint32_t)d; br = (d >> 63) & 1; } };
        for (int sh = 8; sh >= 0; sh--) {        // subtract L << sh while t >= L << sh (t < 2^259 = 2^7 * 4L)
            uint32_t m[9] = {0};
            for (int i = 0; i < 8; i++) { uint64_t v = (uint64_t)Lw[i] << sh; m[i] |= (uint32_t)v; m[i + 1] |= (uint32_t)(v >> 32); }
            while (geq(t, m)) sub(t, m);
        }
        uint32_t nt[9] = {0}; uint32_t L9[9] = {0}; for (int i = 0; i < 8; i++) L9[i] = Lw[i];
        bool zero = true; for (int i = 0; i < 9; i++) if (t[i]) zero = false;
        if (!zero) { for (int i = 0; i < 9; i++) nt[i] = L9[i]; sub(nt, t); if (!geq(nt, t)) for (int i = 0; i < 9; i++) t[i] = nt[i]; }
        uint8_t tb[32]; for (int i = 0; i < 8; i++) { tb[4 * i] = t[i]; tb[4 * i + 1] = t[i] >> 8; tb[4 * i + 2] = t[i] >> 16; tb[4 * i + 3] = t[i] >> 24; }
        int top = 0; for (int b = 255; b >= 0; b--) if ((tb[b >> 3] >> (b & 7)) & 1) { top = b; break; }
        CUDA_CHECK(cudaMemcpyToSymbol(c_xscalar, tb, 32));
        CUDA_CHECK(cudaMemcpyToSymbol(c_xladder_top, &top, sizeof top));
    }
#endif
    uint8_t k[40]; memset(k, 0, sizeof k); memcpy(k, scalar, 32);
    int8_t digits[260]; memset(digits, 0, sizeof digits);
    int len = 0;
    auto isZero = [&]() { for (int i = 0; i < 40; i++) if (k[i]) return false; return true; };
    auto shr1 = [&]() { for (int i = 0; i < 39; i++) k[i] = (k[i] >> 1) | (k[i + 1] << 7); k[39] >>= 1; };
    auto addSmall = [&](int v) { // k += v, v in [-16, 16]
        if (v >= 0) { unsigned c = v; for (int i = 0; i < 40 && c; i++) { unsigned t = k[i] + c; k[i] = t & 0xff; c = t >> 8; } }
        else { unsigned b = -v; for (int i = 0; i < 40 && b; i++) { int t = (int)k[i] - (int)b; if (t < 0) { k[i] = t + 256; b = 1; } else { k[i] = t; b = 0; } } }
    };
    for (int i = 0; i < 258 && !isZero(); i++) {
        int d = 0;
        if (k[0] & 1) { d = k[0] & 31; if (d >= 16) d -= 32; addSmall(-d); }
        digits[i] = (int8_t)d;
        if (d) len = i + 1;
        shr1();
    }
    CUDA_CHECK(cudaMemcpyToSymbol(c_wnaf, digits, sizeof digits));
    CUDA_CHECK(cudaMemcpyToSymbol(c_wnaf_len, &len, sizeof len));
    int8_t next[260]; int8_t last = -1;
    for (int i = 0; i < 260; i++) { next[i] = last; if (digits[i]) last = (int8_t)((digits[i] > 0 ? digits[i] : -digits[i]) >> 1); }
    CUDA_CHECK(cudaMemcpyToSymbol(c_wnaf_next, next, sizeof next));
}
static int v_init(const uint8_t *scalar, uint8_t pk_out[32]) {
    uint8_t *d_sc, *d_pk; int *d_ok;
    CUDA_CHECK(cudaMalloc(&d_sc, 32)); CUDA_CHECK(cudaMalloc(&d_pk, 32)); CUDA_CHECK(cudaMalloc(&d_ok, 4));
    CUDA_CHECK(cudaMemcpy(d_sc, scalar, 32, cudaMemcpyHostToDevice));
    v_set_scalar(scalar);
    init_kernel<<<1, 1>>>(d_sc, d_pk, d_ok);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    int ok; CUDA_CHECK(cudaMemcpy(&ok, d_ok, 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pk_out, d_pk, 32, cudaMemcpyDeviceToHost));
    cudaFree(d_sc); cudaFree(d_pk); cudaFree(d_ok);
    return ok;
}
// scratch for the compacted pipeline: points found, two alternating retry queues, counters
// (cnt[0] = points, cnt[1..PX_MAX_PASSES] = retries per pass, cnt[PX_MAX_PASSES+1] = N)
static void *v_alloc_one(size_t n) {
#if PX_COMPACT
    Scratch *sc = new Scratch(); sc->n = n;
    CUDA_CHECK(cudaMalloc(&sc->pts, n * sizeof(HPoint)));
#ifdef PX_SPLIT_FINISH
    CUDA_CHECK(cudaMalloc(&sc->projective,n*sizeof(ProjectivePoint)));
#endif
    for (int b = 0; b < 2; b++) { CUDA_CHECK(cudaMalloc(&sc->nonce[b], n * sizeof(uint64_t))); CUDA_CHECK(cudaMalloc(&sc->ctr[b], n)); }
#if PX_H2C_MID
    CUDA_CHECK(cudaMalloc(&sc->mid, 17 * sizeof(uint64_t)));
#endif
    CUDA_CHECK(cudaMalloc(&sc->cnt, (PX_MAX_PASSES + 3) * sizeof(unsigned)));
#if PX_TAIL_SPLIT
    {   int lo = 0, hi = 0, dev = 0; CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
        CUDA_CHECK(cudaStreamCreateWithPriority(&sc->sa, cudaStreamNonBlocking, hi));
        CUDA_CHECK(cudaStreamCreateWithPriority(&sc->sb, cudaStreamNonBlocking, lo));
        CUDA_CHECK(cudaEventCreateWithFlags(&sc->ev0, cudaEventDisableTiming)); CUDA_CHECK(cudaEventCreateWithFlags(&sc->evk, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&sc->evda, cudaEventDisableTiming)); CUDA_CHECK(cudaEventCreateWithFlags(&sc->evdb, cudaEventDisableTiming));
        CUDA_CHECK(cudaGetDevice(&dev)); CUDA_CHECK(cudaDeviceGetAttribute(&sc->sms, cudaDevAttrMultiProcessorCount, dev)); }
#endif
    unsigned N = (unsigned)n; CUDA_CHECK(cudaMemcpy(sc->cnt + PX_MAX_PASSES + 1, &N, sizeof N, cudaMemcpyHostToDevice));
    return sc;
#else
    (void)n; return nullptr;
#endif
}
static void v_free_one(void *scratch) {
#if PX_COMPACT
    Scratch *sc = (Scratch *)scratch; if (!sc) return;
#ifdef PX_SPLIT_FINISH
    cudaFree(sc->projective);
#endif
#if PX_H2C_MID
    cudaFree(sc->mid);
#endif
#if PX_TAIL_SPLIT
    cudaStreamDestroy(sc->sa); cudaStreamDestroy(sc->sb); cudaEventDestroy(sc->ev0); cudaEventDestroy(sc->evk); cudaEventDestroy(sc->evda); cudaEventDestroy(sc->evdb);
#endif
    cudaFree(sc->pts); for (int b = 0; b < 2; b++) { cudaFree(sc->nonce[b]); cudaFree(sc->ctr[b]); } cudaFree(sc->cnt); delete sc;
#else
    (void)scratch;
#endif
}
#if !(PX_XONLY && PX_SLICES > 1)
static void *v_alloc(size_t n) { return v_alloc_one(n); }
static void v_free(void *scratch) { v_free_one(scratch); }
#endif
#if PX_XONLY
#define PX_H2C_K h2cx_kernel
#else
#define PX_H2C_K h2c_kernel
#endif
#if PX_XONLY && PX_SLICES > 1
// PX_SLICES: the launch is split into slices on their own streams. Slice i hashes to the curve only after slice
// i-1 has (event chain), so its short, poorly occupied kernels (hash-to-curve passes, prep) run underneath the
// other slices' ladders instead of on an otherwise idle GPU; the finish stages overlap the same way.
struct SliceSet { Scratch *sc[PX_SLICES]; size_t n[PX_SLICES]; cudaStream_t st[PX_SLICES]; cudaEvent_t h2c_done[PX_SLICES], prep_done[PX_SLICES], lad_done[PX_SLICES]; int sms; cudaStream_t hs, ls; };
static void *v_alloc_one(size_t n);
static void v_free_one(void *scratch);
static void *v_alloc_slices(size_t n) {
    SliceSet *ss = new SliceSet();
    { int dev = 0; CUDA_CHECK(cudaGetDevice(&dev)); CUDA_CHECK(cudaDeviceGetAttribute(&ss->sms, cudaDevAttrMultiProcessorCount, dev)); }
    size_t base = n / PX_SLICES, rem = n % PX_SLICES;
    for (int i = 0; i < PX_SLICES; i++) {
        ss->n[i] = base + ((size_t)i < rem ? 1 : 0);
        ss->sc[i] = (Scratch *)v_alloc_one(ss->n[i] ? ss->n[i] : 1);
        if (!ss->n[i]) { unsigned z = 0; CUDA_CHECK(cudaMemcpy(ss->sc[i]->cnt + PX_MAX_PASSES + 1, &z, sizeof z, cudaMemcpyHostToDevice)); }
        CUDA_CHECK(cudaStreamCreate(&ss->st[i]));
        CUDA_CHECK(cudaEventCreateWithFlags(&ss->h2c_done[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ss->prep_done[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ss->lad_done[i], cudaEventDisableTiming));
    }
    {   int lo = 0, hi = 0; CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
        CUDA_CHECK(cudaStreamCreateWithPriority(&ss->ls, cudaStreamDefault, hi));     // ladder: placed first when an SM frees up
        CUDA_CHECK(cudaStreamCreateWithPriority(&ss->hs, cudaStreamDefault, lo)); }
    return ss;
}
static void v_free_slices(void *scratch) {
    SliceSet *ss = (SliceSet *)scratch; if (!ss) return;
    for (int i = 0; i < PX_SLICES; i++) { v_free_one(ss->sc[i]); cudaStreamDestroy(ss->st[i]); cudaEventDestroy(ss->h2c_done[i]); cudaEventDestroy(ss->prep_done[i]); cudaEventDestroy(ss->lad_done[i]); }
    cudaStreamDestroy(ss->hs); cudaStreamDestroy(ss->ls);
    delete ss;
}
static void mine_x_slice(Scratch *sc, size_t N, cudaStream_t st, cudaEvent_t wait_ev, cudaEvent_t done_ev, int blocks, int threads,
                         const JobDev *job, uint64_t nonce_base, Hit *hits, unsigned *nhits, int sms) {
    (void)sms;
    if (wait_ev) CUDA_CHECK(cudaStreamWaitEvent(st, wait_ev, 0));
    CUDA_CHECK(cudaMemsetAsync(sc->cnt, 0, (PX_MAX_PASSES + 1) * sizeof(unsigned), st));
#if PX_CORUN
    const int HT = 128;
#else
    const int HT = 256;
#endif
    int g1 = (int)((N + HT - 1) / HT); if (g1 < 1) g1 = 1;
#if PX_H2C_MID
    const uint64_t *mid = sc->mid;
    sha_prep_kernel<<<1, 32, 0, st>>>(job, sc->mid);
#else
    const uint64_t *mid = nullptr;
#endif
    h2cx_kernel<<<g1, HT, 0, st>>>(job, mid, nonce_base, nullptr, nullptr, sc->cnt + PX_MAX_PASSES + 1, sc->pts, sc->cnt, sc->nonce[0], sc->ctr[0], sc->cnt + 1);
    for (int p = 1; p < PX_MAX_PASSES; p++) {
        int in = (p - 1) & 1, out = p & 1;
        int gp = max(1, g1 >> p);
        h2cx_kernel<<<gp, HT, 0, st>>>(job, mid, nonce_base, sc->nonce[in], sc->ctr[in], sc->cnt + p, sc->pts, sc->cnt, sc->nonce[out], sc->ctr[out], sc->cnt + p + 1);
    }
    CUDA_CHECK(cudaEventRecord(done_ev, st));
#if PX_CORUN
    int gp_ = (int)((N + (size_t)HT * PX_PREP_BATCH - 1) / ((size_t)HT * PX_PREP_BATCH)); if (gp_ < 1) gp_ = 1;
    prepx_kernel<<<gp_, HT, 0, st>>>(sc->cnt, sc->pts);
    ladderx_kernel<<<sms, 384, 0, st>>>(sc->pts, sc->cnt, sc->projective);
    int gf_ = (int)((N + (size_t)HT * PX_SPLIT_FINISH - 1) / ((size_t)HT * PX_SPLIT_FINISH)); if (gf_ < 1) gf_ = 1;
    finishx_kernel<<<gf_, HT, 0, st>>>(job, sc->projective, sc->cnt, hits, nhits);
#else
    prepx_kernel<<<max(1, (blocks + PX_PREP_BATCH - 1) / PX_PREP_BATCH), threads, 0, st>>>(sc->cnt, sc->pts);
    ladderx_kernel<<<max(1, blocks), threads, 0, st>>>(sc->pts, sc->cnt, sc->projective);
    finishx_kernel<<<max(1, (blocks + PX_SPLIT_FINISH - 1) / PX_SPLIT_FINISH), threads, 0, st>>>(job, sc->projective, sc->cnt, hits, nhits);
#endif
}
#endif
#if PX_CORUN == 2
// PX_CORUN 2: every slice hashes to the curve first, back to back at full width (those kernels do not fit beside the
// ladder); then the ladders run one after another, one persistent 384-thread block per multiprocessor, and the prep of
// slice i+1 and the finish of slice i-1 run in the register space the ladder leaves free (128-thread blocks).
#ifndef PX_SMALL_TPB
#define PX_SMALL_TPB 128
#endif
static void mine_x_pipe(SliceSet *ss, int threads, const JobDev *job, uint64_t nonce_base, Hit *hits, unsigned *nhits) {
    (void)threads;
    const int HT = PX_SMALL_TPB;
    uint64_t off = 0;
    for (int i = 0; i < PX_SLICES; i++) {
        Scratch *sc = ss->sc[i]; size_t N = ss->n[i]; cudaStream_t st = ss->st[i];
        if (i) CUDA_CHECK(cudaStreamWaitEvent(st, ss->h2c_done[i - 1], 0));
        CUDA_CHECK(cudaMemsetAsync(sc->cnt, 0, (PX_MAX_PASSES + 1) * sizeof(unsigned), st));
        int g1 = (int)((N + HT - 1) / HT); if (g1 < 1) g1 = 1;
#if PX_H2C_MID
        const uint64_t *mid = sc->mid;
        sha_prep_kernel<<<1, 32, 0, st>>>(job, sc->mid);
#else
        const uint64_t *mid = nullptr;
#endif
        h2cx_kernel<<<g1, HT, 0, st>>>(job, mid, nonce_base + off, nullptr, nullptr, sc->cnt + PX_MAX_PASSES + 1, sc->pts, sc->cnt, sc->nonce[0], sc->ctr[0], sc->cnt + 1);
        for (int p = 1; p < PX_MAX_PASSES; p++) {
            int in = (p - 1) & 1, out = p & 1;
            h2cx_kernel<<<max(1, g1 >> p), HT, 0, st>>>(job, mid, nonce_base + off, sc->nonce[in], sc->ctr[in], sc->cnt + p, sc->pts, sc->cnt, sc->nonce[out], sc->ctr[out], sc->cnt + p + 1);
        }
        CUDA_CHECK(cudaEventRecord(ss->h2c_done[i], st));
        off += N;
    }
    for (int i = 0; i < PX_SLICES; i++) {
        Scratch *sc = ss->sc[i]; size_t N = ss->n[i]; cudaStream_t st = ss->st[i];
        CUDA_CHECK(cudaStreamWaitEvent(st, ss->h2c_done[PX_SLICES - 1], 0));
        if (i) CUDA_CHECK(cudaStreamWaitEvent(st, ss->prep_done[i - 1], 0));     // prep i beside ladder i-1
        int gp = (int)((N + (size_t)HT * PX_PREP_BATCH - 1) / ((size_t)HT * PX_PREP_BATCH)); if (gp < 1) gp = 1;
        prepx_kernel<<<gp, HT, 0, st>>>(sc->cnt, sc->pts);
        CUDA_CHECK(cudaEventRecord(ss->prep_done[i], st));
        if (i) CUDA_CHECK(cudaStreamWaitEvent(st, ss->lad_done[i - 1], 0));
        ladderx_kernel<<<ss->sms * PX_LADDER_BPS, PX_LADDER_TPB, 0, st>>>(sc->pts, sc->cnt, sc->projective);
        CUDA_CHECK(cudaEventRecord(ss->lad_done[i], st));
        int gf = (int)((N + (size_t)HT * PX_SPLIT_FINISH - 1) / ((size_t)HT * PX_SPLIT_FINISH)); if (gf < 1) gf = 1;
        finishx_kernel<<<gf, HT, 0, st>>>(job, sc->projective, sc->cnt, hits, nhits);   // beside ladder i+1
    }
}
#endif
#if PX_CORUN == 3
// PX_CORUN 3: one persistent ladder block per multiprocessor (<= 128 registers, 384 threads) on a high-priority stream;
// everything else (hash to curve, prep, finish) runs as 128-thread blocks on a low-priority helper stream in the
// register space the ladder leaves, one slice ahead (hash, prep) and one slice behind (finish) of the ladder.
#ifndef PX_SMALL_TPB
#define PX_SMALL_TPB 128
#endif
static void pipe3_front(SliceSet *ss, int i, const JobDev *job, uint64_t nb) {
    Scratch *sc = ss->sc[i]; size_t N = ss->n[i]; cudaStream_t st = ss->hs; const int HT = PX_SMALL_TPB;
    CUDA_CHECK(cudaMemsetAsync(sc->cnt, 0, (PX_MAX_PASSES + 1) * sizeof(unsigned), st));
    int g1 = (int)((N + HT - 1) / HT); if (g1 < 1) g1 = 1;
    sha_prep_kernel<<<1, 32, 0, st>>>(job, sc->mid);
    h2cx_kernel<<<g1, HT, 0, st>>>(job, sc->mid, nb, nullptr, nullptr, sc->cnt + PX_MAX_PASSES + 1, sc->pts, sc->cnt, sc->nonce[0], sc->ctr[0], sc->cnt + 1);
    for (int p = 1; p < PX_MAX_PASSES; p++) {
        int in = (p - 1) & 1, out = p & 1;
        h2cx_kernel<<<max(1, g1 >> p), HT, 0, st>>>(job, sc->mid, nb, sc->nonce[in], sc->ctr[in], sc->cnt + p, sc->pts, sc->cnt, sc->nonce[out], sc->ctr[out], sc->cnt + p + 1);
    }
    int gp = (int)((N + (size_t)HT * PX_PREP_BATCH - 1) / ((size_t)HT * PX_PREP_BATCH)); if (gp < 1) gp = 1;
    prepx_kernel<<<gp, HT, 0, st>>>(sc->cnt, sc->pts);
    CUDA_CHECK(cudaEventRecord(ss->prep_done[i], st));
}
static void mine_x_pipe3(SliceSet *ss, const JobDev *job, uint64_t nonce_base, Hit *hits, unsigned *nhits) {
    const int HT = PX_SMALL_TPB;
    uint64_t off[PX_SLICES]; uint64_t o = 0;
    for (int i = 0; i < PX_SLICES; i++) { off[i] = o; o += ss->n[i]; }
    pipe3_front(ss, 0, job, nonce_base + off[0]);
    for (int i = 0; i < PX_SLICES; i++) {
        Scratch *sc = ss->sc[i];
        CUDA_CHECK(cudaStreamWaitEvent(ss->ls, ss->prep_done[i], 0));
        ladderx_kernel<<<ss->sms * PX_LADDER_BPS, PX_LADDER_TPB, 0, ss->ls>>>(sc->pts, sc->cnt, sc->projective);
        CUDA_CHECK(cudaEventRecord(ss->lad_done[i], ss->ls));
        if (i + 1 < PX_SLICES) pipe3_front(ss, i + 1, job, nonce_base + off[i + 1]);
        CUDA_CHECK(cudaStreamWaitEvent(ss->hs, ss->lad_done[i], 0));
        size_t N = ss->n[i];
        int gf = (int)((N + (size_t)HT * PX_SPLIT_FINISH - 1) / ((size_t)HT * PX_SPLIT_FINISH)); if (gf < 1) gf = 1;
        finishx_kernel<<<gf, HT, 0, ss->hs>>>(job, sc->projective, sc->cnt, hits, nhits);
    }
}
#endif
static void v_mine(int blocks, int threads, const JobDev *job, uint64_t nonce_base, uint64_t stride, int iters, Hit *hits, unsigned *nhits, void *scratch) {
#if PX_XONLY && PX_SLICES > 1
    {   SliceSet *ss = (SliceSet *)scratch;
#if PX_CORUN == 3
        mine_x_pipe3(ss, job, nonce_base, hits, nhits);
        (void)blocks; (void)threads; (void)stride; (void)iters;
        resolve_kernel<<<1, 128>>>(job, hits, nhits);
        return;
#endif
#if PX_CORUN == 2
        mine_x_pipe(ss, threads, job, nonce_base, hits, nhits);
        (void)blocks; (void)stride; (void)iters;
        resolve_kernel<<<1, 128>>>(job, hits, nhits);
        return;
#endif
        int sb = (blocks + PX_SLICES - 1) / PX_SLICES;
        uint64_t off = 0;
        for (int i = 0; i < PX_SLICES; i++) {
            mine_x_slice(ss->sc[i], ss->n[i], ss->st[i], i ? ss->h2c_done[i - 1] : nullptr, ss->h2c_done[i], sb, threads, job, nonce_base + off, hits, nhits, ss->sms);
            off += ss->n[i];
        }
        (void)stride; (void)iters;
        resolve_kernel<<<1, 128>>>(job, hits, nhits);   // legacy default stream: after every slice
        return; }
#endif
#if PX_TAIL_SPLIT
    {   // PX_TAIL_SPLIT = K: hash-to-curve passes 0..K-1, then prep / ladder / finish of those points on the main stream,
        // while passes K.. (a few, latency-bound retries) and the prep / ladder / finish of their points run on a
        // low-priority stream beside them instead of on an otherwise idle GPU. Same points and outputs as the passes.
        Scratch *sc = (Scratch *)scratch;
        size_t N = stride * (size_t)iters;
        cudaStream_t A = sc->sa, B = sc->sb;
        unsigned *nk = sc->cnt + PX_MAX_PASSES + 2;
        CUDA_CHECK(cudaEventRecord(sc->ev0, 0));
        CUDA_CHECK(cudaStreamWaitEvent(A, sc->ev0, 0));
        CUDA_CHECK(cudaMemsetAsync(sc->cnt, 0, (PX_MAX_PASSES + 1) * sizeof(unsigned), A));
        int g1 = (int)((N + 255) / 256);
        sha_prep_kernel<<<1, 32, 0, A>>>(job, sc->mid);
        h2cx_kernel<<<g1, 256, 0, A>>>(job, sc->mid, nonce_base, nullptr, nullptr, sc->cnt + PX_MAX_PASSES + 1, sc->pts, sc->cnt, sc->nonce[0], sc->ctr[0], sc->cnt + 1);
        for (int p = 1; p < PX_TAIL_SPLIT; p++) {
            int in = (p - 1) & 1, out = p & 1;
            h2cx_kernel<<<max(1, g1 >> p), 256, 0, A>>>(job, sc->mid, nonce_base, sc->nonce[in], sc->ctr[in], sc->cnt + p, sc->pts, sc->cnt, sc->nonce[out], sc->ctr[out], sc->cnt + p + 1);
        }
        CUDA_CHECK(cudaMemcpyAsync(nk, sc->cnt, sizeof(unsigned), cudaMemcpyDeviceToDevice, A));
        CUDA_CHECK(cudaEventRecord(sc->evk, A));
        prepx_kernel<<<(blocks + PX_PREP_BATCH - 1) / PX_PREP_BATCH, threads, 0, A>>>(nk, sc->pts);
        ladderx_kernel<<<blocks, threads, 0, A>>>(sc->pts, nk, sc->projective);
        finishx_kernel<<<(blocks + PX_SPLIT_FINISH - 1) / PX_SPLIT_FINISH, threads, 0, A>>>(job, sc->projective, nk, hits, nhits);
        CUDA_CHECK(cudaEventRecord(sc->evda, A));
        CUDA_CHECK(cudaStreamWaitEvent(B, sc->evk, 0));
        for (int p = PX_TAIL_SPLIT; p < PX_MAX_PASSES; p++) {
            int in = (p - 1) & 1, out = p & 1;
            h2cx_kernel<<<max(1, g1 >> p), 256, 0, B>>>(job, sc->mid, nonce_base, sc->nonce[in], sc->ctr[in], sc->cnt + p, sc->pts, sc->cnt, sc->nonce[out], sc->ctr[out], sc->cnt + p + 1);
        }
        int gt = max(1, (blocks >> PX_TAIL_SPLIT));
        prepx_kernel<<<max(1, (gt + PX_PREP_BATCH - 1) / PX_PREP_BATCH), threads, 0, B>>>(sc->cnt, sc->pts, nk);
        ladderx_kernel<<<gt, threads, 0, B>>>(sc->pts, sc->cnt, sc->projective, nk);
        finishx_kernel<<<max(1, (gt + PX_SPLIT_FINISH - 1) / PX_SPLIT_FINISH), threads, 0, B>>>(job, sc->projective, sc->cnt, hits, nhits, nk);
        CUDA_CHECK(cudaEventRecord(sc->evdb, B));
        CUDA_CHECK(cudaStreamWaitEvent(0, sc->evda, 0)); CUDA_CHECK(cudaStreamWaitEvent(0, sc->evdb, 0));
        resolve_kernel<<<1, 128>>>(job, hits, nhits);
        return;
    }
#endif
#if PX_COMPACT
    Scratch *sc = (Scratch *)scratch;
    size_t N = stride * (size_t)iters;
    CUDA_CHECK(cudaMemsetAsync(sc->cnt, 0, (PX_MAX_PASSES + 1) * sizeof(unsigned)));
    int g1 = (int)((N + 255) / 256);
    // pass 0 over all nonces, then the retry queues, alternating buffers
#if PX_H2C_MID
    const uint64_t *mid = sc->mid;
    sha_prep_kernel<<<1, 32>>>(job, sc->mid);
#else
    const uint64_t *mid = nullptr;
#endif
#if PX_H2C_PERSIST
    {   int sms = 0, dev = 0; CUDA_CHECK(cudaGetDevice(&dev)); CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
        int gr = sms * PX_LB_H2C; if (gr > g1) gr = g1;
        h2cr_kernel<<<gr, 256>>>(job, mid, nonce_base, sc->cnt + PX_MAX_PASSES + 1, sc->cnt + 1, sc->pts, sc->cnt); }
    if (0)
#endif
    PX_H2C_K<<<g1, 256>>>(job, mid, nonce_base, nullptr, nullptr, sc->cnt + PX_MAX_PASSES + 1, sc->pts, sc->cnt, sc->nonce[0], sc->ctr[0], sc->cnt + 1);
#if PX_H2C_PERSIST
    const int last = 1;
#elif PX_H2C_TAIL
    const int last = PX_H2C_TAIL;
#else
    const int last = PX_MAX_PASSES;
#endif
    for (int p = 1; p < last; p++) {
        int in = (p - 1) & 1, out = p & 1;
        int gp = max(1, g1 >> p); // the queue halves every pass
        PX_H2C_K<<<gp, 256>>>(job, mid, nonce_base, sc->nonce[in], sc->ctr[in], sc->cnt + p, sc->pts, sc->cnt, sc->nonce[out], sc->ctr[out], sc->cnt + p + 1);
    }
#if PX_H2C_TAIL
    { int in = (last - 1) & 1;
      h2c_tail_kernel<<<max(1, g1 >> last), 256>>>(job, mid, sc->nonce[in], sc->ctr[in], sc->cnt + last, sc->pts, sc->cnt); }
#endif
#if PX_XONLY
    prepx_kernel<<<(blocks + PX_PREP_BATCH - 1) / PX_PREP_BATCH, threads>>>(sc->cnt, sc->pts);
    ladderx_kernel<<<blocks, threads>>>(sc->pts, sc->cnt, sc->projective);
    finishx_kernel<<<(blocks+PX_SPLIT_FINISH-1)/PX_SPLIT_FINISH,threads>>>(job,sc->projective,sc->cnt,hits,nhits);
#if PX_FIN_SPLIT
    fsha_kernel<<<blocks, threads>>>(job, sc->projective, sc->cnt, hits, nhits);
#endif
    resolve_kernel<<<1, 128>>>(job, hits, nhits);
#elif PX_LADDER
    prep_kernel<<<(blocks + PX_PREP_BATCH - 1) / PX_PREP_BATCH, threads>>>(sc->cnt, sc->pts);
    ladder_kernel<<<blocks, threads>>>(sc->pts, sc->cnt, sc->projective);
    finish_batch_kernel<<<(blocks+PX_SPLIT_FINISH-1)/PX_SPLIT_FINISH,threads>>>(job,sc->projective,sc->cnt,hits,nhits);
#elif defined(PX_SPLIT_FINISH)
#if PX_SCALAR_THREADS != 256
    // the scalar stage runs its own block size (launch bounds); same total thread count
    scalar_kernel<<<(int)(((size_t)blocks*threads+PX_SCALAR_THREADS-1)/PX_SCALAR_THREADS),PX_SCALAR_THREADS>>>(sc->pts,sc->cnt,sc->projective);
#else
    scalar_kernel<<<blocks,threads>>>(sc->pts,sc->cnt,sc->projective);
#endif
    finish_batch_kernel<<<(blocks+PX_SPLIT_FINISH-1)/PX_SPLIT_FINISH,threads>>>(job,sc->projective,sc->cnt,hits,nhits);
#else
    mult_kernel<<<(blocks + PX_INV_BATCH - 1) / PX_INV_BATCH, threads>>>(job, sc->pts, sc->cnt, hits, nhits);
#endif
#else
    (void)scratch;
    mine_kernel<<<blocks, threads>>>(job, nonce_base, stride, iters, hits, nhits);
#endif
}
#if PX_XONLY && PX_SLICES > 1
static void *v_alloc(size_t n) { return v_alloc_slices(n); }
static void v_free(void *scratch) { v_free_slices(scratch); }
#endif
static int v_test(const JobDev *d_job, uint64_t nonce, uint8_t beta[64], uint8_t h[32], uint8_t cg[32]) {
    uint8_t *d_beta, *d_h, *d_cg; int *d_ok;
    CUDA_CHECK(cudaMalloc(&d_beta, 64)); CUDA_CHECK(cudaMalloc(&d_h, 32)); CUDA_CHECK(cudaMalloc(&d_cg, 32)); CUDA_CHECK(cudaMalloc(&d_ok, 4));
    test_kernel<<<1, 1>>>(d_job, nonce, d_beta, d_h, d_cg, d_ok);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    int ok; CUDA_CHECK(cudaMemcpy(&ok, d_ok, 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(beta, d_beta, 64, cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(h, d_h, 32, cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(cg, d_cg, 32, cudaMemcpyDeviceToHost));
    cudaFree(d_beta); cudaFree(d_h); cudaFree(d_cg); cudaFree(d_ok);
    return ok;
}
static int v_fetest(uint32_t seed, int n, uint8_t dump[128]) {
    int *d_fail; uint8_t *d_dump; CUDA_CHECK(cudaMalloc(&d_fail, 4)); CUDA_CHECK(cudaMalloc(&d_dump, 128));
    fe_test_kernel<<<1, 1>>>(seed, n, d_fail, d_dump);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    int fail; CUDA_CHECK(cudaMemcpy(&fail, d_fail, 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dump, d_dump, 128, cudaMemcpyDeviceToHost));
    cudaFree(d_fail); cudaFree(d_dump);
    return fail;
}
static const VariantOps ops = { PX_VARIANT_NAME,
#ifdef PX_FE32
    true,
#else
    false,
#endif
    PX_WINDOW, PX_LB_BLOCKS, PX_COMPACT != 0, v_set_scalar, v_init, v_alloc, v_free, v_mine, v_test, v_fetest };
#undef PX_INV_BATCH
#undef PX_PREFETCH
#undef PX_FE26
#undef PX_CSTEP
#undef PX_CSTEP2
#undef PX_CSHIFT
#undef PX_MERGE8
#undef PX_SQCHAIN
#undef PX_WIDECHAIN
#undef PX_FOLDSHIFT
#undef PX_LB_SCALAR
#undef PX_LADDER
#undef PX_LADDER_CSWAP
#undef PX_CORUN
#undef PX_SMALL_TPB
#undef PX_LADDER_MAXREG
#undef PX_LADDER_TPB
#undef PX_LADDER_BPS
#undef PX_LADDER_ATTR
#undef PX_SMALL_ATTR
#undef PX_SLICES
#undef PX_LADDER_NB
#undef PX_XSTEP
#undef PX_XONLY
#undef PX_H2C_K
#undef PX_JAC_STEPS
#undef PX_JAC_TRIM
#undef PX_SMALLMUL_WIDE
#undef PX_JAC_MASK
#undef PX_PFX_MEM
#undef PX_ZFREE
#undef PX_TAIL_SPLIT
#undef PX_LAD_KARA
#undef PX_ADDFOLD_ALU
#undef PX_H2C_PERSIST
#undef PX_ROW0C
#undef PX_FIN_SPLIT
#undef PX_LB_FSHA
#undef PX_LB_PREP
#undef PX_LB_FIN
#undef PX_PREP_ATTR
#undef PX_FIN_ATTR
#undef PX_JAC_B1
#undef PX_LAD_FSS
#undef PX_LAD_FSM
#undef PX_LB_SMALL
#undef PX_JAC_MAXR
#undef PX_NOBRANCH
#undef PX_PREP_BATCH
#undef PX_LB_LADDER
#undef PX_MONT_A
#undef PX_MONT_A24
#undef PX_SCALAR_THREADS
#undef PX_SQRT_DEC
#undef PX_H2C_MID
#undef PX_H2C_TAIL
#undef PX_MID_PREFIX
#undef PX_FOLDSHIFT_POW
#undef PX_HAVE_SQ_POW
#undef fe_sq_pow
#undef PX_LB_H2C
#undef PX_DBL6
#undef PX_ADD_NOT
#undef PX_TBL_CACHED
#undef PX_LOOP_ADD
#undef ge_dbl_opt
