// proxichorizo_device.cuh — the device side of ProxiChorizo: field arithmetic, Ed25519 points,
// SHA-512, the VRF attempt and the kernels. Included once per variant from
// ProxiChorizo.cu inside a namespace, with PX_FE32 / PX_WINDOW / PX_LB_BLOCKS /
// PX_VARIANT_NAME set by the includer; see PLATFORMS.md for what each costs.
#ifdef PX_FE32
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
    if (c) { if (fe_addsmall(h, 38u)) fe_addsmall(h, 38u); }
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
    // a wrap past zero added 2^256 = 38 too much
    if (b & 1u) { if (fe_subsmall(h, 38u)) fe_subsmall(h, 38u); }
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
    uint32_t top = r[8];
    if (top) {
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

__device__ __forceinline__ void fe_pow2k(fe &h, const fe &f, int k) { fe_copy(h, f); for (int i = 0; i < k; i++) fe_sq(h, h); }

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
__device__ __forceinline__ void fe_pow2k(fe &h, const fe &f, int k) { fe_copy(h, f); for (int i = 0; i < k; i++) fe_sq(h, h); }

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
    fe_mul(nui, nu, d_SQRT_M1);
    bool correct = fe_eq(check, u);
    bool flipped = fe_eq(check, nu);
    bool flipped_i = fe_eq(check, nui);
    fe_mul(rp, r, d_SQRT_M1);
    if (flipped || flipped_i) fe_copy(r, rp);
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

#if PX_WINDOW == 5
// width-5 wNAF over the constant scalar: 8 cached odd multiples P, 3P, ..., 15P,
// about 255 doublings and 43 mixed additions, no branch divergence (digits uniform)
__device__ void ge_scalarmult_const(ge &r, const ge &p) {
    geCached T[8];
    ge P2, acc;
    ge_dbl(P2, p);
    ge_copy(acc, p);
    ge_cache(T[0], acc);
    for (int i = 1; i < 8; i++) { ge_add(acc, acc, P2); ge_cache(T[i], acc); }
    ge_identity(r);
    for (int i = c_wnaf_len - 1; i >= 0; i--) {
        int d = c_wnaf[i];
        // the digit is uniform across the grid: no divergence
        if (d) ge_dbl(r, r); else ge_dbl_noT(r, r);
        if (d > 0) ge_add_cached(r, r, T[d >> 1], false);
        else if (d < 0) ge_add_cached(r, r, T[(-d) >> 1], true);
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
    ge T[16];
    ge_copy(T[1], p);
    for (int i = 2; i < 16; i++) {
        if (i & 1) ge_add(T[i], T[i - 1], p); else ge_dbl(T[i], T[i >> 1]);
    }
    // top window (bits 252..255) is 4..7 for a clamped scalar, never zero
    int w = (c_scalar[31] >> 4) & 0xF;
    ge_copy(r, T[w]);
    for (int i = 62; i >= 0; i--) {
        w = (c_scalar[i >> 1] >> ((i & 1) * 4)) & 0xF;
        ge_dbl_noT(r, r); ge_dbl_noT(r, r); ge_dbl_noT(r, r);
        if (w) { ge_dbl(r, r); ge_add(r, r, T[w]); } else ge_dbl_noT(r, r);
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
    ge G;
    ge_scalarmult_const(G, H);
    ge_mul8(G, G);
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
struct Scratch { HPoint *pts; uint64_t *nonce[2]; uint8_t *ctr[2]; unsigned *cnt; size_t n; };
#define PX_MAX_PASSES 12

__global__ void __launch_bounds__(256) h2c_kernel(const JobDev *job, uint64_t nonce_base, const uint64_t *in_nonce, const uint8_t *in_ctr, const unsigned *n_in_ptr,
                                                  HPoint *pts, unsigned *n_pts, uint64_t *out_nonce, uint8_t *out_ctr, unsigned *n_out) {
    unsigned n_in = *n_in_ptr;
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x, total = gridDim.x * blockDim.x;
    if (gid >= n_in) return;
    for (unsigned i = gid; i < n_in; i += total) {
        uint64_t nonce = in_nonce ? in_nonce[i] : nonce_base + i;
        uint8_t ctr = in_nonce ? in_ctr[i] : 0;
        uint8_t msg[128];
        int n = 0;
        msg[n++] = 0x03; msg[n++] = 0x01;
        for (int k = 0; k < 32; k++) msg[n++] = job->pk[k];
        for (int k = 0; k < job->prefix_len; k++) msg[n++] = job->prefix[k];
        for (int k = 0; k < 8; k++) msg[n++] = (uint8_t)(nonce >> (56 - 8 * k));
        msg[n++] = ctr; msg[n++] = 0;
        uint8_t dg[64];
        sha512_single(msg, n, dg);
        ge H;
        if (ge_frombytes(H, dg)) {
            unsigned o = atomicAdd(n_pts, 1u);
            fe_copy(pts[o].X, H.X); fe_copy(pts[o].Y, H.Y); pts[o].nonce = nonce;
        } else if (ctr < 255) {
            unsigned o = atomicAdd(n_out, 1u);
            out_nonce[o] = nonce; out_ctr[o] = ctr + 1;
        }
    }
}

// one point: cofactor, scalar multiplication, cofactor; false for the identity
__device__ __forceinline__ bool mult_point(ge &G, const HPoint &pt) {
    ge H;
    fe_copy(H.X, pt.X); fe_copy(H.Y, pt.Y); fe_1(H.Z); // T is not read by the doublings
    ge_mul8(H, H);
    if (ge_is_identity(H)) return false;
    ge_scalarmult_const(G, H);
    ge_mul8(G, G);
    return true;
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
__global__ void __launch_bounds__(PX_LB_THREADS, PX_LB_MULT_BLOCKS) mult_kernel(const JobDev *job, const HPoint *pts, const unsigned *n_ptr, Hit *hits, unsigned *nhits) {
    unsigned n = *n_ptr;
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x, total = gridDim.x * blockDim.x;
    for (unsigned i = gid; i < n; i += total) {
        ge G;
        if (!mult_point(G, pts[i])) continue;
        fe zi;
        fe_invert(zi, G.Z);
        finish_point(job, G, zi, pts[i].nonce, hits, nhits);
    }
}
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
static void v_set_scalar(const uint8_t *scalar) {
    CUDA_CHECK(cudaMemcpyToSymbol(c_scalar, scalar, 32));
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
static void *v_alloc(size_t n) {
#if PX_COMPACT
    Scratch *sc = new Scratch(); sc->n = n;
    CUDA_CHECK(cudaMalloc(&sc->pts, n * sizeof(HPoint)));
    for (int b = 0; b < 2; b++) { CUDA_CHECK(cudaMalloc(&sc->nonce[b], n * sizeof(uint64_t))); CUDA_CHECK(cudaMalloc(&sc->ctr[b], n)); }
    CUDA_CHECK(cudaMalloc(&sc->cnt, (PX_MAX_PASSES + 2) * sizeof(unsigned)));
    unsigned N = (unsigned)n; CUDA_CHECK(cudaMemcpy(sc->cnt + PX_MAX_PASSES + 1, &N, sizeof N, cudaMemcpyHostToDevice));
    return sc;
#else
    (void)n; return nullptr;
#endif
}
static void v_free(void *scratch) {
#if PX_COMPACT
    Scratch *sc = (Scratch *)scratch; if (!sc) return;
    cudaFree(sc->pts); for (int b = 0; b < 2; b++) { cudaFree(sc->nonce[b]); cudaFree(sc->ctr[b]); } cudaFree(sc->cnt); delete sc;
#else
    (void)scratch;
#endif
}
static void v_mine(int blocks, int threads, const JobDev *job, uint64_t nonce_base, uint64_t stride, int iters, Hit *hits, unsigned *nhits, void *scratch) {
#if PX_COMPACT
    Scratch *sc = (Scratch *)scratch;
    size_t N = stride * (size_t)iters;
    CUDA_CHECK(cudaMemsetAsync(sc->cnt, 0, (PX_MAX_PASSES + 1) * sizeof(unsigned)));
    int g1 = (int)((N + 255) / 256);
    // pass 0 over all nonces, then the retry queues, alternating buffers
    h2c_kernel<<<g1, 256>>>(job, nonce_base, nullptr, nullptr, sc->cnt + PX_MAX_PASSES + 1, sc->pts, sc->cnt, sc->nonce[0], sc->ctr[0], sc->cnt + 1);
    for (int p = 1; p < PX_MAX_PASSES; p++) {
        int in = (p - 1) & 1, out = p & 1;
        int gp = max(1, g1 >> p); // the queue halves every pass
        h2c_kernel<<<gp, 256>>>(job, nonce_base, sc->nonce[in], sc->ctr[in], sc->cnt + p, sc->pts, sc->cnt, sc->nonce[out], sc->ctr[out], sc->cnt + p + 1);
    }
    mult_kernel<<<blocks, threads>>>(job, sc->pts, sc->cnt, hits, nhits);
#else
    (void)scratch;
    mine_kernel<<<blocks, threads>>>(job, nonce_base, stride, iters, hits, nhits);
#endif
}
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
