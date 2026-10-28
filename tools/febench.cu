// febench.cu — field multiply/square microbenchmark: shipped 8x32 row/carry-chain arithmetic vs
// product-scanning (comba) with fused IMAD.WIDE + predicate carries. Checks canonical equality on random
// (lazily reduced, full 256-bit) inputs, then times G field-ops/s with 4 independent chains per thread.
// nvcc -O3 -arch=sm_120 -o febench febench.cu
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s line %d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
struct fe { uint32_t v[8]; };

// ---------------- shipped arithmetic (copied from proxichorizo_device.cuh, PX_FE32) ----------------
__device__ __forceinline__ uint32_t fe_addsmall(fe &h, uint32_t a) {
    uint32_t c;
    asm("add.cc.u32 %0, %0, %9;\n\taddc.cc.u32 %1, %1, 0;\n\taddc.cc.u32 %2, %2, 0;\n\taddc.cc.u32 %3, %3, 0;\n\taddc.cc.u32 %4, %4, 0;\n\taddc.cc.u32 %5, %5, 0;\n\taddc.cc.u32 %6, %6, 0;\n\taddc.cc.u32 %7, %7, 0;\n\taddc.u32 %8, 0, 0;"
        : "+r"(h.v[0]), "+r"(h.v[1]), "+r"(h.v[2]), "+r"(h.v[3]), "+r"(h.v[4]), "+r"(h.v[5]), "+r"(h.v[6]), "+r"(h.v[7]), "=&r"(c) : "r"(a));
    return c;
}
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
#define PX_ROW1(R, a, b) asm("mad.lo.cc.u32 %0, %3, %4, %0;\n\taddc.u32 %1, %1, 0;\n\tmad.hi.cc.u32 %1, %3, %4, %1;\n\taddc.u32 %2, %2, 0;" : "+r"((R)[0]), "+r"((R)[1]), "+r"((R)[2]) : "r"(a), "r"((b)[0]))
#define PX_ROW2(R, a, b) asm("mad.lo.cc.u32 %0, %4, %5, %0;\n\tmadc.lo.cc.u32 %1, %4, %6, %1;\n\taddc.u32 %2, %2, 0;\n\tmad.hi.cc.u32 %1, %4, %5, %1;\n\tmadc.hi.cc.u32 %2, %4, %6, %2;\n\taddc.u32 %3, %3, 0;" : "+r"((R)[0]), "+r"((R)[1]), "+r"((R)[2]), "+r"((R)[3]) : "r"(a), "r"((b)[0]), "r"((b)[1]))
#define PX_ROW3(R, a, b) asm("mad.lo.cc.u32 %0, %5, %6, %0;\n\tmadc.lo.cc.u32 %1, %5, %7, %1;\n\tmadc.lo.cc.u32 %2, %5, %8, %2;\n\taddc.u32 %3, %3, 0;\n\tmad.hi.cc.u32 %1, %5, %6, %1;\n\tmadc.hi.cc.u32 %2, %5, %7, %2;\n\tmadc.hi.cc.u32 %3, %5, %8, %3;\n\taddc.u32 %4, %4, 0;" : "+r"((R)[0]), "+r"((R)[1]), "+r"((R)[2]), "+r"((R)[3]), "+r"((R)[4]) : "r"(a), "r"((b)[0]), "r"((b)[1]), "r"((b)[2]))
#define PX_ROW4(R, a, b) asm("mad.lo.cc.u32 %0, %6, %7, %0;\n\tmadc.lo.cc.u32 %1, %6, %8, %1;\n\tmadc.lo.cc.u32 %2, %6, %9, %2;\n\tmadc.lo.cc.u32 %3, %6, %10, %3;\n\taddc.u32 %4, %4, 0;\n\tmad.hi.cc.u32 %1, %6, %7, %1;\n\tmadc.hi.cc.u32 %2, %6, %8, %2;\n\tmadc.hi.cc.u32 %3, %6, %9, %3;\n\tmadc.hi.cc.u32 %4, %6, %10, %4;\n\taddc.u32 %5, %5, 0;" : "+r"((R)[0]), "+r"((R)[1]), "+r"((R)[2]), "+r"((R)[3]), "+r"((R)[4]), "+r"((R)[5]) : "r"(a), "r"((b)[0]), "r"((b)[1]), "r"((b)[2]), "r"((b)[3]))
#define PX_ROW5(R, a, b) asm("mad.lo.cc.u32 %0, %7, %8, %0;\n\tmadc.lo.cc.u32 %1, %7, %9, %1;\n\tmadc.lo.cc.u32 %2, %7, %10, %2;\n\tmadc.lo.cc.u32 %3, %7, %11, %3;\n\tmadc.lo.cc.u32 %4, %7, %12, %4;\n\taddc.u32 %5, %5, 0;\n\tmad.hi.cc.u32 %1, %7, %8, %1;\n\tmadc.hi.cc.u32 %2, %7, %9, %2;\n\tmadc.hi.cc.u32 %3, %7, %10, %3;\n\tmadc.hi.cc.u32 %4, %7, %11, %4;\n\tmadc.hi.cc.u32 %5, %7, %12, %5;\n\taddc.u32 %6, %6, 0;" : "+r"((R)[0]), "+r"((R)[1]), "+r"((R)[2]), "+r"((R)[3]), "+r"((R)[4]), "+r"((R)[5]), "+r"((R)[6]) : "r"(a), "r"((b)[0]), "r"((b)[1]), "r"((b)[2]), "r"((b)[3]), "r"((b)[4]))
#define PX_ROW6(R, a, b) asm("mad.lo.cc.u32 %0, %8, %9, %0;\n\tmadc.lo.cc.u32 %1, %8, %10, %1;\n\tmadc.lo.cc.u32 %2, %8, %11, %2;\n\tmadc.lo.cc.u32 %3, %8, %12, %3;\n\tmadc.lo.cc.u32 %4, %8, %13, %4;\n\tmadc.lo.cc.u32 %5, %8, %14, %5;\n\taddc.u32 %6, %6, 0;\n\tmad.hi.cc.u32 %1, %8, %9, %1;\n\tmadc.hi.cc.u32 %2, %8, %10, %2;\n\tmadc.hi.cc.u32 %3, %8, %11, %3;\n\tmadc.hi.cc.u32 %4, %8, %12, %4;\n\tmadc.hi.cc.u32 %5, %8, %13, %5;\n\tmadc.hi.cc.u32 %6, %8, %14, %6;\n\taddc.u32 %7, %7, 0;" : "+r"((R)[0]), "+r"((R)[1]), "+r"((R)[2]), "+r"((R)[3]), "+r"((R)[4]), "+r"((R)[5]), "+r"((R)[6]), "+r"((R)[7]) : "r"(a), "r"((b)[0]), "r"((b)[1]), "r"((b)[2]), "r"((b)[3]), "r"((b)[4]), "r"((b)[5]))
#define PX_ROW7(R, a, b) asm("mad.lo.cc.u32 %0, %9, %10, %0;\n\tmadc.lo.cc.u32 %1, %9, %11, %1;\n\tmadc.lo.cc.u32 %2, %9, %12, %2;\n\tmadc.lo.cc.u32 %3, %9, %13, %3;\n\tmadc.lo.cc.u32 %4, %9, %14, %4;\n\tmadc.lo.cc.u32 %5, %9, %15, %5;\n\tmadc.lo.cc.u32 %6, %9, %16, %6;\n\taddc.u32 %7, %7, 0;\n\tmad.hi.cc.u32 %1, %9, %10, %1;\n\tmadc.hi.cc.u32 %2, %9, %11, %2;\n\tmadc.hi.cc.u32 %3, %9, %12, %3;\n\tmadc.hi.cc.u32 %4, %9, %13, %4;\n\tmadc.hi.cc.u32 %5, %9, %14, %5;\n\tmadc.hi.cc.u32 %6, %9, %15, %6;\n\tmadc.hi.cc.u32 %7, %9, %16, %7;\n\taddc.u32 %8, %8, 0;" : "+r"((R)[0]), "+r"((R)[1]), "+r"((R)[2]), "+r"((R)[3]), "+r"((R)[4]), "+r"((R)[5]), "+r"((R)[6]), "+r"((R)[7]), "+r"((R)[8]) : "r"(a), "r"((b)[0]), "r"((b)[1]), "r"((b)[2]), "r"((b)[3]), "r"((b)[4]), "r"((b)[5]), "r"((b)[6]))
__device__ __forceinline__ void fe_fold512(fe &h, uint32_t r[18]) {
    uint32_t hi[8];
    for (int i = 0; i < 8; i++) { hi[i] = r[8 + i]; r[8 + i] = 0; }
    PX_MUL_ROW(r + 0, 38u, hi);
    for (int i = 0; i < 8; i++) h.v[i] = r[i];
    uint32_t top = r[8];
    { uint32_t c, k = 38u;
      asm("mad.lo.cc.u32 %0, %9, %10, %0;\n\taddc.cc.u32 %1, %1, 0;\n\taddc.cc.u32 %2, %2, 0;\n\taddc.cc.u32 %3, %3, 0;\n\taddc.cc.u32 %4, %4, 0;\n\taddc.cc.u32 %5, %5, 0;\n\taddc.cc.u32 %6, %6, 0;\n\taddc.cc.u32 %7, %7, 0;\n\taddc.u32 %8, 0, 0;"
          : "+r"(h.v[0]), "+r"(h.v[1]), "+r"(h.v[2]), "+r"(h.v[3]), "+r"(h.v[4]), "+r"(h.v[5]), "+r"(h.v[6]), "+r"(h.v[7]), "=&r"(c) : "r"(top), "r"(k));
      if (c) fe_addsmall(h, 38u); }
}
__device__ void fe_mul_old(fe &h, const fe &f, const fe &g) {
    uint32_t r[18]; for (int i = 0; i < 18; i++) r[i] = 0;
    PX_MUL_ROW(r + 0, f.v[0], g.v); PX_MUL_ROW(r + 1, f.v[1], g.v); PX_MUL_ROW(r + 2, f.v[2], g.v); PX_MUL_ROW(r + 3, f.v[3], g.v);
    PX_MUL_ROW(r + 4, f.v[4], g.v); PX_MUL_ROW(r + 5, f.v[5], g.v); PX_MUL_ROW(r + 6, f.v[6], g.v); PX_MUL_ROW(r + 7, f.v[7], g.v);
    fe_fold512(h, r);
}
__device__ void fe_sq_old(fe &h, const fe &f) {
    uint32_t r[18]; for (int i = 0; i < 18; i++) r[i] = 0;
    const uint32_t *a = f.v;
    PX_ROW7(r + 1, a[0], a + 1); PX_ROW6(r + 3, a[1], a + 2); PX_ROW5(r + 5, a[2], a + 3);
    PX_ROW4(r + 7, a[3], a + 4); PX_ROW3(r + 9, a[4], a + 5); PX_ROW2(r + 11, a[5], a + 6); PX_ROW1(r + 13, a[6], a + 7);
    for (int k = 15; k > 0; k--) r[k] = (r[k] << 1) | (r[k - 1] >> 31);
    r[0] <<= 1;
    uint32_t cin = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint32_t cout;
        asm("add.cc.u32 %0, %0, %4;\n\tmadc.lo.cc.u32 %0, %3, %3, %0;\n\tmadc.hi.cc.u32 %1, %3, %3, %1;\n\taddc.u32 %2, 0, 0;"
            : "+r"(r[2 * i]), "+r"(r[2 * i + 1]), "=&r"(cout) : "r"(a[i]), "r"(cin));
        cin = cout;
    }
    fe_fold512(h, r);
}

// ---------------- comba with fused IMAD.WIDE + predicate carry ----------------
// (c2:c1:c0) += a*b   -> IMAD.WIDE.U32 Rc0c1, P, a, b, Rc0c1 ; IADD.X c2, c2, RZ, P
#define CSTEP(c0, c1, c2, a, b) asm("mad.lo.cc.u32 %0, %3, %4, %0;\n\tmadc.hi.cc.u32 %1, %3, %4, %1;\n\taddc.u32 %2, %2, 0;" : "+r"(c0), "+r"(c1), "+r"(c2) : "r"(a), "r"(b))
// (c1:c0) += a*b, no third word (top column)
#define CSTEP2(c0, c1, a, b) asm("mad.lo.cc.u32 %0, %2, %3, %0;\n\tmadc.hi.u32 %1, %2, %3, %1;" : "+r"(c0), "+r"(c1) : "r"(a), "r"(b))
#define CSHIFT(c0, c1, c2) do { c0 = c1; c1 = c2; c2 = 0; } while (0)

// h += (t1:t0) at words 0,1 with the carry propagated; a wrap past 2^256 is folded as 38
__device__ __forceinline__ void fe_addtop(fe &h, uint32_t t0, uint32_t t1) {
    uint32_t c;
    asm("add.cc.u32 %0, %0, %9;\n\taddc.cc.u32 %1, %1, %10;\n\taddc.cc.u32 %2, %2, 0;\n\taddc.cc.u32 %3, %3, 0;\n\t"
        "addc.cc.u32 %4, %4, 0;\n\taddc.cc.u32 %5, %5, 0;\n\taddc.cc.u32 %6, %6, 0;\n\taddc.cc.u32 %7, %7, 0;\n\taddc.u32 %8, 0, 0;"
        : "+r"(h.v[0]), "+r"(h.v[1]), "+r"(h.v[2]), "+r"(h.v[3]), "+r"(h.v[4]), "+r"(h.v[5]), "+r"(h.v[6]), "+r"(h.v[7]), "=&r"(c) : "r"(t0), "r"(t1));
    if (c) fe_addsmall(h, 38u);
}
// h = f * g
__device__ void fe_mul_comba(fe &h, const fe &f, const fe &g) {
    const uint32_t *a = f.v, *b = g.v;
    uint32_t r[8];                      // words 8..15 of the product
    uint32_t o[8];                      // output kept apart: h may alias f or g (fe_mul(x, x, y))
    uint32_t c0 = 0, c1 = 0, c2 = 0;
    // high columns k = 8..14
    CSTEP(c0, c1, c2, a[1], b[7]); CSTEP(c0, c1, c2, a[2], b[6]); CSTEP(c0, c1, c2, a[3], b[5]); CSTEP(c0, c1, c2, a[4], b[4]); CSTEP(c0, c1, c2, a[5], b[3]); CSTEP(c0, c1, c2, a[6], b[2]); CSTEP(c0, c1, c2, a[7], b[1]);
    r[0] = c0; CSHIFT(c0, c1, c2);
    CSTEP(c0, c1, c2, a[2], b[7]); CSTEP(c0, c1, c2, a[3], b[6]); CSTEP(c0, c1, c2, a[4], b[5]); CSTEP(c0, c1, c2, a[5], b[4]); CSTEP(c0, c1, c2, a[6], b[3]); CSTEP(c0, c1, c2, a[7], b[2]);
    r[1] = c0; CSHIFT(c0, c1, c2);
    CSTEP(c0, c1, c2, a[3], b[7]); CSTEP(c0, c1, c2, a[4], b[6]); CSTEP(c0, c1, c2, a[5], b[5]); CSTEP(c0, c1, c2, a[6], b[4]); CSTEP(c0, c1, c2, a[7], b[3]);
    r[2] = c0; CSHIFT(c0, c1, c2);
    CSTEP(c0, c1, c2, a[4], b[7]); CSTEP(c0, c1, c2, a[5], b[6]); CSTEP(c0, c1, c2, a[6], b[5]); CSTEP(c0, c1, c2, a[7], b[4]);
    r[3] = c0; CSHIFT(c0, c1, c2);
    CSTEP(c0, c1, c2, a[5], b[7]); CSTEP(c0, c1, c2, a[6], b[6]); CSTEP(c0, c1, c2, a[7], b[5]);
    r[4] = c0; CSHIFT(c0, c1, c2);
    CSTEP(c0, c1, c2, a[6], b[7]); CSTEP(c0, c1, c2, a[7], b[6]);
    r[5] = c0; CSHIFT(c0, c1, c2);
    CSTEP2(c0, c1, a[7], b[7]);
    r[6] = c0; r[7] = c1;
    // low columns k = 0..7, each seeded with 38 * word(8+k)
    c0 = c1 = c2 = 0;
    CSTEP(c0, c1, c2, r[0], 38u); CSTEP(c0, c1, c2, a[0], b[0]);
    o[0] = c0; CSHIFT(c0, c1, c2);
    CSTEP(c0, c1, c2, r[1], 38u); CSTEP(c0, c1, c2, a[0], b[1]); CSTEP(c0, c1, c2, a[1], b[0]);
    o[1] = c0; CSHIFT(c0, c1, c2);
    CSTEP(c0, c1, c2, r[2], 38u); CSTEP(c0, c1, c2, a[0], b[2]); CSTEP(c0, c1, c2, a[1], b[1]); CSTEP(c0, c1, c2, a[2], b[0]);
    o[2] = c0; CSHIFT(c0, c1, c2);
    CSTEP(c0, c1, c2, r[3], 38u); CSTEP(c0, c1, c2, a[0], b[3]); CSTEP(c0, c1, c2, a[1], b[2]); CSTEP(c0, c1, c2, a[2], b[1]); CSTEP(c0, c1, c2, a[3], b[0]);
    o[3] = c0; CSHIFT(c0, c1, c2);
    CSTEP(c0, c1, c2, r[4], 38u); CSTEP(c0, c1, c2, a[0], b[4]); CSTEP(c0, c1, c2, a[1], b[3]); CSTEP(c0, c1, c2, a[2], b[2]); CSTEP(c0, c1, c2, a[3], b[1]); CSTEP(c0, c1, c2, a[4], b[0]);
    o[4] = c0; CSHIFT(c0, c1, c2);
    CSTEP(c0, c1, c2, r[5], 38u); CSTEP(c0, c1, c2, a[0], b[5]); CSTEP(c0, c1, c2, a[1], b[4]); CSTEP(c0, c1, c2, a[2], b[3]); CSTEP(c0, c1, c2, a[3], b[2]); CSTEP(c0, c1, c2, a[4], b[1]); CSTEP(c0, c1, c2, a[5], b[0]);
    o[5] = c0; CSHIFT(c0, c1, c2);
    CSTEP(c0, c1, c2, r[6], 38u); CSTEP(c0, c1, c2, a[0], b[6]); CSTEP(c0, c1, c2, a[1], b[5]); CSTEP(c0, c1, c2, a[2], b[4]); CSTEP(c0, c1, c2, a[3], b[3]); CSTEP(c0, c1, c2, a[4], b[2]); CSTEP(c0, c1, c2, a[5], b[1]); CSTEP(c0, c1, c2, a[6], b[0]);
    o[6] = c0; CSHIFT(c0, c1, c2);
    CSTEP(c0, c1, c2, r[7], 38u); CSTEP(c0, c1, c2, a[0], b[7]); CSTEP(c0, c1, c2, a[1], b[6]); CSTEP(c0, c1, c2, a[2], b[5]); CSTEP(c0, c1, c2, a[3], b[4]); CSTEP(c0, c1, c2, a[4], b[3]); CSTEP(c0, c1, c2, a[5], b[2]); CSTEP(c0, c1, c2, a[6], b[1]); CSTEP(c0, c1, c2, a[7], b[0]);
    o[7] = c0;
    // (c2:c1) is the part above 2^256 (< 2^36: the low columns carry up to 2^35 past 2^256); times 38 it spans words 0 and 1
    uint64_t t; asm("mul.wide.u32 %0, %1, 38;" : "=l"(t) : "r"(c1));
    for (int i = 0; i < 8; i++) h.v[i] = o[i];
    fe_addtop(h, (uint32_t)t, (uint32_t)(t >> 32) + c2 * 38u);
}

// h = f^2: the 28 cross products once (comba), doubled by a 16-word shift, the 8 squares added in one
// carry chain of fused wide multiplies, then the 38-fold over exact words
__device__ void fe_sq_comba(fe &h, const fe &f) {
    const uint32_t *a = f.v;
    uint32_t s[16], o[8];
    uint32_t c0 = 0, c1 = 0, c2 = 0;
    s[0] = 0;
    CSTEP(c0, c1, c2, a[0], a[1]); s[1] = c0; CSHIFT(c0, c1, c2);
    CSTEP(c0, c1, c2, a[0], a[2]); s[2] = c0; CSHIFT(c0, c1, c2);
    CSTEP(c0, c1, c2, a[0], a[3]); CSTEP(c0, c1, c2, a[1], a[2]); s[3] = c0; CSHIFT(c0, c1, c2);
    CSTEP(c0, c1, c2, a[0], a[4]); CSTEP(c0, c1, c2, a[1], a[3]); s[4] = c0; CSHIFT(c0, c1, c2);
    CSTEP(c0, c1, c2, a[0], a[5]); CSTEP(c0, c1, c2, a[1], a[4]); CSTEP(c0, c1, c2, a[2], a[3]); s[5] = c0; CSHIFT(c0, c1, c2);
    CSTEP(c0, c1, c2, a[0], a[6]); CSTEP(c0, c1, c2, a[1], a[5]); CSTEP(c0, c1, c2, a[2], a[4]); s[6] = c0; CSHIFT(c0, c1, c2);
    CSTEP(c0, c1, c2, a[0], a[7]); CSTEP(c0, c1, c2, a[1], a[6]); CSTEP(c0, c1, c2, a[2], a[5]); CSTEP(c0, c1, c2, a[3], a[4]); s[7] = c0; CSHIFT(c0, c1, c2);
    CSTEP(c0, c1, c2, a[1], a[7]); CSTEP(c0, c1, c2, a[2], a[6]); CSTEP(c0, c1, c2, a[3], a[5]); s[8] = c0; CSHIFT(c0, c1, c2);
    CSTEP(c0, c1, c2, a[2], a[7]); CSTEP(c0, c1, c2, a[3], a[6]); CSTEP(c0, c1, c2, a[4], a[5]); s[9] = c0; CSHIFT(c0, c1, c2);
    CSTEP(c0, c1, c2, a[3], a[7]); CSTEP(c0, c1, c2, a[4], a[6]); s[10] = c0; CSHIFT(c0, c1, c2);
    CSTEP(c0, c1, c2, a[4], a[7]); CSTEP(c0, c1, c2, a[5], a[6]); s[11] = c0; CSHIFT(c0, c1, c2);
    CSTEP(c0, c1, c2, a[5], a[7]); s[12] = c0; CSHIFT(c0, c1, c2);
    CSTEP2(c0, c1, a[6], a[7]); s[13] = c0; s[14] = c1; s[15] = 0;
    #pragma unroll
    for (int k = 15; k > 0; k--) s[k] = (s[k] << 1) | (s[k - 1] >> 31);   // the cross sum is < 2^480: nothing lost
    s[0] <<= 1;
    // squares in one carry chain through the flags: s[0] is written, not read (it is zero before this point),
    // so no two operands of the statement carry a provably equal value
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
    // fold: column k = 38*s[8+k] + s[k] + carry never exceeds two words
    uint32_t cin = 0;
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        asm("mad.lo.cc.u32 %0, %2, 38, %3;\n\tmadc.hi.u32 %1, %2, 38, 0;\n\tadd.cc.u32 %0, %0, %4;\n\taddc.u32 %1, %1, 0;"
            : "=&r"(c0), "=&r"(c1) : "r"(s[8 + k]), "r"(s[k]), "r"(cin));
        o[k] = c0; cin = c1;
    }
    for (int i = 0; i < 8; i++) h.v[i] = o[i];
    fe_addtop(h, cin * 38u, 0u);
}

// ---------------- Karatsuba: three independent 4x4 comba products, 48 wide multiplies instead of 64 ----------------
// (c1:c0) column sums of a 4x4 product never exceed 4 products, so the third word is needed
__device__ __forceinline__ void mul4(uint32_t r[8], const uint32_t *a, const uint32_t *b) {
    uint32_t c0 = 0, c1 = 0, c2 = 0;
    CSTEP(c0, c1, c2, a[0], b[0]); r[0] = c0; CSHIFT(c0, c1, c2);
    CSTEP(c0, c1, c2, a[0], b[1]); CSTEP(c0, c1, c2, a[1], b[0]); r[1] = c0; CSHIFT(c0, c1, c2);
    CSTEP(c0, c1, c2, a[0], b[2]); CSTEP(c0, c1, c2, a[1], b[1]); CSTEP(c0, c1, c2, a[2], b[0]); r[2] = c0; CSHIFT(c0, c1, c2);
    CSTEP(c0, c1, c2, a[0], b[3]); CSTEP(c0, c1, c2, a[1], b[2]); CSTEP(c0, c1, c2, a[2], b[1]); CSTEP(c0, c1, c2, a[3], b[0]); r[3] = c0; CSHIFT(c0, c1, c2);
    CSTEP(c0, c1, c2, a[1], b[3]); CSTEP(c0, c1, c2, a[2], b[2]); CSTEP(c0, c1, c2, a[3], b[1]); r[4] = c0; CSHIFT(c0, c1, c2);
    CSTEP(c0, c1, c2, a[2], b[3]); CSTEP(c0, c1, c2, a[3], b[2]); r[5] = c0; CSHIFT(c0, c1, c2);
    CSTEP2(c0, c1, a[3], b[3]); r[6] = c0; r[7] = c1;
}
__device__ void fe_mul_kara(fe &h, const fe &f, const fe &g) {
    const uint32_t *a = f.v, *b = g.v;
    uint32_t p0[8], p2[8], m[8];
    mul4(p0, a, b);            // B*D   (low halves)
    mul4(p2, a + 4, b + 4);    // A*C   (high halves)
    // S = A+B, T = C+D (129 bits each: 4 words + a carry bit)
    uint32_t S[4], T[4], sc, tc;
    asm("add.cc.u32 %0, %5, %9;\n\taddc.cc.u32 %1, %6, %10;\n\taddc.cc.u32 %2, %7, %11;\n\taddc.cc.u32 %3, %8, %12;\n\taddc.u32 %4, 0, 0;"
        : "=&r"(S[0]), "=&r"(S[1]), "=&r"(S[2]), "=&r"(S[3]), "=&r"(sc) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]));
    asm("add.cc.u32 %0, %5, %9;\n\taddc.cc.u32 %1, %6, %10;\n\taddc.cc.u32 %2, %7, %11;\n\taddc.cc.u32 %3, %8, %12;\n\taddc.u32 %4, 0, 0;"
        : "=&r"(T[0]), "=&r"(T[1]), "=&r"(T[2]), "=&r"(T[3]), "=&r"(tc) : "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), "r"(b[4]), "r"(b[5]), "r"(b[6]), "r"(b[7]));
    mul4(m, S, T);             // S*T, 256 bits
    // M = S*T + sc*T<<128 + tc*S<<128 + sc*tc<<256   (fits 258 bits: words m[0..7], m8, m9)
    uint32_t ms = 0u - sc, mt = 0u - tc, m8, m9;
    asm("add.cc.u32 %0, %0, %5;\n\taddc.cc.u32 %1, %1, %6;\n\taddc.cc.u32 %2, %2, %7;\n\taddc.cc.u32 %3, %3, %8;\n\taddc.u32 %4, 0, 0;"
        : "+r"(m[4]), "+r"(m[5]), "+r"(m[6]), "+r"(m[7]), "=&r"(m8) : "r"(T[0] & ms), "r"(T[1] & ms), "r"(T[2] & ms), "r"(T[3] & ms));
    asm("add.cc.u32 %0, %0, %6;\n\taddc.cc.u32 %1, %1, %7;\n\taddc.cc.u32 %2, %2, %8;\n\taddc.cc.u32 %3, %3, %9;\n\taddc.cc.u32 %4, %4, %10;\n\taddc.u32 %5, 0, 0;"
        : "+r"(m[4]), "+r"(m[5]), "+r"(m[6]), "+r"(m[7]), "+r"(m8), "=&r"(m9) : "r"(S[0] & mt), "r"(S[1] & mt), "r"(S[2] & mt), "r"(S[3] & mt), "r"(sc & tc));
    // M -= p0, M -= p2 (M = A*D + B*C >= 0, so no final borrow)
    asm("sub.cc.u32 %0, %0, %10;\n\tsubc.cc.u32 %1, %1, %11;\n\tsubc.cc.u32 %2, %2, %12;\n\tsubc.cc.u32 %3, %3, %13;\n\tsubc.cc.u32 %4, %4, %14;\n\tsubc.cc.u32 %5, %5, %15;\n\tsubc.cc.u32 %6, %6, %16;\n\tsubc.cc.u32 %7, %7, %17;\n\tsubc.cc.u32 %8, %8, 0;\n\tsubc.u32 %9, %9, 0;"
        : "+r"(m[0]), "+r"(m[1]), "+r"(m[2]), "+r"(m[3]), "+r"(m[4]), "+r"(m[5]), "+r"(m[6]), "+r"(m[7]), "+r"(m8), "+r"(m9)
        : "r"(p0[0]), "r"(p0[1]), "r"(p0[2]), "r"(p0[3]), "r"(p0[4]), "r"(p0[5]), "r"(p0[6]), "r"(p0[7]));
    asm("sub.cc.u32 %0, %0, %10;\n\tsubc.cc.u32 %1, %1, %11;\n\tsubc.cc.u32 %2, %2, %12;\n\tsubc.cc.u32 %3, %3, %13;\n\tsubc.cc.u32 %4, %4, %14;\n\tsubc.cc.u32 %5, %5, %15;\n\tsubc.cc.u32 %6, %6, %16;\n\tsubc.cc.u32 %7, %7, %17;\n\tsubc.cc.u32 %8, %8, 0;\n\tsubc.u32 %9, %9, 0;"
        : "+r"(m[0]), "+r"(m[1]), "+r"(m[2]), "+r"(m[3]), "+r"(m[4]), "+r"(m[5]), "+r"(m[6]), "+r"(m[7]), "+r"(m8), "+r"(m9)
        : "r"(p2[0]), "r"(p2[1]), "r"(p2[2]), "r"(p2[3]), "r"(p2[4]), "r"(p2[5]), "r"(p2[6]), "r"(p2[7]));
    // product words: w[0..3] = p0[0..3]; w[4..7] = p0[4..7] + m[0..3]; w[8..15] = p2 + m[4..7], m8, m9 with carries
    uint32_t w8, w9, w10, w11, w12, w13, w14, w15;
    asm("add.cc.u32 %0, %0, %12;\n\taddc.cc.u32 %1, %1, %13;\n\taddc.cc.u32 %2, %2, %14;\n\taddc.cc.u32 %3, %3, %15;\n\t"
        "addc.cc.u32 %4, %16, %20;\n\taddc.cc.u32 %5, %17, %21;\n\taddc.cc.u32 %6, %18, %22;\n\taddc.cc.u32 %7, %19, %23;\n\t"
        "addc.cc.u32 %8, %24, %26;\n\taddc.cc.u32 %9, %25, %27;\n\taddc.cc.u32 %10, %28, 0;\n\taddc.u32 %11, %29, 0;"
        : "+r"(p0[4]), "+r"(p0[5]), "+r"(p0[6]), "+r"(p0[7]), "=&r"(w8), "=&r"(w9), "=&r"(w10), "=&r"(w11), "=&r"(w12), "=&r"(w13), "=&r"(w14), "=&r"(w15)
        : "r"(m[0]), "r"(m[1]), "r"(m[2]), "r"(m[3]), "r"(p2[0]), "r"(p2[1]), "r"(p2[2]), "r"(p2[3]), "r"(m[4]), "r"(m[5]), "r"(m[6]), "r"(m[7]), "r"(p2[4]), "r"(p2[5]), "r"(m8), "r"(m9), "r"(p2[6]), "r"(p2[7]));
    // fold: h = lo + 38*hi over exact words (leftover < 39)
    uint32_t hi[8] = { w8, w9, w10, w11, w12, w13, w14, w15 }, o[8];
    uint32_t c0 = 0, c1 = 0, c2 = 0;
    #pragma unroll
    for (int k = 0; k < 7; k++) {
        asm("add.cc.u32 %0, %0, %2;\n\taddc.u32 %1, %1, 0;" : "+r"(c0), "+r"(c1) : "r"(p0[k]));
        CSTEP(c0, c1, c2, hi[k], 38u);
        o[k] = c0; CSHIFT(c0, c1, c2);
    }
    asm("add.cc.u32 %0, %0, %2;\n\taddc.u32 %1, %1, 0;" : "+r"(c0), "+r"(c1) : "r"(p0[7]));
    CSTEP2(c0, c1, hi[7], 38u);
    o[7] = c0;
    for (int i = 0; i < 8; i++) h.v[i] = o[i];
    fe_addtop(h, c1 * 38u, 0u);
}

// canonical representative in [0, p)
__device__ void fe_canon(uint32_t out[8], const fe &f) {
    fe t = f;
    for (int rep = 0; rep < 2; rep++) {
        uint32_t s[8], b;
        asm("sub.cc.u32 %0, %9, 0xFFFFFFED;\n\tsubc.cc.u32 %1, %10, 0xFFFFFFFF;\n\tsubc.cc.u32 %2, %11, 0xFFFFFFFF;\n\tsubc.cc.u32 %3, %12, 0xFFFFFFFF;\n\tsubc.cc.u32 %4, %13, 0xFFFFFFFF;\n\tsubc.cc.u32 %5, %14, 0xFFFFFFFF;\n\tsubc.cc.u32 %6, %15, 0xFFFFFFFF;\n\tsubc.cc.u32 %7, %16, 0x7FFFFFFF;\n\tsubc.u32 %8, 0, 0;"
            : "=&r"(s[0]), "=&r"(s[1]), "=&r"(s[2]), "=&r"(s[3]), "=&r"(s[4]), "=&r"(s[5]), "=&r"(s[6]), "=&r"(s[7]), "=&r"(b)
            : "r"(t.v[0]), "r"(t.v[1]), "r"(t.v[2]), "r"(t.v[3]), "r"(t.v[4]), "r"(t.v[5]), "r"(t.v[6]), "r"(t.v[7]));
        if (!(b & 1u)) for (int i = 0; i < 8; i++) t.v[i] = s[i];
    }
    for (int i = 0; i < 8; i++) out[i] = t.v[i];
}

__device__ __forceinline__ void fe_copy_(fe &h, const fe &f) { for (int i = 0; i < 8; i++) h.v[i] = f.v[i]; }

__global__ void temps_kernel(unsigned *bad, uint32_t *dump) {
    fe x; for (int i = 0; i < 8; i++) x.v[i] = 0; x.v[0] = 121666;
    fe u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13;
    fe m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13;
    int st = 0;
#define SQM(hh, ff) fe_mul_comba(hh, ff, ff)
#define CHK(A, B) { uint32_t ca[8], cb[8]; fe_canon(ca, A); fe_canon(cb, B); bool ok = true; for (int i = 0; i < 8; i++) ok &= ca[i] == cb[i]; if (!ok) { *bad = 100 + st; for (int i = 0; i < 8; i++) { dump[i] = ca[i]; dump[8 + i] = cb[i]; } return; } st++; }
    fe_sq_comba(u0, x); SQM(m0, x); CHK(u0, m0)
    fe_sq_comba(u1, u0); fe_sq_comba(u1, u1); SQM(m1, m0); SQM(m1, m1); CHK(u1, m1)
    fe_mul_comba(u2, x, u1); fe_mul_comba(m2, x, m1); CHK(u2, m2)
    fe_mul_comba(u3, u0, u2); fe_mul_comba(m3, m0, m2); CHK(u3, m3)
    fe_sq_comba(u4, u3); SQM(m4, m3); CHK(u4, m4)
    fe_mul_comba(u5, u2, u4); fe_mul_comba(m5, m2, m4); CHK(u5, m5)
    fe_copy_(u6, u5); for (int i = 0; i < 5; i++) fe_sq_comba(u6, u6); fe_copy_(m6, m5); for (int i = 0; i < 5; i++) SQM(m6, m6); CHK(u6, m6)
    fe_mul_comba(u7, u6, u5); fe_mul_comba(m7, m6, m5); CHK(u7, m7)
    fe_copy_(u8, u7); for (int i = 0; i < 10; i++) fe_sq_comba(u8, u8); fe_copy_(m8, m7); for (int i = 0; i < 10; i++) SQM(m8, m8); CHK(u8, m8)
    fe_mul_comba(u9, u8, u7); fe_mul_comba(m9, m8, m7); CHK(u9, m9)
    fe_copy_(u10, u9); for (int i = 0; i < 20; i++) fe_sq_comba(u10, u10); fe_copy_(m10, m9); for (int i = 0; i < 20; i++) SQM(m10, m10); CHK(u10, m10)
    fe_mul_comba(u11, u10, u9); fe_mul_comba(m11, m10, m9); CHK(u11, m11)
    fe_copy_(u12, u11); for (int i = 0; i < 10; i++) fe_sq_comba(u12, u12); fe_copy_(m12, m11); for (int i = 0; i < 10; i++) SQM(m12, m12); CHK(u12, m12)
    fe_mul_comba(u13, u12, u7); fe_mul_comba(m13, m12, m7); CHK(u13, m13)
    *bad = 0;
}
__device__ __forceinline__ uint64_t splitmix(uint64_t &s) { uint64_t z = (s += 0x9E3779B97F4A7C15ULL); z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL; return z ^ (z >> 31); }
// correctness: random full-range inputs (lazy representatives), compare canonical outputs
__global__ void check_kernel(uint64_t seed, int n, unsigned *bad, uint32_t *dump) {
    uint64_t st = seed + blockIdx.x * 1315423911ULL + threadIdx.x;
    for (int it = 0; it < n; it++) {
        fe x, y;
        for (int i = 0; i < 4; i++) { uint64_t r = splitmix(st); x.v[2 * i] = (uint32_t)r; x.v[2 * i + 1] = (uint32_t)(r >> 32); r = splitmix(st); y.v[2 * i] = (uint32_t)r; y.v[2 * i + 1] = (uint32_t)(r >> 32); }
        if (it % 5 == 1) { for (int i = 0; i < 8; i++) x.v[i] = 0xFFFFFFFFu; }          // 2^256-1
        if (it % 5 == 2) { for (int i = 0; i < 8; i++) y.v[i] = 0xFFFFFFFFu; }
        if (it % 5 == 3) { for (int i = 0; i < 8; i++) x.v[i] = y.v[i] = 0xFFFFFFFFu; }
        if (it % 7 == 4) { x.v[0] = 0; }
        if (it % 11 == 5) { for (int i = 1; i < 8; i++) x.v[i] = 0; x.v[0] = (uint32_t)(it * 7919); }
        if (it % 11 == 6) { for (int i = 0; i < 8; i++) x.v[i] = (i == (it % 8)) ? (1u << (it % 32)) : 0u; }
        if (it % 13 == 7) { for (int i = 0; i < 8; i++) x.v[i] = 0; x.v[0] = 121666; }
        fe m1, m2, s1, s2; uint32_t c1[8], c2[8];
        fe_mul_old(m1, x, y); fe_mul_comba(m2, x, y); fe_canon(c1, m1); fe_canon(c2, m2);
        bool ok = true; for (int i = 0; i < 8; i++) ok &= c1[i] == c2[i];
        fe_sq_old(s1, x); fe_sq_comba(s2, x); uint32_t d1[8], d2[8]; fe_canon(d1, s1); fe_canon(d2, s2);
        { fe mk; fe_mul_kara(mk, x, y); uint32_t ck[8]; fe_canon(ck, mk); for (int i = 0; i < 8; i++) ok &= ck[i] == c1[i]; }
        bool ok2 = true; for (int i = 0; i < 8; i++) ok2 &= d1[i] == d2[i];
        if (!ok || !ok2) { unsigned idx = atomicAdd(bad, 1u); if (idx == 0) { for (int i = 0; i < 8; i++) { dump[i] = x.v[i]; dump[8 + i] = y.v[i]; dump[16 + i] = c1[i]; dump[24 + i] = c2[i]; dump[32 + i] = d1[i]; dump[40 + i] = d2[i]; } dump[48] = ok ? 0 : 1; dump[49] = ok2 ? 0 : 1; } }
    }
}
template<int V> __global__ void __launch_bounds__(256, 2) bench_kernel(const fe *in, fe *out, int iters) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    fe x0 = in[(tid * 4) & 1023], x1 = in[(tid * 4 + 1) & 1023], x2 = in[(tid * 4 + 2) & 1023], x3 = in[(tid * 4 + 3) & 1023];
    fe k = in[(tid + 7) & 1023];
    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        if (V == 0) { fe_mul_old(x0, x0, k); fe_mul_old(x1, x1, k); fe_mul_old(x2, x2, k); fe_mul_old(x3, x3, k); }
        if (V == 1) { fe_mul_comba(x0, x0, k); fe_mul_comba(x1, x1, k); fe_mul_comba(x2, x2, k); fe_mul_comba(x3, x3, k); }
        if (V == 2) { fe_sq_old(x0, x0); fe_sq_old(x1, x1); fe_sq_old(x2, x2); fe_sq_old(x3, x3); }
        if (V == 3) { fe_sq_comba(x0, x0); fe_sq_comba(x1, x1); fe_sq_comba(x2, x2); fe_sq_comba(x3, x3); }
        if (V == 4) { fe_mul_kara(x0, x0, k); fe_mul_kara(x1, x1, k); fe_mul_kara(x2, x2, k); fe_mul_kara(x3, x3, k); }
        if (V == 5) { fe_mul_comba(x0, x0, k); }
        if (V == 6) { fe_mul_kara(x0, x0, k); }
    }
    fe o; for (int i = 0; i < 8; i++) o.v[i] = x0.v[i] ^ x1.v[i] ^ x2.v[i] ^ x3.v[i];
    out[tid] = o;
}
static const char *VN[] = { "fe_mul shipped (rows, mad.lo/hi carry chains)", "fe_mul comba (IMAD.WIDE + predicate carry)", "fe_sq shipped", "fe_sq comba", "fe_mul karatsuba 3x(4x4 comba)", "fe_mul comba, 1 chain/thread (latency)", "fe_mul karatsuba, 1 chain/thread (latency)" };
template<int V> void run(int blocks, const fe *in, fe *out, int iters) {
    bench_kernel<V><<<blocks, 256>>>(in, out, 10); CK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0); bench_kernel<V><<<blocks, 256>>>(in, out, iters); cudaEventRecord(e1); CK(cudaDeviceSynchronize());
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    double ops = (double)blocks * 256.0 * (V >= 5 ? 1.0 : 4.0) * iters;
    printf("%-48s %8.1f G field-ops/s  (%.1f ms)\n", VN[V], ops / ms / 1e6, ms);
}
int main() {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0)); printf("%s sm_%d%d %d SMs\n", p.name, p.major, p.minor, p.multiProcessorCount);
    unsigned *bad; uint32_t *dump; CK(cudaMalloc(&bad, 4)); CK(cudaMalloc(&dump, 64 * 4)); CK(cudaMemset(bad, 0, 4));
    temps_kernel<<<1,1>>>(bad, dump); CK(cudaDeviceSynchronize()); { unsigned b2; CK(cudaMemcpy(&b2, bad, 4, cudaMemcpyDeviceToHost)); printf("temps_kernel (header bodies, many live fe temporaries, x=121666): %s\n", b2 ? "FAIL" : "PASS"); if (b2) { uint32_t d[16]; CK(cudaMemcpy(d, dump, 64, cudaMemcpyDeviceToHost)); printf("step %u\nsq :", b2-100); for (int i = 7; i >= 0; i--) printf(" %08x", d[i]); printf("\nmul:"); for (int i = 7; i >= 0; i--) printf(" %08x", d[8+i]); printf("\n"); } CK(cudaMemset(bad, 0, 4)); }
    check_kernel<<<512, 256>>>(12345, 400, bad, dump); CK(cudaDeviceSynchronize());
    unsigned nb; CK(cudaMemcpy(&nb, bad, 4, cudaMemcpyDeviceToHost));
    printf("exactness: %u mismatches over %d random mul+sq pairs (full-range lazy inputs incl. 2^256-1)\n", nb, 512 * 256 * 400);
    if (nb) { uint32_t d[64]; CK(cudaMemcpy(d, dump, 256, cudaMemcpyDeviceToHost)); printf("mul bad %u sq bad %u\nx:", d[48], d[49]); for (int i = 7; i >= 0; i--) printf(" %08x", d[i]); printf("\ny:"); for (int i = 7; i >= 0; i--) printf(" %08x", d[8 + i]); printf("\nmul old:"); for (int i = 7; i >= 0; i--) printf(" %08x", d[16 + i]); printf("\nmul new:"); for (int i = 7; i >= 0; i--) printf(" %08x", d[24 + i]); printf("\nsq old:"); for (int i = 7; i >= 0; i--) printf(" %08x", d[32 + i]); printf("\nsq new:"); for (int i = 7; i >= 0; i--) printf(" %08x", d[40 + i]); printf("\n"); return 1; }
    std::vector<fe> hin(1024); uint64_t st = 99; for (auto &f : hin) for (int i = 0; i < 8; i++) { st = st * 6364136223846793005ULL + 1442695040888963407ULL; f.v[i] = (uint32_t)(st >> 32); }
    fe *in, *out; int blocks = p.multiProcessorCount * 2 * 8;
    CK(cudaMalloc(&in, sizeof(fe) * 1024)); CK(cudaMalloc(&out, sizeof(fe) * blocks * 256)); CK(cudaMemcpy(in, hin.data(), sizeof(fe) * 1024, cudaMemcpyHostToDevice));
    run<0>(blocks, in, out, 2000); run<1>(blocks, in, out, 2000); run<4>(blocks, in, out, 2000); run<2>(blocks, in, out, 2000); run<3>(blocks, in, out, 2000);
    run<1>(blocks, in, out, 2000); run<4>(blocks, in, out, 2000); run<5>(blocks, in, out, 2000); run<6>(blocks, in, out, 2000);
    return 0;
}
