// proxichorizo.cl — ProxiChorizo device code for OpenCL (AMD and any other OpenCL 1.2 GPU).
// Same algorithm as proxichorizo_device.cuh: Ed25519 field + points, SHA-512, the ECVRF
// attempt, the fused kernel and the compacted two-kernel pipeline. Built per variant with
//   -DPX_FE32=1|0   field element as 8x32-bit limbs with 64-bit accumulators (1) or 5x51-bit (0)
//   -DPX_WINDOW=0|4|5   plain double-and-add / 4-bit table / width-5 wNAF scalar multiplication
//   -DPX_COMPACT=0|1    fused kernel / h2c passes + mult kernel
//   -DPX_WG=256         work-group size the kernels are compiled for
typedef uchar u8; typedef uint u32; typedef ulong u64;
#ifndef PX_FE32
#define PX_FE32 1
#endif
#ifndef PX_WINDOW
#define PX_WINDOW 5
#endif
#ifndef PX_COMPACT
#define PX_COMPACT 1
#endif
#ifndef PX_WG
#define PX_WG 256
#endif
#define MAX_HITS 256

typedef struct { u8 pk[32]; u8 prefix[64]; int prefix_len; int k; u8 beat[64]; } JobDev;
typedef struct { u64 nonce; u8 beta[64]; } Hit;

// ----------------------------------------------------------------------------
// field element
// ----------------------------------------------------------------------------
#if PX_FE32
// 8 x 32-bit limbs, lazily reduced (any 256-bit representative), 2^256 = 38 (mod p)
typedef struct { u32 v[8]; } fe;
static inline void fe_0(fe *h) { for (int i = 0; i < 8; i++) h->v[i] = 0; }
static inline void fe_1(fe *h) { fe_0(h); h->v[0] = 1; }
static inline void fe_copy(fe *h, const fe *f) { for (int i = 0; i < 8; i++) h->v[i] = f->v[i]; }
static inline void fe_reduce(fe *h) { (void)h; }
static inline u32 fe_addsmall(fe *h, u32 a) {
    u64 t = (u64)h->v[0] + a; h->v[0] = (u32)t; u32 c = (u32)(t >> 32);
    for (int i = 1; i < 8; i++) { t = (u64)h->v[i] + c; h->v[i] = (u32)t; c = (u32)(t >> 32); }
    return c;
}
static inline u32 fe_subsmall(fe *h, u32 a) {
    u64 t = (u64)h->v[0] - a; h->v[0] = (u32)t; u32 b = (u32)(t >> 32) & 1u;
    for (int i = 1; i < 8; i++) { t = (u64)h->v[i] - b; h->v[i] = (u32)t; b = (u32)(t >> 32) & 1u; }
    return b;
}
static inline void fe_add(fe *h, const fe *f, const fe *g) {
    u64 c = 0;
    for (int i = 0; i < 8; i++) { c += (u64)f->v[i] + g->v[i]; h->v[i] = (u32)c; c >>= 32; }
    if (c) { if (fe_addsmall(h, 38u)) fe_addsmall(h, 38u); }
}
static inline void fe_sub(fe *h, const fe *f, const fe *g) {
    u64 b = 0;
    for (int i = 0; i < 8; i++) { u64 t = (u64)f->v[i] - g->v[i] - b; h->v[i] = (u32)t; b = (t >> 32) & 1u; }
    if (b) { if (fe_subsmall(h, 38u)) fe_subsmall(h, 38u); }
}
static inline void fe_neg(fe *h, const fe *f) { fe z; fe_0(&z); fe_sub(h, &z, f); }
// fold the 512-bit product r[0..15] into h: lo + 38*hi, then the top limb once more
static inline void fe_fold512(fe *h, const u32 r[16]) {
    u64 c = 0;
    for (int i = 0; i < 8; i++) { c += (u64)r[i] + (u64)38u * r[8 + i]; h->v[i] = (u32)c; c >>= 32; }
    if (c) {
        u64 t = (u64)h->v[0] + c * 38u; h->v[0] = (u32)t; u32 cc = (u32)(t >> 32);
        for (int i = 1; i < 8; i++) { t = (u64)h->v[i] + cc; h->v[i] = (u32)t; cc = (u32)(t >> 32); }
        if (cc) fe_addsmall(h, 38u);
    }
}
// column-wise (comba) schoolbook: low and high halves of the products summed separately
static void fe_mul(fe *h, const fe *f, const fe *g) {
    u32 r[16];
    u64 lo = 0, hi = 0;
    #pragma unroll
    for (int k = 0; k < 15; k++) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int j = k - i;
            if (j >= 0 && j < 8) { u64 p = (u64)f->v[i] * g->v[j]; lo += (u32)p; hi += p >> 32; }
        }
        r[k] = (u32)lo; lo = (lo >> 32) + hi; hi = 0;
    }
    r[15] = (u32)lo;
    fe_fold512(h, r);
}
static void fe_sq(fe *h, const fe *f) {
    u32 r[16];
    u64 lo = 0, hi = 0;
    #pragma unroll
    for (int k = 0; k < 15; k++) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int j = k - i;
            if (j > i && j < 8) { u64 p = (u64)f->v[i] * f->v[j]; lo += (u64)(u32)p << 1; hi += (p >> 32) << 1; }
        }
        if ((k & 1) == 0) { u64 p = (u64)f->v[k >> 1] * f->v[k >> 1]; lo += (u32)p; hi += p >> 32; }
        r[k] = (u32)lo; lo = (lo >> 32) + hi; hi = 0;
    }
    r[15] = (u32)lo;
    fe_fold512(h, r);
}
static inline void fe_pow2k(fe *h, const fe *f, int k) { fe_copy(h, f); for (int i = 0; i < k; i++) fe_sq(h, h); }
__constant u32 FE_P[8] = { 0xFFFFFFEDu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0x7FFFFFFFu };
static inline void fe_canon(u32 out[8], const fe *f) {
    fe t; fe_copy(&t, f);
    for (int rep = 0; rep < 2; rep++) {
        u32 s[8]; u64 b = 0;
        for (int i = 0; i < 8; i++) { u64 x = (u64)t.v[i] - FE_P[i] - b; s[i] = (u32)x; b = (x >> 32) & 1u; }
        if (!b) for (int i = 0; i < 8; i++) t.v[i] = s[i];
    }
    for (int i = 0; i < 8; i++) out[i] = t.v[i];
}
static void fe_tobytes(u8 s[32], const fe *f) {
    u32 c[8]; fe_canon(c, f);
    for (int i = 0; i < 8; i++) { s[4 * i] = (u8)c[i]; s[4 * i + 1] = (u8)(c[i] >> 8); s[4 * i + 2] = (u8)(c[i] >> 16); s[4 * i + 3] = (u8)(c[i] >> 24); }
}
static void fe_frombytes(fe *h, const u8 s[32]) {
    for (int i = 0; i < 8; i++) h->v[i] = (u32)s[4 * i] | ((u32)s[4 * i + 1] << 8) | ((u32)s[4 * i + 2] << 16) | ((u32)s[4 * i + 3] << 24);
    h->v[7] &= 0x7FFFFFFFu;
}
static inline bool fe_isnegative(const fe *f) { u32 c[8]; fe_canon(c, f); return (c[0] & 1u) != 0; }
static inline bool fe_iszero(const fe *f) { u32 c[8]; fe_canon(c, f); u32 o = 0; for (int i = 0; i < 8; i++) o |= c[i]; return o == 0; }
static inline bool fe_eq(const fe *f, const fe *g) { fe t; fe_sub(&t, f, g); return fe_iszero(&t); }
#else
// 5 x 51-bit limbs with 64x64->128 products (mul_hi); the libsodium-style layout
typedef struct { u64 v[5]; } fe;
#define MASK51 0x7FFFFFFFFFFFFUL
typedef struct { u64 lo, hi; } u128;
static inline void mac(u128 *c, u64 a, u64 b) { u64 lo = a * b, hi = mul_hi(a, b); u64 nlo = c->lo + lo; c->hi += hi + (nlo < c->lo ? 1u : 0u); c->lo = nlo; }
static inline void add64(u128 *c, u64 a) { u64 n = c->lo + a; c->hi += (n < c->lo ? 1u : 0u); c->lo = n; }
static inline u64 shr51(const u128 *c) { return (c->lo >> 51) | (c->hi << 13); }
static inline void fe_0(fe *h) { for (int i = 0; i < 5; i++) h->v[i] = 0; }
static inline void fe_1(fe *h) { fe_0(h); h->v[0] = 1; }
static inline void fe_copy(fe *h, const fe *f) { for (int i = 0; i < 5; i++) h->v[i] = f->v[i]; }
static inline void fe_reduce(fe *h) {
    u64 c0 = h->v[0] >> 51, c1 = h->v[1] >> 51, c2 = h->v[2] >> 51, c3 = h->v[3] >> 51, c4 = h->v[4] >> 51;
    h->v[0] &= MASK51; h->v[1] &= MASK51; h->v[2] &= MASK51; h->v[3] &= MASK51; h->v[4] &= MASK51;
    h->v[0] += c4 * 19; h->v[1] += c0; h->v[2] += c1; h->v[3] += c2; h->v[4] += c3;
}
static inline void fe_add(fe *h, const fe *f, const fe *g) { for (int i = 0; i < 5; i++) h->v[i] = f->v[i] + g->v[i]; }
static inline void fe_sub(fe *h, const fe *f, const fe *g) {
    h->v[0] = f->v[0] + 36028797018963664UL - g->v[0];
    h->v[1] = f->v[1] + 36028797018963952UL - g->v[1];
    h->v[2] = f->v[2] + 36028797018963952UL - g->v[2];
    h->v[3] = f->v[3] + 36028797018963952UL - g->v[3];
    h->v[4] = f->v[4] + 36028797018963952UL - g->v[4];
    fe_reduce(h);
}
static inline void fe_neg(fe *h, const fe *f) { fe z; fe_0(&z); fe_sub(h, &z, f); }
static void fe_mul(fe *h, const fe *f, const fe *g) {
    u64 a0 = f->v[0], a1 = f->v[1], a2 = f->v[2], a3 = f->v[3], a4 = f->v[4];
    u64 b0 = g->v[0], b1 = g->v[1], b2 = g->v[2], b3 = g->v[3], b4 = g->v[4];
    u64 b1_19 = b1 * 19, b2_19 = b2 * 19, b3_19 = b3 * 19, b4_19 = b4 * 19;
    u128 c0 = {0, 0}, c1 = {0, 0}, c2 = {0, 0}, c3 = {0, 0}, c4 = {0, 0};
    mac(&c0, a0, b0); mac(&c0, a4, b1_19); mac(&c0, a3, b2_19); mac(&c0, a2, b3_19); mac(&c0, a1, b4_19);
    mac(&c1, a1, b0); mac(&c1, a0, b1); mac(&c1, a4, b2_19); mac(&c1, a3, b3_19); mac(&c1, a2, b4_19);
    mac(&c2, a2, b0); mac(&c2, a1, b1); mac(&c2, a0, b2); mac(&c2, a4, b3_19); mac(&c2, a3, b4_19);
    mac(&c3, a3, b0); mac(&c3, a2, b1); mac(&c3, a1, b2); mac(&c3, a0, b3); mac(&c3, a4, b4_19);
    mac(&c4, a4, b0); mac(&c4, a3, b1); mac(&c4, a2, b2); mac(&c4, a1, b3); mac(&c4, a0, b4);
    u64 o0, o1, o2, o3, o4;
    add64(&c1, shr51(&c0)); o0 = c0.lo & MASK51;
    add64(&c2, shr51(&c1)); o1 = c1.lo & MASK51;
    add64(&c3, shr51(&c2)); o2 = c2.lo & MASK51;
    add64(&c4, shr51(&c3)); o3 = c3.lo & MASK51;
    u64 carry = shr51(&c4); o4 = c4.lo & MASK51;
    o0 += carry * 19; carry = o0 >> 51; o0 &= MASK51; o1 += carry;
    h->v[0] = o0; h->v[1] = o1; h->v[2] = o2; h->v[3] = o3; h->v[4] = o4;
}
static void fe_sq(fe *h, const fe *f) {
    u64 a0 = f->v[0], a1 = f->v[1], a2 = f->v[2], a3 = f->v[3], a4 = f->v[4];
    u64 a3_19 = a3 * 19, a4_19 = a4 * 19, a0_2 = a0 * 2, a1_2 = a1 * 2;
    u128 c0 = {0, 0}, c1 = {0, 0}, c2 = {0, 0}, c3 = {0, 0}, c4 = {0, 0};
    mac(&c0, a0, a0);   mac(&c0, a1_2, a4_19); mac(&c0, a2 * 2, a3_19);
    mac(&c1, a3, a3_19); mac(&c1, a0_2, a1);   mac(&c1, a2 * 2, a4_19);
    mac(&c2, a1, a1);   mac(&c2, a0_2, a2);    mac(&c2, a4 * 2, a3_19);
    mac(&c3, a4, a4_19); mac(&c3, a0_2, a3);   mac(&c3, a1_2, a2);
    mac(&c4, a2, a2);   mac(&c4, a0_2, a4);    mac(&c4, a1_2, a3);
    u64 o0, o1, o2, o3, o4;
    add64(&c1, shr51(&c0)); o0 = c0.lo & MASK51;
    add64(&c2, shr51(&c1)); o1 = c1.lo & MASK51;
    add64(&c3, shr51(&c2)); o2 = c2.lo & MASK51;
    add64(&c4, shr51(&c3)); o3 = c3.lo & MASK51;
    u64 carry = shr51(&c4); o4 = c4.lo & MASK51;
    o0 += carry * 19; carry = o0 >> 51; o0 &= MASK51; o1 += carry;
    h->v[0] = o0; h->v[1] = o1; h->v[2] = o2; h->v[3] = o3; h->v[4] = o4;
}
static inline void fe_pow2k(fe *h, const fe *f, int k) { fe_copy(h, f); for (int i = 0; i < k; i++) fe_sq(h, h); }
static void fe_tobytes(u8 s[32], const fe *f) {
    fe h; fe_copy(&h, f); fe_reduce(&h);
    u64 q = (h.v[0] + 19) >> 51;
    q = (h.v[1] + q) >> 51; q = (h.v[2] + q) >> 51; q = (h.v[3] + q) >> 51; q = (h.v[4] + q) >> 51;
    h.v[0] += 19 * q;
    h.v[1] += h.v[0] >> 51; h.v[0] &= MASK51;
    h.v[2] += h.v[1] >> 51; h.v[1] &= MASK51;
    h.v[3] += h.v[2] >> 51; h.v[2] &= MASK51;
    h.v[4] += h.v[3] >> 51; h.v[3] &= MASK51;
    h.v[4] &= MASK51;
    u64 l0 = h.v[0], l1 = h.v[1], l2 = h.v[2], l3 = h.v[3], l4 = h.v[4];
    s[0] = (u8)l0; s[1] = (u8)(l0 >> 8); s[2] = (u8)(l0 >> 16); s[3] = (u8)(l0 >> 24); s[4] = (u8)(l0 >> 32); s[5] = (u8)(l0 >> 40);
    s[6] = (u8)((l0 >> 48) | (l1 << 3)); s[7] = (u8)(l1 >> 5); s[8] = (u8)(l1 >> 13); s[9] = (u8)(l1 >> 21); s[10] = (u8)(l1 >> 29); s[11] = (u8)(l1 >> 37);
    s[12] = (u8)((l1 >> 45) | (l2 << 6)); s[13] = (u8)(l2 >> 2); s[14] = (u8)(l2 >> 10); s[15] = (u8)(l2 >> 18); s[16] = (u8)(l2 >> 26); s[17] = (u8)(l2 >> 34); s[18] = (u8)(l2 >> 42);
    s[19] = (u8)((l2 >> 50) | (l3 << 1)); s[20] = (u8)(l3 >> 7); s[21] = (u8)(l3 >> 15); s[22] = (u8)(l3 >> 23); s[23] = (u8)(l3 >> 31); s[24] = (u8)(l3 >> 39);
    s[25] = (u8)((l3 >> 47) | (l4 << 4)); s[26] = (u8)(l4 >> 4); s[27] = (u8)(l4 >> 12); s[28] = (u8)(l4 >> 20); s[29] = (u8)(l4 >> 28); s[30] = (u8)(l4 >> 36); s[31] = (u8)(l4 >> 44);
}
static inline u64 load8(const u8 *p) { u64 r = 0; for (int i = 7; i >= 0; i--) r = (r << 8) | p[i]; return r; }
static void fe_frombytes(fe *h, const u8 s[32]) {
    h->v[0] = load8(s + 0) & MASK51;
    h->v[1] = (load8(s + 6) >> 3) & MASK51;
    h->v[2] = (load8(s + 12) >> 6) & MASK51;
    h->v[3] = (load8(s + 19) >> 1) & MASK51;
    h->v[4] = (load8(s + 24) >> 12) & MASK51;
}
static inline bool fe_isnegative(const fe *f) { u8 s[32]; fe_tobytes(s, f); return (s[0] & 1) != 0; }
static inline bool fe_iszero(const fe *f) { u8 s[32]; fe_tobytes(s, f); u8 o = 0; for (int i = 0; i < 32; i++) o |= s[i]; return o == 0; }
static inline bool fe_eq(const fe *f, const fe *g) { fe t; fe_sub(&t, f, g); return fe_iszero(&t); }
#endif

// derived constants and the key, written once by init_kernel / the host
typedef struct { fe sqrt_m1, d, d2; u8 scalar[32]; char wnaf[260]; int wnaf_len; int pad; } Consts;
typedef struct { fe sqrt_m1, d, d2; } Cfe;

// (x^(2^250-1), x^11)
static void fe_pow22501(fe *t19, fe *t3, const fe *x) {
    fe t0, t1, t2, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18;
    fe_sq(&t0, x);
    fe_sq(&t1, &t0); fe_sq(&t1, &t1);
    fe_mul(&t2, x, &t1);
    fe_mul(t3, &t0, &t2);
    fe_sq(&t4, t3);
    fe_mul(&t5, &t2, &t4);
    fe_pow2k(&t6, &t5, 5);
    fe_mul(&t7, &t6, &t5);
    fe_pow2k(&t8, &t7, 10);
    fe_mul(&t9, &t8, &t7);
    fe_pow2k(&t10, &t9, 20);
    fe_mul(&t11, &t10, &t9);
    fe_pow2k(&t12, &t11, 10);
    fe_mul(&t13, &t12, &t7);
    fe_pow2k(&t14, &t13, 50);
    fe_mul(&t15, &t14, &t13);
    fe_pow2k(&t16, &t15, 100);
    fe_mul(&t17, &t16, &t15);
    fe_pow2k(&t18, &t17, 50);
    fe_mul(t19, &t18, &t13);
}
static void fe_invert(fe *h, const fe *x) { fe t19, t3, t20; fe_pow22501(&t19, &t3, x); fe_pow2k(&t20, &t19, 5); fe_mul(h, &t20, &t3); }
static void fe_pow_p58(fe *h, const fe *x) { fe t19, t3, t20; fe_pow22501(&t19, &t3, x); fe_pow2k(&t20, &t19, 2); fe_mul(h, x, &t20); }

// r = sqrt(u/v) non-negative; returns whether u/v was a square
static bool fe_sqrt_ratio_i(fe *r, const fe *u, const fe *v, const Cfe *C) {
    fe v3, v7, t, uv3, uv7, check, nu, nui, rp;
    fe_sq(&v3, v); fe_mul(&v3, &v3, v);
    fe_sq(&v7, &v3); fe_mul(&v7, &v7, v);
    fe_mul(&uv3, u, &v3);
    fe_mul(&uv7, u, &v7);
    fe_pow_p58(&t, &uv7);
    fe_mul(r, &uv3, &t);
    fe_sq(&check, r); fe_mul(&check, &check, v);
    fe_neg(&nu, u);
    fe_mul(&nui, &nu, &C->sqrt_m1);
    bool correct = fe_eq(&check, u);
    bool flipped = fe_eq(&check, &nu);
    bool flipped_i = fe_eq(&check, &nui);
    fe_mul(&rp, r, &C->sqrt_m1);
    if (flipped || flipped_i) fe_copy(r, &rp);
    if (fe_isnegative(r)) fe_neg(r, r);
    return correct || flipped;
}

// ----------------------------------------------------------------------------
// points, extended coordinates (X:Y:Z:T), a = -1
// ----------------------------------------------------------------------------
typedef struct { fe X, Y, Z, T; } ge;
static inline void ge_identity(ge *p) { fe_0(&p->X); fe_1(&p->Y); fe_1(&p->Z); fe_0(&p->T); }
static inline void ge_copy(ge *r, const ge *p) { fe_copy(&r->X, &p->X); fe_copy(&r->Y, &p->Y); fe_copy(&r->Z, &p->Z); fe_copy(&r->T, &p->T); }
static bool ge_frombytes(ge *p, const u8 s[32], const Cfe *C) {
    fe Y, YY, u, v, X;
    fe_frombytes(&Y, s);
    fe_1(&p->Z);
    fe_sq(&YY, &Y);
    fe_sub(&u, &YY, &p->Z);
    fe_mul(&v, &YY, &C->d); fe_add(&v, &v, &p->Z);
    if (!fe_sqrt_ratio_i(&X, &u, &v, C)) return false;
    if (s[31] >> 7) fe_neg(&X, &X);
    fe_copy(&p->X, &X); fe_copy(&p->Y, &Y);
    fe_mul(&p->T, &X, &Y);
    return true;
}
static void ge_tobytes(u8 s[32], const ge *p) {
    fe zi, x, y;
    fe_invert(&zi, &p->Z);
    fe_mul(&x, &p->X, &zi); fe_mul(&y, &p->Y, &zi);
    fe_tobytes(s, &y);
    s[31] ^= (fe_isnegative(&x) ? 0x80 : 0);
}
static bool ge_is_identity(const ge *p) { return fe_iszero(&p->X) && fe_eq(&p->Y, &p->Z); }
static void ge_dbl(ge *r, const ge *p) {
    fe A, B, Cc, D, E, F, G, H, t;
    fe_sq(&A, &p->X); fe_sq(&B, &p->Y);
    fe_sq(&Cc, &p->Z); fe_add(&Cc, &Cc, &Cc);
    fe_neg(&D, &A);
    fe_add(&t, &p->X, &p->Y); fe_sq(&t, &t); fe_sub(&t, &t, &A); fe_sub(&E, &t, &B);
    fe_add(&G, &D, &B); fe_sub(&F, &G, &Cc); fe_sub(&H, &D, &B);
    fe_mul(&r->X, &E, &F); fe_mul(&r->Y, &G, &H); fe_mul(&r->T, &E, &H); fe_mul(&r->Z, &F, &G);
}
static void ge_dbl_noT(ge *r, const ge *p) {
    fe A, B, Cc, D, E, F, G, H, t;
    fe_sq(&A, &p->X); fe_sq(&B, &p->Y);
    fe_sq(&Cc, &p->Z); fe_add(&Cc, &Cc, &Cc);
    fe_neg(&D, &A);
    fe_add(&t, &p->X, &p->Y); fe_sq(&t, &t); fe_sub(&t, &t, &A); fe_sub(&E, &t, &B);
    fe_add(&G, &D, &B); fe_sub(&F, &G, &Cc); fe_sub(&H, &D, &B);
    fe_mul(&r->X, &E, &F); fe_mul(&r->Y, &G, &H); fe_mul(&r->Z, &F, &G);
}
static void ge_add(ge *r, const ge *p, const ge *q, const Cfe *C) {
    fe A, B, Cc, D, E, F, G, H, t1, t2;
    fe_sub(&t1, &p->Y, &p->X); fe_sub(&t2, &q->Y, &q->X); fe_mul(&A, &t1, &t2);
    fe_add(&t1, &p->Y, &p->X); fe_add(&t2, &q->Y, &q->X); fe_mul(&B, &t1, &t2);
    fe_mul(&Cc, &p->T, &C->d2); fe_mul(&Cc, &Cc, &q->T);
    fe_mul(&D, &p->Z, &q->Z); fe_add(&D, &D, &D);
    fe_sub(&E, &B, &A); fe_sub(&F, &D, &Cc); fe_add(&G, &D, &Cc); fe_add(&H, &B, &A);
    fe_mul(&r->X, &E, &F); fe_mul(&r->Y, &G, &H); fe_mul(&r->T, &E, &H); fe_mul(&r->Z, &F, &G);
}
static inline void ge_mul8(ge *r, const ge *p) { ge_dbl_noT(r, p); ge_dbl_noT(r, r); ge_dbl(r, r); }

typedef struct { fe YpX, YmX, T2d, Z2; } geCached;
static inline void ge_cache(geCached *c, const ge *p, const Cfe *C) {
    fe_add(&c->YpX, &p->Y, &p->X); fe_sub(&c->YmX, &p->Y, &p->X); fe_mul(&c->T2d, &p->T, &C->d2); fe_add(&c->Z2, &p->Z, &p->Z);
}
static void ge_add_cached(ge *r, const ge *p, const geCached *q, bool neg) {
    fe A, B, Cc, D, E, F, G, H, t1, t2;
    fe_sub(&t1, &p->Y, &p->X); fe_add(&t2, &p->Y, &p->X);
    fe_mul(&A, &t1, neg ? &q->YpX : &q->YmX);
    fe_mul(&B, &t2, neg ? &q->YmX : &q->YpX);
    fe_mul(&Cc, &p->T, &q->T2d); if (neg) fe_neg(&Cc, &Cc);
    fe_mul(&D, &p->Z, &q->Z2);
    fe_sub(&E, &B, &A); fe_sub(&F, &D, &Cc); fe_add(&G, &D, &Cc); fe_add(&H, &B, &A);
    fe_mul(&r->X, &E, &F); fe_mul(&r->Y, &G, &H); fe_mul(&r->T, &E, &H); fe_mul(&r->Z, &F, &G);
}

#if PX_WINDOW == 5
// width-5 wNAF over the constant scalar (digits uniform across the grid: no divergence)
static void ge_scalarmult_const(ge *r, const ge *p, const Cfe *C, __global const Consts *K) {
    geCached T[8];
    ge P2, acc;
    ge_dbl(&P2, p);
    ge_copy(&acc, p);
    ge_cache(&T[0], &acc, C);
    for (int i = 1; i < 8; i++) { ge_add(&acc, &acc, &P2, C); ge_cache(&T[i], &acc, C); }
    ge_identity(r);
    int len = K->wnaf_len;
    for (int i = len - 1; i >= 0; i--) {
        int d = K->wnaf[i];
        if (d) ge_dbl(r, r); else ge_dbl_noT(r, r);
        if (d > 0) ge_add_cached(r, r, &T[d >> 1], false);
        else if (d < 0) ge_add_cached(r, r, &T[(-d) >> 1], true);
    }
}
#elif PX_WINDOW == 0
static void ge_scalarmult_const(ge *r, const ge *p, const Cfe *C, __global const Consts *K) {
    ge_copy(r, p);
    for (int i = 253; i >= 0; i--) {
        if ((K->scalar[i >> 3] >> (i & 7)) & 1) { ge_dbl(r, r); ge_add(r, r, p, C); } else ge_dbl_noT(r, r);
    }
}
#else
static void ge_scalarmult_const(ge *r, const ge *p, const Cfe *C, __global const Consts *K) {
    ge T[16];
    ge_copy(&T[1], p);
    for (int i = 2; i < 16; i++) { if (i & 1) ge_add(&T[i], &T[i - 1], p, C); else ge_dbl(&T[i], &T[i >> 1]); }
    int w = (K->scalar[31] >> 4) & 0xF;
    ge_copy(r, &T[w]);
    for (int i = 62; i >= 0; i--) {
        w = (K->scalar[i >> 1] >> ((i & 1) * 4)) & 0xF;
        ge_dbl_noT(r, r); ge_dbl_noT(r, r); ge_dbl_noT(r, r);
        if (w) { ge_dbl(r, r); ge_add(r, r, &T[w], C); } else ge_dbl_noT(r, r);
    }
}
#endif
// generic scalar mult with a per-call scalar (key derivation)
static void ge_scalarmult(ge *r, const u8 *sc, const ge *p, const Cfe *C) {
    ge_identity(r);
    for (int i = 255; i >= 0; i--) { ge_dbl(r, r); if ((sc[i >> 3] >> (i & 7)) & 1) ge_add(r, r, p, C); }
}

// ----------------------------------------------------------------------------
// SHA-512, single block (message <= 111 bytes), rolling 16-word schedule
// ----------------------------------------------------------------------------
__constant u64 K512[80] = {
    0x428a2f98d728ae22UL, 0x7137449123ef65cdUL, 0xb5c0fbcfec4d3b2fUL, 0xe9b5dba58189dbbcUL, 0x3956c25bf348b538UL, 0x59f111f1b605d019UL, 0x923f82a4af194f9bUL, 0xab1c5ed5da6d8118UL,
    0xd807aa98a3030242UL, 0x12835b0145706fbeUL, 0x243185be4ee4b28cUL, 0x550c7dc3d5ffb4e2UL, 0x72be5d74f27b896fUL, 0x80deb1fe3b1696b1UL, 0x9bdc06a725c71235UL, 0xc19bf174cf692694UL,
    0xe49b69c19ef14ad2UL, 0xefbe4786384f25e3UL, 0x0fc19dc68b8cd5b5UL, 0x240ca1cc77ac9c65UL, 0x2de92c6f592b0275UL, 0x4a7484aa6ea6e483UL, 0x5cb0a9dcbd41fbd4UL, 0x76f988da831153b5UL,
    0x983e5152ee66dfabUL, 0xa831c66d2db43210UL, 0xb00327c898fb213fUL, 0xbf597fc7beef0ee4UL, 0xc6e00bf33da88fc2UL, 0xd5a79147930aa725UL, 0x06ca6351e003826fUL, 0x142929670a0e6e70UL,
    0x27b70a8546d22ffcUL, 0x2e1b21385c26c926UL, 0x4d2c6dfc5ac42aedUL, 0x53380d139d95b3dfUL, 0x650a73548baf63deUL, 0x766a0abb3c77b2a8UL, 0x81c2c92e47edaee6UL, 0x92722c851482353bUL,
    0xa2bfe8a14cf10364UL, 0xa81a664bbc423001UL, 0xc24b8b70d0f89791UL, 0xc76c51a30654be30UL, 0xd192e819d6ef5218UL, 0xd69906245565a910UL, 0xf40e35855771202aUL, 0x106aa07032bbd1b8UL,
    0x19a4c116b8d2d0c8UL, 0x1e376c085141ab53UL, 0x2748774cdf8eeb99UL, 0x34b0bcb5e19b48a8UL, 0x391c0cb3c5c95a63UL, 0x4ed8aa4ae3418acbUL, 0x5b9cca4f7763e373UL, 0x682e6ff3d6b2b8a3UL,
    0x748f82ee5defb2fcUL, 0x78a5636f43172f60UL, 0x84c87814a1f0ab72UL, 0x8cc702081a6439ecUL, 0x90befffa23631e28UL, 0xa4506cebde82bde9UL, 0xbef9a3f7b2c67915UL, 0xc67178f2e372532bUL,
    0xca273eceea26619cUL, 0xd186b8c721c0c207UL, 0xeada7dd6cde0eb1eUL, 0xf57d4f7fee6ed178UL, 0x06f067aa72176fbaUL, 0x0a637dc5a2c898a6UL, 0x113f9804bef90daeUL, 0x1b710b35131c471bUL,
    0x28db77f523047d84UL, 0x32caab7b40c72493UL, 0x3c9ebe0a15c9bebcUL, 0x431d67c49c100d4cUL, 0x4cc5d4becb3e42b6UL, 0x597f299cfc657e2aUL, 0x5fcb6fab3ad6faecUL, 0x6c44198c4a475817UL
};
static inline u64 rotr64(u64 x, int n) { return (x >> n) | (x << (64 - n)); }
static void sha512_single(const u8 *msg, int len, u8 out[64]) {
    u64 W[16];
    for (int i = 0; i < 16; i++) {
        u64 w = 0;
        for (int j = 0; j < 8; j++) { int k = i * 8 + j; u8 b = k < len ? msg[k] : (k == len ? (u8)0x80 : (u8)0); w = (w << 8) | b; }
        W[i] = w;
    }
    W[15] = (u64)len * 8;
    u64 a = 0x6a09e667f3bcc908UL, b = 0xbb67ae8584caa73bUL, c = 0x3c6ef372fe94f82bUL, d = 0xa54ff53a5f1d36f1UL;
    u64 e = 0x510e527fade682d1UL, f = 0x9b05688c2b3e6c1fUL, g = 0x1f83d9abfb41bd6bUL, h = 0x5be0cd19137e2179UL;
    #pragma unroll
    for (int i = 0; i < 80; i++) {
        u64 wi;
        if (i < 16) wi = W[i];
        else {
            u64 w15 = W[(i - 15) & 15], w2 = W[(i - 2) & 15];
            u64 s0 = rotr64(w15, 1) ^ rotr64(w15, 8) ^ (w15 >> 7);
            u64 s1 = rotr64(w2, 19) ^ rotr64(w2, 61) ^ (w2 >> 6);
            wi = W[i & 15] + s0 + W[(i - 7) & 15] + s1;
            W[i & 15] = wi;
        }
        u64 S1 = rotr64(e, 14) ^ rotr64(e, 18) ^ rotr64(e, 41);
        u64 ch = (e & f) ^ (~e & g);
        u64 t1 = h + S1 + ch + K512[i] + wi;
        u64 S0 = rotr64(a, 28) ^ rotr64(a, 34) ^ rotr64(a, 39);
        u64 mj = (a & b) ^ (a & c) ^ (b & c);
        u64 t2 = S0 + mj;
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    u64 st[8] = { a + 0x6a09e667f3bcc908UL, b + 0xbb67ae8584caa73bUL, c + 0x3c6ef372fe94f82bUL, d + 0xa54ff53a5f1d36f1UL,
                  e + 0x510e527fade682d1UL, f + 0x9b05688c2b3e6c1fUL, g + 0x1f83d9abfb41bd6bUL, h + 0x5be0cd19137e2179UL };
    for (int i = 0; i < 8; i++) for (int j = 0; j < 8; j++) out[i * 8 + j] = (u8)(st[i] >> (56 - 8 * j));
}

// ----------------------------------------------------------------------------
// the VRF output for one nonce
// ----------------------------------------------------------------------------
static inline void load_cfe(Cfe *C, __global const Consts *K) { C->sqrt_m1 = K->sqrt_m1; C->d = K->d; C->d2 = K->d2; }
static inline int trailing_zero_bits(const u8 beta[64]) {
    int n = 0;
    for (int i = 63; i >= 0; i--) {
        u32 b = beta[i];
        if (b == 0) { n += 8; continue; }
        return n + (31 - (int)clz(b & (0u - b)));
    }
    return n;
}
static inline bool below(const u8 a[64], __global const u8 *b) {
    for (int i = 0; i < 64; i++) { if (a[i] < b[i]) return true; if (a[i] > b[i]) return false; }
    return false;
}
static bool vrf_beta(__global const JobDev *job, u64 nonce, u8 beta[64], u8 *dbgH, u8 *dbgCG, const Cfe *C, __global const Consts *K) {
    u8 msg[128];
    int n = 0;
    msg[n++] = 0x03; msg[n++] = 0x01;
    for (int i = 0; i < 32; i++) msg[n++] = job->pk[i];
    int pl = job->prefix_len;
    for (int i = 0; i < pl; i++) msg[n++] = job->prefix[i];
    for (int i = 0; i < 8; i++) msg[n++] = (u8)(nonce >> (56 - 8 * i));
    int ctrpos = n; n += 2;
    ge H;
    bool found = false;
    for (int ctr = 0; ctr < 256 && !found; ctr++) {
        msg[ctrpos] = (u8)ctr; msg[ctrpos + 1] = 0;
        u8 dg[64];
        sha512_single(msg, n, dg);
        if (!ge_frombytes(&H, dg, C)) continue;
        ge_mul8(&H, &H);
        if (ge_is_identity(&H)) continue;
        found = true;
    }
    if (!found) return false;
    if (dbgH) ge_tobytes(dbgH, &H);
    ge G;
    ge_scalarmult_const(&G, &H, C, K);
    ge_mul8(&G, &G);
    u8 m2[35];
    m2[0] = 0x03; m2[1] = 0x03;
    ge_tobytes(m2 + 2, &G);
    m2[34] = 0x00;
    if (dbgCG) for (int i = 0; i < 32; i++) dbgCG[i] = m2[2 + i];
    sha512_single(m2, 35, beta);
    return true;
}

// ----------------------------------------------------------------------------
// kernels
// ----------------------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size(PX_WG, 1, 1)))
void mine_kernel(__global const JobDev *job, u64 nonce_base, u64 stride, int iters, __global Hit *hits, __global uint *nhits, __global const Consts *K) {
    Cfe C; load_cfe(&C, K);
    u64 nonce = nonce_base + get_global_id(0);
    for (int it = 0; it < iters; it++, nonce += stride) {
        u8 beta[64];
        if (!vrf_beta(job, nonce, beta, 0, 0, &C, K)) continue;
        if (trailing_zero_bits(beta) >= job->k && below(beta, job->beat)) {
            uint idx = atomic_inc(nhits);
            if (idx < MAX_HITS) { hits[idx].nonce = nonce; for (int i = 0; i < 64; i++) hits[idx].beta[i] = beta[i]; }
        }
    }
}

// compacted pipeline: one decompression per item per pass, failures queued for the next pass;
// then the scalar multiplications over the found points with no divergence
typedef struct { fe X, Y; u64 nonce; } HPoint;
#define PX_MAX_PASSES 12
__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
// cnt[0] = points found, cnt[1..PX_MAX_PASSES] = retries queued per pass, cnt[PX_MAX_PASSES+1] = N;
// a pass reads its input count at cnt[in_idx] (first pass: N) and queues failures under cnt[out_idx]
void h2c_kernel(__global const JobDev *job, u64 nonce_base, __global const u64 *in_nonce, __global const u8 *in_ctr, __global uint *cnt, int in_idx, int out_idx, int first_pass,
                __global HPoint *pts, __global u64 *out_nonce, __global u8 *out_ctr, __global const Consts *K) {
    uint n_in = cnt[in_idx];
    __global uint *n_pts = cnt; __global uint *n_out = cnt + out_idx;
    uint gid = get_global_id(0), total = get_global_size(0);
    if (gid >= n_in) return;
    Cfe C; load_cfe(&C, K);
    u8 msg[128];
    int n = 0;
    msg[n++] = 0x03; msg[n++] = 0x01;
    for (int k = 0; k < 32; k++) msg[n++] = job->pk[k];
    int pl = job->prefix_len;
    for (int k = 0; k < pl; k++) msg[n++] = job->prefix[k];
    int npos = n; n += 10;
    for (uint i = gid; i < n_in; i += total) {
        u64 nonce = first_pass ? nonce_base + i : in_nonce[i];
        u8 ctr = first_pass ? (u8)0 : in_ctr[i];
        for (int k = 0; k < 8; k++) msg[npos + k] = (u8)(nonce >> (56 - 8 * k));
        msg[npos + 8] = ctr; msg[npos + 9] = 0;
        u8 dg[64];
        sha512_single(msg, n, dg);
        ge H;
        if (ge_frombytes(&H, dg, &C)) {
            uint o = atomic_inc(n_pts);
            pts[o].X = H.X; pts[o].Y = H.Y; pts[o].nonce = nonce;
        } else if (ctr < 255) {
            uint o = atomic_inc(n_out);
            out_nonce[o] = nonce; out_ctr[o] = ctr + 1;
        }
    }
}
static inline bool mult_point(ge *G, __global const HPoint *pt, const Cfe *C, __global const Consts *K) {
    ge H;
    H.X = pt->X; H.Y = pt->Y; fe_1(&H.Z);
    ge_mul8(&H, &H);
    if (ge_is_identity(&H)) return false;
    ge_scalarmult_const(G, &H, C, K);
    ge_mul8(G, G);
    return true;
}
__kernel __attribute__((reqd_work_group_size(PX_WG, 1, 1)))
void mult_kernel(__global const JobDev *job, __global const HPoint *pts, __global const uint *cnt, __global Hit *hits, __global uint *nhits, __global const Consts *K) {
    uint n = cnt[0];
    uint gid = get_global_id(0), total = get_global_size(0);
    Cfe C; load_cfe(&C, K);
    for (uint i = gid; i < n; i += total) {
        ge G;
        if (!mult_point(&G, &pts[i], &C, K)) continue;
        fe zi, x, y;
        fe_invert(&zi, &G.Z);
        fe_mul(&x, &G.X, &zi); fe_mul(&y, &G.Y, &zi);
        u8 m2[35];
        m2[0] = 0x03; m2[1] = 0x03;
        fe_tobytes(m2 + 2, &y);
        m2[33] ^= (fe_isnegative(&x) ? 0x80 : 0);
        m2[34] = 0x00;
        u8 beta[64];
        sha512_single(m2, 35, beta);
        if (trailing_zero_bits(beta) >= job->k && below(beta, job->beat)) {
            uint idx = atomic_inc(nhits);
            if (idx < MAX_HITS) { hits[idx].nonce = pts[i].nonce; for (int k = 0; k < 64; k++) hits[idx].beta[k] = beta[k]; }
        }
    }
}

// self-test: beta for a complete alpha plus intermediates
__kernel void test_kernel(__global const JobDev *job, u64 nonce, __global u8 *out_beta, __global u8 *out_h, __global u8 *out_cg, __global int *ok, __global const Consts *K) {
    Cfe C; load_cfe(&C, K);
    u8 beta[64], hh[32], cg[32];
    int r = vrf_beta(job, nonce, beta, hh, cg, &C, K) ? 1 : 0;
    for (int i = 0; i < 64; i++) out_beta[i] = beta[i];
    for (int i = 0; i < 32; i++) { out_h[i] = hh[i]; out_cg[i] = cg[i]; }
    *ok = r;
}

// randomized field-arithmetic test
__kernel void fe_test_kernel(uint seed, int n, __global int *fail, __global u8 *dump, __global const Consts *K) {
    u64 st = (u64)seed * 0x9E3779B97F4A7C15UL + 1;
    for (int it = 0; it < n; it++) {
        u8 xb[32], yb[32];
        for (int i = 0; i < 32; i++) { st = st * 6364136223846793005UL + 1442695040888963407UL; xb[i] = (u8)(st >> 56); st = st * 6364136223846793005UL + 1442695040888963407UL; yb[i] = (u8)(st >> 56); }
        fe x, y, s1, s2, t, u, v, w;
        fe_frombytes(&x, xb); fe_frombytes(&y, yb);
        if (it & 1) { fe_add(&x, &x, &x); fe_add(&y, &y, &x); }
        fe_sq(&s1, &x); fe_mul(&s2, &x, &x);
        int code = 0;
        if (!fe_eq(&s1, &s2)) code = 1;
        fe_add(&t, &x, &y); fe_sq(&t, &t);
        fe_mul(&u, &x, &y); fe_add(&u, &u, &u); fe_sq(&v, &y); fe_add(&u, &u, &v); fe_add(&u, &u, &s2);
        if (!code && !fe_eq(&t, &u)) code = 2;
        fe_sub(&w, &x, &y); fe_add(&t, &x, &y); fe_mul(&w, &w, &t);
        fe_sub(&v, &s2, &v);
        if (!code && !fe_eq(&w, &v)) code = 3;
        fe_copy(&t, &x); fe_sq(&t, &t);
        if (!code && !fe_eq(&t, &s2)) code = 4;
        // encoding: the top bit of the last byte is always clear, isnegative is the low bit,
        // decoding the bytes gives the value back (for x < p), and a product's encoding is canonical
        { u8 b1[32]; fe_tobytes(b1, &s2); if (!code && (b1[31] & 0x80)) code = 9;
          if (!code && fe_isnegative(&s2) != ((b1[0] & 1) != 0)) code = 10;
          fe f2; fe_frombytes(&f2, b1); if (!code && !fe_eq(&f2, &s2)) code = 11;
          u8 b2[32]; fe_tobytes(b2, &f2); int diff = 0; for (int i = 0; i < 32; i++) diff |= b1[i] ^ b2[i]; if (!code && diff) code = 12; }
        if (code) { *fail = code * 1000000 + it; for (int i = 0; i < 32; i++) { dump[i] = xb[i]; dump[32 + i] = yb[i]; } u8 b1[32], b2[32]; fe_tobytes(b1, &s1); fe_tobytes(b2, &s2); for (int i = 0; i < 32; i++) { dump[64 + i] = b1[i]; dump[96 + i] = b2[i]; } return; }
    }
    // inversion sanity
    fe xx; fe_0(&xx); xx.v[0] = 121666;
    fe inv, one, o; fe_invert(&inv, &xx); fe_mul(&one, &inv, &xx); fe_1(&o);
    if (!fe_eq(&one, &o)) { *fail = 7 * 1000000; return; }
    *fail = 0;
}

// constants: d = -121665/121666, sqrt(-1) = 2^((p-1)/4), base point B (y = 4/5); pk = scalar*B
__kernel void init_kernel(__global const u8 *scalar, __global u8 *pk_out, __global int *ok, __global Consts *K) {
    fe a, b, t;
    fe_0(&a); a.v[0] = 121665; fe_neg(&a, &a);
    fe_0(&b); b.v[0] = 121666; fe_invert(&t, &b);
    fe d; fe_mul(&d, &a, &t);
    fe d2; fe_add(&d2, &d, &d); fe_reduce(&d2);
    { fe s1, s2; fe_sq(&s1, &d); fe_mul(&s2, &d, &d); if (!fe_eq(&s1, &s2)) { *ok = -2; return; } }
    fe r; fe_0(&r); r.v[0] = 2;
    fe acc; fe_1(&acc);
    for (int i = 253; i >= 0; i--) {
        fe_sq(&acc, &acc);
        int bit = (i >= 3 && i <= 252) || i == 1 || i == 0;
        if (bit) fe_mul(&acc, &acc, &r);
    }
    fe m1; fe_1(&m1); fe_neg(&m1, &m1);
    fe chk; fe_sq(&chk, &acc);
    if (!fe_eq(&chk, &m1)) { *ok = 0; return; }
    K->sqrt_m1 = acc; K->d = d; K->d2 = d2;
    Cfe C; C.sqrt_m1 = acc; C.d = d; C.d2 = d2;
    fe four, five, y; fe_0(&four); four.v[0] = 4; fe_0(&five); five.v[0] = 5;
    fe_invert(&t, &five); fe_mul(&y, &four, &t);
    u8 yb[32]; fe_tobytes(yb, &y);
    ge B;
    if (!ge_frombytes(&B, yb, &C)) { *ok = 0; return; }
    u8 sc[32]; for (int i = 0; i < 32; i++) sc[i] = scalar[i];
    ge P; ge_scalarmult(&P, sc, &B, &C);
    u8 pk[32]; ge_tobytes(pk, &P);
    for (int i = 0; i < 32; i++) pk_out[i] = pk[i];
    *ok = 1;
}
