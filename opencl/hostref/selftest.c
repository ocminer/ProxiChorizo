// Host-side reference run of the complete VRF attempt (test_kernel) over vectors.json: the OpenCL
// source compiled as C (see shim.h). Fills the constants the way the OpenCL host does (scalar +
// width-5 wNAF digits) and compares H, 8*Gamma and beta with every vector.
#include "shim.h"
#include "proxichorizo.cl"   // found via -I.. (the real kernel source, unmodified)
static int unhex(const char *h, int n, u8 *out) { for (int i = 0; i < n; i++) { unsigned v; if (sscanf(h + 2 * i, "%2x", &v) != 1) return 0; out[i] = (u8)v; } return 1; }
static const char *field(const char *line, const char *key) { char k[64]; snprintf(k, sizeof k, "\"%s\":\"", key); const char *p = strstr(line, k); return p ? p + strlen(k) : NULL; }
static void prepare_job(JobDev *jd) {
    u8 m[72]; memset(m, 0, sizeof m); m[0] = 0x03; m[1] = 0x01; memcpy(m + 2, jd->pk, 32); memcpy(m + 34, jd->prefix, 37);
    for (int i = 0; i < 9; i++) { u64 w = 0; for (int j = 0; j < 8; j++) w = (w << 8) | (u64)(i * 8 + j < 71 ? m[i * 8 + j] : 0); jd->mw[i] = w; }
    for (int i = 0; i < 8; i++) { u64 w = 0; for (int j = 0; j < 8; j++) w = (w << 8) | jd->beat[8 * i + j]; jd->beat_w[i] = w; }
}
// same digit computation as set_scalar() in proxichorizo_cl.cpp
static void set_scalar(Consts *K, const u8 *scalar) {
    u8 k[40]; memset(k, 0, sizeof k); memcpy(k, scalar, 32);
    memset(K->wnaf, 0, sizeof K->wnaf); int len = 0;
    for (int i = 0; i < 258; i++) {
        int zero = 1; for (int j = 0; j < 40; j++) if (k[j]) { zero = 0; break; }
        if (zero) break;
        int d = 0;
        if (k[0] & 1) {
            d = k[0] & 31; if (d >= 16) d -= 32;
            // k -= d
            if (d < 0) { unsigned c = -d; for (int j = 0; j < 40 && c; j++) { unsigned t = k[j] + c; k[j] = t & 0xff; c = t >> 8; } }
            else { unsigned b = d; for (int j = 0; j < 40 && b; j++) { int t = (int)k[j] - (int)b; if (t < 0) { k[j] = (u8)(t + 256); b = 1; } else { k[j] = (u8)t; b = 0; } } }
        }
        K->wnaf[i] = (char)d; if (d) len = i + 1;
        for (int j = 0; j < 39; j++) k[j] = (k[j] >> 1) | (k[j + 1] << 7);
        k[39] >>= 1;
    }
    memcpy(K->scalar, scalar, 32); K->wnaf_len = len;
}
int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "../../vectors.json";
    FILE *f = fopen(path, "r"); if (!f) { perror(path); return 1; }
    char line[4096]; int n = 0, bad = 0; u8 cur_x[32] = {0}; int have_x = 0; Consts K; memset(&K, 0, sizeof K);
    while (fgets(line, sizeof line, f)) {
        const char *x = field(line, "x"), *pk = field(line, "pk"), *al = field(line, "alpha"), *h = field(line, "h"), *cg = field(line, "cg"), *beta = field(line, "beta");
        if (!x || !pk || !al || !h || !cg || !beta) continue;
        u8 xb[32], pkb[32], ab[45], hb[32], cgb[32], bb[64];
        if (!unhex(x, 32, xb) || !unhex(pk, 32, pkb) || !unhex(al, 45, ab) || !unhex(h, 32, hb) || !unhex(cg, 32, cgb) || !unhex(beta, 64, bb)) { printf("bad vector line\n"); bad++; continue; }
        if (!have_x || memcmp(xb, cur_x, 32)) {
            set_scalar(&K, xb);
            u8 pk2[32]; int ok = 0; init_kernel(xb, pk2, &ok, &K);
            if (ok != 1 || memcmp(pk2, pkb, 32)) { printf("constant derivation failed: ok=%d, pk %s\n", ok, memcmp(pk2, pkb, 32) ? "MISMATCH" : "ok"); return 1; }
            memcpy(cur_x, xb, 32); have_x = 1;
        }
        JobDev jd; memset(&jd, 0, sizeof jd); memcpy(jd.pk, pkb, 32); memcpy(jd.prefix, ab, 37); jd.prefix_len = 37; prepare_job(&jd);
        u64 nonce = 0; for (int i = 0; i < 8; i++) nonce = (nonce << 8) | ab[37 + i];
        u8 ob[64], oh[32], ocg[32]; int ok = 0;
        test_kernel(&jd, nonce, ob, oh, ocg, &ok, &K);
        n++;
        if (ok != 1 || memcmp(oh, hb, 32) || memcmp(ocg, cgb, 32) || memcmp(ob, bb, 64)) { bad++; printf("vector %d FAILED: ok=%d H %s 8G %s beta %s\n", n, ok, memcmp(oh, hb, 32) ? "MISMATCH" : "ok", memcmp(ocg, cgb, 32) ? "MISMATCH" : "ok", memcmp(ob, bb, 64) ? "MISMATCH" : "ok"); }
    }
    printf("host selftest PX_FE=%d: %d vectors, %d failed\n", PX_FE, n, bad);
    return bad != 0;
}
