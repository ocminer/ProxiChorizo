// proxichorizo_cl.cpp — ProxiChorizo for OpenCL GPUs (AMD first). The host side that does not
// touch the GPU (sockets, HTTP, job poller, submit, stats API, key loading) is shared with the
// CUDA miner through host_common.inc (generated from proxichorizo.cu); the device code is
// proxichorizo.cl, embedded as kernel_blob.h and compiled per GPU and variant at start-up.
#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <random>
#include <fstream>
#include <sstream>
#include <map>

#include "host_common.inc"
#include "kernel_blob.h"

// must match proxichorizo.cl
struct JobDev { uint8_t pk[32]; uint8_t prefix[64]; int prefix_len; int k; uint8_t beat[64]; uint64_t mw[9]; uint64_t beat_w[8]; };
// the kernels hash on 64-bit words: precompute the fixed part of the hash-to-curve message
// (03 01 pk prefix = 71 bytes -> words 0..7 and the top 56 bits of word 8) and the beat as words
static void prepare_job(JobDev &jd) {
    uint8_t m[72]; memset(m, 0, sizeof m); m[0] = 0x03; m[1] = 0x01; memcpy(m + 2, jd.pk, 32); memcpy(m + 34, jd.prefix, 37);
    for (int i = 0; i < 9; i++) { uint64_t w = 0; for (int j = 0; j < 8; j++) w = (w << 8) | (uint64_t)(i * 8 + j < 71 ? m[i * 8 + j] : 0); jd.mw[i] = w; }
    for (int i = 0; i < 8; i++) { uint64_t w = 0; for (int j = 0; j < 8; j++) w = (w << 8) | jd.beat[8 * i + j]; jd.beat_w[i] = w; }
}
struct Hit { uint64_t nonce; uint8_t beta[64]; };
#define MAX_HITS 256
#define PX_MAX_PASSES 12

#define CL_CHECK(x) do { cl_int e_ = (x); if (e_ != CL_SUCCESS) { fprintf(stderr, "OpenCL error %d at %s:%d\n", (int)e_, __FILE__, __LINE__); exit(2); } } while (0)

// ---- devices -------------------------------------------------------------------
struct ClDev {
    cl_platform_id plat; cl_device_id id; cl_context ctx; cl_command_queue q;
    std::string name, board, platname; cl_uint cus; int bus = 0; size_t maxwg = 256;
};
static std::vector<ClDev> g_devs;
static void enumerate_devices() {
    cl_uint np = 0; clGetPlatformIDs(0, nullptr, &np);
    std::vector<cl_platform_id> plats(np); if (np) clGetPlatformIDs(np, plats.data(), nullptr);
    for (auto p : plats) {
        char pn[256] = {0}; clGetPlatformInfo(p, CL_PLATFORM_NAME, sizeof pn, pn, nullptr);
        cl_uint nd = 0; if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd) != CL_SUCCESS || nd == 0) continue;
        std::vector<cl_device_id> ds(nd); clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, nd, ds.data(), nullptr);
        for (auto d : ds) {
            ClDev c; c.plat = p; c.id = d; c.platname = pn;
            char nm[256] = {0}; clGetDeviceInfo(d, CL_DEVICE_NAME, sizeof nm, nm, nullptr); c.name = nm;
            char bd[256] = {0}; if (clGetDeviceInfo(d, 0x4038 /*CL_DEVICE_BOARD_NAME_AMD*/, sizeof bd, bd, nullptr) == CL_SUCCESS) c.board = bd;
            clGetDeviceInfo(d, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof c.cus, &c.cus, nullptr);
            clGetDeviceInfo(d, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof c.maxwg, &c.maxwg, nullptr);
            struct { cl_uint type; char unused[17]; char bus, device, function; } topo; memset(&topo, 0, sizeof topo);
            if (clGetDeviceInfo(d, 0x4037 /*CL_DEVICE_TOPOLOGY_AMD*/, sizeof topo, &topo, nullptr) == CL_SUCCESS) c.bus = (unsigned char)topo.bus;
            cl_int err; c.ctx = clCreateContext(nullptr, 1, &d, nullptr, nullptr, &err); CL_CHECK(err);
            c.q = clCreateCommandQueue(c.ctx, d, 0, &err); CL_CHECK(err);
            g_devs.push_back(c);
        }
    }
}
static std::string dev_label(const ClDev &d) { return d.board.empty() ? d.name : d.board + " (" + d.name + ")"; }

// ---- variants: one kernel source, several build configurations --------------------
struct Variant { const char *name; int fe, window, compact, wg; };
static const Variant VARIANTS[] = {
    { "fe29_wnaf_c", 29, 5, 1, 256 },
    { "fe29_w4_c",   29, 4, 1, 256 },
    { "fe29_w0_c",   29, 0, 1, 256 },
    { "fe32_wnaf_c", 32, 5, 1, 256 },
    { "fe32_w4_c",   32, 4, 1, 256 },
    { "fe51_wnaf_c", 51, 5, 1, 256 },
    { "fe29_wnaf",   29, 5, 0, 256 },
    { "fe32_wnaf",   32, 5, 0, 256 },
};
static const int NVARIANTS = (int)(sizeof VARIANTS / sizeof VARIANTS[0]);
static const Variant *variantByName(const std::string &n) { for (int i = 0; i < NVARIANTS; i++) if (n == VARIANTS[i].name) return &VARIANTS[i]; return nullptr; }
struct Platform { const char *prefix; const char *variant; int iters; const char *note; };
static const Platform PLATFORMS[] = {
    { "gfx11", "fe29_wnaf_c", 8, "RDNA3: 9x29 limbs straight into v_mad_u64_u32, wNAF, compacted pipeline" },
    { "gfx12", "fe29_wnaf_c", 8, "RDNA4: as RDNA3" },
    { "gfx10", "fe29_wnaf_c", 8, "RDNA2/1" },
    { "gfx9",  "fe29_wnaf_c", 8, "Vega/CDNA" },
    { "",      "fe29_wnaf_c", 8, "default; run --autotune" },
};
static const Platform &platformFor(const ClDev &d) { for (auto &p : PLATFORMS) if (d.name.rfind(p.prefix, 0) == 0) return p; return PLATFORMS[sizeof PLATFORMS / sizeof PLATFORMS[0] - 1]; }

static std::string g_kernel_src;   // from the blob or --kernel-file
static std::string g_extra_build;  // --build-opts

// a built variant on one device
struct Prog {
    const Variant *v = nullptr; ClDev *dev = nullptr;
    cl_program p = nullptr; cl_kernel k_mine = nullptr, k_h2c = nullptr, k_mult = nullptr, k_test = nullptr, k_fetest = nullptr, k_init = nullptr;
    cl_mem consts = nullptr;  // Consts struct: 3 fe + scalar[32] + wnaf[260] + int len + pad
    size_t fe_size = 32, consts_size = 0;
};
static std::map<std::pair<cl_device_id, const Variant *>, Prog *> g_progs;
static std::mutex g_prog_mu;
static Prog *get_prog(ClDev &dev, const Variant *v) {
    std::lock_guard<std::mutex> l(g_prog_mu);
    auto key = std::make_pair(dev.id, v);
    auto it = g_progs.find(key); if (it != g_progs.end()) return it->second;
    Prog *pr = new Prog(); pr->v = v; pr->dev = &dev;
    const char *src = g_kernel_src.c_str(); size_t len = g_kernel_src.size(); cl_int err;
    pr->p = clCreateProgramWithSource(dev.ctx, 1, &src, &len, &err); CL_CHECK(err);
    std::string opts = "-DPX_FE=" + std::to_string(v->fe) + " -DPX_WINDOW=" + std::to_string(v->window) + " -DPX_COMPACT=" + std::to_string(v->compact) + " -DPX_WG=" + std::to_string(v->wg) + " " + g_extra_build;
    auto t0 = std::chrono::steady_clock::now();
    err = clBuildProgram(pr->p, 1, &dev.id, opts.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t ll = 0; clGetProgramBuildInfo(pr->p, dev.id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &ll);
        std::string log(ll, 0); clGetProgramBuildInfo(pr->p, dev.id, CL_PROGRAM_BUILD_LOG, ll, log.data(), nullptr);
        fprintf(stderr, "[%s] build failed on %s with '%s':\n%s\n", v->name, dev.name.c_str(), opts.c_str(), log.c_str()); exit(2);
    }
    double bs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    pr->k_mine = clCreateKernel(pr->p, "mine_kernel", &err); CL_CHECK(err);
    pr->k_h2c = clCreateKernel(pr->p, "h2c_kernel", &err); CL_CHECK(err);
    pr->k_mult = clCreateKernel(pr->p, "mult_kernel", &err); CL_CHECK(err);
    pr->k_test = clCreateKernel(pr->p, "test_kernel", &err); CL_CHECK(err);
    pr->k_fetest = clCreateKernel(pr->p, "fe_test_kernel", &err); CL_CHECK(err);
    pr->k_init = clCreateKernel(pr->p, "init_kernel", &err); CL_CHECK(err);
    pr->fe_size = v->fe == 32 ? 32 : v->fe == 51 ? 40 : 36;
    pr->consts_size = 3 * pr->fe_size + 32 + 260 + 8;
    pr->consts = clCreateBuffer(dev.ctx, CL_MEM_READ_WRITE, pr->consts_size, nullptr, &err); CL_CHECK(err);
    logln("[" + std::string(v->name) + "] built for " + dev.name + " in " + std::to_string((int)(bs * 1000)) + " ms");
    g_progs[key] = pr;
    return pr;
}

// a variant is trusted on a device only after it reproduced the Go test vectors there: the
// kernel is compiled by the installed driver, and one driver version miscompiled a layout
static std::map<Prog *, bool> g_validated;
static int run_selftest(Prog *pr, const std::string &path, bool quiet);
static bool prog_validated(Prog *pr) {
    auto it = g_validated.find(pr); if (it != g_validated.end()) return it->second;
    bool ok = run_selftest(pr, "embedded", true) == 0;
    if (!ok) fprintf(stderr, "[%s] does not reproduce the test vectors on %s with this driver: variant disabled\n", pr->v->name, pr->dev->name.c_str());
    g_validated[pr] = ok; return ok;
}

// width-5 wNAF of the little-endian scalar (same as the CUDA v_set_scalar) into the Consts buffer
static void set_scalar(Prog *pr, const uint8_t *scalar) {
    uint8_t k[40]; memset(k, 0, sizeof k); memcpy(k, scalar, 32);
    int8_t digits[260]; memset(digits, 0, sizeof digits);
    int len = 0;
    auto isZero = [&]() { for (int i = 0; i < 40; i++) if (k[i]) return false; return true; };
    auto shr1 = [&]() { for (int i = 0; i < 39; i++) k[i] = (k[i] >> 1) | (k[i + 1] << 7); k[39] >>= 1; };
    auto addSmall = [&](int v) {
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
    size_t off = 3 * pr->fe_size;
    CL_CHECK(clEnqueueWriteBuffer(pr->dev->q, pr->consts, CL_TRUE, off, 32, scalar, 0, nullptr, nullptr));
    CL_CHECK(clEnqueueWriteBuffer(pr->dev->q, pr->consts, CL_TRUE, off + 32, 260, digits, 0, nullptr, nullptr));
    CL_CHECK(clEnqueueWriteBuffer(pr->dev->q, pr->consts, CL_TRUE, off + 32 + 260, 4, &len, 0, nullptr, nullptr));
}
// derive the constants and the public key on the device; 1 = ok, 0 = sqrt(-1)/basepoint check failed, -2 = sq != mul
static int prog_init(Prog *pr, const uint8_t *scalar, uint8_t pk_out[32]) {
    set_scalar(pr, scalar);
    cl_int err; ClDev &d = *pr->dev;
    cl_mem b_sc = clCreateBuffer(d.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 32, (void *)scalar, &err); CL_CHECK(err);
    cl_mem b_pk = clCreateBuffer(d.ctx, CL_MEM_WRITE_ONLY, 32, nullptr, &err); CL_CHECK(err);
    cl_mem b_ok = clCreateBuffer(d.ctx, CL_MEM_WRITE_ONLY, 4, nullptr, &err); CL_CHECK(err);
    clSetKernelArg(pr->k_init, 0, sizeof b_sc, &b_sc); clSetKernelArg(pr->k_init, 1, sizeof b_pk, &b_pk);
    clSetKernelArg(pr->k_init, 2, sizeof b_ok, &b_ok); clSetKernelArg(pr->k_init, 3, sizeof pr->consts, &pr->consts);
    size_t one = 1; CL_CHECK(clEnqueueNDRangeKernel(d.q, pr->k_init, 1, nullptr, &one, &one, 0, nullptr, nullptr));
    int ok = 0; CL_CHECK(clEnqueueReadBuffer(d.q, b_ok, CL_TRUE, 0, 4, &ok, 0, nullptr, nullptr));
    CL_CHECK(clEnqueueReadBuffer(d.q, b_pk, CL_TRUE, 0, 32, pk_out, 0, nullptr, nullptr));
    clReleaseMemObject(b_sc); clReleaseMemObject(b_pk); clReleaseMemObject(b_ok);
    return ok;
}

// per-launch scratch of the compacted pipeline
struct Scratch { cl_mem pts = nullptr, nonce[2] = {nullptr, nullptr}, ctr[2] = {nullptr, nullptr}, cnt = nullptr; size_t n = 0; };
static Scratch *scratch_alloc(Prog *pr, size_t n) {
    if (!pr->v->compact) return nullptr;
    cl_int err; ClDev &d = *pr->dev; Scratch *sc = new Scratch(); sc->n = n;
    size_t hpoint = 2 * pr->fe_size + 8;
    sc->pts = clCreateBuffer(d.ctx, CL_MEM_READ_WRITE, n * hpoint, nullptr, &err); CL_CHECK(err);
    for (int b = 0; b < 2; b++) { sc->nonce[b] = clCreateBuffer(d.ctx, CL_MEM_READ_WRITE, n * 8, nullptr, &err); CL_CHECK(err); sc->ctr[b] = clCreateBuffer(d.ctx, CL_MEM_READ_WRITE, n, nullptr, &err); CL_CHECK(err); }
    sc->cnt = clCreateBuffer(d.ctx, CL_MEM_READ_WRITE, (PX_MAX_PASSES + 2) * 4, nullptr, &err); CL_CHECK(err);
    cl_uint N = (cl_uint)n; CL_CHECK(clEnqueueWriteBuffer(d.q, sc->cnt, CL_TRUE, (PX_MAX_PASSES + 1) * 4, 4, &N, 0, nullptr, nullptr));
    return sc;
}
static void scratch_free(Scratch *sc) {
    if (!sc) return;
    clReleaseMemObject(sc->pts); for (int b = 0; b < 2; b++) { clReleaseMemObject(sc->nonce[b]); clReleaseMemObject(sc->ctr[b]); } clReleaseMemObject(sc->cnt); delete sc;
}
// one launch over stride*iters nonces starting at nonce_base; hits appended to d_hits/d_n
static void prog_mine(Prog *pr, cl_mem d_job, uint64_t nonce_base, uint64_t stride, int iters, cl_mem d_hits, cl_mem d_n, Scratch *sc) {
    ClDev &d = *pr->dev; cl_uint zero = 0;
    if (pr->v->compact) {
        size_t N = stride * (size_t)iters;
        CL_CHECK(clEnqueueFillBuffer(d.q, sc->cnt, &zero, 4, 0, (PX_MAX_PASSES + 1) * 4, 0, nullptr, nullptr));
        size_t g1 = ((N + 255) / 256);
        for (int p = 0; p < PX_MAX_PASSES; p++) {
            int in = (p + 1) & 1, out = p & 1; // pass p reads queue (p-1)&1, writes queue p&1
            int in_idx = p == 0 ? PX_MAX_PASSES + 1 : p, out_idx = p + 1, first = p == 0 ? 1 : 0;
            size_t groups = std::max<size_t>(1, g1 >> p), gsz = groups * 256, lsz = 256;
            cl_mem in_nonce = sc->nonce[in], in_ctr = sc->ctr[in], out_nonce = sc->nonce[out], out_ctr = sc->ctr[out];
            clSetKernelArg(pr->k_h2c, 0, sizeof d_job, &d_job); clSetKernelArg(pr->k_h2c, 1, sizeof nonce_base, &nonce_base);
            clSetKernelArg(pr->k_h2c, 2, sizeof in_nonce, &in_nonce); clSetKernelArg(pr->k_h2c, 3, sizeof in_ctr, &in_ctr);
            clSetKernelArg(pr->k_h2c, 4, sizeof sc->cnt, &sc->cnt); clSetKernelArg(pr->k_h2c, 5, sizeof in_idx, &in_idx);
            clSetKernelArg(pr->k_h2c, 6, sizeof out_idx, &out_idx); clSetKernelArg(pr->k_h2c, 7, sizeof first, &first);
            clSetKernelArg(pr->k_h2c, 8, sizeof sc->pts, &sc->pts); clSetKernelArg(pr->k_h2c, 9, sizeof out_nonce, &out_nonce);
            clSetKernelArg(pr->k_h2c, 10, sizeof out_ctr, &out_ctr); clSetKernelArg(pr->k_h2c, 11, sizeof pr->consts, &pr->consts);
            CL_CHECK(clEnqueueNDRangeKernel(d.q, pr->k_h2c, 1, nullptr, &gsz, &lsz, 0, nullptr, nullptr));
        }
        size_t gsz = stride, lsz = pr->v->wg;
        clSetKernelArg(pr->k_mult, 0, sizeof d_job, &d_job); clSetKernelArg(pr->k_mult, 1, sizeof sc->pts, &sc->pts);
        clSetKernelArg(pr->k_mult, 2, sizeof sc->cnt, &sc->cnt); clSetKernelArg(pr->k_mult, 3, sizeof d_hits, &d_hits);
        clSetKernelArg(pr->k_mult, 4, sizeof d_n, &d_n); clSetKernelArg(pr->k_mult, 5, sizeof pr->consts, &pr->consts);
        CL_CHECK(clEnqueueNDRangeKernel(d.q, pr->k_mult, 1, nullptr, &gsz, &lsz, 0, nullptr, nullptr));
    } else {
        size_t gsz = stride, lsz = pr->v->wg;
        clSetKernelArg(pr->k_mine, 0, sizeof d_job, &d_job); clSetKernelArg(pr->k_mine, 1, sizeof nonce_base, &nonce_base);
        clSetKernelArg(pr->k_mine, 2, sizeof stride, &stride); clSetKernelArg(pr->k_mine, 3, sizeof iters, &iters);
        clSetKernelArg(pr->k_mine, 4, sizeof d_hits, &d_hits); clSetKernelArg(pr->k_mine, 5, sizeof d_n, &d_n);
        clSetKernelArg(pr->k_mine, 6, sizeof pr->consts, &pr->consts);
        CL_CHECK(clEnqueueNDRangeKernel(d.q, pr->k_mine, 1, nullptr, &gsz, &lsz, 0, nullptr, nullptr));
    }
}
static int prog_test(Prog *pr, cl_mem d_job, uint64_t nonce, uint8_t beta[64], uint8_t h[32], uint8_t cg[32]) {
    cl_int err; ClDev &d = *pr->dev;
    cl_mem b_b = clCreateBuffer(d.ctx, CL_MEM_WRITE_ONLY, 64, nullptr, &err), b_h = clCreateBuffer(d.ctx, CL_MEM_WRITE_ONLY, 32, nullptr, &err), b_c = clCreateBuffer(d.ctx, CL_MEM_WRITE_ONLY, 32, nullptr, &err), b_ok = clCreateBuffer(d.ctx, CL_MEM_WRITE_ONLY, 4, nullptr, &err);
    clSetKernelArg(pr->k_test, 0, sizeof d_job, &d_job); clSetKernelArg(pr->k_test, 1, sizeof nonce, &nonce);
    clSetKernelArg(pr->k_test, 2, sizeof b_b, &b_b); clSetKernelArg(pr->k_test, 3, sizeof b_h, &b_h); clSetKernelArg(pr->k_test, 4, sizeof b_c, &b_c);
    clSetKernelArg(pr->k_test, 5, sizeof b_ok, &b_ok); clSetKernelArg(pr->k_test, 6, sizeof pr->consts, &pr->consts);
    size_t one = 1; CL_CHECK(clEnqueueNDRangeKernel(d.q, pr->k_test, 1, nullptr, &one, &one, 0, nullptr, nullptr));
    int ok = 0;
    CL_CHECK(clEnqueueReadBuffer(d.q, b_b, CL_TRUE, 0, 64, beta, 0, nullptr, nullptr)); CL_CHECK(clEnqueueReadBuffer(d.q, b_h, CL_TRUE, 0, 32, h, 0, nullptr, nullptr));
    CL_CHECK(clEnqueueReadBuffer(d.q, b_c, CL_TRUE, 0, 32, cg, 0, nullptr, nullptr)); CL_CHECK(clEnqueueReadBuffer(d.q, b_ok, CL_TRUE, 0, 4, &ok, 0, nullptr, nullptr));
    clReleaseMemObject(b_b); clReleaseMemObject(b_h); clReleaseMemObject(b_c); clReleaseMemObject(b_ok);
    return ok;
}
static int prog_fetest(Prog *pr, uint32_t seed, int n, uint8_t dump[128]) {
    cl_int err; ClDev &d = *pr->dev;
    cl_mem b_f = clCreateBuffer(d.ctx, CL_MEM_WRITE_ONLY, 4, nullptr, &err), b_d = clCreateBuffer(d.ctx, CL_MEM_WRITE_ONLY, 128, nullptr, &err);
    clSetKernelArg(pr->k_fetest, 0, sizeof seed, &seed); clSetKernelArg(pr->k_fetest, 1, sizeof n, &n);
    clSetKernelArg(pr->k_fetest, 2, sizeof b_f, &b_f); clSetKernelArg(pr->k_fetest, 3, sizeof b_d, &b_d); clSetKernelArg(pr->k_fetest, 4, sizeof pr->consts, &pr->consts);
    size_t one = 1; CL_CHECK(clEnqueueNDRangeKernel(d.q, pr->k_fetest, 1, nullptr, &one, &one, 0, nullptr, nullptr));
    int fail = 0; CL_CHECK(clEnqueueReadBuffer(d.q, b_f, CL_TRUE, 0, 4, &fail, 0, nullptr, nullptr)); CL_CHECK(clEnqueueReadBuffer(d.q, b_d, CL_TRUE, 0, 128, dump, 0, nullptr, nullptr));
    clReleaseMemObject(b_f); clReleaseMemObject(b_d);
    return fail;
}
static cl_mem make_job(ClDev &d, JobDev jd) { prepare_job(jd); cl_int err; cl_mem m = clCreateBuffer(d.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof jd, (void *)&jd, &err); CL_CHECK(err); return m; }

// ---- the per-GPU worker ----------------------------------------------------------
struct GpuCfg { int dev; int groups; int iters; const Variant *v; int ordinal = 0; };
static void gpu_worker(Shared &sh, GpuCfg cfg) {
    ClDev &d = g_devs[cfg.dev];
    Prog *pr = get_prog(d, cfg.v);
    cl_int err;
    cl_mem d_job = clCreateBuffer(d.ctx, CL_MEM_READ_ONLY, sizeof(JobDev), nullptr, &err); CL_CHECK(err);
    cl_mem d_hits = clCreateBuffer(d.ctx, CL_MEM_READ_WRITE, sizeof(Hit) * MAX_HITS, nullptr, &err); CL_CHECK(err);
    cl_mem d_n = clCreateBuffer(d.ctx, CL_MEM_READ_WRITE, 4, nullptr, &err); CL_CHECK(err);
    std::vector<Hit> hits(MAX_HITS);
    std::mt19937_64 rng(std::random_device{}() ^ (uint64_t)cfg.dev * 0x9E3779B97F4A7C15ULL);
    uint64_t working = 0, nonce = 0;
    uint64_t stride = (uint64_t)cfg.groups * cfg.v->wg;
    Scratch *sc = scratch_alloc(pr, stride * (size_t)cfg.iters);
    logln("GPU " + std::to_string(cfg.dev) + " " + dev_label(d) + ": variant " + cfg.v->name + ", " + std::to_string(cfg.groups) + "x" + std::to_string(cfg.v->wg) + " work-items, " + std::to_string(cfg.iters) + " nonces/item/launch");
    auto t_last = std::chrono::steady_clock::now(); uint64_t local_attempts = 0;
    double idle_nojob = 0, idle_stale = 0, kernel_ms = 0; int launches = 0;
    while (!sh.stop) {
        Job j; { std::lock_guard<std::mutex> l(sh.mu); j = sh.job; }
        if (j.id == 0 || j.stale) {
            if (j.id == 0) idle_nojob += 10; else idle_stale += 10;
            std::this_thread::sleep_for(std::chrono::milliseconds(10)); continue;
        }
        if (j.id != working) { working = j.id; nonce = j.has_base ? (j.nonce_base | ((uint64_t)(cfg.ordinal & 0xff) << 40)) : rng(); }
        auto k0 = std::chrono::steady_clock::now();
        JobDev jd; memset(&jd, 0, sizeof jd);
        memcpy(jd.pk, sh.pk, 32); memcpy(jd.prefix, j.prefix.data(), j.prefix.size()); jd.prefix_len = (int)j.prefix.size();
        if (j.has_beat) memcpy(jd.beat, j.beat, 64); else memset(jd.beat, 0xff, 64);
        jd.k = j.k;
        if (jd.prefix_len != 37) { logln("job with alpha_prefix of " + std::to_string(jd.prefix_len) + " bytes; this miner expects 37"); std::this_thread::sleep_for(std::chrono::seconds(5)); continue; }
        prepare_job(jd);
        CL_CHECK(clEnqueueWriteBuffer(d.q, d_job, CL_FALSE, 0, sizeof jd, &jd, 0, nullptr, nullptr));
        cl_uint zero = 0; CL_CHECK(clEnqueueFillBuffer(d.q, d_n, &zero, 4, 0, 4, 0, nullptr, nullptr));
        prog_mine(pr, d_job, nonce, stride, cfg.iters, d_hits, d_n, sc);
        CL_CHECK(clFinish(d.q));
        kernel_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - k0).count(); launches++;
        uint64_t done = stride * cfg.iters;
        nonce += done;
        sh.attempts += done; local_attempts += done;
        cl_uint n = 0; CL_CHECK(clEnqueueReadBuffer(d.q, d_n, CL_TRUE, 0, 4, &n, 0, nullptr, nullptr));
        if (n > 0) {
            if (n > MAX_HITS) n = MAX_HITS;
            CL_CHECK(clEnqueueReadBuffer(d.q, d_hits, CL_TRUE, 0, sizeof(Hit) * n, hits.data(), 0, nullptr, nullptr));
            std::sort(hits.begin(), hits.begin() + n, [](const Hit &a, const Hit &b) { return memcmp(a.beta, b.beta, 64) < 0; });
            for (unsigned i = 0; i < n; i++) {
                Job cur; { std::lock_guard<std::mutex> l(sh.mu); cur = sh.job; }
                if (cur.id != j.id || cur.stale) break;
                if (cur.has_beat && memcmp(hits[i].beta, cur.beat, 64) >= 0) continue;
                submit(sh, cur, hits[i].nonce, hits[i].beta, sh.attempts.exchange(0));
            }
        }
        auto now = std::chrono::steady_clock::now();
        if (now - t_last >= std::chrono::seconds(30)) {
            double s = std::chrono::duration<double>(now - t_last).count();
            char line[300]; snprintf(line, sizeof line, "GPU %d: %.2f MH/s avg over %.0fs | in kernel %.0f%% (%d launches, %.1f ms each) | idle: no job %.0f%%, stale %.0f%%",
                cfg.dev, local_attempts / s / 1e6, s, kernel_ms / 10 / s, launches, launches ? kernel_ms / launches : 0, idle_nojob / 10 / s, idle_stale / 10 / s);
            logln(line);
            { std::lock_guard<std::mutex> l(sh.mu); if ((size_t)cfg.ordinal < sh.gpus.size()) sh.gpus[cfg.ordinal].rate = local_attempts / s; }
            t_last = now; local_attempts = 0; idle_nojob = idle_stale = kernel_ms = 0; launches = 0;
        }
    }
    scratch_free(sc); clReleaseMemObject(d_job); clReleaseMemObject(d_hits); clReleaseMemObject(d_n);
}

// ---- tests, bench, autotune, verify (as in the CUDA miner) -------------------------------
static int run_selftest(Prog *pr, const std::string &path, bool quiet = false) {
    std::string text;
    if (path.empty() || path == "embedded") text.assign((const char *)VEC_SRC, VEC_SRC_LEN);
    else { std::ifstream f(path); if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); return 2; } std::stringstream ss; ss << f.rdbuf(); text = ss.str(); }
    std::stringstream f(text);
    std::string line; int n = 0, bad = 0; std::string cur_x;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::string pk, x, alpha, h, cg, beta; uint64_t ctr = 0;
        json_str(line, "pk", pk); json_str(line, "x", x); json_str(line, "alpha", alpha);
        json_str(line, "h", h); json_str(line, "cg", cg); json_str(line, "beta", beta); json_u64(line, "ctr", ctr);
        std::vector<uint8_t> xb, pkb, ab;
        unhex(x, xb); unhex(pk, pkb); unhex(alpha, ab);
        if (xb.size() != 32 || pkb.size() != 32 || ab.size() != 45) { fprintf(stderr, "bad vector line\n"); bad++; continue; }
        if (x != cur_x) { uint8_t pk2[32]; prog_init(pr, xb.data(), pk2); cur_x = x; }
        JobDev jd; memset(&jd, 0, sizeof jd);
        memcpy(jd.pk, pkb.data(), 32); memcpy(jd.prefix, ab.data(), 37); jd.prefix_len = 37;
        uint64_t nonce = 0; for (int i = 0; i < 8; i++) nonce = (nonce << 8) | ab[37 + i];
        cl_mem d_job = make_job(*pr->dev, jd);
        uint8_t ob[64], oh[32], ocg[32];
        int ok = prog_test(pr, d_job, nonce, ob, oh, ocg);
        clReleaseMemObject(d_job);
        n++;
        bool hok = hexs(oh, 32) == h, cgok = hexs(ocg, 32) == cg, bok = hexs(ob, 64) == beta;
        if (!ok || !hok || !cgok || !bok) {
            bad++;
            fprintf(stderr, "[%s] vector %d FAILED (ctr=%llu, ok=%d): H %s, 8G %s, beta %s\n", pr->v->name, n, (unsigned long long)ctr, ok, hok ? "ok" : "MISMATCH", cgok ? "ok" : "MISMATCH", bok ? "ok" : "MISMATCH");
            if (!hok) fprintf(stderr, "    H   got %s\n    H   exp %s\n", hexs(oh, 32).c_str(), h.c_str());
            if (!cgok) fprintf(stderr, "    8G  got %s\n    8G  exp %s\n", hexs(ocg, 32).c_str(), cg.c_str());
        }
    }
    if (!quiet || bad) fprintf(stderr, "[%s] selftest on %s: %d vectors, %d failed\n", pr->v->name, pr->dev->name.c_str(), n, bad);
    return bad ? 1 : 0;
}
static double run_bench(Prog *pr, const uint8_t pk[32], int groups, int iters, double seconds, bool verbose) {
    ClDev &d = *pr->dev; cl_int err;
    if (groups <= 0) groups = (int)d.cus * 8;
    JobDev jd; memset(&jd, 0, sizeof jd); jd.prefix_len = 37; jd.k = 20; memset(jd.beat, 0xff, 64);
    memcpy(jd.pk, pk, 32);
    for (int i = 0; i < 37; i++) jd.prefix[i] = (uint8_t)(i * 31 + 7);
    cl_mem d_job = make_job(d, jd);
    cl_mem d_hits = clCreateBuffer(d.ctx, CL_MEM_READ_WRITE, sizeof(Hit) * MAX_HITS, nullptr, &err); CL_CHECK(err);
    cl_mem d_n = clCreateBuffer(d.ctx, CL_MEM_READ_WRITE, 4, nullptr, &err); CL_CHECK(err);
    uint64_t stride = (uint64_t)groups * pr->v->wg, total = 0, nonce = 0x123456789ULL; cl_uint zero = 0;
    Scratch *sc = scratch_alloc(pr, stride * (size_t)iters);
    CL_CHECK(clEnqueueFillBuffer(d.q, d_n, &zero, 4, 0, 4, 0, nullptr, nullptr)); prog_mine(pr, d_job, nonce, stride, iters, d_hits, d_n, sc); CL_CHECK(clFinish(d.q));
    auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < seconds) {
        CL_CHECK(clEnqueueFillBuffer(d.q, d_n, &zero, 4, 0, 4, 0, nullptr, nullptr));
        prog_mine(pr, d_job, nonce, stride, iters, d_hits, d_n, sc);
        CL_CHECK(clFinish(d.q));
        nonce += stride * iters; total += stride * iters;
    }
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    double mhs = total / s / 1e6;
    if (verbose) fprintf(stderr, "GPU %s [%s %dx%d iters %d]: %.3f MH/s\n", dev_label(d).c_str(), pr->v->name, groups, pr->v->wg, iters, mhs);
    scratch_free(sc); clReleaseMemObject(d_job); clReleaseMemObject(d_hits); clReleaseMemObject(d_n);
    return mhs;
}
static GpuCfg autotune(int dev, const uint8_t scalar[32], const uint8_t pk[32], double seconds, int fixedGroups) {
    ClDev &d = g_devs[dev];
    const Platform &pl = platformFor(d);
    GpuCfg best{dev, fixedGroups, pl.iters, variantByName(pl.variant)};
    double bestRate = 0;
    const int iters_list[] = { 2, 4, 8 };
    fprintf(stderr, "autotune GPU %d %s, %.0fs per trial:\n", dev, dev_label(d).c_str(), seconds);
    for (int v = 0; v < NVARIANTS; v++) {
        Prog *pr = get_prog(d, &VARIANTS[v]);
        uint8_t pk2[32];
        if (prog_init(pr, scalar, pk2) != 1 || memcmp(pk2, pk, 32) != 0) { fprintf(stderr, "  %-12s: init failed, skipped\n", VARIANTS[v].name); continue; }
        if (!prog_validated(pr)) { fprintf(stderr, "  %-12s: fails the test vectors here, skipped\n", VARIANTS[v].name); continue; }
        prog_init(pr, scalar, pk2);
        for (int it : iters_list) {
            double r = run_bench(pr, pk, fixedGroups, it, seconds, false);
            fprintf(stderr, "  %-12s x%d: %.3f MH/s%s\n", VARIANTS[v].name, it, r, r > bestRate ? "  <- best so far" : "");
            if (r > bestRate) { bestRate = r; best = GpuCfg{dev, fixedGroups, it, &VARIANTS[v]}; }
        }
    }
    fprintf(stderr, "autotune GPU %d: %s x%d (%.3f MH/s)\n", dev, best.v->name, best.iters, bestRate);
    return best;
}
// every variant over the same nonce range at low difficulty must find exactly the same hits
static int run_verify(const uint8_t scalar[32], const uint8_t pk[32], int dev) {
    ClDev &d = g_devs[dev]; cl_int err;
    int groups = (int)d.cus * 4, iters = 2, launches = 3;
    JobDev jd; memset(&jd, 0, sizeof jd); jd.prefix_len = 37; jd.k = 15; memset(jd.beat, 0xff, 64);
    memcpy(jd.pk, pk, 32);
    for (int i = 0; i < 37; i++) jd.prefix[i] = (uint8_t)(i * 31 + 7);
    cl_mem d_job = make_job(d, jd);
    cl_mem d_hits = clCreateBuffer(d.ctx, CL_MEM_READ_WRITE, sizeof(Hit) * MAX_HITS, nullptr, &err); CL_CHECK(err);
    cl_mem d_n = clCreateBuffer(d.ctx, CL_MEM_READ_WRITE, 4, nullptr, &err); CL_CHECK(err);
    std::vector<std::vector<std::pair<uint64_t, std::string>>> found(NVARIANTS);
    int rc = 0;
    for (int v = 0; v < NVARIANTS; v++) {
        Prog *pr = get_prog(d, &VARIANTS[v]);
        uint8_t pk2[32]; if (prog_init(pr, scalar, pk2) != 1) { fprintf(stderr, "[%s] init failed\n", VARIANTS[v].name); rc = 1; continue; }
        uint64_t stride = (uint64_t)groups * pr->v->wg, nonce = 0x5555000000ULL; cl_uint zero = 0;
        Scratch *sc = scratch_alloc(pr, stride * (size_t)iters);
        for (int l = 0; l < launches; l++) {
            CL_CHECK(clEnqueueFillBuffer(d.q, d_n, &zero, 4, 0, 4, 0, nullptr, nullptr));
            prog_mine(pr, d_job, nonce, stride, iters, d_hits, d_n, sc); CL_CHECK(clFinish(d.q));
            cl_uint n = 0; CL_CHECK(clEnqueueReadBuffer(d.q, d_n, CL_TRUE, 0, 4, &n, 0, nullptr, nullptr)); if (n > MAX_HITS) n = MAX_HITS;
            std::vector<Hit> hits(n); if (n) CL_CHECK(clEnqueueReadBuffer(d.q, d_hits, CL_TRUE, 0, sizeof(Hit) * n, hits.data(), 0, nullptr, nullptr));
            for (auto &h : hits) found[v].push_back({h.nonce, hexs(h.beta, 64)});
            nonce += stride * iters;
        }
        scratch_free(sc);
        std::sort(found[v].begin(), found[v].end());
        bool same = v == 0 || found[v] == found[0];
        fprintf(stderr, "[%s] %zu hits, %s\n", VARIANTS[v].name, found[v].size(), v == 0 ? "reference" : same ? "identical" : "DIFFERENT");
        if (!same) rc = 1;
    }
    if (found[0].empty()) { fprintf(stderr, "verify: no hits at all — suspicious\n"); rc = 1; }
    clReleaseMemObject(d_job); clReleaseMemObject(d_hits); clReleaseMemObject(d_n);
    return rc;
}

static void usage() {
    fprintf(stderr,
        "ProxiChorizo " PX_VERSION " (OpenCL) — nonce seeker for Proxima (suprnova.cc)\n"
        "  --proxi URL        job server of the pool or of `proxi node mine` (e.g. http://proxi.suprnova.cc:8100)\n"
        "  --token T          bearer token (pool registration / mine.seeker.token)\n"
        "  --key-file FILE    unencrypted proxima.key (or --seed-hex HEX / NONCE_SEEKER_SEED)\n"
        "  --seed-hex HEX     32-byte seed\n"
        "  --name NAME        seeker name in the pool's statistics (default proxichorizo-<pid>)\n"
        "  --api-bind [H:]P   stats API, JSON at http://H:P/ and the mmpOS line at /mmpos (default off)\n"
        "  --gpus 0,1         devices to use (default all; --list-devices shows them)\n"
        "  --variant NAME     kernel variant for all GPUs (default: per-GPU table; --list-variants)\n"
        "  --autotune [SECS]  measure every variant and iteration count on each GPU first (default 6s per trial)\n"
        "  --groups N --iters N   work-groups per launch (default 8 per compute unit) and nonces per work-item\n"
        "  --kernel-file F    read the OpenCL source from F instead of the embedded copy; --build-opts \"...\" extra compiler flags\n"
        "  --selftest [FILE]  verify every variant against the embedded test vectors (or FILE) and exit\n"
        "  --fetest           randomized field-arithmetic test of every variant and exit\n"
        "  --verify           every variant over one nonce range at low difficulty must find identical hits; exit\n"
        "  --bench [SECS]     measure the hashrate of the chosen variant and exit\n");
}

int main(int argc, char **argv) {
    std::string proxi, token, keyfile, seedhex, name, selftest, gpus, variant, apiBind, kernelFile;
    int groups = 0, iters = 0; double bench = 0, autotuneSecs = 0; bool fetest = false, listVariants = false, listDevices = false, verify = false;
    sock_init();
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](std::string &dst) { if (i + 1 < argc) dst = argv[++i]; };
        auto optnum = [&](double dflt) { if (i + 1 < argc && isdigit((unsigned char)argv[i + 1][0])) return atof(argv[++i]); return dflt; };
        if (a == "--proxi") next(proxi);
        else if (a == "--token") next(token);
        else if (a == "--key-file") next(keyfile);
        else if (a == "--seed-hex") next(seedhex);
        else if (a == "--name") next(name);
        else if (a == "--api-bind" || a == "--api-port") next(apiBind);
        else if (a == "--gpus") next(gpus);
        else if (a == "--variant") next(variant);
        else if (a == "--kernel-file") next(kernelFile);
        else if (a == "--build-opts") next(g_extra_build);
        else if (a == "--selftest") { selftest = "embedded"; if (i + 1 < argc && argv[i + 1][0] != '-') selftest = argv[++i]; }
        else if (a == "--fetest") fetest = true;
        else if (a == "--verify") verify = true;
        else if (a == "--list-variants") listVariants = true;
        else if (a == "--list-devices") listDevices = true;
        else if (a == "--groups" || a == "--blocks") { std::string s; next(s); groups = atoi(s.c_str()); }
        else if (a == "--threads") { std::string s; next(s); }  // accepted for CUDA command-line compatibility; the work-group size is per variant
        else if (a == "--iters") { std::string s; next(s); iters = atoi(s.c_str()); }
        else if (a == "--bench") bench = optnum(20);
        else if (a == "--autotune") autotuneSecs = optnum(6);
        else { usage(); return 1; }
    }
    if (listVariants) { for (int v = 0; v < NVARIANTS; v++) printf("%s\n", VARIANTS[v].name); return 0; }
    if (!kernelFile.empty()) { std::ifstream f(kernelFile); if (!f) { fprintf(stderr, "cannot open %s\n", kernelFile.c_str()); return 1; } std::stringstream ss; ss << f.rdbuf(); g_kernel_src = ss.str(); }
    else g_kernel_src.assign((const char *)CL_SRC, CL_SRC_LEN);
    enumerate_devices();
    if (g_devs.empty()) { fprintf(stderr, "no OpenCL GPU found (is the OpenCL ICD installed? ROCm: /etc/OpenCL/vendors)\n"); return 2; }
    if (listDevices) { for (size_t i = 0; i < g_devs.size(); i++) printf("%zu: %s, %u CUs, bus %d, platform %s\n", i, dev_label(g_devs[i]).c_str(), g_devs[i].cus, g_devs[i].bus, g_devs[i].platname.c_str()); return 0; }
    std::vector<int> devs;
    if (gpus.empty()) { for (size_t i = 0; i < g_devs.size(); i++) devs.push_back((int)i); }
    else { std::stringstream ss(gpus); std::string t; while (std::getline(ss, t, ',')) { int d = atoi(t.c_str()); if (d < 0 || d >= (int)g_devs.size()) { fprintf(stderr, "no GPU %d\n", d); return 1; } devs.push_back(d); } }
    const Variant *forced = nullptr;
    if (!variant.empty()) { forced = variantByName(variant); if (!forced) { fprintf(stderr, "unknown variant %s (--list-variants)\n", variant.c_str()); return 1; } }

    // key -> clamped scalar (SHA-512 of the seed), as the Go/Rust seekers do
    uint8_t seed[32], scalar[32], pk[32];
    bool keyGiven = !seedhex.empty() || !keyfile.empty() || getenv("NONCE_SEEKER_SEED") != nullptr;
    if (keyGiven) { if (!load_seed(seedhex, keyfile, seed)) return 1; }
    else { for (int i = 0; i < 32; i++) seed[i] = (uint8_t)(i * 97 + 13); }  // tests/bench without a key: fixed pseudo-random seed
    { uint8_t h[64]; host_sha512(seed, 32, h); memcpy(scalar, h, 32); scalar[0] &= 248; scalar[31] &= 127; scalar[31] |= 64; }

    if (!selftest.empty() || fetest || verify) {
        int rc = 0; ClDev &d = g_devs[devs[0]];
        for (int v = 0; v < NVARIANTS; v++) {
            if (forced && &VARIANTS[v] != forced) continue;
            Prog *pr = get_prog(d, &VARIANTS[v]);
            uint8_t pk2[32]; int ok = prog_init(pr, scalar, pk2);
            if (ok != 1) { fprintf(stderr, "[%s] constant derivation FAILED (%s) %s\n", VARIANTS[v].name, ok == -2 ? "fe_sq != fe_mul" : ok == -3 ? "sqrt(-1)^2 != -1, got" : ok == -4 ? "basepoint decode, y =" : "?", hexs(pk2, 32).c_str()); rc = 1; }
            if (ok == 1 && !selftest.empty()) rc |= run_selftest(pr, selftest);
            if (fetest) {
                int worst = 0;
                for (int sd = 1; sd <= 16 && !worst; sd++) { uint8_t dump[128]; int fail = prog_fetest(pr, sd, 2000, dump); if (fail) { worst = fail; fprintf(stderr, "[%s] fetest FAIL code %d at iter %d\n x %s\n y %s\n a %s\n b %s\n", VARIANTS[v].name, fail / 1000000, fail % 1000000, hexs(dump, 32).c_str(), hexs(dump + 32, 32).c_str(), hexs(dump + 64, 32).c_str(), hexs(dump + 96, 32).c_str()); } }
                if (!worst) fprintf(stderr, "[%s] fetest: 32000 random values ok\n", VARIANTS[v].name); else rc = 1;
            }
        }
        if (verify) rc |= run_verify(scalar, pk, devs[0]);
        return rc;
    }

    std::vector<GpuCfg> cfgs; std::vector<Shared::GpuStat> gstats;
    for (int dv : devs) {
        ClDev &d = g_devs[dv];
        const Platform &pl = platformFor(d);
        GpuCfg cfg{dv, groups > 0 ? groups : (int)d.cus * 8, iters > 0 ? iters : pl.iters, forced ? forced : variantByName(pl.variant)};
        cfg.ordinal = (int)cfgs.size();
        Prog *pr = get_prog(d, cfg.v);
        if (!prog_validated(pr)) {
            if (forced) { fprintf(stderr, "device %d: the forced variant %s is wrong on this driver; refusing to mine with it\n", dv, cfg.v->name); return 2; }
            cfg.v = variantByName("fe32_wnaf_c"); pr = get_prog(d, cfg.v);  // the layout known to be right on every driver so far
            if (!prog_validated(pr)) { fprintf(stderr, "device %d: no variant reproduces the test vectors\n", dv); return 2; }
        }
        uint8_t pk2[32]; int ok = prog_init(pr, scalar, pk2);
        if (ok != 1) { fprintf(stderr, "device %d: constant derivation failed on %s\n", dv, cfg.v->name); return 2; }
        if (dv == devs[0]) memcpy(pk, pk2, 32); else if (memcmp(pk, pk2, 32) != 0) { fprintf(stderr, "device %d derives a different public key\n", dv); return 2; }
        if (autotuneSecs > 0) { cfg = autotune(dv, scalar, pk, autotuneSecs, groups); if (iters > 0) cfg.iters = iters; cfg.ordinal = (int)cfgs.size(); prog_init(get_prog(d, cfg.v), scalar, pk2); }
        else if (!prog_validated(pr)) return 2;
        else fprintf(stderr, "GPU %d %s: %s x%d — %s\n", dv, dev_label(d).c_str(), cfg.v->name, cfg.iters, pl.note);
        char bus[32]; snprintf(bus, sizeof bus, "0000:%02x:00.0", d.bus);
        gstats.push_back(Shared::GpuStat{dv, cfg.ordinal, dev_label(d), bus, d.bus, 0.0});
        cfgs.push_back(cfg);
    }
    if (bench > 0) { for (auto &c : cfgs) run_bench(get_prog(g_devs[c.dev], c.v), pk, c.groups, c.iters, bench, true); return 0; }
    if (proxi.empty()) { if (autotuneSecs > 0) return 0; usage(); return 1; }

    Shared sh;
    sh.base_url = proxi; if (!parse_url(proxi, sh.url)) return 1;
    sh.token = token; sh.name = name.empty() ? "proxichorizo-" + std::to_string(getpid()) : name;
    memcpy(sh.pk, pk, 32); memcpy(sh.scalar, scalar, 32);
    sh.gpus = gstats; sh.api_bind = apiBind;
    logln("ProxiChorizo " PX_VERSION " (OpenCL): key " + hexs(pk, 32) + ", job server " + sh.url.host + ":" + std::to_string(sh.url.port) + ", " + std::to_string(devs.size()) + " GPU(s), name " + sh.name);
    std::vector<std::thread> th;
    th.emplace_back(poller, std::ref(sh));
    th.emplace_back(reporter, std::ref(sh));
    if (!apiBind.empty()) th.emplace_back(api_server, std::ref(sh));
    for (auto &c : cfgs) th.emplace_back(gpu_worker, std::ref(sh), c);
    for (auto &t : th) t.join();
    return 0;
}
