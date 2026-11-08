// ProxiChorizo - Copyright (c) 2026 suprnova.cc. ProxiChorizo License 1.0: free to run for mining;
// reuse in commercial or closed-source miners needs written permission. See LICENSE.
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
#include <cmath>
#include <cerrno>
#include <ctime>
#include <condition_variable>

#include "../txid_blake2b.h"   // txid-blake2b work of library upgrade 6: CPU reference and per-job precomputation
#include "host_common.inc"
#include "kernel_blob.h"

// must match proxichorizo.cl
struct JobDev { uint8_t pk[32]; uint8_t prefix[64]; int prefix_len; int k; uint8_t beat[64]; uint64_t mw[9]; uint64_t beat_w[8]; uint64_t h2c_mid[8]; };
#include "job_midstate.h"
// the kernels hash on 64-bit words: precompute the fixed part of the hash-to-curve message
// (03 01 pk prefix = 71 bytes -> words 0..7 and the top 56 bits of word 8) and the beat as words
static void prepare_job(JobDev &jd) {
    uint8_t m[72]; memset(m, 0, sizeof m); m[0] = 0x03; m[1] = 0x01; memcpy(m + 2, jd.pk, 32); memcpy(m + 34, jd.prefix, 37);
    for (int i = 0; i < 9; i++) { uint64_t w = 0; for (int j = 0; j < 8; j++) w = (w << 8) | (uint64_t)(i * 8 + j < 71 ? m[i * 8 + j] : 0); jd.mw[i] = w; }
    prepare_sha_mid(&jd);
    for (int i = 0; i < 8; i++) { uint64_t w = 0; for (int j = 0; j < 8; j++) w = (w << 8) | jd.beat[8 * i + j]; jd.beat_w[i] = w; }
}
struct Hit { uint64_t nonce; uint8_t beta[64]; };
#define MAX_HITS 8192  // also accommodates complete low-difficulty verification batches
#define PX_MAX_PASSES 12

#define CL_CHECK(x) do { cl_int e_ = (x); if (e_ != CL_SUCCESS) { fprintf(stderr, "OpenCL error %d at %s:%d\n", (int)e_, __FILE__, __LINE__); exit(2); } } while (0)

// ---- devices -------------------------------------------------------------------
struct ClDev {
    cl_platform_id plat; cl_device_id id; cl_context ctx; cl_command_queue q;
    std::string name, board, platname, tuning_id, driver; cl_uint cus; int bus = 0; size_t maxwg = 256;
};
static std::vector<ClDev> g_devs;
static thread_local std::map<cl_command_queue, std::vector<cl_event>> mining_events;
static cl_int enqueue_mining_kernel(cl_command_queue q, cl_kernel kernel, size_t global, size_t local) {
    cl_event event = nullptr;
    cl_int rc = clEnqueueNDRangeKernel(q, kernel, 1, nullptr, &global, &local, 0, nullptr, &event);
    if (rc == CL_SUCCESS) mining_events[q].push_back(event);
    return rc;
}
// Wait for a flushed end marker without spinning a CPU core. The callback
// also reports negative execution status; failed work never becomes a share.
struct ClCompletion {
    std::mutex mu;
    std::condition_variable cv;
    bool done = false;
    cl_int status = CL_QUEUED;
};
static void CL_CALLBACK mining_complete(cl_event, cl_int status, void *opaque) {
    ClCompletion *c = static_cast<ClCompletion *>(opaque);
    std::lock_guard<std::mutex> lock(c->mu);
    c->status = status;
    c->done = true;
    // Notify while holding the lock: the waiter cannot destroy this state
    // until the callback has finished accessing it.
    c->cv.notify_one();
}
static cl_int mining_wait(cl_command_queue q) {
    cl_event event = nullptr;
    cl_int rc = clEnqueueMarkerWithWaitList(q, 0, nullptr, &event);
    if (rc == CL_SUCCESS) rc = clFlush(q);
    if (rc == CL_SUCCESS) {
        ClCompletion completion;
        rc = clSetEventCallback(event, CL_COMPLETE, mining_complete, &completion);
        if (rc == CL_SUCCESS) {
            std::unique_lock<std::mutex> lock(completion.mu);
            completion.cv.wait(lock, [&] { return completion.done; });
            rc = completion.status;
        }
    }
    // A marker completing does not imply that earlier commands succeeded.
    // Inspect the retained dispatch events as well, then release every handle.
    for (cl_event dispatch : mining_events[q]) {
        cl_int status = CL_QUEUED;
        cl_int query = clGetEventInfo(dispatch, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof status, &status, nullptr);
        if (rc == CL_SUCCESS) rc = query != CL_SUCCESS ? query : status < 0 ? status : status != CL_COMPLETE ? CL_INVALID_OPERATION : CL_SUCCESS;
        clReleaseEvent(dispatch);
    }
    mining_events[q].clear();
    if (event) clReleaseEvent(event);
    return rc;
}
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
            char driver[256] = {0};
            clGetDeviceInfo(d, CL_DRIVER_VERSION, sizeof driver, driver, nullptr); c.driver = driver;
            unsigned char uuid[16] = {0};
            if (clGetDeviceInfo(d, 0x106a /*CL_DEVICE_UUID_KHR*/, sizeof uuid, uuid, nullptr) == CL_SUCCESS) {
                c.tuning_id = "uuid" + hexs(uuid, sizeof uuid);
            } else if (topo.type == 1) {
                char host[256] = {0}; gethostname(host, sizeof host - 1);
                c.tuning_id = std::string(host) + ":pci:" + std::to_string(c.bus) + ":" +
                    std::to_string((unsigned char)topo.device) + ":" + std::to_string((unsigned char)topo.function);
            }
            cl_int err; c.ctx = clCreateContext(nullptr, 1, &d, nullptr, nullptr, &err); CL_CHECK(err);
            c.q = clCreateCommandQueue(c.ctx, d, 0, &err); CL_CHECK(err);
            g_devs.push_back(c);
        }
    }
}
static std::string dev_label(const ClDev &d) { return d.board.empty() ? d.name : d.board + " (" + d.name + ")"; }

// ---- variants: one kernel source, several build configurations --------------------
struct Variant { const char *name; int fe, window, compact, wg; int split = 0; int fast = 0; int cofactor = 0; int karatsuba = 0; };
static const Variant VARIANTS[] = {
    { "fe29_mlsp_c", 29, 5, 1, 256, 16, 1 },
    { "fe29_mlsp_cf3_c", 29, 5, 1, 256, 16, 1, 3 },
    { "fe29_mlsp_kara_c", 29, 5, 1, 256, 16, 1, 0, 1 },
    { "fe29_wnaf_c", 29, 5, 1, 256 },
    { "fe29_wnaf_split8_c", 29, 5, 1, 256, 8 },
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
// measured: benchmarked on this model. An unmeasured model is tuned on first
// start using the current and earlier kernels with appropriate launch sizes; the choice
// cached in proxichorizo-autotune.cache, as the CUDA miner does.
struct Platform {
    const char *prefix; const char *variant; int iters; bool measured; const char *note;
    int groups_per_cu = 8;
    unsigned required_cus = 0;
    const char *board_match = nullptr;
};
static const Platform PLATFORMS[] = {
    { "gfx1100", "fe29_mlsp_cf3_c", 1, true, "RX 7900 XTX: measured model profile", 128, 48, "7900 XTX" },
    { "gfx1102", "fe29_mlsp_kara_c", 1, true, "RX 7600 XT: measured model profile", 128, 16, "7600 XT" },
    { "gfx906",  "fe29_mlsp_c", 1, true, "MI50: measured 60-CU profile", 128, 60, "MI50" },
    { "gfx11", "fe29_wnaf_c", 8, false, "RDNA3 model not measured here, tuned on first start" },
    { "gfx12", "fe29_wnaf_c", 8, false, "RDNA4: not measured here, tuned on first start" },
    { "gfx10", "fe29_wnaf_c", 8, false, "RDNA2/1: not measured here, tuned on first start" },
    { "gfx9",  "fe29_wnaf_c", 8, false, "Vega/CDNA model not measured here, tuned on first start" },
    { "",      "fe29_wnaf_c", 8, false, "unknown family: tuned on first start" },
};
static const Platform &platformFor(const ClDev &d) {
    for (const auto &p : PLATFORMS) {
        if (d.name.rfind(p.prefix, 0) != 0) continue;
        if (p.required_cus && d.cus != p.required_cus) continue;
        if (p.board_match && d.board.find(p.board_match) == std::string::npos) continue;
        return p;
    }
    return PLATFORMS[sizeof PLATFORMS / sizeof PLATFORMS[0] - 1];
}

static std::string g_kernel_src;   // from the blob or --kernel-file
static std::string g_extra_build;  // --build-opts

// a built variant on one device
struct Prog {
    const Variant *v = nullptr; ClDev *dev = nullptr;
    cl_program p = nullptr; cl_kernel k_mine = nullptr, k_h2c = nullptr, k_mult = nullptr, k_test = nullptr, k_fetest = nullptr, k_init = nullptr;
    cl_kernel k_scalar = nullptr, k_finish = nullptr, k_prep = nullptr, k_resolve = nullptr;
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
    if (v->fast) opts += " -DPX_XONLY=1 -DPX_JACOBI=1 -DPX_SHA_MID=1 -DPX_LADDER=1 -DPX_EDWARDS_OPT=1";
    if (v->cofactor) opts += " -DPX_LAD_ENDPOINT=1 -DPX_LAD_COFACTOR=" + std::to_string(v->cofactor);
    if (v->karatsuba) opts += " -DPX_FE29_KARA=1";
    if (v->split) opts += " -DPX_SPLIT_FINISH=" + std::to_string(v->split);
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
    if (v->split) {
        pr->k_scalar = clCreateKernel(pr->p, v->fast ? "ladder_kernel" : "scalar_kernel", &err); CL_CHECK(err);
        pr->k_finish = clCreateKernel(pr->p, "finish_batch_kernel", &err); CL_CHECK(err);
    }
    if(v->fast) { pr->k_prep=clCreateKernel(pr->p,"prep_kernel",&err); CL_CHECK(err); }
    if(v->fast) { pr->k_resolve=clCreateKernel(pr->p,"resolve_kernel",&err); CL_CHECK(err); }
    pr->fe_size = v->fe == 32 ? 32 : v->fe == 51 ? 40 : 36;
    pr->consts_size = 3 * pr->fe_size + 32 + 260 + 8;
    if(v->fast) pr->consts_size += 36;
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

#include "merged_scalar.h"

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
    if(pr->v->fast) {
        uint8_t merged[32];
        if(pr->v->cofactor)v_xonly_scalar(scalar,merged);else v_merged_scalar(scalar,merged);
        int top=0;for(int b=255;b>=0;b--)if((merged[b>>3]>>(b&7))&1){top=b;break;}
        size_t off=3*pr->fe_size+300;
        CL_CHECK(clEnqueueWriteBuffer(pr->dev->q,pr->consts,CL_TRUE,off,32,merged,0,nullptr,nullptr));
        CL_CHECK(clEnqueueWriteBuffer(pr->dev->q,pr->consts,CL_TRUE,off+32,4,&top,0,nullptr,nullptr));
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
struct Scratch { cl_mem candidates = nullptr, ncandidates = nullptr; cl_mem projective = nullptr; cl_mem pts = nullptr, nonce[2] = {nullptr, nullptr}, ctr[2] = {nullptr, nullptr}, cnt = nullptr; size_t n = 0; };
static Scratch *scratch_alloc(Prog *pr, size_t n) {
    if (!pr->v->compact) return nullptr;
    cl_int err; ClDev &d = *pr->dev; Scratch *sc = new Scratch(); sc->n = n;
    size_t hpoint = 2 * pr->fe_size + 8;
    if(pr->v->fast) {
        sc->candidates=clCreateBuffer(d.ctx,CL_MEM_READ_WRITE,n*sizeof(uint64_t),nullptr,&err);CL_CHECK(err);
        sc->ncandidates=clCreateBuffer(d.ctx,CL_MEM_READ_WRITE,sizeof(cl_uint),nullptr,&err);CL_CHECK(err);
    }
    if (pr->v->split) {
        // Three field elements, an 8-byte aligned nonce, valid flag and tail padding.
        size_t bytes = ((3 * pr->fe_size + 7) / 8) * 8 + 16;
        sc->projective = clCreateBuffer(d.ctx, CL_MEM_READ_WRITE, n * bytes, nullptr, &err); CL_CHECK(err);
    }
    sc->pts = clCreateBuffer(d.ctx, CL_MEM_READ_WRITE, n * hpoint, nullptr, &err); CL_CHECK(err);
    for (int b = 0; b < 2; b++) { sc->nonce[b] = clCreateBuffer(d.ctx, CL_MEM_READ_WRITE, n * 8, nullptr, &err); CL_CHECK(err); sc->ctr[b] = clCreateBuffer(d.ctx, CL_MEM_READ_WRITE, n, nullptr, &err); CL_CHECK(err); }
    sc->cnt = clCreateBuffer(d.ctx, CL_MEM_READ_WRITE, (PX_MAX_PASSES + 2) * 4, nullptr, &err); CL_CHECK(err);
    cl_uint N = (cl_uint)n; CL_CHECK(clEnqueueWriteBuffer(d.q, sc->cnt, CL_TRUE, (PX_MAX_PASSES + 1) * 4, 4, &N, 0, nullptr, nullptr));
    return sc;
}
static void scratch_free(Scratch *sc) {
    if (!sc) return;
    if(sc->candidates)clReleaseMemObject(sc->candidates);
    if(sc->ncandidates)clReleaseMemObject(sc->ncandidates);
    if (sc->projective) clReleaseMemObject(sc->projective);
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
            CL_CHECK(enqueue_mining_kernel(d.q, pr->k_h2c, gsz, lsz));
        }
        if (pr->v->split) {
            size_t gsz = stride, lsz = pr->v->wg;
            if(pr->v->fast) {
                size_t prep_gsz=((stride+16*lsz-1)/(16*lsz))*lsz;
                CL_CHECK(clSetKernelArg(pr->k_prep,0,sizeof sc->cnt,&sc->cnt));
                CL_CHECK(clSetKernelArg(pr->k_prep,1,sizeof sc->pts,&sc->pts));
                CL_CHECK(clSetKernelArg(pr->k_prep,2,sizeof pr->consts,&pr->consts));
                CL_CHECK(enqueue_mining_kernel(d.q, pr->k_prep, prep_gsz, lsz));
            }
            CL_CHECK(clSetKernelArg(pr->k_scalar, 0, sizeof sc->pts, &sc->pts));
            CL_CHECK(clSetKernelArg(pr->k_scalar, 1, sizeof sc->cnt, &sc->cnt));
            CL_CHECK(clSetKernelArg(pr->k_scalar, 2, sizeof sc->projective, &sc->projective));
            CL_CHECK(clSetKernelArg(pr->k_scalar, 3, sizeof pr->consts, &pr->consts));
            CL_CHECK(enqueue_mining_kernel(d.q, pr->k_scalar, gsz, lsz));
            gsz = ((stride + pr->v->split * lsz - 1) / (pr->v->split * lsz)) * lsz;
            CL_CHECK(clSetKernelArg(pr->k_finish, 0, sizeof d_job, &d_job));
            CL_CHECK(clSetKernelArg(pr->k_finish, 1, sizeof sc->projective, &sc->projective));
            CL_CHECK(clSetKernelArg(pr->k_finish, 2, sizeof sc->cnt, &sc->cnt));
            cl_mem finish_hits=pr->v->fast?sc->candidates:d_hits,finish_n=pr->v->fast?sc->ncandidates:d_n;
            if(pr->v->fast)CL_CHECK(clEnqueueFillBuffer(d.q,finish_n,&zero,4,0,4,0,nullptr,nullptr));
            CL_CHECK(clSetKernelArg(pr->k_finish, 3, sizeof finish_hits, &finish_hits));
            CL_CHECK(clSetKernelArg(pr->k_finish, 4, sizeof finish_n, &finish_n));
            CL_CHECK(enqueue_mining_kernel(d.q, pr->k_finish, gsz, lsz));
            if(pr->v->fast) {
                cl_kernel k=pr->k_resolve;
                CL_CHECK(clSetKernelArg(k,0,sizeof d_job,&d_job));
                CL_CHECK(clSetKernelArg(k,1,sizeof sc->candidates,&sc->candidates));
                CL_CHECK(clSetKernelArg(k,2,sizeof sc->ncandidates,&sc->ncandidates));
                CL_CHECK(clSetKernelArg(k,3,sizeof d_hits,&d_hits));
                CL_CHECK(clSetKernelArg(k,4,sizeof d_n,&d_n));
                CL_CHECK(clSetKernelArg(k,5,sizeof pr->consts,&pr->consts));
                size_t resolve_groups=std::min<size_t>(64,(sc->n+255)/256),resolve_size=resolve_groups*256,resolve_local=256;
                CL_CHECK(enqueue_mining_kernel(d.q, k, resolve_size, resolve_local));
            }
        } else {
            size_t gsz = stride, lsz = pr->v->wg;
            clSetKernelArg(pr->k_mult, 0, sizeof d_job, &d_job); clSetKernelArg(pr->k_mult, 1, sizeof sc->pts, &sc->pts);
            clSetKernelArg(pr->k_mult, 2, sizeof sc->cnt, &sc->cnt); clSetKernelArg(pr->k_mult, 3, sizeof d_hits, &d_hits);
            clSetKernelArg(pr->k_mult, 4, sizeof d_n, &d_n); clSetKernelArg(pr->k_mult, 5, sizeof pr->consts, &pr->consts);
            CL_CHECK(enqueue_mining_kernel(d.q, pr->k_mult, gsz, lsz));
        }
    } else {
        size_t gsz = stride, lsz = pr->v->wg;
        clSetKernelArg(pr->k_mine, 0, sizeof d_job, &d_job); clSetKernelArg(pr->k_mine, 1, sizeof nonce_base, &nonce_base);
        clSetKernelArg(pr->k_mine, 2, sizeof stride, &stride); clSetKernelArg(pr->k_mine, 3, sizeof iters, &iters);
        clSetKernelArg(pr->k_mine, 4, sizeof d_hits, &d_hits); clSetKernelArg(pr->k_mine, 5, sizeof d_n, &d_n);
        clSetKernelArg(pr->k_mine, 6, sizeof pr->consts, &pr->consts);
        CL_CHECK(enqueue_mining_kernel(d.q, pr->k_mine, gsz, lsz));
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

// ---- txid-blake2b (library upgrade 6): the hash search kernels of txid_blake2b.cl -------------------
#define TX_WG 256
struct TxProg { cl_program p = nullptr; cl_kernel k[32] = {}; ClDev *dev = nullptr; };
static std::map<cl_device_id, TxProg *> g_txprogs;
static TxProg *get_txprog(ClDev &dev) {
    std::lock_guard<std::mutex> l(g_prog_mu);
    auto it = g_txprogs.find(dev.id); if (it != g_txprogs.end()) return it->second;
    TxProg *tp = new TxProg(); tp->dev = &dev;
    const char *src = (const char *)TXID_CL_SRC; size_t len = TXID_CL_SRC_LEN; cl_int err;
    tp->p = clCreateProgramWithSource(dev.ctx, 1, &src, &len, &err); CL_CHECK(err);
    std::string opts = "-cl-std=CL1.2 -DTX_WG=" + std::to_string(TX_WG) + " -DTX_MAX_HITS=" + std::to_string(TXID_MAX_HITS) + " " + g_extra_build;
    auto t0 = std::chrono::steady_clock::now();
    err = clBuildProgram(tp->p, 1, &dev.id, opts.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t ll = 0; clGetProgramBuildInfo(tp->p, dev.id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &ll);
        std::string log(ll, 0); clGetProgramBuildInfo(tp->p, dev.id, CL_PROGRAM_BUILD_LOG, ll, log.data(), nullptr);
        fprintf(stderr, "[txid-blake2b] build failed on %s:\n%s\n", dev.name.c_str(), log.c_str()); exit(2);
    }
    for (int i = 0; i < 32; i++) { char nm[32]; snprintf(nm, sizeof nm, "txid_k%d", i); tp->k[i] = clCreateKernel(tp->p, nm, &err); CL_CHECK(err); }
    logln("[txid-blake2b] built for " + dev.name + " in " + std::to_string((int)(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1000)) + " ms");
    g_txprogs[dev.id] = tp;
    return tp;
}
static cl_int tx_enqueue(TxProg *tp, cl_mem params, int kernel, uint64_t nbase, uint64_t count, cl_uint iters, cl_mem hits, cl_mem nhit, size_t groups) {
    cl_kernel k = tp->k[kernel & 31]; cl_ulong nb = nbase, cn = count;
    clSetKernelArg(k, 0, sizeof params, &params); clSetKernelArg(k, 1, sizeof nb, &nb); clSetKernelArg(k, 2, sizeof cn, &cn);
    clSetKernelArg(k, 3, sizeof iters, &iters); clSetKernelArg(k, 4, sizeof hits, &hits); clSetKernelArg(k, 5, sizeof nhit, &nhit);
    return enqueue_mining_kernel(tp->dev->q, k, groups * TX_WG, TX_WG);
}
// a synchronous search for the self-check
struct ClTxSearch {
    TxProg *tp; cl_mem params, hits, nhit;
    explicit ClTxSearch(ClDev &d) : tp(get_txprog(d)) {
        cl_int err;
        params = clCreateBuffer(d.ctx, CL_MEM_READ_ONLY, sizeof(txid::Params), nullptr, &err); CL_CHECK(err);
        hits = clCreateBuffer(d.ctx, CL_MEM_READ_WRITE, sizeof(txid::HitRec) * TXID_MAX_HITS, nullptr, &err); CL_CHECK(err);
        nhit = clCreateBuffer(d.ctx, CL_MEM_READ_WRITE, 4, nullptr, &err); CL_CHECK(err);
    }
    ~ClTxSearch() { clReleaseMemObject(params); clReleaseMemObject(hits); clReleaseMemObject(nhit); }
    std::vector<txid::HitRec> operator()(const txid::Params &p, uint64_t nbase, uint64_t count) {
        std::vector<txid::HitRec> out; cl_command_queue q = tp->dev->q; cl_uint zero = 0;
        CL_CHECK(clEnqueueWriteBuffer(q, params, CL_TRUE, 0, sizeof p, &p, 0, nullptr, nullptr));
        CL_CHECK(clEnqueueWriteBuffer(q, nhit, CL_TRUE, 0, 4, &zero, 0, nullptr, nullptr));
        cl_uint iters = 4; size_t groups = (size_t)((count + (uint64_t)TX_WG * iters - 1) / ((uint64_t)TX_WG * iters));
        if (tx_enqueue(tp, params, (int)p.kernel, nbase, count, iters, hits, nhit, groups) != CL_SUCCESS || mining_wait(q) != CL_SUCCESS) { out.push_back(txid::HitRec{~0ULL, 0}); return out; }
        cl_uint n = 0; CL_CHECK(clEnqueueReadBuffer(q, nhit, CL_TRUE, 0, 4, &n, 0, nullptr, nullptr));
        n = std::min<cl_uint>(n, TXID_MAX_HITS); out.resize(n);
        if (n) CL_CHECK(clEnqueueReadBuffer(q, hits, CL_TRUE, 0, sizeof(txid::HitRec) * n, out.data(), 0, nullptr, nullptr));
        return out;
    }
};
static int cl_txid_selfcheck(ClDev &d, std::string &why) { ClTxSearch s(d); return txid::selfcheck(std::ref(s), why); }
// one GPU's hash search (as TxidWorker in the CUDA miner)
struct ClTxidWorker {
    Shared &sh; int ordinal; ClDev &d; TxProg *tp; size_t groups; cl_uint iters = 16;
    txid::Params p; uint64_t job = 0; uint16_t en1 = 0; int k = -1; uint64_t cur = 0, lo = 0, hi = 0; bool exhausted = false, dirty = true;
    std::map<std::pair<uint64_t, uint16_t>, uint64_t> cursors;
    cl_mem params, d_hits, d_n; std::vector<txid::HitRec> hits;
    ClTxidWorker(Shared &s, int ord, ClDev &dev) : sh(s), ordinal(ord), d(dev), tp(get_txprog(dev)), hits(TXID_MAX_HITS) {
        groups = (size_t)dev.cus * 32; if (const char *e = getenv("PX_TXID_GROUPS")) groups = (size_t)atoi(e);
        cl_int err;
        params = clCreateBuffer(d.ctx, CL_MEM_READ_ONLY, sizeof(txid::Params), nullptr, &err); CL_CHECK(err);
        d_hits = clCreateBuffer(d.ctx, CL_MEM_READ_WRITE, sizeof(txid::HitRec) * TXID_MAX_HITS, nullptr, &err); CL_CHECK(err);
        d_n = clCreateBuffer(d.ctx, CL_MEM_READ_WRITE, 4, nullptr, &err); CL_CHECK(err);
    }
    void idle(int ms) { { std::lock_guard<std::mutex> l(sh.mu); sh.gpus[ordinal].win_idle_ms += ms; } std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }
    void step(const Job &j) {
        if (j.has_ttl && std::chrono::steady_clock::now() > j.deadline) { idle(10); return; }
        if (j.id != job || j.en1 != en1) {
            if (job) { cursors[{job, en1}] = cur; if (cursors.size() > 256) cursors.clear(); }
            job = j.id; en1 = j.en1; k = j.k; exhausted = false; dirty = true;
            txid::prepare(j.hprefix.data(), (int)j.hprefix.size(), k, p);
            txid_range(sh, ordinal, lo, hi);
            auto it = cursors.find({job, en1}); cur = it != cursors.end() ? it->second : lo;
        } else if (j.k != k) { k = j.k; p.mask = txid::mask_for_k(k); dirty = true; }
        if (cur >= hi) {
            if (!exhausted) { exhausted = true; log_warn("GPU #" + std::to_string(sh.gpus[ordinal].dev) + ": searched its whole extranonce2 range of job " + j.jid + ", idle until the next job"); }
            idle(10); return;
        }
        auto t0 = std::chrono::steady_clock::now();
        cl_uint zero = 0;
        if (dirty) { CL_CHECK(clEnqueueWriteBuffer(d.q, params, CL_FALSE, 0, sizeof p, &p, 0, nullptr, nullptr)); dirty = false; }
        CL_CHECK(clEnqueueFillBuffer(d.q, d_n, &zero, 4, 0, 4, 0, nullptr, nullptr));
        uint64_t per = (uint64_t)groups * TX_WG * iters, count = std::min<uint64_t>(per, hi - cur);
        CL_CHECK(tx_enqueue(tp, params, (int)p.kernel, ((uint64_t)en1 << 48) | cur, count, iters, d_hits, d_n, groups));
        CL_CHECK(mining_wait(d.q));
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        cur += count;
        if (count == per && ms > 0) { double want = (double)iters * sh.txid_launch_ms / ms; iters = (cl_uint)std::max(1.0, std::min(65536.0, iters * 0.5 + want * 0.5)); }
        sh.attempts += count;
        { std::lock_guard<std::mutex> l(sh.mu); auto &g = sh.gpus[ordinal]; g.win_attempts += count; g.win_kernel_ms += ms; g.win_launches++; g.hash = true; }
        cl_uint n = 0; CL_CHECK(clEnqueueReadBuffer(d.q, d_n, CL_TRUE, 0, 4, &n, 0, nullptr, nullptr));
        if (n) {
            n = std::min<cl_uint>(n, TXID_MAX_HITS);
            CL_CHECK(clEnqueueReadBuffer(d.q, d_hits, CL_TRUE, 0, sizeof(txid::HitRec) * n, hits.data(), 0, nullptr, nullptr));
            txid_found(sh, j, hits.data(), n, ordinal);
        }
    }
};

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
    ClTxidWorker tw(sh, cfg.ordinal, d);
    logln("GPU " + std::to_string(cfg.dev) + " " + dev_label(d) + ": variant " + cfg.v->name + ", " + std::to_string(cfg.groups) + "x" + std::to_string(cfg.v->wg) + " work-items, " + std::to_string(cfg.iters) + " nonces/item/launch");
    while (!sh.stop) {
        if (!thermal_gate(sh, cfg.ordinal)) {
            { std::lock_guard<std::mutex> l(sh.mu); sh.gpus[cfg.ordinal].win_idle_ms += 1000; }
            std::this_thread::sleep_for(std::chrono::seconds(1)); continue;
        }
        Job j; { std::lock_guard<std::mutex> l(sh.mu); j = sh.job; }
        if (j.id == 0 || j.stale) {
            { std::lock_guard<std::mutex> l(sh.mu); sh.gpus[cfg.ordinal].win_idle_ms += 10; }
            std::this_thread::sleep_for(std::chrono::milliseconds(10)); continue;
        }
        if (j.txid) { tw.step(j); continue; }
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
        CL_CHECK(mining_wait(d.q));
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - k0).count();
        uint64_t done = stride * cfg.iters;
        nonce += done;
        sh.attempts += done;
        { std::lock_guard<std::mutex> l(sh.mu); auto &g = sh.gpus[cfg.ordinal]; g.win_attempts += done; g.win_kernel_ms += ms; g.win_launches++; g.hash = false; }
        cl_uint n = 0; CL_CHECK(clEnqueueReadBuffer(d.q, d_n, CL_TRUE, 0, 4, &n, 0, nullptr, nullptr));
        if (n > 0) {
            if (n > MAX_HITS) n = MAX_HITS;
            CL_CHECK(clEnqueueReadBuffer(d.q, d_hits, CL_TRUE, 0, sizeof(Hit) * n, hits.data(), 0, nullptr, nullptr));
            std::sort(hits.begin(), hits.begin() + n, [](const Hit &a, const Hit &b) { return memcmp(a.beta, b.beta, 64) < 0; });
            for (unsigned i = 0; i < n; i++) {
                Job cur; { std::lock_guard<std::mutex> l(sh.mu); cur = sh.job; }
                if (cur.id != j.id || cur.stale) break;
                if (cur.has_beat && memcmp(hits[i].beta, cur.beat, 64) >= 0) continue;
                submit(sh, cur, hits[i].nonce, hits[i].beta, sh.attempts.exchange(0), cfg.ordinal);
            }
        }
    }
    scratch_free(sc); clReleaseMemObject(d_job); clReleaseMemObject(d_hits); clReleaseMemObject(d_n);
}

// Replay an independent vector through the actual compacted mining pipeline.
// The search intentionally retains its 12 hash-to-curve passes; later decodes
// must produce no hit rather than being compared to the unbounded test kernel.
static bool pipeline_vector_ok(Prog *pr, JobDev jd, uint64_t nonce,
                               uint64_t counter, const std::string &beta) {
    if (!pr->v->compact) return true;
    ClDev &d = *pr->dev; cl_int err;
    jd.k = 0; memset(jd.beat, 255, sizeof jd.beat);
    cl_mem job = make_job(d, jd);
    cl_mem hits = clCreateBuffer(d.ctx, CL_MEM_READ_WRITE, sizeof(Hit) * MAX_HITS, nullptr, &err); CL_CHECK(err);
    cl_mem nhits = clCreateBuffer(d.ctx, CL_MEM_READ_WRITE, sizeof(cl_uint), nullptr, &err); CL_CHECK(err);
    Scratch *sc = scratch_alloc(pr, 1);
    cl_uint n = 0; CL_CHECK(clEnqueueWriteBuffer(d.q, nhits, CL_TRUE, 0, sizeof n, &n, 0, nullptr, nullptr));
    prog_mine(pr, job, nonce, pr->v->wg, 1, hits, nhits, sc);
    CL_CHECK(mining_wait(d.q));
    CL_CHECK(clEnqueueReadBuffer(d.q, nhits, CL_TRUE, 0, sizeof n, &n, 0, nullptr, nullptr));
    bool ok = n == (counter < PX_MAX_PASSES ? 1u : 0u);
    if (ok && n) {
        Hit hit; CL_CHECK(clEnqueueReadBuffer(d.q, hits, CL_TRUE, 0, sizeof hit, &hit, 0, nullptr, nullptr));
        ok = hit.nonce == nonce && hexs(hit.beta, sizeof hit.beta) == beta;
    }
    scratch_free(sc); clReleaseMemObject(job); clReleaseMemObject(hits); clReleaseMemObject(nhits);
    return ok;
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
        if (x != cur_x) {
            uint8_t pk2[32];
            if (prog_init(pr, xb.data(), pk2) != 1 || memcmp(pk2, pkb.data(), 32)) {
                fprintf(stderr, "[%s] vector public-key derivation failed\n", pr->v->name); bad++; continue;
            }
            cur_x = x;
        }
        JobDev jd; memset(&jd, 0, sizeof jd);
        memcpy(jd.pk, pkb.data(), 32); memcpy(jd.prefix, ab.data(), 37); jd.prefix_len = 37;
        uint64_t nonce = 0; for (int i = 0; i < 8; i++) nonce = (nonce << 8) | ab[37 + i];
        cl_mem d_job = make_job(*pr->dev, jd);
        uint8_t ob[64], oh[32], ocg[32];
        int ok = prog_test(pr, d_job, nonce, ob, oh, ocg);
        clReleaseMemObject(d_job);
        if (pr->v->split && !pipeline_vector_ok(pr, jd, nonce, ctr, beta)) {
            fprintf(stderr, "[%s] mining pipeline vector %d FAILED\n", pr->v->name, n + 1); bad++;
        }
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
    return bad || n == 0 ? 1 : 0;
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
    CL_CHECK(clEnqueueFillBuffer(d.q, d_n, &zero, 4, 0, 4, 0, nullptr, nullptr)); prog_mine(pr, d_job, nonce, stride, iters, d_hits, d_n, sc); CL_CHECK(mining_wait(d.q));
    auto t0 = std::chrono::steady_clock::now();
    std::clock_t cpu0 = std::clock();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < seconds) {
        CL_CHECK(clEnqueueFillBuffer(d.q, d_n, &zero, 4, 0, 4, 0, nullptr, nullptr));
        prog_mine(pr, d_job, nonce, stride, iters, d_hits, d_n, sc);
        CL_CHECK(mining_wait(d.q));
        nonce += stride * iters; total += stride * iters;
    }
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    double mhs = total / s / 1e6;
    if(verbose) fprintf(stderr,"BENCH_CPU %.3f percent of one core, %.6f seconds\n",100.0*(std::clock()-cpu0)/CLOCKS_PER_SEC/s,s);
    if (verbose) fprintf(stderr, "GPU %s [%s %dx%d iters %d]: %.6f MH/s\n", dev_label(d).c_str(), pr->v->name, groups, pr->v->wg, iters, mhs);
    scratch_free(sc); clReleaseMemObject(d_job); clReleaseMemObject(d_hits); clReleaseMemObject(d_n);
    return mhs;
}
static GpuCfg autotune(int dev, const uint8_t scalar[32], const uint8_t pk[32], double seconds, int fixedGroups, bool quick = false) {
    ClDev &d = g_devs[dev];
    const Platform &pl = platformFor(d);
    GpuCfg best{dev, fixedGroups, pl.iters, variantByName(pl.variant)};
    double bestRate = 0;
    fprintf(stderr, "autotune GPU %d %s, %.0fs per trial:\n", dev, dev_label(d).c_str(), seconds);
    for (int v = 0; v < NVARIANTS; v++) {
        if (quick && !VARIANTS[v].split && strcmp(VARIANTS[v].name, "fe29_wnaf_c") != 0 && strcmp(VARIANTS[v].name, "fe32_wnaf_c") != 0) continue;
        Prog *pr = get_prog(d, &VARIANTS[v]);
        uint8_t pk2[32];
        if (!prog_validated(pr)) { fprintf(stderr, "  %-12s: fails the test vectors here, skipped\n", VARIANTS[v].name); continue; }
        if (prog_init(pr, scalar, pk2) != 1 || memcmp(pk2, pk, 32) != 0) { fprintf(stderr, "  %-12s: init failed, skipped\n", VARIANTS[v].name); continue; }
        // The split/ladder kernels need wide one-iteration launches. The old
        // search only tried 8 groups/CU with 2, 4 or 8 iterations and could
        // never measure their production launch shape.
        const int splitGeo[][2] = {{128,1},{256,1},{64,1}};
        const int otherGeo[][2] = {{8,8},{8,4},{8,2},{64,1}};
        const int (*geo)[2] = VARIANTS[v].split ? splitGeo : otherGeo;
        int ngeo = VARIANTS[v].split ? 3 : quick ? 2 : 4;
        for (int gi = 0; gi < ngeo; gi++) {
            if (fixedGroups > 0 && VARIANTS[v].split && gi > 0) continue;
            int groups = fixedGroups > 0 ? fixedGroups : (int)d.cus * geo[gi][0];
            int it = geo[gi][1];
            double r = run_bench(pr, pk, groups, it, seconds, false);
            fprintf(stderr, "  %-18s %dx%dx%d: %.3f MH/s%s\n", VARIANTS[v].name, groups, VARIANTS[v].wg, it, r, r > bestRate ? "  <- best so far" : "");
            if (r > bestRate) { bestRate = r; best = GpuCfg{dev, groups, it, &VARIANTS[v]}; }
        }
    }
    fprintf(stderr, "autotune GPU %d: %s %dx%dx%d (%.3f MH/s)\n", dev, best.v->name, best.groups, best.v->wg, best.iters, bestRate);
    return best;
}
// tuning cache shared in format with the CUDA miner: "<name>|<family>|<version>|<variant> <groups> <wg> <iters>"
static const char *TUNE_CACHE = "proxichorizo-autotune.cache";
static std::string tuneKey(const ClDev &d) { return dev_label(d) + "|" + d.name + "|cu" + std::to_string(d.cus) + "|" + d.tuning_id + "|driver" + d.driver + "|" PX_VERSION "|"; }
static bool loadTuned(const ClDev &d, int dev, GpuCfg &cfg) {
    if (d.tuning_id.empty()) return false;
    std::ifstream in(TUNE_CACHE);
    std::string line, key = tuneKey(d);
    while (std::getline(in, line)) {
        if (line.compare(0, key.size(), key) != 0) continue;
        std::istringstream ss(line.substr(key.size()));
        std::string v; int g = 0, wg = 0, it = 0;
        if (!(ss >> v >> g >> wg >> it) || it <= 0) return false;
        const Variant *vp = variantByName(v);
        if (!vp) return false;
        cfg = GpuCfg{dev, g, it, vp};
        return true;
    }
    return false;
}
static void saveTuned(const ClDev &d, const GpuCfg &cfg) {
    if (d.tuning_id.empty()) { logln("no persistent GPU identity: tuning result is not cached"); return; }
    std::ofstream out(TUNE_CACHE, std::ios::app);
    if (!out) { logln(std::string("cannot write ") + TUNE_CACHE + ": the result is not kept for the next start"); return; }
    out << tuneKey(d) << cfg.v->name << ' ' << cfg.groups << ' ' << cfg.v->wg << ' ' << cfg.iters << '\n';
}
// every variant over the same nonce range at low difficulty must find exactly the same hits
#include "pipeline_verify.inc"

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
            prog_mine(pr, d_job, nonce, stride, iters, d_hits, d_n, sc); CL_CHECK(mining_wait(d.q));
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
        "  --proxi URL        the pool: stratum+tcp://HOST:PORT or stratum+ssl://HOST:PORT (stratum), or the HTTP job server of `proxi node mine`\n"
        "                     and older pools (http://HOST:PORT); -o and --url are the same option\n"
        "  --user ADDR.WORKER stratum login (-u): your payout address and worker name, or just the worker name;\n"
        "                     default: the --name\n"
        "  --ca-file FILE     CA certificates for stratum+ssl:// besides the system store (PEM)\n"
        "  --tls-insecure     stratum+ssl:// without checking the pool's certificate (private pools only)\n"
        "  --check-pool URL   connect to a stratum pool, subscribe, print the answer and exit (no GPU needed)\n"
        "  --token T          the account token from the pool registration (stratum password; --pass and -p are the same)\n"
        "  --key-file FILE    unencrypted proxima.key (or --seed-hex HEX / NONCE_SEEKER_SEED)\n"
        "  --seed-hex HEX     32-byte seed\n"
        "  --name NAME        seeker name in the pool's statistics (default proxichorizo-<pid>)\n"
        "  --api-bind [H:]P   stats API, JSON at http://H:P/ and the mmpOS line at /mmpos (default off)\n"
        "  -d 0,1,2,3         GPUs to use; without it every GPU mines in one process (--list-devices shows them; --gpus/--devices are the same)\n"
        "  --variant NAME     kernel variant for all GPUs (default: per-GPU table; --list-variants)\n"
        "  --autotune [SECS]  measure every variant and iteration count on each GPU first (default 6s per trial)\n"
        "  --groups N --iters N   work-groups per launch (default 8 per compute unit) and nonces per work-item\n"
        "  --kernel-file F    read the OpenCL source from F instead of the embedded copy; --build-opts \"...\" extra compiler flags\n"
        "  --selftest [FILE]  verify every variant against the embedded test vectors (or FILE) and exit\n"
        "  --fetest           randomized field-arithmetic test of every variant and exit\n"
        "  --verify-pipeline  complete compacted-pipeline output checks, including partial batches; exit\n"
        "  --verify           every variant over one nonce range at low difficulty must find identical hits; exit\n"
        "%s"
        "  --bench [SECS]     measure the hashrate of the chosen variant and exit\n"
        "txid-blake2b hash work (Proxima library upgrade 6, from slot 197000; needs no key, so --key-file is optional with a stratum pool):\n"
        "  --hash-selftest    check the hash kernels on every selected GPU against the upgrade-6 vectors and the CPU, then exit\n"
        "  --hash-bench [SECS] measure the hash rate (GH/s) on every selected GPU and exit (--hash-prefix-len N, default 600)\n"
        "  --hash-launch-ms N target length of one hash launch (default 40)\n", CONSOLE_USAGE);
}

// -d 0,1,2,3 (also "all"): device indices, each once, within the device count
static bool parse_devices(const std::string &gpus, int ndev, std::vector<int> &devs) {
    devs.clear();
    if (gpus.empty() || gpus == "all") { for (int i = 0; i < ndev; i++) devs.push_back(i); return true; }
    std::stringstream ss(gpus); std::string t;
    while (std::getline(ss, t, ',')) {
        while (!t.empty() && t.front() == ' ') t.erase(0, 1);
        while (!t.empty() && t.back() == ' ') t.pop_back();
        if (t.empty()) continue;
        char *end = nullptr; long d = strtol(t.c_str(), &end, 10);
        if (*end || d < 0 || d >= ndev) { fprintf(stderr, "-d %s: no GPU %s (this machine has %d, numbered 0..%d; --list-devices)\n", gpus.c_str(), t.c_str(), ndev, ndev - 1); return false; }
        if (std::find(devs.begin(), devs.end(), (int)d) != devs.end()) continue;
        devs.push_back((int)d);
    }
    if (devs.empty()) { fprintf(stderr, "-d %s selects no GPU\n", gpus.c_str()); return false; }
    return true;
}

// Ctrl-C / SIGTERM: one summary line, then out
static Shared *g_sh = nullptr;
static void on_signal(int) {
    if (g_sh) {
        char b[200];
        snprintf(b, sizeof b, "\n%s stopped: accepted %llu, rejected %llu, stale %llu, uptime %s\n", now_str().c_str(), (unsigned long long)g_sh->shares.load(),
                 (unsigned long long)g_sh->rejected.load(), (unsigned long long)g_sh->stale.load(), fmt_uptime(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - g_sh->started).count()).c_str());
        fputs(b, stderr); fflush(stderr);
    }
    _Exit(0);
}

int main(int argc, char **argv) {
    std::string proxi, token, keyfile, seedhex, name, selftest, gpus, variant, apiBind, kernelFile, user, checkPool;
    int groups = 0, iters = 0; double bench = 0, autotuneSecs = 0; bool fetest = false, listVariants = false, listDevices = false, verify = false, verifyPipeline = false;
    double txidBench = 0; bool txidSelftest = false; int txidPrefix = 600, txidLaunchMs = 40;
    ConsoleOpts con;
    sock_init();
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](std::string &dst) { if (i + 1 < argc) dst = argv[++i]; };
        auto optnum = [&](double dflt) { if (i + 1 < argc && isdigit((unsigned char)argv[i + 1][0])) return atof(argv[++i]); return dflt; };
        if (a == "--help" || a == "-h") { usage(); return 0; }
        else if (a == "--proxi" || a == "-o" || a == "--url") next(proxi);
        else if (a == "--user" || a == "-u") next(user);
        else if (a == "--ca-file") next(tls::ca_file);
        else if (a == "--tls-insecure") tls::insecure = true;
        else if (a == "--check-pool") next(checkPool);
        else if (a == "--token" || a == "--pass" || a == "-p") next(token);
        else if (a == "--key-file") next(keyfile);
        else if (a == "--seed-hex") next(seedhex);
        else if (a == "--name") next(name);
        else if (a == "--api-bind" || a == "--api-port") next(apiBind);
        else if (a == "--gpus" || a == "-d" || a == "--devices") next(gpus);
        else if (a == "--variant") next(variant);
        else if (a == "--kernel-file") next(kernelFile);
        else if (a == "--build-opts") next(g_extra_build);
        else if (a == "--selftest") { selftest = "embedded"; if (i + 1 < argc && argv[i + 1][0] != '-') selftest = argv[++i]; }
        else if (a == "--fetest") fetest = true;
        else if (a == "--verify") verify = true;
        else if (a == "--verify-pipeline") verifyPipeline = true;
        else if (a == "--list-variants") listVariants = true;
        else if (a == "--list-devices") listDevices = true;
        else if (a == "--groups" || a == "--blocks") { std::string s; next(s); groups = atoi(s.c_str()); }
        else if (a == "--threads") { std::string s; next(s); }  // accepted for CUDA command-line compatibility; the work-group size is per variant
        else if (a == "--iters") { std::string s; next(s); iters = atoi(s.c_str()); }
        else if (a == "--bench") bench = optnum(20);
        else if (a == "--autotune") autotuneSecs = optnum(6);
        else if (a == "--hash-bench") txidBench = optnum(20);
        else if (a == "--hash-selftest") txidSelftest = true;
        else if (a == "--hash-prefix-len") { std::string s; next(s); txidPrefix = atoi(s.c_str()); }
        else if (a == "--hash-launch-ms") { std::string s; next(s); txidLaunchMs = std::max(5, std::min(1000, atoi(s.c_str()))); }
        else if (parse_console_opt(a, i, argc, argv, con)) {}
        else { fprintf(stderr, "unknown option %s\n\n", a.c_str()); usage(); return 1; }
    }
    console_setup(con);
    if (!checkPool.empty()) return check_pool(checkPool);
    if (listVariants) { for (int v = 0; v < NVARIANTS; v++) printf("%s\n", VARIANTS[v].name); return 0; }
    if (!kernelFile.empty()) { std::ifstream f(kernelFile); if (!f) { fprintf(stderr, "cannot open %s\n", kernelFile.c_str()); return 1; } std::stringstream ss; ss << f.rdbuf(); g_kernel_src = ss.str(); }
    else g_kernel_src.assign((const char *)CL_SRC, CL_SRC_LEN);
    enumerate_devices();
    if (g_devs.empty()) { fprintf(stderr, "no OpenCL GPU found (is the OpenCL ICD installed? ROCm: /etc/OpenCL/vendors)\n"); return 2; }
    if (listDevices) { for (size_t i = 0; i < g_devs.size(); i++) printf("%zu: %s, %u CUs, bus %d, platform %s\n", i, dev_label(g_devs[i]).c_str(), g_devs[i].cus, g_devs[i].bus, g_devs[i].platname.c_str()); return 0; }
    std::vector<int> devs;
    if (!parse_devices(gpus, (int)g_devs.size(), devs)) return 1;
    // txid-blake2b (hash work, library upgrade 6): the self-check on every selected GPU, also before mining
    auto hash_selfcheck = [&]() {
        int rc = 0;
        for (int dv : devs) {
            std::string why; auto t0 = std::chrono::steady_clock::now(); int r = cl_txid_selfcheck(g_devs[dv], why);
            fprintf(stderr, "GPU %d %s: txid-blake2b self-check %s: %s (%.1f s)\n", dv, dev_label(g_devs[dv]).c_str(), r ? "FAILED" : "passed", why.c_str(), std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            rc |= r;
        }
        return rc;
    };
    if (txidSelftest) return hash_selfcheck();
    if (txidBench > 0) {
        if (hash_selfcheck()) return 1;
        for (int dv : devs) {
            ClDev &d = g_devs[dv]; TxProg *tp = get_txprog(d); cl_int err;
            std::vector<uint8_t> prefix((size_t)std::max(0, txidPrefix)); for (size_t i = 0; i < prefix.size(); i++) prefix[i] = (uint8_t)(i * 31 + 7);
            txid::Params p; txid::Shape shp = txid::prepare(prefix.data(), (int)prefix.size(), 40, p);
            cl_mem params = clCreateBuffer(d.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof p, &p, &err); CL_CHECK(err);
            cl_mem hb = clCreateBuffer(d.ctx, CL_MEM_READ_WRITE, sizeof(txid::HitRec) * TXID_MAX_HITS, nullptr, &err); CL_CHECK(err);
            cl_uint zero = 0; cl_mem nb = clCreateBuffer(d.ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 4, &zero, &err); CL_CHECK(err);
            size_t grp = (size_t)d.cus * 32; if (const char *e = getenv("PX_TXID_GROUPS")) grp = (size_t)atoi(e);
            cl_uint it = 8; uint64_t n = 0x0102000000000000ULL, total = 0, lastTotal = 0; int launches = 0;
            auto t0 = std::chrono::steady_clock::now(), last = t0;
            char bus[32]; snprintf(bus, sizeof bus, "0000:%02x:00.0", d.bus);
            while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < txidBench) {
                uint64_t per = (uint64_t)grp * TX_WG * it; auto k0 = std::chrono::steady_clock::now();
                CL_CHECK(tx_enqueue(tp, params, (int)p.kernel, n, per, it, hb, nb, grp)); CL_CHECK(mining_wait(d.q));
                double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - k0).count();
                n += per; total += per; launches++;
                if (ms > 0) it = (cl_uint)std::max(1.0, std::min(65536.0, it * 0.5 + it * (double)txidLaunchMs / ms * 0.5));
                auto now = std::chrono::steady_clock::now();
                if (now - last >= std::chrono::seconds(5)) {
                    fprintf(stderr, "GPU %d %s: %.3f GH/s, %s\n", dv, dev_label(d).c_str(), (total - lastTotal) / std::chrono::duration<double>(now - last).count() / 1e9, sensors_str(read_sensors(bus)).c_str());
                    last = now; lastTotal = total;
                }
            }
            double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            fprintf(stderr, "GPU %d %s: txid-blake2b %d-byte prefix (kernel %d): %.3f GH/s over %.0f s (%d launches, %zux%dx%u)\n", dv, dev_label(d).c_str(), txidPrefix, shp.kernel, total / el / 1e9, el, launches, grp, TX_WG, it);
            clReleaseMemObject(params); clReleaseMemObject(hb); clReleaseMemObject(nb);
        }
        return 0;
    }

    const Variant *forced = nullptr;
    if (!variant.empty()) { forced = variantByName(variant); if (!forced) { fprintf(stderr, "unknown variant %s (--list-variants)\n", variant.c_str()); return 1; } }

    // key -> clamped scalar (SHA-512 of the seed), as the Go/Rust seekers do
    uint8_t seed[32], scalar[32], pk[32];
    bool keyGiven = !seedhex.empty() || !keyfile.empty() || getenv("NONCE_SEEKER_SEED") != nullptr;
    if (keyGiven) { if (!load_seed(seedhex, keyfile, seed)) return 1; }
    else { for (int i = 0; i < 32; i++) seed[i] = (uint8_t)(i * 97 + 13); }  // tests/bench without a key: fixed pseudo-random seed
    { uint8_t h[64]; host_sha512(seed, 32, h); memcpy(scalar, h, 32); scalar[0] &= 248; scalar[31] &= 127; scalar[31] |= 64; }

    if (!selftest.empty() || fetest || verify || verifyPipeline) {
        int rc = 0; ClDev &d = g_devs[devs[0]];
        for (int v = 0; v < NVARIANTS; v++) {
            if (forced && &VARIANTS[v] != forced) continue;
            if (verifyPipeline && selftest.empty() && !fetest && !verify && !VARIANTS[v].split && strcmp(VARIANTS[v].name, "fe29_wnaf_c")) continue;
            Prog *pr = get_prog(d, &VARIANTS[v]);
            uint8_t pk2[32]; int ok = prog_init(pr, scalar, pk2);
            if (ok != 1) { fprintf(stderr, "[%s] constant derivation FAILED (%s) %s\n", VARIANTS[v].name, ok == -2 ? "fe_sq != fe_mul" : ok == -3 ? "sqrt(-1)^2 != -1, got" : ok == -4 ? "basepoint decode, y =" : "?", hexs(pk2, 32).c_str()); rc = 1; }
            if (ok == 1) memcpy(pk, pk2, sizeof pk);
            if (ok == 1 && !selftest.empty()) rc |= run_selftest(pr, selftest);
            if (fetest) {
                int worst = 0;
                for (int sd = 1; sd <= 16 && !worst; sd++) { uint8_t dump[128]; int fail = prog_fetest(pr, sd, 2000, dump); if (fail) { worst = fail; fprintf(stderr, "[%s] fetest FAIL code %d at iter %d\n x %s\n y %s\n a %s\n b %s\n", VARIANTS[v].name, fail / 1000000, fail % 1000000, hexs(dump, 32).c_str(), hexs(dump + 32, 32).c_str(), hexs(dump + 64, 32).c_str(), hexs(dump + 96, 32).c_str()); } }
                if (!worst) fprintf(stderr, "[%s] fetest: 32000 random values ok\n", VARIANTS[v].name); else rc = 1;
            }
        }
        if (verifyPipeline) rc |= run_pipeline_edges(scalar, pk, devs[0], forced);
        if (verify) rc |= run_verify(scalar, pk, devs[0]);
        return rc;
    }

    std::vector<GpuCfg> cfgs; std::vector<Shared::GpuStat> gstats;
    for (int dv : devs) {
        ClDev &d = g_devs[dv];
        const Platform &pl = platformFor(d);
        GpuCfg cfg{dv, groups > 0 ? groups : (int)d.cus * pl.groups_per_cu, iters > 0 ? iters : pl.iters, forced ? forced : variantByName(pl.variant)};
        cfg.ordinal = (int)cfgs.size();
        Prog *pr = get_prog(d, cfg.v);
        if (!prog_validated(pr)) {
            if (forced) { fprintf(stderr, "device %d: the forced variant %s is wrong on this driver; refusing to mine with it\n", dv, cfg.v->name); return 2; }
            // Try the established fe29 path before the portable fe32 fallback.
            if (cfg.v->split) {
                cfg.v = variantByName("fe29_wnaf_c"); pr = get_prog(d, cfg.v);
                cfg.groups = groups > 0 ? groups : (int)d.cus * 8;
                cfg.iters = iters > 0 ? iters : 8;
            }
            if (!prog_validated(pr)) { cfg.v = variantByName("fe32_wnaf_c"); pr = get_prog(d, cfg.v); }
            if (!prog_validated(pr)) { fprintf(stderr, "device %d: no variant reproduces the test vectors\n", dv); return 2; }
        }
        uint8_t pk2[32]; int ok = prog_init(pr, scalar, pk2);
        if (ok != 1) { fprintf(stderr, "device %d: constant derivation failed on %s\n", dv, cfg.v->name); return 2; }
        if (dv == devs[0]) memcpy(pk, pk2, 32); else if (memcmp(pk, pk2, 32) != 0) { fprintf(stderr, "device %d derives a different public key\n", dv); return 2; }
        if (autotuneSecs > 0) { cfg = autotune(dv, scalar, pk, autotuneSecs, groups); if (iters > 0) cfg.iters = iters; cfg.ordinal = (int)cfgs.size(); prog_init(get_prog(d, cfg.v), scalar, pk2); }
        else if (!prog_validated(pr)) return 2;
        else if (!pl.measured && !forced && groups <= 0 && iters <= 0 && bench <= 0) {
            GpuCfg tuned;
            if (loadTuned(d, dv, tuned) && prog_validated(get_prog(d, tuned.v))) {
                cfg = tuned; if (cfg.groups <= 0) cfg.groups = (int)d.cus * 8; cfg.ordinal = (int)cfgs.size(); prog_init(get_prog(d, cfg.v), scalar, pk2);
                fprintf(stderr, "GPU %d %s: %s x%d — tuned on this card earlier (%s)\n", dv, dev_label(d).c_str(), cfg.v->name, cfg.iters, TUNE_CACHE);
            } else {
                fprintf(stderr, "GPU %d %s: %s; this build has no measurements for this model, measuring candidate kernels and launch sizes now (once; kept in %s)\n", dv, dev_label(d).c_str(), pl.note, TUNE_CACHE);
                cfg = autotune(dv, scalar, pk, 3, groups, true); cfg.ordinal = (int)cfgs.size(); prog_init(get_prog(d, cfg.v), scalar, pk2);
                saveTuned(d, cfg);
            }
        }
        else fprintf(stderr, "GPU %d %s: %s x%d — %s\n", dv, dev_label(d).c_str(), cfg.v->name, cfg.iters, pl.note);
        char bus[32]; snprintf(bus, sizeof bus, "0000:%02x:00.0", d.bus);
        gstats.push_back(Shared::GpuStat{dv, cfg.ordinal, dev_label(d), bus, d.bus, 0.0});
        { auto &g = gstats.back(); g.arch = std::to_string(d.cus) + " CUs"; g.variant = cfg.v->name; g.geometry = std::to_string(cfg.groups) + "x" + std::to_string(cfg.v->wg) + "x" + std::to_string(cfg.iters); }
        cfgs.push_back(cfg);
    }
    // Every GPU must own a distinct ordinal (its nonce range under the pool's base and its stats row).
    for (size_t i = 0; i < cfgs.size(); i++) {
        if (cfgs[i].ordinal != (int)i) {
            fprintf(stderr, "internal: GPU %d had ordinal %d, using %zu\n", cfgs[i].dev, cfgs[i].ordinal, i);
            cfgs[i].ordinal = (int)i;
            if (i < gstats.size()) gstats[i].ordinal = (int)i;
        }
    }
    if (bench > 0) { for (auto &c : cfgs) run_bench(get_prog(g_devs[c.dev], c.v), pk, c.groups, c.iters, bench, true); return 0; }
    if (proxi.empty()) { if (autotuneSecs > 0) return 0; usage(); return 1; }

    if (hash_selfcheck()) { fprintf(stderr, "a GPU failed the txid-blake2b self-check: refusing to mine (hash shares would be wrong)\n"); return 2; }
    Shared sh;
    sh.base_url = proxi; if (!parse_url(proxi, sh.url)) return 1;
    sh.token = token; sh.name = name.empty() ? "proxichorizo-" + std::to_string(getpid()) : name;
    sh.user = user.empty() ? sh.name : user;
    memcpy(sh.pk, pk, 32); memcpy(sh.scalar, scalar, 32);
    sh.gpus = gstats; sh.api_bind = apiBind; sh.con = con;
    sh.have_key = keyGiven; sh.txid_launch_ms = txidLaunchMs;
    if (!keyGiven && !sh.url.stratum) { fprintf(stderr, "the HTTP job server of `proxi node mine` sends VRF jobs, which need --key-file\n"); return 1; }
    print_banner(sh, "OpenCL");
    if (!keyGiven) log_warn("no --key-file: mining txid-blake2b hash jobs only (library upgrade 6, from slot 197000); VRF jobs before it need the key");
    logln("ProxiChorizo " PX_VERSION " (OpenCL): key " + hexs(pk, 32) + ", " + (sh.url.stratum ? "stratum " : "job server ") + sh.url.host + ":" + std::to_string(sh.url.port) + ", " + std::to_string(devs.size()) + " GPU(s), " + (sh.url.stratum ? "user " + sh.user : "name " + sh.name));
    for (auto &g : sh.gpus) if (read_sensors(g.bus).temp < 0) log_warn(gpu_tag(g) + ": no temperature sensor readable (hwmon), the thermal limit is not enforced on it");
    g_sh = &sh; signal(SIGINT, on_signal); signal(SIGTERM, on_signal);
    std::vector<std::thread> th;
    if (sh.url.stratum) th.emplace_back(stratum_client, std::ref(sh));
    else { th.emplace_back(poller, std::ref(sh)); th.emplace_back(reporter, std::ref(sh)); }
    th.emplace_back(console, std::ref(sh));
    if (!apiBind.empty()) th.emplace_back(api_server, std::ref(sh));
    for (auto &c : cfgs) th.emplace_back(gpu_worker, std::ref(sh), c);
    for (auto &t : th) t.join();
    return 0;
}
