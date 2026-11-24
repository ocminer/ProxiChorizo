// ProxiChorizo - Copyright (c) 2026 suprnova.cc. ProxiChorizo License 1.0; see LICENSE.
#pragma once
#include "txid_aligned.cuh"
#include "txid_zero_tail.cuh"
#include <array>
#include <cstdio>

namespace txid {
// Written during the startup checks, before any worker threads are created.
inline bool bw_verified[128] = {};
static inline bool bw_device(const cudaDeviceProp &p) {
#ifdef PX_BW_PTX
    if (p.major != 12 || p.minor != 0) return false;
    if (const char *e = getenv("PX_TXID_BW")) {
        if (!strcmp(e, "0")) return false;
        if (!strcmp(e, "1")) return true; // explicit diagnostic override, still self-checked
    }
    return !strcmp(p.name, "NVIDIA GeForce RTX 5090") || !strcmp(p.name, "GeForce RTX 5090") ||
           !strcmp(p.name, "NVIDIA GeForce RTX 5080") || !strcmp(p.name, "GeForce RTX 5080") ||
           !strcmp(p.name, "NVIDIA GeForce RTX 5070 Ti") || !strcmp(p.name, "GeForce RTX 5070 Ti");
#else
    (void)p; return false;
#endif
}
static inline int bw_selfcheck(const cudaDeviceProp &p, std::string &why) {
    if (!bw_device(p)) { why.clear(); return 0; }
    int dev = -1; cudaGetDevice(&dev);
    if (dev < 0 || dev >= 128) { why = "Blackwell device index unsupported"; return 1; }
    bw_verified[dev] = false;
    fprintf(stderr, "GPU %d: checking Blackwell kernels (the first run may compile them for this driver)\n", dev);
    int rc = aligned_selfcheck(why, true, true);
    if (rc) {
        // An older driver may not understand this PTX. Keep its proven native
        // kernel; actual calculation or execution failures must stop startup.
        if (why == cudaGetErrorString(cudaErrorUnsupportedPtxVersion) ||
            why == cudaGetErrorString(cudaErrorNoKernelImageForDevice) ||
            why == cudaGetErrorString(cudaErrorJitCompilerNotFound) ||
            why == cudaGetErrorString(cudaErrorJitCompilationDisabled)) {
            cudaGetLastError();
            why = "Blackwell PTX unavailable on this driver; native kernel retained";
            fprintf(stderr, "GPU %d: %s\n", dev, why.c_str());
            return 0;
        }
        return rc;
    }
    bw_verified[dev] = true;
    why = "Blackwell PTX " + why;
    return 0;
}

// A choice is fixed for the lifetime of a job and cached by message layout.
// 0 = ordinary native, 1 = aligned native, 2 = independently checked PTX.
// Calibration never contributes to reported hashes or submits shares.
struct BlackwellChoice {
    std::array<int,128> choices;
    // A padded tail and an ordinary tail at offset 120 run different PTX kernels.
    int padded_generic = -1;
    bool enabled = false, legacy_device = false, legacy_all = false, wide_zero = false;
    int device = -1, platform = 0, zero_choice = -1;
    explicit BlackwellChoice(const cudaDeviceProp &p) {
        choices.fill(-1); cudaGetDevice(&device);
        enabled = bw_device(p) && device >= 0 && device < 128 && bw_verified[device];
        legacy_device = aligned_device(p); legacy_all = aligned_all_layouts(p);
        platform = platform_profile(p);
        wide_zero=p.major==6&&p.minor==1;
    }
    int select_generic(const Params &input, int plen, unsigned grid, HitRec *hits, unsigned *nhit,
                       bool padded, bool compare_zero_tail, std::string &why) {
        if (platform && plen > 0 && plen % 128 == 113) return 2 + platform;
        const int native = legacy_device && aligned_shape(plen, legacy_all) ? 1 : 0;
        if (!enabled || !aligned_placeable(plen)) return native;
        int &chosen = padded ? padded_generic : choices[plen % 128];
        if (chosen >= 0) return chosen;
        if (const char *e = getenv("PX_TXID_BW")) if (!strcmp(e, "1")) return chosen = 2;
        Params p = input; p.mask = ~0ULL;
        const uint64_t stride = (uint64_t)grid * TXID_TPB;
        if (!stride || stride > (1ULL << 30)) return chosen = native;
        const uint32_t iters = (uint32_t)std::min<uint64_t>(256, (1ULL << 31) / stride);
        const uint64_t count = stride * iters, first = 0xa1b2000000000000ULL;
        cudaEvent_t begin = nullptr, end = nullptr;
        cudaError_t err = cudaEventCreate(&begin);
        if (err == cudaSuccess) err = cudaEventCreateWithFlags(&end, cudaEventBlockingSync);
        // Both offset-120 input classes get their own reversed comparison.
        // Use four short timing windows per contender across its two arms.
        const float window_ms = plen % 128 == 120 ? 32.f : 75.f;
        // On a padded job both ordinary kernels can hash the same exact bytes.
        // Time the ordinary PTX body explicitly before caching a choice for
        // unpadded offset-120 jobs. Its padded specialization has a different cost.
        constexpr int ordinary_ptx = 8;
        const bool measure_ordinary = compare_zero_tail && choices[120] < 0;
        auto sample = [&](int mode, float &ms) {
            if (err != cudaSuccess) return;
            AlignedSearch cache;
            cache.fast = mode == 2 || mode == ordinary_ptx;
            cache.allow_bw_tail = mode != ordinary_ptx;
            // Let each kernel reach its own power/clock operating point before
            // timing it. A cold, few-millisecond sample can pick a slower path.
            for (int phase = 0; phase < 2; ++phase) {
                float total_ms = 0; unsigned samples = 0;
                do {
                    err = cudaMemset(nhit, 0, sizeof(unsigned));
                    if (err == cudaSuccess) err = cudaEventRecord(begin);
                    if (err != cudaSuccess) return;
                    if (mode == 6) launch_zero_tail(grid, TXID_TPB, 0, p, first, count, iters, hits, nhit, 256);
                    else if (mode) launch_aligned(grid, TXID_TPB, 0, p, plen, first, count, hits, nhit, cache);
                    else launch(grid, TXID_TPB, 0, p, first, count, iters, hits, nhit);
                    err = cudaGetLastError();
                    if (err == cudaSuccess) err = cudaEventRecord(end);
                    if (err == cudaSuccess) err = cudaEventSynchronize(end);
                    float elapsed = 0;
                    if (err == cudaSuccess) err = cudaEventElapsedTime(&elapsed, begin, end);
                    if (err != cudaSuccess) return;
                    if (!(elapsed > 0)) { err = cudaErrorUnknown; return; }
                    total_ms += elapsed; ++samples;
                } while (total_ms < window_ms);
                if (phase) ms = total_ms / samples;
            }
        };
        float a = 0, b = 0, zero = 0, ordinary = 0;
        // Every pair has a symmetric A/B/B/A order on the identical input,
        // nonce range and geometry. Measure the winner once for both comparisons.
        const int modes[] = {native, 2, 6, ordinary_ptx, ordinary_ptx, 6, 2, native};
        for (int mode : modes) {
            if (mode == 6 && !compare_zero_tail) continue;
            if (mode == ordinary_ptx && !measure_ordinary) continue;
            float ms = 0; sample(mode, ms);
            if (mode == ordinary_ptx) ordinary += ms;
            else if (mode == 6) zero += ms;
            else if (mode == 2) b += ms;
            else a += ms;
        }
        if (begin) cudaEventDestroy(begin);
        if (end) cudaEventDestroy(end);
        if (err != cudaSuccess) { why = cudaGetErrorString(err); return -1; }
        if (!(a > 0 && b > 0)) { why = "invalid Blackwell calibration timing"; return -1; }
        chosen = b < a * 0.98f ? 2 : native;
        // Near a crossover, time the first real ordinary job as usual.
        if (measure_ordinary && ordinary > 0 && (ordinary < a * 0.92f || ordinary > a * 1.08f))
            choices[120] = ordinary < a ? 2 : native;
        fprintf(stderr, "GPU %d: hash layout %d selects %s (native %.3f ms, Blackwell %.3f ms)\n",
            device, plen % 128, chosen == 2 ? "Blackwell" : "native", a * 0.5f, b * 0.5f);
        if (compare_zero_tail) {
            const float incumbent = chosen == 2 ? b : a;
            if (!(zero > 0)) { why = "invalid zero-tail calibration timing"; return -1; }
            zero_choice = zero < incumbent * 0.98f ? 6 : chosen;
            fprintf(stderr, "GPU %d: aligned-tail job selects %s (existing %.3f ms, zero-tail %.3f ms)\n",
                    device, zero_choice == 6 ? "zero-tail" : "existing kernel", incumbent * 0.5f, zero * 0.5f);
        }
        return chosen;
    }
    // Mode 6 is the zero-tail kernel. Calibrate against the existing selection
    // on this GPU; never let a new layout replace another platform's fast path.
    int select(const Params &input, int plen, unsigned grid, HitRec *hits, unsigned *nhit, std::string &why) {
        const bool padded = is_zero_tail(input);
        const char *force = padded ? getenv("PX_TXID_ZERO_TAIL") : nullptr;
        const bool compare_zero_tail = padded && zero_choice < 0 && !wide_zero &&
            !(force && (!strcmp(force, "0") || !strcmp(force, "1")));
        const int baseline = select_generic(input, plen, grid, hits, nhit, padded, compare_zero_tail, why);
        if (baseline < 0 || !padded) return baseline;
        if (force && !strcmp(force, "0")) return baseline;
        if (zero_choice >= 0) return zero_choice;
        if (force && !strcmp(force, "1")) return zero_choice = 6; // diagnostic; startup checks still run
        Params p = input; p.mask = ~0ULL;
        const uint64_t stride = (uint64_t)grid * TXID_TPB;
        if (!stride || stride > (1ULL << 30)) return baseline;
        const uint32_t iters = (uint32_t)std::min<uint64_t>(256, (1ULL << 31) / stride);
        const uint64_t count = stride * iters, first = 0xa1b2000000000000ULL;
        cudaEvent_t begin = nullptr, end = nullptr;
        cudaError_t err = cudaEventCreate(&begin);
        if (err == cudaSuccess) err = cudaEventCreateWithFlags(&end, cudaEventBlockingSync);
        auto sample = [&](int mode, float &ms) {
            AlignedSearch cache; cache.fast = mode == 2; cache.platform = mode >= 3 && mode <= 5 ? mode - 2 : 0;
            for (int phase = 0; phase < 2 && err == cudaSuccess; ++phase) {
                float total = 0; unsigned samples = 0;
                do {
                    err = cudaMemset(nhit, 0, sizeof(unsigned));
                    if (err == cudaSuccess) err = cudaEventRecord(begin);
                    if (err != cudaSuccess) break;
                    if (mode == 6 || mode == 7) launch_zero_tail(grid, TXID_TPB, 0, p, first, count, iters, hits, nhit,mode==7?512:256);
                    else if (mode) launch_aligned(grid, TXID_TPB, 0, p, plen, first, count, hits, nhit, cache);
                    else launch(grid, TXID_TPB, 0, p, first, count, iters, hits, nhit);
                    err = cudaGetLastError();
                    if (err == cudaSuccess) err = cudaEventRecord(end);
                    if (err == cudaSuccess) err = cudaEventSynchronize(end);
                    float elapsed = 0;
                    if (err == cudaSuccess) err = cudaEventElapsedTime(&elapsed, begin, end);
                    if (err != cudaSuccess) break;
                    if (!(elapsed > 0)) { err = cudaErrorUnknown; break; }
                    total += elapsed; ++samples;
                } while (total < 75.0f);
                if (phase && samples) ms = total / samples;
            }
        };
        float a = 0, b = 0, wide = 0;
        const std::vector<int> modes=wide_zero?std::vector<int>{baseline,6,7,7,6,baseline}:std::vector<int>{baseline,6,6,baseline};
        for (int mode : modes) {
            float ms = 0; sample(mode, ms);
            if (mode == 6) b += ms; else if(mode==7)wide+=ms;else a += ms;
        }
        if (begin) cudaEventDestroy(begin);
        if (end) cudaEventDestroy(end);
        if (err != cudaSuccess || !(a > 0 && b > 0)) { why = err == cudaSuccess ? "invalid zero-tail calibration" : cudaGetErrorString(err); return -1; }
        zero_choice = b < a * 0.98f ? 6 : baseline;
        if(wide_zero&&wide>0&&wide<(zero_choice==6?b:a)*.98f&&device>=0&&device<128) {
            zero_tail_blocks[device]=512;zero_choice=6;
        }
        fprintf(stderr, "GPU %d: aligned-tail job selects %s (existing %.3f ms, zero-tail %.3f ms)\n",
                device, zero_choice == 6 ? "zero-tail" : "existing kernel", a * .5f, b * .5f);
        if(wide_zero)fprintf(stderr,"GPU %d: zero-tail launch %u threads (512-thread %.3f ms)\n",device,
            device>=0&&device<128&&zero_tail_blocks[device]==512?512u:256u,wide*.5f);
        return zero_choice;
    }

};
} // namespace txid
