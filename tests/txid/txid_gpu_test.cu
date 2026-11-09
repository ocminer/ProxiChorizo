// GPU test and bench of the txid-blake2b kernels outside the miner.
// nvcc -O3 -std=c++17 -arch=sm_120 -I../.. -o txid_gpu_test txid_gpu_test.cu
// ./txid_gpu_test check [dev]          self-check (vectors + all shapes)
// ./txid_gpu_test bench dev plen secs grid block iters
// ./txid_gpu_test fuzz dev [log2 nonces = 25] [k = 12] [seed]
//     random prefixes over every kernel shape, mining launch geometry (SMs x 16 blocks of TXID_TPB, several iterations),
//     some ranges ending a few nonces before or crossing a 2^32 boundary of the nonce's low word, partial last launches;
//     every nonce is also hashed by the CPU reference (blake2b256): the GPU hit set must equal the CPU set exactly
#include <cuda_runtime.h>
#include <thread>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <functional>
#include "txid_cuda.cuh"
int main(int argc, char **argv) {
    std::string mode = argc > 1 ? argv[1] : "check"; int dev = argc > 2 ? atoi(argv[2]) : 0;
    cudaSetDevice(dev);
    if (mode == "check") { std::string why; int rc = txid::cuda_selfcheck(why); printf("%s: %s\n", rc ? "FAIL" : "PASS", why.c_str()); return rc; }
    if (mode == "fuzz") {
        const int lg = argc > 3 ? atoi(argv[3]) : 25, k = argc > 4 ? atoi(argv[4]) : 12;
        uint64_t seed = argc > 5 ? strtoull(argv[5], nullptr, 0) : 0x5EED2026110900ULL;
        auto rnd = [&]() { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; return seed; };
        cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
        const unsigned grid = prop.multiProcessorCount * 16, block = TXID_TPB;
        txid::HitRec *dh; unsigned *dn; cudaMalloc(&dh, sizeof(txid::HitRec) * TXID_MAX_HITS); cudaMalloc(&dn, 4);
        const uint64_t total = 1ULL << lg; uint64_t done = 0, hits = 0; int cases = 0, shapes[32] = {};
        const unsigned nth = std::max(1u, std::thread::hardware_concurrency());
        while (done < total) {
            // the shapes in turn: kernel s = 2W + SH (nonce at byte offset 8W(+1..7) of its block), 31 = straddle (offset 121..127)
            const int s = cases % 32, off = s == 31 ? 121 + (int)(rnd() % 7) : 8 * (s / 2) + ((s & 1) ? 1 + (int)(rnd() % 7) : 0);
            int plen = 128 * (int)(rnd() % 6) + off; if (plen > 760) plen -= 128;
            std::vector<uint8_t> prefix((size_t)plen); for (auto &c : prefix) c = (uint8_t)rnd();
            txid::Params p; txid::Shape sh = txid::prepare(prefix.data(), plen, k, p); shapes[sh.kernel]++;
            const uint32_t iters = 1 + (uint32_t)(rnd() % 4); const uint64_t per = (uint64_t)grid * block * iters;
            uint64_t nbase = rnd() & 0xffffffffffffULL | ((uint64_t)(uint16_t)rnd() << 48);
            const int kind = cases % 4;   // 0: full launch, 1: partial, 2: ends just below a 2^32 boundary, 3: crosses it
            uint64_t count = per;
            if (kind == 1) count = 1 + rnd() % per;
            if (kind == 2) nbase = (nbase | 0xffffffffULL) - per + 1;
            if (kind == 3) nbase = (nbase | 0xffffffffULL) - (rnd() % per);
            cudaMemset(dn, 0, 4);
            txid::launch(grid, block, 0, p, nbase, count, iters, dh, dn);
            if (cudaDeviceSynchronize() != cudaSuccess) { printf("FAIL: CUDA error %s\n", cudaGetErrorString(cudaGetLastError())); return 2; }
            unsigned n = 0; cudaMemcpy(&n, dn, 4, cudaMemcpyDeviceToHost);
            if (n > TXID_MAX_HITS) { printf("FAIL: hit buffer overflow (%u), use a larger k\n", n); return 2; }
            std::vector<txid::HitRec> got(n); if (n) cudaMemcpy(got.data(), dh, sizeof(txid::HitRec) * n, cudaMemcpyDeviceToHost);
            std::vector<std::vector<txid::HitRec>> part(nth);
            std::vector<std::thread> th;
            for (unsigned t = 0; t < nth; t++) th.emplace_back([&, t]() {
                std::vector<uint8_t> msg(prefix); msg.resize(prefix.size() + 8); uint8_t hh[32];
                for (uint64_t i = t; i < count; i += nth) {
                    const uint64_t nn = nbase + i;
                    for (int b = 0; b < 8; b++) msg[prefix.size() + b] = (uint8_t)(nn >> (56 - 8 * b));
                    txid::blake2b256(msg.data(), msg.size(), hh);
                    if (txid::work_bits(hh) >= k) part[t].push_back({nn, txid::load64le(hh + 24)});
                }
            });
            for (auto &x : th) x.join();
            std::vector<txid::HitRec> want; for (auto &v : part) want.insert(want.end(), v.begin(), v.end());
            auto lt = [](const txid::HitRec &a, const txid::HitRec &c) { return a.n < c.n; };
            std::sort(got.begin(), got.end(), lt); std::sort(want.begin(), want.end(), lt);
            bool same = got.size() == want.size();
            for (size_t i = 0; same && i < got.size(); i++) same = got[i].n == want[i].n && got[i].h3 == want[i].h3;
            if (!same) { printf("FAIL: case %d prefix %d (kernel %d) kind %d nbase %016llx count %llu iters %u: GPU %zu hits, CPU %zu\n", cases, plen, sh.kernel, kind,
                                (unsigned long long)nbase, (unsigned long long)count, iters, got.size(), want.size()); return 1; }
            done += count; hits += want.size(); cases++;
        }
        int covered = 0; for (int i = 0; i < 32; i++) covered += shapes[i] > 0;
        printf("PASS: %d cases, %llu nonces (2^%.2f), k=%d, %llu hits all equal to the CPU reference, %d/32 kernel shapes\n", cases,
               (unsigned long long)done, std::log2((double)done), k, (unsigned long long)hits, covered);
        return covered == 32 ? 0 : 3;
    }
    int plen = argc > 3 ? atoi(argv[3]) : 600; double secs = argc > 4 ? atof(argv[4]) : 10;
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    unsigned grid = argc > 5 ? atoi(argv[5]) : prop.multiProcessorCount * 16, block = argc > 6 ? atoi(argv[6]) : 256, iters = argc > 7 ? atoi(argv[7]) : 64;
    std::vector<uint8_t> prefix(plen); for (int i = 0; i < plen; i++) prefix[i] = (uint8_t)(i * 31 + 7);
    txid::Params p; txid::Shape s = txid::prepare(prefix.data(), plen, 40, p);
    txid::HitRec *dh; unsigned *dn; cudaMalloc(&dh, sizeof(txid::HitRec) * TXID_MAX_HITS); cudaMalloc(&dn, 4); cudaMemset(dn, 0, 4);
    uint64_t per = (uint64_t)grid * block * iters, n = 0x0102000000000000ULL, total = 0;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    txid::launch(grid, block, 0, p, n, per, iters, dh, dn); cudaDeviceSynchronize();
    auto t0 = std::chrono::steady_clock::now(); float kms = 0; int launches = 0;
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < secs) {
        cudaEventRecord(e0); txid::launch(grid, block, 0, p, n, per, iters, dh, dn); cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1); kms += ms; launches++; n += per; total += per;
    }
    double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    cudaError_t e = cudaGetLastError();
    printf("%s plen %d kernel %d grid %u block %u iters %u: %.3f GH/s wall, %.3f GH/s kernel, %.1f ms/launch %s\n", prop.name, plen, s.kernel, grid, block, iters, total / el / 1e9, total / (kms / 1e3) / 1e9, kms / launches, cudaGetErrorString(e));
    return 0;
}
