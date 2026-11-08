// GPU test and bench of the txid-blake2b kernels outside the miner.
// nvcc -O3 -std=c++17 -arch=sm_120 -I../.. -o txid_gpu_test txid_gpu_test.cu
// ./txid_gpu_test check [dev]          self-check (vectors + all shapes)
// ./txid_gpu_test bench dev plen secs grid block iters
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <functional>
#include "txid_cuda.cuh"
int main(int argc, char **argv) {
    std::string mode = argc > 1 ? argv[1] : "check"; int dev = argc > 2 ? atoi(argv[2]) : 0;
    cudaSetDevice(dev);
    if (mode == "check") { std::string why; int rc = txid::cuda_selfcheck(why); printf("%s: %s\n", rc ? "FAIL" : "PASS", why.c_str()); return rc; }
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
