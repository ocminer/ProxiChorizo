// Exercise the production two-block CUDA kernel over partial and wrapping nonce tiles.
#include <cuda_runtime.h>
#include "../../txid_cuda.cuh"
#include <cassert>
#include <cstdio>
#include <map>

int main() {
    txid::CudaSearch buffers;
    unsigned cases = 0;
    size_t nonces = 0;
    for (int length = 377; length <= 383; ++length) {
        std::vector<uint8_t> prefix(length);
        for (int i = 0; i < length; ++i) prefix[i] = uint8_t(i * 37 + length);
        for (unsigned bits : {0u, 5u, 63u}) {
            txid::Params params;
            txid::prepare(prefix.data(), length, bits, params);
            assert(params.kernel == 31);
            for (uint64_t first : {0x12345678ffffffc1ULL, 0xffffffffffffffe1ULL}) {
                for (unsigned count : {0u, 1u, 63u, 64u, 65u, 257u, 8195u, 32771u}) {
                    constexpr unsigned threads = 128;
                    unsigned iterations = (count + threads - 1) / threads;
                    assert(cudaMemset(buffers.d_n, 0, sizeof(unsigned)) == cudaSuccess);
                    txid::search_kernel<31><<<1, threads>>>(params, first, count,
                        iterations, buffers.d_hits, buffers.d_n, params.q);
                    assert(cudaGetLastError() == cudaSuccess);
                    assert(cudaDeviceSynchronize() == cudaSuccess);
                    unsigned found = 0;
                    assert(cudaMemcpy(&found, buffers.d_n, sizeof(found), cudaMemcpyDeviceToHost) == cudaSuccess);
                    std::map<uint64_t, uint64_t> expected;
                    for (unsigned i = 0; i < count; ++i) {
                        uint8_t digest[32];
                        txid::hash_nonce(prefix, first + i, digest);
                        if (txid::work_bits(digest) >= int(bits))
                            expected.emplace(first + i, txid::load64le(digest + 24));
                    }
                    assert(found == expected.size());
                    std::vector<txid::HitRec> hits(std::min<unsigned>(found, TXID_MAX_HITS));
                    if (!hits.empty())
                        assert(cudaMemcpy(hits.data(), buffers.d_hits, sizeof(hits[0]) * hits.size(),
                                          cudaMemcpyDeviceToHost) == cudaSuccess);
                    for (auto hit : hits) {
                        auto pos = expected.find(hit.n);
                        assert(pos != expected.end() && pos->second == hit.h3);
                        expected.erase(pos);
                    }
                    assert(expected.size() == found - hits.size());
                    ++cases;
                    nonces += count;
                }
            }
        }
    }
    printf("PASS: %u cases, %zu reference nonces across all seven splits, partial tiles, wraps and overflow\n",
           cases, nonces);
}
