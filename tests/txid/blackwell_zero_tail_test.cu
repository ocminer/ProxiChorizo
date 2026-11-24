// ProxiChorizo License 1.0; see LICENSE.
// Production Blackwell dispatch: partial grids, counter boundaries, overflow,
// and ordinary fallback. Output is suitable for independent hashlib checking.
#define PX_BW_PTX 1
#include "../../txid_aligned.cuh"
#include <cassert>
#include <cstring>
#include <set>
int main(int argc, char **argv) {
    const bool ordinary_only = argc == 2 && !std::strcmp(argv[1], "--ordinary-kernel");
    if (argc > 1 && !ordinary_only) {
        fprintf(stderr, "usage: %s [--ordinary-kernel]\n", argv[0]);
        return 2;
    }
    cudaDeviceProp prop{};
    assert(cudaGetDeviceProperties(&prop, 0) == cudaSuccess);
    assert(prop.major == 12 && prop.minor == 0);
    txid::CudaSearch buffers;
    size_t verified = 0;
    for (int plen : {120, 248, 504, 632}) for (bool mutate : {false, true}) {
        std::vector<uint8_t> prefix(plen, 0);
        for (int i = 0; i < plen - 120; ++i) prefix[i] = (uint8_t)(i * 37 + 11);
        if (mutate) prefix[plen - 17] = 1; // Must use the old generic module.
        txid::Params base; txid::prepare(prefix.data(), plen, 0, base);
        for (uint64_t first : {0xa1b27856ffffff01ULL, 0xfffffffffffffff1ULL}) {
            txid::AlignedSearch cache; cache.fast = true;
            cache.allow_bw_tail = !ordinary_only;
            uint64_t left = 5513, cursor = first;
            while (left) {
                const uint64_t count = txid::aligned_count(cursor, left);
                assert(cudaMemset(buffers.d_n, 0, 4) == cudaSuccess);
                txid::launch_aligned(3, 128, 0, base, plen, cursor, count,
                    buffers.d_hits, buffers.d_n, cache);
                assert(cudaDeviceSynchronize() == cudaSuccess);
                unsigned got = 0;
                assert(cudaMemcpy(&got, buffers.d_n, 4, cudaMemcpyDeviceToHost) == cudaSuccess);
                assert(got == count);
                std::vector<txid::HitRec> hits(std::min<unsigned>(got, TXID_MAX_HITS));
                assert(cudaMemcpy(hits.data(), buffers.d_hits, hits.size() * sizeof(hits[0]), cudaMemcpyDeviceToHost) == cudaSuccess);
                std::set<uint64_t> expected, seen;
                for (uint64_t j = 0; j < count; ++j) expected.insert(txid::aligned_nonce(4, cursor + j));
                for (auto h : hits) {
                    assert(expected.count(h.n) && seen.insert(h.n).second);
                    uint8_t digest[32]; txid::hash_nonce(prefix, h.n, digest);
                    assert(h.h3 == txid::load64le(digest + 24)); ++verified;
                    printf("%d %d %016llx %016llx\n", plen, (int)mutate,
                        (unsigned long long)h.n, (unsigned long long)h.h3);
                }
                if (count <= TXID_MAX_HITS) assert(seen == expected);
                cursor += count; left -= count;
            }
        }
    }
    fprintf(stderr, "PASS: %zu full-reference hits; production Blackwell dispatch, 4 lengths, ordinary fallback, overflow, 32/64-bit boundaries\n", verified);
}
