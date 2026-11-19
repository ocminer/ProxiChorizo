// ProxiChorizo - Copyright (c) 2026 suprnova.cc. ProxiChorizo License 1.0; see LICENSE.
#pragma once
#include "txid_blake2b.h"

namespace txid {
// proxima.aligned-tail.v1 describes bytes, not a trusted kernel selector.
// Validate the actual prepared final block, including its counter and shape.
static inline bool is_zero_tail(const ParamsCore &p) {
    if (p.kernel != 30 || p.shift != 0 || p.tfin < 128 || p.tfin % 128 != 0) return false;
    uint64_t nonzero = 0;
    for (int i = 0; i < 16; ++i) nonzero |= p.m[i];
    return nonzero == 0;
}

// Padded jobs may mix native sequential and aligned-permuted search on one rig.
// At offset 120 the aligned mapping swaps extranonce2's first two bytes and
// reverses its last four. Partition by XOR of the first bytes' high GPU bits:
// that domain is invariant under BOTH enumerations. Every GPU retains exactly
// its previous-sized search space, and single-GPU enumeration is unchanged.
// A batch must end before the low 32-bit counter wraps.
static inline bool zero_tail_nonce_base(uint16_t en1, uint64_t cursor,
        uint64_t lo, uint64_t hi, uint64_t &out) {
    if (hi <= lo || cursor < lo || cursor >= hi) return false;
    const uint64_t span = hi - lo;
    if (span > (1ULL << 48) || span < (1ULL << 40) || (span & (span - 1)) || lo % span) return false;
    unsigned bits = 0;
    for (uint64_t n = span; n < (1ULL << 48); n <<= 1) ++bits;
    const uint64_t gpu = lo / span, local = cursor - lo;
    if (gpu >= (1ULL << bits)) return false;
    const unsigned second = (unsigned)((local >> 32) & 255);
    const unsigned first = (unsigned)(local >> 40) |
            ((unsigned)(gpu ^ (second >> (8 - bits))) << (8 - bits));
    out = ((uint64_t)en1 << 48) | ((uint64_t)first << 40) |
            ((uint64_t)second << 32) | (local & 0xffffffffULL);
    return true;
}

// Literal zero message words disappear from the native integer additions.
// Only word 15 is variable. Keep round 0's existing exact precomputation.
TXQ uint64_t zero_tail_h3(const ParamsCore &p, uint64_t X) {
    uint64_t m[16], v[16];
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (int i = 0; i < 15; ++i) m[i] = 0;
    m[15] = X;
    round0<15, false>(v, p, m);
    rounds<1>(v, m);
    return p.h[3] ^ v[3] ^ v[11];
}

// Used by CPU and each backend's startup check. Includes different message
// counters, partial batches, both nonce-word boundaries and full nonce wrap.
template <class Search> static inline int zero_tail_selfcheck(Search search, std::string &why) {
    for (int len : {120, 248, 504, 632, 1016}) {
        std::vector<uint8_t> prefix((size_t)len);
        for (int i = 0; i < len - 120; ++i) prefix[i] = (uint8_t)(31 * i + len);
        Params p;
        for (int k : {0, 3, 32, 63}) {
            prepare(prefix.data(), len, k, p);
            if (!is_zero_tail(p)) { why = "zero-tail shape not recognized"; return 1; }
            for (uint64_t first : {0x0102abcffffffe01ULL, 0xfffffffffffffe01ULL}) {
                const uint64_t count = 1025;
                std::vector<HitRec> expected, got = search(p, first, count);
                for (uint64_t i = 0; i < count; ++i) {
                    uint8_t hash[32]; hash_nonce(prefix, first + i, hash);
                    if (work_bits(hash) >= k) expected.push_back({first + i, load64le(hash + 24)});
                }
                auto less = [](const HitRec &a, const HitRec &b) { return a.n < b.n; };
                std::sort(expected.begin(), expected.end(), less); std::sort(got.begin(), got.end(), less);
                if (expected.size() != got.size()) { why = "zero-tail hit count mismatch"; return 1; }
                for (size_t i = 0; i < got.size(); ++i)
                    if (got[i].n != expected[i].n || got[i].h3 != expected[i].h3) { why = "zero-tail hash or nonce mismatch"; return 1; }
            }
        }
    }
    why = "zero-tail full hashes and hit sets passed";
    return 0;
}
} // namespace txid
