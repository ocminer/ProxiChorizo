// ProxiChorizo License 1.0; see LICENSE.
// Host-only test of exact zero-tail math and fail-closed shape detection.
#include "../../txid_zero_tail.h"
#include <cassert>
#include <cstdio>
int main() {
    auto search = [](const txid::Params &p, uint64_t first, uint64_t count) {
        std::vector<txid::HitRec> hits;
        for (uint64_t i = 0; i < count; ++i) {
            uint64_t n = first + i, h = txid::zero_tail_h3(p, txid::nonce_word(n));
            if (!(h & p.mask)) hits.push_back({n,h});
        }
        return hits;
    };
    std::string why; assert(!txid::zero_tail_selfcheck(search, why));
    size_t negative = 0;
    for (int len : {120,248,504,632,1016}) {
        std::vector<uint8_t> prefix(len, 0);
        txid::Params p; txid::prepare(prefix.data(),len,0,p);
        assert(txid::is_zero_tail(p));
        for (int word = 0; word < 16; ++word) for (int bit = 0; bit < 64; ++bit) {
            auto bad = p; bad.m[word] ^= 1ULL << bit;
            assert(!txid::is_zero_tail(bad)); ++negative;
        }
        auto bad = p; bad.kernel = 31; assert(!txid::is_zero_tail(bad)); ++negative;
        bad = p; bad.shift = 8; assert(!txid::is_zero_tail(bad)); ++negative;
        bad = p; bad.tfin--; assert(!txid::is_zero_tail(bad)); ++negative;
        bad = p; bad.tfin = 0; assert(!txid::is_zero_tail(bad)); ++negative;
    }
    for (int len = 1; len <= 1024; ++len) {
        std::vector<uint8_t> prefix(len,0);
        txid::Params p; txid::prepare(prefix.data(),len,3,p);
        assert(txid::is_zero_tail(p) == (len % 128 == 120));
    }
    printf("PASS: 41000 reference nonces, %zu mutated-parameter rejections, all 1024 prefix lengths\n",negative);
}
