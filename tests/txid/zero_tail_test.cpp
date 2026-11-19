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
    // Exhaust the 16-bit high counter domain for every supported GPU split.
    // The aligned path swaps these bytes; native/zero-tail leave them in place.
    for (unsigned bits = 0; bits <= 8; ++bits) {
        const unsigned gpus = 1u << bits;
        const uint64_t span = (1ULL << 48) / gpus;
        std::vector<int> owner(65536,-1);
        for (unsigned gpu = 0; gpu < gpus; ++gpu) {
            for (uint64_t high = 0; high < (1ULL << (16-bits)); ++high) {
                uint64_t n = 0, cur = gpu*span + (high << 32);
                assert(txid::zero_tail_nonce_base(0xa1b2,cur,gpu*span,(gpu+1)*span,n));
                assert(n>>48==0xa1b2 && (uint32_t)n==0);
                const unsigned word = (unsigned)((n>>32)&65535);
                assert(owner[word]==-1); owner[word]=(int)gpu;
            }
        }
        for (unsigned word=0;word<65536;++word) {
            assert(owner[word]>=0 && owner[word]==owner[((word&255)<<8)|(word>>8)]);
        }
    }
    uint64_t n=0;
    assert(!txid::zero_tail_nonce_base(0,0,0,0,n));
    assert(!txid::zero_tail_nonce_base(0,1ULL<<48,0,1ULL<<48,n));
    assert(!txid::zero_tail_nonce_base(0,0,0,1ULL<<39,n));
    printf("PASS: 41000 reference nonces, %zu mutated-parameter rejections, all 1024 prefix lengths\n",negative);
    puts("PASS: exhaustive high-counter domains for 1..256 GPUs, native/aligned mixed enumeration remains disjoint");
}
