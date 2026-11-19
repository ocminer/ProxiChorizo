// ProxiChorizo License 1.0; see LICENSE.
// Exact GPU hit sets, overflow, nonce wrap; stdout is checked by Python hashlib.
#include "../../txid_zero_tail.cuh"
#include <set>
#include <cassert>
int main() {
    txid::CudaZeroTailSearch search; std::string why;
    if (txid::zero_tail_selfcheck(std::ref(search),why)) { fprintf(stderr,"%s\n",why.c_str());return 1; }
    std::vector<uint8_t> prefix(504,0);
    for(int i=0;i<384;++i)prefix[i]=(uint8_t)(i*37+11);
    txid::Params p;txid::prepare(prefix.data(),504,0,p);
    assert(search(p,0,0).empty());
    for(uint64_t first:{0xa1b27856fffff800ULL,0xfffffffffffff800ULL}) {
        auto hits=search(p,first,5000);assert(hits.size()==TXID_MAX_HITS);
        unsigned count=0;assert(cudaMemcpy(&count,search.d_n,4,cudaMemcpyDeviceToHost)==cudaSuccess);assert(count==5000);
        std::set<uint64_t> seen;
        for(auto h:hits) {
            assert(h.n-first<5000 && seen.insert(h.n).second);
            uint8_t digest[32];txid::hash_nonce(prefix,h.n,digest);assert(h.h3==txid::load64le(digest+24));
            printf("%016llx %016llx\n",(unsigned long long)h.n,(unsigned long long)h.h3);
        }
    }
    fprintf(stderr,"PASS: 41000 reference nonces plus two 5000-hit overflow/counter-wrap cases, zero count\n");
}
