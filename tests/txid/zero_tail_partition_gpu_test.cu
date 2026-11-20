// ProxiChorizo License 1.0; see LICENSE.
// Exercise mixed native/aligned/zero-tail enumeration with virtual GPU splits.
#include "../../txid_aligned.cuh"
#include "../../txid_zero_tail.cuh"
#include <cassert>
#include <set>
// The aligned production kernel is Blackwell-only. On other cards exercise
// its nonce permutation with the ordinary exact hash, without claiming that
// this executes or qualifies the Blackwell kernel itself.
__global__ void aligned_enumeration_oracle(txid::ParamsCore p, txid::Plan01 q, uint64_t base,
        uint64_t count, txid::HitRec *hits, unsigned *nhit) {
    for (uint64_t i=threadIdx.x; i<count; i+=blockDim.x) {
        const uint64_t logical=base+i;
        const uint64_t n=(logical&0xffff000000000000ULL) |
            ((logical&0x000000ff00000000ULL)<<8) |
            ((logical&0x0000ff0000000000ULL)>>8) |
            (uint64_t)__byte_perm((uint32_t)logical,0,0x0123);
        const uint64_t h=txid::txid_h3<15,false>(p,txid::bswap64d(n),&q);
        if ((h&p.mask)==0) txid::record(n,h,hits,nhit);
    }
}
int main() {
    cudaDeviceProp prop{};assert(cudaGetDeviceProperties(&prop,0)==cudaSuccess);
    const bool blackwell=prop.major==12 && prop.minor==0;
    txid::CudaSearch memory;
    std::vector<uint8_t> prefix(504,0);
    for(int i=0;i<384;++i)prefix[i]=(uint8_t)(i*13+9);
    txid::Params p;txid::prepare(prefix.data(),504,0,p);
    size_t checked=0,batches=0;
    for(unsigned bits:{1,2,3,8}) {
        const unsigned gpus=1u<<bits;const uint64_t span=(1ULL<<48)/gpus;
        for(unsigned gpu=0;gpu<gpus;++gpu) for(int mode:{0,1,6}) {
            const uint64_t lo=gpu*span,hi=lo+span;
            const uint64_t starts[] = {lo+0xfffffe01ULL,lo+span/2-255,hi-513};
            for(uint64_t start:starts) {
                std::set<uint64_t> expected,seen;uint64_t cur=start,remaining=513;
                txid::AlignedSearch cache;
                while(remaining) {
                    uint64_t base=0;assert(txid::zero_tail_nonce_base(0xa1b2,cur,lo,hi,base));
                    uint64_t count=txid::aligned_count(base,remaining);
                    assert(cudaMemset(memory.d_n,0,4)==cudaSuccess);
                    if(mode==6)txid::launch_zero_tail(1,256,0,p,base,count,3,memory.d_hits,memory.d_n);
                    else if(mode==1 && blackwell)txid::launch_aligned(1,256,0,p,504,base,count,memory.d_hits,memory.d_n,cache);
                    else if(mode==1)aligned_enumeration_oracle<<<1,256>>>(p,p.q,base,count,memory.d_hits,memory.d_n);
                    else txid::launch(1,256,0,p,base,count,3,memory.d_hits,memory.d_n);
                    const cudaError_t sync=cudaDeviceSynchronize();
                    if(sync!=cudaSuccess)fprintf(stderr,"partition oracle mode %d: %s\n",mode,cudaGetErrorString(sync));
                    assert(sync==cudaSuccess);
                    unsigned found=0;assert(cudaMemcpy(&found,memory.d_n,4,cudaMemcpyDeviceToHost)==cudaSuccess);assert(found==count);
                    std::vector<txid::HitRec> hits(found);assert(cudaMemcpy(hits.data(),memory.d_hits,found*sizeof(hits[0]),cudaMemcpyDeviceToHost)==cudaSuccess);
                    for(auto hit:hits) {
                        assert(hit.n>>48==0xa1b2);
                        unsigned first=(unsigned)((hit.n>>40)&255),second=(unsigned)((hit.n>>32)&255);
                        assert(((first^second)>>(8-bits))==gpu);
                        uint8_t hash[32];txid::hash_nonce(prefix,hit.n,hash);assert(hit.h3==txid::load64le(hash+24));
                        assert(seen.insert(hit.n).second);++checked;
                    }
                    for(uint64_t i=0;i<count;++i)expected.insert(mode==1?txid::aligned_nonce(4,base+i):base+i);
                    cur+=count;remaining-=count;++batches;
                }
                assert(expected==seen && seen.size()==513);
            }
        }
    }
    printf("PASS: %zu CPU/GPU full-reference hits in %zu batches; 2,4,8,256 virtual GPUs, all three kernel enumerations, counter crossings and range ends\n",checked,batches);
    printf("Aligned enumeration checked using %s\n",blackwell?"production Blackwell kernel":"portable test oracle (Blackwell execution not qualified)");
}
