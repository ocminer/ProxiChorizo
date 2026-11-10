// ProxiChorizo License 1.0; see LICENSE.
// nvcc -O3 -std=c++17 -arch=sm_120 -I../.. -o aligned_gpu_test aligned_gpu_test.cu
// Exhaustively compare every tested nonce with the full CPU hash, not only GPU hits.
#include <cuda_runtime.h>
#include <functional>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include "txid_aligned.cuh"
#define CK(x) do { auto e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s: %s\n",#x,cudaGetErrorString(e));return 2;} } while(0)
int main(int argc,char**argv) {
    CK(cudaSetDevice(0));cudaDeviceProp prop;CK(cudaGetDeviceProperties(&prop,0));
    if(prop.major!=12||prop.minor!=0){fprintf(stderr,"This kernel test requires SM120\n");return 2;}
    std::string why;int rc=txid::aligned_selfcheck(why);printf("%s: %s\n",rc?"FAIL":"PASS",why.c_str());if(rc)return rc;
    unsigned cases=argc>1?atoi(argv[1]):40;
    uint64_t seed=0xd31e7b49a1c4f652ULL,total=0,hits=0;
    auto rnd=[&](){seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;return seed;};
    txid::HitRec*dh;unsigned*dn;CK(cudaMalloc(&dh,sizeof(txid::HitRec)*TXID_MAX_HITS));CK(cudaMalloc(&dn,4));
    unsigned block=TXID_TPB,grid=prop.multiProcessorCount*16;
    int offsets[]={52,97,108,113};
    for(unsigned c=0;c<cases;c++) {
        int plen=(rnd()%6)*128+offsets[c%4],k=12+(c%3);
        std::vector<uint8_t> prefix(plen);for(auto&v:prefix)v=(uint8_t)rnd();
        uint64_t first=rnd(),count=(1ULL<<20)+(rnd()%(1ULL<<18));
        if(c%4==0)first=(first&0xffffffff00000000ULL)|0xfff80000ULL;
        if(c%4==1)count=(uint64_t)grid*block*2;
        if(c%4==2)first&=0xffffffff00000000ULL;
        txid::Params p;txid::prepare(prefix.data(),plen,k,p);txid::AlignedSearch cache;
        std::vector<txid::HitRec> got,want;
        for(uint64_t done=0;done<count;) {
            uint64_t nb=first+done,n=txid::aligned_count(nb,count-done);
            CK(cudaMemset(dn,0,4));txid::launch_aligned(grid,block,0,p,plen,nb,n,dh,dn,cache);CK(cudaDeviceSynchronize());
            unsigned nh;CK(cudaMemcpy(&nh,dn,4,cudaMemcpyDeviceToHost));if(nh>TXID_MAX_HITS){fprintf(stderr,"overflow\n");return 3;}
            size_t at=got.size();got.resize(at+nh);if(nh)CK(cudaMemcpy(got.data()+at,dh,nh*sizeof(txid::HitRec),cudaMemcpyDeviceToHost));done+=n;
        }
        constexpr int NT=8;std::vector<txid::HitRec> part[NT];std::vector<std::thread> threads;
        for(int t=0;t<NT;t++)threads.emplace_back([&,t]{
            std::vector<uint8_t> msg(prefix);msg.resize(plen+8);uint8_t h[32];
            for(uint64_t i=t;i<count;i+=NT){
                uint64_t logical=first+i,n=txid::aligned_nonce(txid::aligned_offset(plen),logical);
                if((n>>48)!=(logical>>48))abort();
                for(int b=0;b<8;b++)msg[plen+b]=(uint8_t)(n>>(56-8*b));
                txid::blake2b256(msg.data(),msg.size(),h);
                if(txid::work_bits(h)>=k)part[t].push_back({n,txid::load64le(h+24)});
            }
        });
        for(auto&t:threads)t.join();for(auto &v:part)want.insert(want.end(),v.begin(),v.end());
        auto less=[](const txid::HitRec&a,const txid::HitRec&b){return a.n<b.n;};std::sort(got.begin(),got.end(),less);std::sort(want.begin(),want.end(),less);
        bool same=got.size()==want.size();for(size_t i=0;same&&i<got.size();i++)same=got[i].n==want[i].n&&got[i].h3==want[i].h3;
        if(!same){fprintf(stderr,"FAIL case %u plen%d count%llu CPU%zu GPU%zu\n",c,plen,(unsigned long long)count,want.size(),got.size());return 1;}
        total+=count;hits+=got.size();
    }
    printf("PASS: %u randomized cases, %llu nonces, %llu hits, all four aligned layouts, partial launches and 2^32 carries\n",cases,(unsigned long long)total,(unsigned long long)hits);
    cudaFree(dh);cudaFree(dn);return 0;
}
