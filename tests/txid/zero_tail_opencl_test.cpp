// ProxiChorizo License 1.0; see LICENSE.
#define main proxichorizo_program_main
#include "../../opencl/proxichorizo_cl.cpp"
#undef main
#include <cassert>
int main(int argc,char **argv) {
    if(argc!=2)return 2;
    enumerate_devices();int selected=-1;
    for(size_t i=0;i<g_devs.size();++i)if(g_devs[i].bus==atoi(argv[1])){assert(selected<0);selected=(int)i;}
    assert(selected>=0);ClDev &d=g_devs[selected];cl_int err;
    d.ctx=clCreateContext(nullptr,1,&d.id,nullptr,nullptr,&err);CL_CHECK(err);
    d.q=clCreateCommandQueue(d.ctx,d.id,0,&err);CL_CHECK(err);configure_wait(d);
    ClTxSearch search(d);std::string why;
    if(txid::zero_tail_selfcheck(std::ref(search),why)){fprintf(stderr,"%s\n",why.c_str());return 1;}
    std::vector<uint8_t> prefix(504,0);
    for(int i=0;i<384;++i)prefix[i]=(uint8_t)(i*37+11);
    txid::Params p;txid::prepare(prefix.data(),504,0,p);
    assert(search(p,0,0).empty());
    for(uint64_t first:{0xa1b27856fffff800ULL,0xfffffffffffff800ULL}) {
        auto hits=search(p,first,5000);assert(hits.size()==TXID_MAX_HITS);
        cl_uint count=0;CL_CHECK(clEnqueueReadBuffer(d.q,search.nhit,CL_TRUE,0,4,&count,0,nullptr,nullptr));assert(count==5000);
        std::set<uint64_t> seen;
        for(auto h:hits) {
            assert(h.n-first<5000 && seen.insert(h.n).second);
            uint8_t digest[32];txid::hash_nonce(prefix,h.n,digest);assert(h.h3==txid::load64le(digest+24));
            printf("%016llx %016llx\n",(unsigned long long)h.n,(unsigned long long)h.h3);
        }
    }
    fprintf(stderr,"PASS: 41000 reference nonces plus two 5000-hit overflow/counter-wrap cases, zero count\n");
}
