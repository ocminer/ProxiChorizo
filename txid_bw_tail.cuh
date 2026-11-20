// ProxiChorizo License 1.0; see LICENSE.
// Exact final-block specialization for a verified 120-byte zero tail.
#pragma once
namespace txid_bw {
using namespace txid;
#ifdef __CUDA_ARCH__
template<int STYLE> __device__ __forceinline__ uint64_t add(uint64_t a,uint64_t b) {
    if constexpr(STYLE==1)return a+b;
    else return bw_add(a,b);
}
template<int STYLE> __device__ __forceinline__ uint64_t addc(uint64_t a,uint64_t b) {
    if constexpr(STYLE==1)return a+b;
    else if constexpr(STYLE==2)return bw_add(a,b);
    else return bw_add_c(a,b);
}
template<int STYLE> __device__ __forceinline__ uint64_t r63(uint64_t a) {
    if constexpr(STYLE==1||STYLE==3)return rotr<63>(a);
    else return bw_rot63(a);
}
template<int STYLE,bool X,bool Y> __device__ __forceinline__ void g(uint64_t &a,uint64_t &b,uint64_t &c,uint64_t &d,uint64_t nonce) {
    a=add<STYLE>(a,b);if constexpr(X)a=add<STYLE>(a,nonce);
    d=rotr<32>(d^a);c=addc<STYLE>(c,d);b=rotr<24>(b^c);
    a=add<STYLE>(a,b);if constexpr(Y)a=add<STYLE>(a,nonce);
    d=rotr<16>(d^a);c=addc<STYLE>(c,d);b=r63<STYLE>(b^c);
}
template<int STYLE,int R> __device__ __forceinline__ void tail_rounds(uint64_t v[16],uint64_t nonce) {
    constexpr int S[12][16]=TXID_SIGMA_ROWS;
    g<STYLE,S[R][0]==15,S[R][1]==15>(v[0],v[4],v[8],v[12],nonce);
    g<STYLE,S[R][2]==15,S[R][3]==15>(v[1],v[5],v[9],v[13],nonce);
    g<STYLE,S[R][4]==15,S[R][5]==15>(v[2],v[6],v[10],v[14],nonce);
    g<STYLE,S[R][6]==15,S[R][7]==15>(v[3],v[7],v[11],v[15],nonce);
    g<STYLE,S[R][8]==15,S[R][9]==15>(v[0],v[5],v[10],v[15],nonce);
    g<STYLE,S[R][10]==15,S[R][11]==15>(v[1],v[6],v[11],v[12],nonce);
    g<STYLE,S[R][12]==15,S[R][13]==15>(v[2],v[7],v[8],v[13],nonce);
    g<STYLE,S[R][14]==15,S[R][15]==15>(v[3],v[4],v[9],v[14],nonce);
    if constexpr(R<11)tail_rounds<STYLE,R+1>(v,nonce);
}
#endif
template<int MIN_BLOCKS>
__global__ void __launch_bounds__(512,MIN_BLOCKS) zero_tail_search(const ParamsCore p,uint32_t low,uint64_t fixed,uint32_t full,uint32_t tail,HitRec *hits,unsigned *nhit) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200
    const uint32_t gid=blockIdx.x*blockDim.x+threadIdx.x,stride=gridDim.x*blockDim.x;
    uint32_t n=low+gid,limit=full+(gid<tail);
    for(uint32_t i=0;i<limit;++i,n+=stride) {
        uint64_t v[16];
#pragma unroll
        for(int j=0;j<16;++j)v[j]=p.vr[j];
        // Fixed low nonce bytes were included by AlignedSearch::prepare.
        // Only this upper-word addition varies in the last G of round zero.
        v[3]=p.vr[3]+((uint64_t)n<<32);
        v[14]=rotr<16>(p.vhd[14]^v[3]);
        v[9]=bw_add_c(p.vhd[9],v[14]);
        v[4]=bw_rot63(p.vhd[4]^v[9]);
        tail_rounds<0,1>(v,p.m[15]|((uint64_t)n<<32));
        const uint64_t h3=p.h[3]^v[3]^v[11];
        if((h3&p.mask)==0){unsigned k=atomicAdd(nhit,1u);if(k<TXID_MAX_HITS)hits[k]={fixed|(uint64_t)__byte_perm(n,0,0x0123),h3};}
    }
#endif
}
} // namespace txid_bw
