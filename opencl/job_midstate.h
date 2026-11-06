// ProxiChorizo License 1.0; see LICENSE.
#pragma once
#include <stdint.h>
#include <string.h>
static inline uint64_t px_rotr64(uint64_t x,int n) { return (x>>n)|(x<<(64-n)); }
static inline void prepare_sha_mid(JobDev *jd) {
    // First eight SHA-512 rounds depend only on the public key and job prefix.
    static const uint64_t k[8]={0x428a2f98d728ae22ULL,0x7137449123ef65cdULL,0xb5c0fbcfec4d3b2fULL,0xe9b5dba58189dbbcULL,0x3956c25bf348b538ULL,0x59f111f1b605d019ULL,0x923f82a4af194f9bULL,0xab1c5ed5da6d8118ULL};
    uint64_t a=0x6a09e667f3bcc908ULL,b=0xbb67ae8584caa73bULL,c=0x3c6ef372fe94f82bULL,d=0xa54ff53a5f1d36f1ULL;
    uint64_t e=0x510e527fade682d1ULL,f=0x9b05688c2b3e6c1fULL,g=0x1f83d9abfb41bd6bULL,h=0x5be0cd19137e2179ULL;
    for(int i=0;i<8;i++) {
        uint64_t t1=h+(px_rotr64(e,14)^px_rotr64(e,18)^px_rotr64(e,41))+((e&f)^(~e&g))+k[i]+jd->mw[i];
        uint64_t t2=(px_rotr64(a,28)^px_rotr64(a,34)^px_rotr64(a,39))+((a&b)^(a&c)^(b&c));
        h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
    }
    uint64_t mid[8]={a,b,c,d,e,f,g,h};memcpy(jd->h2c_mid,mid,sizeof mid);
}
