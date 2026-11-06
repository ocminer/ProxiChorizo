// ProxiChorizo License 1.0; see LICENSE.
#include "../platform_profiles.h"
#include <cassert>
#include <cstdio>
int main() {
    struct Case {int major,minor;const char *name;bool measured;const char *variant;};
    const Case cases[]={
        {8,6,"NVIDIA GeForce RTX 3070",true,"fe32x_mlsp_c"},
        {8,6,"NVIDIA GeForce RTX 3060",false,"fe32x_mlsp_c"},
        {8,6,"NVIDIA GeForce RTX 3060 Ti",false,"fe32x_mlsp_c"},
        {8,6,"NVIDIA GeForce RTX 3070 Ti",false,"fe32x_mlsp_c"},
        {8,6,"NVIDIA GeForce RTX 3070 Laptop GPU",false,"fe32x_mlsp_c"},
        {8,6,"NVIDIA GeForce RTX 3090",false,"fe32x_mlsp_c"},
        {8,9,"NVIDIA GeForce RTX 4070 Ti SUPER",true,"fe32x_mlsp_c"},
        {8,9,"NVIDIA GeForce RTX 4070 Ti",false,"fe32x_mlsp_c"},
        {12,0,"NVIDIA GeForce RTX 5090",true,"fe32x_mlsp_c"},
        {12,0,"NVIDIA GeForce RTX 5080",true,"fe32x_mlsp_c"},
        {12,0,"NVIDIA GeForce RTX 5070 Ti",true,"fe32x_mlsp_c"},
        {12,0,"NVIDIA GeForce RTX 5070",false,"fe32x_mlsp_c"},
        {12,0,"NVIDIA GeForce RTX 5090 Laptop GPU",false,"fe32x_mlsp_c"},
        {12,0,"NVIDIA RTX PRO 6000 Blackwell Server Edition",true,"fe32x_mlsp_c"},
        {8,0,"NVIDIA CMP 170HX",true,"fe32x_mlsp_c"},
        {8,0,"NVIDIA A100-SXM4-80GB",true,"fe32_wnaf_split16_c"},
        {9,0,"NVIDIA H100 80GB HBM3",true,"fe32_w4_linear_split8_c"},
        {7,5,"NVIDIA GeForce GTX 1660 SUPER",false,"fe32x_wnaf_s32_c"},
        {8,6,nullptr,false,"fe32x_mlsp_c"},
        {8,6,"",false,"fe32x_mlsp_c"},
    };
    const auto &a=platformFor(8,0,"NVIDIA A100-SXM4-80GB");
    const auto &h=platformFor(9,0,"NVIDIA H100 80GB HBM3");
    assert(a.threads==256 && a.iters==1 && a.blocksPerSM==128 && a.measured);
    assert(h.threads==256 && h.iters==1 && h.blocksPerSM==128 && h.measured);
    for(int i=0;i<20;i++) {
        assert(&platformFor(8,0,"NVIDIA A100-SXM4-80GB")==&a);
        assert(&platformFor(9,0,"NVIDIA H100 80GB HBM3")==&h);
        assert(platformFor(8,0,"NVIDIA CMP 170HX").blocksPerSM==256);
    }
    assert(&platformFor(8,0,"NVIDIA A100-SXM4-80GB MIG 1g.10gb")!=&a);
    assert(&platformFor(9,0,"NVIDIA H100 80GB HBM3 MIG 1g.10gb")!=&h);
    assert(!platformFor(8,6,"NVIDIA RTX A1000").measured);
    assert(!platformFor(13,0,"unmeasured future GPU").measured);
    for(const auto &c:cases){const auto&p=platformFor(c.major,c.minor,c.name);assert(p.measured==c.measured);assert(!std::strcmp(p.variant,c.variant));}
    std::printf("PASS: %zu model profiles, including 3060/Ti and laptop separation\n",sizeof cases/sizeof cases[0]);
}
