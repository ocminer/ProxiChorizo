// CPU regression checks: selecting A100 must never change CMP170HX or RTX profiles.
#include "../platform_profiles.h"
#include <cassert>
#include <cstdio>
#include <cstring>

int main() {
    const auto &a = platformFor(8,0,"NVIDIA A100-SXM4-80GB");
    const auto &h = platformFor(9,0,"NVIDIA H100 80GB HBM3");
    assert(std::strcmp(a.variant,"fe32_wnaf_split16_c")==0);
    assert(std::strcmp(h.variant,"fe32_w4_linear_split8_c")==0);
    assert(a.threads==256 && a.iters==1 && a.measured);
    assert(h.threads==256 && h.iters==1 && h.measured);
    struct Device {int major,minor;const char *name;};
    const Device unchanged[] = {
        {8,0,"NVIDIA CMP 170HX"},{12,0,"NVIDIA RTX PRO 6000 Blackwell Server Edition"},
        {12,0,"NVIDIA GeForce RTX 5090"},{12,0,"NVIDIA GeForce RTX 5080"},
        {8,6,"NVIDIA GeForce RTX 3070"},{8,9,"NVIDIA GeForce RTX 4070 Ti SUPER"},
        {8,6,"NVIDIA RTX A6000"},{7,5,"NVIDIA Tesla T4"},{9,0,"NVIDIA H200"},
        {8,0,"NVIDIA A100-SXM4-80GB MIG 1g.10gb"},
        {9,0,"NVIDIA H100 80GB HBM3 MIG 1g.10gb"},
        {8,6,"NVIDIA RTX A1000"},{13,0,"unmeasured future GPU"},{8,0,nullptr}
    };
    for (const auto &d : unchanged) {
        const Platform *legacy=nullptr;
        for (const auto &p : PLATFORMS)
            if(p.major==-1 || (p.major==d.major && (p.minor==-1 || p.minor==d.minor))) {legacy=&p;break;}
        assert(&platformFor(d.major,d.minor,d.name)==legacy);
    }
    // Mixed devices in one process must resolve independently of lookup order.
    for(int i=0;i<20;i++) {
        assert(&platformFor(8,0,"NVIDIA A100-SXM4-80GB")==&a);
        assert(&platformFor(9,0,"NVIDIA H100 80GB HBM3")==&h);
        assert(std::strcmp(platformFor(8,0,"NVIDIA CMP 170HX").variant,"fe32_wnaf_lb2_c")==0);
    }
    puts("PASS model-specific datacenter selection and unchanged legacy/MIG profiles");
}
