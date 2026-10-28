// ProxiChorizo License 1.0; see LICENSE.
#pragma once
#include <cstring>

struct Platform { int major, minor; const char *variant; int threads, iters, blocksPerSM; bool measured; const char *note; };
static const Platform PLATFORMS[] = {
    { 12, -1, "fe32c_wnaf_s16_c", 256, 1, 128, true,  "Blackwell (RTX 50xx, RTX PRO 6000): 0.5.8 comba field arithmetic, split normalisation (64.0 MH/s on a 5090 at 600 W, 33.0 on a 5080)" },
    {  9, -1, "fe32_w4_lb2_c",    256, 1, 128, true,  "Hopper (H100): the 4-bit table beats wNAF here (21.2 vs 19.3 MH/s on an H100 SXM), large launches; 0.5.8 kernels not measured on Hopper" },
    {  8,  9, "fe32c_wnaf_s16_c", 256, 1, 128, false, "Ada (RTX 40xx): not measured here, tuned on first start" },
    {  8,  6, "fe32c_wnaf_s16_c", 256, 1, 128, true,  "Ampere GA10x (RTX 30xx): 0.5.8 comba field arithmetic, split normalisation (11.5 MH/s on a 3070)" },
    {  8,  0, "fe32c_wnaf_s16_c", 256, 1, 128, true,  "Ampere GA100 (CMP 170HX): 0.5.8 comba field arithmetic, split normalisation (13.3 MH/s on a CMP 170HX)" },
    {  7,  5, "fe32c_wnaf_s16_c", 256, 1, 128, false, "Turing: not measured here, tuned on first start" },
    { -1, -1, "fe32c_wnaf_s16_c", 256, 1, 128, false, "unknown generation: tuned on first start" },
};

// A100 and CMP 170HX share SM80 but have different measured profiles. The A100 and H100
// rows keep their 0.5.7 datacenter profiles: the 0.5.8 kernels were not measured there
// (use --autotune or --variant fe32c_wnaf_s16_c to try them). MIG partitions follow the
// generation default.
static const Platform &platformFor(int major, int minor, const char *name) {
    static const Platform a100 = {8,0,"fe32_wnaf_split16_c",256,1,128,true,
        "A100: measured datacenter profile"};
    static const Platform h100 = {9,0,"fe32_w4_linear_split8_c",256,1,128,true,
        "H100: measured datacenter profile"};
    if (name && !std::strstr(name,"MIG")) {
        if (major==8 && minor==0 && std::strstr(name,"A100")) return a100;
        if (major==9 && minor==0 && std::strstr(name,"H100")) return h100;
    }
    for (const Platform &pl : PLATFORMS)
        if (pl.major==-1 || (pl.major==major && (pl.minor==-1 || pl.minor==minor))) return pl;
    return PLATFORMS[sizeof(PLATFORMS)/sizeof(PLATFORMS[0])-1];
}
