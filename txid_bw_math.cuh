// ProxiChorizo - Copyright (c) 2026 suprnova.cc. ProxiChorizo License 1.0; see LICENSE.
// Device-only arithmetic for the separately compiled Blackwell PTX module.
// Explicit carries let newer drivers use their light 32-bit add instructions.
// The ordinary native modules retain their original arithmetic.
#pragma once
TXQ uint64_t bw_add(uint64_t a, uint64_t b) {
    uint32_t lo, hi;
    asm("add.cc.u32 %0, %2, %4;\n\taddc.u32 %1, %3, %5;"
        : "=&r"(lo), "=&r"(hi)
        : "r"((uint32_t)a), "r"((uint32_t)(a >> 32)), "r"((uint32_t)b), "r"((uint32_t)(b >> 32)));
    return ((uint64_t)hi << 32) | lo;
}
TXQ uint64_t bw_add_c(uint64_t a, uint64_t b) {
    uint32_t lo, hi, one;
    asm("mov.u32 %0, 1;" : "=r"(one));
    asm("add.cc.u32 %0, %2, %4;\n\tmadc.lo.u32 %1, %3, %6, %5;"
        : "=&r"(lo), "=&r"(hi)
        : "r"((uint32_t)a), "r"((uint32_t)(a >> 32)), "r"((uint32_t)b), "r"((uint32_t)(b >> 32)), "r"(one));
    return ((uint64_t)hi << 32) | lo;
}
TXQ uint64_t bw_rot63(uint64_t x) {
    uint32_t lo, hi;
    // Doubling each half with the two overflow bits carried across the wrap.
    asm("add.cc.u32 %0, %2, %2;\n\taddc.cc.u32 %1, %3, %3;\n\taddc.u32 %0, %0, 0;"
        : "=&r"(lo), "=&r"(hi) : "r"((uint32_t)x), "r"((uint32_t)(x >> 32)));
    return ((uint64_t)hi << 32) | lo;
}
