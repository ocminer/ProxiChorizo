// ProxiChorizo License 1.0; see LICENSE.
// Preserve the full Edwards group, including torsion: 64*x modulo 8L.
#pragma once
#include <stdint.h>
#include <string.h>
static inline void v_merged_scalar(const uint8_t *scalar, uint8_t out[32]) {
    static const uint8_t Lb[32] = { 0xed, 0xd3, 0xf5, 0x5c, 0x1a, 0x63, 0x12, 0x58, 0xd6, 0x9c, 0xf7, 0xa2, 0xde, 0xf9, 0xde, 0x14,
                                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10 };
    // t = scalar << 6 in 40 bytes
    uint8_t t[40]; memset(t, 0, sizeof t);
    for (int i = 0; i < 32; i++) { unsigned v = (unsigned)scalar[i] << 6; t[i] |= v & 0xff; t[i + 1] |= (v >> 8) & 0xff; }
    // subtract 8L << sh while t >= 8L << sh, for sh = 16..0 bits (64x < 2^262 = 2^7 * 8L, so 16 shifts are plenty)
    for (int sh = 16; sh >= 0; sh--) {
        uint8_t m[40]; memset(m, 0, sizeof m);
        for (int i = 0; i < 32; i++) { unsigned v = (unsigned)Lb[i] << ((sh + 3) & 7); int o = i + ((sh + 3) >> 3); m[o] |= v & 0xff; m[o + 1] |= (v >> 8) & 0xff; }
        for (;;) {
            int ge = 0; for (int i = 39; i >= 0; i--) { if (t[i] != m[i]) { ge = t[i] > m[i]; break; } if (i == 0) ge = 1; }
            if (!ge) break;
            int b = 0; for (int i = 0; i < 40; i++) { int d = (int)t[i] - (int)m[i] - b; b = d < 0; t[i] = (uint8_t)(d + (b ? 256 : 0)); }
        }
    }
    memcpy(out, t, 32);
}
