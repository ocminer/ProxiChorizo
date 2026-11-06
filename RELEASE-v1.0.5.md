# ProxiChorizo 1.0.5

Faster NVIDIA kernel: +7 to +19 % on every card we measured. Recommended for every NVIDIA rig.

## Faster

A new default kernel (faster hash-to-curve and point multiplication) for Blackwell (RTX 50xx, RTX PRO 6000),
Ada (RTX 40xx), Ampere GA10x (RTX 30xx) and the CMP 170HX, with larger launches (256 blocks per multiprocessor).
Turing measures it on its first start along with the previous kernels. Matched `--bench` runs (eight 60 s A/B/B/A
arms) of 1.0.5 against the published 1.0.4 on the same card, power cap and clocks, platform defaults:

| GPU | Power cap | 1.0.4 | **1.0.5** | Gain over 1.0.4 | Power drawn (1.0.5) | MH/s per W (1.0.5) |
|---|---:|---:|---:|---:|---:|---:|
| RTX 5090 | 600 W | 70.1 MH/s | **78.6 MH/s** | +12.2 % | 600 W | 0.131 |
| RTX 5090 | 420 W | 56.5 MH/s | **66.9 MH/s** | +18.6 % | 420 W | 0.159 |
| RTX 5080 | 360 W | 37.4 MH/s | **40.2 MH/s** | +7.4 % | 326 W | 0.123 |
| RTX 5080 (second card) | 360 W | 36.7 MH/s | **39.4 MH/s** | +7.4 % | 292 W | 0.135 |
| RTX 4070 Ti SUPER | 285 W | 28.5 MH/s | **31.3 MH/s** | +10.1 % | 282 W | 0.111 |
| RTX 3070 | 240 W | 13.8 MH/s | **15.1 MH/s** | +9.7 % | 230 W | 0.066 |
| RTX 3070 (second card) | 240 W | 13.6 MH/s | **14.8 MH/s** | +9.4 % | 231 W | 0.064 |
| CMP 170HX (default clocks) | 250 W | 15.9 MH/s | **17.0 MH/s** | +7.3 % | 180 W | 0.095 |

(RTX 4070 Ti SUPER: single-architecture builds of the 1.0.4 and 1.0.5 sources.)

The outputs are bit-identical to 1.0.4. Every card above passed the node's test vectors, 512 independently computed
VRF outputs, the field-arithmetic test and a comparison of complete outputs against the previous kernels, and the
release build mined on the pool with 0 rejected shares. The A100/H100 datacenter profiles and AMD/OpenCL keep their
kernels; `--variant fe32x_wnaf_s32_c` selects the 1.0.4 kernel. The full table, including cards measured with earlier
releases, is in the README.

CPU use stays as in 1.0.4 (about 3 % of one core per GPU). The stratum protocol, options, console and launchers are
unchanged from 1.0.4.
