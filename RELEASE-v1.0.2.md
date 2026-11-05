# ProxiChorizo 1.0.2

Faster NVIDIA kernels. Recommended for every NVIDIA rig: the same shares, more of them per second.

## Faster

A new default kernel for Blackwell (RTX 50xx, RTX PRO 6000), Ada (RTX 40xx), Ampere GA10x (RTX 30xx) and the
CMP 170HX. Turing still picks the faster kernel on its first start. Matched `--bench` runs (eight 60 s A/B/B/A arms)
of 1.0.2 against the published 1.0.1 on the same card, power cap and clocks, platform defaults:

| GPU | Power cap | 1.0.1 | **1.0.2** | Gain over 1.0.1 | Power drawn (1.0.2) | MH/s per W (1.0.2) |
|---|---:|---:|---:|---:|---:|---:|
| RTX 5090 | 600 W | 67.4 MH/s | **69.8 MH/s** | +3.5 % | 600 W | 0.116 |
| RTX 5090 | 420 W | 54.1 MH/s | **56.6 MH/s** | +4.6 % | 418 W | 0.135 |
| RTX 5080 | 360 W | 36.2 MH/s | **37.4 MH/s** | +3.4 % | 360 W | 0.104 |
| RTX 5080 (second card) | 360 W | 35.6 MH/s | **36.6 MH/s** | +3.0 % | 322 W | 0.114 |
| RTX 4070 Ti SUPER | 285 W | 27.3 MH/s | **28.4 MH/s** | +3.8 % | 284 W | 0.100 |
| RTX 3070 | 240 W | 13.1 MH/s | **13.7 MH/s** | +4.7 % | 238 W | 0.058 |
| CMP 170HX (default clocks) | 250 W | 14.9 MH/s | **15.7 MH/s** | +5.5 % | 182 W | 0.087 |

The outputs are bit-identical to 1.0.1. Every card above passed the node's test vectors, 512 independently computed
VRF outputs, the field-arithmetic test and a comparison of complete outputs against the previous kernels, and the
release package mined on two different GPUs in one process (with and without `--autotune`) with 0 rejected and
0 duplicate shares. The A100/H100 datacenter profiles and AMD/OpenCL keep their 1.0.1 kernels; the full table,
including cards measured with earlier releases, is in the README.

## Tip for power-limited RTX 5090s

At a reduced power cap the RTX 5090 gains a little from a lower memory clock (`sudo nvidia-smi -lmc 7001,7001`):
+1.4 % at 420 W, +0.8 % at 575 W, nothing measurable at 600 W. Do not go below 7001 MHz: the kernels need the
memory bandwidth (810 MHz cuts the rate to a third). `sudo nvidia-smi -rmc` restores the default. RTX 5080 and
RTX 3070 do not gain from a lower memory clock.

The stratum protocol, options and launchers are unchanged from 1.0.1.
