# ProxiChorizo 1.0.2

Faster NVIDIA kernels. Recommended for every NVIDIA rig: same shares, more of them per second.

## Faster

A new default kernel for Blackwell (RTX 50xx, RTX PRO 6000) and Ampere GA10x (RTX 30xx). Ada and Turing cards pick it on
their first start when it measures faster there. Matched `--bench` runs against 1.0.1 on the same card and power cap:

| GPU | Power cap | 1.0.1 | **1.0.2** | Gain |
|---|---:|---:|---:|---:|
| RTX 5090 | 600 W | __R5090_600_OLD__ MH/s | **__R5090_600_NEW__ MH/s** | __R5090_600_GAIN__ |
| RTX 5090 | 420 W | __R5090_420_OLD__ MH/s | **__R5090_420_NEW__ MH/s** | __R5090_420_GAIN__ |
| RTX 5080 | 360 W | __R5080_OLD__ MH/s | **__R5080_NEW__ MH/s** | __R5080_GAIN__ |
| RTX 3070 | 240 W | __R3070_OLD__ MH/s | **__R3070_NEW__ MH/s** | __R3070_GAIN__ |

The outputs are bit-identical to 1.0.1: every release build passes the node's test vectors, 512 independently computed
VRF outputs, the field-arithmetic test and a comparison of complete outputs against the previous kernel. CMP 170HX
(sm_80), the A100/H100 datacenter profiles and AMD/OpenCL keep their 1.0.1 kernels.

## Tip for power-limited RTX 5090s

At a reduced power cap the RTX 5090 gains a little from a lower memory clock (`sudo nvidia-smi -lmc 7001,7001`):
+1.4 % at 420 W, +0.8 % at 575 W, nothing measurable at 600 W. Do not go below 7001 MHz: the kernels need the
memory bandwidth (810 MHz cuts the rate to a third). `sudo nvidia-smi -rmc` restores the default. RTX 5080 and
RTX 3070 do not gain from a lower memory clock.

The stratum protocol, options and launchers are unchanged from 1.0.1.
