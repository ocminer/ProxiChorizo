# ProxiChorizo 1.0.3

Console and statistics fix on top of 1.0.2. The GPU kernels are unchanged from 1.0.2 (the same device code), so
the 1.0.2 speed-up and the measurements below apply unchanged.

## Fixed

- **Memory clock in the per-GPU lines.** Each GPU line now shows the core and the memory clock, labelled
  (`core 2550 / mem 13801 MHz`); 1.0.2 showed only the core clock, without a label. AMD/OpenCL shows the memory
  clock where the driver reports it.
- **Rig power in the total line.** The total line adds the power of all mining GPUs and the rig's MH/s per watt
  (`Total: 51.13 MH/s on 2/2 GPUs, 598 W (0.086 MH/W) | ...`).
- **Statistics API** (`--api-bind`): `mem_clock` per GPU next to `clock`, and the rig's total `power` in watts.

## Speed (unchanged kernels, measured with 1.0.2 against 1.0.1)

| GPU | Power cap | 1.0.1 | **1.0.2** | Gain over 1.0.1 | Power drawn (1.0.2) | MH/s per W (1.0.2) |
|---|---:|---:|---:|---:|---:|---:|
| RTX 5090 | 600 W | 67.4 MH/s | **69.8 MH/s** | +3.5 % | 600 W | 0.116 |
| RTX 5090 | 420 W | 54.1 MH/s | **56.6 MH/s** | +4.6 % | 418 W | 0.135 |
| RTX 5080 | 360 W | 36.2 MH/s | **37.4 MH/s** | +3.4 % | 360 W | 0.104 |
| RTX 5080 (second card) | 360 W | 35.6 MH/s | **36.6 MH/s** | +3.0 % | 322 W | 0.114 |
| RTX 4070 Ti SUPER | 285 W | 27.3 MH/s | **28.4 MH/s** | +3.8 % | 284 W | 0.100 |
| RTX 3070 | 240 W | 13.1 MH/s | **13.7 MH/s** | +4.7 % | 238 W | 0.058 |
| CMP 170HX (default clocks) | 250 W | 14.9 MH/s | **15.7 MH/s** | +5.5 % | 182 W | 0.087 |

Upgrading from 1.0.1 brings these gains; from 1.0.2 only the display changes. The stratum protocol, options and
launchers are unchanged.
