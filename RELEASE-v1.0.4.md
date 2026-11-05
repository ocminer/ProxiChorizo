# ProxiChorizo 1.0.4

CPU fix. Recommended for every NVIDIA rig, especially rigs with a small CPU or several GPUs.

## Fixed

- **One CPU core per GPU was busy the whole time.** Each GPU worker waited for its GPU with a busy wait, so a rig used
  one full CPU core per mining GPU (three GPUs: about 300 % CPU). The worker now sleeps while the GPU works and checks
  only around the expected end of each launch: about 3 % of one core per GPU, with the same hashrate. Measured:
  RTX 5090, 100 % -> 3.0 % of one core, 68.25 -> 68.22 MH/s; one process on 2x RTX 3070 + RTX 5080, 300 % -> 8.8 % in
  total, 13.51 / 13.71 / 37.42 MH/s per GPU before and after.

The GPU kernels, console, options and the stratum protocol are unchanged from 1.0.3.

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
