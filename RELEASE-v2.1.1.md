# ProxiChorizo 2.1.1

Faster long-job mining on Blackwell and Pascal, plus faster AMD startup and an
explicit `--no-autotune` option on CUDA, OpenCL and HiveOS. No miner fee.

- **RTX 5090: 19.802 → 21.611 GH/s (+9.13%)** on the matched long-job benchmark.
- **Four GTX 1080 Ti cards: +4.39% to +4.97%**, reaching 3.155–3.369 GH/s.
- `--no-autotune` skips legacy VRF tuning trials, using a valid cached profile or
  platform default. HiveOS: `AUTOTUNE=0` or `EXTRA_ARGS="--no-autotune"`. Mandatory
  verification and the brief Blake2b kernel comparison remain enabled.
- Keyless OpenCL hash mining skips unused legacy VRF compilation. Cached legacy
  profiles compile only the chosen variant. CUDA hash workers also avoid unused
  legacy VRF scratch allocations, improving startup with limited free VRAM.
- Automatic long-job negotiation and ordinary-job fallback remain enabled on all
  platforms. Every submitted hit is checked on the CPU. All kernels remain plain.

Matched Linux A/B/B/A results versus 2.1.0, same 504-byte padded prefix and full
existing firmware allowances: NVIDIA 30-second arms, RX 7900 XTX 15-second arms.
The measured executables are development builds carrying the 2.1.1 kernels; the
README and CSV retain executable hashes, power, clocks and individual results.
RTX 4070 Ti SUPER measured 7.333 GH/s and RX 7900 XTX 7.198 GH/s, unchanged within
run variation. MI50's 3.245 GH/s is a cooled four-second diagnostic, not sustained
thermal qualification. Existing AMD long-job gains are not new gains in 2.1.1.

Linux CUDA/OpenCL, Windows NVIDIA/AMD, HiveOS, mmpOS, SMOS and Docker NVIDIA/AMD
packages include current setup documentation and the public PROTOCOL-UPGRADE.
Docker: `ocminersupr/proxichorizo:2.1.1` and `:2.1.1-amd`, also mirrored on GHCR.
Windows CI checks startup, CPU tests and pool connections; physical Windows GPU
execution has not been tested. The ten-card RX 5700 XT report could not be timed
on matching hardware, so no specific startup-duration claim is made.

[Setup and pool](https://proxi.suprnova.cc/StartMining.html) ·
[Usage](https://github.com/ocminer/ProxiChorizo/blob/v2.1.1/USAGE.md) ·
[Community](https://discord.gg/29RHZwvVPN)
