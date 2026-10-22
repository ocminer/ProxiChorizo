ProxiChorizo 0.5.6 makes the CUDA field arithmetic branch-free and is 3–7 % faster on every NVIDIA generation measured. Options and the pool protocol are unchanged; the OpenCL miner is the same as in 0.5.5.

## Changes

- **Branch-free carry handling.** The field add, subtract and the 512→256-bit fold each had a small data-dependent branch (a carry that is set in about half of the lanes, a top word that is non-zero in 97 % of them). Profiling on an H100 showed these branches diverging in nearly every warp, with instruction fetch after the branch as the largest stall. The fix-ups now run unconditionally with the same instruction count. Measured: H100 SXM +7 %, RTX PRO 6000 Blackwell +3.4 %, RTX A6000 +3.5 %.
- **Cached table in the 4-bit window loop.** The 15 odd multiples are kept in the cached (Y+X, Y−X, 2dT, 2Z) form, saving one multiplication per point addition. +1.7 % on Hopper, where the 4-bit table is the default.
- **Hopper platform row.** H100 (sm_90) starts on `fe32_w4_lb2_c` at 256 threads × 128 blocks per SM; the quick first-start tuning is no longer needed there. PLATFORMS.md documents the H100 measurements (Round 9), the pipe microbenchmark that shows why the card cannot be driven to its 700 W power limit by 256-bit integer work (Round 9b), the RTX A6000 round (8) and the rejected experiments (5-bit signed window, non-inlined field functions, other register caps and inversion batch sizes).

## Measured performance

| GPU | 0.5.5 | 0.5.6 |
|---|---|---|
| H100 SXM 80 GB (sm_90) | 19.8 MH/s | 21.2 MH/s |
| RTX PRO 6000 Blackwell Server (sm_120) | 41.6 MH/s | 43.0 MH/s |
| RTX A6000 (sm_86) | 11.6 MH/s | 12.0 MH/s |

Ada and consumer Blackwell cards get the same change; rates on RTX 5090 / 5080 / 3070 / CMP 170HX are noted in PLATFORMS.md as they are measured.

## Downloads

| Platform | Asset |
|---|---|
| Linux / HiveOS / MMPOS / SMOS, NVIDIA + AMD | `proxichorizo-0.5.6.tar.gz` |
| HiveOS custom miner | `proxichorizo-hiveos.tar.gz` |
| Native Windows x64 (NVIDIA) | `proxichorizo-0.5.6-windows-x86_64.zip` |

The Linux tarball contains `proxichorizo` (CUDA) and `proxichorizo-cl` (OpenCL), `h-run.sh`, `mmpos-run.sh`, `smos-run.sh`, `USAGE.md` and a checksum manifest; external `.sha256` files and `SHA256SUMS` cover the archives. Docker: `ocminersupr/proxichorizo:0.5.6` and `:latest` (mirror `ghcr.io/ocminer/proxichorizo`), NVIDIA only.

## Requirements and validation

Same requirements as 0.5.5 (driver for CUDA 12.9 on Linux and Windows). Every build passed the embedded self-test vectors and a 128,000-value random field test (`--fetest`) on sm_86, sm_90 and sm_120 before release.

Upgrading: recommended for all NVIDIA cards (+3–7 %); replace the binary, no option changes. AMD / OpenCL users gain nothing and can stay on 0.5.5.
