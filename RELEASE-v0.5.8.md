# ProxiChorizo 0.5.8

New field-arithmetic kernels for NVIDIA GPUs: about 1.5× the hashrate of 0.5.7 on RTX 50xx and RTX 30xx cards and 1.75× on the CMP 170HX, selected automatically. Datacenter (A100, H100) and AMD profiles are unchanged.

## Measured improvements

Matched A/B/B/A measurements against the 0.5.7 release source on the same card, driver, mining key and power cap:

| GPU | Power cap | 0.5.7 | 0.5.8 | Gain |
|---|---:|---:|---:|---:|
| RTX 5090 | 600 W | 42.3 MH/s | 64.0 MH/s | **+51 %** |
| RTX 5090 | 420 W | 32.7 MH/s | 49.9 MH/s | **+53 %** |
| RTX 5080 | 360 W | 22.1 MH/s | 33.0 MH/s | **+50 %** |
| RTX 3070 | 240 W | 7.65 MH/s | 11.49 MH/s | **+50 %** |
| CMP 170HX | 250 W | 7.6 MH/s | 13.3 MH/s | **+76 %** |

Benchmark rates (`--bench`) at the listed caps; pool rates vary with job changes, clocks, cooling and the mining key. The 5090 and 3070 run into their power caps; the 5080 and CMP 170HX do not. See PLATFORMS.md (round 10) for the series, the profile and the rejected experiments.

## What changed

- New `fe32c_wnaf_s16_c` kernel variant: product-scanning field multiplication and squaring, the cofactor folded into the scalar multiplication, a separate normalisation stage sharing one inversion over 16 points. Default for Blackwell, Ampere GA10x and GA100 (CMP 170HX); Ada and Turing pick it up through the first-start quick tune, which now includes it. The previous variants stay available (`--variant`, `--autotune`).
- `--token-file FILE` (and `PROXI_TOKEN` in the environment) keep the pool token off the command line.
- A100 and H100 keep their 0.5.7 measured profiles; the OpenCL miner is unchanged apart from the version.

## Downloads

| Platform | Asset |
|---|---|
| Linux / HiveOS / MMPOS / SMOS, NVIDIA + AMD | `proxichorizo-0.5.8.tar.gz` |
| HiveOS custom miner | `proxichorizo-hiveos.tar.gz` |
| Native Windows x64, NVIDIA | `proxichorizo-0.5.8-windows-x86_64.zip` |

The Linux archives include both the CUDA and OpenCL miners and all three mining-OS launchers. Every archive includes `LICENSE`, `NOTICE`, documentation, `BUILD-INFO.json` and internal checksums. External `.sha256` files and `SHA256SUMS` cover the downloads.

Docker, NVIDIA only: `ocminersupr/proxichorizo:0.5.8` and `:latest`, mirrored at `ghcr.io/ocminer/proxichorizo:0.5.8` and `:latest`.

## Upgrade and validation

Replace the miner and remove old `--variant`, `--blocks` and `--iters` overrides to get the new defaults; a cached first-start tune from an older version is re-measured automatically. Linux builds target glibc 2.31; NVIDIA builds need a driver compatible with CUDA 12.9 and the GPU.

Every kernel variant passed, on sm_120 (RTX 5090, RTX 5080), sm_86 (RTX 3070) and sm_80 (CMP 170HX): the node's VRF vectors, 128,000 randomized field checks, 512 independent VRF vectors across 256 keys through the real pipeline, identical hit sets and complete-output comparisons over 58 dense launch shapes, and live pool mining with accepted shares and no rejects (RTX 5090: 10 minutes, 22 accepted; RTX 5080: 8 minutes, 18 accepted). The Windows build is validated by compilation, package checks and executable startup on CI; Windows GPU hardware execution has not been tested.
