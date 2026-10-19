ProxiChorizo 0.4.0 is the first packaged release of the Suprnova CUDA nonce seeker for Proxima (PROX): pool mining on proxi.suprnova.cc and solo mining with `proxi node mine`, on Linux, HiveOS, MMPOS, SMOS, Windows and Docker.

## Changes

- One binary for Turing to Blackwell (sm_75, 80, 86, 89, 120 + PTX), kernel variant chosen per GPU generation or by `--autotune`.
- Field arithmetic in 8×32-bit limbs with PTX carry chains and a dedicated squaring, compacted hash-to-curve pipeline, width-5 wNAF scalar multiplication, T-skipping doublings.
- Pool support: per-user tokens and keys, vardiff shares, pool-assigned nonce ranges per worker and GPU (`nonce_base`), long-poll job updates (new job within one launch of a new tip).
- Solo support: the external-nonce-seeker protocol of `proxi node mine` (verified live).
- `--api-bind`: JSON statistics (per-GPU hashrate with PCI bus numbers, shares, uptime) and the mmpOS stats line.
- Windows build (MSVC) through a small POSIX/Winsock socket layer.
- Launchers for HiveOS (`h-run.sh`), MMPOS (`mmpos-run.sh`, plus the mmp-* external-miner trio) and SMOS (`smos-run.sh`), Docker image with an environment-variable front end.

## Measured performance

Platform defaults, one process per GPU, pool mining; rates vary with clocks and power limits.

| GPU | Variant | MH/s |
|---|---|---:|
| RTX 5080 | fe32_wnaf_lb2_c 256×4 | ≈ 17 |
| RTX 3070 | fe32_wnaf_lb2_c 256×4 | ≈ 6.9 |
| CMP 170HX | fe32_w4_lb2_c 256×2 | 4.6 – 6.3 |

The reference CPU seeker does about 37 kH/s per core. The work is pure integer ALU; PCIe link width and memory bandwidth do not matter.

## Downloads

| Platform | Asset |
|---|---|
| Linux / MMPOS / SMOS | `proxichorizo-0.4.0.tar.gz` |
| HiveOS custom miner | `proxichorizo-hiveos.tar.gz` |
| Native Windows x64 | `proxichorizo-0.4.0-windows-x86_64.zip` |

The Linux tarball includes `h-run.sh`, `mmpos-run.sh`, `smos-run.sh` and `USAGE.md`. The separate HiveOS archive uses the directory name required by its custom-miner installer. Every archive includes a file checksum manifest; external `.sha256` files and `SHA256SUMS` verify the downloads.

Docker: `ocminersupr/proxichorizo:0.4.0` and `:latest` (mirror `ghcr.io/ocminer/proxichorizo`).

## Requirements and validation

- NVIDIA driver 550 or newer (CUDA 12.x); Linux packages need glibc 2.31 or newer (built on Ubuntu 20.04, so they run on HiveOS and MMPOS images); Windows 10/11 x64.
- Every build passes the self-test against the Proxima node's VRF test vectors, a randomized field-arithmetic test, and the cross-variant hit check (`--verify`).
- Linux packages are tested mining live on the Suprnova pool and solo against `proxi node mine`. The hosted Windows runner has no GPU: Windows is validated by compiling, packaging and starting the executable.
