ProxiChorizo 0.5.0 adds AMD GPUs. The OpenCL miner `proxichorizo-cl` ships in the Linux tarball next to the CUDA binary, with a new 9×29-bit field layout that doubled the AMD rate measured in 0.4.0's development builds.

## Changes

- **AMD / OpenCL miner** (`opencl/`, binary `proxichorizo-cl`): the same algorithm in OpenCL C 1.2, compiled per GPU and variant by the installed driver at start-up (about 45 s per variant on ROCm) and validated against the embedded VRF test vectors before it is allowed to mine. Variants `fe29_*` (9×29-bit limbs, the AMD default), `fe32_*`, `fe51_*`; `--autotune`, `--list-devices`, `--groups`, `--kernel-file`, `--build-opts`. Same pool/solo protocol, options and statistics API as the CUDA miner.
- **9×29-bit field layout**: every partial product of the schoolbook multiply is a single `v_mad_u64_u32` into a 64-bit column, the 17 columns are folded once (2^261 ≡ 1216 mod p), subtraction adds 1024p limb-wise so lazily reduced operands never wrap. 2× on RDNA3, RDNA3 small and Vega 20.
- The HiveOS, MMPOS and SMOS launchers start `proxichorizo-cl` automatically when no NVIDIA driver is present (or when `MINER=cl` is set), so one package serves both vendors.
- **CPU reference build** `opencl/hostref/`: the unmodified kernel source compiled as C; `make check` runs the field test and all vectors for the three layouts with limb-invariant assertions. Test and init kernels keep the per-lane code path (`get_global_id` mixed in).
- CUDA miner unchanged apart from the version string.

## Measured performance

| GPU | Miner / variant | MH/s |
|---|---|---:|
| RTX 5080 | CUDA fe32_wnaf_lb2_c 256×4 | ≈ 17 |
| RTX 3070 | CUDA fe32_wnaf_lb2_c 256×4 | ≈ 6.9 |
| CMP 170HX | CUDA fe32_w4_lb2_c 256×2 | 4.6 – 6.3 |
| RX 7900 XTX (ROCm 7.1) | OpenCL fe29_wnaf_c ×8 | ≈ 10.5 (10.8 bench) |
| MI50 (ROCm 6.4) | OpenCL fe29_wnaf_c ×8 | ≈ 4.5 |
| RX 7600 XT (ROCm 6.4) | OpenCL fe29_wnaf_c ×8 | ≈ 3.6 |

## Downloads

| Platform | Asset |
|---|---|
| Linux / HiveOS / MMPOS / SMOS, NVIDIA + AMD | `proxichorizo-0.5.0.tar.gz` |
| HiveOS custom miner | `proxichorizo-hiveos.tar.gz` |
| Native Windows x64 (NVIDIA) | `proxichorizo-0.5.0-windows-x86_64.zip` |

The Linux tarball contains `proxichorizo` (CUDA) and `proxichorizo-cl` (OpenCL), `h-run.sh`, `mmpos-run.sh`, `smos-run.sh`, `USAGE.md` and a checksum manifest; external `.sha256` files and `SHA256SUMS` cover the archives. Docker: `ocminersupr/proxichorizo:0.5.0` and `:latest` (mirror `ghcr.io/ocminer/proxichorizo`), NVIDIA only.

## Requirements and validation

- NVIDIA: driver 550 or newer (CUDA 12.x). AMD: an OpenCL ICD with a working compiler for your card — ROCm 6.4 and 7.1 are tested (gfx906, gfx1100, gfx1102); the start-up validation refuses any variant your driver miscompiles and falls back to `fe32_wnaf_c`. Linux packages need glibc 2.31 or newer; Windows 10/11 x64 (CUDA only — no OpenCL build for Windows yet).
- Every build passes the self-test against the Proxima node's VRF test vectors, the randomized field-arithmetic test, the cross-variant hit check (`--verify`) and, for OpenCL, the CPU reference harness.
- Tested mining live on the Suprnova pool from NVIDIA and AMD rigs.
