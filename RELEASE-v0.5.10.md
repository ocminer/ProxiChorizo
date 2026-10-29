# ProxiChorizo 0.5.10

A kernel release: the field arithmetic issues its 32×32-bit products in chains whose carries ride as carry-ins of the next wide multiply, which removes about 40 % of the integer adds and most of the register moves from the hot loop. Hashrates rise on every NVIDIA card we measured, most on the RTX 30xx series. Everything else — console, thermal pause, device selection, API, launchers, OpenCL miner — is 0.5.9.

## What changed

- **Field multiply and square (`fe32w_wnaf_s16_c`, the new default for Blackwell, Ada, Ampere and Turing):** row-ordered schoolbook whose products are issued two columns apart, so the 2^64 carry of one `IMAD.WIDE.U32` is the carry-in of the next and no per-product add is needed; even and odd columns accumulate in two register arrays that are added once at the end, the 2^256 = 38 fold is a row of the same form. Same multiply count as 0.5.8's product scanning, about half its integer adds. Details, the Nsight Compute profile that led there and the variants that lost are in `PLATFORMS.md`, round 11.
- **Measured** (`--bench 45`, matched A/B/B/A at the same cap against the 0.5.9 kernels): RTX 3070 11.5 → 12.8 MH/s (+11 %), RTX 5090 64.0 → 68.1 MH/s at 600 W (+6.7 %) and 48.3 → 51.0 at 420 W (+5.7 %), RTX 5080 33.0 → 35.3 MH/s (+7.1 %). CMP 170HX: see `PLATFORMS.md`. Ada and Turing are not measured here; they get the new kernel through the first-start quick tune, which now includes it.
- The 0.5.8 comba variant stays in the binary (`--variant fe32c_wnaf_s16_c`, `--autotune`). The A100 and H100 datacenter profiles are unchanged.
- Build-time options documented for reference: `PX_FOLDSHIFT`, `PX_FOLDSHIFT_POW` (the fold from shifts, slower on sm_86), `PX_LB_SCALAR` (blocks per SM of the scalar kernel).
- Console, thermal limits, API, launchers, packaging and the OpenCL miner are unchanged from 0.5.9.

## Downloads

| Platform | Asset |
|---|---|
| Linux / HiveOS / MMPOS / SMOS, NVIDIA + AMD | `proxichorizo-0.5.10.tar.gz` |
| HiveOS custom miner | `proxichorizo-hiveos.tar.gz` |
| Native Windows x64, NVIDIA | `proxichorizo-0.5.10-windows-x86_64.zip` |

The Linux archives include both the CUDA and OpenCL miners and all three mining-OS launchers. Every archive includes `LICENSE`, `NOTICE`, documentation, `BUILD-INFO.json` and internal checksums. External `.sha256` files and `SHA256SUMS` cover the downloads.

Docker, NVIDIA only: `ocminersupr/proxichorizo:0.5.10` and `:latest`, mirrored at `ghcr.io/ocminer/proxichorizo:0.5.10` and `:latest`.

## Upgrade and validation

Replace the miner; options and launchers are unchanged. A cached first-start tune (`proxichorizo-autotune.cache`) is keyed by miner version and refreshes itself. Linux builds target glibc 2.31; NVIDIA builds need a driver compatible with CUDA 12.9 and the GPU.

Validation of this release: on sm_86 (RTX 3070), sm_120 (RTX 5080, RTX 5090) and sm_80 (CMP 170HX) every kernel variant passes `--selftest vectors.json` (16 node vectors), `--fetest` (128,000 random field checks), 512 independent VRF vectors across 256 keys, and `--verify` (identical hit sets over one nonce range plus the dense launch-shape edge cases); in isolation the new arithmetic agrees with the 0.5.8 arithmetic on 52 million full-range operand pairs. Live pool runs with the final build: see `PLATFORMS.md`, round 11.
