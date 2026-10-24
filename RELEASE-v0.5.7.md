# ProxiChorizo 0.5.7

Faster mining on A100, H100, RX 7900 XTX, RX 7600 XT and MI50, with measured defaults selected automatically for each supported GPU. Existing CMP 170HX, RTX PRO 6000 and other NVIDIA profiles are preserved.

## Measured improvements

Matched optimization measurements included in this release:

| GPU | Power cap | Previous defaults | Optimized defaults | Gain |
|---|---:|---:|---:|---:|
| A100-SXM4-80GB | 400 W | 11.329 MH/s | 12.699 MH/s | **+12.09%** |
| H100 80GB HBM3 | 700 W | 21.181 MH/s | 21.626 MH/s | **+2.10%** |
| Radeon RX 7900 XTX | 244 W | 11.016 MH/s | 11.271 MH/s | **+2.31%** |
| Radeon RX 7600 XT | 155 W | 3.818 MH/s | 3.999 MH/s | **+4.73%** |
| Radeon Instinct MI50, sample 1 | 100 W | 4.687 MH/s | 4.837 MH/s | **+3.20%** |
| Radeon Instinct MI50, sample 2 | 100 W | 4.583 MH/s | 4.748 MH/s | **+3.59%** |

Each comparison used the same GPU, driver, mining key and power cap, averaging two measurements per build in an A–B–B–A sequence. Measurements lasted 45 seconds per stage on A100, H100 and RX 7900 XTX, 30 seconds on MI50, and 15 seconds on RX 7600 XT. A100/H100 measurements used CUDA 13.4 builds of the optimized source; distribution builds use CUDA 12.9 for compatibility. The H100 control was a native build of the previous source. These are benchmark rates at the listed caps; pool rates vary with job changes, clocks, cooling and the mining key.

## Fixes and packaging

- H100 support is now included in every NVIDIA distribution build, fixing the missing kernel image in the previous Linux archive.
- Corrected a benchmark buffer overrun.
- HiveOS now preserves automatic GPU settings unless an override is explicitly configured. The optional `MINER_GROUPS` setting replaces the conflicting Bash `GROUPS` variable.
- The MMPOS external launcher can select the AMD/OpenCL miner and no longer prints pool credentials in its startup message.
- `--help` and `-h` exit successfully without requiring a GPU.
- Release builds check launcher behavior, GPU profile selection, packaged license notices, versions and checksums before publication. The Windows executable also passes a startup check.

## Downloads

| Platform | Asset |
|---|---|
| Linux / HiveOS / MMPOS / SMOS, NVIDIA + AMD | `proxichorizo-0.5.7.tar.gz` |
| HiveOS custom miner | `proxichorizo-hiveos.tar.gz` |
| Native Windows x64, NVIDIA | `proxichorizo-0.5.7-windows-x86_64.zip` |

The Linux archives include both the CUDA and OpenCL miners and all three mining-OS launchers. Every archive includes `LICENSE`, `NOTICE`, documentation, `BUILD-INFO.json` and internal checksums. External `.sha256` files and `SHA256SUMS` cover the downloads.

Docker, NVIDIA only: `ocminersupr/proxichorizo:0.5.7` and `:latest`, mirrored at `ghcr.io/ocminer/proxichorizo:0.5.7` and `:latest`.

## Upgrade and validation

Replace the miner and remove old `--variant`, `--groups`, `--blocks` and `--iters` overrides to use automatic defaults. Explicit overrides remain supported. AMD users need a working OpenCL driver; Linux builds target glibc 2.31. NVIDIA builds require a driver compatible with CUDA 12.9 and the GPU.

The optimized source passed independent VRF vectors, randomized field checks, complete-output pipeline comparisons and live pool mining on Linux. The Windows build is validated by compilation, package checks and executable startup on CI; Windows GPU hardware execution has not been tested.

## License

ProxiChorizo License 1.0 remains unchanged. Mining, including for profit, is allowed. Incorporation into commercial or closed-source miners requires written permission from suprnova.cc. This is source-available software; earlier grants remain unchanged.
