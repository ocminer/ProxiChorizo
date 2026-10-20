ProxiChorizo 0.5.3 mines 4–17 % faster on NVIDIA than 0.5.2 by making the scalar-multiplication loop fit the GPU's instruction cache and sharing one field inversion between two points.

## Changes

- **Smaller hot loop**: the wNAF scalar multiplication now has one doubling site (the T coordinate computed only when an addition follows) and one addition site (the digit's sign is an argument), and the table loop is not unrolled. The loop body halves, no register spills remain, and the "no instruction" stalls Nsight Compute showed on Blackwell drop by half.
- **Two points per inversion** (`PX_INV_BATCH 2`): the mult kernel multiplies two points, parks them and inverts once (Montgomery's trick). A loss in 0.5.0 because of the parked points, a gain now that the loop fits the cache.
- **GA100 (A100, CMP 170HX)** switches to the wNAF variant with large launches: the old "4-bit window is faster on GA100" was an instruction-cache effect.

Measured (`--bench`, realistic scalar, defaults):

| GPU | 0.5.2 | 0.5.3 |
|---|---:|---:|
| RTX PRO 6000 Blackwell | 36.7 MH/s | 41.5 |
| RTX 5080 | 18.3 | 21.5 |
| RTX 3070 | 7.06 | 7.37 |
| CMP 170HX | 5.99 | 6.91 |

Also kept in the source, off by default: a 10×26-bit limb representation with 64-bit column products (`PX_FE26`, 1.4× the raw multiply rate but slower overall because of its carry pass) and an L1 prefetch of the next wNAF table entry (`PX_PREFETCH`, no effect). The AMD/OpenCL miner is unchanged apart from the version string.

## Downloads

| Platform | Asset |
|---|---|
| Linux / HiveOS / MMPOS / SMOS, NVIDIA + AMD | `proxichorizo-0.5.3.tar.gz` |
| HiveOS custom miner | `proxichorizo-hiveos.tar.gz` |
| Native Windows x64 (NVIDIA) | `proxichorizo-0.5.3-windows-x86_64.zip` |

The Linux tarball contains `proxichorizo` (CUDA) and `proxichorizo-cl` (OpenCL), `h-run.sh`, `mmpos-run.sh`, `smos-run.sh`, `USAGE.md` and a checksum manifest; external `.sha256` files and `SHA256SUMS` cover the archives. Docker: `ocminersupr/proxichorizo:0.5.3` and `:latest` (mirror `ghcr.io/ocminer/proxichorizo`), NVIDIA only.

Upgrading from 0.5.2: replace the binary; no option changes.
