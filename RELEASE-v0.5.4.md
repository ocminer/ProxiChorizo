ProxiChorizo 0.5.4 brings the 0.5.3 kernel improvements to AMD (+3.8 % on an RX 7900 XTX) and makes every card get a kernel chosen on itself.

## Changes

- **AMD / OpenCL**: the scalar-multiplication loop has one doubling and one addition site (as the CUDA kernel since 0.5.3) and two points share one inversion. RX 7900 XTX: 10.72 → 11.13 MH/s in `--bench`.
- **First-start tuning**: GPU generations this build has no measurements for (Ada RTX 40xx, Turing, RDNA2/RDNA4, unknown) are measured on the card at the first start (two candidate kernels at two launch shapes, ~15 s) and the winner is kept in `proxichorizo-autotune.cache` next to the miner, keyed by card and version. Measured generations (RTX 30xx/50xx, RTX PRO 6000, A100/CMP 170HX, RX 7900 XTX, MI50) start immediately as before. The start-up line names the kernel, the launch shape and the reason; `--autotune`, `--variant`, `--blocks`, `--iters` override as before.
- NVIDIA kernels unchanged from 0.5.3.

## Downloads

| Platform | Asset |
|---|---|
| Linux / HiveOS / MMPOS / SMOS, NVIDIA + AMD | `proxichorizo-0.5.4.tar.gz` |
| HiveOS custom miner | `proxichorizo-hiveos.tar.gz` |
| Native Windows x64 (NVIDIA) | `proxichorizo-0.5.4-windows-x86_64.zip` |

The Linux tarball contains `proxichorizo` (CUDA) and `proxichorizo-cl` (OpenCL), `h-run.sh`, `mmpos-run.sh`, `smos-run.sh`, `USAGE.md` and a checksum manifest; external `.sha256` files and `SHA256SUMS` cover the archives. Docker: `ocminersupr/proxichorizo:0.5.4` and `:latest` (mirror `ghcr.io/ocminer/proxichorizo`), NVIDIA only.

Upgrading: replace the binary; no option changes.
