ProxiChorizo 0.5.5 is a license release: the first version published under the ProxiChorizo License 1.0. Kernels, options and rates are those of 0.5.4.

## License

- **Mining stays free.** Run the miner on any hardware and any pool, for yourself or as a paid operation, no permission needed.
- **Copy, study, modify and share** it as before, with the source and under the same license, notices intact.
- **New:** taking the code or a substantial part of it (field and point arithmetic, hash-to-curve, VRF attempt, kernel designs, autotuning, pool protocol) into another miner that is commercial (sold, dev fee, paid service) or closed-source needs written permission from suprnova.cc, whether as source, PTX, SPIR-V or binary. Requests: open an issue on GitHub.
- Versions up to and including 0.5.4 keep the MIT-with-restriction license they were released under.

See `LICENSE`, `NOTICE` and the README.

## Changes

- Version string 0.5.5; license notices in `LICENSE`, `NOTICE`, the README and the main source files. No functional change.

## Downloads

| Platform | Asset |
|---|---|
| Linux / HiveOS / MMPOS / SMOS, NVIDIA + AMD | `proxichorizo-0.5.5.tar.gz` |
| HiveOS custom miner | `proxichorizo-hiveos.tar.gz` |
| Native Windows x64 (NVIDIA) | `proxichorizo-0.5.5-windows-x86_64.zip` |

The Linux tarball contains `proxichorizo` (CUDA) and `proxichorizo-cl` (OpenCL), `h-run.sh`, `mmpos-run.sh`, `smos-run.sh`, `USAGE.md` and a checksum manifest; external `.sha256` files and `SHA256SUMS` cover the archives. Docker: `ocminersupr/proxichorizo:0.5.5` and `:latest` (mirror `ghcr.io/ocminer/proxichorizo`), NVIDIA only.

Upgrading: optional; replace the binary if you want the version to show the license it ships under. No option changes.
