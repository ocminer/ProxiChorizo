# ProxiChorizo 2.0.4

Faster Blake2b mining on RTX 5090, RTX 5080 and RTX 5070 Ti, with protected
kernel payloads across all release formats.

| GPU | Power limit | v2.0.3 | 2.0.4 kernels | Gain | Measured watts, before → after |
|---|---:|---:|---:|---:|---:|
| RTX 5090 | 600 W | 16.902 GH/s | 19.857 GH/s | +17.48% | 600 → 600 W |
| RTX 5090 | 450 W | 15.393 GH/s | 16.785 GH/s | +9.04% | 450 → 450 W |
| RTX 5080 | 360 W | 8.351 GH/s | 10.671 GH/s | +27.78% | 281 → 353 W |
| RTX 5070 Ti | 300 W | 6.912 GH/s | 8.883 GH/s | +28.51% | 242 → 297 W |

Matched A/B/B/A Linux development measurements, 369-byte prefix plus 8-byte
nonce, unchanged clock policies and power limits. The 5080 and 5070 Ti use more
of their available power. The README and packaged CSV include drivers, clocks,
run lengths and measured binary hashes. These are not guaranteed pool rates;
Windows GPU performance has not been measured.

Each supported card checks the new path against the CPU and chooses the faster
kernel for each message layout after warming both candidates. Unsupported
models keep their established kernels. Driver compilation failures that indicate
an unavailable PTX compiler retain the native path; incorrect hashes stop startup.
AMD hash calculations and all existing native NVIDIA kernels are retained.
The tested RX 7900 XTX host still consumes approximately one CPU core in the
ROCm async-event runtime, as v2.0.3 did; that driver behavior remains unresolved.

Release CUDA images, Blackwell PTX and OpenCL source payloads are encrypted and
authenticated, decoded only in memory, and checked before packaging. Corrupted
payloads fail closed. The default build uses the protected path.

Downloads cover Linux CUDA and OpenCL, HiveOS, mmpOS/SMOS, native Windows NVIDIA
and native Windows AMD. Docker images are `ocminersupr/proxichorizo:2.0.4` /
`:latest` and `:2.0.4-amd` / `:amd`, mirrored under `ghcr.io/ocminer/proxichorizo`.

ProxiChorizo License 1.0 is unchanged. Packages also include the preserved
Monocypher BSD 2-clause license for the new payload protection dependency.
