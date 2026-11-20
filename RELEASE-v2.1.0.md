# ProxiChorizo 2.1.0

Automatic support for the optional `proxima.aligned-tail.v1` job format on all
NVIDIA CUDA and AMD OpenCL platforms, including Pascal and mixed-GPU rigs.
Each card measures the available kernels and keeps its faster valid path.
Ordinary jobs and older pools remain compatible, with the same mining command.

Padded jobs permit an exact zero-tail Blake2b specialization. Short Linux
development comparisons on the same 504-byte layout measured RX 7900 XTX
6.115 → 7.237 GH/s (+18.35%) and RTX 4070 Ti SUPER 6.759 → 7.346 GH/s (+8.69%).
These eight-second diagnostics are not sustained benchmarks or earnings claims.
The tested GTX 1080 Ti keeps its existing faster kernel; no Pascal gain is claimed.
See the README for measurement context and earlier ordinary-job benchmarks.

The development integration recorded 3,606 accepted staging shares, zero invalid
hashes or non-stale rejects, and five job-change stales. Sixteen padded transactions
were independently confirmed by the test node and verified with the official
transaction parser, full hashes and signatures. Staging block records were checked;
production PPLNS credits and payouts were not part of that test.

Pool maintainers can use [PROTOCOL-UPGRADE](https://github.com/ocminer/ProxiChorizo/blob/v2.1.0/PROTOCOL-UPGRADE.md) and the included
public hash vectors to implement the extension alongside their ordinary protocol.
The guide covers negotiation, canonical transaction padding, shared job IDs across
connections, late negotiation, validation and fallback. No upgrade-7 wait is part
of the protocol.

Every submitted GPU hit remains CPU-verified. Multi-GPU padded searches keep
disjoint nonce domains across kernel types. No power, clock or fan settings are
changed. Kernels are plain source and ordinary CUDA/OpenCL payloads, without
encryption or obfuscation. ProxiChorizo License 1.0 is unchanged.

Release packages cover Linux CUDA/OpenCL, native Windows NVIDIA/AMD, HiveOS,
mmpOS, SMOS and NVIDIA/AMD Docker images. Physical Windows GPU execution is not
qualified; the native compiler, CPU checks, startup, packaged TLS and GPU payload
checks are used for those builds.

Pool: https://proxi.suprnova.cc · Setup: https://proxi.suprnova.cc/StartMining.html

Docker: `ocminersupr/proxichorizo:2.1.0` / `:latest`, and
`ocminersupr/proxichorizo:2.1.0-amd` / `:amd`, mirrored on `ghcr.io/ocminer/proxichorizo`.
