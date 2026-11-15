# ProxiChorizo 2.0.5

Easier setup for mining at suprnova.cc, with all of 2.0.4's per-card hash optimizations retained.

The CUDA and OpenCL builds now compile the published kernel sources directly.
All per-card optimizations from 2.0.4 are retained. Matched Linux A/B/B/A checks
against the published 2.0.4 binaries used 40-second runs, a 369-byte prefix and
unchanged clocks/power settings:

| GPU | Power limit | v2.0.4 GH/s | v2.0.5 check GH/s | Difference |
|---|---:|---:|---:|---:|
| RTX 5090 | 600 W | 19.728 | 19.707 | -0.11% |
| RTX 5080 | 360 W | 10.663 | 10.656 | -0.07% |
| RTX 5070 Ti | 300 W | 8.882 | 8.881 | -0.01% |
| RTX 3070 | 240 W | 3.204 | 3.204 | +0.02% |
| RX 7900 XTX | 303 W | 6.046 | 6.026 | -0.34% |

These are regression checks with platform-specific Linux validation builds,
not a new performance-gain claim or guaranteed pool rates. The small differences
are within the observed run variation. The [measurement CSV](benchmarks/hash-2.0.5-regression-2026-11-15.csv)
records binary hashes, power draw and test details. Full-reference hash checks
passed on all five cards. The 5090 also passed the legacy VRF vectors; NVIDIA
native fallback paths passed. Physical Windows GPU performance remains unmeasured.

**New: ready-to-edit example configurations** in the `examples/` folder of every package:

- **Linux:** `start-suprnova.sh` (TCP), `start-suprnova-ssl.sh` (TLS) and
  `start-suprnova-amd.sh` (OpenCL), plus a `proxichorizo.service` systemd unit for
  headless rigs.
- **Windows:** `start-suprnova.cmd` and `start-suprnova-ssl.cmd` in the NVIDIA and AMD
  packages.
- **HiveOS, mmpOS and SMOS:** the exact flight sheet / miner config values, including a
  HiveOS installation URL that always points to the latest release.
- **Docker:** run commands for NVIDIA and AMD.

You need only your pool token and a worker name. Register at
https://proxi.suprnova.cc/StartMining.html; pool `stratum+tcp://proxi.suprnova.cc:3355`
or `stratum+ssl://proxi.suprnova.cc:3356`.

**Fixed:** the HiveOS, mmpOS, SMOS, Docker and Windows launchers no longer print
"no SEED/KEY_FILE: mining hash jobs only … VRF jobs need the key". Since Proxima library
upgrade 6 the pool sends hash jobs only, and they need no key seed or key file; leave the
password field empty. The documentation and the Docker Compose examples were updated to match.

Downloads cover Linux CUDA and OpenCL, HiveOS, mmpOS/SMOS, native Windows NVIDIA and
native Windows AMD. Docker images are `ocminersupr/proxichorizo:2.0.5` / `:latest` and
`:2.0.5-amd` / `:amd`, mirrored under `ghcr.io/ocminer/proxichorizo`.

ProxiChorizo License 1.0 is unchanged.
