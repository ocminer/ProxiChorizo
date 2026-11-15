# ProxiChorizo 2.0.5

Easier setup for mining at suprnova.cc. Mining kernels and hash rates are unchanged
from 2.0.4.

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
