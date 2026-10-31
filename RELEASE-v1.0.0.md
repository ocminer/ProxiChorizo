# ProxiChorizo 1.0.0

The stratum release: the miner talks to the pool over **stratum** — one persistent TCP connection, newline-delimited JSON-RPC as in stratum v1 — instead of polling an HTTP job server. The pool gives every connection its own nonce range and its own share difficulty, retargeted to about one share every 15–25 seconds per connection. It also brings new field arithmetic for NVIDIA GPUs: hashrates rise over 0.5.9 on every NVIDIA card we measured, most on the RTX 30xx series and the CMP 170HX.

## Faster kernels (NVIDIA)

The field multiply and square issue their 32×32-bit products in chains whose carries ride as carry-ins of the next wide multiply, which removes about 40 % of the integer adds and most of the register moves from the hot loop (`fe32w_wnaf_s16_c`, the new default for Blackwell, Ada, Ampere and Turing). Same multiply count as the 0.5.8/0.5.9 product scanning; details, the profile that led there and the variants that lost are in `PLATFORMS.md`, round 11.

Measured with `--bench 45`, matched A/B/B/A on the same card and power cap against the 0.5.9 release, automatic profiles on both sides:

| GPU | Power cap | 0.5.9 | 1.0.0 | Gain |
|---|---:|---:|---:|---:|
| RTX 5090 | 600 W | 63.9 MH/s | **68.0 MH/s** | +6.4 % |
| RTX 5090 | 420 W | 48.6 MH/s | **51.0 MH/s** | +5.0 % |
| RTX 5080 | 360 W | 33.1 MH/s | **35.5 MH/s** | +7.2 % |
| RTX 3070 (two cards) | 240 W / 220 W | 11.5 MH/s | **12.7 MH/s** | +11.0 % / +10.8 % |
| CMP 170HX (core locked at 1300 MHz) | 250 W | 12.5 MH/s | **14.0 MH/s** | +12.7 % |

Ada and Turing were not measured here; they get the new kernel through the first-start quick tune, which now includes it. The A100 and H100 datacenter profiles and the AMD OpenCL kernels are unchanged. The 0.5.8/0.5.9 variant stays in the binary (`--variant fe32c_wnaf_s16_c`, `--autotune`). A cached first-start tune (`proxichorizo-autotune.cache`) is keyed by miner version and refreshes itself.

## What changed

- **Stratum:** `--proxi stratum+tcp://proxi.suprnova.cc:3355` (also `-o` / `--url`). The miner subscribes (`mining.subscribe`), authorizes (`mining.authorize`), receives the difficulty (`mining.set_difficulty`, applied from the next job) and the jobs (`mining.notify`; a new job replaces the old one at once), and submits shares (`mining.submit`). Answers are counted as before: accepted, rejected, and stale (`21 job not found`); the other refusals (`22` duplicate, `23` low difficulty, `24` unauthorized, `25` not subscribed) are logged with their reason. A broken connection is re-established with backoff (1 s up to 30 s); `client.reconnect` from the pool is honoured.
- **Stratum over TLS:** `stratum+ssl://proxi.suprnova.cc:3356`. The miner loads OpenSSL at run time — the system's libssl (1.1 or 3) on Linux, the DLLs shipped in the Windows archive — and checks the pool's certificate and host name against the system CA store (Windows: the bundled `cacert.pem`). `--ca-file FILE` adds CA certificates, `--tls-insecure` skips the check for a private pool. Plain `stratum+tcp://` needs no OpenSSL.
- **`--check-pool URL`:** connects, subscribes, prints the pool's answer and exits — a connection test without a GPU, also over TLS.
- **Login:** `-u ADDRESS.WORKER` (`--user`) with your PROX address and a worker name, or just the worker name; without `-u` the miner logs in with its `--name`. The password is the account token from the pool registration: `--token`, `--token-file` or `PROXI_TOKEN` as before (`-p` / `--pass` are the same option). The key file is still needed: Proxima shares are VRF outputs under the mining key the pool created for your account.
- **Own nonce range per connection:** the pool hands every connection its own 16 bits on top of the nonce (stratum extranonce1); the miner puts the GPU number in the next 8 bits and counts the rest. Two rigs with the same worker name no longer search the same nonces.
- **Share difficulty per connection:** a connection starts low (2^24) and the pool's vardiff raises or lowers it per connection, so a single small GPU and a large rig both report about one share every 20 s. Difficulties are powers of two (the number of trailing zero bits of the VRF output), so the time per share settles within a factor of 1.4 of 20 s.
- **HTTP job servers still work:** `http://HOST:PORT` (or `HOST:PORT`) keeps the 0.5.x protocol for `proxi node mine` (solo) and for pools that have not moved to stratum.
- **Launchers:** HiveOS, MMPOS, SMOS, Docker and the Windows start scripts pass the pool URL through unchanged; the defaults and examples now use `stratum+tcp://proxi.suprnova.cc:3355`.
- **Kernels:** the wide-chain field arithmetic above is the default on every consumer and CMP NVIDIA generation (platform profiles for sm_120, sm_89, sm_86, sm_80 GA100 and sm_75). Build-time options kept for reference: `PX_FOLDSHIFT`, `PX_FOLDSHIFT_POW`, `PX_LB_SCALAR`.
- The console, the thermal limits and the stats API are unchanged.

## Downloads

| Platform | Asset |
|---|---|
| Linux / HiveOS / MMPOS / SMOS, NVIDIA + AMD | `proxichorizo-1.0.0.tar.gz` |
| HiveOS custom miner | `proxichorizo-hiveos.tar.gz` |
| Native Windows x64, NVIDIA (with OpenSSL 3 DLLs and `cacert.pem` for TLS) | `proxichorizo-1.0.0-windows-x86_64.zip` |

The Linux archives include both the CUDA and OpenCL miners and all three mining-OS launchers. Every archive includes `LICENSE`, `NOTICE`, documentation, `BUILD-INFO.json` and internal checksums. External `.sha256` files and `SHA256SUMS` cover the downloads.

Docker, NVIDIA only: `ocminersupr/proxichorizo:1.0.0` and `:latest`, mirrored at `ghcr.io/ocminer/proxichorizo:1.0.0` and `:latest`.

## Upgrade

Replace the miner and change the pool URL to `stratum+tcp://proxi.suprnova.cc:3355` (or `stratum+ssl://proxi.suprnova.cc:3356`); token, key file and worker name stay as they are. On HiveOS, MMPOS and SMOS put that URL in the flight sheet's pool field. The pool's HTTP port 8100 keeps working for 0.5.x miners for a transition period and will be switched off later.

## Validation

Every kernel variant of the release-style Linux build (CUDA 12.9, glibc 2.31) passes on sm_86 (RTX 3070) and sm_120 (two RTX 5080, RTX 5090): `--selftest vectors.json` (16 node vectors), 512 independent VRF vectors across 256 keys, `--fetest` (128,000 random field checks) and `--verify` (identical hit sets over one nonce range plus 58 dense launch shapes / 5,189 complete outputs). The OpenCL miner on an RX 7900 XTX passes the 16 + 512 vectors, `--fetest` and `--verify-pipeline` (70 shapes / 83,393 outputs). The same kernels passed the 16 + 512 vectors and `--fetest` on sm_80 (CMP 170HX) during development; the A/B rates above were measured then on the identical kernels. Both the CUDA and the OpenCL binaries pass `--check-pool` against `stratum+tcp://` and `stratum+ssl://`.

Live stratum mining with the release-style build on the public pool, about 13 minutes per card (7 minutes for the second XTX run), automatic profiles:

| GPU | Connection | Average | Accepted | Rejected |
|---|---|---:|---:|---:|
| RTX 3070 (240 W) | `stratum+tcp://` | 12.81 MH/s | 58 | 0 |
| RTX 3070 (240 W) | `stratum+ssl://` | 12.98 MH/s | 59 | 0 |
| RTX 5080 (360 W) | `stratum+tcp://` | 35.37 MH/s | 52 | 0 |
| RTX 5080 (360 W) | `stratum+ssl://` | 36.11 MH/s | 43 | 0 |
| RTX 5090 (420 W) | `stratum+ssl://` | 51.95 MH/s | 73 | 0 |
| RX 7900 XTX (OpenCL, 244 W) | `stratum+tcp://` | 11.35 MH/s | 59 | 0 |
| RX 7900 XTX (OpenCL, 244 W) | `stratum+ssl://` | 11.32 MH/s | 56 | 0 |

Three shares in total arrived after a job change and were counted stale.
