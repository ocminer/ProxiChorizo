# ProxiChorizo 1.0.0

The stratum release: the miner talks to the pool over **stratum** — one persistent TCP connection, newline-delimited JSON-RPC as in stratum v1 — instead of polling an HTTP job server. The pool gives every connection its own nonce range and its own share difficulty, retargeted to about one share every 15–25 seconds per connection. The kernels are the 0.5.9 kernels, so hashrates are unchanged.

## What changed

- **Stratum:** `--proxi stratum+tcp://proxi.suprnova.cc:3355` (also `-o` / `--url`). The miner subscribes (`mining.subscribe`), authorizes (`mining.authorize`), receives the difficulty (`mining.set_difficulty`, applied from the next job) and the jobs (`mining.notify`; a new job replaces the old one at once), and submits shares (`mining.submit`). Answers are counted as before: accepted, rejected, and stale (`21 job not found`); the other refusals (`22` duplicate, `23` low difficulty, `24` unauthorized, `25` not subscribed) are logged with their reason. A broken connection is re-established with backoff (1 s up to 30 s); `client.reconnect` from the pool is honoured.
- **Login:** `-u ADDRESS.WORKER` (`--user`) with your PROX address and a worker name, or just the worker name; without `-u` the miner logs in with its `--name`. The password is the account token from the pool registration: `--token`, `--token-file` or `PROXI_TOKEN` as before (`-p` / `--pass` are the same option). The key file is still needed: Proxima shares are VRF outputs under the mining key the pool created for your account.
- **Own nonce range per connection:** the pool hands every connection its own 16 bits on top of the nonce (stratum extranonce1); the miner puts the GPU number in the next 8 bits and counts the rest. Two rigs with the same worker name no longer search the same nonces.
- **Share difficulty per connection:** a connection starts low (2^24) and the pool's vardiff raises or lowers it per connection, so a single small GPU and a large rig both report about one share every 20 s. Difficulties are powers of two (the number of trailing zero bits of the VRF output), so the time per share settles within a factor of 1.4 of 20 s.
- **HTTP job servers still work:** `http://HOST:PORT` (or `HOST:PORT`) keeps the 0.5.x protocol for `proxi node mine` (solo) and for pools that have not moved to stratum.
- **Launchers:** HiveOS, MMPOS, SMOS, Docker and the Windows start scripts pass the pool URL through unchanged; the defaults and examples now use `stratum+tcp://proxi.suprnova.cc:3355`.
- Kernels, platform profiles, the console, the thermal limits and the stats API are unchanged.

## Downloads

| Platform | Asset |
|---|---|
| Linux / HiveOS / MMPOS / SMOS, NVIDIA + AMD | `proxichorizo-1.0.0.tar.gz` |
| HiveOS custom miner | `proxichorizo-hiveos.tar.gz` |
| Native Windows x64, NVIDIA | `proxichorizo-1.0.0-windows-x86_64.zip` |

The Linux archives include both the CUDA and OpenCL miners and all three mining-OS launchers. Every archive includes `LICENSE`, `NOTICE`, documentation, `BUILD-INFO.json` and internal checksums. External `.sha256` files and `SHA256SUMS` cover the downloads.

Docker, NVIDIA only: `ocminersupr/proxichorizo:1.0.0` and `:latest`, mirrored at `ghcr.io/ocminer/proxichorizo:1.0.0` and `:latest`.

## Upgrade

Replace the miner and change the pool URL to `stratum+tcp://proxi.suprnova.cc:3355`; token, key file and worker name stay as they are. On HiveOS, MMPOS and SMOS put that URL in the flight sheet's pool field. The pool's HTTP port 8100 keeps working for 0.5.x miners for a transition period and will be switched off later.
