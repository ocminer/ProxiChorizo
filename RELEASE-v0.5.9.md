# ProxiChorizo 0.5.9

A console and launch-behaviour release: the miner takes every GPU of the machine in one process by default, pauses an overheating GPU on its own, keeps mining when one GPU fails, and reports like a classic GPU miner. The kernels are the 0.5.8 kernels, byte for byte, so hashrates are unchanged.

## What changed

- **Launch:** without a GPU argument the miner mines on every GPU, in one process, numbered `GPU #0 … GPU #n` in PCI bus order (the order `nvidia-smi` shows; `CUDA_DEVICE_ORDER=PCI_BUS_ID` is set by the miner unless you set it yourself). `-d 0,1,2,3` restricts it to those GPUs (`--gpus` and `--devices` are the same option); unknown or repeated indices are reported instead of silently ignored.
- **One GPU failing does not stop the others:** a CUDA error on one GPU marks that GPU dead (red log line, `"status":"dead"` in the API) and its worker ends; the other GPUs keep mining. The process exits with code 2 only when no working GPU is left.
- **Per-GPU thermal limit inside the miner:** `--max-temp N` (default 88 °C core) and `--max-mem-temp N` (default 92 °C memory, on GPUs that report it) pause only the hot GPU, with a yellow log line, and resume it once it has cooled 8 °C below the limit. `0` disables a limit. The API shows `"status":"paused","paused":true,"reason":"thermal"` for a paused GPU. A GPU without a readable temperature sensor is never paused (and says so at start-up).
- **Console** in the style of ccminer/sgminer: a start-up banner (version, licence, pool, worker, key, every GPU with bus id, architecture, kernel variant and launch shape, thermal limits, stats API), then every `--log-interval` seconds (default 30) one line per GPU — `GPU #0: RTX PRO 6000 Blackwell Server Edition [05:00.0] 63.52 MH/s, 73C, 548 W, 2422 MHz, fan n/a` — and a total line with accepted/rejected/stale counters, the current job and the uptime. Share results name the GPU that found them (`GPU #2: share accepted (62/63)`). Colours: green accepted, red rejected/errors, yellow warnings; off with `--no-color`, `NO_COLOR` in the environment, or automatically when the output is not a terminal, so log files stay plain. `--quiet` keeps only the total line, job changes, warnings and errors; `--verbose` appends kernel occupancy and idle diagnostics. Ctrl-C prints a final summary. The HH:MM:SS prefix of the log lines is unchanged.
- **Share accounting:** accepted, rejected and stale (409, the pool had moved on) are counted separately, in total and per GPU. In 0.5.8 a share the pool rejected with HTTP 422 was logged but not counted; both counters are exact now.
- **Stats API (`--api-bind`):** fields were added, none removed or renamed. New: `shares.stale`, `job_id`, `gpus_active`, `max_temp`, `max_mem_temp`; per GPU `mem_temp`, `clock` (SM MHz), `status`, `paused`, `reason`, `accepted`, `rejected`, `stale`. The HiveOS, mmpOS and SMOS stats scripts work unchanged.
- **OpenCL miner (AMD):** same console, thermal limits, device selection and API through the shared host code; memory temperature and clock come from the kernel's hwmon files.
- Kernels, platform profiles and the Windows build are unchanged.

## Downloads

| Platform | Asset |
|---|---|
| Linux / HiveOS / MMPOS / SMOS, NVIDIA + AMD | `proxichorizo-0.5.9.tar.gz` |
| HiveOS custom miner | `proxichorizo-hiveos.tar.gz` |
| Native Windows x64, NVIDIA | `proxichorizo-0.5.9-windows-x86_64.zip` |

The Linux archives include both the CUDA and OpenCL miners and all three mining-OS launchers. Every archive includes `LICENSE`, `NOTICE`, documentation, `BUILD-INFO.json` and internal checksums. External `.sha256` files and `SHA256SUMS` cover the downloads.

Docker, NVIDIA only: `ocminersupr/proxichorizo:0.5.9` and `:latest`, mirrored at `ghcr.io/ocminer/proxichorizo:0.5.9` and `:latest`.

## Upgrade and validation

Replace the miner. If you ran one process per GPU with `CUDA_VISIBLE_DEVICES`, that still works; the recommended way is now one process without a GPU argument (or with `-d`). External per-card temperature guards can be retired in favour of `--max-temp`. Linux builds target glibc 2.31; NVIDIA builds need a driver compatible with CUDA 12.9 and the GPU.

Validation of this release: the device code of the 0.5.9 build is identical to 0.5.8 (kernel machine code compared instruction by instruction), `--selftest vectors.json` and `--fetest` pass, and a live run on four RTX PRO 6000 (one process, all GPUs) mined at the 0.5.8 rate with accepted shares and no rejects while the per-GPU pause, the `-d` selection and a simulated device failure on one GPU were exercised.
