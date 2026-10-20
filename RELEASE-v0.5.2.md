ProxiChorizo 0.5.2 stops idling at pool round boundaries (about +8 % effective hashrate for every pool miner), mines 5–13 % faster on NVIDIA through larger launches, shows GPU temperature, fan and power, and accepts `-d` for device selection. It replaces 0.5.2, which was tagged minutes earlier without the poller fix.

## Changes

- **No more idling between rounds**: a pool retires its job for a few milliseconds at every round boundary (about once per slot); the poller used to sleep 500 ms before asking again, so every GPU sat idle ~5–9 % of the time (`idle: no job` in the 30-second line). It now asks again at once. Measured on an RTX PRO 6000: 33.7 → 36.5 MH/s effective at the same in-kernel rate. CUDA and OpenCL.
- **Larger launches** (CUDA): the compacted pipeline runs its passes over the whole launch, so a launch of 128 blocks per multiprocessor with one nonce per thread amortises the per-pass overhead better than 8 blocks per SM looping over 4 nonces. New default for Blackwell, Ada, Ampere GA10x and Turing: +13 % on an RTX PRO 6000, +10 % on an RTX 5080, +4.5 % on an RTX 3070. GA100 (A100, CMP 170HX) keeps its previous geometry. `--autotune` also tries 64 and 128 blocks per SM; `--blocks` still overrides. A launch is ~150–170 ms.
- **Sensors**: every 30-second GPU line ends with temperature, fan and power (`| 64C, fan 45%, 318 W`), and the stats API carries `temp`, `fan` and `power` per GPU plus `temp`/`fan`/`power` arrays in the mmpOS line. NVIDIA: NVML loaded at run time (Linux and Windows, no new dependency; passive server cards report no fan). AMD on Linux: the kernel's hwmon files. Without either the fields are `null` / omitted.
- **`-d 0,1,2`** (also `--devices`) selects GPUs like in other miners; `--gpus` still works. CUDA and OpenCL.
- Build: `-ldl` on the Linux link lines (dlopen for NVML).

No change to the OpenCL kernels or the AMD defaults.

## Measured performance

| GPU | Miner / variant | 0.5.0 | 0.5.2 |
|---|---|---:|---:|
| RTX PRO 6000 Blackwell | CUDA fe32_wnaf_lb2_c, 24064×256×1 | 32.4 | 36.7 MH/s |
| RTX 5080 | CUDA fe32_wnaf_lb2_c, 10752×256×1 | 17.4 | 19.1 |
| RTX 3070 | CUDA fe32_wnaf_lb2_c, 5888×256×1 | 6.8 | 7.1 |
| CMP 170HX | CUDA fe32_w4_lb2_c 256×2 (unchanged) | 4.6 – 6.3 | same |
| RX 7900 XTX (ROCm 7.1) | OpenCL fe29_wnaf_c ×8 (unchanged) | ≈ 10.5 | same |

## Downloads

| Platform | Asset |
|---|---|
| Linux / HiveOS / MMPOS / SMOS, NVIDIA + AMD | `proxichorizo-0.5.2.tar.gz` |
| HiveOS custom miner | `proxichorizo-hiveos.tar.gz` |
| Native Windows x64 (NVIDIA) | `proxichorizo-0.5.2-windows-x86_64.zip` |

The Linux tarball contains `proxichorizo` (CUDA) and `proxichorizo-cl` (OpenCL), `h-run.sh`, `mmpos-run.sh`, `smos-run.sh`, `USAGE.md` and a checksum manifest; external `.sha256` files and `SHA256SUMS` cover the archives. Docker: `ocminersupr/proxichorizo:0.5.2` and `:latest` (mirror `ghcr.io/ocminer/proxichorizo`), NVIDIA only.

Upgrading from 0.5.0: replace the binary; no option changes needed. Pool miners need nothing for the Proxima library upgrade at slot 29000 (see README).
