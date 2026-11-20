# ProxiChorizo 2.1.1 on HiveOS

Use the custom miner package for NVIDIA or AMD:
https://github.com/ocminer/ProxiChorizo/releases/download/v2.1.1/proxichorizo-hiveos.tar.gz

| Flight sheet field | Value |
|---|---|
| Miner | Custom; name `proxichorizo` |
| Wallet and worker template | Your pool token from https://proxi.suprnova.cc/StartMining.html |
| Pool URL | `stratum+tcp://proxi.suprnova.cc:3355` |
| TLS alternative | `stratum+ssl://proxi.suprnova.cc:3356` |
| Pass / SEED | Leave empty for current Blake2b jobs |
| Hash algorithm | `txid-blake2b` (display only) |

The HiveOS worker name becomes the pool worker name. The launcher selects CUDA
when NVIDIA is available, otherwise OpenCL. For an explicit backend use
`MINER=cuda` or `MINER=cl`; `DEVICES=0,1` selects devices within that backend.

## Skip legacy autotuning

Add this to **Extra config**, one setting per line:

```sh
AUTOTUNE=0
```

Alternatively use `EXTRA_ARGS="--no-autotune"`. When combining options, keep the
whole value quoted, for example `EXTRA_ARGS="--no-autotune --max-temp 85"`.
Do not combine `--no-autotune` with `--autotune`.

The flag skips legacy VRF tuning trials and uses a valid cached result or platform
default. Required driver compilation, correctness checks and the brief Blake2b
kernel comparison remain. In keyless Blake2b mode, OpenCL skips legacy VRF program
compilation entirely. A cached legacy profile now compiles only its selected
variant. These changes address repeated startup work, not a new AMD hashrate gain.
The reported ten-card RX 5700 XT setup was not available for a physical timing test.

Long jobs are negotiated automatically on both backends. Older pools and ordinary
jobs still work. The miner does not change clocks, fans or power allowances.
See the packaged README and USAGE for benchmarks, solo mining and API details.
