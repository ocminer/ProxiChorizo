# ProxiChorizo — how to mine Proxima (PROX) with it

ProxiChorizo is a CUDA miner ("nonce seeker") for the Proxima fair-launch mine chain.
It works with a **pool** (the pool runs the wallet side and pays you per share) or
**solo** against your own Proxima node and `proxi node mine`. One NVIDIA GPU of the
Turing, Ampere, Ada or Blackwell generation is enough; several GPUs in one rig are fine.

Requirements: NVIDIA driver 550 or newer (CUDA 12.x capable), Linux x64 or Windows 10/11
x64. Nothing else is installed by the miner.

---

## 1. Pool mining — example: suprnova.cc

The pool is https://proxi.suprnova.cc. A Proxima transit is signed with the key that
mined it, so a pool cannot let you mine with your own wallet key: at registration it
creates a **mining key** for your account and gives you a **token**. Rewards are credited
to your PROX address by the pool (PPLNS, 1 % fee) and paid out to it.

1. Open https://proxi.suprnova.cc/StartMining.html and register with your **PROX
   address** (the 64-hex holder ID of your wallet, shown by `proxi node balance` as
   `a/<hex>`; type it without the `a/`).
2. The page gives you a **token** and a **key file**. Save the key file as `proxima.key`
   next to the miner. This key can only mine for your account; it holds no funds.
3. Start the miner:

   Linux
   ```bash
   ./proxichorizo --proxi http://proxi.suprnova.cc:8100 --token YOUR_TOKEN --key-file proxima.key --name rig1
   ```
   Windows
   ```bat
   proxichorizo.exe --proxi http://proxi.suprnova.cc:8100 --token YOUR_TOKEN --key-file proxima.key --name rig1
   ```
   `--name` is the worker name shown on the pool; use a different one per rig/GPU.

4. After a minute the log shows lines like
   ```
   GPU 0: 16.90 MH/s avg over 30s | in kernel 95% ... | idle: no job 4%, stale 0%
   job 1792…: slot 11234, k=28, ttl 9680 ms
   ```
   and the pool's dashboard lists your worker. Shares are accepted silently and counted in
   the 30-second rate line; a rejected share is logged.

Several GPUs in one rig: run one process per GPU — it gives the steadiest rates and the
pool hands every worker its own nonce range:
```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 ./proxichorizo ... --name rig1-gpu0 &
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ./proxichorizo ... --name rig1-gpu1 &
```
(or one process with `--gpus 0,1`; each GPU still searches its own range).

Instead of `--key-file` you may pass the key as `--seed-hex <first 64 hex chars of
"private_key" in proxima.key>` or in the environment variable `NONCE_SEEKER_SEED` —
that is what the HiveOS/mmpOS/Docker packages do.

## 2. Solo mining — your own node and `proxi node mine`

Solo, the reference miner `proxi node mine` builds, proves and submits the transits with
your wallet key and ProxiChorizo only does the nonce search for it (the "external nonce
seeker" protocol of Proxima). You need a synced Proxima access node and the `proxi` wallet
(see https://lunfardo314.github.io/#/participate/proxi).

1. In the wallet profile `proxi.yaml` set the seeker server:
   ```yaml
   mine:
       seeker:
           listen: 0.0.0.0:8200          # where seekers fetch jobs (127.0.0.1:8200 if on the same box)
           token: some-long-random-string
   ```
2. Start the miner without CPU workers (`--workers N` would add CPU search beside the GPUs):
   ```bash
   proxi node mine --workers 0
   ```
3. Start ProxiChorizo with the **same key file as the wallet** and the token from step 1:
   ```bash
   ./proxichorizo --proxi http://your-node:8200 --token some-long-random-string --key-file proxima.key
   ```
   If the key file is passphrase-encrypted, pass the seed with `--seed-hex` instead.

**Library upgrade at slot 29000 (2026-10-21 17:01 UTC, proxima v1.1.0-take1):** from that
slot transits are stamped at tick 83 and carry miner version 5. `proxi node mine` does this
for you, but only from v1.1.0-take1 on — update `proxi` and your node before the slot, or
every transit your older `proxi` builds after it is rejected. ProxiChorizo itself needs no
update (the nonce search is unchanged); pool miners have nothing to do.

`proxi node mine` logs `SOLVED …` / `confirmed` for every transit; the first rewards of a
new account are 15 PROX "opening" payouts until the account is known on the ledger, then
50 and 95 PROX (see the Proxima docs on mine payout options). Expect contests: several
miners solve the same height and the smallest VRF output wins, so a small solo miner wins
rarely but exactly in proportion to its hashrate.

## 3. HiveOS

Custom miner, installation URL = the `proxichorizo-hiveos.tar.gz` asset of the
release, miner name `proxichorizo`. Flight sheet:

| Field | Value |
|---|---|
| Pool URL | `proxi.suprnova.cc:8100` (or your own `proxi node mine` seeker address) |
| Wallet / template | your pool **token** (or the mine.seeker.token) |
| Pass | the **seed**: first 64 hex characters of `private_key` in your key file |
| Extra config | optional `KEY=VALUE` lines: `DEVICES=0,1`, `VARIANT=fe32_w4_lb2_c`, `THREADS=256`, `ITERS=2`, `EXTRA_ARGS=...` |

Hashrate, shares and per-GPU figures come from the miner's stats API on `127.0.0.1:4078`.

## 4. MMPOS and SMOS

The Linux tarball `proxichorizo-<version>.tar.gz` contains `mmpos-run.sh` and `smos-run.sh`:
set `POOL_URL` (job server), `TOKEN` (pool token) and `SEED` (as above) in the miner config
(`WALLET`/`PASSWORD` are accepted as aliases), optionally `DEVICES`, `VARIANT`, `EXTRA_ARGS`.
MMPOS's API-based "external miner" format is served by `mmp-launch.sh` / `mmp-stats.sh` /
`mmp-external.conf` from the same tarball (pool = job server, user = token, password = seed).

## 5. Docker

```bash
docker run -d --name proxichorizo --gpus all --restart unless-stopped -p 4078:4078 \
  -e TOKEN=YOUR_TOKEN -e SEED=FIRST_64_HEX_OF_private_key -e WORKER=docker1 ocminersupr/proxichorizo:latest
```
`POOL_URL` defaults to `proxi.suprnova.cc:8100`; `KEY_FILE` (a mounted key) replaces `SEED`;
arguments after the image name go straight to the miner. Statistics at `http://localhost:4078/`.

## 5b. AMD GPUs (OpenCL)

Build `opencl/proxichorizo-cl` (see README) on a machine with an OpenCL driver (ROCm, or the
vendor ICD on HiveOS/mmpOS). Usage is identical: `./proxichorizo-cl --proxi ... --token ... --key-file proxima.key --name rig1`;
`--list-devices` shows the GPUs, `--gpus 0,1` selects them, `--autotune` picks the fastest kernel variant.
The first start compiles the kernels (about a minute per variant) and verifies them against the
built-in test vectors; a variant that does not pass on your driver is disabled automatically.
Expect roughly 10.5 MH/s on an RX 7900 XTX, 3.6 MH/s on an RX 7600 XT and 4.5 MH/s on an MI50
with the 9×29-bit field layout (`fe29_wnaf_c`, the default since 0.5.0; the 0.4.0 numbers were
about half of that). `--iters 8` is the best launch geometry on every AMD card measured so far.

## 6. Options

```
--proxi URL          job server: pool (http://proxi.suprnova.cc:8100) or your proxi node mine
--token T            bearer token from the pool registration / mine.seeker.token
--key-file FILE      unencrypted proxima.key (or --seed-hex HEX, or env NONCE_SEEKER_SEED)
--name NAME          worker name shown by the pool (default proxichorizo-<pid>)
--gpus 0,1           GPUs to use (default all; CUDA order, set CUDA_DEVICE_ORDER=PCI_BUS_ID for PCI order)
--api-bind [H:]PORT  statistics: JSON at http://H:PORT/, mmpOS line at /mmpos (default off)
--autotune [SECS]    try every kernel variant on each GPU first, use the fastest (6 s per trial)
--variant NAME       force one kernel variant (--list-variants); default: per-GPU-generation table
--threads N --iters N --blocks N   launch geometry (defaults from the table)
--bench [SECS]       measure the hashrate and exit      --selftest vectors.json   verify the math
```

Typical rates: RTX 5080 ≈ 17 MH/s, RTX 3070 ≈ 6.9 MH/s, CMP 170HX ≈ 5–6 MH/s. The
work is pure integer ALU; memory, PCIe link width and risers do not matter.

## 7. Troubleshooting

- `cannot fetch the job: HTTP 401` — wrong token. `HTTP 409` — the pool moved on to the
  next height while you were submitting; normal, nothing to do.
- `the miner mines under key … but this seeker holds …` — the key file does not belong to
  the token (pool) or to the wallet (solo). Use the key the pool gave you / the wallet's key.
- `no job (miner idle)` for long — the job server is between rounds or unreachable; check
  the URL and the pool status page.
- Low in-kernel percentage — too little work per launch: raise `--iters` (4 on most GPUs,
  2 on a CMP 170HX), or run one process per GPU.
- Windows: `start.cmd` expects `proxima.key` in the unpacked folder; `start.ps1 -KeyFile` points elsewhere.

The source and issue tracker: https://github.com/ocminer/ProxiChorizo
