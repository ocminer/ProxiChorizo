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
   ./proxichorizo --proxi stratum+tcp://proxi.suprnova.cc:3355 --token YOUR_TOKEN --key-file proxima.key --name rig1
   ```
   Windows
   ```bat
   proxichorizo.exe --proxi stratum+tcp://proxi.suprnova.cc:3355 --token YOUR_TOKEN --key-file proxima.key --name rig1
   ```
   `--name` is the worker name shown on the pool; use a different one per rig/GPU.

4. The miner prints a start-up banner (version, licence, pool, worker, the GPUs with
   bus id and kernel) and then, every 30 seconds, one line per GPU and a total:
   ```
   12:04:31 GPU #0: RTX PRO 6000 Blackwell Server Edition [05:00.0] 63.52 MH/s, 73C, 548 W, 2422 MHz, fan n/a
   12:04:31 GPU #1: RTX PRO 6000 Blackwell Server Edition [06:00.0] 63.47 MH/s, 75C, 555 W, 2422 MHz, fan n/a
   12:04:31 Total: 253.95 MH/s on 4/4 GPUs | accepted 61, rejected 0, stale 1 | job 1792 k=28 | uptime 00:10:30
   12:04:33 GPU #2: share accepted (62/63)
   job 1793: slot 11235, k=28, ttl 9680 ms
   ```
   Accepted shares are green, rejected ones red, warnings yellow; colours switch off by
   themselves when the output is not a terminal (log files stay plain), or with `--no-color`.
   `--quiet` keeps only the total line, job changes, warnings and errors; `--log-interval 10`
   changes the cadence; `--verbose` adds kernel occupancy and idle diagnostics per GPU.

**Several GPUs in one rig: start the miner once, without any GPU argument.** It takes every
GPU in the machine in PCI bus order (`GPU #0 … GPU #7`, the numbers nvidia-smi shows), runs
them from one process and gives each GPU its own nonce range. `-d 0,1,2,3` (the same as
`--gpus` / `--devices`) restricts it to those cards. A GPU that fails with a device error is
marked dead and reported; the others keep mining (the process exits only when no GPU is left).

**Temperature:** every GPU is watched by the miner itself. At `--max-temp` (default 88 °C core)
or `--max-mem-temp` (default 92 °C memory, on cards that report it) only that GPU pauses, with
a yellow log line, and resumes by itself once it has cooled 8 °C below the limit (after at least
30 s); the other GPUs keep mining. `--max-temp 0` disables it. The stats API shows a paused GPU as
`"status":"paused","reason":"thermal"`.

Instead of `--key-file` you may pass the key as `--seed-hex <first 64 hex chars of
"private_key" in proxima.key>` or in the environment variable `NONCE_SEEKER_SEED` —
that is what the HiveOS/mmpOS/Docker packages do.

### Stratum

From 1.0.0 the miner talks to the pool over stratum (`stratum+tcp://HOST:PORT`): one
persistent connection, on which the pool sends the jobs and the share difficulty and the
miner sends its shares. The pool gives every connection its own nonce range and its own
difficulty, starting low and adjusted to about one share every 20 seconds, so a single GPU
and a large rig both report regularly. The login is `-u ADDRESS.WORKER` (or just the
worker name; without `-u`, the `--name`), the password is your token. Two rigs may use the
same worker name; they still search different nonces. The pool's old HTTP port
(`http://proxi.suprnova.cc:8100`) works with 0.5.x miners until it is switched off.

**Encrypted:** `stratum+ssl://proxi.suprnova.cc:3356` is the same pool over TLS. On Linux the
miner uses the system's OpenSSL (1.1 or 3; present on HiveOS, MMPOS, SMOS and every current
distribution) and CA store; the Windows archive brings OpenSSL's DLLs and a `cacert.pem`. The
pool's certificate and host name are checked; `--ca-file FILE` adds CA certificates,
`--tls-insecure` skips the check (only for a private pool with its own certificate).

**Check the connection** without mining (no GPU needed):
`./proxichorizo --check-pool stratum+ssl://proxi.suprnova.cc:3356` prints `pool OK, …` or why not.

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
| Pool URL | `stratum+tcp://proxi.suprnova.cc:3355` (or your own `proxi node mine` seeker address, `http://HOST:PORT`) |
| Wallet / template | your pool **token** (or the mine.seeker.token) |
| Pass | the **seed**: first 64 hex characters of `private_key` in your key file |
| Extra config | optional `KEY=VALUE` lines: `DEVICES=0,1`, `API_BIND=127.0.0.1:4078`, `EXTRA_ARGS=...`; omit geometry overrides to use automatic GPU profiles. `MINER_GROUPS` overrides the block/workgroup count. |

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
`POOL_URL` defaults to `stratum+tcp://proxi.suprnova.cc:3355`; `KEY_FILE` (a mounted key) replaces `SEED`;
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
--proxi URL          the pool: stratum+tcp://proxi.suprnova.cc:3355 or stratum+ssl://proxi.suprnova.cc:3356 (stratum),
                     or the HTTP job server of your proxi node mine (http://HOST:PORT); -o and --url are the same
-u ADDRESS.WORKER    stratum login (--user): your PROX address and worker name, or just the worker name (default: --name)
--ca-file FILE       CA certificates (PEM) for stratum+ssl:// in addition to the system store
--tls-insecure       stratum+ssl:// without checking the pool's certificate (private pools only)
--check-pool URL     connect to a stratum pool, subscribe, print the answer and exit
--token T            the account token from the pool registration (stratum password; -p, --pass) / mine.seeker.token
--token-file FILE    read the token from a file instead (keeps it off the command line); PROXI_TOKEN in the environment works too
--key-file FILE      unencrypted proxima.key (or --seed-hex HEX, or env NONCE_SEEKER_SEED)
--name NAME          worker name shown by the pool (default proxichorizo-<pid>)
-d 0,1,2,3           GPUs to use (default: all of them, one process, numbered in PCI bus order like nvidia-smi;
                     --gpus and --devices are the same option)
--api-bind [H:]PORT  statistics: JSON at http://H:PORT/, mmpOS line at /mmpos (default off)
--max-temp N         pause a GPU at N C core temperature, resume 8 C lower after at least 30 s (default 88; 0 = off); only that GPU pauses
--max-mem-temp N     the same for the memory temperature where the driver reports one (default 92; 0 = off)
--log-interval N     seconds between the per-GPU and total lines (default 30)
--quiet              only the total line, job changes, warnings and errors
--verbose            kernel occupancy and idle diagnostics on the per-GPU lines
--no-color           plain output (automatic when stderr is not a terminal or NO_COLOR is set)
--autotune [SECS]    try every kernel variant on each GPU first, use the fastest (6 s per trial)
--variant NAME       force one kernel variant (--list-variants); default: per-GPU-generation table
--threads N --iters N --blocks N   launch geometry (defaults from the table)
--bench [SECS]       measure the hashrate and exit      --selftest vectors.json   verify the math
```

The stats API (`--api-bind`) answers `GET /` with JSON: `hashrate` (H/s), `uptime`, `shares`
(`accepted`, `rejected`, `stale`), `job_id`, `job_k`, `gpus_active`, `max_temp`, `max_mem_temp`
and one `gpus[]` entry per GPU with `id`, `cuda_device`, `name`, `bus_id`, `bus_number`,
`hashrate`, `temp`, `mem_temp`, `fan`, `power`, `clock`, `status` (`mining`, `paused`, `dead`),
`paused`, `reason`, `accepted`, `rejected`, `stale`. Fields are only ever added, never removed.

Typical rates: RTX 5090 ≈ 29 MH/s, RTX 5080 ≈ 21 MH/s, RTX 3070 ≈ 7.4 MH/s, CMP 170HX ≈ 7 MH/s. The
work is pure integer ALU; memory, PCIe link width and risers do not matter.

## 6b. Kernel choice per card

Measured generations (RTX 30xx/50xx, PRO 6000, A100/CMP 170HX; RX 7900 XTX, MI50) start with their
measured kernel at once. Any other card is measured on first start (~15 s, two candidate kernels) and
the result kept in `proxichorizo-autotune.cache` in the directory the miner runs from (HiveOS: the
miner directory); delete the file to re-measure, or pass `--autotune` for the full sweep. The start-up
line shows the kernel, the launch shape and the reason.

## 7. Troubleshooting

- `cannot fetch the job: HTTP 401` — wrong token. `HTTP 409` — the pool moved on to the
  next height while you were submitting; normal, nothing to do.
- `the miner mines under key … but this seeker holds …` — the key file does not belong to
  the token (pool) or to the wallet (solo). Use the key the pool gave you / the wallet's key.
- `no job (miner idle)` for long — the job server is between rounds or unreachable; check
  the URL and the pool status page.
- Low kernel percentage in `--verbose` lines — too little work per launch: raise `--iters`
  (4 on most GPUs, 2 on a CMP 170HX).
- `GPU #n: … reaches the limit … paused` — that card is too hot; it resumes by itself 8 °C
  below the limit (after at least 30 s) while the others keep mining. Improve the cooling or lower its power cap;
  `--max-temp` raises the limit if your card is rated for it.
- `GPU #n: CUDA error … this GPU is disabled` — a device error (Xid, fallen off the bus) on
  one card; the others keep mining. Check `dmesg`/`nvidia-smi` and restart the miner once the
  card is back. The process exits with code 2 only when no working GPU is left.
- Windows: `start.cmd` expects `proxima.key` in the unpacked folder; `start.ps1 -KeyFile` points elsewhere.

The source and issue tracker: https://github.com/ocminer/ProxiChorizo
