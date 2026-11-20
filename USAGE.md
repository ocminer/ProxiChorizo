# ProxiChorizo — how to mine Proxima (PROX) with it

ProxiChorizo is a CUDA / OpenCL miner ("nonce seeker") for the Proxima fair-launch mine chain.
It works with a **pool** (the pool runs the wallet side and pays you per share) or
**solo** against your own Proxima node and `proxi node mine`. One NVIDIA GPU of the
Pascal SM61, Turing, Ampere, Ada, Hopper or Blackwell generation is enough; several GPUs in one rig are fine.

Requirements: Linux x64 or Windows 10/11 x64. NVIDIA needs a driver compatible
with CUDA 12.9 and the selected GPU; AMD needs a working OpenCL driver. Nothing
is installed by the miner itself.

---

## 1. Pool mining — example: suprnova.cc

The pool is https://proxi.suprnova.cc (PPLNS, 1% pool fee). It builds and signs
mining transactions; your wallet key stays with you.

1. Copy your PROX address: the wallet's 64-hex holder ID (the `a/` prefix is optional).
   No registration, new token or key download is required.
2. Download the miner for your platform.
3. Start it using address login:

   ```sh
   ./proxichorizo -o stratum+tcp://proxi.suprnova.cc:3355 -u YOUR_PROX_ADDRESS.rig1 -p x
   ```

   Windows: use `proxichorizo.exe`; AMD: `proxichorizo-cl` / `proxichorizo-cl.exe`.
   The part after the dot is your worker name. For TLS use
   `stratum+ssl://proxi.suprnova.cc:3356`. Current jobs need no key file or seed.
   Existing tokens continue to work with `--token YOUR_EXISTING_TOKEN --name rig1`.
   Packaged launchers retain the variable name `TOKEN`: set it to your PROX
   address or existing token. See [current quick start](QUICKSTART-v2.1.1.md).

4. The miner prints a start-up banner (version, licence, pool, worker, the GPUs with
   bus id and kernel) and then, every 30 seconds, one line per GPU and a total:
   Accepted shares are green, rejected ones red, warnings yellow; colours switch off by
   themselves when the output is not a terminal (log files stay plain), or with `--no-color`.
   `--quiet` keeps only the total line, job changes, warnings and errors; `--log-interval 10`
   changes the cadence; `--verbose` adds kernel occupancy and idle diagnostics per GPU and
   reports each launch that a job change stopped early.

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

Legacy VRF jobs (before library upgrade 6) also needed the pool's key, as `--key-file proxima.key`,
`--seed-hex <first 64 hex chars of "private_key">` or the environment variable `NONCE_SEEKER_SEED`.
The options remain for compatibility. For keyless hash startup, omit them and
keep old `proxima.key` files outside the miner working directory; loading a legacy
key can initialize the legacy code even when the pool supplies hash jobs.

### Stratum

From 1.0.0 the miner talks to the pool over stratum (`stratum+tcp://HOST:PORT`): one
persistent connection, on which the pool sends the jobs and the share difficulty and the
miner sends its shares. The pool gives every connection its own nonce range and its own
difficulty, starting low and adjusted to about one share every 20 seconds, so a single GPU
and a large rig both report regularly. The login is `-u ADDRESS.WORKER` (or just the
worker name for existing-token login; without `-u`, the `--name`). Use `-p x` for
address login, or your existing token for the legacy route. Two rigs may use the
same worker name; they still search different nonces. The pool's old HTTP port and 1.x miners are no longer supported; use current Stratum endpoints. HTTP remains available for your own solo seeker.

**Encrypted:** `stratum+ssl://proxi.suprnova.cc:3356` is the same pool over TLS. On Linux the
miner uses the system's OpenSSL (1.1 or 3; present on HiveOS, MMPOS, SMOS and every current
distribution) and CA store; the Windows archive brings OpenSSL's DLLs and a `cacert.pem`. The
pool's certificate and host name are checked; `--ca-file FILE` adds CA certificates,
`--tls-insecure` skips the check (only for a private pool with its own certificate).

**Check the connection** without mining (no GPU needed):
`./proxichorizo --check-pool stratum+ssl://proxi.suprnova.cc:3356` prints `pool OK, …` or why not.

The wire protocol between miner and pool is documented in [STRATUM-SPEC.md](STRATUM-SPEC.md).

## 2. Solo mining — your own node and `proxi node mine`

ProxiChorizo supports the **Blake2b transaction-hash work of library upgrade 6**
through the version-2 external nonce seeker protocol. Run a current, synced Proxima
node and a matching `proxi` wallet (v1.6.3 was used for the current solo check).
The wallet builds, signs and submits each transit; ProxiChorizo searches its nonce.
The wallet key stays with `proxi`: **no key file or seed is needed by ProxiChorizo
for Blake2b jobs**.

1. In the wallet's `proxi.yaml`, enable the seeker endpoint:
   ```yaml
   mine:
       seeker:
           listen: 127.0.0.1:8200
           token: some-long-random-string
   ```
   For a GPU on another machine, bind the endpoint to an address reachable by that
   machine and use a private network or tunnel. The seeker endpoint uses HTTP;
   `--proxi` points to this endpoint, not the node's general API.
2. Start the wallet's mining controller with no CPU search workers:
   ```bash
   proxi node mine --workers 0
   ```
3. Put the same token in a local `seeker.token` file, then start the GPU miner:
   ```bash
   ./proxichorizo --proxi http://127.0.0.1:8200 --token-file seeker.token --name solo1
   ```
   `--devices 0,1` selects GPUs; the default is all supported GPUs. The OpenCL
   executable `proxichorizo-cl` uses the same solo options.

ProxiChorizo recomputes every GPU hit on the CPU, submits only improving work and
continues searching until the job expires or is replaced. A `SOLVED ... accepted`
message means the wallet verified the nonce. Check the wallet's **confirmed transit
... (1 of ours)** message for ledger inclusion: competing solutions can beat an
accepted nonce before settlement. The miner reconnects if the controller restarts.

Legacy version-1 VRF jobs still require `--key-file proxima.key` (an unencrypted key
file) or a seed. That key requirement does not apply to current Blake2b solo mining.
See the [upstream seeker protocol](https://github.com/lunfardo314/proxima/blob/master/kb/external_nonce_seeker.md)
for the job format and the wallet's own documentation for payout configuration.

## 3. HiveOS

Custom miner, installation URL = the `proxichorizo-hiveos.tar.gz` asset of the
release, miner name `proxichorizo`. Flight sheet:

| Field | Value |
|---|---|
| Pool URL | `stratum+tcp://proxi.suprnova.cc:3355` (or your own `proxi node mine` seeker address, `http://HOST:PORT`) |
| Wallet / template | your PROX **address** (an existing pool token also works; solo uses mine.seeker.token) |
| Pass | leave empty for Blake2b; for legacy VRF only, the first 64 hex characters of `private_key` |
| Extra config | optional `KEY=VALUE` lines: `DEVICES=0,1`, `API_BIND=127.0.0.1:4078`, `AUTOTUNE=0`, `EXTRA_ARGS=...`; omit geometry overrides to use automatic GPU profiles. `MINER_GROUPS` overrides the block/workgroup count. |

Hashrate, shares and per-GPU figures come from the miner's stats API on `127.0.0.1:4078`.
`AUTOTUNE=0` passes `--no-autotune` to either backend. You can also use
`EXTRA_ARGS="--no-autotune"`. These options are available from **2.1.1** on CUDA and OpenCL.

## 4. MMPOS and SMOS

The Linux tarball `proxichorizo-<version>.tar.gz` contains `mmpos-run.sh` and `smos-run.sh`:
set `POOL_URL` (job server) and `TOKEN` (your PROX address, existing pool token or solo seeker token) in the miner config
(`WALLET` is an alias for `TOKEN`), optionally `DEVICES`, `VARIANT`, `EXTRA_ARGS`.
Only legacy VRF needs `SEED` (`PASSWORD` is its alias).
MMPOS's API-based "external miner" format is served by `mmp-launch.sh` / `mmp-stats.sh` /
`mmp-external.conf` from the same tarball (pool = job server, user = PROX address or existing token, password = `x` or empty).

## 5. Docker

```bash
docker run -d --name proxichorizo --gpus all --restart unless-stopped -p 4078:4078 \
  -e TOKEN=YOUR_PROX_ADDRESS -e WORKER=docker1 ocminersupr/proxichorizo:latest
```
`POOL_URL` defaults to `stratum+tcp://proxi.suprnova.cc:3355`; for solo, set it to the
reachable HTTP seeker endpoint. `SEED` or `KEY_FILE` (a mounted key) is needed only for legacy VRF;
arguments after the image name go straight to the miner. Statistics at `http://localhost:4078/`.

AMD GPUs use the `-amd` image (OpenCL miner + ROCm OpenCL runtime; the host needs only the amdgpu driver):
```bash
docker run -d --name proxichorizo --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  --restart unless-stopped -p 4078:4078 \
  -e TOKEN=YOUR_PROX_ADDRESS -e WORKER=docker1 ocminersupr/proxichorizo:amd
```

## 5b. AMD GPUs (OpenCL)

Build `opencl/proxichorizo-cl` (see README) on a machine with an OpenCL driver (ROCm, or the
vendor ICD on HiveOS/mmpOS). Usage is identical: `./proxichorizo-cl --proxi ... --token ... --name rig1`;
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
--token T            stratum password (-p, --pass): x for ADDRESS.worker login, or an existing pool token / solo mine.seeker.token
--token-file FILE    read the token from a file instead (keeps it off the command line); PROXI_TOKEN in the environment works too
--key-file FILE      unencrypted proxima.key (or --seed-hex HEX, or env NONCE_SEEKER_SEED); VRF jobs only:
                     with a stratum pool the miner runs without a key and then mines the hash jobs only
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
--autotune [SECS]    try every legacy VRF kernel variant on each GPU first (6 s per trial)
--no-autotune        skip legacy VRF tuning trials; use a valid cached profile or platform default
--variant NAME       force one kernel variant (--list-variants); default: per-GPU-generation table
--threads N --iters N --blocks N   launch geometry (defaults from the table)
--bench [SECS]       measure the hashrate and exit      --selftest vectors.json   verify the math
--hash-selftest      check the txid-blake2b kernels on every GPU (upgrade-6 vectors, all 32 nonce positions vs the CPU)
--hash-bench [SECS]  measure the txid-blake2b hash rate (GH/s) and exit; --hash-prefix-len N (default 600)
--hash-launch-ms N   target length of one hash launch (default 40)
```

The stats API (`--api-bind`) answers `GET /` with JSON: `hashrate` (H/s), `uptime`, `shares`
(`accepted`, `rejected`, `stale`), `job_id`, `job_k`, `gpus_active`, `max_temp`, `max_mem_temp`
and one `gpus[]` entry per GPU with `id`, `cuda_device`, `name`, `bus_id`, `bus_number`,
`hashrate`, `temp`, `mem_temp`, `fan`, `power`, `clock`, `status` (`mining`, `paused`, `dead`),
`paused`, `reason`, `accepted`, `rejected`, `stale`. Fields are only ever added, never removed.
2.0 adds `algo` (`vrf` or `txid-blake2b`, the kind of the last job), `paused` (the ledger pause of
library upgrade 6) and `invalid` (GPU results that failed the CPU check and were not sent).
`hashrate` stays in H/s for both kinds of work (hash work: about 10^10 H/s per GPU).

Hash work (txid-blake2b, 2.0): RTX 5090 ≈ 16.1 GH/s at 450 W (17.3 at 575 W), RTX 5070 Ti ≈ 7.3 GH/s, RX 7900 XTX ≈ 6.3 GH/s,
MI50 ≈ 2.8 GH/s, RX 7600 XT ≈ 2.2 GH/s (see README). VRF work (older figures): RTX 5090 ≈ 29 MH/s, RTX 5080 ≈ 21 MH/s, RTX 3070 ≈ 7.4 MH/s, CMP 170HX ≈ 7 MH/s. The
work is pure integer ALU; memory, PCIe link width and risers do not matter.

## 6b. Kernel choice per card

Current Blake2b work uses its own per-card kernel comparison and mandatory
correctness checks. These brief checks remain enabled with `--no-autotune`.
Keyless OpenCL pool mining skips legacy VRF compilation and tuning entirely.

For legacy VRF work, measured generations start with their platform profile.
Unmeasured cards normally run a bounded first-start comparison and keep its result
in `proxichorizo-autotune.cache` in the miner's working directory (HiveOS: the
miner directory). Driver compilation can take much longer than the timed trials.
`--no-autotune` suppresses those trials, uses a valid saved profile when available,
and otherwise keeps the platform default. Cached profiles are selected before
compilation so the unused default variant is not compiled first. Delete the cache
to remeasure, or use `--autotune` for a full sweep; it cannot be combined with
`--no-autotune`. The startup line reports the chosen kernel and launch shape.

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
- Windows: set `TOKEN` in `start.cmd` to your PROX address or existing token; `examples\start-suprnova*.cmd` are ready-made alternatives (TCP and TLS).

The source and issue tracker: https://github.com/ocminer/ProxiChorizo
