# ProxiChorizo — example configurations for suprnova.cc

Ready-to-edit setups for mining Proxima (PROX) on **https://proxi.suprnova.cc**.
Every example needs only two things from you: your **pool token** and a **worker name**.

## Pool settings

| Setting | Value |
|---|---|
| Pool (stratum, TCP) | `stratum+tcp://proxi.suprnova.cc:3355` |
| Pool (stratum, TLS) | `stratum+ssl://proxi.suprnova.cc:3356` |
| Token | from https://proxi.suprnova.cc/StartMining.html (register with your PROX address) |
| Worker | any name, one per rig, e.g. `rig1` |
| Key file / seed | **not needed** — current Proxima jobs (txid-blake2b) need only the token |

Check that the pool is reachable without mining (no GPU needed):
`./proxichorizo --check-pool stratum+ssl://proxi.suprnova.cc:3356`

## Plain Linux

| File | What it does |
|---|---|
| `linux/start-suprnova.sh` | all NVIDIA GPUs, TCP |
| `linux/start-suprnova-ssl.sh` | all NVIDIA GPUs, TLS |
| `linux/start-suprnova-amd.sh` | AMD / OpenCL GPUs (`proxichorizo-cl`), TCP |
| `linux/proxichorizo.service` | systemd unit for a headless rig (starts at boot, restarts on failure) |

Edit `TOKEN` and `WORKER` at the top, then run it from the unpacked miner directory, e.g.
`./examples/linux/start-suprnova.sh`. Extra miner options can be appended: `-d 0,1`
(only these GPUs), `--no-autotune` (skip legacy VRF tuning trials), `--max-temp 85` (per-GPU pause limit, default 88 °C).

## Windows

| File | What it does |
|---|---|
| `windows/start-suprnova.cmd` | NVIDIA package, TCP |
| `windows/start-suprnova-ssl.cmd` | NVIDIA package, TLS |
| `windows-amd/start-suprnova.cmd` | AMD package (`proxichorizo-cl.exe`), TCP |
| `windows-amd/start-suprnova-ssl.cmd` | AMD package, TLS |

Copy the file next to `proxichorizo.exe` (or `proxichorizo-cl.exe`), set `TOKEN` and `WORKER`
in Notepad, save and double-click it. The `start.cmd` shipped in the package works the same way.

## HiveOS

Flight sheet → Coin `PROX` (or any name) → Miner **Custom**:

| Field | Value |
|---|---|
| Miner name | `proxichorizo` |
| Installation URL | `https://github.com/ocminer/ProxiChorizo/releases/latest/download/proxichorizo-hiveos.tar.gz` |
| Hash algorithm | `txid-blake2b` (display only) |
| Wallet and worker template | your **pool token** |
| Pool URL | `stratum+tcp://proxi.suprnova.cc:3355` (or `stratum+ssl://proxi.suprnova.cc:3356`) |
| Pass | leave empty |
| Extra config arguments | optional, one per line: `DEVICES=0,1`, `AUTOTUNE=0`, `EXTRA_ARGS="--no-autotune"` |

The HiveOS worker name becomes the pool worker name. AMD rigs use the same package; it picks
`proxichorizo-cl` automatically when no NVIDIA driver is present.

## mmpOS

Add a custom miner with the Linux tarball
(`https://github.com/ocminer/ProxiChorizo/releases/latest/download/proxichorizo-<version>.tar.gz`)
and use the external-miner format shipped in it (`mmp-external.conf`, `mmp-launch.sh`):

| Field | Value |
|---|---|
| Pool / URL | `stratum+tcp://proxi.suprnova.cc:3355` |
| User / wallet | your **pool token** |
| Password | `x` (or empty) |
| Extra arguments | optional: `-d 0,1`, `--no-autotune`, `--max-temp 85` |

Alternatively start `mmpos-run.sh` with `POOL_URL=stratum+tcp://proxi.suprnova.cc:3355` and
`TOKEN=<your token>` set in the miner environment.

## SMOS

Same Linux tarball, launcher `smos-run.sh`. In the miner configuration set
`POOL_URL=stratum+tcp://proxi.suprnova.cc:3355`, `TOKEN=<your token>` (or put the token as
the wallet), optionally `WORKER=rig1`, `DEVICES=0,1`, `EXTRA_ARGS="--no-autotune"`.

## Docker

`docker/run-suprnova.sh` (NVIDIA and AMD commands), or `docker/docker-compose.yml` in the
repository with `TOKEN` and `WORKER` filled in. Images: `ocminersupr/proxichorizo:latest`
(NVIDIA) and `ocminersupr/proxichorizo:amd`.

## Solo mining

Solo against your own node with `proxi node mine` is described in `USAGE.md`, section 2.

From 2.1.1, `--no-autotune` skips legacy tuning trials; saved profiles, mandatory
correctness checks and the brief Blake2b kernel comparison remain. HiveOS also
accepts `AUTOTUNE=0`. Docker accepts `EXTRA_ARGS="--no-autotune"`. See the current
[HiveOS guide](../hiveos/README.md) and [Docker guide](../docker/README.md).
