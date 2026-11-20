# ProxiChorizo 2.1.1 — current pool setup

Suprnova now accepts address login without registration. Existing tokens continue
to work. This guide updates the legacy registration/token wording in the frozen
2.1.1 archives; no binary, kernel, release tag or existing payload was changed.

## Linux and Windows

Copy your wallet's 64-hex PROX holder ID (omit the optional `a/` prefix here):

```sh
./proxichorizo -o stratum+tcp://proxi.suprnova.cc:3355 -u YOUR_PROX_ADDRESS.rig1 -p x
```

On Windows use `proxichorizo.exe`. For AMD use `proxichorizo-cl` or
`proxichorizo-cl.exe`. Encrypted endpoint: `stratum+ssl://proxi.suprnova.cc:3356`.
Replace `rig1` with your worker name. No wallet key or seed is needed on the rig.

Existing tokens remain valid: `--token YOUR_EXISTING_TOKEN --name rig1`. The
packaged start scripts retain the variable name `TOKEN`: set its value to your
PROX address or an existing token. The Windows PowerShell launcher similarly
accepts `-Token YOUR_PROX_ADDRESS -Worker rig1`.

## HiveOS

Custom miner name: `proxichorizo`. Installation URL:
https://github.com/ocminer/ProxiChorizo/releases/download/v2.1.1/proxichorizo-hiveos.tar.gz

Use your PROX address in Wallet / `%WAL%`, either endpoint above in Pool, and leave
Pass / SEED empty for Blake2b. An existing token in Wallet continues to work.
Add `AUTOTUNE=0` to Extra config to skip legacy tuning trials; alternatively use
`EXTRA_ARGS="--no-autotune"`. NVIDIA and AMD use the same archive.

## Docker, mmpOS and SMOS

Docker images: `ocminersupr/proxichorizo:2.1.1` (NVIDIA) and `:2.1.1-amd` (AMD),
also mirrored under `ghcr.io/ocminer/proxichorizo`. Set
`TOKEN=YOUR_PROX_ADDRESS`, `WORKER=rig1` and optionally `EXTRA_ARGS="--no-autotune"`.
The default pool is the TCP endpoint above; set `POOL_URL` for TLS.

For mmpOS use the Linux archive and supplied external-miner launcher; User/Wallet
is your PROX address, Password `x` or empty, and Extra arguments may include
`--no-autotune`. SMOS uses `smos-run.sh` with the same `TOKEN`, `POOL_URL`,
`WORKER` and `EXTRA_ARGS` variables as Docker.

## Startup and safety checks

`--no-autotune` skips legacy VRF tuning trials, using a valid cached profile or the
platform default. Required driver compilation, CPU share verification and the
brief Blake2b kernel comparison remain enabled. Do not combine it with `--autotune`.
For keyless hash startup, omit legacy seed/key options and keep any old
`proxima.key` outside the miner directory (retain your backup).

Long jobs are negotiated automatically; ordinary jobs remain supported. No clock,
fan or power setting is changed by the miner.

[Pool setup](https://proxi.suprnova.cc/StartMining.html) ·
[Current full usage guide](https://github.com/ocminer/ProxiChorizo/blob/main/USAGE.md) ·
[HiveOS](https://github.com/ocminer/ProxiChorizo/blob/main/hiveos/README.md) ·
[Docker](https://github.com/ocminer/ProxiChorizo/blob/main/docker/README.md) ·
[Community](https://discord.gg/29RHZwvVPN)
