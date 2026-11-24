# ProxiChorizo 2.1.2 on mmpOS and SMOS

Both platforms use the standard Linux archive, containing CUDA and OpenCL miners:
https://github.com/ocminer/ProxiChorizo/releases/download/v2.1.2/proxichorizo-2.1.2.tar.gz

## mmpOS

Use the supplied `mmp-external.conf`, `mmp-launch.sh` and `mmp-stats.sh`. Set Pool to
`stratum+tcp://proxi.suprnova.cc:3355` (or TLS port 3356 with `stratum+ssl://`),
User/Wallet to your PROX address (or existing token) and Password to `x` or empty for current Blake2b jobs.
Extra arguments can include `--no-autotune` and `-d 0,1`. The alternative
`mmpos-run.sh` accepts `POOL_URL`, `TOKEN`, `WORKER`, `DEVICES` and `EXTRA_ARGS`.

## SMOS

Use `smos-run.sh`, with `POOL_URL=stratum+tcp://proxi.suprnova.cc:3355`,
`TOKEN=<your PROX address or existing token>` and an optional `WORKER`. To skip legacy tuning trials
set `EXTRA_ARGS="--no-autotune"`. `MINER=cl` explicitly selects the OpenCL backend;
`MINER=cuda` selects CUDA. Leave the legacy seed/key unset for Blake2b pool work.

The new flag uses a valid cached legacy profile or platform default. It keeps
mandatory verification and the brief Blake2b kernel comparison. Do not combine it
with `--autotune`. Long-job negotiation is automatic and ordinary jobs remain
supported. The miner never changes clocks, fans or power allowances.
