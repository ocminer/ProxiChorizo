# Docker deployment

`ocminersupr/proxichorizo:2.1.2` (also `:latest`; mirrored as `ghcr.io/ocminer/proxichorizo`).
Install the NVIDIA Container Toolkit on the host. `TOKEN` is the launcher
variable name: set it to your PROX address, or keep an existing pool token.
No registration or wallet key is required for current pool mining.

```sh
docker run -d --name proxichorizo --gpus '"device=0"' --restart unless-stopped -p 4078:4078 \
  -e TOKEN='YOUR_PROX_ADDRESS' -e WORKER=rig1 \
  ocminersupr/proxichorizo:2.1.2
```

`POOL_URL` defaults to `stratum+tcp://proxi.suprnova.cc:3355`; for solo mining point it at your own
`proxi node mine` seeker address and use `mine.seeker.token` as `TOKEN`. Blake2b
mining needs neither `SEED` nor `KEY_FILE`; the wallet controller keeps its key.
Legacy VRF jobs need `SEED` or a mounted key file selected with `KEY_FILE`. Multi-GPU: `--gpus '"device=0,1"'`
and `DEVICES=0,1`. Statistics: `curl http://localhost:4078/`.

## AMD GPUs

`ocminersupr/proxichorizo:2.1.2-amd` (also `:amd`; mirrored as `ghcr.io/ocminer/proxichorizo:2.1.2-amd`) carries the
OpenCL miner `proxichorizo-cl` and AMD's ROCm 6.4 OpenCL runtime (Vega 20 / MI50, CDNA, RDNA2, RDNA3, RDNA4).
The host needs only the amdgpu kernel driver; no ROCm install and no container toolkit:

```sh
docker run -d --name proxichorizo --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  --restart unless-stopped -p 4078:4078 \
  -e TOKEN='YOUR_PROX_ADDRESS' -e WORKER=rig1 \
  ocminersupr/proxichorizo:2.1.2-amd
```

The same environment variables apply. `DEVICES=0,1` selects GPUs (`--list-devices` shows them:
`docker run --rm --device /dev/kfd --device /dev/dri ocminersupr/proxichorizo:amd --list-devices`).
If the host's render group has a different id than the image's, use `--group-add $(getent group render | cut -d: -f3)`.
The driver compiles the required kernels and the miner verifies them before mining.
Compilation time varies with the card and driver. Keyless Blake2b mining in 2.1.1
skips unused legacy VRF programs; a valid cached legacy profile also avoids compiling
an unused default variant. No exact startup time is promised for a particular rig.

Arguments after the image name bypass the env front end and go straight to the miner:
`docker run --rm --gpus all ocminersupr/proxichorizo:2.1.2 --hash-bench 10`.

## Startup control (2.1.1)

Both images accept `-e EXTRA_ARGS="--no-autotune"`. This skips legacy VRF tuning
trials and uses a valid cache entry or platform default. The brief Blake2b kernel
comparison and mandatory verification remain enabled. Leave `SEED` and `KEY_FILE`
unset for current pool hash jobs. Do not combine the flag with `--autotune`.

Long jobs are negotiated automatically on NVIDIA and AMD; older pools and ordinary
jobs continue to work. For TLS set `POOL_URL=stratum+ssl://proxi.suprnova.cc:3356`.
Images do not change clocks, fans or power allowances. Use the versioned tags above
for a fixed release; `latest` (NVIDIA) and `amd` track the latest respective image.
