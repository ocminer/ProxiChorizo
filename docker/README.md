# Docker deployment

`ocminersupr/proxichorizo:2.0.1` (also `:latest`; mirrored as `ghcr.io/ocminer/proxichorizo`).
Install the NVIDIA Container Toolkit on the host.

```sh
docker run -d --name proxichorizo --gpus '"device=0"' --restart unless-stopped -p 4078:4078 \
  -e TOKEN='YOUR_POOL_TOKEN' -e SEED='FIRST_64_HEX_CHARS_OF_private_key' -e WORKER=rig1 \
  ocminersupr/proxichorizo:2.0.1
```

`POOL_URL` defaults to `stratum+tcp://proxi.suprnova.cc:3355`; for solo mining point it at your own
`proxi node mine` seeker address and use `mine.seeker.token` as `TOKEN`. Instead of `SEED`,
mount the key file and set `KEY_FILE=/keys/proxima.key`. The key is needed for VRF jobs only (until slot 196000); hash jobs (txid-blake2b, from slot 197000) need neither `SEED` nor `KEY_FILE`. Multi-GPU: `--gpus '"device=0,1"'`
and `DEVICES=0,1`. Statistics: `curl http://localhost:4078/`.

## AMD GPUs

`ocminersupr/proxichorizo:2.0.1-amd` (also `:amd`; mirrored as `ghcr.io/ocminer/proxichorizo:2.0.1-amd`) carries the
OpenCL miner `proxichorizo-cl` and AMD's ROCm 6.4 OpenCL runtime (Vega 20 / MI50, CDNA, RDNA2, RDNA3, RDNA4).
The host needs only the amdgpu kernel driver; no ROCm install and no container toolkit:

```sh
docker run -d --name proxichorizo --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  --restart unless-stopped -p 4078:4078 \
  -e TOKEN='YOUR_POOL_TOKEN' -e SEED='FIRST_64_HEX_CHARS_OF_private_key' -e WORKER=rig1 \
  ocminersupr/proxichorizo:2.0.1-amd
```

The same environment variables apply. `DEVICES=0,1` selects GPUs (`--list-devices` shows them:
`docker run --rm --device /dev/kfd --device /dev/dri ocminersupr/proxichorizo:amd --list-devices`).
If the host's render group has a different id than the image's, use `--group-add $(getent group render | cut -d: -f3)`.
The first start compiles and verifies the kernels for each GPU (about a minute).

Arguments after the image name bypass the env front end and go straight to the miner:
`docker run --rm --gpus all ocminersupr/proxichorizo:2.0.1 --autotune --bench 10`.
