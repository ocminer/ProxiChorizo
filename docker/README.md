# Docker deployment

`ocminersupr/proxichorizo:0.5.0` (also `:latest`; mirrored as `ghcr.io/ocminer/proxichorizo`).
Install the NVIDIA Container Toolkit on the host.

```sh
docker run -d --name proxichorizo --gpus '"device=0"' --restart unless-stopped -p 4078:4078 \
  -e TOKEN='YOUR_POOL_TOKEN' -e SEED='FIRST_64_HEX_CHARS_OF_private_key' -e WORKER=rig1 \
  ocminersupr/proxichorizo:0.5.0
```

`POOL_URL` defaults to `proxi.suprnova.cc:8100`; for solo mining point it at your own
`proxi node mine` seeker address and use `mine.seeker.token` as `TOKEN`. Instead of `SEED`,
mount the key file and set `KEY_FILE=/keys/proxima.key`. Multi-GPU: `--gpus '"device=0,1"'`
and `DEVICES=0,1`. Statistics: `curl http://localhost:4078/`.

Arguments after the image name bypass the env front end and go straight to the miner:
`docker run --rm --gpus all ocminersupr/proxichorizo:0.5.0 --autotune --bench 10`.
