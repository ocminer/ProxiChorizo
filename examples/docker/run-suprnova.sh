#!/usr/bin/env bash
# ProxiChorizo in Docker on suprnova.cc. Needs Docker and, for NVIDIA, the NVIDIA Container Toolkit.
TOKEN="YOUR_POOL_TOKEN"; WORKER="docker1"
# NVIDIA, all GPUs:
docker run -d --name proxichorizo --gpus all --restart unless-stopped -p 127.0.0.1:4078:4078 \
  -e TOKEN="$TOKEN" -e WORKER="$WORKER" \
  -e POOL_URL=stratum+tcp://proxi.suprnova.cc:3355 \
  ocminersupr/proxichorizo:latest
# AMD (OpenCL / ROCm runtime in the image; the host needs only the amdgpu driver) — use instead of the command above:
# docker run -d --name proxichorizo --device /dev/kfd --device /dev/dri --group-add video --group-add render \
#   --restart unless-stopped -p 127.0.0.1:4078:4078 -e TOKEN="$TOKEN" -e WORKER="$WORKER" ocminersupr/proxichorizo:amd
# TLS: -e POOL_URL=stratum+ssl://proxi.suprnova.cc:3356     Only GPUs 0,1: --gpus '"device=0,1"'
# Logs: docker logs -f proxichorizo     Stats: curl http://127.0.0.1:4078/     Stop: docker rm -f proxichorizo
