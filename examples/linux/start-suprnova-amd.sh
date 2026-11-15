#!/usr/bin/env bash
# ProxiChorizo — mine Proxima (PROX) on suprnova.cc, AMD / OpenCL GPUs (proxichorizo-cl), plain TCP stratum.
# 1. Register at https://proxi.suprnova.cc/StartMining.html with your PROX address; copy the TOKEN it shows.
# 2. Put the token and a worker name below, save, then run:  ./examples/linux/start-suprnova-amd.sh
TOKEN="YOUR_POOL_TOKEN"
WORKER="rig1"                                   # one name per rig; shows up as a worker on the pool
POOL="stratum+tcp://proxi.suprnova.cc:3355"      # TLS: stratum+ssl://proxi.suprnova.cc:3356

cd "$(dirname "$0")/../.." || exit 1             # the miner directory (where ./proxichorizo-cl is)
[ "$TOKEN" = "YOUR_POOL_TOKEN" ] && { echo "Edit $0 and set TOKEN first."; exit 2; }
# first start compiles the OpenCL kernels (about a minute) and self-tests them
# Optional: -d 0,1 (only these GPUs), --max-temp 85 (per-GPU pause, default 88 C), --api-bind 127.0.0.1:4078 (stats)
exec ./proxichorizo-cl --proxi "$POOL" --token "$TOKEN" --name "$WORKER" "$@"
