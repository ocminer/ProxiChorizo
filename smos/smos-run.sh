#!/usr/bin/env bash
# SMOS custom-miner launcher for ProxiChorizo. SMOS exports the miner settings as
# environment variables and captures stdout. Set POOL_URL (the pool) and TOKEN (pool token)
# in the SMOS miner config (WALLET is an alias for TOKEN). SEED/PASSWORD: legacy VRF only, not needed.
cd "$(dirname "$0")" || exit 1
POOL_URL="${POOL_URL:-${POOL:-}}"; TOKEN="${TOKEN:-${WALLET:-}}"; SEED="${SEED:-${PASSWORD:-}}"
[[ -z $POOL_URL ]] && { echo "POOL_URL missing (e.g. stratum+tcp://proxi.suprnova.cc:3355)"; exit 1; }
[[ -z $TOKEN    ]] && { echo "TOKEN missing (your pool registration token)"; exit 1; }
[[ $POOL_URL == *://* ]] || POOL_URL="http://$POOL_URL"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# binary: the CUDA miner with an NVIDIA driver, the OpenCL miner (AMD) otherwise or with MINER=cl
BIN=./proxichorizo
if [[ ${MINER:-} == cl || ( ${MINER:-} != cuda && ! -e /dev/nvidiactl && -x ./proxichorizo-cl ) ]]; then BIN=./proxichorizo-cl; fi
args=( --proxi "$POOL_URL" --token "$TOKEN" --name "${WORKER:-smos}" --api-bind "${API_BIND:-127.0.0.1:4078}" )
[[ -n $SEED     ]] && args+=( --seed-hex "${SEED:0:64}" )
[[ -n $KEY_FILE ]] && args+=( --key-file "$KEY_FILE" )
[[ -n $DEVICES  ]] && args+=( --gpus "$DEVICES" )
[[ -n $VARIANT  ]] && args+=( --variant "$VARIANT" )
[[ -n $EXTRA_ARGS ]] && args+=( $EXTRA_ARGS )
exec "$BIN" "${args[@]}"
