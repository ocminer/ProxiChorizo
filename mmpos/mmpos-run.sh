#!/usr/bin/env bash
# MMPOS custom-miner launcher for ProxiChorizo. MMPOS exports the miner settings as
# environment variables and captures stdout. Set POOL_URL (job server), TOKEN (pool token)
# and SEED (key seed, 64 hex) in the MMPOS miner config; WALLET/PASSWORD are accepted as aliases.
cd "$(dirname "$0")" || exit 1
POOL_URL="${POOL_URL:-${POOL:-}}"; TOKEN="${TOKEN:-${WALLET:-}}"; SEED="${SEED:-${PASSWORD:-}}"
[[ -z $POOL_URL ]] && { echo "POOL_URL missing (e.g. proxi.suprnova.cc:8100)"; exit 1; }
[[ -z $TOKEN    ]] && { echo "TOKEN missing (your pool registration token)"; exit 1; }
[[ -z $SEED && -z $KEY_FILE ]] && { echo "SEED missing (first 64 hex chars of private_key in your proxima.key) or KEY_FILE"; exit 1; }
[[ $POOL_URL == *://* ]] || POOL_URL="http://$POOL_URL"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# binary: the CUDA miner with an NVIDIA driver, the OpenCL miner (AMD) otherwise or with MINER=cl
BIN=./proxichorizo
if [[ ${MINER:-} == cl || ( ${MINER:-} != cuda && ! -e /dev/nvidiactl && -x ./proxichorizo-cl ) ]]; then BIN=./proxichorizo-cl; fi
args=( --proxi "$POOL_URL" --token "$TOKEN" --name "${WORKER:-mmpos}" --api-bind "${API_BIND:-127.0.0.1:4078}" )
[[ -n $SEED     ]] && args+=( --seed-hex "${SEED:0:64}" )
[[ -n $KEY_FILE ]] && args+=( --key-file "$KEY_FILE" )
[[ -n $DEVICES  ]] && args+=( --gpus "$DEVICES" )
[[ -n $VARIANT  ]] && args+=( --variant "$VARIANT" )
[[ -n $EXTRA_ARGS ]] && args+=( $EXTRA_ARGS )
exec "$BIN" "${args[@]}"
