#!/usr/bin/env bash
# Launch ProxiChorizo. cwd is the package directory; HiveOS captures stdout.
cd "$(dirname "$0")" || exit 1
. h-manifest.conf
[[ -f $CUSTOM_CONFIG_FILENAME ]] && . "$CUSTOM_CONFIG_FILENAME"
[[ -z $POOL_URL ]] && echo "POOL_URL missing — set the pool (job server) in the flight sheet" && exit 1
[[ -z $TOKEN    ]] && echo "TOKEN missing — set your pool token as the wallet template" && exit 1
[[ -z $SEED && -z $KEY_FILE ]] && echo "SEED missing — set the key seed (64 hex) as the pass, or KEY_FILE=/path/proxima.key" && exit 1
[[ $POOL_URL == *://* ]] || POOL_URL="http://$POOL_URL"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# binary: the CUDA miner with an NVIDIA driver, the OpenCL miner (AMD) otherwise or with MINER=cl
BIN=./proxichorizo
if [[ ${MINER:-} == cl || ( ${MINER:-} != cuda && ! -e /dev/nvidiactl && -x ./proxichorizo-cl ) ]]; then BIN=./proxichorizo-cl; fi
args=( --proxi "$POOL_URL" --token "$TOKEN" --name "${WORKER:-hiveos}" --api-bind "${API_BIND:-127.0.0.1:4078}" )
[[ -n $SEED     ]] && args+=( --seed-hex "${SEED:0:64}" )
[[ -n $KEY_FILE ]] && args+=( --key-file "$KEY_FILE" )
[[ -n $DEVICES  ]] && args+=( --gpus "$DEVICES" )
[[ -n $VARIANT  ]] && args+=( --variant "$VARIANT" )
[[ -n $THREADS  ]] && args+=( --threads "$THREADS" )
[[ -n $ITERS    ]] && args+=( --iters "$ITERS" )
[[ -n $GROUPS   ]] && args+=( --groups "$GROUPS" )
[[ -n $EXTRA_ARGS ]] && args+=( $EXTRA_ARGS )
mkdir -p "$(dirname "$CUSTOM_LOG_BASENAME")"
"$BIN" "${args[@]}" 2>&1 | tee -a "${CUSTOM_LOG_BASENAME}.log"
exit "${PIPESTATUS[0]}"
