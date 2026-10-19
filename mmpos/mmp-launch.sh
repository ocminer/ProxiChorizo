#!/usr/bin/env bash
# mmpOS "external miner" launcher: --pool -> job server, --user -> pool token, --password -> key seed,
# --api-port -> stats API; --algo/--coin are ignored; anything else is passed through.
cd "$(dirname "$(realpath "$0")")"
ARGS=("$@"); FINAL=(); POOL=""; USERV=""; PASS=""; API_PORT=4078; i=0
while [[ $i -lt ${#ARGS[@]} ]]; do
  case "${ARGS[$i]}" in
    --pool) POOL="${ARGS[$((i+1))]}"; ((i+=2));;  --user) USERV="${ARGS[$((i+1))]}"; ((i+=2));;
    --password) PASS="${ARGS[$((i+1))]}"; ((i+=2));;  --api-port) API_PORT="${ARGS[$((i+1))]}"; ((i+=2));;
    --algo|--coin) ((i+=2));;  *) FINAL+=("${ARGS[$i]}"); ((i+=1));;
  esac
done
[[ -z $API_PORT || $API_PORT == 0 ]] && API_PORT=4078
[[ -n $POOL && $POOL != *://* ]] && POOL="http://$POOL"
sed -i "s/^CUSTOM_API_PORT=.*/CUSTOM_API_PORT=$API_PORT/" mmp-external.conf 2>/dev/null
export CUDA_DEVICE_ORDER=PCI_BUS_ID
CMD=(./proxichorizo --proxi "$POOL" --token "$USERV" --api-bind "127.0.0.1:$API_PORT" --name "${WORKER_NAME:-mmpos}")
[[ -n $PASS && $PASS != x ]] && CMD+=(--seed-hex "${PASS:0:64}")
CMD+=("${FINAL[@]}")
echo "Running: ${CMD[*]}"
exec "${CMD[@]}"
