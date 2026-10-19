#!/usr/bin/env bash
# Env-var front end for ProxiChorizo.
#   POOL_URL   job server (default proxi.suprnova.cc:8100; your own proxi node mine seeker for solo)
#   TOKEN      pool registration token (or mine.seeker.token)                         REQUIRED
#   SEED       first 64 hex chars of "private_key" in the pool's proxima.key          REQUIRED unless KEY_FILE
#   KEY_FILE   path of a mounted unencrypted proxima.key (alternative to SEED)
#   WORKER, DEVICES (e.g. 0,1), VARIANT, THREADS, ITERS, API_BIND (default 0.0.0.0:4078), EXTRA_ARGS
# Any command-line arguments are passed straight to the miner instead (docker run … proxichorizo --help).
set -euo pipefail
cd /app
if [[ $# -gt 0 ]]; then exec ./proxichorizo "$@"; fi
POOL_URL="${POOL_URL:-proxi.suprnova.cc:8100}"
[[ -n "${TOKEN:-}" ]] || { echo "error: TOKEN is required (pool registration token)" >&2; exit 2; }
[[ -n "${SEED:-}" || -n "${KEY_FILE:-}" ]] || { echo "error: SEED (64 hex) or KEY_FILE is required" >&2; exit 2; }
[[ $POOL_URL == *://* ]] || POOL_URL="http://$POOL_URL"
args=( --proxi "$POOL_URL" --token "$TOKEN" --name "${WORKER:-docker}" --api-bind "${API_BIND:-0.0.0.0:4078}" )
[[ -n "${SEED:-}"     ]] && args+=( --seed-hex "${SEED:0:64}" )
[[ -n "${KEY_FILE:-}" ]] && args+=( --key-file "$KEY_FILE" )
[[ -n "${DEVICES:-}"  ]] && args+=( --gpus "$DEVICES" )
[[ -n "${VARIANT:-}"  ]] && args+=( --variant "$VARIANT" )
[[ -n "${THREADS:-}"  ]] && args+=( --threads "$THREADS" )
[[ -n "${ITERS:-}"    ]] && args+=( --iters "$ITERS" )
[[ -n "${EXTRA_ARGS:-}" ]] && args+=( $EXTRA_ARGS )
exec ./proxichorizo "${args[@]}"
