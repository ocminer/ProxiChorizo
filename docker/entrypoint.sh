#!/usr/bin/env bash
# Env-var front end for ProxiChorizo.
#   POOL_URL   the pool (default stratum+tcp://proxi.suprnova.cc:3355; your own proxi node mine seeker, http://HOST:PORT, for solo)
#   TOKEN      pool registration token (or mine.seeker.token)                         REQUIRED
#   SEED       first 64 hex chars of "private_key" in the pool's proxima.key          VRF jobs only (hash jobs need no key)
#   KEY_FILE   path of a mounted unencrypted proxima.key (alternative to SEED)
#   WORKER, DEVICES (e.g. 0,1), VARIANT, THREADS, ITERS, API_BIND (default 0.0.0.0:4078), EXTRA_ARGS
#   PX_BIN     the miner binary: proxichorizo (CUDA image) or proxichorizo-cl (AMD image); set by the image
# Any command-line arguments are passed straight to the miner instead (docker run … IMAGE --help).
set -euo pipefail
cd /app
BIN="./${PX_BIN:-proxichorizo}"
if [[ $# -gt 0 ]]; then exec "$BIN" "$@"; fi
POOL_URL="${POOL_URL:-stratum+tcp://proxi.suprnova.cc:3355}"
[[ -n "${TOKEN:-}" ]] || { echo "error: TOKEN is required (pool registration token)" >&2; exit 2; }
[[ -n "${SEED:-}" || -n "${KEY_FILE:-}" ]] || echo "no SEED/KEY_FILE: mining hash jobs only (txid-blake2b, Proxima library upgrade 6); VRF jobs need the key" >&2
[[ $POOL_URL == *://* ]] || POOL_URL="http://$POOL_URL"
args=( --proxi "$POOL_URL" --token "$TOKEN" --name "${WORKER:-docker}" --api-bind "${API_BIND:-0.0.0.0:4078}" )
[[ -n "${SEED:-}"     ]] && args+=( --seed-hex "${SEED:0:64}" )
[[ -n "${KEY_FILE:-}" ]] && args+=( --key-file "$KEY_FILE" )
[[ -n "${DEVICES:-}"  ]] && args+=( --gpus "$DEVICES" )
[[ -n "${VARIANT:-}"  ]] && args+=( --variant "$VARIANT" )
[[ -n "${THREADS:-}"  ]] && args+=( --threads "$THREADS" )
[[ -n "${ITERS:-}"    ]] && args+=( --iters "$ITERS" )
[[ -n "${EXTRA_ARGS:-}" ]] && args+=( $EXTRA_ARGS )
exec "$BIN" "${args[@]}"
