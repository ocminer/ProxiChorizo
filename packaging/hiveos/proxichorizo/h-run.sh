#!/usr/bin/env bash
# Starts ProxiChorizo in the foreground, log to $CUSTOM_LOG_BASENAME.log (h-stats.sh reads the API, the log is for you).
cd "$(dirname "$(realpath "$0")")"
. h-manifest.conf
[[ -f "$CUSTOM_CONFIG_FILENAME" ]] && . "$CUSTOM_CONFIG_FILENAME"
if [[ -z "${CLI_ARGS:-}" ]]; then . h-config.sh; . "$CUSTOM_CONFIG_FILENAME"; fi
mkdir -p "$(dirname "$CUSTOM_LOG_BASENAME")"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
echo "[proxichorizo] ./proxichorizo $CLI_ARGS"
./proxichorizo $CLI_ARGS 2>&1 | tee "$CUSTOM_LOG_BASENAME.log"
exit "${PIPESTATUS[0]}"
