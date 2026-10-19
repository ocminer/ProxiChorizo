#!/usr/bin/env bash
# Build the release archives from one Linux binary: plain tar.gz, HiveOS and mmpOS packages.
# Usage: packaging/package.sh <version> <path-to-linux-binary> [outdir]
set -euo pipefail
VER=$1; BIN=$2; OUT=${3:-dist}
cd "$(dirname "$0")/.."
[[ -x "$BIN" ]] || { echo "binary $BIN missing"; exit 1; }
mkdir -p "$OUT"; STAGE=$(mktemp -d); trap 'rm -rf "$STAGE"' EXIT
# 1. plain Linux
D="$STAGE/proxichorizo-$VER-linux-x64"; mkdir -p "$D"
cp "$BIN" "$D/proxichorizo"; cp README.md USAGE.md LICENSE PLATFORMS.md vectors.json "$D/"
tar -C "$STAGE" -czf "$OUT/proxichorizo-$VER-linux-x64.tar.gz" "proxichorizo-$VER-linux-x64"
# 2. HiveOS: archive proxichorizo-<ver>.tar.gz with the top-level directory "proxichorizo"
D="$STAGE/hiveos/proxichorizo"; mkdir -p "$D"
cp packaging/hiveos/proxichorizo/* "$D/"; cp "$BIN" "$D/proxichorizo"; cp LICENSE USAGE.md "$D/"
sed -i "s/^CUSTOM_VERSION=.*/CUSTOM_VERSION=$VER/" "$D/h-manifest.conf"; chmod +x "$D"/*.sh "$D/proxichorizo"
tar -C "$STAGE/hiveos" -czf "$OUT/proxichorizo-$VER.tar.gz" proxichorizo
# 3. mmpOS: proxichorizo-mmpos_<ver>.tar.gz, internal directory of the same name
D="$STAGE/proxichorizo-mmpos_$VER"; mkdir -p "$D"
cp packaging/mmpos/proxichorizo/* "$D/"; cp "$BIN" "$D/proxichorizo"; cp LICENSE USAGE.md "$D/"
sed -i "s/^EXTERNAL_VERSION=.*/EXTERNAL_VERSION=\"$VER\"/" "$D/mmp-external.conf"; chmod +x "$D"/*.sh "$D/proxichorizo"
tar -C "$STAGE" -czf "$OUT/proxichorizo-mmpos_$VER.tar.gz" "proxichorizo-mmpos_$VER"
(cd "$OUT" && sha256sum proxichorizo-$VER-linux-x64.tar.gz proxichorizo-$VER.tar.gz proxichorizo-mmpos_$VER.tar.gz > SHA256SUMS-linux.txt && cat SHA256SUMS-linux.txt)
