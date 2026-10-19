#!/usr/bin/env bash
# Build the Linux release archives from one binary. One tarball serves plain Linux, HiveOS
# (h-run.sh), MMPOS (mmpos-run.sh) and SMOS (smos-run.sh); a second archive carries the
# directory name HiveOS's custom-miner installer requires.
# Usage: scripts/package.sh <version> <binary> [cuda-version] [opencl-binary]      -> dist/
set -euo pipefail
cd "$(dirname "$0")/.."
VER=$1; BIN=$2; CUDA=${3:-$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9.]+' || echo unknown)}; CLBIN=${4:-}
[[ -x "$BIN" ]] || { echo "binary $BIN missing"; exit 1; }
[[ -z "$CLBIN" || -x "$CLBIN" ]] || { echo "OpenCL binary $CLBIN missing"; exit 1; }
D=dist/proxichorizo-$VER
rm -rf "$D"; mkdir -p "$D"
cp "$BIN" "$D/proxichorizo"; strip "$D/proxichorizo" 2>/dev/null || true
[[ -n "$CLBIN" ]] && { cp "$CLBIN" "$D/proxichorizo-cl"; strip "$D/proxichorizo-cl" 2>/dev/null || true; }
cp README.md USAGE.md LICENSE NOTICE PLATFORMS.md vectors.json "$D/"
[[ -f RELEASE-v$VER.md ]] && cp "RELEASE-v$VER.md" "$D/"
printf '{"version":"%s","source_commit":"%s","ci_run":"%s","platform":"linux-x86_64","glibc_min":"%s","cuda":"%s","architectures":[75,80,86,89,120],"opencl":%s}\n' \
  "$VER" "$(git rev-parse HEAD 2>/dev/null || echo unknown)" "${GITHUB_SERVER_URL:-}/${GITHUB_REPOSITORY:-}/actions/runs/${GITHUB_RUN_ID:-local}" \
  "$(objdump -T "$D/proxichorizo" 2>/dev/null | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1)" "$CUDA" "$([[ -n "$CLBIN" ]] && echo true || echo false)" > "$D/BUILD-INFO.json"
cp hiveos/h-config.sh hiveos/h-run.sh hiveos/h-stats.sh hiveos/h-manifest.conf "$D/"
cp mmpos/mmpos-run.sh mmpos/mmp-external.conf mmpos/mmp-launch.sh mmpos/mmp-stats.sh smos/smos-run.sh "$D/"
sed -i "s/^CUSTOM_VERSION=.*/CUSTOM_VERSION=$VER/" "$D/h-manifest.conf"
sed -i "s/^EXTERNAL_VERSION=.*/EXTERNAL_VERSION=\"$VER\"/" "$D/mmp-external.conf"
chmod +x "$D"/*.sh "$D"/proxichorizo*
( cd "$D" && find . -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS.txt )
( cd dist && tar czf "proxichorizo-$VER.tar.gz" "proxichorizo-$VER" && sha256sum "proxichorizo-$VER.tar.gz" > "proxichorizo-$VER.tar.gz.sha256" )
( cd dist && rm -rf proxichorizo && cp -r "proxichorizo-$VER" proxichorizo && tar czf proxichorizo-hiveos.tar.gz proxichorizo && rm -rf proxichorizo \
    && sha256sum proxichorizo-hiveos.tar.gz > proxichorizo-hiveos.tar.gz.sha256 )
rm -rf "$D"
echo "== dist =="; ls -la dist/ | grep -E "$VER|hiveos"
echo "== package contents =="; tar tzf "dist/proxichorizo-$VER.tar.gz"
