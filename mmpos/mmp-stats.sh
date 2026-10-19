#!/usr/bin/env bash
# mmpOS stats line {busid, hash, units, air, miner_name, miner_version} from the miner's /mmpos endpoint.
DIR="$(dirname "$(realpath "$0")")"; PORT=4078
[[ -f "$DIR/mmp-external.conf" ]] && PORT=$(grep -m1 '^CUSTOM_API_PORT=' "$DIR/mmp-external.conf" | cut -d= -f2)
json=$(curl -s --max-time 3 "http://127.0.0.1:${PORT:-4078}/mmpos" 2>/dev/null)
[[ "$json" == *'"hash"'* ]] && { echo "$json"; exit 0; }
echo '{"busid":[],"hash":[],"units":"hs","air":["0","0","0"],"miner_name":"proxichorizo","miner_version":"unknown"}'
