#!/usr/bin/env bash
# HiveOS stats from the miner's API (127.0.0.1:4078 unless API_BIND is set): hashrate per GPU in kH/s,
# shares, uptime; temperatures and fans from the agent's GPU list matched by PCI bus number.
cd "$(dirname "$0")" || exit 1
. h-manifest.conf
[[ -f $CUSTOM_CONFIG_FILENAME ]] && . "$CUSTOM_CONFIG_FILENAME"
api="${API_BIND:-127.0.0.1:4078}"; [[ $api == :* || $api != *:* ]] && api="127.0.0.1:${api#:}"
json=$(curl -s --max-time 3 "http://$api/" 2>/dev/null)
if [[ -z $json ]]; then
  khs=0; stats='{"hs":[],"hs_units":"khs","temp":[],"fan":[],"uptime":0,"ar":[0,0],"algo":"proxima","bus_numbers":[]}'
else
  readarray -t gpu_stats < <( jq --slurp -r -c '.[] | .busids, .temp, .fan | join(" ")' $GPU_STATS_JSON 2>/dev/null)
  busids=(${gpu_stats[0]}); temps=(${gpu_stats[1]}); fans=(${gpu_stats[2]})
  hs=(); bus=(); temp=(); fan=()
  while IFS=$'\t' read -r bn hr; do
    hs+=("$(awk -v h="$hr" 'BEGIN{printf "%.3f", h/1000}')"); bus+=("$bn"); t=0; f=0
    for i in "${!busids[@]}"; do
      [[ ${busids[i]} =~ ^([A-Fa-f0-9]+): ]] && [[ $((16#${BASH_REMATCH[1]})) -eq $bn ]] && { t=${temps[i]:-0}; f=${fans[i]:-0}; break; }
    done
    temp+=("$t"); fan+=("$f")
  done < <(jq -r '.gpus[] | "\(.bus_number)\t\(.hashrate)"' <<< "$json")
  j() { local IFS=,; echo "[$*]"; }
  khs=$(jq -r '.hashrate/1000' <<< "$json")
  stats=$(jq -nc --argjson hs "$(j "${hs[@]}")" --argjson temp "$(j "${temp[@]}")" --argjson fan "$(j "${fan[@]}")" --argjson bus "$(j "${bus[@]}")" \
    --argjson up "$(jq -r '.uptime' <<< "$json")" --argjson acc "$(jq -r '.shares.accepted' <<< "$json")" --argjson rej "$(jq -r '.shares.rejected' <<< "$json")" --arg ver "$CUSTOM_VERSION" \
    '{hs:$hs, hs_units:"khs", temp:$temp, fan:$fan, uptime:$up, ver:$ver, ar:[$acc,$rej], algo:"proxima", bus_numbers:$bus}')
fi
echo "$stats"; echo "$khs"
