#!/usr/bin/env bash
# HiveOS stats: hashrate and shares from the miner's API (127.0.0.1:4078), temps/fans from the agent.
. /hive/miners/custom/proxichorizo/h-manifest.conf
json=$(curl -s --max-time 3 http://127.0.0.1:4078/ 2>/dev/null)
if [[ -z "$json" ]]; then khs=0; stats='{"hs":[],"hs_units":"khs","temp":[],"fan":[],"uptime":0,"ar":[0,0],"algo":"proxima","bus_numbers":[]}'; return 2>/dev/null || exit 0; fi
total=$(jq -r '.hashrate' <<< "$json")
khs=$(awk -v h="$total" 'BEGIN{printf "%.3f", h/1000}')
uptime=$(jq -r '.uptime' <<< "$json")
acc=$(jq -r '.shares.accepted' <<< "$json"); rej=$(jq -r '.shares.rejected' <<< "$json")
ver=$(jq -r '.version' <<< "$json")
# per-GPU: the miner reports PCI bus numbers (decimal); match them to the agent's GPU list for temp/fan
readarray -t gpu_stats < <( jq --slurp -r -c '.[] | .busids, .temp, .fan | join(" ")' $GPU_STATS_JSON 2>/dev/null)
busids=(${gpu_stats[0]}); temps=(${gpu_stats[1]}); fans=(${gpu_stats[2]})
hs_arr=(); bus_arr=(); temp_arr=(); fan_arr=()
while IFS=$'\t' read -r bn hr; do
  hs_arr+=($(awk -v h="$hr" 'BEGIN{printf "%.3f", h/1000}')); bus_arr+=($bn)
  t=0; f=0
  for i in "${!busids[@]}"; do
    [[ "${busids[i]}" =~ ^([A-Fa-f0-9]+): ]] && [[ $((16#${BASH_REMATCH[1]})) -eq $bn ]] && { t=${temps[i]}; f=${fans[i]}; break; }
  done
  temp_arr+=($t); fan_arr+=($f)
done < <(jq -r '.gpus[] | "\(.bus_number)\t\(.hashrate)"' <<< "$json")
j() { local IFS=,; echo "[$*]"; }
stats=$(jq -c -n --argjson hs "$(j "${hs_arr[@]}")" --argjson temp "$(j "${temp_arr[@]}")" --argjson fan "$(j "${fan_arr[@]}")" --argjson bus "$(j "${bus_arr[@]}")" \
  --argjson up "$uptime" --argjson acc "$acc" --argjson rej "$rej" --arg ver "$ver" \
  '{hs:$hs, hs_units:"khs", temp:$temp, fan:$fan, uptime:$up, ver:$ver, ar:[$acc,$rej], algo:"proxima", bus_numbers:$bus}')
