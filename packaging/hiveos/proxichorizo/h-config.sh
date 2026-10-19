#!/usr/bin/env bash
# Flight sheet -> ProxiChorizo command line. Written to $CUSTOM_CONFIG_FILENAME as CLI_ARGS.
#
#   Pool URL   ($CUSTOM_URL)         -> --proxi http://host:port   (the job server; pool or your own proxi node mine)
#   Wallet     ($CUSTOM_TEMPLATE)    -> --token  (the bearer token the pool gave you at registration,
#                                       or mine.seeker.token of your own proxi node mine)
#   Password   ($CUSTOM_PASS)        -> --seed-hex (the 32-byte seed = first 64 hex chars of "private_key"
#                                       in the proxima.key the pool gave you). %WORKER_NAME% becomes --name.
#   Extra args ($CUSTOM_USER_CONFIG) -> appended (e.g. --variant fe32_w4_lb2_c --threads 256 --iters 2)
# The key never leaves the rig other than to this process; a pool key can only mine for your account.
[[ -t 1 ]] || exec 2>/dev/null
. /hive/miners/custom/proxichorizo/h-manifest.conf 2>/dev/null
url="$CUSTOM_URL"; [[ "$url" == *://* ]] || url="http://$url"
args="--proxi ${url} --token ${CUSTOM_TEMPLATE} --name ${WORKER_NAME:-hiveos} --api-bind 127.0.0.1:4078"
seed="$CUSTOM_PASS"
[[ -n "$seed" && "$seed" != "x" ]] && args="$args --seed-hex ${seed:0:64}"
args="$args ${CUSTOM_USER_CONFIG}"
echo "CLI_ARGS=\"${args}\"" > "$CUSTOM_CONFIG_FILENAME"
