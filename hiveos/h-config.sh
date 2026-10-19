#!/usr/bin/env bash
# Writes proxichorizo.conf from the flight sheet (sourced by h-run.sh).
#   Pool URL  -> POOL_URL  (job server: proxi.suprnova.cc:8100, or your own proxi node mine seeker address)
#   Wallet    -> TOKEN     (the token the pool gave you at registration / mine.seeker.token)
#   Pass      -> SEED      (first 64 hex characters of "private_key" in the pool's proxima.key)
#   Extra config: KEY=VALUE lines (DEVICES, VARIANT, THREADS, ITERS, API_BIND, EXTRA_ARGS)
[[ -z $CUSTOM_URL ]] && echo -e "${RED}CUSTOM_URL is empty${NOCOLOR}" && exit 1
[[ -z $CUSTOM_TEMPLATE ]] && echo -e "${RED}CUSTOM_TEMPLATE (wallet = pool token) is empty${NOCOLOR}" && exit 1
conf="POOL_URL=\"$CUSTOM_URL\"\nTOKEN=\"$CUSTOM_TEMPLATE\"\nSEED=\"${CUSTOM_PASS:-}\"\nWORKER=\"${WORKER_NAME:-hiveos}\"\n"
[[ -n $CUSTOM_USER_CONFIG ]] && conf+="$CUSTOM_USER_CONFIG\n"
[[ -z $CUSTOM_CONFIG_FILENAME ]] && echo -e "${RED}No CUSTOM_CONFIG_FILENAME${NOCOLOR}" && exit 1
echo -e "$conf" > "$CUSTOM_CONFIG_FILENAME"
