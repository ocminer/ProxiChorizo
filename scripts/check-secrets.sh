#!/bin/bash
# Pre-commit exposure check for the public ProxiChorizo repository.
# Refuses the commit if any staged file contains private network addresses,
# internal host names, credentials, key material or other sensitive patterns.
# Usage: scripts/check-secrets.sh            (checks the staged files)
#        scripts/check-secrets.sh --all      (checks the whole tree)
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
if [ "${1:-}" = "--all" ]; then FILES=$(git ls-files); else FILES=$(git diff --cached --name-only --diff-filter=ACM); fi
[ -z "$FILES" ] && exit 0
PATTERNS=(
  # private / internal networks and VPN ranges
  '\b10\.(21|1)\.[0-9]+\.[0-9]+\b'
  '\b192\.168\.[0-9]+\.[0-9]+\b'
  '\b172\.(1[6-9]|2[0-9]|3[01])\.[0-9]+\.[0-9]+\b'
  '\b5\.9\.193\.22\b' '\b51\.89\.7\.224\b'
  # internal host names
  'hp0[0-9]\b' 'rig0[0-9]\b' 'us-rig' 'hay-rig' 'sprdlm' 'gadtr' 'haynau' 'bri4-' 'server[1-6]\b'
  # credentials and key material
  # credential VALUES (field names like private_key in parsing code are fine)
  '(password|passphrase|secret|token|dsn|seed)[a-z_]* *[:=] *["'"'"'][^"'"'"' $]{8,}' 'PRIVATE KEY' 'Bearer [A-Za-z0-9+/=_-]{20,}'
  'wg set' 'PublicKey *=' 'AllowedIPs' 'ssh-ed25519 ' 'ssh-rsa '
  # anything that looks like a 64-hex key or a 32+ char base64 blob assigned to something
  '"(private|secret|seed|token)[a-z_]*" *: *"[0-9a-fA-F]{32,}'
)
rc=0
for f in $FILES; do
  [ -f "$f" ] || continue
  # vectors.json holds published TEST vectors (key derived from a fixed public string), not a real key
  case "$f" in *.png|*.jpg|*.gif|*.ico|*.woff*|*.ttf|vectors.json|scripts/check-secrets.sh|.gitignore) continue;; esac
  for p in "${PATTERNS[@]}"; do
    if grep -nE -- "$p" "$f" | grep -v 'check-secrets.sh' > /tmp/px-secret-hit.$$ 2>/dev/null; then
      echo "EXPOSURE in $f (pattern: $p):"; head -5 /tmp/px-secret-hit.$$; rc=1
    fi
  done
done
rm -f /tmp/px-secret-hit.$$
if [ $rc -ne 0 ]; then echo; echo "commit refused: remove the lines above or move them out of the repository"; fi
exit $rc
