#!/usr/bin/env bash
# Re-issue the CLAUDE_CODE_OAUTH_TOKEN across every plist that carries it,
# then restart the backend. Operator-run; ask #26.
#
# The token is READ SILENTLY (never echoed, never in argv, never in shell
# history) and is written only into the launchd plists, which already hold it.
# It is never printed, logged, or committed -- the script reports only lengths,
# counts and a truncated hash, the same discipline used in the ask.
#
#   bash scripts/ops/reissue_cc_oauth_token.sh
#
# Get a fresh token first with:  claude setup-token
set -euo pipefail

PLISTS=(
  "$HOME/Library/LaunchAgents/com.pyfinagent.backend.plist"
  "$HOME/Library/LaunchAgents/com.pyfinagent.away-session-am.plist"
  "$HOME/Library/LaunchAgents/com.pyfinagent.away-session-pm.plist"
  "$HOME/Library/LaunchAgents/com.pyfinagent.away-watchdog.plist"
)

echo "This will replace CLAUDE_CODE_OAUTH_TOKEN in ${#PLISTS[@]} plists and restart the backend."
echo
printf 'Paste the new token (input hidden), then press Enter: '
IFS= read -r -s TOKEN
echo

# ---- structural validation, BEFORE touching anything -----------------------
# The current live value fails all three of these checks, which is the defect.
fail=0
len=${#TOKEN}
prefixes=$(printf '%s' "$TOKEN" | grep -o 'sk-ant-oat' | wc -l | tr -d ' ')
case "$TOKEN" in *$'\n'*) newline=yes ;; *) newline=no ;; esac

[ "$len" -lt 40 ]        && { echo "REFUSED: length $len is too short to be a token."; fail=1; }
[ "$prefixes" != "1" ]   && { echo "REFUSED: found $prefixes 'sk-ant-oat' prefixes, expected exactly 1 (a double-paste is what broke it last time)."; fail=1; }
[ "$newline" = yes ]     && { echo "REFUSED: the value contains an embedded newline."; fail=1; }
case "$TOKEN" in sk-ant-oat*) ;; *) echo "REFUSED: does not start with sk-ant-oat."; fail=1 ;; esac
[ "$fail" -eq 1 ] && { echo; echo "Nothing was changed."; exit 1; }

echo "Validated: len=$len, prefixes=1, no embedded newline, sha256[:12]=$(printf '%s' "$TOKEN" | shasum -a 256 | cut -c1-12)"
echo

# ---- write ------------------------------------------------------------------
for p in "${PLISTS[@]}"; do
  [ -f "$p" ] || { echo "  SKIP (absent): $p"; continue; }
  cp "$p" "$p.bak.$(date +%Y%m%dT%H%M%S)"
  TOKEN="$TOKEN" /usr/bin/python3 - "$p" <<'PY'
import os, plistlib, sys
path = sys.argv[1]
with open(path, "rb") as f:
    d = plistlib.load(f)
d.setdefault("EnvironmentVariables", {})["CLAUDE_CODE_OAUTH_TOKEN"] = os.environ["TOKEN"]
with open(path, "wb") as f:
    plistlib.dump(d, f)
print(f"  updated: {os.path.basename(path)}")
PY
done
unset TOKEN

# ---- restart the backend ----------------------------------------------------
echo
echo "Restarting the backend (launchd owns the pid; kickstart, never pkill)..."
launchctl kickstart -k "gui/$(id -u)/com.pyfinagent.backend"
sleep 6
for i in 1 2 3 4 5 6 7 8 9 10; do
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://localhost:8000/api/health || true)
  [ "$code" = "200" ] && break
  sleep 3
done
echo "  /api/health = ${code:-no-response}"
launchctl list | grep com.pyfinagent.backend | awk '{print "  launchd pid="$1}'

echo
echo "DONE. Next: tell Claude the token is replaced. It will verify the rail with"
echo "the one remaining authorized verification cycle and report whether the book trades."
