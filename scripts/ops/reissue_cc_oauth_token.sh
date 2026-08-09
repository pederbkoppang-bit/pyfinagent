#!/usr/bin/env bash
# Re-issue the CLAUDE_CODE_OAUTH_TOKEN across every plist that carries it.
# Operator-run; ask #26.
#
# The token is READ SILENTLY (never echoed, never in argv, never in shell
# history) and written only into the launchd plists, which already hold it. It
# is never printed, logged or committed -- this script reports only lengths,
# counts and a truncated hash.
#
#   bash scripts/ops/reissue_cc_oauth_token.sh            # install
#   bash scripts/ops/reissue_cc_oauth_token.sh --verify   # check only, changes nothing
#
# Get a fresh token first with:  claude setup-token
set -euo pipefail

RELOAD_HINT_1='launchctl bootout gui/$(id -u)/com.pyfinagent.backend'
RELOAD_HINT_2='launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.pyfinagent.backend.plist'

# ---- --verify: compare the RUNNING env against the plist. Changes nothing. ---
if [ "${1:-}" = "--verify" ]; then
  /usr/bin/python3 - <<'PY'
import subprocess, plistlib, os, hashlib, sys
lst = subprocess.run(["launchctl", "list"], capture_output=True, text=True).stdout
pid = next((l.split()[0] for l in lst.splitlines()
            if l.rstrip().endswith("com.pyfinagent.backend")), None)
if not pid or not pid.isdigit():
    print("  backend is not running"); sys.exit(1)
out = subprocess.run(["ps", "-Eww", "-o", "command=", "-p", pid],
                     capture_output=True, text=True).stdout
run = next((p.split("=", 1)[1] for p in out.split()
            if p.startswith("CLAUDE_CODE_OAUTH_TOKEN=")), None)
d = plistlib.load(open(os.path.expanduser(
    "~/Library/LaunchAgents/com.pyfinagent.backend.plist"), "rb"))
want = d["EnvironmentVariables"]["CLAUDE_CODE_OAUTH_TOKEN"]
h = lambda v: hashlib.sha256(v.encode()).hexdigest()[:12]
if run is None:
    print("  token not visible in the process environment"); sys.exit(1)
print(f"  plist  : len={len(want)} prefixes={want.count('sk-ant-oat')} sha={h(want)}")
print(f"  running: len={len(run)}  prefixes={run.count('sk-ant-oat')} sha={h(run)}  (pid {pid})")
if h(run) == h(want):
    print("  MATCH -- the running backend has the token from the plist.")
    sys.exit(0)
print("  MISMATCH -- the process is running a STALE token.")
print("  A kickstart is NOT enough: launchd serves the job definition it cached")
print("  at bootstrap time. Only a bootout + bootstrap re-reads the plist.")
sys.exit(1)
PY
  exit $?
fi

PLISTS=(
  "$HOME/Library/LaunchAgents/com.pyfinagent.backend.plist"
  "$HOME/Library/LaunchAgents/com.pyfinagent.away-session-am.plist"
  "$HOME/Library/LaunchAgents/com.pyfinagent.away-session-pm.plist"
  "$HOME/Library/LaunchAgents/com.pyfinagent.away-watchdog.plist"
)

echo "This writes CLAUDE_CODE_OAUTH_TOKEN into ${#PLISTS[@]} plists. It does NOT restart anything."
echo
printf 'Paste the new token (input hidden), then press Enter: '
IFS= read -r -s TOKEN
echo

# ---- structural validation, BEFORE touching anything -----------------------
# The value that broke the rail on 2026-08-09 fails two of these.
fail=0
len=${#TOKEN}
prefixes=$(printf '%s' "$TOKEN" | grep -o 'sk-ant-oat' | wc -l | tr -d ' ')
case "$TOKEN" in *$'\n'*) newline=yes ;; *) newline=no ;; esac

[ "$len" -lt 40 ]      && { echo "REFUSED: length $len is too short to be a token."; fail=1; }
[ "$prefixes" != "1" ] && { echo "REFUSED: found $prefixes 'sk-ant-oat' prefixes, expected exactly 1 (a double-paste is what broke it)."; fail=1; }
[ "$newline" = yes ]   && { echo "REFUSED: the value contains an embedded newline."; fail=1; }
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

# ---- reload -----------------------------------------------------------------
# MEASURED 2026-08-09, THE HARD WAY. `launchctl kickstart -k` restarts the
# PROCESS but does NOT re-read the plist: launchd serves the job definition it
# cached at bootstrap. A fresh backend came up at 14:57:07 -- one second after
# the plist was written -- still holding a STALE token, while the file on disk
# was already correct. The first version of this script called kickstart and
# reported success on exactly that state. Hence --verify above, and hence this
# is now an explicit operator step rather than something the script pretends to
# have done.
#
# bootout/bootstrap is deliberately NOT automated: away-ops rail 9 reserves it
# for the operator, because a bootout that succeeds followed by a bootstrap that
# fails leaves paper trading UNLOADED rather than merely stopped.
cat <<EOF

The plists are written. The RUNNING backend does not have the new token yet.

Run these two lines yourself -- a kickstart will NOT pick it up:

    $RELOAD_HINT_1
    $RELOAD_HINT_2

Then confirm it actually took effect:

    bash scripts/ops/reissue_cc_oauth_token.sh --verify

That prints MATCH only when the running process and the plist agree. Do not
trust a restart that you have not verified this way -- that is the exact trap
that cost 2026-08-09.
EOF
