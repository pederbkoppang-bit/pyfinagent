#!/usr/bin/env bash
# verify_phase_4000_1_baseline.sh -- step 4000.1 gate.
# Inspects the CC-rail baseline artifact for the six required sections (a)-(f)
# by CONTENT MARKERS scoped to each section's own text (masterplan 4000.1
# criterion 1; hardened after Q/A cycle-1 measured four file-wide-grep vacuity
# survivors). Sections (a) and (d) must carry fenced blocks; (d) must carry
# both computed cadence figures; (b) must carry the recorded NO conclusion.
# Accepts the artifact in handoff/current/ OR handoff/archive/phase-4000.1/ so
# the gate stays green after the archive hook rotates the step. An explicit
# path may be passed as $1 (used by the criterion-8 mutation demonstration).
set -u

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
CUR="$REPO/handoff/current/cc_rail_baseline_4000_1.md"
ARC="$REPO/handoff/archive/phase-4000.1/cc_rail_baseline_4000_1.md"

if [ "${1:-}" != "" ]; then
  ARTIFACT="$1"
elif [ -f "$CUR" ]; then
  ARTIFACT="$CUR"
elif [ -f "$ARC" ]; then
  ARTIFACT="$ARC"
else
  echo "FAIL: baseline artifact not found at $CUR or $ARC"
  exit 1
fi
if [ ! -f "$ARTIFACT" ]; then
  echo "FAIL: not a file: $ARTIFACT"
  exit 1
fi

echo "artifact: $ARTIFACT"
fails=0

# Extract one section's text: from the line starting "## Section (x)" up to
# (excluding) the next "## " heading. index()==1 sidesteps regex metachars in
# the parenthesised header.
section_text() {
  awk -v h="## Section ($1)" 'index($0, h) == 1 {f=1; next} /^## / {f=0} f' "$ARTIFACT"
}

# sec_check <letter> <label> <marker>... : every marker must appear in the
# section's OWN text; a missing section fails outright.
sec_check() {
  local letter="$1" label="$2"; shift 2
  local body marker
  body="$(section_text "$letter")"
  if [ -z "$body" ]; then
    echo "FAIL [$label]: section ($letter) missing or empty"
    fails=$((fails + 1))
    return
  fi
  for marker in "$@"; do
    if ! printf '%s\n' "$body" | grep -qF -- "$marker"; then
      echo "FAIL [$label]: marker not in section ($letter): $marker"
      fails=$((fails + 1))
      return
    fi
  done
  echo "ok   [$label]"
}

# (a) fenced projection capture: fence present, flag value inside, timestamp,
#     and the producing command disclosed in-section.
sec_check a "a: live flag capture" \
  '```' '"paper_use_claude_code_route": true' "captured 2026-" "curl -s"
# (b) process start vs 78.2 side by side + the recorded explicit NO conclusion
#     (guards the frozen fact against inversion, checksum-style).
sec_check b "b: process age vs 78.2" \
  "lstart" "5e51f4a9" "a75c209f" "predate the --model change? **NO**"
# (c) executable WHERE-clause rule + deriving citations, in-section.
sec_check c "c: rail row rule" \
  "provider = 'claude-code' OR agent = 'cc_rail' OR agent LIKE 'cc_rail:%'" \
  "spend.py:228-230" "backend/services/autonomous_loop.py:2347-2358"
# (d) cadence: fence, the stated cast+window, pairing column, verbatim query,
#     and BOTH computed figures.
sec_check d "d: cadence baseline" \
  '```' "SAFE_CAST(created_at AS TIMESTAMP)" "INTERVAL 28 DAY" "round_trip_id" \
  "1.75 trades/week" "0.75 RT/week"
# (e) chosen bounded entrypoint.
sec_check e "e: entrypoint" \
  "/api/analysis/" "abort" "MID-analysis"
# (f) all seven pre-registered checks with substantive bodies, not bare bullets.
sec_check f "f: E1-E7 definitions" \
  "E1 rail-used" "E2 metered-dark" "E3 model-truth" "E4 output-integrity" \
  "E5 guard-health" "E6 quota-math" "E7 UI" \
  "modelUsage" "fetch_llm_spend" "%-of-weekly-Max-pool" "Pass shape"

if [ "$fails" -gt 0 ]; then
  echo "RESULT: FAIL ($fails section check(s) failed)"
  exit 1
fi
echo "RESULT: PASS (all six sections present with required in-section content)"
exit 0
