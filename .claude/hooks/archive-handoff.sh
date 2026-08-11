#!/usr/bin/env bash
# archive-handoff.sh — Copy/move handoff/current/* into
# handoff/archive/phase-<id>/ when a step in .claude/masterplan.json has
# status=done AND its archive dir does not yet exist.
# Triggered by PostToolUse on Write(.claude/masterplan.json).
#
# Source-of-truth for "already archived" = presence of
# handoff/archive/phase-<sid>/. Idempotent by construction: running the
# hook twice with no masterplan changes is a no-op.
#
# Historical note (2026-04-20 -> 2026-04-24): the previous version of
# this hook diffed HEAD:masterplan.json vs working tree to find newly
# done steps. When masterplan.json accumulated many done flips before
# the next commit, every write saw N steps as "newly done" and the
# -v2/-v3 suffix logic minted duplicate versioned archive dirs on every
# run. The filesystem-as-SoT approach below removes that failure mode,
# so the `.claude/archive-handoff.disabled` remediation flag is no
# longer load-bearing. Kept as an emergency kill switch only.
if [ -f "${CLAUDE_PROJECT_DIR:-$(pwd)}/.claude/archive-handoff.disabled" ]; then
    exit 0
fi

# Fail-open: 2026-05-16 fix. trap exit 0 ensures the hook never
# returns non-zero to the harness even when `set -e` would otherwise
# kill the script silently mid-flow.
trap 'exit 0' EXIT
set -uo pipefail

REPO="${CLAUDE_PROJECT_DIR:-}"
if [ -z "$REPO" ]; then
  REPO="$(git rev-parse --show-toplevel 2>/dev/null || true)"
fi
if [ -z "$REPO" ]; then
  REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
[ -z "$REPO" ] && exit 0

MASTERPLAN="$REPO/.claude/masterplan.json"
CURRENT_DIR="$REPO/handoff/current"
ARCHIVE_ROOT="$REPO/handoff/archive"
# State file owned by this hook. Records the set of step ids already
# seen as `done` so we archive only genuinely-new transitions. Self-
# seeded on first run (see NEWLY_DONE below): if missing, populated
# with every currently-done id and the hook emits nothing that turn.
STATE_FILE="$REPO/.claude/.archive-baseline.json"

[ -f "$MASTERPLAN" ] || exit 0
[ -d "$CURRENT_DIR" ] || exit 0

# Gather step ids that transitioned to `done` SINCE the last time the
# hook ran (baseline state file). First run seeds the baseline with all
# currently-done ids as "already seen" so we never retro-archive 100+
# historical steps against the wrong rolling-file snapshot.
NEWLY_DONE=$(python3 - "$REPO" "$MASTERPLAN" "$STATE_FILE" << 'PYEOF'
import json, sys, os

repo, mp_path, state_path = sys.argv[1], sys.argv[2], sys.argv[3]
archive_root = os.path.join(repo, "handoff", "archive")

with open(mp_path) as f:
    doc = json.load(f)

def walk(node, out):
    if isinstance(node, dict):
        sid = node.get("id")
        status = node.get("status")
        if sid and status == "done":
            out.append(str(sid))
        for v in node.values():
            walk(v, out)
    elif isinstance(node, list):
        for v in node:
            walk(v, out)

current_done = []
walk(doc, current_done)
# Dedupe preserving order (masterplan can list same id under multiple parents).
seen_once = set()
current_set = []
for sid in current_done:
    if sid not in seen_once:
        seen_once.add(sid)
        current_set.append(sid)

# Seed baseline on first run: treat every currently-done id as already
# seen. Emit nothing and let the hook write the baseline.
if not os.path.isfile(state_path):
    with open(state_path, "w") as f:
        json.dump({"seen_done": sorted(seen_once)}, f, indent=2)
    sys.exit(0)

with open(state_path) as f:
    state = json.load(f)
baseline = set(state.get("seen_done", []))

newly = [sid for sid in current_set if sid not in baseline]

# Filter further: belt-and-suspenders -- if archive dir already exists,
# skip (covers the case where state file was deleted + recreated).
to_archive = []
for sid in newly:
    short = sid[len("phase-"):] if sid.startswith("phase-") else sid
    if not os.path.isdir(os.path.join(archive_root, f"phase-{short}")):
        to_archive.append(sid)

# Update baseline to include the ones we are about to archive (and any
# skipped-because-dir-already-exists ones too, so we never look at them
# again).
new_baseline = baseline | set(newly)
with open(state_path, "w") as f:
    json.dump({"seen_done": sorted(new_baseline)}, f, indent=2)

for sid in to_archive:
    print(sid)
PYEOF
)

[ -z "$NEWLY_DONE" ] && exit 0

# phase-86.29: does a rolling file DECLARE the step we are archiving?
#
# Exit 0 (true) ONLY on an explicit declaration whose step id equals $2.
# Everything else -- no declaration found, a DIFFERENT step declared, an
# unreadable file -- is FALSE. That asymmetry is the whole fix: copying a
# rolling file that does not declare this step is precisely the defect
# (`handoff/archive/phase-86.6/contract.md` began "# Contract -- phase-82.54"),
# so "unsure" must mean "do not copy", never "copy anyway".
#
# Implemented in python3 rather than sed/grep because this hook must behave
# identically under BSD (macOS, the only deployment) and GNU userland, and the
# patterns below are deliberately the SAME set used by
# scripts/qa/derive_archive_misattribution_86_29.py -- one declaration grammar,
# two consumers, so the census and the hook cannot drift into disagreeing about
# what "declares a step" means.
rolling_declares_step() {
    python3 - "$1" "$2" << 'PYEOF'
import re, sys
path, want = sys.argv[1], sys.argv[2]
try:
    with open(path, encoding="utf-8", errors="replace") as fh:
        head = fh.read(4000)
except OSError:
    sys.exit(1)
SID = r"[0-9]+(?:\.[0-9A-Za-z]+)*"
for pattern in (
    rf"^#\s*Contract\s*--\s*step\s*`?({SID})`?",
    rf"^#\s*(?:Sprint\s+)?Contract\s*--\s*(?:.*?)phase-({SID})",
    rf"^\*\*Step(?:\s+ID)?\*\*:\s*`?(?:phase-)?({SID})`?",
    rf"^#.*?\bphase-({SID})\b",
):
    m = re.search(pattern, head, re.M | re.I)
    if m:
        sys.exit(0 if m.group(1) == want else 1)
sys.exit(1)
PYEOF
}

# Accumulates human-visible warnings so they can be surfaced as a
# `systemMessage` at the end. A warning written only to stderr or to a
# gitignored log is invisible in practice -- which is how this defect ran
# unnoticed from 2026-08-06.
ARCHIVE_WARNINGS=""

# Archive each newly-done step. Exit codes swallowed so a partial failure
# does not block the tool call that triggered us.
archive_step() {
    local sid="$1"
    # phase-4.16.2 fix: masterplan step ids are inconsistent --
    # some are bare `4.14.1`, others already prefixed `phase-6.1`.
    # Strip any leading `phase-` so we do not produce `phase-phase-6.1/`.
    local short_sid="${sid#phase-}"
    local target="$ARCHIVE_ROOT/phase-$short_sid"

    # Defensive idempotency: the caller (NEWLY_DONE) already filters out
    # steps whose archive dir exists, but if concurrent runs racy-create
    # the same dir we skip rather than mint a -v2.
    if [ -d "$target" ]; then
        echo "[archive-handoff] step $sid -> phase-$short_sid already archived, skipping" >&2
        return 0
    fi

    mkdir -p "$target"

    # phase-86.29 -- PROVENANCE: every archived file records where it came from.
    # The audit trail's failure mode was not that it lost data, it was that it
    # answered confidently with another step's work. A dir that states its own
    # sources cannot do that silently again.
    local prov="$target/PROVENANCE.md"
    {
        echo "# Archive provenance -- phase-$short_sid"
        echo
        echo "Written by \`.claude/hooks/archive-handoff.sh\` (phase-86.29 revision)."
        echo "Masterplan step id as recorded: \`$sid\` -> archive dir \`phase-$short_sid\`."
        echo "Archived at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo
        echo "| archived as | source | how |"
        echo "|---|---|---|"
    } > "$prov" 2>/dev/null || true

    # phase-86.29 -- BRANCH 1, THE PRIMARY ONE: DERIVE the source names from the
    # step id instead of hoping a glob written for the 2024 convention still
    # matches. `contract_86.29.md` -> `<archive>/contract.md`.
    #
    # This removes the hook/convention agreement dependency rather than
    # re-tuning it. The old step-specific globs at the bottom of this function
    # expected the sid at the FRONT before a hyphen (`4.5.9-contract.md`); the
    # convention has put it at the END after an underscore since ~phase-4.9, so
    # those globs matched ZERO files for every sid and only the rolling branch
    # below ever fired -- measured for sids 86.29 / 86.6 / 82.54 / 86.31 / 86.26,
    # 0 matches each, against a positive control proving the count can be 1.
    local derived=0
    local base src
    for base in contract experiment_results evaluator_critique research_brief live_check; do
        src="$CURRENT_DIR/${base}_${short_sid}.md"
        if [ -f "$src" ] && cp "$src" "$target/${base}.md" 2>/dev/null; then
            derived=$((derived + 1))
            printf '| `%s.md` | `handoff/current/%s` | derived from step id |\n' \
                "$base" "$(basename "$src")" >> "$prov" 2>/dev/null || true
        fi
        # Variants such as `research_brief_86.29_rerun.md` keep their own name;
        # dropping them would lose the artifact that actually passed the gate.
        for extra in "$CURRENT_DIR/${base}_${short_sid}_"*.md; do
            if [ -f "$extra" ] && cp "$extra" "$target/$(basename "$extra")" 2>/dev/null; then
                derived=$((derived + 1))
                printf '| `%s` | `handoff/current/%s` | derived variant |\n' \
                    "$(basename "$extra")" "$(basename "$extra")" >> "$prov" 2>/dev/null || true
            fi
        done
    done

    # phase-86.29 -- BRANCH 2: the rolling phase-level files are now a GUARDED
    # FALLBACK, not the default. Two conditions, both required:
    #   (a) the derived branch did not already supply this artifact, and
    #   (b) the rolling file DECLARES this step.
    # Condition (b) is the fix. Blindly substituting `contract.md` for a missing
    # `contract_<sid>.md` IS the bug -- it is how 153 archive dirs came to hold
    # another step's work. COPY (not move) so downstream verifiers keep reading
    # the live file between step transitions.
    local copied=0
    local skipped_rolling=0
    for f in contract.md experiment_results.md evaluator_critique.md research.md research_brief.md; do
        [ -f "$CURRENT_DIR/$f" ] || continue
        [ -f "$target/$f" ] && continue          # derived branch already won
        if rolling_declares_step "$CURRENT_DIR/$f" "$short_sid"; then
            if cp "$CURRENT_DIR/$f" "$target/$f" 2>/dev/null; then
                copied=$((copied + 1))
                printf '| `%s` | `handoff/current/%s` | rolling file, declares this step |\n' \
                    "$f" "$f" >> "$prov" 2>/dev/null || true
            fi
        else
            skipped_rolling=$((skipped_rolling + 1))
            printf '| -- | `handoff/current/%s` | SKIPPED: does not declare phase-%s |\n' \
                "$f" "$short_sid" >> "$prov" 2>/dev/null || true
        fi
    done

    # Step-specific files: MOVE (these are the per-substep contracts like
    # handoff/current/4.5.9-contract.md, which do belong only to one step).
    # phase-4.16.2 fix: match BOTH `<sid>-*.md` AND `phase-<sid>-*.md`
    # (the `phase-` prefix became the convention from ~phase-4.14 onward
    # and the old single-glob left 150 files stranded).
    local moved=0
    for f in "$CURRENT_DIR/${sid}-"*.md "$CURRENT_DIR/phase-${sid}-"*.md; do
        if [ -f "$f" ]; then
            local base="$(basename "$f")"
            if git -C "$REPO" mv "handoff/current/$base" "handoff/archive/$(basename "$target")/$base" 2>/dev/null; then
                moved=$((moved + 1))
            elif mv "$f" "$target/$base" 2>/dev/null; then
                moved=$((moved + 1))
            fi
        fi
    done

    # phase-86.29 -- FAIL LOUDLY ON AN EMPTY ARCHIVE (criterion 4, second branch).
    # Before this, a run that archived three other steps' files printed
    # `copied=4 moved=0` and read as success. An archive that captured nothing
    # for the step it is named after is a VISIBLE FAILURE, and it is recorded in
    # three places: stderr, the durable log, and PROVENANCE.md inside the dir
    # itself -- so it survives even if nobody was watching the transcript.
    local total=$((derived + copied + moved))
    if [ "$total" -eq 0 ]; then
        local msg="[archive-handoff] FAILURE: step $sid -> phase-$short_sid archived NOTHING. No handoff/current/<artifact>_${short_sid}.md exists, and $skipped_rolling rolling file(s) were skipped because none declares this step. The archive directory is EMPTY BY DESIGN rather than silently filled with another step's work."
        echo "$msg" >&2
        {
            echo
            echo "## RESULT: FAILED -- nothing was archived"
            echo
            echo "$msg"
        } >> "$prov" 2>/dev/null || true
        mkdir -p "$REPO/handoff/logs" 2>/dev/null || true
        printf '%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$msg" \
            >> "$REPO/handoff/logs/archive-handoff.log" 2>/dev/null || true
        ARCHIVE_WARNINGS="${ARCHIVE_WARNINGS}${msg}
"
    else
        {
            echo
            echo "## RESULT: ok -- derived=$derived rolling_copied=$copied legacy_moved=$moved rolling_skipped=$skipped_rolling"
        } >> "$prov" 2>/dev/null || true
    fi

    echo "[archive-handoff] step $sid -> $(basename "$target") (derived=$derived copied=$copied moved=$moved skipped_rolling=$skipped_rolling)" >&2
}

while IFS= read -r sid; do
    [ -z "$sid" ] && continue
    archive_step "$sid" || true
done <<< "$NEWLY_DONE"

# phase-86.29: surface any empty-archive failure to the operator. stderr from a
# PostToolUse hook lands in the transcript at best; a `systemMessage` on stdout
# is the documented channel that is actually seen, and it does NOT block the
# masterplan Write (this hook must never fail it -- see the fail-open trap).
# python3 does the serialisation so a message containing quotes or newlines
# cannot emit malformed JSON.
if [ -n "$ARCHIVE_WARNINGS" ]; then
    python3 - "$ARCHIVE_WARNINGS" << 'PYEOF' 2>/dev/null || true
import json, sys
print(json.dumps({"systemMessage": sys.argv[1].strip()}))
PYEOF
fi

exit 0
