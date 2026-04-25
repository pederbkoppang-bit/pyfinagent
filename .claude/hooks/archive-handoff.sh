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

set -euo pipefail

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

    # Rolling phase-level files: COPY (not move) so downstream verifiers can
    # keep reading them between step transitions. The per-step snapshot goes
    # to the archive; the live file keeps serving cross-verification.
    # phase-4.16.2 fix: add research_brief.md (actual file name since
    # phase-4.9; `research.md` was the old name and never matched).
    local copied=0
    for f in contract.md experiment_results.md evaluator_critique.md research.md research_brief.md; do
        if [ -f "$CURRENT_DIR/$f" ]; then
            if cp "$CURRENT_DIR/$f" "$target/$f" 2>/dev/null; then
                copied=$((copied + 1))
            fi
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

    echo "[archive-handoff] step $sid -> $(basename "$target") (copied=$copied moved=$moved)" >&2
}

while IFS= read -r sid; do
    [ -z "$sid" ] && continue
    archive_step "$sid" || true
done <<< "$NEWLY_DONE"

exit 0
