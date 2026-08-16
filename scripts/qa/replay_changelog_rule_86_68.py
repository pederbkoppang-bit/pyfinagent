#!/usr/bin/env python3
"""phase-86.68 criteria 1/3/4/6 -- replay the OLD and NEW changelog rules over
real history and count version bumps per step.

Read-only: runs `git show` and parses. Nothing is written, nothing is committed.
Run as a FILE (macOS spawn re-imports __main__ by path).
"""
from __future__ import annotations
import json, os, re, subprocess, sys

REPO = "."

def sh(*a):
    r = subprocess.run(list(a), capture_output=True, text=True, cwd=REPO, timeout=60)
    return r.returncode, r.stdout

# ---- the OLD rule, verbatim from the retired classifier -------------------
def old_rule(subject: str, body: str) -> str:
    s = subject.strip(); b = body or ""
    if re.match(r"^[a-z]+(?:\([^)]*\))?!:", s): return "major"
    if re.search(r"^BREAKING CHANGE:", b, re.M): return "major"
    m = re.match(r"^phase-(\d+)\.(\d+)(?:\.\d+)?:", s)
    if m: return "minor" if m.group(2) == "0" else "patch"
    m = re.match(r"^([a-z]+)(?:\([^)]*\))?:", s)
    if m:
        t = m.group(1)
        if t == "feat": return "minor"
        if t in ("fix","bug","perf"): return "patch"
        if t in ("chore","docs","refactor","test","style","ci","build"): return "none"
    return "patch"

# ---- the NEW rule: subject may only force MAJOR; else the masterplan diff --
def statuses(ref):
    rc, out = sh("git","show",f"{ref}:.claude/masterplan.json")
    if rc != 0 or not out.strip(): return None
    acc = {}
    def walk(o):
        if isinstance(o, dict):
            if isinstance(o.get("id"),str) and isinstance(o.get("status"),str):
                acc[o["id"]] = o["status"]
            for v in o.values(): walk(v)
        elif isinstance(o, list):
            for v in o: walk(v)
    try: walk(json.loads(out))
    except Exception: return None
    return acc

_ABSENT = object()

def newly_done_ids(after, before, *, count_created):
    """The three-state membership test.

    phase-86.91: `count_created=False` is the SHIPPED (defective) predicate --
    `before.get(sid) not in (None, "done")` drops every step that DID NOT EXIST
    at HEAD~1, i.e. exactly the file-it-and-fix-it workflow. `count_created=True`
    is the FIXED one. Keeping BOTH here is what lets this script report three
    numbers from one execution; when only the hook was fixed, this file kept
    measuring the old predicate and the "baseline" silently went stale.
    """
    created = [s for s,st in after.items()
               if st=="done" and before.get(s, _ABSENT) is _ABSENT]
    transitioned = [s for s,st in after.items()
                    if st=="done" and before.get(s, _ABSENT) is not _ABSENT
                    and before.get(s) != "done"]
    return (created + transitioned) if count_created else transitioned

def flip_magnitude(sha, *, enabled=True, count_created=True):
    """enabled=False is the MUTANT: the flip gate is removed.
    count_created=False replays the SHIPPED pre-86.91 predicate."""
    if not enabled:
        return None  # caller falls back to the subject verdict = old behaviour
    after, before = statuses(sha), statuses(sha + "~1")
    if after is None or before is None: return "none"
    newly = newly_done_ids(after, before, count_created=count_created)
    if not newly: return "none"
    for sid in newly:
        top = sid.split(".")[0]
        sib = [st for s2,st in after.items() if s2.split(".")[0]==top]
        if sib and all(st=="done" for st in sib): return "major"
    for sid in newly:
        if re.fullmatch(r"\d+\.0", sid): return "minor"
    return "patch"

def new_rule(sha, subject, body, *, flip_enabled=True, count_created=True):
    bt = old_rule(subject, body)
    if bt != "major":
        fm = flip_magnitude(sha, enabled=flip_enabled, count_created=count_created)
        if fm is not None: bt = fm
    return bt

# ---- corpus ---------------------------------------------------------------
# phase-86.91: PINNED, and the pin is the point. `--since=2026-08-11` (a BARE
# date) is applied by git at the CURRENT TIME OF DAY, so the corpus SLIDES as the
# clock advances: measured 2026-08-16, the identical command returned 621 commits
# at 09:56 and 592 at 10:17, while `--since=2026-08-11T00:00:00` returns 706. The
# phase-86.68 artifacts' "348 commits from 2026-08-11" was produced by that
# sliding cutoff and is NOT reproducible -- it is a number about a clock, not
# about a corpus. An explicit timestamp makes the replay deterministic.
CORPUS_SINCE = "2026-08-11T00:00:00Z"
# PINNED AT BOTH ENDS, and the second pin is phase-86.91 cycle-2. Pinning only
# CORPUS_SINCE fixed the LOWER bound while the upper bound still floated with
# HEAD, so the headline counts still moved: this file's own artifact claimed
# "anyone re-running it gets 706 / 250 / 9 / 11" and the cycle-1 Q/A measured
# 710 / 252 / 9 / 11 two hours later, the delta being exactly the four commits
# that landed in between. In a step whose finding is "that is a number about a
# clock", a half-pinned corpus is the same defect one end over.
#
# This is a REPLAY of history, not a live gate: its job is to validate a rule
# against a fixed corpus, and the live gate is verify_changelog_flip_86_91.py.
# Override with CORPUS_UNTIL=HEAD to measure a moving window on purpose -- the
# resolved endpoint is printed either way, so no count is ever quoted without
# the endpoint it was measured against.
CORPUS_UNTIL = os.environ.get("CORPUS_UNTIL", "8dc70502")
_log_args = ["git","log",f"--since={CORPUS_SINCE}","--format=%H%x1f%s%x1f%b%x1e"]
if CORPUS_UNTIL: _log_args.append(CORPUS_UNTIL)
rc, out = sh(*_log_args)
commits = []
for rec in out.split("\x1e"):
    rec = rec.strip("\n")
    if not rec.strip(): continue
    parts = rec.split("\x1f")
    if len(parts) >= 2: commits.append((parts[0], parts[1], parts[2] if len(parts)>2 else ""))

_rc_end, _end = sh("git", "rev-parse", "--short", CORPUS_UNTIL or "HEAD")
print(f"corpus: {len(commits)} commits in [{CORPUS_SINCE} .. {(CORPUS_UNTIL or 'HEAD')}"
      f" = {_end.strip()}]")
print("        BOTH ENDS PINNED -- a bare --since date slides with the clock AND an")
print("        unpinned upper bound slides with HEAD; every count below is quoted")
print("        against the endpoint printed above.")
print("RULE STATED: OLD = subject-only (phase-X.Y -> patch). NEW = subject may force")
print("             MAJOR only; otherwise the parsed masterplan id->status diff decides.\n")

old_b = sum(1 for _,s,b in commits if old_rule(s,b) != "none")
shipped_b = sum(1 for h,s,b in commits if new_rule(h,s,b, count_created=False) != "none")
fixed_b   = sum(1 for h,s,b in commits if new_rule(h,s,b, count_created=True)  != "none")
print(f"  version bumps under OLD rule (subject prefix)     : {old_b}")
print(f"  version bumps under SHIPPED flip rule (pre-86.91) : {shipped_b}")
print(f"  version bumps under FIXED flip rule (86.91)       : {fixed_b}\n")

# phase-86.91 criterion 3: the increase must be accounted for COMMIT BY COMMIT.
gained = [(h,s) for h,s,b in commits
          if new_rule(h,s,b, count_created=False) == "none"
          and new_rule(h,s,b, count_created=True)  != "none"]
print(f"CRITERION 3 -- commits the SHIPPED rule swallowed and the FIXED rule bumps: {len(gained)}")
for h,subj in gained:
    after, before = statuses(h), statuses(h+"~1")
    created = [x for x,st in after.items()
               if st=="done" and before.get(x, _ABSENT) is _ABSENT]
    print(f"  {h[:8]}  created-and-closed={created}  {subj[:64]}")
if not gained:
    print("  (none -- if this is empty the fix changed nothing on this corpus)")
print()

# ---- criterion 3: the two PARKED steps ------------------------------------
print("CRITERION 3 -- PARKED steps must not bump:")
for step in ("86.9","86.44"):
    pat = re.compile(rf"^phase-{re.escape(step)}[.:\s]")
    sel = [(h,s,b) for h,s,b in commits if pat.match(s)]
    o = sum(1 for _,s,b in sel if old_rule(s,b) != "none")
    n = sum(1 for h,s,b in sel if new_rule(h,s,b) != "none")
    st = statuses("HEAD").get(step)
    print(f"  {step:6} commits={len(sel):3}  OLD bumps={o:3}  NEW bumps={n:3}  masterplan status={st}")

# ---- criterion 6: MUTATION -- remove the flip gate ------------------------
print("\nCRITERION 6 -- MUTATION (flip gate removed):")
all_killed_ok = []
ctrl_ok = True
for step in ("86.9","86.44"):
    pat = re.compile(rf"^phase-{re.escape(step)}[.:\s]")
    sel = [(h,s,b) for h,s,b in commits if pat.match(s)]
    ctrl = sum(1 for h,s,b in sel if new_rule(h,s,b,flip_enabled=True)  != "none")
    mut  = sum(1 for h,s,b in sel if new_rule(h,s,b,flip_enabled=False) != "none")
    green = (ctrl == 0)
    ctrl_ok &= green
    killed = mut > ctrl
    # phase-86.68 cycle-2, from the Q/A's residual note: the exit code used to gate
    # ONLY on control-greenness, so a run whose cells all SURVIVED still exited 0 and
    # the quoted `REAL exit=0` evidenced nothing about a kill. Gate on BOTH.
    all_killed_ok.append(green and killed)
    print(f"  {step:6} CONTROL={ctrl} ({'GREEN' if green else 'NOT GREEN -- cell UNSCORABLE'})"
          f"  MUTANT={mut}  -> {'KILLED' if (green and killed) else 'SURVIVED/UNSCORABLE'}")

print("\nCRITERION 4 -- rows are independent of the bump (see experiment_results C4):")
print("  The row-insert is UNCONDITIONAL; the version header and bullet are gated on")
print("  bump_type. NOTE: the table is capped at MAX_ROWS=20, so a row COUNT is not a")
print("  census -- older eligible commits are trimmed. Do not read row-count as coverage.")
ok = ctrl_ok and all(all_killed_ok) and len(all_killed_ok) > 0
print(f"\nexit gate: control_green={ctrl_ok} all_cells_killed={all(all_killed_ok)} "
      f"cells_scored={len(all_killed_ok)} -> exit {0 if ok else 1}")
sys.exit(0 if ok else 1)
