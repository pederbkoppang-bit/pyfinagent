#!/usr/bin/env python3
"""phase-86.68 criteria 1/3/4/6 -- replay the OLD and NEW changelog rules over
real history and count version bumps per step.

Read-only: runs `git show` and parses. Nothing is written, nothing is committed.
Run as a FILE (macOS spawn re-imports __main__ by path).
"""
from __future__ import annotations
import json, re, subprocess, sys

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

def flip_magnitude(sha, *, enabled=True):
    """enabled=False is the MUTANT: the flip gate is removed."""
    if not enabled:
        return None  # caller falls back to the subject verdict = old behaviour
    after, before = statuses(sha), statuses(sha + "~1")
    if after is None or before is None: return "none"
    newly = [s for s,st in after.items() if st=="done" and before.get(s) not in (None,"done")]
    if not newly: return "none"
    for sid in newly:
        top = sid.split(".")[0]
        sib = [st for s2,st in after.items() if s2.split(".")[0]==top]
        if sib and all(st=="done" for st in sib): return "major"
    for sid in newly:
        if re.fullmatch(r"\d+\.0", sid): return "minor"
    return "patch"

def new_rule(sha, subject, body, *, flip_enabled=True):
    bt = old_rule(subject, body)
    if bt != "major":
        fm = flip_magnitude(sha, enabled=flip_enabled)
        if fm is not None: bt = fm
    return bt

# ---- corpus ---------------------------------------------------------------
rc, out = sh("git","log","--since=2026-08-11","--format=%H%x1f%s%x1f%b%x1e")
commits = []
for rec in out.split("\x1e"):
    rec = rec.strip("\n")
    if not rec.strip(): continue
    parts = rec.split("\x1f")
    if len(parts) >= 2: commits.append((parts[0], parts[1], parts[2] if len(parts)>2 else ""))

print(f"corpus: {len(commits)} commits since 2026-08-11 (re-derived at execution time)")
print("RULE STATED: OLD = subject-only (phase-X.Y -> patch). NEW = subject may force")
print("             MAJOR only; otherwise the parsed masterplan id->status diff decides.\n")

old_b = sum(1 for _,s,b in commits if old_rule(s,b) != "none")
new_b = sum(1 for h,s,b in commits if new_rule(h,s,b) != "none")
print(f"  version bumps under OLD rule : {old_b}")
print(f"  version bumps under NEW rule : {new_b}\n")

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
