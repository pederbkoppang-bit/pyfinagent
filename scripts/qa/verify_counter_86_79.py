#!/usr/bin/env python3
"""phase-86.79 -- re-runnable checker for the attempt-counter fix in qa_wip.py.

    source .venv/bin/activate && python scripts/qa/verify_counter_86_79.py

Exit 0 iff EVERY check passes AND the expected number of checks actually ran.
The cardinality floor is not decoration: a checker whose loop silently covers
nothing prints no failures and exits 0, which is indistinguishable from success.

WHAT IT PROVES, criterion by criterion
--------------------------------------
1. `records_retained` includes the current spawn (prior + 1), with the producing
   line GREP-DERIVED at runtime -- never a hardcoded line number, because the
   line has already moved twice during this step.
2. The old number was correct only because write-first ran first; the new
   `attempt_number` refuses rather than reporting the wrong number.
3. Pruning saturates `records_retained` at `keep`; `attempt_number` survives it
   via the loss ledger.
5. The 3rd-consecutive-CONDITIONAL boundary and F1b's 5-attempt budget both
   still fire against the corrected number, INCLUDING through a prune.
6. Verdict semantics are untouched, and every uncomputable path returns None.

Every write goes to a temp dir. The live repo is only ever READ.
"""
from __future__ import annotations

import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(REPO / "scripts" / "harness"))

# TEST-ONLY SEAM, identical in shape and purpose to the one in
# verify_wip_retention_86_36.py: `PYFIN_QA_WIP_OVERRIDE` points at a MUTANT COPY
# of qa_wip.py so mutation_matrix_86_79.py can drive this checker against a
# mutated subject without ever opening the tracked file for writing. Unset (the
# normal case) this imports the real module, so a plain run is unaffected. The
# RED half of criterion 7 is unprovable without a seam like this.
import importlib.util                            # noqa: E402
import os                                        # noqa: E402

_override = os.environ.get("PYFIN_QA_WIP_OVERRIDE")
if _override:
    _spec = importlib.util.spec_from_file_location("qa_wip", _override)
    qa_wip = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(qa_wip)
else:
    import qa_wip                                # noqa: E402
import verdict_history_86_21 as vh               # noqa: E402
import attempt_budget as ab                      # noqa: E402

SID = "99.1"
#: Cardinality floor. Cycle-1 Q/A noted 12 checks of slack against the real count
#: and that the constant's own instruction ("raise it when adding checks") had not
#: been followed; raised to sit just under the current total so a silently-skipped
#: block is caught rather than absorbed.
EXPECTED_CHECKS = 53

_results: list[tuple[str, bool, str]] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    _results.append((label, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))


def section(t: str) -> None:
    print(f"\n{'=' * 74}\n{t}\n{'=' * 74}")


def mk(tmp: pathlib.Path, sid: str, stamp: str) -> pathlib.Path:
    """One write-first record carrying an explicit WRITTEN stamp."""
    sink = tmp / qa_wip.MEMORY_DIR / qa_wip.WIP_SUBDIR
    sink.mkdir(parents=True, exist_ok=True)
    p = sink / f"verdict_wip_{sid}__{stamp}.md"
    iso = (f"{stamp[0:4]}-{stamp[4:6]}-{stamp[6:8]}T"
           f"{stamp[9:11]}:{stamp[11:13]}:{stamp[13:15]}Z")
    p.write_text(f"{qa_wip.COMPLETE_MARKER}\nWRITTEN: {iso}\nSTEP: {sid}\n\nbody\n",
                 encoding="utf-8")
    return p


def iso_of(stamp: str) -> str:
    return (f"{stamp[0:4]}-{stamp[4:6]}-{stamp[6:8]}T"
            f"{stamp[9:11]}:{stamp[11:13]}:{stamp[13:15]}Z")


STAMPS = [f"20260814T{h:02d}0000Z" for h in range(9, 20)]


# ══════════════════════════════════════════════════════ CRITERION 1
section("C1 -- records_retained counts the CURRENT spawn (prior + 1)")

# Read the SUBJECT UNDER TEST, not always the tracked file -- otherwise every
# source-text assertion below is blind to a mutant and would score a false kill.
SUBJECT_SRC = pathlib.Path(_override) if _override else REPO / "scripts/qa/qa_wip.py"
src_text = SUBJECT_SRC.read_text(encoding="utf-8")
producing = [(i + 1, ln) for i, ln in enumerate(src_text.splitlines())
             if '"records_retained": len(records)' in ln]
check("the producing statement is found by grep (line NOT hardcoded)",
      len(producing) == 1,
      f"qa_wip.py:{producing[0][0]}: {producing[0][1].strip()}" if producing
      else "NOT FOUND -- the field was renamed or removed")

tmp = pathlib.Path(tempfile.mkdtemp(prefix="c1_"))
for s in STAMPS[:2]:
    mk(tmp, SID, s)
cur = mk(tmp, SID, STAMPS[2])
rep = qa_wip.report(SID, repo=tmp, spawned_at=iso_of(STAMPS[2]))
check("records_retained == priors + 1", rep["records_retained"] == 2 + 1,
      f"priors=2 records_retained={rep['records_retained']}")
check("len(prior_records) == priors", len(rep["prior_records"]) == 2)
check("the NEW attempt_number is the same number, but unit-stated",
      rep["attempt_number"] == 3 and rep["prior_attempts"] == 2,
      f"attempt_number={rep['attempt_number']} prior_attempts={rep['prior_attempts']}")
check("records_retained_unit states it is a GAUGE, not a counter",
      "GAUGE" in rep.get("records_retained_unit", ""))
shutil.rmtree(tmp)


# ══════════════════════════════════════════════════════ CRITERION 2
section("C2 -- the old number was right ONLY because write-first ran first")

tmp = pathlib.Path(tempfile.mkdtemp(prefix="c2_"))
for s in STAMPS[:2]:
    mk(tmp, SID, s)
before = qa_wip.report(SID, repo=tmp, spawned_at=iso_of(STAMPS[2]))
mk(tmp, SID, STAMPS[2])
after = qa_wip.report(SID, repo=tmp, spawned_at=iso_of(STAMPS[2]))

check("records_retained DIFFERS across the write-first boundary (the coupling)",
      before["records_retained"] != after["records_retained"],
      f"before={before['records_retained']} after={after['records_retained']}")
check("the pre-write number is LOWER -- the coupling fails OPEN",
      before["records_retained"] < after["records_retained"])
check("attempt_number REFUSES before the write-first record exists",
      before["attempt_number"] is None,
      f"status={before['attempt_number_status']}")
check("...and it refuses with the right reason",
      before["attempt_number_status"] == "no_record_for_this_spawn")
check("...while still reporting the priors it genuinely knows",
      before["prior_attempts"] == 2, f"prior_attempts={before['prior_attempts']}")
check("...and it tells the reader what to do instead of falling back",
      "records_retained" in before["attempt_number_guidance"]
      and "prior_attempts + 1" in before["attempt_number_guidance"])
check("attempt_number is correct once write-first has run",
      after["attempt_number"] == 3, f"attempt_number={after['attempt_number']}")
shutil.rmtree(tmp)


# ══════════════════════════════════════════════════════ CRITERION 3
section("C3 -- pruning saturates records_retained; attempt_number survives it")

tmp = pathlib.Path(tempfile.mkdtemp(prefix="c3_"))
for s in STAMPS[:6]:
    mk(tmp, SID, s)
pre = qa_wip.report(SID, repo=tmp, spawned_at=iso_of(STAMPS[5]))
check("6 records -> records_retained == 6 and attempt_number == 6",
      pre["records_retained"] == 6 and pre["attempt_number"] == 6)

removed = qa_wip.prune_wip_records(SID, repo=tmp, keep=qa_wip.DEFAULT_KEEP)
post = qa_wip.report(SID, repo=tmp, spawned_at=iso_of(STAMPS[5]))
check(f"prune(keep={qa_wip.DEFAULT_KEEP}) removed 3", len(removed) == 3)
check("records_retained SATURATES at keep (the defect, still visible)",
      post["records_retained"] == qa_wip.DEFAULT_KEEP,
      f"records_retained={post['records_retained']} (true attempts: 6)")
check("attempt_number SURVIVES the prune and still reports 6",
      post["attempt_number"] == 6,
      f"attempt_number={post['attempt_number']} lost={post['records_pruned_known']}")
check("the loss ledger accounts for exactly what was destroyed",
      post["records_pruned_known"] == 3)

ledger = qa_wip.resolve_loss_ledger_path(SID, repo=tmp)
check("the loss ledger is dot-prefixed (invisible to the verdict_wip_* globs)",
      ledger.name.startswith("."))
check("...and list_wip_records does NOT pick it up",
      ledger not in qa_wip.list_wip_records(SID, repo=tmp))

# monotonic: a second prune that removes nothing must never lower the account
qa_wip.prune_wip_records(SID, repo=tmp, keep=qa_wip.DEFAULT_KEEP)
check("the loss account is MONOTONIC across a no-op prune",
      qa_wip.read_loss(SID, repo=tmp) == 3,
      f"lost={qa_wip.read_loss(SID, repo=tmp)}")
# and across a deeper prune it rises
qa_wip.prune_wip_records(SID, repo=tmp, keep=1)
check("...and RISES when more is destroyed",
      qa_wip.read_loss(SID, repo=tmp) == 5,
      f"lost={qa_wip.read_loss(SID, repo=tmp)}")
shutil.rmtree(tmp)

# ── C3b: the two halves COMBINED. Cycle-1 Q/A finding (mutant N6). ────────────
# C2 exercised the no_record_for_this_spawn branch with NO prune and NO ledger,
# so `lost_n` was 0 and the `lost_n +` term was DEAD -- deleting it changed
# nothing and the mutant survived both gates. C3 exercised the ledger only on the
# MATCHED path. The failure lives exactly where the two meet, so the fixture has
# to meet them: a pruned step WITH a loss account, reported by a spawn that has
# NOT yet written its record.
section("C3b -- loss account x no_record_for_this_spawn (the branch N6 survived in)")

tmp = pathlib.Path(tempfile.mkdtemp(prefix="c3b_"))
for s in STAMPS[:6]:
    mk(tmp, SID, s)
qa_wip.prune_wip_records(SID, repo=tmp, keep=qa_wip.DEFAULT_KEEP)
check("precondition: the loss account is NON-ZERO, so the add-back term is LIVE",
      qa_wip.read_loss(SID, repo=tmp) == 3,
      f"lost={qa_wip.read_loss(SID, repo=tmp)} -- if this were 0 the assertion below "
      f"could not fail and would be vacuous")
# A 7th spawn that has NOT written its write-first record yet.
nr = qa_wip.report(SID, repo=tmp, spawned_at=iso_of(STAMPS[6]))
check("...the branch under test is actually reached",
      nr["attempt_number_status"] == "no_record_for_this_spawn",
      f"status={nr['attempt_number_status']}")
check("prior_attempts counts PRUNED-AWAY records too, not just retained ones",
      nr["prior_attempts"] == 6,
      f"prior_attempts={nr['prior_attempts']} (retained={nr['records_retained']}, "
      f"lost={nr['records_pruned_known']}; without the add-back it would be 3)")
# ...and that this is the difference between escalating and not.
st_here = ab.BudgetState(step_id=SID)
for _ in range(nr["prior_attempts"] + 1):
    st_here.record(ab.Outcome.CONDITIONAL)
check("...and that difference is what makes F1b's ceiling reachable on this path",
      st_here.disposition() is ab.Disposition.ESCALATE,
      f"attempt {nr['prior_attempts'] + 1}/{st_here.max_attempts} -> "
      f"{st_here.disposition().value} (dropping the add-back gives 4/5 -> CONTINUE)")
shutil.rmtree(tmp)

# ── C3c: attempt_number_is_lower_bound. Cycle-1 Q/A finding (mutant N1). ──────
# The field shipped with ZERO assertions anywhere -- flipping True->False was
# invisible to every gate. A shipped output field with no guard has no guard.
section("C3c -- attempt_number_is_lower_bound actually discriminates")

# CYCLE-2 Q/A finding (its mutant Q-A): the first version of this section drove
# retained=2 and retained=5 only, so the BOUNDARY VALUE -- exactly DEFAULT_KEEP --
# was never exercised, and `>=` -> `>` survived. The boundary is not a detail here:
# `retained == DEFAULT_KEEP` with no ledger is EXACTLY the state a legacy prune
# leaves behind, i.e. the state the flag exists for. Every regime is now driven,
# and the boundary has its own named assertion.
tmp = pathlib.Path(tempfile.mkdtemp(prefix="c3c_"))
regimes = {}
for s in STAMPS[:qa_wip.DEFAULT_KEEP - 1]:            # BELOW the window
    mk(tmp, SID, s)
regimes["below"] = qa_wip.report(
    SID, repo=tmp, spawned_at=iso_of(STAMPS[qa_wip.DEFAULT_KEEP - 2]))
mk(tmp, SID, STAMPS[qa_wip.DEFAULT_KEEP - 1])         # EXACTLY at the window
regimes["at"] = qa_wip.report(
    SID, repo=tmp, spawned_at=iso_of(STAMPS[qa_wip.DEFAULT_KEEP - 1]))
for s in STAMPS[qa_wip.DEFAULT_KEEP:qa_wip.DEFAULT_KEEP + 2]:   # ABOVE
    mk(tmp, SID, s)
regimes["above"] = qa_wip.report(
    SID, repo=tmp, spawned_at=iso_of(STAMPS[qa_wip.DEFAULT_KEEP + 1]))
qa_wip.prune_wip_records(SID, repo=tmp, keep=qa_wip.DEFAULT_KEEP)   # writes a ledger
regimes["accounted"] = qa_wip.report(
    SID, repo=tmp, spawned_at=iso_of(STAMPS[qa_wip.DEFAULT_KEEP + 1]))

check("precondition: the three unaccounted regimes really are below / AT / above",
      (regimes["below"]["records_retained"] == qa_wip.DEFAULT_KEEP - 1
       and regimes["at"]["records_retained"] == qa_wip.DEFAULT_KEEP
       and regimes["above"]["records_retained"] > qa_wip.DEFAULT_KEEP),
      f"retained = {regimes['below']['records_retained']} / "
      f"{regimes['at']['records_retained']} / {regimes['above']['records_retained']}, "
      f"keep={qa_wip.DEFAULT_KEEP}")
check("below the retention window with no ledger -> NOT flagged a floor",
      regimes["below"]["attempt_number_is_lower_bound"] is False)
check("EXACTLY AT the window with no ledger -> FLAGGED (the boundary itself)",
      regimes["at"]["attempt_number_is_lower_bound"] is True,
      f"retained == keep == {qa_wip.DEFAULT_KEEP}; this is the state a legacy "
      f"prune leaves behind, so `>` instead of `>=` must NOT survive")
check("above the window with no loss account -> FLAGGED as a floor",
      regimes["above"]["attempt_number_is_lower_bound"] is True)
check("once the loss IS accounted -> no longer a floor",
      regimes["accounted"]["attempt_number_is_lower_bound"] is False,
      f"lost={regimes['accounted']['records_pruned_known']}")
check("the field discriminates (not all regimes agree)",
      len({r["attempt_number_is_lower_bound"] for r in regimes.values()}) == 2)
shutil.rmtree(tmp)


# ── C3d: the ledger-before-unlink CRASH SAFETY claim. Cycle-2 Q/A finding Q-E. ──
# qa_wip.prune_wip_records documents an explicit safety invariant: the loss is
# recorded BEFORE the unlink so a crash mid-prune OVER-counts (escalates early,
# safe) rather than UNDER-counts (suppresses escalation, unsafe). That was a
# documented claim with no assertion behind it -- and a documented invariant with
# no guard is a claim, not a property. Driven by making the unlink actually fail.
section("C3d -- crash mid-prune OVER-counts (the documented safe direction)")

tmp = pathlib.Path(tempfile.mkdtemp(prefix="c3d_"))
for s in STAMPS[:6]:
    mk(tmp, SID, s)
_real_unlink = pathlib.Path.unlink


def _exploding_unlink(self, *a, **kw):        # simulate a crash mid-prune
    raise OSError("simulated crash mid-prune")


pathlib.Path.unlink = _exploding_unlink
try:
    removed_crash = qa_wip.prune_wip_records(SID, repo=tmp, keep=qa_wip.DEFAULT_KEEP)
finally:
    pathlib.Path.unlink = _real_unlink

survivors = len(qa_wip.list_wip_records(SID, repo=tmp))
lost_after_crash = qa_wip.read_loss(SID, repo=tmp)
check("precondition: the unlink really did fail (nothing was deleted)",
      removed_crash == [] and survivors == 6,
      f"removed={len(removed_crash)} still_on_disk={survivors}")
check("the loss was ALREADY recorded when the crash hit -> OVER-count, not None",
      lost_after_crash == 3,
      f"read_loss={lost_after_crash} -- recording AFTER the unlink would give "
      f"None (or 0), which UNDER-counts and suppresses escalation")
crashed = qa_wip.report(SID, repo=tmp, spawned_at=iso_of(STAMPS[7]))
check("...so the derived count errs STRICTLY HIGH after a crash, never low",
      crashed["prior_attempts"] > 6,
      f"prior_attempts={crashed['prior_attempts']} -- must EXCEED the 6 real "
      f"attempts (6 survivors + 3 already-accounted). Accounting after the unlink "
      f"gives exactly 6, so `>= 6` would not discriminate and `> 6` does")
shutil.rmtree(tmp)

print("\n  ENUMERATION -- automatic callers of prune_wip_records in the live tree.")
CMD = (r"grep -rn 'prune_wip_records' --include='*.py' --include='*.js' "
       r"--include='*.sh' . | grep -v -e '/\.venv/' -e '/node_modules/' "
       r"-e '/\.git/' -e '^\./handoff/' -e '^\./docs/'")
print(f"  command: {CMD}")
out = subprocess.run(["bash", "-lc", CMD], cwd=REPO, capture_output=True,
                     text=True).stdout.strip().splitlines()
for ln in out:
    print(f"    {ln}")
# POPULATION RULE, stated: a hit is NON-production iff its file is on this
# EXPLICIT allowlist. A pattern like `verify_*` would let a future production
# caller hide behind a checker-shaped filename; an allowlist cannot.
NON_PRODUCTION = {
    "./scripts/qa/qa_wip.py",                    # its own definition
    "./scripts/qa/verify_wip_retention_86_36.py",  # 86.36's checker
    "./scripts/qa/mutation_matrix_86_36.py",     # 86.36's mutation matrix
    "./scripts/qa/mutate_counter_source_86_21.py",  # comments only
    "./scripts/qa/verify_counter_86_79.py",      # THIS checker
    "./scripts/qa/mutation_matrix_86_79.py",     # this step's mutation matrix
}
prod = [ln for ln in out if ln.split(":", 1)[0] not in NON_PRODUCTION]
check("prune_wip_records has NO production caller (defect is LATENT, not live)",
      prod == [], f"non-allowlisted hits: {prod}")
check("the enumeration is not vacuous -- it found the definition itself",
      any(ln.startswith("./scripts/qa/qa_wip.py:") for ln in out),
      f"{len(out)} hits total")


# ══════════════════════════════════════════════════════ CRITERION 4
section("C4 -- the DOC and the CODE agree (in-module), and the residual is LOUD")

# The comment block immediately above DEFAULT_KEEP, grep-derived.
m = re.search(r"((?:^#:.*\n)+)DEFAULT_KEEP = (\d+)", src_text, re.M)
comment, keep_val = (m.group(1), int(m.group(2))) if m else ("", -1)
check("DEFAULT_KEEP and its comment block are found", bool(m),
      f"DEFAULT_KEEP = {keep_val}")
check("the comment states the UNIT (TOTAL / INCLUSIVE), not just a name",
      "INCLUSIVE" in comment and "TOTAL" in comment)
check("the old off-by-one wording is GONE from the normative sentence",
      "#: Current record + this many prior attempts" not in src_text)

# Behavioural half: what the comment claims must be what prune delivers.
tmp = pathlib.Path(tempfile.mkdtemp(prefix="c4_"))
for s in STAMPS[:6]:
    mk(tmp, SID, s)
qa_wip.prune_wip_records(SID, repo=tmp, keep=qa_wip.DEFAULT_KEEP)
retained = len(qa_wip.list_wip_records(SID, repo=tmp))
check("measured retention == keep TOTAL, exactly as the comment now claims",
      retained == qa_wip.DEFAULT_KEEP,
      f"retained={retained} keep={qa_wip.DEFAULT_KEEP}")
shutil.rmtree(tmp)

# The residual divergence (qa.md) must be LOUD, which is what criterion 4 forbids
# leaving silent. Two independent channels, neither of which is a qa.md edit.
patch = REPO / "handoff" / "current" / "qa_md_patch_86.79.md"
check("the un-applied qa.md correction is written out for the operator",
      patch.is_file(), str(patch.relative_to(REPO)))
check("...and it names the exact line it would change",
      patch.is_file() and "records_retained" in patch.read_text(encoding="utf-8"))


# ══════════════════════════════════════════════════════ CRITERION 5
section("C5 -- both escalation bounds still fire against the CORRECTED number")

# --- F1b: 5-attempt cumulative budget, fed from attempt_number THROUGH a prune
tmp = pathlib.Path(tempfile.mkdtemp(prefix="c5_"))
for s in STAMPS[:6]:
    mk(tmp, SID, s)
qa_wip.prune_wip_records(SID, repo=tmp, keep=qa_wip.DEFAULT_KEEP)
r = qa_wip.report(SID, repo=tmp, spawned_at=iso_of(STAMPS[5]))

st_old = ab.BudgetState(step_id=SID)
for _ in range(r["records_retained"]):
    st_old.record(ab.Outcome.CONDITIONAL)
st_new = ab.BudgetState(step_id=SID)
for _ in range(r["attempt_number"]):
    st_new.record(ab.Outcome.CONDITIONAL)

check("OLD number does NOT reach F1b's ceiling after a prune (the bug)",
      st_old.disposition() is ab.Disposition.CONTINUE,
      f"attempts_used={st_old.attempts_used}/{st_old.max_attempts}")
check("NEW number DOES escalate after a prune (the fix)",
      st_new.disposition() is ab.Disposition.ESCALATE,
      f"attempts_used={st_new.attempts_used}/{st_new.max_attempts}")
check("the escalation summary is written and refuses to read as a pass",
      "THIS IS NOT A PASS AND NOT A FAIL" in st_new.escalation_summary())
shutil.rmtree(tmp)

# --- the 3rd-consecutive-CONDITIONAL boundary (verdict-keyed; must be UNCHANGED)
led = pathlib.Path(tempfile.mkdtemp(prefix="c5b_")) / "ledger.jsonl"
led.write_text("".join(
    json.dumps({"step_id": SID, "verdict": v}) + "\n" for v in ("CONDITIONAL",)), encoding="utf-8")
h1 = vh.read_ledger(SID, path=led)
led.write_text("".join(
    json.dumps({"step_id": SID, "verdict": v}) + "\n"
    for v in ("CONDITIONAL", "CONDITIONAL")), encoding="utf-8")
h2 = vh.read_ledger(SID, path=led)
led.write_text("".join(
    json.dumps({"step_id": SID, "verdict": v}) + "\n"
    for v in ("CONDITIONAL", "CONDITIONAL", "PASS")), encoding="utf-8")
h3 = vh.read_ledger(SID, path=led)

check("1 CONDITIONAL -> auto-FAIL NOT armed", h1.would_auto_fail is False,
      f"consecutive={h1.consecutive_conditionals}")
check("2 consecutive CONDITIONALs -> auto-FAIL ARMED (the boundary fires)",
      h2.would_auto_fail is True, f"consecutive={h2.consecutive_conditionals}")
check("a PASS RESETS the run", h3.would_auto_fail is False,
      f"consecutive={h3.consecutive_conditionals}")
check("a missing ledger returns None, NOT 0",
      vh.read_ledger(SID, path=led.parent / "nope.jsonl")
        .consecutive_conditionals is None)
shutil.rmtree(led.parent)


# ══════════════════════════════════════════════════════ CRITERION 6
section("C6 -- verdict semantics untouched; uncomputable == None, never 0")

tmp = pathlib.Path(tempfile.mkdtemp(prefix="c6_"))
no_sink = qa_wip.report(SID, repo=tmp, spawned_at=iso_of(STAMPS[0]))
(tmp / qa_wip.MEMORY_DIR / qa_wip.WIP_SUBDIR).mkdir(parents=True)
empty = qa_wip.report(SID, repo=tmp, spawned_at=iso_of(STAMPS[0]))
mk(tmp, SID, STAMPS[0])
no_id = qa_wip.report(SID, repo=tmp)                     # no spawned_at

check("missing sink -> attempt_number is None, not 0",
      no_sink["attempt_number"] is None
      and no_sink["attempt_number_status"] == "source_missing")
check("empty sink -> attempt_number is None, not 0",
      empty["attempt_number"] is None)
check("no spawned_at -> attempt_number is None, not 0",
      no_id["attempt_number"] is None
      and no_id["attempt_number_status"] == "no_spawn_identity")
check("NO report() variant ever carries a `verdict` key",
      all("verdict" not in d for d in (no_sink, empty, no_id)))
check("every report() variant states is_verdict: false",
      all(d["is_verdict"] is False for d in (no_sink, empty, no_id)))
shutil.rmtree(tmp)

# exhaustion can never manufacture a PASS -- replayed, not asserted from the doc
st = ab.BudgetState(step_id=SID)
for _ in range(ab.DEFAULT_MAX_ATTEMPTS + 3):
    st.record(ab.Outcome.CONDITIONAL)
kinds = {st.close_kind(p, e) for p in (True, False) for e in (True, False)}
check("budget exhaustion cannot produce a PASS under ANY flag combination",
      kinds == {"ESCALATE"}, f"close_kind over all flags -> {sorted(kinds)}")


# ══════════════════════════════════════════════════════ verdict
section("RESULT")
failed = [r for r in _results if not r[1]]
ran = len(_results)
print(f"  checks run : {ran}   (cardinality floor {EXPECTED_CHECKS})")
print(f"  failed     : {len(failed)}")
for lbl, _, det in failed:
    print(f"    FAIL {lbl} -- {det}")
if ran < EXPECTED_CHECKS:
    print(f"  *** CARDINALITY FLOOR BREACHED: {ran} < {EXPECTED_CHECKS}. "
          f"A checker that covers nothing exits 0 and looks identical to success.")
ok = not failed and ran >= EXPECTED_CHECKS
print(f"\n  {'ALL CHECKS PASS' if ok else 'CHECKER RED'}")
sys.exit(0 if ok else 1)
