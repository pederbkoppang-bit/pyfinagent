#!/usr/bin/env python3
"""phase-86.21 criterion 6 -- mutate the COUNTER and prove each guard can fail.

The criterion asks for the source to be corrupted or emptied and the counter to
NOTICE. `verdict_history_86_21.py --self-test` does that. This harness closes the
other half: it mutates the COUNTER ITSELF and requires the self-test to go RED.
A self-test that cannot fail is not a guard.

SAFETY: in-memory only. The target is read, mutated in a temp copy, and its md5
is asserted unchanged across the whole run -- the house pattern from phase-36.17,
adopted because an earlier scratch harness once left a mutant in a live file.

    source .venv/bin/activate
    python scripts/qa/mutation_matrix_86_21.py
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import pathlib
import tempfile

TARGET = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "qa" / "verdict_history_86_21.py"

MUTANTS = [
    # ---- phase-86.21 cycle 4: the three survivors the cycle-3 Q/A found -----
    # All three lived in `_report`'s PRINTED OUTPUT, which had zero automated
    # coverage because the self-test discarded the stdout buffer. They are the
    # reason cases (vi-b)/(vi-c)/(vi-d) now keep it.
    ("S1", "the CLI prints a hard zero for every step forever (the silent zero returns, in the OUTPUT)",
     'print(f"consecutive     : {c}")',
     'print(f"consecutive     : 0")'),
    ("S2", "the two CAUSE explanations swap -- blindness attributed to a predicate mismatch and vice versa",
     "        if g == 0 and c > 0:",
     "        if not (g == 0 and c > 0):"),
    ("S3", "the whole DISAGREEMENT block disappears silently",
     "    if c is not None and g != c:",
     "    if False:"),
    ("M1", "unparseable/empty report 0 instead of None (the silent zero returns)",
     "        if self.status in (UNPARSEABLE, LEDGER_EMPTY, LEDGER_MISSING):\n            return None",
     "        if False:\n            return None"),
    ("M2", "reset becomes == 'PASS' (misses PASS_WITH_FINDINGS / PASS_AFTER_RETRY)",
     "            if v == CONDITIONAL:\n                n += 1\n            else:\n                break",
     "            if v != 'PASS':\n                n += 1\n            else:\n                break"),
    ("M3", "corrupt rows are ignored instead of counted (fail-open restored)",
     "        except json.JSONDecodeError:\n            bad += 1",
     "        except json.JSONDecodeError:\n            bad += 0"),
    # ---- cells added in cycle 2, each one a mutant the Q/A built that SURVIVED
    # ---- the cycle-1 matrix. They are here so they cannot survive again.
    ("M4", "arming threshold drops to one CONDITIONAL (one-sided guard, Q/A's Q1)",
     "        return c >= 2                     # a THIRD would be the auto-FAIL",
     "        return c >= 1                     # MUTANT M4"),
    ("M5", "a present-but-EMPTY ledger reports a silent zero again (Q/A's finding 1)",
     "    if path.stat().st_size == 0:",
     "    if False:"),
    ("M6", "step matching becomes a PREFIX match (86.2 would swallow 86.20/86.21)",
     "        if sid.strip() != step_id:",
     "        if not sid.strip().startswith(step_id):"),
    ("M7", "verdict tokens stop being case-normalised (Q/A's Q3)",
     '        verdicts.append(v.strip().upper())',
     '        verdicts.append(v.strip())'),
    # ---- cycle-3 cells: every one is a mutant the Q/A built that SURVIVED the
    # ---- cycle-2 matrix. The cluster is telling -- three of them live in the
    # ---- CLI/contrast half, which self_test() never touched until now.
    ("M8", "a row with NO step_id is silently skipped again (fail-OPEN under-count)",
     "        if sid is None or not isinstance(sid, str) or not sid.strip():\n            bad += 1\n            continue",
     "        if False:\n            bad += 1\n            continue"),
    ("M9", "prescribed_grep_count always returns 0 (Q/A's N1 -- contrast half unguarded)",
     '    if not path.exists():\n        return 0\n    pat = re.compile(',
     '    if True:\n        return 0\n    pat = re.compile('),
    ("M10", "_report always exits 0 (Q/A's N2 -- the fail-CLOSED signal goes dark)",
     '    return 0 if h.status in (OK, NO_ROWS_FOR_STEP) else 1  # empty/missing/corrupt -> 1',
     '    return 0  # MUTANT M10'),
    ("M11", "would_auto_fail returns False instead of None when unknowable (Q/A's N4)",
     "        c = self.consecutive_conditionals\n        if c is None:\n            return None",
     "        c = self.consecutive_conditionals\n        if c is None:\n            return False"),
]


def _load(src_text: str, tag: str):
    d = tempfile.mkdtemp()
    p = pathlib.Path(d) / f"vh_{tag}.py"
    p.write_text(src_text)
    spec = importlib.util.spec_from_file_location(f"vh_{tag}", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def verify_broken_scoring(text: str) -> bool:
    """phase-86.21 cycle 4 -- prove the BROKEN path actually fires.

    The cycle-3 Q/A showed this harness scoring guard-IRRELEVANT mutants as
    KILLED and then printing "every guard IN THIS MATRIX can fail". The fix
    separates load-time failure from self_test() failure -- but a fix that is
    never observed working is exactly what this step is about. So: inject a
    mutant that CANNOT COMPILE and require `_load` to raise, which is what the
    BROKEN branch keys on. Run as part of every matrix run, not on request.
    """
    broken_src = text.replace("def self_test() -> int:", "def self_test(((", 1)
    if broken_src == text:
        print("  [broken-scoring self-check] ANCHOR MISSING -- cannot verify")
        return False
    try:
        _load(broken_src, "xcheck")
    except Exception as exc:                                   # noqa: BLE001
        print(f"  [broken-scoring self-check] _load raised {type(exc).__name__}"
              " -> the BROKEN branch is reachable, guard-irrelevant mutants"
              " cannot be scored as kills")
        return True
    print("  [broken-scoring self-check] an UNCOMPILABLE mutant LOADED -- the"
          " BROKEN branch is unreachable and kills cannot be trusted")
    return False


def main() -> int:
    before = hashlib.md5(TARGET.read_bytes()).hexdigest()
    text = TARGET.read_text()
    print("phase-86.21 criterion 6 -- mutation matrix (in-memory; repo never written)")
    print(f"target : {TARGET.name}")
    print(f"md5    : {before}\n")

    ctl = _load(text, "ctl")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = ctl.self_test()
    print(f"[control] un-mutated self-test rc={rc} (0 = PASSED)")
    if rc != 0:
        print("CONTROL IS RED -- a killed mutant would prove nothing.")
        print(buf.getvalue())
        return 2

    if not verify_broken_scoring(text):
        print("REFUSING TO SCORE -- the BROKEN path could not be verified.")
        return 5

    killed, survived, broken = 0, [], []
    for mid, desc, old, new in MUTANTS:
        n = text.count(old)
        if n != 1:
            broken.append(mid)
            print(f"  BROKEN  | {mid}: {desc}\n            anchor matched {n} time(s)")
            continue
        # A mutant that makes the module RAISE is killed just as surely as one
        # that fails an assertion -- louder, in fact. Cycle 3 hit this: removing
        # the step_id guard leaves `sid` None and the next line raises
        # AttributeError. Letting that propagate aborted the whole matrix and
        # reported nothing about the remaining cells, which is a worse failure
        # than the mutant itself.
        # phase-86.21 cycle 4 -- A LOAD-TIME CRASH IS NOT A KILL, IT IS A BROKEN
        # CELL. The cycle-3 Q/A demonstrated the hole by monkeypatching in three
        # guard-IRRELEVANT mutants -- a syntax error, an unimportable module, a
        # broken indent -- and this harness scored all three "KILLED" and then
        # printed "every guard IN THIS MATRIX can fail". None of those mutants
        # ever reached a guard. A mutant that cannot execute licenses NOTHING,
        # and crediting it to the guard is the mis-attributed-kill shape (qa.md
        # 4c #11): the mutation died, but not by the assertion claimed.
        #
        # `_load()` compiles and execs the module, so it is separated out. Only
        # exceptions raised INSIDE self_test() are kills -- and those are real
        # ones, e.g. removing the step_id guard leaves `sid` None and the next
        # line raises AttributeError, which IS the guard's subject failing.
        try:
            mod = _load(text.replace(old, new, 1), mid)
        except Exception as exc:  # noqa: BLE001 -- did not reach any guard
            broken.append(mid)
            print(f"  BROKEN  | {mid}: {desc}")
            print(f"            mutant failed to LOAD ({type(exc).__name__}: {exc})")
            print("            -- it never reached a guard, so this cell scores NOTHING")
            continue
        try:
            b = io.StringIO()
            with contextlib.redirect_stdout(b):
                r = mod.self_test()
        except Exception as exc:  # noqa: BLE001 -- raised INSIDE the guard: a kill
            killed += 1
            print(f"  KILLED  | {mid}: {desc}\n            self_test() raised {type(exc).__name__}: {exc}")
            continue
        if r != 0:
            killed += 1
            print(f"  KILLED  | {mid}: {desc}\n            self-test rc={r}")
        else:
            survived.append(mid)
            print(f"  SURVIVED| {mid}: {desc}\n            self-test rc={r}")

    after = hashlib.md5(TARGET.read_bytes()).hexdigest()
    same = before == after
    print(f"\n[integrity] target md5 unchanged: {same}")
    if broken:
        print(f"MATRIX BROKEN -- anchors failed for: {', '.join(broken)}")
        return 3
    if survived:
        print(f"{killed} of {len(MUTANTS)} killed. SURVIVORS: {', '.join(survived)}")
        return 1
    if not same:
        print("TARGET CHANGED DURING THE RUN -- investigate.")
        return 4
    print(f"ALL {len(MUTANTS)} MUTANTS KILLED -- every guard IN THIS MATRIX can fail.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
