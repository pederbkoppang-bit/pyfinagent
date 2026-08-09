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
    ("M1", "unparseable/empty report 0 instead of None (the silent zero returns)",
     "        if self.status in (UNPARSEABLE, LEDGER_EMPTY):\n            return None",
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
     '        if str(row.get("step_id", "")) != step_id:',
     '        if not str(row.get("step_id", "")).startswith(step_id):'),
    ("M7", "verdict tokens stop being case-normalised (Q/A's Q3)",
     '        verdicts.append(v.strip().upper())',
     '        verdicts.append(v.strip())'),
]


def _load(src_text: str, tag: str):
    d = tempfile.mkdtemp()
    p = pathlib.Path(d) / f"vh_{tag}.py"
    p.write_text(src_text)
    spec = importlib.util.spec_from_file_location(f"vh_{tag}", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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

    killed, survived, broken = 0, [], []
    for mid, desc, old, new in MUTANTS:
        n = text.count(old)
        if n != 1:
            broken.append(mid)
            print(f"  BROKEN  | {mid}: {desc}\n            anchor matched {n} time(s)")
            continue
        mod = _load(text.replace(old, new, 1), mid)
        b = io.StringIO()
        with contextlib.redirect_stdout(b):
            r = mod.self_test()
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
