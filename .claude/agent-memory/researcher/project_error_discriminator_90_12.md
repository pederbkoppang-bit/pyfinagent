---
name: error-discriminator-90-12
description: Step 90.12 research -- non-viable-mutant exclusion is settled prior art in 3 tool ecosystems; exception TYPE is an INCOMPLETE oracle (TypeError hole); the fail-open handler already puts the type on the wire; caplog is inapplicable
metadata:
  type: project
---

Step 90.12 (2026-08-21): "distinguish 'the mutant could not run' from 'the mutant ran and misbehaved'
when a fail-open handler swallows the failure." Research brief:
`handoff/current/research_brief_90.12.md` (8 sources read in full, 26 URLs).

**The doctrine is NOT a local invention -- three independent tool ecosystems converge.** Stryker puts
Runtime error / Compile error / Ignored / Pending in an **excluded** bucket, score `detected / valid`;
PIT ships **NON_VIABLE** ("could not be loaded by the JVM as the bytecode was in some way invalid") and
RUN_ERROR as statuses distinct from KILLED/SURVIVED; cosmic-ray names it **"incompetent"**. So 90.1
criterion 5 clause 3 restates settled practice. Useful when a Q/A asks "is this bar real?"

**Why:** the step had already relocated the same seam FOUR times (parse -> import -> run ->
runs-but-swallowed) and each cycle risked being argued as a local judgment call rather than a
published requirement.

**How to apply:**

1. **The published risk is the ORACLE's PRECISION, not the doctrine.** Best equivalent-mutant detector
   is 94.33% precision (arXiv:2408.01760, ISSTA 2024); TCE baselines F1 39-51%; Google validated
   arid-node suppression on **100 labelled nodes, 99 correct**. The field's own discipline is: measure
   the false-exclusion rate on a labelled sample and report it. An over-eager ERROR probe deleting a
   legitimate cell already happened here once (90.1 cycle 4, M14).
2. **The failure mode is SYMMETRIC and has a wild counterexample in the OTHER direction.** cosmic-ray
   issue #310: a non-viable mutant raising `TypeError` at pytest COLLECTION was scored **SURVIVED**.
   pyfinagent's is the mirror (scored KILLED). Same root cause: **the crash lands outside the
   observation window, so the mutant inherits the runner's default.** A fail-open handler IS a
   window-collapsing device -- it converts "never ran" into "ran and returned 0".
3. **Exception TYPE is sound but INCOMPLETE -- two traps.** (a) `TypeError` is a DOMAIN error by the
   Python taxonomy yet cosmic-ray #310's non-viable mutant died with one, so type alone cannot decide
   viability; it is safe ONLY as the last rung of a ladder whose earlier rungs (`ast.parse`, an import
   probe) already caught non-viability structurally. (b) **Subclass names do not inherit in a string
   match**: `UnboundLocalError` subclasses `NameError`, `ModuleNotFoundError` subclasses `ImportError`.
   `UNRESOLVABLE_ERRORS` lists ModuleNotFoundError but NOT UnboundLocalError.
4. **No production change is needed to read the type.** `attempt_gate.py`'s fail-open handler already
   interpolates `type(exc).__name__` into its one-liner -- the type is on the wire today. That matters
   because the handler is CORRECT and must stay (only exit 2 blocks a hook), and because backend
   restarts are batched to session end.
5. **`caplog` is INAPPLICABLE here.** The handler uses `print(..., file=sys.stderr)`, not the `logging`
   module; the channel is raw stderr from `subprocess.run(capture_output=True)`. Do not let pytest
   logging idiom into a contract for this seam.
6. **Easy misread to avoid:** Google's arid nodes SUPPRESS mutants placed ON logging calls (an
   unproductive-mutant argument). That is NOT "a log line is not a signal". Reading it the wrong way
   points the design 180 degrees off.
7. **GAP, stated not padded:** no authoritative source treats testing a *deliberately fail-open
   handler* as a named methodology problem. The design must be argued from mechanism, not cited.

**Live-state caveats (verify before reusing):** the draft fix was ALREADY on disk at research time --
`mutation_matrix_90_1.py` (`_drive_traceback` renamed `_drive_unresolvable`, swallowed-exception branch
added) and `verify_error_discriminator_90_12.py` both mtime 2026-08-21. And the masterplan
`audit_basis` cites `mutation_matrix_90_1.py:341` / `:337`, both **already stale** (the draft moved them
to `:353` / `:349`) -- re-derive line numbers, per [[feedback_url_count_must_be_re_derived]]'s sibling
discipline.

Related: [[project_phase90_accounting_and_the_relocating_seam]], [[project_phase90_research_findings]].
