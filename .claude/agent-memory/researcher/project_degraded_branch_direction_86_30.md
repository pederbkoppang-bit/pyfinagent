---
name: degraded-branch-direction-86-30
description: phase-86.30 -- is_global answers ROUTABILITY not OWNERSHIP (6 of this Mac's 17 own addresses are is_global=True); the finding was already Q/A note N1 in a prior critique; psutil is in NO requirements file
metadata:
  type: project
---

Fail-safe direction for the degraded branch of `_is_this_machine`
(`scripts/qa/live_backend_origin.py:186`, `return not ip.is_global`).

**Why:** the branch's own comment says "Over-refusal, which is the safe
direction" while the expression under-refuses. Researched 2026-08-10.

**How to apply:** four transferable facts, each a class not an instance.

1. **`is_global` is a ROUTABILITY property, never an OWNERSHIP one.** CPython
   defines it as "globally reachable by iana-ipv{4,6}-special-registry". RFC 4291
   §2.8 requires a host to recognise "any additional Unicast ... addresses
   configured for interfaces" as identifying itself -- ordinarily global unicast.
   MEASURED on this Mac: **17 own addresses, 6 with `is_global == True`** (six
   `2001:...` GUAs on en1 = one stable SLAAC + accumulated RFC 8981 temporaries,
   which rotate on a 1-day preferred / 2-day valid clock). So `not is_global`
   inverts for exactly the addresses a modern host carries. Whenever a predicate
   asks "is this mine", enumeration is the ONLY answer -- RFC 7136: "no reliable
   deductions can be made" from an address's bits. Never substitute a
   registry-derived boolean; `is_global` was itself CVE'd (CVE-2024-4032, CVSS
   7.5, fixed 3.12.4) and changed again in 3.13.

2. **Check prior evaluator critiques' NOTE-level findings BEFORE researching.**
   This entire defect -- same measurement ("SIX globally-routable IPv6
   addresses"), same one-line remedy ("return True unconditionally") -- was
   already written down as note N1 at `evaluator_critique_86.27.md:69`, with the
   materiality bound (psutil IS installed; uvicorn binds IPv4-only, so no IPv6
   spelling reaches the book -> LATENT, not live). Q/A notes are a free research
   corpus; grep them first. See [[feedback_queue_discovered_defects_in_masterplan]].

3. **A repo can hold two degraded paths pointing opposite ways.** In this same
   file family, `conftest.py:96-110` degrades to port-only over-refusal and says
   so, and `targets_this_machine` (`live_backend_origin.py:243`) returns True
   when it cannot canonicalise. `_is_this_machine:186` is the lone dissenter.
   When auditing a degraded branch, enumerate the OTHER degraded branches in the
   same subsystem and compare directions -- disagreement is the finding.

4. **Undeclared transitive deps make a "theoretical" branch reachable.**
   `psutil` (7.2.2 installed) appears in **none** of the 6 requirements files.
   A fresh `pip install -r` venv hits the degraded path. Always grep every
   requirements file before calling a fallback unreachable.

Harness note: `scripts/qa/mutation_matrix_86_27.py` has cells **M1-M7** (a
caller's scope statement said M1-M3). It documents twice (`:54-61`, `:85-89`)
that a probe must DISCRIMINATE -- a mutant survives when the control answer and
the mutant's fail-safe answer coincide, which is the trap for any cell covering
a fix that returns a constant. See [[feedback_mutation_probe_must_discriminate]].
