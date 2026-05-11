---
step: phase-23.6.0
title: Ship dotenv-syntax validator + document operator-fix for backend/.env leading-space bug
cycle_date: 2026-05-10
harness_required: true
verification: 'python3 tests/verify_phase_23_6_0.py'
research_brief: handoff/current/phase-23.6.0-research-brief.md
---

# Contract — phase-23.6.0

## Hypothesis

A pure-Python stdlib-only dotenv syntax validator + a refined
idempotent `sed` sequence + a pre-commit hook block prevent
recurrence of the `backend/.env` leading-space bug class that has
caused autoresearch (and previously ablation) to fail nightly
since 2026-04-24.

**Key constraint:** `backend/.env` is sandbox-blocked from this
session. The deliverable is **defensive tooling** that the
operator runs to (a) detect leading-space lines BEFORE editing,
(b) apply the fix, (c) verify the fix landed, (d) prevent
recurrence via pre-commit hook. Main does NOT attempt to read or
edit `backend/.env` from this session.

## Research-gate summary

`researcher` agent `ab35b2da5f6a31d8e` ran tier=moderate and
returned `gate_passed: true` with:
- 8 external sources fetched in full (≥5 floor)
- 10 snippet-only + 8 read-in-full = 18 URLs (≥10 floor)
- Recency scan 2024-2026 performed
- Three-query discipline followed
- 8 internal files inspected

Brief: `handoff/current/phase-23.6.0-research-brief.md`.

**Researcher's four answers:**

1. **Implementation:** pure Python regex, stdlib only. `python-dotenv`'s
   `dotenv_values()` SILENTLY STRIPS the leading space — returns
   `KEY=value` for `KEY= value`, so it can't catch the exact bug
   that crashes bash. Rust dotenv-linter requires global install;
   wemake dotenv-linter has no library API. ~40 lines of regex +
   pathlib does the job zero-dep.

2. **Rules:** TWO CRITICAL rules suffice to prevent the exit-127/exit-1
   recurrence:
   - Leading space after `=` (`KEY= value`)
   - Leading space before key (` KEY=value`)
   Plus 3 WARNING/INFO rules (trailing whitespace, inline comment
   on unquoted value, missing trailing newline) for hygiene.

3. **Operator-fix sed sequence (idempotent global pattern):**
   ```bash
   # Pre-fix scan
   grep -n '^\([A-Z_][A-Z0-9_]*\)= ' backend/.env

   # Idempotent fix (BSD sed; no-op on clean lines)
   sed -i '' 's/^\([A-Z_][A-Z0-9_]*\)=  *\([^ ]\)/\1=\2/' backend/.env

   # Verify
   grep -n '^\([A-Z_][A-Z0-9_]*\)= ' backend/.env && echo "STILL BROKEN" || echo "CLEAN"

   # Restart autoresearch
   launchctl bootout gui/$(id -u)/com.pyfinagent.autoresearch 2>/dev/null
   launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.pyfinagent.autoresearch.plist
   launchctl kickstart gui/$(id -u)/com.pyfinagent.autoresearch
   ```

4. **Where to put it:**
   - `scripts/validators/check_dotenv_syntax.py` — operator-runnable
   - `tests/services/test_dotenv_syntax.py` — pytest against synthetic
     fixtures (does NOT open the real `backend/.env` — sandbox-blocked
     and would make tests non-portable)
   - Extend `.git/hooks/pre-commit` (already exists at 32 lines) with
     a 5-line block triggering only when `backend/.env` is staged

## Immutable success criteria (verbatim — DO NOT EDIT)

```
python3 tests/verify_phase_23_6_0.py
```

The verifier exits 0 only when:

1. `scripts/validators/check_dotenv_syntax.py` exists, is executable,
   and runs the 5 rules (2 CRITICAL + 3 hygiene) against a synthetic
   buggy `.env` fixture and exits non-zero with the offending line
   number; runs against a synthetic clean fixture and exits 0.
2. `tests/services/test_dotenv_syntax.py` exists and the suite passes
   (covers each rule + idempotency).
3. `handoff/runbooks/dotenv-leading-space-fix.md` exists and contains
   the verbatim sed sequence + verification command.
4. `.git/hooks/pre-commit` (or equivalent) contains a block invoking
   the validator on staged `.env` files (or, alternatively,
   `.pre-commit-config.yaml` exists with the validator entry).

## Plan steps

1. (DONE — RESEARCH) `gate_passed: true`.
2. (DONE — PLAN) This contract.
3. **GENERATE phase:**
   a. Create `scripts/validators/check_dotenv_syntax.py` — pure
      Python, stdlib only, takes path arg, runs 5 regex rules.
      Exit 0 clean / 1 dirty.
   b. Create `tests/services/test_dotenv_syntax.py` — pytest with
      synthetic fixtures (in-memory `tempfile`s) for each rule +
      idempotency check.
   c. Create `handoff/runbooks/dotenv-leading-space-fix.md` —
      operator runbook with the exact sed sequence,
      pre/post-fix verification commands, and restart steps.
   d. Extend `.git/hooks/pre-commit` to call the validator on
      staged `.env` files. Idempotent: skip if no .env staged.
   e. Add `tests/verify_phase_23_6_0.py` — 4-check verifier.
4. **EVALUATE phase:** spawn fresh `qa` agent.
5. **LOG phase:** append `harness_log.md` AFTER Q/A. Flip status.

## Anti-patterns guarded

1. **Attempting to read backend/.env** from this session —
   sandbox-blocked; would crash. Documented as operator-only.
2. **Using `python-dotenv`'s `dotenv_values()`** as the
   validator — silently strips leading space; can't catch the
   bug. Researcher confirmed empirically.
3. **Adding a heavy dep** (Rust dotenv-linter) — local-only Mac
   deployment; pure Python suffices.
4. **Self-evaluation by Main** — Q/A is mandatory.

## Out of scope

- Actually editing `backend/.env` (operator-action; this step
  ships the *tooling* the operator runs).
- Migrating to a secret manager.
- Refactoring the autoresearch python entrypoint (the residual
  exit-1 may live there; out of scope here).

## Backwards compatibility

- Pure additive: new files only.
- `.git/hooks/pre-commit` extended (not replaced); skip-if-no-env
  guard preserves existing behavior on every other commit.
- No `requirements.txt` change.
- No backend code change.

## Risk

- **Pre-commit hook may slow commits** — mitigated by skip-if-no-env
  guard.
- **Validator regex misses an edge case** — extensible: more rules
  can be added later. The 2 CRITICAL rules are the load-bearing
  ones; the 3 hygiene rules are defense-in-depth.
- **Operator never runs the validator** — that's an operational
  problem, not a tooling problem. The runbook documents both
  the manual sed sequence AND the validator command.

## References

- Research brief: `handoff/current/phase-23.6.0-research-brief.md`.
- Phase-23.3.5 prior audit:
  `handoff/archive/phase-23.3.5/phase-23.3.5-audit-findings.md`.
- Phase-23.5.19 archive (current `failed` state):
  `handoff/archive/phase-23.5.19/`.
- `scripts/autoresearch/run_nightly.sh` (the consumer of `.env`).
