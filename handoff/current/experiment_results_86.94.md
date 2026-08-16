# experiment_results — phase-86.94

## What changed

| file | change |
|---|---|
| `scripts/qa/verify_no_sliding_windows_86_94.py` | **NEW.** The regression guard: known-member recall, source-derived enumeration, criterion-4 disclosure check, mutation cells. |
| `scripts/qa/replay_changelog_rule_86_68.py` | `CORPUS_SINCE` → `2026-08-11T00:00:00Z`. **One character.** Figures unchanged; the window is now TZ-invariant. |
| `scripts/qa/verify_changelog_flip_86_91.py` | Comment only — discloses that its window assertion is a *substring* test and therefore a pin-presence check, not a TZ check. |
| `handoff/current/experiment_results_86.91.md`, `live_check_86.91.md` | The "reproducible figures" claim qualified **in place** (criterion 5). |

Full measured evidence: `live_check_86.94.md` §A–G.

---

## The finding

**Two defects, one class.** A window is reproducible only if it names an
*instant*. Two ways to fail that, and phase-86.91 fixed one and believed it had
fixed both:

1. **A bare date is not midnight.** `git rev-parse --since=2026-08-11` resolved
   to `2026-08-11 23:09:03 CEST` — today's clock time carried onto the target
   date. `--since=today` likewise resolves to *now*, not midnight (0 commits vs
   64), contradicting `git-log(1)`.
2. **A pinned timestamp is still TZ-local.** `2026-08-11T00:00:00` with *both
   ends pinned* measured **707** under Oslo/UTC/New York and **787** under
   Seoul. An 80-commit spread decided by `$TZ`.

Defect 2 is the one that matters for the harness, because 86.91's remediation
*was* defect 2 and it looked complete. The fix is one character (`Z`), the
published figures are unchanged (707 / 251 / 9 / 11, exit gate green), and the
corpus is now regenerable off this laptop.

**The drift is intermittently invisible, which is what makes it dangerous.** The
obvious boundary date, `2026-08-11`, would have shown *no* change tonight: its
last commit is 22:36:46, already behind the 22:50 cutoff, so its drift window is
exhausted. Two runs an hour apart would have agreed — a true result that reads
like a refutation. The date had to be chosen by measuring where commits actually
sit relative to the sliding band.

---

## The guard

`scripts/qa/verify_no_sliding_windows_86_94.py`, 24 assertions, exit 0.

- **Known-member recall is a hard gate.** The rule must find the pre-86.91 form
  of the replay, recovered from git at `06c3265f`, and classify it SLIDING. If
  the blob becomes unreachable the section **fails** rather than skipping.
- **The rule is written down in source and is wider than "bare date"** — a
  bare-date-only rule would have declared 86.91's TZ-naive pin clean.
- **It is an allowlist, not a ban.** One member is legitimately relative:
  `backend/slack_bot/scheduler.py` builds the Slack "shipped today" digest, and a
  report about today must move with today. A blanket prohibition would break
  correct code and then be switched off.
- **Criterion 4 is enforced as disclosure, not absence** (see below).

### Two defects my own rule had first, both of this step's classes

1. **It matched its own documentation.** The replay *documents* this defect in a
   comment quoting `` `--since=2026-08-11` ``; the scanner reported that prose as
   two SLIDING sites. Comments are now stripped, with a control in **both**
   directions — a stripper that quietly does nothing looks identical to a
   correct one.
2. **It failed OPEN on indirection.** `--since={CORPUS_SINCE}` was classified
   REPRODUCIBLE because the value is "decided at the call site". That is exactly
   how the TZ-naive constant stayed invisible. The resolver now reads the literal
   through one level of indirection and **fails closed** when it cannot. This is
   the leg that found the real defect — without it the guard would have shipped
   green over the very thing it was written for.

---

## Criterion 4 — the judgement, stated

| member | class | judgement (measured) |
|---|---|---|
| `scheduler.py:503` `midnight` | LEGITIMATELY RELATIVE | Name appears in 37 files; every hit descriptive (em-dash cleanup, an APScheduler job description, a different scheduler at `:761-795`). No count from this window is quoted. |
| `frontend_route_inventory.py:70,73` `30.days` | SLIDING, left | Mentioned in **0** files outside this step's artifacts. It *does* print per-route figures; they are never quoted as evidence. |
| `verify_decision_log_86_97.py:360` `{first_stamp}` | runtime-derived, allowed | Figures **are** quoted — always with the clock time they were taken at, and the checker asserts a *relationship*, not a number. |
| `replay_changelog_rule_86_68.py:114` `{CORPUS_SINCE}` | was SLIDING → **FIXED** | The TZ-naive pin. |

**The check enforces disclosure rather than absence, and that was a correction.**
My first version asserted the script name was absent from the quote corpus. It
immediately falsified two of my own allowlist claims — correctly as to the proxy,
misleadingly as to the question, because every hit was descriptive prose rather
than a quoted count. Criterion 4 asks for a judgement to be *stated*, so the
check now surfaces mention sites for audit and requires the entry to state one.
This step's own artifacts are excluded from the count, and the exclusion is
stated: they necessarily name every member, which would guarantee a hit for each
and make the check vacuous.

---

## Criterion 5 — corrected in place, not annotated

The figure found TZ-dependent was the replay's corpus and the
"reproducible figures are 707 / 251 / 9 / 11" claim built on it. Both 86.91
artifacts asserting that claim now carry the bound **in the same sentence**, and
the enumeration was driven by the *claim* (the window string and the figures),
not by my own phrasing — the 86.97 FAIL earlier tonight was caused by sweeping
for my own wording, and I was not going to repeat it in the step whose criterion
3 is a recall test.

`verify_changelog_flip_86_91.py:440` asserts the window by **substring**, so it
matched the `Z` form and stayed green through the fix — but it would equally stay
green if the `Z` were removed again. That bound is now disclosed in its own
comment, and the TZ property is asserted by the new checker, which resolves the
constant and classifies its *value*.

---

## Verification

```
$ bash -c 'source .venv/bin/activate && python scripts/qa/verify_changelog_flip_86_91.py > /dev/null && echo green'
green
```

**Disclosed:** the immutable command runs the *86.91* checker, which is green
today and would have stayed green through every defect this step is about. It
cannot fail on the class. The real evidence is in `live_check_86.94.md`.

```
verify_no_sliding_windows_86_94.py   ALL GREEN: 24 passed, 0 failed   (exit 0)
verify_changelog_flip_86_91.py       ALL GREEN: 42 passed, 0 failed
verify_workflow_args_boundary.mjs    ALL GREEN: 96 passed, 0 failed
ruff (default ruleset, new file)     All checks passed!
```

---

## Scope honesty — what I did NOT do

- Did **not** ban relative windows. One member is correctly relative; a ban would
  break it and would then be disabled.
- Did **not** change `frontend_route_inventory.py`. Classified SLIDING, judged
  acceptable, and the judgement is recorded with the measurement behind it.
- Did **not** re-run or amend 86.68's or 86.91's closed criteria. 86.91's
  *claim* was qualified; its criteria were not touched.
- The resolver sees through **one** level of indirection (a module-level string
  literal in the same file). A value assembled at runtime, or imported from
  another module, resolves to nothing and is failed **closed** — correct, but it
  means such a site needs an allowlist entry rather than being proven safe. The
  `{first_stamp}` entry is exactly that case and says so.
- The criterion-4 mention check uses **name presence** as its proxy. It cannot
  tell a quoted count from a descriptive mention; that is why it enforces a
  stated judgement instead of asserting absence, and why the mention sites are
  printed for an auditor rather than swallowed.
