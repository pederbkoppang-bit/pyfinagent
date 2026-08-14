# Research gate — step 86.67 RE-RUN — **PASSED**

**Run:** `wf_40b64505-346` | **Date:** 2026-08-14 ~09:15 CEST
**Brief:** `handoff/current/research_brief_86.67_rerun.md` (33,211 chars)
**2 agents, 0 errors, 0 empty returns, 182,569 subagent tokens, 506s**

> Verdict transcribed in the same turn it landed. The prior brief
> (`research_brief_86.67.md`, 40,839 bytes) is **preserved unmodified**.

---

## Why the first gate failed — NOT the reason recorded

The goal said *"gate `false` on a 38-vs-37 over-claim."* Measured before the re-run: the
brief claims `urls_collected: 38` and **38 distinct URLs are genuinely present**. The count
was fine. The actual blocker was the same structural defect as 86.59: **the envelope
carried no `sources_read_in_full` array**, so `enforceGate` could corroborate nothing.

## The recomputed result

| check | value |
|---|---|
| `sources_floor_ok` | **9 ≥ 5** |
| `urls_floor_ok` | **31 ≥ 10** |
| `urls_collected_corroborated` | **31 ≤ 31 distinct URLs in the brief** |
| `all_9_claimed_sources_present_in_brief` | 0 missing |
| `brief_status_in_brief` | `COMPLETE` |
| `self_report_disagreed` | **false** |
| `rail_dropped` | `null` |

---

## The finding that ranks the remediation

**The three options are layers, not alternatives.**

**1. Redaction-at-write — FIRST, and it names our exact line.**
arXiv:2604.03070v1 (n=17,022 agent skills, κ=0.88) measures **73.5% of agent credential
leaks as stdout/log capture** and names stripping stdout before persistence the
*"highest-impact remediation point."*

**Verified by me** — `scripts/away_ops/run_away_session.sh:170`:
```
< "$PROMPT_FILE" > "$OUT_JSON" 2>> "$SLOG"
```
Raw agent stdout is redirected **straight into a file in the tree**, with
`--output-format json`. That is the leak mechanism, confirmed at the line the gate cited.

**2. Commit-boundary scanning — a hook exists, but it is NOT a secret scanner.**
`.git/hooks/pre-commit` exists (49 lines, executable), and `auto-commit-and-push.sh` passes
**no `--no-verify`** (measured: 0 occurrences), so it *does* fire on the publish path.

**But it would not have caught this leak.** Measured: **0** secret-ish patterns
(`sk-ant|secret|token|credential|api_key`) anywhere in it. Its three guards are:

| guard | pattern | matches `handoff/away_ops/*.json`? |
|---|---|---|
| `.bak-` files | `^\.claude/.*\.bak-` | **no** |
| stale model IDs | `^(backend\|scripts)/.*\.py$` | **no** |
| staged `.env` | `(^\|/)\.env$` | **no** |

**3. `.gitignore` — confirmed incapable, from the primary source.** git-scm: an ignore rule
does not apply to **tracked** files, and all five leakers are tracked. This closes the
option rather than deferring it.

**Ordering: revoke-then-scrub** (GitHub, OWASP), mechanistically because scrubbed
credentials survive in forks and clones — **>64% of 2022-leaked secrets were still valid in
Jan 2026**. That is the quantitative backing for §0 of the goal.

---

## Two corrections to the gate's own summary, verified rather than accepted

1. **It conflated two files in one sentence.** *"a rejection is SILENT (:380 logs and exit
   0)"* — the pre-commit hook is **49 lines**, so `:380` cannot be in it. That line is
   `auto-commit-and-push.sh:380`, the `git commit` failure path. Both facts are real; the
   citation attaches to the wrong file.
2. **The `set -e` fail-closed risk is HYPOTHETICAL, not current.** The gate warns *"a grep
   without `|| true` blocks every clean commit."* Measured: **all four** grep command
   substitutions (`:10`, `:19`, `:21`, `:35`) are `|| true`-guarded — **0 unguarded**. So it
   is a real hazard for a **future edit**, not a live defect. Worth keeping as a warning
   attached to any change to that hook.

---

## What this does NOT license

- **No remediation was performed.** No rotation, no history rewrite, no `.gitignore` edit,
  no hook change. All operator-gated under ask 06-2 / #20.
- **The gate answers "what should be done", not "it is done."**
- `coverage.dry` is `false` (2 rounds, 0 dry) — informational, since the step is not
  audit-class.
- **No Q/A has graded this**, and the step is not flipped.
