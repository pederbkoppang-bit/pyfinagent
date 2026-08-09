# Goal — UNATTENDED OVERNIGHT masterplan drain
### Spans 2026-08-09 evening → 2026-08-10 morning. Cycles continue at 195.

Every number here was **measured at 20:16 CEST / 18:16Z on 2026-08-09**, not
recalled. **Re-derive anything you rely on.**

---

## 0. STANDING AUTHORIZATIONS (operator, 2026-08-09) — new, and they matter

1. **Run unattended all night.** Do not stop to ask permission for ordinary work.
   Queue every operator decision as a numbered ask and keep going.
2. **Backend restarts are AUTHORIZED without asking** — this supersedes the
   batch-to-session-end rule for this session. Discipline in §5 is still binding.
3. **Fable 5 is AUTHORIZED for genuinely complex tasks.** Budget ceiling and the
   places NOT to use it are in §6. Read it before switching.

Everything NOT on this list is unchanged: paper trading only, no flag
promotions, no `backend/.env` writes, `historical_macro` untouched.

## 1. Startup

`git checkout main && git pull origin main`. Confirm `.claude/settings.json` has
`defaultMode: bypassPermissions`. Run **`ListAgents`** — see §8, a scheduled
Claude session fires at 22:00 and again at 07:30 and you may not be alone.

**BINDING, read before acting:** this file, `CLAUDE.md`,
`.claude/rules/research-gate.md`, auto-memory `MEMORY.md`,
`handoff/current/day_report_2026-08-09b.md`, and
`handoff/current/operator_ask_2026-08-07.md`.

## 2. Measured state

| | measured 2026-08-09 20:16 CEST |
|---|---|
| backend | pid **6644**, started 18:56:00. The 36.17 fix **IS in force** |
| kill switch | `paused: false`, `armed: true`, `sod_date: 2026-08-09`, NAV 23833.94, peak 24666.57, trailing DD 3.3755% / 10% |
| last cycle | `ae2284ba` **completed** 15:04:32Z, 5,945s, **0 trades**, no errors, rail alive, meta-scorer NOT degraded |
| cycle lock | `state: released`, pid 17402 **dead**, lifetime ~1s → a **pytest run**, not a cycle (§7) |
| masterplan | **382 pending** (19 P0, 87 P1, 167 P2, 63 P3, 8 P4) |
| git | pushed clean, 0 unpushed |

**`sod_date` will roll to 2026-08-10 at the next cycle's Step 0.** That is
correct and self-clearing. **Do not "fix" it.**

## 3. The work, in order

### 3.1 FIRST — close 36.17 (it is 95% done and its fix is already live)

Five Q/A cycles: CONDITIONAL, FAIL, FAIL, CONDITIONAL, CONDITIONAL. The
**production code has been correct and byte-identical since cycle 2**
(md5 `58bbf24bde4c5161ac05f26f70fb264e`, confirmed by four separate Q/A passes)
and is in force. Every remaining finding was about **evidence quality**.

**Do this before spawning cycle 6, or you will repeat the loop:**

- **SLIM THE ARTIFACTS.** `experiment_results_36.17.md` has grown four
  meta-sections (§11–§14) narrating the Q/A process. Each one added claims,
  anchors and numbers — i.e. new attack surface. **Measured proof it compounds:**
  a correct "3 of 44 anchors" became wrong ("3 of 48") purely because the
  sentence reporting it added four more. Move the cycle-by-cycle narrative to
  `evaluator_critique_36.17.md` (already verbatim) and the harness log. The
  evidence file should state what was built and carry its verbatim proof.
- Then **one** Q/A on a quiet tree. Say explicitly in the spawn prompt that
  evidence HAS changed (it has: 11/11 mutants, tool-reported recall, rewritten
  in-force proof, ABRIDGED labels).
- **THE 3rd-CONDITIONAL RULE IS ARMED** — cycles 193 and 194 are both CONDITIONAL
  and are now visible in `harness_log.md`. A third **auto-FAILs**.
- **If cycle 6 is also CONDITIONAL: STOP, do not run a seventh.** Write it up as
  an operator disposition ask and move on. The fix is live and correct; burning
  the night on artifact prose is the wrong trade.

### 3.2 THEN — 86.20 (P1, money path)

`'Strong Buy'` → `'STRONG BUY'` never matches `'STRONG_BUY'`, so it is dropped by
`continue` with no log. **1 genuine signal at score 8.36 — higher than any row
that did match.** Full spec in the step. Normalise the **separator**, cover
SELL/DOWNGRADE too, and do **not** claim a lost trade (candidate ≠ trade).

### 3.3 THEN, in this order

1. **86.17** — Layer-3 args boundary. **Research gate already PASSED** (10
   sources, 54 URLs, `handoff/current/research_brief_86.17.md`). Go straight to
   the contract. **Do not re-run the gate.** The brief carries the trap that
   would break it: the empty `catch` is **load-bearing** — the checker imports
   the slice with `args` unbound, so use `typeof args === 'undefined'`, never a
   bare `args === undefined`. Measured at $0: a no-args launch leaves `args`
   genuinely unbound, so this bites production too.
2. **86.6** — filesystem + subprocess channels. **Widen its digest set**: today a
   pytest run wrote the live `handoff/.autonomous_loop.lock`, which my md5 checks
   missed because that file is **untracked**. Digest the whole live-state set.
3. **86.21** — the 3rd-CONDITIONAL counter is blind to in-flight steps.
4. **86.10 / 86.11 / 86.14** — UI. Playwright works behind the auth wall; a
   SessionStart hook mints the cookie.
5. **86.12, 86.7, 86.5**, then the remaining 36.x P1s.

### 3.4 DO NOT

- **Do not promote 61.2 or any flag.** 61.2 is built-and-dark and needs an
  operator decision on a live book. **And its 83.5% baseline is stale** — that was
  measured across a dead-rail period; on a healthy rail today it was **1 of 6**.
  Any live_check must re-derive first.
- **Do not treat 61.2 as the reason the book isn't trading.** Measured: the BUY
  rate is **21.1% (103 of 489) over 90 days**. Today's 0-of-6 is ordinary
  variance, not a broken selector. A previous session generalised from n=6 and
  was wrong — do not repeat it.
- Operator-gated work (**79.x, 62.1.1, 61.3, 68.1, 61.2, 72.0.2**, asks #10, #13,
  #14, #19, #30–#33) → numbered ask, then **SKIP**.

## 4. Unattended rules

- **Every step still runs the full harness**: research gate → contract →
  generate → qa-verdict → harness_log → flip. No shortcuts because it is night.
- **Contract BEFORE generate.** If you breach it, disclose it AND name which
  automated check is blind to the breach.
- **Search the masterplan before filing a step.**
- **Never loosen a safety gate or weaken an assertion to get green.**
- **No metered spend.** The Max rail is $0; anything metered breaks the standing
  constraint.
- **Do not trigger a manual paper-trading cycle.** The scheduled one runs on its
  own. No verification cycle is authorized.

## 5. Backend restart — AUTHORIZED, with binding discipline

You may restart without asking. You may **not** skip these:

```
# 1. NEVER restart into a live cycle. Read the LOCK, never last_result:
cat handoff/.autonomous_loop.lock          # require "state": "released"
#    AND the lock pid must be DEAD and must NOT equal the backend pid
#    AND a ~1-2s lifetime means a pytest run, not a cycle (§7)

# 2. Restart:
launchctl kickstart -k gui/$(id -u)/com.pyfinagent.backend

# 3. Prove it took, on the RUNNING process:
launchctl list | grep com.pyfinagent.backend      # new pid
curl -s localhost:8000/api/health                 # status ok
tail -80 backend.log | grep -ci "EADDRINUSE"      # expect 0
```

- **Use `kickstart -k`, NOT `bootout`.** `bootout` is blocked by the 62.0 guard
  (away-ops rail 9) and it is the verb that caused the ~4-minute outage on 08-09.
  `kickstart -k` restarts the process, so `backend/.env` **is** re-read; only
  *plist* `EnvironmentVariables` need bootout, and nothing here does.
- **`kickstart -k` was measured at ~14s with no downtime signal, twice today.**
- **To prove a code change is IN FORCE, use content-last-changed, not mtime:**
  `git log -1 --format=%cd -- <file>` < process start, plus a clean tree.
  **mtime is NOT durable** — a same-content rewrite (e.g. a mutation run) pushes
  it past process start and the claim reads backwards. That happened today.

## 6. Fable 5 — AUTHORIZED, with a hard ceiling

**Where to use it:** genuinely hard analysis or research — a gnarly root-cause, a
wide audit, a design with real trade-offs. Switch with `/model fable`, or pass
`model: 'fable'` on an `Agent` call for one sub-task.

**Where NOT to, and why — do not "fix" these:**
- **Leave `qa-verdict.js` and `research-gate.js` pinned to `opus`.** Both carry an
  explicit rider trap (R4) saying so, and two Fable Q/A spawns **stalled
  mid-evaluation** on 2026-07-09. Opus is the reliable evaluator.
- **No bulk fan-out on Fable.** A 13-agent run is not free.
- **All Layer-2 / in-app pins stay off Fable** (2026-07-08 revert holds).

**The ceiling, and it is hard:** Fable draws the **same weekly Max budget** and
burns it faster. Past **50% of the weekly allowance** it silently moves to
**metered usage credits**, which breaks the standing `$0 metered` constraint.
Treat 50% as operator-gated. If you are unsure where you are, stay on Opus.

## 7. Traps that cost real time today — do not rediscover these

- **Derive every number LAST, and generate it from a command.** Three Q/A cycles
  were lost to numbers that were correct when measured and invalidated by the
  next edit. Never hand-copy a count, a ratio or a line anchor into prose. State
  the **relation** ("the line immediately before X"), or have the tool print it.
- **Edit the masterplan by TEXTUAL SPLICE, never `json.dump`.** A `json.dump`
  rewrite produced **24,200 insertions / 24,178 deletions** today (that is defect
  86.18) against 43 for a splice. With two sessions live it is a lost-update
  hazard. **Check `git diff --stat` before committing a masterplan edit.**
- **A guard from the instance is not a guard against the class.** Enumerate the
  member set and recall-test all of it. This cost four separate findings today.
- **Never run the mutation harness against the live production file while the
  backend is armed.** Mutate a copy, or use in-memory `sys.modules` injection.
  A tool timeout left a mutant in `autonomous_loop.py` today.
- **Do not run two pytest invocations at once.** The autouse
  `_live_audit_file_is_write_protected` fixture byte-compares a live file, so a
  concurrent run turns the other one RED. It produced a **false "3 failed"** today.
- **`handoff/.autonomous_loop.lock` is written by TESTS.** A ~1-2s lifetime with
  a dead, non-backend pid is a pytest run. It is **untracked**, so tracked-file
  digests miss it.
- **Small samples lie.** 0-of-6 looked like a broken selector; the 90-day rate is
  21.1%.

## 8. Scheduled jobs that will fire overnight — plan around them

| Job | Fires | Why you care |
|---|---|---|
| **`com.pyfinagent.away-session-pm`** | **22:00** | **Starts ANOTHER Claude session.** |
| **`com.pyfinagent.away-session-am`** | **07:30** | Same. |
| `com.pyfinagent.autoresearch` | 02:00 | Known RED (dead metered rail). Do not chase it tonight. |
| `com.pyfinagent.ablation` | 03:00 | |
| `com.pyfinagent.away-watchdog` | every 30 min | |

**ONLY ONE SESSION FLIPS MASTERPLAN STEPS.** Re-run `ListAgents` after 22:00 and
after 07:30. The auto-commit lock serialises commits, but `git add -A` still
cross-attributes a peer's files into your step's commit (that is 86.15). **Stage
with explicit pathspecs, and run `git add -An` before every flip** to see what
you are about to ship.

## 9. Stop conditions

- **HARD STOP immediately** on: any real-money action, any safety-gate loosening,
  metered spend, or **3 consecutive infrastructure failures**.
- **Soft stop** when only operator-gated work remains, or at ~07:00.
- **A step that will not close after 2 Q/A cycles: park it** with an honest
  disposition ask and move to the next one. Do not spend the night on one step.

## 10. Morning deliverable

Write **`handoff/current/day_report_2026-08-10.md`**, leading with **whether the
book can trade, yes or no**, and if no, exactly what blocks it. Then: what
closed, what was filed, numbered asks, and — separately and plainly — **what you
could not verify**. Regenerate the next goal from measured state. Commit and
push everything.

**Be honest about what you could not verify. A defect reported as fixed without
proof is worse than one reported as open — and a claim that is true but narrower
than it sounds is the one that will mislead.**
