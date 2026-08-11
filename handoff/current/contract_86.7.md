# Contract -- step 86.7

**Step**: `86.7` (phase-86, **P1**, `harness_required: true`) | **Phase**: PLAN
**Date**: 2026-08-11 (~11:3x CEST) | **Driver**: Main (`pyfinagent-06`), Opus 5 / effort max
**Written BEFORE any code.** No production file is modified at this moment.

**Concurrency**: a peer session owns 86.29/86.38 and has parked 86.21. It
acked 86.7 as mine and independently endorsed the keychain-lock refusal below.

---

## 1. Research gate

**PASSED** -- `wf_d6c07606-eb3`, tier `moderate`, brief `research_brief_86.7.md`
(55,714 chars). Script-enforced: **10 sources read in full** (floor 5), **22
URLs** (floor 10), recency scan present, all 10 claimed URLs verified in the
brief, `brief_status: COMPLETE`, `rail_dropped: null`, self-report agreed.
**Disclosed:** the WebSearch budget was exhausted session-wide, so currency of
the external survey is weaker than the tier implies. The 10 full reads are
canonical (RFC 8628, RFC 9700, Apple TN2083, systemd credentials, Google SRE,
the Claude Code auth doc, `security(1)`, Entra refresh tokens) and the
load-bearing findings are **internal measurements**, which I re-derived.

### The gate refuted the step's own premise

**The step's title says the plist token "no longer exists". IT DOES.** Criterion
5 asks whether 62.1.1 and 85.3.3 are closed; the honest answer is **STILL-OPEN**,
and 85.3.3's git-history criterion now resolves **POSITIVE**, which by its own
text **mandates rotation**.

**I verified this independently, by hash, printing no values:**

```
5 GIT-TRACKED handoff/away_ops/session_*.json files contain a 92-char sk-ant-*
  one distinct value, sha256[:16] = 32fd305146379e49
  window: 2026-08-08T20:00:08Z .. 2026-08-10T20:00:10Z
  the NEXT session file (2026-08-11T05:30:09Z) is CLEAN -- bounded and stopped
separately: 12 plist-like files carry 3 OTHER distinct tokens (79 / 79 / 123 chars)
```

**AND I GOT THIS WRONG FIRST, IN THE DANGEROUS DIRECTION.** My initial read said
the leak was **ONGOING**, because I sorted filenames alphabetically and
`session_pm_*` outranks `session_am_*` — so `pm_20260810` looked newer than
`am_20260811`. That would have escalated a contained incident into an active one.
Corrected by sorting on the embedded timestamp. Same sort-key error I made two
hours earlier on the verdict ledger; both are now one rule in the goal's TRAPS.

### Four other findings that reshape the plan

1. **Every pyfinagent launchd job is an AGENT; there are zero LaunchDaemons.**
   That is *why* the keychain is reachable at all (TN2083). Any conclusion about
   headless auth that ignores this is unsound.
2. **`claude auth status` proves credential PRESENCE, not validity** —
   `healthcheck.sh:86-89` says so itself. A 401 exits **0** and lands at
   `claude_code_client.py:474-483`. So criterion 1 cannot use it.
3. **The login keychain measures `no-timeout`: screen lock does NOT lock it.**
   Reboot is the real trigger. This **partly refutes the step's own audit basis**.
4. **Every alert is a one-shot latch and `run_away_session.sh:157` exits 0 on
   auth-dead.** The gap is **recall, not precision** — which is criterion 2's
   "must alert, not silently degrade", already broken.

## 2. Immutable success criteria

Six, copied verbatim into `experiment_results_86.7.md` §2 at GENERATE rather than
duplicated here — a paraphrase in two places is the divergence defect that made
86.36 CONDITIONAL this morning (`qa.md` vs `qa-verdict.js`).

## 3. Plan

**P0 -- CRITERION 5 IS ANSWERED NOW, IN WRITING: STILL-OPEN, BOTH.** The premise
that the token "no longer exists in any plist" is false on two counts (12 plists,
5 tracked JSONs). This is recorded regardless of what happens to the rest.

**P1 -- ROTATION IS AN OPERATOR ASK (#2), NOT A TASK.** Raised in the binding
goal. I will not rotate, delete, or rewrite history. Criterion 3 can still be
*planned* — the options are enumerable without acting — but a fallback that
depends on a rotated credential cannot be validated until the ask returns.

**P2 -- CRITERION 1 NEEDS THE REAL PROBE.** Run an away-session entrypoint (or a
faithful stand-in with the same launchd context) with **no env token**, and
assert on `is_error` / `subtype` / `duration_api_ms` from the JSON envelope —
**not** on `claude auth status`, which finding 2 rules out. The step forbids
inferring from the attended backend, and that prohibition is correct.

**P3 -- CRITERION 2 WITHOUT LOCKING THE KEYCHAIN.** The criterion permits *"lock
the keychain, **or** run in a context without it"*. I take the second leg only:
the backend (pid 66306) authenticates through the login keychain and the book
runs at 20:00 CEST. **A dark rail at 20:00 is a worse outcome than an unproven
criterion**, and that trade is not mine to make unilaterally. If the lock proves
genuinely necessary, it becomes an operator ask rather than an action.

**P4 -- CRITERION 4 is a capture task, not an engineering one.** Record the
reproduction for Anthropic. No fix is available from this side.

**P5 -- CRITERION 6, WITH ITS LIMIT STATED IN THE CELL.** Re-introduce the token
into a **copy** of the plist and show the rail check fails again. **The copy is
never loaded**: `bootout`+`bootstrap` is the operator's verb (away-ops rail 9),
and `kickstart -k` does not re-read `EnvironmentVariables` (measured 2026-08-09).
So the cell proves **the CHECK reacts**, not that launchd would. Saying so in the
artifact is strictly better than a green cell that implies more.

### Guard shapes to avoid — named in advance

From the peer's four CONDITIONALs today, every one a guard that could not fail:
a source-order assertion; a self-check asserting a library fact; a one-sided
self-check pinning two of three outcomes; and a guard defeated by its own
docstring quoting the literal it grepped for. Criteria 2 and 6 are where I could
most easily write one here.

### Explicitly NOT doing

- **Not** locking the login keychain (P3). **Not** rotating, deleting, or
  history-rewriting (P1). **Not** writing `backend/.env`.
- **Not** touching the peer's steps.

### Risk

The subject is the auth path of the live analysis rail, ~8h before a cycle. Every
probe must be read-only on credentials. The immutable command itself makes a real
rail call (drops the env token, asks the production client for one token) — $0 on
the Max rail, but live, and it is the one place this step touches production.

## 4. References

- `handoff/current/research_brief_86.7.md` (gate `wf_d6c07606-eb3`)
- RFC 8628 / RFC 9700; Apple TN2083; `security(1)`; systemd CREDENTIALS;
  Google SRE monitoring + alerting-on-SLOs; Claude Code authentication doc
- `backend/agents/claude_code_client.py:474-483`; `healthcheck.sh:86-89`;
  `run_away_session.sh:157`; away-ops rail 9
