# Experiment results — Step 78.0 (Anthropic call-site census, audit-class, READ-ONLY)

Date: 2026-07-25 | Cycle: 158 | Execution: Main (Opus 5) GENERATE

## What was produced

### 1. `handoff/current/census_78.json` (NEW) — machine-readable decision table

**28 role rows**: `max_rail_cli` 19, `max_rail_proxy` 1, `stay_metered` 8. Each row
carries `id, role, anchor, model_today, structured_output, volume_30d,
latency_budget, instrumented, decision, reason, owner_step`.

### 2. `handoff/current/census_78.md` (NEW) — human-readable twin

**Generated FROM the JSON by a single renderer**, so the two artifacts cannot drift —
there is one source of truth, not two hand-maintained copies.

### 3. `.claude/masterplan.json` — 9 follow-up steps authored (78.1 … 78.9)

Each executor-tagged, research-gated, with immutable criteria and a live_check
requirement, written for an executor with no memory of this session:

| Step | P | Owns |
|------|---|------|
| 78.1 | P1 | C1–C6 overlay rewire off direct `ClaudeClient` (the phase-72 rail-bypass class) |
| 78.2 | P2 | B1/B2/E1 pass no `--model` on the rail; ticket queue's agent_model_map silently ignored |
| 78.3 | P2 | F1–F3 + G1 rewire; make the F-block's silent key-prefix→Gemini fallback loud |
| 78.4 | P2 | D1/D2 rewire, D3 explicitly EXCLUDED (guarded by a test); loud 401 latch |
| 78.5 | P3 | BatchClient fix-or-retire (latent no-args TypeError behind a doubly-dark flag) |
| 78.6 | P3 | openclaw_client disposition (dormant + hardcoded model table) |
| 78.7 | P2 | sonnet-5 pricing rows + `_VALID_MODELS` + close 76.11 |
| 78.8 | P1 | **NEW** — llm_call_log is blind to every raw-SDK metered site |
| 78.9 | P1 | **NEW** — 70% rail failure rate, unalarmed |

## Method (what makes this a census rather than a restatement)

1. **Anchors re-derived MECHANICALLY.** A script opened every claimed `file:line` and
   asserted the expected symbol was there, reporting actual line + drift — 42 anchors,
   42 resolved, 5 small drifts found. NOTE: cycle-1 detected the drifts but did not propagate 4 of the 5 into the deliverables; they are now applied to census_78.json/md and to steps 78.3/78.8 (stale-anchor grep over the masterplan returns 0). Not an LLM re-read of the brief.
2. **Volumes measured**, not inherited: a verbatim 30d `GROUP BY` over
   `llm_call_log`. The goal text's 2,241/4.1M figure is from the 2026-07-24 window;
   this cycle re-measured on a slid window and reports **its own** numbers.
3. **Instrumentation audited per site**, which is what makes "0 rows" interpretable.
4. **Scope disambiguation section** (7 entries) — every scope area that turns out NOT
   to be a call site is recorded with its verdict and why it was checked, rather than
   silently dropped. This is what closes criterion 1's explicit instruction to
   disambiguate top-level `backend/autonomous_loop.py` from
   `services/autonomous_loop.py`: the top-level file configures and constructs
   `PlannerAgent` (:75, :369) but issues no API call itself — the calls are in
   `planner_agent.py:166/:273`, censused as G1. Also enumerated: **all**
   `claude_code_invoke` callers (B1, B2, E1 — no others exist outside tests and the
   definition), the import-for-exception-typing sites that an import sweep would
   wrongly count, and the stale `evaluator_agent.py` "Uses Claude Sonnet" docstring.
   Keeping these OUT of `roles` (rather than padding the count with fake
   `stay_metered` rows) is deliberate: the roles list is call sites only.

## Two findings that were NOT in the research brief

Both were produced by the mechanical passes above, and both are queued rather than
fixed here (READ-ONLY step):

**(a) The spend meter is blind to the metered path — 78.8.** **11 of 12** raw-SDK
sites write no `llm_call_log` row (derived from the census's own `instrumented` field
over the denominator {A4, A5, A6, D1-D4, F1-F3, G1, H1}; only A4 is instrumented).
Instrumentation follows the CC rail and the wrapper clients, not the raw SDK. On
dual-rail sites the asymmetry is exact — the rail branch logs, the `else:` metered
branch does not. Since `fetch_llm_spend` and the $25/day breaker read that table,
**the spend they exist to govern is the spend they cannot see**, and an unlogged call
is indistinguishable from a call that never happened.

*Corrected after Q/A cycle-1:* this section first said "9 of 12", which did not
reproduce against the census's own field — three different cardinalities (9, 10, 11)
appeared across the artifacts for one set. The number is now derived rather than
asserted, and 78.8 carries the corrected figure. The error understated the blindness,
so the conclusion stands.

**A second, distinct hole the Q/A surfaced**, now also owned by 78.8: even the
*instrumented* wrapper clients log **successes only**. `ClaudeClient` hardcodes
`ok=True` and `llm_client.py` has no `ok=False` writer (errors re-raise at
`:1739`/`:1746`/`:1790`, before the log block at `:1886`), while the CC rail *does*
log failures — which is why the rail shows 1,547 failures and the wrappers show none.
**Consequence for this census: a zero row count can never prove a path was dormant.**
My original C1–C6 reading — "0 rows, and ClaudeClient is instrumented, therefore they
genuinely did not run" — was an unsound inference and is corrected in the census rows
and in live_check §4. 78.1 now carries an explicit instruction not to assume the six
overlays are dormant, since "ran and failed every time" is exactly the dead-credits
scenario that step exists to fix.

**(b) A 70% rail failure rate, unalarmed — 78.9.** 1,547/2,192 sonnet and 294/357
opus-4-7 `cc_rail` rows have `ok=false`. 78.9's first task is to establish what
`ok=false` means on that write path before anyone concludes anything, because the
severity depends entirely on it.

## Criteria status

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | Covers every scope area incl. the 4 missed raw-SDK sites; anchors re-derived | MET | live_check §2 (42/42); the 4 sites are C2–C5, surfaced by the gate's round-2 `ClaudeClient(` sweep |
| 2 | Audit-class gate with coverage.dry=true | MET | live_check §1: 9 rounds, rounds 8+9 dry; the envelope EXCERPT there is relabelled as such (the full envelope lives in research_brief_78.0.md) |
| 3 | Every row has a decision + reason; volume measured or honestly unmeasurable | MET | census_78.json all 28 rows; live_check §3 (query) + §4 (why some are unmeasurable) |
| 4 | advisor_call + BatchClient stay_metered with no-CC-equivalent evidence | MET | live_check §5: beta advisor tool (hard-raises under the flag) and Batches API (50%/24h, no CLI equivalent) |
| 5 | Census names the follow-up step owning each max_rail decision | MET | `owner_step` on every row; 78.1–78.9 authored above |

## Verification (verbatim)

```
$ python3 -c "import json; c=json.load(open('handoff/current/census_78.json')); assert len(c['roles'])>=12, ...; assert all(r.get('decision') in ('max_rail_cli','max_rail_proxy','stay_metered') for r in c['roles']), 'undecided rows'"
exit=0

$ python3 -c "import json; json.load(open('.claude/masterplan.json')); print('masterplan JSON valid')"
masterplan JSON valid
```

## Boundary honesty

READ-ONLY as contracted: **no production code changed, no flag flipped, no model pin
touched.** The only repository writes are the two census artifacts and the 9 new
`pending` masterplan steps. Every routing recommendation remains the operator's call.
