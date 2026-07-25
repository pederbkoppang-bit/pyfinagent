# Contract — phase-78.2

**Step id:** 78.2
**Phase:** phase-78 (Anthropic Max-rail LLM routing — full audit + fix)
**Priority:** P1 · `harness_required: true`
**Executor tag (masterplan):** sonnet-4.6/high
**Boundary (masterplan, binding):** `claude_code_client.py` callers + tests
**Date:** 2026-07-25 · Cycle 165

---

## 1. Why this step exists

`claude_code_invoke` builds its argv with no `--model` flag, so every CC-rail
call runs on whatever the `claude` CLI resolves by default, while
`llm_call_log` records the *caller's* label. The log is therefore actively
misleading about which model produced a signal.

This step also **blocks 78.1's criterion 2** ("model tier per service is
unchanged from the census table (all haiku-tier)"), which its own Q/A flagged.
78.1 cannot honestly close until this lands.

---

## 2. Research-gate summary

**Brief:** `handoff/current/research_brief_78.2.md` (researcher subagent, tier
`simple`, run on **Sonnet** — a deliberate cheaper-model choice for a narrow,
well-anchored question, per the operator's standing instruction; the ≥5-source
floor was not relaxed and the brief cleared it).

> **GATE VERDICT — `gate_passed: true`.** Envelope verbatim:
> `external_sources_read_in_full: 6`, `snippet_only_sources: 19`,
> `urls_collected: 25`, `recency_scan_performed: true`,
> `internal_files_inspected: 13`.

Findings that drive the plan:

| # | Finding | Consequence |
|---|---------|-------------|
| R1 | `--model` accepts **both** aliases (`opus`/`sonnet`/`haiku`) and full ids (`claude-haiku-4-5`). | We can pass the existing `model_name` strings unchanged — no id-mapping layer. |
| R2 | Precedence when `--model` is omitted: `/model` (N/A for `-p`) → `--model` → `ANTHROPIC_MODEL` → the `model` field in `~/.claude/settings.json`. The rail sets none of the first three. | Every rail call falls through to the **interactive session's own pin**. Confirmed: `~/.claude/settings.json` `model = "opus[1m]"`. |
| R3 | An invalid `--model` is **not** validated at launch; it surfaces as a normal error on the first request, already handled by `claude_code_invoke`'s non-success-`subtype` path (`claude_code_client.py:362-370`). | No new error branch needed. |
| R4 | The JSON envelope reports the **resolved** model(s) via `modelUsage: {modelName: ModelUsage}`, each entry carrying `canonicalModel`. | We can log what actually ran, not just what we asked for. |
| R5 | The real caller denominator is **9, not 3**: the 3 direct `claude_code_invoke` sites **plus** the 6 C-block services, because `ClaudeCodeClient.generate_content` builds its own internal `claude_code_invoke` call that also omits the model. | The fix must land in `ClaudeCodeClient`, not only at the 3 named sites, or 6 callers stay broken. |
| R6 | `ticket_queue_processor.py:172-206` computes `agent_model_map` and then **discards** it on the rail branch. | Criterion 2's "actually honored or deleted as dead" has a concrete target. |
| R7 | `ClaudeCodeClient.model_name` is currently **write-only for observability** — passed to `_log_cc_call` as the BQ label at `:607`/`:626`, never read back to build argv. | Threading it into argv is additive; the constructor already receives it. |
| R8 | `canonicalModel` / `provider` are **live-observed but undocumented** in the TS-SDK `ModelUsage` type. | Treat as best-effort: `.get()` with a fallback to the dict key, never `KeyError`. |

**Researcher recommendation:** do **both** — pass `--model` (fixes what runs) *and*
log the resolved model from `modelUsage` (catches future silent substitutions,
including Anthropic's own safety-classifier fallback). Adopted.

### My own measurement — what the rail runs TODAY

Live probe, this machine, 2026-07-25, `claude --print --output-format json` with
**no** `--model` flag (the exact shape `claude_code_invoke` builds):

```
NO --model FLAG PASSED (today the rail behaviour):
  key=claude-haiku-4-5-20251001  canonicalModel=claude-haiku-4-5  in=521 out=12 costUSD=0.000581
  key=claude-opus-5[1m]          canonicalModel=claude-opus-5     in=2   out=4  costUSD=0.061417
  subtype: success | is_error: False
```

Compare, same probe **with** `--model claude-haiku-4-5`: the main entry becomes
`canonicalModel: claude-haiku-4-5`, `provider: firstParty`, `maxOutputTokens: 32000`.

**So the defect is not theoretical and it is worse than the census stated.** The
rail's main model today is **`claude-opus-5[1m]`** — the top tier at 1M context —
for calls whose `llm_call_log` row says `claude-haiku-4-5`. The second entry is
the CLI's own internal quick-task helper, which is why `modelUsage` is a dict.

---

## 3. Hypothesis, and the behaviour change this causes (stated up front)

> Thread an explicit model from every rail call site into argv, and log the
> **resolved** model from the envelope alongside the requested one. The rail then
> executes what the caller asked for, and the log stops lying.

**This is a real behaviour change on live signal paths, in the direction of
LOWER model tier, and it must not be buried.** Today the six overlays, the lite
trader and the lite risk judge are all silently running Opus 5 (1M). After this
step they run their configured tier — `claude-haiku-4-5` for the six.

Why ship it in that direction anyway:

- Every downstream system already **believes** they are haiku: `llm_call_log`
  labels, the cost accounting, the census, and 78.1's criterion 2. Reality is the
  outlier, not the config.
- The current state is **accidental**, not designed: it is whatever the operator
  last selected with `/model` in an interactive session. A `/model sonnet` today
  would silently re-tier live trading decisions with no code change and no log
  line — which is the risk the step exists to remove. Leaving it "accidentally
  good" keeps the *variance*, which is the actual hazard.
- Nobody chose Opus-5-1M for a news-screening overlay, and nobody could see that
  they had it.

**But the tier itself is the operator's call, not mine.** So this step ships the
plumbing and the honest labels, and **queues an operator decision** (§7) with the
measured before/after, so re-pinning any role upward is a one-line config change
made deliberately rather than a side effect of an interactive `/model`.

---

## 4. Immutable success criteria (verbatim from `.claude/masterplan.json`)

1. "claude_code_invoke accepts and forwards an explicit model to the CLI; B1, B2 and E1 each pass one"
2. "The ticket queue's agent_model_map is actually honored on the rail branch (test asserts the per-agent model reaches the invocation), or the map is deleted as dead -- whichever the executor justifies, not left silently ignored"
3. "Every rail call logs the resolved model so post-hoc audit is possible"
4. "MUTATION: drop the model argument at one call site -> a test asserting the explicit model goes red"

**Verification command (immutable):**
```
.venv/bin/python -m pytest backend/tests/test_phase_56_2_ops_fixes.py backend/tests/test_phase_75_llm_rail.py -q
```

**live_check (immutable):**
`handoff/current/live_check_78.2.md`: verbatim CLI argv (or its captured
equivalent) for one call per site showing the model flag, plus the new log line.

---

## 5. Plan

1. **`claude_code_invoke` gains `model: Optional[str] = None`** and emits
   `["--model", model]` into argv when set. Default `None` preserves today's
   behaviour for any caller not yet updated (and makes the mutation in criterion 4
   meaningful). Docstring records R2's precedence chain so the next reader knows
   what `None` actually resolves to.

2. **`ClaudeCodeClient.generate_content` passes `self.model_name`** — this is the
   R5 fix that covers 6 of the 9 callers at one seam, including all six C-block
   services from 78.1.

3. **The three direct sites** pass an explicit model:
   `autonomous_loop.py:2454` (lite trader), `:2530` (lite risk judge), and
   `ticket_queue_processor.py:206` (ticket queue).

4. **Criterion 2 — `agent_model_map` honored, not deleted.** It is a real
   per-agent policy (main/q-and-a → opus-4-8, research → sonnet-4-6) that is
   simply unreachable on the rail branch. The fix is to pass its selection into
   the invocation. Deleting it would throw away an intentional policy to make a
   test easier; that is the wrong trade. A test asserts the per-agent model
   reaches argv.

5. **Criterion 3 — log the resolved model.** In `_log_cc_call`, read
   `envelope["modelUsage"]`, take `canonicalModel` (`.get()` with fallback to the
   dict key, per R8), and pick the entry with the largest `outputTokens` as the
   primary — the CLI's internal helper always appears as a second, tiny entry, so
   "largest output" is what distinguishes the model that did the work. Log the
   **requested** model too, so a request/resolved mismatch is visible. A mismatch
   is the only way to catch a future silent substitution once `--model` is
   correct.
   *Open question for GENERATE:* whether the resolved model goes into the existing
   `model` column or a new field. `llm_call_log` has no spare column, and adding
   one is a BQ migration — outside this step's boundary. Default: put the
   **resolved** model in `model` (it is the truthful answer to "what produced this
   row") and emit the request/resolved pair to the application log when they
   differ. If that proves wrong, disclose rather than silently widen scope.

6. **Trace the two irregular call sites** before writing the fix:
   `analyst_narrative_scorer.py` and `call_transcript_gpr.py` take a caller-supplied
   `model` local rather than reading settings inline (the researcher's open gap).

7. **Mutation matrix** (criterion 4), run AFTER the work is complete, purging
   `__pycache__` between cases and reverting **from backup, not `git checkout`**
   (the edits will be uncommitted):
   - M1: drop the model argument at one direct call site → argv test RED.
   - M2: `claude_code_invoke` accepts `model` but never appends it to argv → RED.
   - M3: `ClaudeCodeClient` stops passing `self.model_name` → the C-block coverage RED.
   - M4: `agent_model_map` selection replaced with a constant → the per-agent test RED.
   - M5: mutate the **stub** — the fake `subprocess.run` reports a fixed argv → the
     argv assertions must fail (proves they read the captured value).

---

## 6. Scope fence

- Does **not** change any model *pin* in `model_tiers.py` or settings. It makes
  the configured pin actually take effect.
- Does **not** add a `llm_call_log` column (BQ migration is out of boundary).
- Does **not** touch the metered/direct path.
- Does **not** close 78.1; that is a separate cycle once this lands.

---

## 7. Operator decision queued out of this step

The measured before/after tier change is an operator-visible policy question, so
it goes to the owed-actions phase (phase-79) rather than being decided here:

> **RAIL-MODEL TIER CONFIRMATION.** Until 78.2, every CC-rail call ran the
> interactive session's pinned model — measured 2026-07-25 as
> `claude-opus-5[1m]` — regardless of the caller's configured tier. After 78.2
> the rail runs the configured tier (e.g. `claude-haiku-4-5` for the six signal
> overlays). If any of those roles should keep a higher tier, it is now a
> deliberate one-line config change in `model_tiers.py` / settings rather than an
> accident of `/model`. Confirm the per-role tiers or record "as configured".

---

## 8. Open gaps carried into GENERATE

- **`ANTHROPIC_MODEL` in `backend/.env` could not be read.** Bash access to
  `backend/.env` is denied by a permission rail (the researcher hit the same wall
  from its sandbox). **This is moot for the fix**: per R2, `--model` outranks
  `ANTHROPIC_MODEL`, so threading the flag makes the resolved model deterministic
  regardless of what that env var says. Recorded rather than silently dropped.
- `canonicalModel` is undocumented (R8) — guarded, never load-bearing.

---

## 9. References

- `handoff/current/research_brief_78.2.md` — research gate
- `backend/agents/claude_code_client.py:215` (`claude_code_invoke`), `:263-272` (argv), `:489-515` (`_log_cc_call`), `:607`/`:626` (label call sites)
- `backend/services/autonomous_loop.py:2454`, `:2530`; `backend/services/ticket_queue_processor.py:172-206`
- `scripts/mas_harness/run_cycle.sh:66-71` — the correct in-repo reference pattern
- `backend/tests/test_phase_75_llm_rail.py:111-128` — the argv-capture seam to reuse
