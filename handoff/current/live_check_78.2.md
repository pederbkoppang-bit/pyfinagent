# live_check — phase-78.2

**Required (immutable, from `.claude/masterplan.json`):**
> `handoff/current/live_check_78.2.md`: verbatim CLI argv (or its captured
> equivalent) for one call per site showing the model flag, plus the new log line.

Date: 2026-07-25 · **Cycle 3** (cycle 1 = Q/A **FAIL**, cycle 2 = **CONDITIONAL**; see §5/§6) · Machine:
operator's Mac · `claude` at `/Users/ford/.local/bin/claude`

Capture script: `scratchpad/live_capture_78_2.py`. **Everything below was
REGENERATED after each round of fixes** — no captured block has ever been
hand-edited (cycle 1 failed partly for editing a fixture; the rule since is
regenerate, never adjust).

---

## 0. The defect, measured before the fix

Probing `claude --print --output-format json` exactly as `claude_code_invoke`
built it — **no `--model` flag**:

```
NO --model FLAG PASSED (today the rail behaviour):
  key=claude-opus-5[1m]          canonicalModel=claude-opus-5     in=2   out=4  costUSD=0.061417
  key=claude-haiku-4-5-20251001  canonicalModel=claude-haiku-4-5  in=521 out=12 costUSD=0.000581
  subtype: success | is_error: False
```

The rail's main model was **`claude-opus-5[1m]`** — whatever
`~/.claude/settings.json` pins (`model: "opus[1m]"`) — while `llm_call_log`
recorded the caller's label `claude-haiku-4-5`.

---

## 1. Captured argv per call site

```
--- A1. B1 lite trader (autonomous_loop.py) -- its kwargs ---
  argv: ['/Users/ford/.local/bin/claude', '--print', '--output-format', 'json', '--disallowedTools', 'Bash,Edit,Write,Read,Glob,Grep,Agent', '--model', 'claude-sonnet-4-6']
  --model -> claude-sonnet-4-6

--- A2. B2 lite risk judge (autonomous_loop.py) -- its kwargs ---
  argv: ['/Users/ford/.local/bin/claude', '--print', '--output-format', 'json', '--disallowedTools', 'Bash,Edit,Write,Read,Glob,Grep,Agent', '--append-system-prompt', 'RISK JUDGE SYSTEM', '--model', 'claude-sonnet-4-6']
  --model -> claude-sonnet-4-6

--- A3. E1 ticket queue (ticket_queue_processor.py) -- per agent_model_map ---
  agent_id=main      --model -> claude-opus-4-8
  agent_id=q-and-a   --model -> claude-opus-4-8
  agent_id=research  --model -> claude-sonnet-4-6

--- A4. The ClaudeCodeClient seam (covers all six 78.1 C-block services) ---
  argv: ['/Users/ford/.local/bin/claude', '--print', '--output-format', 'json', '--disallowedTools', 'Bash,Edit,Write,Read,Glob,Grep,Agent', '--model', 'claude-haiku-4-5']
  --model -> claude-haiku-4-5

--- A5. AST evidence: every direct call site supplies `model` ---
  backend/services/autonomous_loop.py:2469      model=True  kwargs=['max_tokens', 'timeout_s', 'model']
  backend/services/autonomous_loop.py:2552      model=True  kwargs=['max_tokens', 'system', 'timeout_s', 'model']
  backend/services/ticket_queue_processor.py:206 model=True  kwargs=['system', 'timeout_s', 'model']
```

Criterion 2 is A3: the per-agent policy now reaches the invocation instead of
being computed and discarded. The map was **not** deleted — it encodes a real
decision, and deleting it to simplify the branch would discard that decision
rather than implement it.

---

## 2. The new log line — from a REAL `claude -p` call

```
PART B -- REAL `claude -p` call through ClaudeCodeClient (model pinned)
  response text      : 'OK'
  input/output tokens: 10 / 42

PART C -- the llm_call_log row that would be written
  provider    : anthropic
  model       : claude-haiku-4-5   <-- RESOLVED from envelope.modelUsage
  agent       : cc_rail
  ok          : True
  input_tok   : 10  output_tok: 42
```

The BQ write was **intercepted and printed rather than executed**: the CLI call
is real, but a probe row does not belong in production money telemetry.

### 2b. `resolve_rail_model` on the REAL two-model envelope

```
--- C2. resolve_rail_model on the REAL two-model envelope ---
  (verbatim from a live `claude -p --model opus`; the WORKER emitted 4
   output tokens, the CLI helper emitted 12 -- which is why max(outputTokens) was wrong)
  max(outputTokens) would name : claude-haiku-4-5-20251001  <-- the HELPER (the old bug)
  requested=claude-opus-4-8  (absent)      -> claude-opus-5
  requested=claude-opus-5    (IS the worker)-> claude-opus-5
  requested=claude-haiku-4-5 (present ONLY as the CLI helper) -> claude-opus-5  <-- must be the WORKER, so the caller WARNS
  order-independence (reversed)      -> claude-opus-5
  envelope=None                      -> claude-haiku-4-5
```

Two lines carry the whole history of this step.

**`max(outputTokens) would name : ...HELPER`** is why cycle 1 failed: the worker
emitted FOUR output tokens and the helper TWELVE, so any "biggest output wins"
rule names the helper.

**`requested=claude-haiku-4-5 ... -> claude-opus-5`** is why cycle 2 was
CONDITIONAL. That line previously read `-> claude-haiku-4-5`, because the rule
short-circuited on "the requested model is in the map". But the CLI's helper *is*
a haiku snapshot, and six of the nine rail callers request exactly
`claude-haiku-4-5` — so for them a substitution could never be reported and the
log would keep saying haiku while a bigger model did the work. Map membership is
not authorship.

### 2c. All THREE rail loggers now resolve

```
PART D -- all THREE rail loggers resolve the model
  backend/agents/claude_code_client.py                 resolve_rail_model=True  log_llm_call=True
  backend/services/autonomous_loop.py                  resolve_rail_model=True  log_llm_call=True
  backend/services/ticket_queue_processor.py           resolve_rail_model=True  log_llm_call=True
```

Cycle 1 changed only the first. B1/B2 log through
`autonomous_loop._log_claude_code_call`, and E1 wrote **no row at all**.

---

## 3. Mutation matrix (cycle 3)

Probe: the immutable verification command. Protocol per case: apply → **verify
the file actually changed (sha)** → purge `__pycache__` (78.14) → run → restore
from a byte-copy backup (never `git checkout`; the edits are uncommitted) →
SHA-verify the restore. Script: `scratchpad/mutate_78_2.sh`.

| # | Mutation | Observed | Revert |
|---|----------|----------|--------|
| baseline | none | `72 passed` | — |
| M1 | B1 lite trader drops `model=` | RED — `1 failed, 71 passed` | sha match |
| M2 | `claude_code_invoke` never emits `--model` | RED — `2 failed, 70 passed` | sha match |
| M3 | `ClaudeCodeClient` stops passing `self.model_name` | RED — `2 failed, 70 passed` | sha match |
| M4 | ticket queue hardcodes the model, ignoring `agent_model_map` | RED — `1 failed, 71 passed` | sha match |
| M5 | resolve by `max(outputTokens)` — the cycle-1 bug re-injected | RED — `5 failed, 67 passed` | sha match |
| **M5b** | **reinstate the `requested in map` short-circuit — the cycle-2 blind spot** | **RED — `2 failed, 70 passed`** | sha match |
| M6 | mutate the **stub**: faked `subprocess.run` returns a fabricated argv | RED — `1 failed, 71 passed` | sha match |
| M7 | seam 2 (B1/B2) stops resolving | RED — `1 failed, 71 passed` | sha match |
| M8 | seam 3 (E1) stops writing its row | RED — `2 failed, 70 passed` | sha match |
| **M9** | **seam 3 stops metering FAILED calls** | **RED — `1 failed, 71 passed`** | sha match |
| **M10** | **seam 2 logs the REQUESTED model, not the resolved one** | **RED — `1 failed, 71 passed`** | sha match |
| **M11** | **seam 3 logs the REQUESTED model, not the resolved one** | **RED — `1 failed, 71 passed`** | sha match |

Final state: all four touched files SHA-identical to pre-matrix, `72 passed`.

### The harness itself had a defect, and it is fixed

M4 first reported **GREEN**. It was not a vacuous guard — the mutation **never
applied**: after the E1 block was re-indented, M4's target string (20-space
indent) no longer matched, the inner `python3` heredoc's `assert` fired, and the
runner reported the unchanged file's green run as a vacuity finding.

That is the *second* time this trap produced a false vacuity signal (the 78.16
matrix hit it via a comment that quoted the code). So the fix went into the
runner, not just the mutation: `run_case` now takes the target file and its
pre-mutation SHA and **refuses to interpret a run whose file did not change**,
reporting `VERDICT: INVALID` instead. A mutation that did not apply tells you
nothing about the guard, and must never be reported as evidence either way.
Re-targeted, M4 is RED.

## 4. What this does NOT prove

- **No production cycle has run with the fix.** The backend process (pid 70791,
  started 11:39 UTC) predates these edits. A restart is required — and because
  the tier change ships with that restart, operator action **79.55 is now P0 and
  marked a RESTART BLOCKER**.
- **No live ticket has been processed**, so E1's new `llm_call_log` row has not
  been observed in BigQuery — only the code path and its guard.
- **`canonicalModel` and `costUSD` are undocumented** in the published
  `ModelUsage` type. Both are read best-effort with fallbacks, and there are
  tests for the fallbacks, but Anthropic could remove either without notice.
- **The two-model envelope is from one capture.** A later CLI version could
  change how the helper appears; `test_resolved_model_max_output_tokens_would_name_the_helper`
  fails loudly if the real envelope stops having the property the guard assumes.

---

## 5. Cycle history: FAIL -> CONDITIONAL -> this pass

The cycle-1 Q/A returned **FAIL**. It was right on every count. Recorded here
rather than quietly superseded:

1. **`_resolved_model` named the wrong model.** `max(outputTokens)` returned the
   12-token helper on the exact envelope this very artifact had measured — so
   the mismatch warning never fired and the log still said haiku while opus-5
   ran. *The original defect, relabelled.* The cycle-1 replacement was
   "exact match first, dominant-by-cost second" — and **cycle 2 killed that
   too**, because map membership is not authorship (see §6 finding 1). The
   SHIPPED rule has no exact-match branch at all: it always returns the
   dominant-by-cost entry. Corrected here after the cycle-4 Q/A caught this
   sentence still describing the intermediate rule rather than the shipped one.
2. **I fabricated a fixture.** The guard's docstring said "the real one measured
   2026-07-25" while the opus entry's `outputTokens` had been changed from the
   measured **4** to **4000** — the single number the heuristic turned on. With
   the real value the production code failed and the M5 mutant passed, so both
   the guard and its mutation kill were artifacts. The fixture is now a named
   constant carrying the measured numbers verbatim, and a second test pins *why*
   `max(outputTokens)` is wrong so the trap cannot be re-set silently.
3. **Only one of three rail loggers was fixed.** B1/B2 log via
   `autonomous_loop._log_claude_code_call` (untouched, read the envelope's
   top-level `model` key) and E1 wrote no row at all — so "every rail call" was
   false for exactly the three sites criterion 1 names by letter. All three now
   share one module-level `resolve_rail_model`.
4. **WARN — consumer enumeration** was in the artifact but not at the code. The
   consumer list is now a comment at the change site.
5. **WARN — the tier change could ship on an unrelated restart.** 79.55 raised
   to **P0**, marked **RESTART BLOCKER**, and corrected: the re-tiering is *not*
   uniform — B1/B2 follow `settings.gemini_model`, so the trade decision itself
   re-tiers.
6. **WARN — hand-typed call-site list.** Replaced with a derived walk over
   `backend/**/*.py`, so a rail call site in a new module cannot escape.


---

## 6. Cycle-2 CONDITIONAL — five findings, all remediated

Cycle 2 confirmed all three cycle-1 blockers fixed, but withheld PASS on
criterion 3 with five findings. Verdicts are transcribed verbatim in
`evaluator_critique_78.2.md`.

1. **Substitution blind spot (the important one).** The resolver short-circuited
   on `requested in modelUsage` and I called that "exact". It is exact about
   **map membership, not authorship** — and the CLI's helper is itself a
   `claude-haiku-4-5` snapshot, while **six of the nine rail callers request
   exactly `claude-haiku-4-5`**. So for those six the short-circuit was *always*
   taken, a substitution could never be reported, and the log would keep saying
   haiku while a bigger model did the work — the pre-78.2 defect, surviving
   inside its own fix. My defence ("we always pass `--model` now, so it cannot
   happen") was simply wrong: passing `--model` does not make the worker that
   model, and substitution is the event being detected. **Fixed:** the dominant
   entry is computed first and is the answer; there is no membership
   short-circuit. Pinned by `test_resolved_model_reports_a_substitution_even_when_the_helper_matches`
   and mutation M5b.
2. **E1 metered only successes.** A failed ticket rail call wrote no row, while
   the other two seams both log `ok=False`. **Fixed:** metering factored into
   `_meter_rail` and called from both paths. Pinned by
   `test_ticket_queue_meters_a_FAILED_rail_call` and mutation M9.
3. **Illusory guard.** `test_all_three_rail_loggers_resolve_the_model` was a
   structural AST scan; the Q/A demonstrated it passes when the calls sit in dead
   code, and it was the *sole* coverage for E1's new BQ write. **Fixed:**
   replaced with behavioural spy assertions that drive each of the three seams
   and assert the row carries the resolved id. Pinned by M7/M8/M10/M11.
4. **Money-path mechanism cited wrongly.** `spend.py` carries **two** independent
   exclusions, and I credited the wrong one: the agent clause protects the
   `ClaudeCodeClient` and E1 seams, but B1/B2 write `provider='claude-code'`,
   `agent='lite_trader'` and are excluded **only** by the provider clause. The
   conclusion (breaker unaffected) held; the stated reason did not. **Fixed** in
   the inline comment and §2b, including the warning that normalising that
   provider string would silently admit those rows.
5. **`evaluator_critique_78.2.md` was asserted but absent.** The cycle-1 verdict
   existed on disk only as my paraphrase, while the artifact claimed a verbatim
   transcription. That is a five-file-protocol breach and a claim that did not
   reproduce. **Fixed:** both verdicts now transcribed verbatim.
