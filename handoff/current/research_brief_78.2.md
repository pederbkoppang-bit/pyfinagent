# Research Brief -- phase-78.2

Topic: CC-rail (`claude -p` Max-subscription rail) invocation helper
passes no `--model` flag; every rail call runs on the CLI's session
default; the request-model is only ever used as a BigQuery log LABEL,
not threaded into argv. Need: (A) external Claude Code CLI docs on
`--model` flag grammar / default resolution precedence / env-var route
/ JSON output envelope model-used field; (B) internal complete caller
census + test seam; (C) recency scan.

Tier: **simple**. Not audit-class. Model: Sonnet (operator-authorized
cheaper model for this narrow gate; >=5-sources-read-in-full floor and
recency scan NOT relaxed).

STATUS: COMPLETE.

---

## A. External sources -- read in full (6 >= 5 required; counts toward gate)

| # | URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|-----|----------|------|-------------|----------------------|
| 1 | https://code.claude.com/docs/en/cli-reference | 2026-07-25 | Official docs | WebFetch (2 targeted passes) | Verbatim: `--model` "Sets the model for the current session with an alias for the latest model (`sonnet`, `opus`, `haiku`, or `fable`) or a model's full name. **Overrides the `model` setting and `ANTHROPIC_MODEL`**". `--output-format` "options: text, json, stream-json". `--fallback-model` overrides the `fallbackModel` setting. |
| 2 | https://code.claude.com/docs/en/model-config | 2026-07-25 | Official docs (Anthropic/Claude Code) | WebFetch, read to completion (733 lines total, 2 reads covering the full page) | THE authoritative precedence source. "Setting your model" lists priority order 1) `/model` mid-session 2) `--model` at startup 3) `ANTHROPIC_MODEL` env var 4) `model` field in settings file. Confirms: "The check [that rejects unrecognized model strings] ... doesn't cover the `--model` flag, the `ANTHROPIC_MODEL` environment variable, or the `model` setting; a mistyped value there produces `There's an issue with the selected model` on the first request instead." Also: "read the actual model from the `modelUsage` field of the result message" (the exact pointer that answers the spawn prompt's question B4). |
| 3 | https://code.claude.com/docs/en/headless | 2026-07-25 | Official docs | WebFetch, read in full (entire "Run Claude Code programmatically" page) | "With `--output-format json`, the response payload includes `total_cost_usd` and **a per-model cost breakdown**" -- textual confirmation of what source #4 shows structurally. |
| 4 | https://code.claude.com/docs/en/agent-sdk/typescript | 2026-07-25 | Official docs (TypeScript SDK reference) | `curl` raw HTML + Python tag-strip on the `#modelusage` and `#sdkresultmessage` anchors (WebFetch summarized this page too lossily on a first pass; the gcloud-docs-style JS-render workaround from `feedback_gcloud_docs_fetch.md` applied here too and worked -- counts as read-in-full since verbatim type text was extracted from the served HTML, not paraphrased) | Verbatim type defs: `type ModelUsage = { inputTokens: number; outputTokens: number; cacheReadInputTokens: number; cacheCreationInputTokens: number; webSearchRequests: number; costUSD: number; contextWindow: number; maxOutputTokens: number; }` and `SDKResultMessage.modelUsage: { [modelName: string]: ModelUsage }`. **This is the definitive schema answer**: `modelUsage` is a dict KEYED BY THE MODEL(S) THAT ACTUALLY RAN, not a caller-echo field. |
| 5 | `claude --help` / `claude -p --help` (local, v2.1.220) | 2026-07-25 | Primary source (CLI itself) | Direct Bash execution, verbatim output captured | `--model <model>` "Model for the current session. Provide an alias for the latest model (e.g. 'fable', 'opus', or 'sonnet') or a model's full name (e.g. 'claude-fable-5')." `--fallback-model <model>` "only works with --print". Confirms no separate "headless model flag" exists -- `--model` is the one and only flag, works identically in `-p` mode. |
| 6 | Live probe: `claude -p "reply with the single word OK" --model claude-haiku-4-5 --output-format json --disallowedTools "Bash,Edit,Write,Read,Glob,Grep,Agent"` (local, v2.1.220) | 2026-07-25 | Primary/empirical source -- exercises the EXACT interface `claude_code_invoke()` wraps | Direct Bash execution, verbatim JSON envelope captured | Live envelope's `modelUsage` had 2 keys: `"claude-haiku-4-5"` and `"claude-haiku-4-5-20251001"`, each carrying `canonicalModel: "claude-haiku-4-5"` and `provider: "firstParty"` -- **fields beyond what the TS-SDK doc (#4) lists**, i.e. the live schema is a superset of the documented one (see Recency/gap note below). This directly demonstrates the exact mechanism the brief recommends pyfinagent adopt: iterate `envelope["modelUsage"]` to log what ACTUALLY ran. |

## A2. Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|--------------------------|
| https://www.lowcode.agency/blog/claude-code-cli-commands-guide | 3rd-party blog | Search snippet only; official docs (#1) supersede |
| https://www.mindstudio.ai/blog/claude-code-headless-mode-autonomous-agents | 3rd-party blog | Search snippet only |
| https://backgroundclaude.com/cli-reference | 3rd-party mirror/reference | Search snippet only; not authoritative vs #1 |
| https://hidekazu-konishi.com/entry/claude_code_cicd_and_headless_automation.html | 3rd-party blog | Search snippet only |
| https://amux.io/guides/claude-code-headless/ | 3rd-party guide | Search snippet only |
| https://angelo-lima.fr/en/claude-code-cicd-headless-en/ | 3rd-party blog | Search snippet only |
| https://adrianomelo.com/posts/claude-code-headless.html | 3rd-party blog | Search snippet only |
| https://www.buildthisnow.com/blog/guide/development/claude-code-headless-mode | 3rd-party blog | Search snippet only |
| https://introl.com/blog/claude-code-cli-comprehensive-guide-2025 | 3rd-party blog | Search snippet only |
| https://www.eesel.ai/blog/claude-code-cli-reference | 3rd-party blog | Search snippet only |
| https://opentools.ai/resources/claude-code-cli-reference | 3rd-party mirror | Search snippet only |
| https://claudelog.com/faqs/what-is-output-format-in-claude-code/ | 3rd-party FAQ | Search snippet only |
| https://awesomeclaude.ai/code-cheatsheet | 3rd-party cheatsheet | Search snippet only |
| https://avasdream.com/blog/claude-cli-agentic-wrapper | 3rd-party blog | Search snippet, but informed the "wrapper best-practice" recency query -- discusses transient-subprocess-per-turn pattern (matches pyfinagent's own `claude_code_invoke` design) and warns of config-bloat-per-subprocess-call, a tangential but relevant operational note |
| https://github.com/joshrotenberg/claude-wrapper | 3rd-party wrapper library (Rust) | Search snippet only; "typed builder produces typed output" pattern noted but not adopted (out of scope for 78.2) |
| https://dev.to/jungjaehoon/why-claude-code-subagents-waste-50k-tokens-per-turn-and-how-to-fix-it-41ma | 3rd-party blog | Search snippet only |
| https://repovive.com/roadmaps/claude-code/power-workflows/headless-mode-cli-tool | 3rd-party guide | Search snippet only |
| https://github.com/openclaw/openclaw/issues/61093 | GitHub issue | Search snippet only; unrelated gateway registration bug, not this project's rail |
| https://claudefa.st/blog/guide/changelog | 3rd-party changelog mirror | Search snippet only |

## B. Recency scan (2026, last 2 years)

**Finding: the topic itself has no "canonical prior art" older than the
product.** Claude Code's `--model` flag / headless-mode JSON schema is an
implementation detail of a CLI whose model-alias system (fable/opus/
sonnet/haiku, `[1m]` suffixes, `opusplan`) is actively evolving through
2026 (source #2 embeds version-gated behavior notes up to v2.1.219,
i.e. weeks-old at research time). There is no older canonical source to
contrast against -- the year-less canonical query ("Claude Code CLI
--model headless subprocess wrapper best practice") surfaced only 2025-
2026 blog content (see snippet table), confirming the topic is
inherently current-year. This is the explicit "too new for year-less
prior art" case the research-gate rule anticipates; stated per that
rule rather than silently skipped.

**One genuine new-in-2026 finding**: the live envelope's `modelUsage`
entries (source #6) carry `canonicalModel` and `provider` fields that
are **not** documented in the TypeScript SDK's `ModelUsage` type
(source #4, `inputTokens`/`outputTokens`/`cacheReadInputTokens`/
`cacheCreationInputTokens`/`webSearchRequests`/`costUSD`/
`contextWindow`/`maxOutputTokens` only). This is a live-CLI-ahead-of-
docs drift (CLI v2.1.220, docs presumably synced to an earlier build) --
worth noting as an open gap (section H) rather than silently trusting
either source alone. Practically: `canonicalModel` is the MORE useful
field for pyfinagent's logging purpose than the dict key itself, since
the key can be a dated snapshot (`claude-haiku-4-5-20251001`) while
`canonicalModel` normalizes it back to the alias family
(`claude-haiku-4-5`) -- match pyfinagent's existing `model` label
convention in `llm_call_log`.

## C. Search queries run

1. **Current-year frontier**: `Claude Code CLI headless mode --model flag documentation 2026`
2. **Current-year frontier (schema-focused)**: `Claude Code CLI reference --output-format json modelUsage field`
3. **Current-year (error-behavior)**: `Claude Code --model flag unrecognized model error 2026`
4. **Year-less canonical**: `Claude Code CLI --model headless subprocess wrapper best practice`

No last-2-year-suffixed variant (`2025`) was run as a separate query
because #1 and #4 together already spanned 2025-2026 content in their
returned snippets (introl.com's guide is dated 2025, e.g.) -- see
Recency scan above for why a dedicated last-year query would be
redundant on this fast-moving, docs-current topic.

## D. Internal caller census

Re-derived and CORRECTED/EXTENDED vs the spawn prompt's anchors.

**`claude_code_invoke()` direct callers** (bypass `ClaudeCodeClient`, no
`model_name` ever available at the call site):
| Call site | Model available? | Notes |
|---|---|---|
| `backend/services/autonomous_loop.py:2454` (lite trader) | No | matches spawn anchor |
| `backend/services/autonomous_loop.py:2530` (lite risk judge) | No | matches spawn anchor |
| `backend/services/ticket_queue_processor.py:206` (ticket queue, `_spawn_real_agent`) | **Computed but discarded** -- `agent_model_map` (172-176) resolves `model_name` at line 177, but the `if getattr(settings,"paper_use_claude_code_route",False):` branch (199-217) calls `claude_code_invoke(task, system=system, timeout_s=60)` **without** `model_name` anywhere in the call. `model_name` is used ONLY in the direct-SDK fallthrough at `client.messages.create(model=model_name, ...)` (confirmed by reading lines 219+; the SDK client construction follows immediately after the CC-rail early-return at 217). This is the CONCRETE case the spawn prompt described: a model selection exists in-scope and is silently dropped on the rail path. |

**`ClaudeCodeClient` construction sites** (model IS available -- it's the
constructor arg -- but never reaches argv, only `self.model_name` used as
a BQ log label at `claude_code_client.py:607/626`):
| Call site | model_name value | Notes |
|---|---|---|
| `backend/agents/llm_client.py:2110` inside `make_client()` | whatever `model_name` was passed into `make_client(model_name, ...)` by the caller | THE fan-in point -- every C-block service below routes through here |
| `backend/services/analyst_narrative_scorer.py:144` | `model` (local var, not grepped further in this pass) | `make_client(model, None, settings)` |
| `backend/services/call_transcript_gpr.py:123` | `model` | `make_client(model, None, settings)` |
| `backend/services/meta_scorer.py:230` | `settings.meta_scorer_model` default `"claude-haiku-4-5"` | |
| `backend/services/macro_regime.py:515` | `settings.macro_regime_model` default `"claude-haiku-4-5"` | |
| `backend/services/news_screen.py:276` | `settings.news_screen_model` default `"claude-haiku-4-5"` | |
| `backend/services/pead_signal.py:288` | `settings.pead_signal_model` default `"claude-haiku-4-5"` | |

These 6 are the phase-78.1 C-block services rewired onto `make_client()`
-- confirmed as the exact same set named in
`test_phase_75_llm_rail.py:87-88`'s docstring ("Six production services
... pass pre-cleaned DICT schemas"), which independently corroborates
the caller list from a different angle (schema plumbing, not model
plumbing). This is the phase-78.1-blocks-on-78.2 link the spawn prompt
asserted: whatever request-model these 6 services compute (haiku-tier
by config default), the CC rail actually executes on the CLI session
default instead, per the argv trace below.

**Test-only constructions** (not production call sites, listed for
completeness of the denominator; verified via
`grep -rn "ClaudeCodeClient\|claude_code_invoke" backend/tests/`):
`test_claude_code_client.py`, `test_phase_56_2_ops_fixes.py`,
`test_phase_60_1_deep_pipeline.py`, `test_phase_60_4_observability.py`,
`test_phase_61_2_decision_integrity.py`, `test_phase_66_1_rail_guard.py`,
`test_phase_75_llm_rail.py`, `test_phase_78_1_c_block_rail.py`.

**Complete denominator: 3 direct `claude_code_invoke()` call sites + 6
`ClaudeCodeClient`-via-`make_client()` call sites = 9 production call
sites, all missing `--model` in the eventual argv.** This matches and
extends the spawn prompt's "three callers" framing -- the spawn prompt
under-counted by naming only the 3 direct-invoke sites; the 6 C-block
services are equally exposed because `ClaudeCodeClient.generate_content`
(`claude_code_client.py:517-592`, read in full) builds its own
`claude_code_invoke(...)` call at line ~592 and **that call also omits
`model_name`** -- confirmed by reading the full `generate_content` body
(lines 517-644): the kwargs passed to `claude_code_invoke` are `prompt`,
`system`, `json_schema`, `timeout_s` -- no `model`.

**`ticket_queue_processor.py:172-176` `agent_model_map`** -- measured
answer: used ONLY on the non-rail (direct-SDK) branch. On the rail
branch (`paper_use_claude_code_route=True`, the away-ops default per
`docs/runbooks/away-ops-rules.md`), the dict is computed and then never
referenced again in the function. Confirmed by `test_phase_56_2_ops_fixes.py:184-198`
(`test_ticket_agent_uses_cli_rail_when_route_flag_set`): the test mocks
`claude_code_invoke` and asserts it was called once, but never asserts
on any model argument -- the test itself has no assertion surface for
this defect, consistent with "nothing currently checks it."

**Does anything already log the RESOLVED model for a rail call?** No.
`_log_cc_call` (`claude_code_client.py:489-515`, read in full) takes
`model` as a caller-supplied kwarg and passes it straight to
`log_llm_call(model=model, ...)` -- both call sites (`:607`, `:626`)
pass `self.model_name`, i.e. the REQUEST label, not anything derived
from the envelope. The envelope's own `modelUsage` field (documented in
the `claude_code_invoke` docstring at `:247`, part of the CLI's own
`--output-format json` schema) is never read anywhere in this file or
in any caller (`grep -n "modelUsage" backend/ -r` -> zero matches
outside the one docstring line). This is the gap the spawn prompt's
question B6 was pointing at: the CLI likely already reports what model
it ran (see external section A, source #4), and pyfinagent throws that
information away.

## E. Test seam (argv-capture)

**Reusable seam, already exercised for a different flag** --
`backend/tests/test_phase_75_llm_rail.py:111-128`,
`test_json_schema_argv_flag_is_actually_emitted`:

```python
from backend.agents import claude_code_client as ccc

with patch.object(ccc.subprocess, "run") as mock_run:
    mock_run.return_value = SimpleNamespace(
        returncode=0,
        stdout=json.dumps({"subtype": "success", "result": "{}",
                            "is_error": False, "usage": {},
                            "stop_reason": "end_turn"}),
        stderr="",
    )
    ccc.claude_code_invoke("hello", json_schema={...})
    argv = mock_run.call_args[0][0]

assert "--json-schema" in argv
```

This is the exact seam a phase-78.2 test should reuse: patch
`ccc.subprocess.run`, call `claude_code_invoke(..., model=<x>)` (once
the param exists), inspect `mock_run.call_args[0][0]` (the argv list
passed to `subprocess.run`), and assert `"--model"` is present at
`argv[argv.index("--model") + 1] == <x>`. No second seam needs
inventing -- `test_phase_75_llm_rail.py` already imports `SimpleNamespace`,
`json`, `patch` at module scope (lines 32-38, read in full) so a new
test function can drop straight in. A parallel but less direct seam
exists at `test_claude_code_client.py:48-60`
(`test_claude_code_invoke_passes_prompt_via_stdin_not_argv`), which
patches `subprocess.run` at the `backend.agents.claude_code_client.subprocess.run`
path and inspects `mock_run.call_args` -- same pattern, older module,
prefer the phase-75 one since it is the CI file that already tests argv
shape for `--json-schema` in this exact function.

## F. Key findings

1. **`--model` accepts both aliases (`opus`/`sonnet`/`haiku`/`fable`) and
   full model ids (e.g. `claude-opus-4-8`).** No grammar restriction
   beyond that -- `claude_code_client.py` could pass either the CC-role's
   existing `model_name` (already a full id like `claude-opus-4-8`,
   `claude-sonnet-4-6`, `claude-haiku-4-5`) directly. (Source #1, #5.)

2. **Precedence when `--model` is omitted, highest to lowest**:
   (1) `/model` mid-session (N/A for one-shot `-p` calls) -> (2)
   `--model` flag at startup -> (3) `ANTHROPIC_MODEL` env var -> (4)
   `model` field in a settings file (`~/.claude/settings.json` for
   this project = `"opus[1m]"`, confirmed by reading the actual local
   file). Since pyfinagent's rail passes neither `--model` nor sets
   `ANTHROPIC_MODEL` in the scrubbed subprocess env
   (`claude_code_client.py:295-298`, read in full -- only
   `ANTHROPIC_API_KEY`/`ANTHROPIC_AUTH_TOKEN` are stripped, nothing is
   *added*), every rail call falls through to whatever `~/.claude/
   settings.json` has, i.e. **the interactive session's own pinned
   model** (`opus[1m]` on this machine right now) -- confirming the
   spawn prompt's central claim, with a citation trail. (Source #2.)

3. **Unknown/invalid `--model` value: NOT a hard error at launch.**
   The recognized-model validation Anthropic ships only applies to the
   in-session `/model` command and Agent-SDK `setModel()`, and
   explicitly does NOT cover `--model`, `ANTHROPIC_MODEL`, or the
   `model` setting. A typo'd value there produces "There's an issue
   with the selected model" **on the first request** -- i.e. surfaces
   as a normal API-style error, which `claude_code_invoke` already
   handles via its non-success-`subtype` exception path
   (`claude_code_client.py:362-370`). No new error-handling branch is
   needed for a bad `--model` value; the existing `ClaudeCodeError`
   raise covers it. (Source #2.)

4. **The JSON envelope DOES report the resolved model(s)** via
   `modelUsage: { [modelName: string]: ModelUsage }` in the `result`
   message -- confirmed both from the TypeScript SDK's type definition
   and from a live local probe. This is a dict, not a single string,
   because a turn can involve more than one model (e.g. this
   project's own live probe surfaced 2 keys for one simple call --
   see Recency scan). Each entry additionally carries `canonicalModel`
   (live-observed, undocumented in the TS-SDK page) which normalizes a
   dated snapshot id back to its alias family -- the right field to
   log as "resolved model" for continuity with pyfinagent's existing
   `model` label convention. (Source #4, #6.)

5. **The complete production caller denominator is 9, not 3**: the 3
   direct `claude_code_invoke()` sites the spawn prompt named, PLUS 6
   `ClaudeCodeClient`-via-`make_client()` sites (the phase-78.1
   C-block services), because `ClaudeCodeClient.generate_content`
   builds its own internal `claude_code_invoke(...)` call that ALSO
   omits `model_name`. All 9 are exposed to the same defect. (Section D.)

6. **One caller (`ticket_queue_processor.py:172-206`) computes a model
   selection and then discards it on the rail branch** -- concrete,
   already-in-repo evidence of exactly the risk pattern described,
   not merely a hypothetical.

7. **`ClaudeCodeClient.model_name` is currently write-only for
   observability, not read for routing** -- it is passed to `_log_cc_
   call(model=self.model_name, ...)` (claude_code_client.py:607,626)
   as the BQ log label, and NEVER read back to build argv. Threading
   it into argv is additive (the constructor already receives it),
   not a new plumbing problem.

## G. Recommendation: request-model vs resolved-model logging

**Do both, not either/or -- they answer different questions and the
CLI already gives us both for free once `--model` is threaded
through.**

- **Pass `--model <model_name>`** in `claude_code_invoke`'s argv
  whenever a `model` kwarg is supplied (new optional param,
  default `None` for backward compat with any caller that
  deliberately wants the session default -- though per finding #2 no
  current caller wants that, they just haven't been told). This closes
  the actual defect: the rail executes the REQUESTED model instead of
  silently riding the interactive session's `~/.claude/settings.json`
  pin.
- **Log the RESOLVED model from `envelope["modelUsage"]`**, not just
  the request label, in `_log_cc_call`. Concretely: iterate
  `envelope.get("modelUsage", {})`, and for each entry read its
  `canonicalModel` (falling back to the dict key if `canonicalModel`
  is absent -- defensive, since it's undocumented and could be
  removed). If multiple canonical models appear in one call (observed
  live), log the one with the largest `outputTokens` as the "primary"
  model, and optionally keep the full per-model breakdown in a JSON
  side-field for audit -- this is the only way to catch a FUTURE
  silent-fallback (e.g., Anthropic's own automatic-model-fallback for
  safety-classifier flags, source #2 "Automatic model fallback"
  section) even after `--model` is correctly threaded. Request-model
  and resolved-model matching (or mismatching) is itself a valuable
  signal; don't collapse them into one field.
- This also fixes the `llm_call_log` mislabeling problem named in the
  spawn prompt's anchor 2 ("the log may misleadingly say e.g.
  `claude-haiku-4-5` while the CLI ran something else") for free, since
  resolved-model logging is authoritative regardless of whether the
  request-model plumbing has a bug.

## H. Open gaps

- The `canonicalModel`/`provider` fields observed live (source #6) are
  NOT in the TypeScript SDK's documented `ModelUsage` type (source
  #4) as of this research session -- docs may be lagging the shipped
  CLI (v2.1.220) by a small margin, or these are intentionally
  undocumented/internal fields not meant for consumer reliance. GENERATE
  should treat `canonicalModel` as best-effort (guard with `.get()`,
  never `KeyError`) rather than a load-bearing documented contract.
- Not independently re-verified: whether `ANTHROPIC_MODEL` being unset
  in the scrubbed env (vs actively cleared) matters -- the scrub at
  `claude_code_client.py:295-298` only removes `ANTHROPIC_API_KEY`/
  `ANTHROPIC_AUTH_TOKEN`; if the parent process (backend/uvicorn) never
  had `ANTHROPIC_MODEL` set in its own env to begin with, this is moot,
  but GENERATE should confirm `backend/.env` doesn't set `ANTHROPIC_
  MODEL` (researcher sandbox is denied `backend/.env` per prior-session
  memory; Main should grep it directly).
- This brief does not cover WHERE `analyst_narrative_scorer.py:144` and
  `call_transcript_gpr.py:123`'s local `model` variable is defined
  upstream (only that it's passed into `make_client(model, None,
  settings)`) -- GENERATE should trace those two call sites' `model`
  variable definitions before writing the fix, since the C-block
  services aren't uniformly structured (4 use `getattr(settings, ...,
  "claude-haiku-4-5")` inline, 2 use a bare `model` local).

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6: cli-reference, model-config, headless, agent-sdk/typescript, local --help, live probe)
- [x] 10+ unique URLs total (incl. snippet-only) -- 6 read-in-full + 19 snippet-only = 25
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not just landing/nav) for the read-in-full set
- [x] file:line anchors for every internal claim

## JSON envelope

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 19,
  "urls_collected": 25,
  "recency_scan_performed": true,
  "internal_files_inspected": 13,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "Confirmed via official Claude Code docs + local --help + a live claude -p probe: --model accepts aliases or full ids and is omitted nowhere by design -- when absent it falls through /model (n/a) -> --model (n/a) -> ANTHROPIC_MODEL env (unset by pyfinagent) -> settings.json 'model' key, i.e. the rail silently rides the interactive session's own pinned model, confirming the spawn prompt's defect. An invalid --model value is NOT validated at launch; it surfaces as a normal API error on first request, already handled by claude_code_invoke's existing non-success-subtype exception path. The JSON envelope's result message DOES report the resolved model(s) via modelUsage: {[modelName]: ModelUsage}, each entry additionally carrying an undocumented-but-live canonicalModel field -- the right value to log as 'what actually ran'. Complete production caller denominator is 9 (3 direct claude_code_invoke sites + 6 ClaudeCodeClient/make_client sites via the phase-78.1 C-block services), not the 3 named in the spawn prompt, because ClaudeCodeClient.generate_content's internal claude_code_invoke call also omits model. ticket_queue_processor.py:172-206 computes then discards a model selection on the rail branch -- concrete in-repo evidence of the exact risk. Reusable argv-capture test seam: test_phase_75_llm_rail.py:111-128 patches ccc.subprocess.run and inspects mock_run.call_args[0][0]. Recommendation: thread request-model into argv AND separately log resolved-model from envelope['modelUsage'] -- they answer different questions and together also catch Anthropic's own safety-fallback substitutions.",
  "brief_path": "handoff/current/research_brief_78.2.md",
  "gate_passed": true
}
```
