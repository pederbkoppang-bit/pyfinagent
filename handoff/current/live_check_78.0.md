# live_check 78.0 — verbatim evidence (2026-07-25)

## 1. Criterion 2 — audit-class gate with `coverage.dry = true`

**Labelling correction (78.0 Q/A):** this block was originally headed "verbatim envelope". It is not verbatim — it is an EXCERPT with the long `summary` and `brief_path` keys elided. Every numeric/boolean field below matches the envelope in `research_brief_78.0.md` exactly, and the Q/A independently confirmed the envelope in the brief itself; but a capture labelled verbatim must be regenerated rather than edited, so it is relabelled here as the excerpt it is.

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 12,
  "urls_collected": 18,
  "recency_scan_performed": true,
  "internal_files_inspected": 45,
  "coverage": {
    "audit_class": true,
    "rounds": 9,
    "dry_rounds": 2,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": true
  },
  "gate_passed": true
}
```

Loop-until-dry is real, not asserted: rounds 8 (async/constructor/beta sweep) and 9
(frontend + curl-in-shell + GitHub-catalog lens) each returned **0 new call sites**,
which is what flipped `dry`. Round 2 (the `ClaudeClient(` direct-construction sweep)
is the round that earned the audit-class treatment — it surfaced **4 sites the seed
list did not have**, all of which bypass `make_client` and therefore never see the
CC-rail flag. A fixed-list census would have missed exactly those.

## 2. Criterion 1 — anchors RE-DERIVED mechanically (not copied, not re-read)

A script opened each claimed `file:line` and asserted the expected symbol was present,
reporting the actual line and any drift. **42 anchors, 42 resolved.** Excerpt:

```
A1    backend/agents/orchestrator.py              :652   OK@652    self.general_client: LLMClient = make_client(settings.gemini_model, ...
A5    backend/agents/orchestrator.py              :1043  OK@1043   bc = BatchClient()
C1    backend/services/meta_scorer.py             :221   OK@221    client = ClaudeClient(
C6    backend/services/call_transcript_gpr.py     :113   OK@113    client = ClaudeClient(
D1    backend/agents/multi_agent_orchestrator.py  :1099  OK@1099   response = client.messages.create(
D4    backend/slack_bot/streaming_integration.py  :526   OK@527 (drift +1)   resp = client.messages.create(
D5b   backend/agents/openclaw_client.py           :48    OK@48     "communication": "anthropic/claude-sonnet-4-6",
F1    backend/meta_evolution/directive_review.py  :139   OK@138 (drift -1)   resp = client.messages.create(
G1    backend/agents/planner_agent.py             :166   OK@166    response = self.client.messages.create(
H1    backend/agents/rag_agent_runtime.py         :259   OK@259    response = client.beta.messages.create(**kwargs)
I1    scripts/autoresearch/run_memo.py            :273   OK@273    "FAST_LLM": f"anthropic:{resolve_model('autoresearch_fast')}",
K1    backend/agents/claude_code_client.py        :215   OK@215    def claude_code_invoke(
```

Drift found: `D4` +1, `F1`/`F2`/`F3` −1, `J1` +1. All other anchors landed exactly.

**Correction (78.0 Q/A cycle-1):** the drift was *detected* in this run but four of the five
corrections were never propagated into the deliverables — `census_78.json`, `census_78.md` and
the queued steps 78.3/78.8 all still carried the pre-drift lines while this file claimed they
had been corrected. They are now applied everywhere (the corrected rows carry a
`[drift-corrected...]` marker in the census) and the stale-anchor grep over the masterplan
returns 0. `J1`'s `:66-71` was already correct as a range. `K1b` is a RANGE claim (`llm_client.py:2099-2113`) and verifies as a
range: `ClaudeCodeClient` occurs at 2104 and 2110, inside the cited span.

**Honest disclosure about this run:** the first pass reported 15 "failures". Every one
was a defect in *my verification script's* path/needle guesses (`llm_client.py` is
`backend/agents/llm_client.py`; `news/sentiment.py` is `backend/news/sentiment.py`;
the C-block rows anchor the `ClaudeClient(` construction, not a `generate_content`
call), **not** an error in the brief. Corrected batch: 16/17, the 17th being the range
claim above. The brief's anchors were accurate.

## 3. Criterion 3 — volumes MEASURED (verbatim 30d query + output)

```sql
SELECT provider, model, agent, COUNT(*) AS calls,
       SUM(input_tok + output_tok) AS tokens,
       COUNTIF(NOT ok) AS failed, MAX(ts) AS last_seen
FROM `sunny-might-477607-p8.pyfinagent_data.llm_call_log`
WHERE ts >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
GROUP BY provider, model, agent ORDER BY calls DESC
```

```
provider     model                      agent                calls     tokens  fail  last_seen
anthropic    claude-sonnet-4-6          cc_rail               2192    4370458  1547  2026-07-24 18:34:17
anthropic    claude-opus-4-7            cc_rail                357     500651   294  2026-07-24 18:34:09
gemini       gemini-2.5-flash           -                      226     232090     0  2026-07-24 18:35:46
claude-code  claude-code-cli            lite_trader              9      20997     4  2026-07-24 18:04:01
claude-code  claude-code-cli            lite_risk_judge          9      28840     3  2026-07-24 18:04:56
anthropic    claude-sonnet-4-6          cc_rail:drill_66_1       7          0     7  2026-07-06 22:27:54
anthropic    claude-haiku-4-5-20251001  -                        3       3150     0  2026-07-07 09:48:08

TOTAL GROUPS: 7
```

Two things worth the operator's attention, both measured here rather than inferred:

- **The bare-`cc_rail` shape dominates**: 2,192 + 357 = 2,549 calls carry agent
  EXACTLY `cc_rail` (no colon), versus 7 rows in the `cc_rail:<agent>` shape. This is
  the 75.5.12 P1 defect's blast radius, re-measured on a fresh 30d window. (The goal
  text cites 2,241/4.1M from the 2026-07-24 window; the window has since slid — the
  numbers above are today's measurement, not a restatement of the goal's.)
- **70% of rail calls are failing**: 1,547 of 2,192 sonnet rows (70.6%) and 294 of 357
  opus-4-7 rows (82.4%) have `ok = false`. That is not a routing question and is out
  of scope for this READ-ONLY census — queued as **78.9** rather than disclosed only
  in prose. 78.9 deliberately makes its FIRST task establishing what `ok=false`
  actually means on this write path (genuine failure vs. retried-then-succeeded),
  because the severity of the finding depends entirely on that distinction and this
  census did not establish it.

## 4. Instrumentation audit — why "0 rows" ≠ "0 calls" (criterion 3, honest-unmeasurable half)

Per-site check for whether the call site writes an `llm_call_log` row at all. **11 of the 12**
raw-SDK sites are uninstrumented — DERIVED from the census's own `instrumented` field over the
denominator {A4, A5, A6, D1-D4, F1-F3, G1, H1}, only A4 being instrumented. (An earlier revision
of this file and of step 78.8 said "9 of 12"; that number did not reproduce and is corrected
here. The error understated the blindness, so the conclusion is unchanged.)

```
site                         instrumented?    evidence
A4 advisor_call              YES              L2321: from ...observability import log_llm_call
A5 BatchClient               NO (dark)        no log_llm_call in llm_client.py:1931-2050
A6 HaikuScorer               NO (dark)        no log_llm_call in news/sentiment.py:760-860
B1/B2 lite (rail branch)     YES              L2462: _log_claude_code_call(envelope, agent="lite_trader", ...)
D1-D3 MAS                    NO (dark)        no log_llm_call in multi_agent_orchestrator.py:1050-1350
D4 slack leak                NO (dark)        no log_llm_call in streaming_integration.py:490-570
E1 ticket queue              NO (dark)        no log_llm_call in ticket_queue_processor.py:180-300
F1 directive_review          NO (dark)        no log_llm_call in directive_review.py:100-200
F2 directive_rewriter        NO (dark)        no log_llm_call in directive_rewriter.py:140-240
F3 skill_mod_review          NO (dark)        no log_llm_call in skill_modification_review.py:160-260
G1 planner_agent             NO (dark)        no log_llm_call in planner_agent.py:60-320
H1 rag multimodal            NO (dark)        no log_llm_call in rag_agent_runtime.py:200-300
```

Wrapper clients ARE instrumented — `ClaudeClient` (llm_client.py:1887),
`GeminiClient` (:1123), `OpenAIClient` (:1275), `ClaudeCodeClient`
(claude_code_client.py:498). **Instrumentation follows the rail and the wrappers, not
the raw-SDK metered path.**

Two consequences the census encodes:

1. **CORRECTED — the C-block's 0 rows is NOT a genuine zero.** This file originally
   argued that because `ClaudeClient` is instrumented, C1–C6's 0 rows in 30d proved
   those overlays never ran. **That inference is unsound**, and the 78.0 Q/A was right
   to reject it: `ClaudeClient` hardcodes `ok=True` at its log site
   (`llm_client.py:~1905`), `llm_client.py` contains **zero** `ok=False` writers, and
   SDK errors re-raise at `:1739`/`:1746`/`:1790` — *before* the log block at `:1886`.
   So 0 rows means **"no call SUCCEEDED"**, which is equally consistent with "ran every
   cycle and failed every time" — precisely the dead-credits outage the C-block's own
   reason field cites. Verified first-hand: `grep -n "ok=False" backend/agents/llm_client.py`
   returns nothing, while the CC rail *does* log failures
   (`claude_code_client.py:607`, `autonomous_loop.py:2469`/`:2545`) — which is exactly
   why the rail shows 1,547 failures and the wrapper clients show none. The census rows
   now say so, and **78.1 carries an explicit instruction not to assume the six are
   dormant**, because a dormant rewire and an actively-failing rewire are different jobs.
   For **D1–D4 / F1–F3 / G1 / H1 / A5 / A6**, 0 rows means **unmeasurable** for the
   separate reason that they write no row at all.
2. Any spend metric built on `llm_call_log` — `fetch_llm_spend`, the $25/day
   breaker — is structurally **blind to exactly the metered raw-SDK paths**. On the
   dual-rail sites this is sharp: the rail branch logs, the `else:` metered branch
   does not (`autonomous_loop.py` rail :2462 logs / direct :2472 does not;
   `ticket_queue_processor.py` rail :206 logs / direct :226 does not). The spend the
   operator wants governed is the spend the meter cannot see. Queued as **78.8**.

## 5. Criterion 4 — advisor_call + BatchClient stay_metered, evidence cited

- **A4 advisor_call** — `llm_client.py:2191`, beta call at `:2273` with
  `betas=["advisor-tool-2026-03-01"]`. Server-side beta *tool-use*; the `claude -p`
  CLI exposes no advisor tool. Already **hard-raises** under
  `paper_use_claude_code_route` (`:2233-2240`) — it fails loud rather than silently
  falling back to metered. DARK today (`enable_advisor_tool=False`, settings.py:391).
- **A5 BatchClient** — `llm_client.py:1931`, `messages.batches.create` at `:1978`. The
  Batches API is a 50%-discount / 24h-window product with **no CLI equivalent**;
  railing it would *cost* money. Also carries a latent no-args `TypeError`
  (`__init__(model_name, api_key)` has no defaults) behind a doubly-dark flag →
  **78.5 fix-or-retire**.

## 6. Criteria 3+5 — immutable command + ownership

```
$ python3 -c "import json; c=json.load(open('handoff/current/census_78.json')); assert len(c['roles'])>=12, ...; assert all(r.get('decision') in ('max_rail_cli','max_rail_proxy','stay_metered') for r in c['roles']), 'undecided rows'"
exit=0
```

**28 roles** — max_rail_cli 19, max_rail_proxy 1, stay_metered 8. Every row carries a
decision, a reason, and an owning follow-up step (criterion 5); the full table is
`handoff/current/census_78.md`, generated from `census_78.json` so the two cannot
drift.
