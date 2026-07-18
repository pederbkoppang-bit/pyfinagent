# Assessment: local LLM on the Mac mini — verdict + shortlist + pilot plan

## Context

You asked whether downloading the best available local LLM onto this Mac mini (which runs the app) is a good idea for the app's LLM tasks, with motives of full replacement + rail reliability + cost, and no hardware upgrade on the table. Assessed via ultracode recon `wf_1ce61081-07f` (internal role census + integration audit ∥ external July-2026 model landscape ∥ synthesis verdict at effort max), grounded in measured hardware and the verified phase-72/73 artifacts.

## Verdict: CONDITIONAL — BAD as asked, GOOD in one narrow form

**"Replace ALL LLM tasks with a local model" is a bad idea on this box, on three independent grounds:**

1. **Hardware**: base M4 mini, 16 GB soldered (upgrade = new machine), with **13.7 GB already resident** from the app stack → real inference budget ≈ 2–6 GB → the "best available" that fits is a **4B model at Q4** (~2.5–3.5 GB). 8B (~5.5 GB) is marginal with OOM/swap risk; 32B/70B are physically impossible.
2. **Quality cliff (measured, not vibes)**: FAITH (ICAIF'25) financial tabular reasoning — Qwen-3-**8B** scores 30.6%, Llama-3.1-8B 47.5% vs Gemini-2.5-Pro 91.9% / Claude-Sonnet-4 95.6%, and ~0–10% on multivariate financial calculation. A 4B at Q4 is below that. Our judgment roles (debate, synthesis/final BUY-SELL-HOLD, risk veto, conviction ranking) are exactly where this cliff is catastrophic.
3. **Mechanics**: the pipeline fires ~18–39 mostly-serial calls per analysis — a single ~40 tok/s local stream blows every cycle budget (the cc_rail's 120s timeouts already proved this failure mode); and the codebase's only local-compatible path (`OpenAIClient` with `base_url`, `llm_client.py:1116-1138`) **skips schema enforcement** (`:1200-1202`) — you'd lose the grammar-enforced JSON (`response_schema=SynthesisReport/RiskJudgeVerdict`) that keeps parsers alive, recreating the 61.2 fabricated-HOLD failure mode.

**Cost motive is a false economy**: the healthy Gemini leg costs ~$0.2/day (~$73/yr) with zero failures all window. There is nothing meaningful to save.

**The GOOD version — a last-resort third rail + overlay roles**: a small local model that can never be credit-exhausted has genuine reliability value — its answer beats the fabricated HOLD/0.0 and the credit-dead flat conviction=10 that cost you two months, and it hedges the Gemini-2.5 retirement cliff (2026-10-16). Per-role: enrichment/debate/synthesis/risk-judge/autoresearch/MAS-harness = **never-local**; meta-scorer = **local-as-last-resort only**; macro_regime + news_screen + reflections = **acceptable local candidates** (low-volume, latency-tolerant, failure-tolerant, off the critical path).

## Model shortlist (fits this box today)

| Model | RAM @Q4 | M4 speed | Notes |
|---|---|---|---|
| **Qwen3-4B-Instruct-2507** (headline pick) | ~2.5-3.5 GB | ~38-48 tok/s | Best quality-per-byte; excellent JSON via Ollama grammar; card scores partially disputed by reproductions — treat as extraction/rail class, never a scorer |
| **Phi-4-mini (3.8B)** | ~3-4 GB | ~38-48 tok/s | Strongest math per byte (GSM8K 88.6); pick if the task is arithmetic-shaped |
| Gemma-3-4B | ~4 GB | ~35-45 tok/s | Only if vision/multilingual needed; weaker reasoning |
| Qwen3-8B / Llama-3.1-8B | ~5.5 GB | ~28-35 tok/s | NOT recommended here — marginal quality gain, double RAM, biggest OOM risk on the shared box |

**Stack**: Ollama (OpenAI-compatible `/v1`, JSON-schema→GBNF grammar enforcement guaranteed-valid, `keep_alive=0` auto-unload — mandatory on this box). MLX is faster but lacks built-in grammar.

## "Some of them" — the ranked local-candidate list (answering the follow-up)

Yes — partial localization is exactly the defensible version. Ranked by fit (low money-risk × low volume × latency-tolerant × currently-broken-or-dark):

1. **Slack chat bot — the BEST candidate** (newly assessed): its conversational brain is `claude-sonnet-4-6` via the `mas_communication` tier (`assistant_handler.py:345`, `model_tiers.py:57`) + a `claude-haiku-4-5` call (`:446`) — i.e. it sits on the **same credit-dead Anthropic direct path** that killed the trading rail, so today it is degraded-or-billing-dead anyway. Zero money risk (a mediocre chat answer ≠ a bad trade), low volume (operator chats), tolerable latency at 4B speeds (~40 tok/s ⇒ a 200-token reply in ~5s). Caveat: the bot uses MCP tools (`mcp_tools.py`) — 4B tool-calling is the weak spot; Qwen3-4B supports tool calls but expect occasional misses; acceptable for a chat convenience surface, and the effort tier is already "low".
2. **news_screen overlay** — mechanical headline extraction, 1/day, dark, failure-tolerant (2-attempt retry + skip).
3. **macro_regime overlay** — bounded classification 1/day cached, graceful `unknown` fallback, dark.
4. **Reflections (learn-loop, post-73.2 build)** — post-close free text, off the decision path, no schema fragility.
5. **Meta-scorer** — last-resort fallback ONLY (beats the flat conviction=10 it collapses to today; never primary — ranking is the #1 quality-sensitive role).
6. **Terminal fail-forward rail** — the local model as the third rail after cc_rail → Vertex, beating the fabricated-HOLD failure mode.

Still never-local: enrichment (volume×latency), debate/synthesis/risk-judge (judgment + enforced schema), autoresearch (deep synthesis), the MAS harness (Opus-class agentic work).

## Recommended implementation (if approved): a two-step pilot queued for cheaper executors

Append to the masterplan as **phase-74 "local last-resort rail pilot"** (pattern-identical to phases 72-73: executor-tagged, flag-dark, immutable live_checks):

- **74.0 [operator + sonnet-4.6/high]**: Install Ollama (`brew install ollama` — operator approves the system change), pull Qwen3-4B-Instruct-2507 Q4, launchd service with `keep_alive=0` + a memory guard (refuse inference if free RAM < 2 GB). Live_check: `/v1` responds; RAM released post-call.
- **74.1 [sonnet-4.6/high]**: Wire the local provider plumbing once: `make_client()` localhost branch (copy the GitHub-Models branch shape, `llm_client.py:2030-2039`); **fix the `base_url` schema-skip** (`llm_client.py:1200-1202`) to pass `json_schema` through (Ollama enforces it as grammar); `($0,$0)` MODEL_PRICING row (`cost_tracker.py:20-83` — else an unknown id books phantom $0.10/$0.40).
- **74.2 [sonnet-4.6/high] — SLACK BOT pilot (the user-suggested surface, ranked #1)**: repoint `mas_communication` (and the `assistant_handler.py:446` haiku call) to the local model behind a flag; keep tool-calling with a graceful "couldn't run that tool" reply on miss. Live_check: a real Slack conversation transcript on the local model + RAM released after replies. This removes the bot's dependency on the dead Anthropic key entirely.
- **74.3 [sonnet-4.6/high, optional second wave]**: `news_screen_model` → local (mechanical proving ground) + the terminal fail-forward slot after the 72.0.2 Vertex leg (`llm_client.py:1983-2044`). Flag-dark, byte-identical OFF.
- **Success bar** (~10 idle-time runs / a week of bot chat): ≥95% schema-valid outputs where schemas apply, no app-stack memory pressure (guard refuses inference under 2 GB free), bot replies subjectively acceptable to the operator. Meta-scorer local fallback only after this bar passes. Wider pipeline localization stays rejected on this hardware.

**Explicitly NOT recommended**: any pipeline scoring role, ever, on this hardware; an 8B model on this box; framing this as cost savings.

## Verification

- Verdict numbers trace to: measured `sysctl`/`vm_stat` (16 GB / 13.7 GB RSS), FAITH arXiv:2508.05201, FinBen 2402.12659, modelfit.io M4 tok/s, Ollama structured-output docs; internal seams re-verified at `llm_client.py:962-963` (Gemini schema), `:1200-1202` (base_url skip, confirmed), `:1983-2044` (fallback order), `cost_tracker.py:83` (default pricing trap).
- Pilot verification is built into the queued steps' immutable live_checks (schema-validity run + RAM-release evidence).
