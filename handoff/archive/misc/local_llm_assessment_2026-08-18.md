# Ollama on the Mac mini — re-assessment (2026-08-18)

Supersedes nothing; **updates** `handoff/archive/misc/local_llm_assessment_2026-07-18.md`
with one month of new evidence. The July verdict still holds. What changed is the
*strength* of the reliability argument (up, sharply) and the *ranking* of the
pilot surfaces (Slack bot demoted).

## Short answer

**Possible: yes, and already planned.** The work is sitting in
`.claude/masterplan.json` as **phase-74.0 → 74.3, all `status: pending`**, with
immutable `live_check`s already written. Nothing has been built — no Ollama
reference exists anywhere in `backend/`.

**Beneficial: for reliability, yes — more than it was a month ago. For cost, no.**

## What changed since 2026-07-18 (all of it strengthens the reliability case)

**1. The rail died again, today.** Commit `4f90a2f` (2026-08-18 14:44 +0200):
weekly Max budget exhausted, resets **Aug 20 19:00 Europe/Oslo**. With
`PAPER_USE_CLAUDE_CODE_ROUTE=true` both fallthrough paths hard-raise
`ValueError('Routing breach...')` (`llm_client.py` `make_client` / `advisor_call`)
— money is safe by design, but the Claude-dependent legs produce **no real
analysis until the reset**. That is a ~53-hour analytical blackout, and it is the
*second* distinct exhaustion class after the credit-exhaustion that the July doc
records as having "cost you two months."

**2. The breaker can't see it.** Step **86.120** (`pending`) is now a live
condition rather than a prediction: the CC rail circuit breaker is not
weekly-limit-aware — the guard resets every cycle, opens only after ~20
consecutive failures, and the health probe checks *login*, not *quota*. So the
analyst spawns doomed subprocesses every cycle for two days.

A local model does not fix 86.120 — fix that separately, it's cheaper. But a
local model is the only rail on the box whose capacity **cannot be exhausted by
anyone else's usage**, which is the property both outages lacked.

**3. The Gemini retirement clock is now ~2 months out**, not ~3
(`model_tiers.py:49,57`; `settings.py:204` — 2.5 family EOL **2026-10-16**, with
`test_phase_75_5_2_model_pins.py` set to go red on purpose). The single healthy
leg has a dated expiry.

## What got weaker

The July doc ranked the **Slack bot as pilot candidate #1** largely because it sat
on the same credit-dead Anthropic path as the trading rail — "degraded-or-billing-dead
anyway", so zero downside. With credits available that premise is gone: the bot
works, and replacing a working Sonnet-4.6 conversational brain
(`model_tiers.py:98`, effort `low` at `:326`) with a 4B model is a **real quality
regression** for a surface you actually talk to. It also stresses exactly the 4B
weak spot the July doc flagged — MCP tool-calling.

**Demote the Slack bot to optional.** Re-ranked candidates:

1. **Terminal fail-forward rail** (was #6) — the third rail after cc_rail → Vertex.
   Now the whole point. Beats the fabricated-HOLD / flat-`conviction=10` failure modes.
2. **news_screen overlay** — mechanical extraction, 1/day, failure-tolerant. Best
   *proving ground* for schema-validity before trusting the rail.
3. **macro_regime overlay** — bounded classification, cached, graceful `unknown`.
4. Reflections / meta-scorer-as-last-resort — unchanged.
5. ~~Slack bot~~ — optional convenience, no longer the lead.

Never-local, unchanged: enrichment, debate, synthesis, risk-judge, autoresearch,
the Layer-3 harness.

## The one code defect that blocks all of it

`backend/agents/llm_client.py` — **re-verified today, still present**:

```python
if (mime == "application/json" or schema) and not self._base_url:
    kwargs["response_format"] = {"type": "json_object"}
```

Any `base_url` client (which is how Ollama would attach, copying the GitHub-Models
branch shape) **silently drops schema enforcement**. Ollama enforces JSON Schema as
GBNF grammar and is *guaranteed*-valid — strictly better than what the local path
would get today. Shipping a local rail without fixing this recreates the 61.2
fabricated-HOLD failure mode. This is step **74.1**, and it must land before 74.2/74.3.

Second trap, also still live: `cost_tracker.py:177` falls back to `_DEFAULT_PRICING`
for unknown model ids — an unpriced local id books **phantom** spend. 74.1 adds the
`($0, $0)` row.

## Cost: still a false economy — say it plainly

The July measurement stands: the healthy Gemini leg runs ~**$0.2/day (~$73/yr)**.
There is nothing meaningful to save. If cost is the motive, don't do this. The
motive that survives scrutiny is **availability**.

## Recommendation

Do **74.0 + 74.1 only**, then stop and look.

- **74.0** — `brew install ollama`, pull Qwen3-4B-Instruct-2507 Q4, launchd service
  with `keep_alive=0` and a refuse-under-2 GB-free memory guard. Requires your
  approval (system change).
- **74.1** — the plumbing: `make_client()` localhost branch, **the `base_url`
  schema-skip fix**, the `($0,$0)` pricing row.

That buys the capability and fixes a real latent defect, at low risk, flag-dark.
Then prove it on **news_screen** (74.3's first half) before wiring the terminal
rail. Leave the Slack bot (74.2) alone unless you want it for its own sake — and
if you do run it, note the masterplan text for 74.2 still carries the July
"credit-dead" rationale, which no longer applies.

**Do not** localize any scoring role on this hardware, ever. Do not put an 8B on
this box.

## Caveats on this assessment

- Written from the remote container; the Mac's RAM headroom (**16 GB total /
  13.7 GB resident → 2–6 GB inference budget**) is the **2026-07-18 measurement**,
  not re-measured today. Re-measure with `vm_stat` before 74.0.
- The quality-cliff numbers (FAITH `arXiv:2508.05201`, FinBen `2402.12659`) are
  carried from the July research gate and were not re-fetched.
- Code seams (`llm_client.py` schema-skip, `cost_tracker.py:177`, `model_tiers.py:98,326`)
  **were** re-verified against the working tree today.
