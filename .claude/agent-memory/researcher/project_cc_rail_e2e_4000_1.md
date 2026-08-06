---
name: cc-rail-e2e-4000-1
description: Phase-4000.1 CC-rail findings -- claude -p flat-fee is a PAUSED policy not a standing fact; modelUsage canonicalModel collapse; rail marker is THREE shapes; PUT /api/settings writes .env
metadata:
  type: project
---

Measured 2026-08-06 for masterplan 4000.1 (prove app Claude traffic runs on the
Claude Code Max rail). Five findings that are NOT derivable from reading the
code, and one that contradicts the code's own comments.

**1. The flat-fee premise is a PAUSED policy, not a standing fact.**
Anthropic announced then paused (dated notice "Update June 15" on
https://support.claude.com/en/articles/15036540-use-the-claude-agent-sdk-with-your-claude-plan)
moving `claude -p` -- exactly what `claude_code_client.py` shells -- onto a
capped monthly credit (Pro $20 / Max 5x $100 / Max 20x $200, billed at API list
rates). Secondary trackers still report the credit as LIVE; they are stale.
**Why:** every phase-4000 cost argument rests on "$0 flat fee". That is true
today and reversible by a vendor decision with no code change and no log line.
**How to apply:** never write "the Max plan is flat-fee" as a standing fact --
date it, cite the pause, and express quota math BOTH as %-of-weekly-pool and as
%-of-$100-credit. Re-check the URL before any step that assumes $0.

**2. `--bare` is documented to become the `-p` DEFAULT, and it disables
subscription auth.** The rail deliberately omits `--bare`
(`claude_code_client.py:364-369`) and scrubs `ANTHROPIC_API_KEY`
(`:408-411`), so when the default flips the rail has NO credential at all.
**Why:** a latent, vendor-scheduled, total-rail-failure mode.
**How to apply:** record `claude --version` in any rail baseline so a future
breakage bisects to a CLI upgrade rather than to our code.

**3. `modelUsage` keys can SHARE one `canonicalModel`, and
`resolve_rail_model` collapses them (last-wins).** Measured: one probe returned
both `claude-haiku-4-5` and `claude-haiku-4-5-20251001`, both with
`canonicalModel: "claude-haiku-4-5"`. The dict build at
`claude_code_client.py:274-277` overwrites, so `max(..., key=_weight)` at `:291`
chooses from a map that already lost an entry. The LABEL survives correct; any
COST/TOKEN total taken from the collapsed map under-counts.
Also: `total_cost_usd` == the sum over ALL raw entries (first key alone was 59%
of it) -- a free arithmetic assertion for any envelope test.
**Why:** this is a second-generation instance of the phase-78 defect class --
78.2 fixed *which key you pick*; this is *how many keys survive to be picked
from*.
**How to apply:** sum the RAW `modelUsage` map for any burn math; only use
`resolve_rail_model` for the label.

**4. Per-call overhead is ~45.6K cache-creation tokens for a 9-token prompt.**
Because the rail omits `--bare`, `claude -p` loads CLAUDE.md, hooks, skills,
MCP servers and auto-memory from the inherited cwd on EVERY call. Subscription
cache lifetime is one hour, so calls spaced across cycles re-pay it.
**Why:** any burn estimate derived from prompt length is wrong by an order of
magnitude; the fixed overhead dominates.
**How to apply:** record the backend's cwd alongside its start time; estimate
from measured envelopes, never from prompt size.

**5. The CC-rail llm_call_log marker is THREE row shapes, not one.**
`provider = 'claude-code' OR agent = 'cc_rail' OR agent LIKE 'cc_rail:%'` --
the exact complement of the exclusion at `spend.py:228-230`. `provider` alone
cannot separate the rails (two of three writers use `provider='anthropic'`,
same as the metered SDK client), and the bare-`cc_rail` shape is DOMINANT
(2,549 vs 7 in 30d). Never simplify to `cc_rail%` -- `spend.py:37-38` records
the exact `!=` was chosen on purpose.
**How to apply:** any "did the rail run?" query uses all three shapes.

**6. `PUT /api/settings` WRITES `backend/.env`.** `_update_env_var` at
`settings_api.py:316-331`. So "flip via the API, never edit .env" is subtler
than it reads: the API edits .env too -- the real difference is that the API
path also runs `get_settings.cache_clear()`, and a manual edit produces exactly
the "lru_cache desync" named in the routing-breach error at
`llm_client.py:2155-2156`. Consequence: **a flag flip survives a backend
restart**, so a crashed smoke leaves the rail ON permanently.
GET is cached 300 s (`api_cache.py:136`) -- a "live value" read can be stale.

**7. `financial_reports.paper_trades.created_at` is STRING, not TIMESTAMP**
(`scripts/migrations/migrate_paper_trading.py:68`), and `save_paper_trade`
drops None-valued columns. A `paper_round_trips` table schema already exists
(`scripts/migrations/add_round_trip_schema.py:66-75`) with `exit_date`
TIMESTAMP -- check whether it is populated BEFORE inventing a pairing rule on
paper_trades. The two halves then use different date columns/types, which
[[normalization-rule-must-be-stated-with-the-ratio]] makes a hard requirement
to state explicitly.

Related: [[cc-rail-vs-claudeclient-78-1]], [[cost-truth-66-3]],
[[cc-rail-guard-66-1]], [[vacuous-type-guards-on-bq-string-columns]].
