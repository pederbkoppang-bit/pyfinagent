# Autoresearch cron exit-1 -- root cause analysis (phase-39.1)

**Date investigated:** 2026-05-25 (cycle 56)
**Failure window:** 2026-04-18 through 2026-05-25 (38 consecutive nights of ERROR)
**Status:** SOURCE FIXED; calendar-bound for 3-consecutive-night PASS verification.

---

## TL;DR

`scripts/autoresearch/run_memo.py` set `FAST_LLM` / `SMART_LLM` /
`STRATEGIC_LLM` env vars to raw model ids (`"claude-haiku-4-5"`), but
`gpt-researcher` requires `"<provider>:<model>"` format
(`"anthropic:claude-haiku-4-5"`). Every nightly cron exited 1 with
`ValueError: Set SMART_LLM or FAST_LLM = '<llm_provider>:<llm_model>'`.

Fixed by prefixing `anthropic:` at the caller boundary in `run_memo.py`.

---

## Trace

1. **Symptom:** `handoff/autoresearch/2026-05-25-ERROR-topic05.md` (and 37
   preceding files) all carry the same error message.
2. **Stack location:** `gpt_researcher.config.config.Config.parse_llm`
   at `.venv/lib/python3.14/site-packages/gpt_researcher/config/config.py:204-221`.
   Implementation: `llm_str.split(":", 1)` -> ValueError if no colon.
3. **Our env vars:** `run_memo.py:145-150` previously read:
   ```python
   "FAST_LLM": resolve_model("autoresearch_fast"),    # -> "claude-haiku-4-5"
   "SMART_LLM": resolve_model("autoresearch_smart"),  # -> "claude-sonnet-4-6"
   "STRATEGIC_LLM": resolve_model("autoresearch_strategic"),  # -> "claude-opus-4-7"
   ```
   `resolve_model` (`backend/config/model_tiers.py:98`) returns just the model
   id by design -- it's the single-source-of-truth for model ids, not for
   gpt-researcher-specific env-var formatting.
4. **Supported providers:** `_SUPPORTED_PROVIDERS` at `gpt_researcher/llm_provider/generic/base.py:13`
   includes `anthropic`. Our key is set in the environment via
   `run_nightly.sh` sourcing `backend/.env`.

---

## Fix

`scripts/autoresearch/run_memo.py:145-156`:

```python
# phase-39.1 (OPEN-29): gpt-researcher Config.parse_llm expects the
# `<llm_provider>:<llm_model>` format ...
env_defaults = {
    "FAST_LLM": f"anthropic:{resolve_model('autoresearch_fast')}",
    "SMART_LLM": f"anthropic:{resolve_model('autoresearch_smart')}",
    "STRATEGIC_LLM": f"anthropic:{resolve_model('autoresearch_strategic')}",
    "EMBEDDING": "huggingface:BAAI/bge-small-en-v1.5",
    "RETRIEVER": "arxiv,semantic_scholar,duckduckgo",
}
```

**Design choice:** prefix at the caller boundary, not inside `resolve_model`.
Rationale: `model_tiers.py` is the single-source-of-truth for model ids
consumed by many different LLM clients (Anthropic SDK, Vertex AI SDK,
gpt-researcher, etc.). Each client formats the provider differently
(Anthropic SDK takes bare `model=`, Vertex takes `publishers/anthropic/models/...`,
gpt-researcher takes `anthropic:...`). Keeping the prefix at the caller
preserves the canonical id.

---

## Why this stayed broken for 38 nights

- Cron failures were silent in the masterplan track (DoD-1 stayed FAIL;
  no per-night human review of `handoff/autoresearch/*.md`).
- Owner-gated until operator unblocked 2026-05-25.
- Sandbox-blocked from automating the launchd reload, so even after
  the fix, operator must `launchctl unload` + `load` the plist OR wait
  for the next 02:00 fire.

---

## Verification path

1. **Source fix locked in:** `backend/tests/test_phase_39_1_autoresearch_env.py`
   3/3 PASS. Mutation-resistant: catches future drift where
   (a) the `anthropic:` prefix is removed from `run_memo.py`, OR
   (b) `resolve_model` starts returning already-prefixed ids (would
   double-up to `anthropic:anthropic:...`).
2. **Manual reproduction** (optional, for operator):
   ```bash
   source .venv/bin/activate
   python scripts/autoresearch/run_memo.py --topic-index 0
   # Should write handoff/autoresearch/YYYY-MM-DD-topic00-<slug>.md
   # (a non-ERROR memo file), exit 0.
   ```
3. **Cron verification (calendar-bound):** wait 3 nights from 2026-05-26
   onward, then check:
   ```bash
   ls handoff/autoresearch/ | grep -E '2026-05-(2[6-9])-[^E]'  # non-ERROR files
   ```
4. **DoD-1 unblock:** once 3 PASS nights observed, flip phase-39.1
   status to `done` in masterplan + update `live_check_39.1.md`.

---

## Operator action required (post-fix)

Reload launchd job so the fixed script is picked up:

```bash
launchctl unload ~/Library/LaunchAgents/com.pyfinagent.autoresearch.plist
launchctl load   ~/Library/LaunchAgents/com.pyfinagent.autoresearch.plist
```

OR just wait for the next 02:00 cron fire -- launchd will re-exec the
script (it reads from disk each invocation).

---

## 2026-07-24 update (phase-76.9) -- two NEW independent causes after the 39.1 fix

The 39.1 provider-prefix fix (above) resolved the `ValueError` class of
failure. Autoresearch and its sibling ablation cron subsequently broke
again via TWO different, unrelated root causes. This section documents
both plus the ablation `.env`-sourcing cause investigated in the same
76.9 pass; carries forward 39.1's `root_cause_documented` intent now
that 39.1 is superseded by 76.9 (its verification grep was unsatisfiable
-- dates in the past, memos never named `-PASS`).

### (a) huggingface embedding soft-skip window (~2026-06-08 onward)

phase-51.4 added `_embedding_preflight()` (`run_memo.py`): if the
configured `EMBEDDING` provider's backing module (e.g.
`langchain_huggingface`) is not importable, the script prints a skip
message to stderr and returns 0 -- deliberately, to avoid crashing every
night on a known-missing optional dependency. While the huggingface
embedding deps were absent, this meant many nights produced **no memo
at all** (clean exit 0, but zero autoresearch output) rather than an
ERROR file, which does not show up in `-ERROR-` counts. Closed when the
embedding dependencies were installed per phase-75.13 (`deps-02`,
`_gpt_researcher_guard()` guard ordering fix + dep install), after which
`_embedding_preflight()` returns `None` and the run proceeds to the real
`GPTResearcher` call.

### (b) arxiv HTTP-429 chain (2026-07-08..07-24)

`scripts/autoresearch/run_memo.py` set `RETRIEVER =
"arxiv,semantic_scholar,duckduckgo"` (:211), putting `arxiv` in
`retrievers[0]`. Installed `gpt_researcher==0.14.8`'s
`plan_research()` (`skills/researcher.py:62`) uses `retrievers[0]`
ONLY for the initial planning search, and that call site is UNGUARDED
(no try/except) -- documented upstream as gpt-researcher issue #1282.
`gpt_researcher`'s `ArxivSearch.search()` calls
`list(arxiv.Client().results(...))` with no try/except of its own, so
an `arxiv.HTTPError` (429) raised there propagates straight out of
`conduct_research()` into `run_memo.py`'s broad `except Exception` at
`_main_async` (:114 pre-fix), which wrote an `-ERROR-` memo and
`return 1` -> `run_nightly.sh` exit 1 -> the 75.11 paging seam fired.
arXiv has been returning 429 to polite (arxiv.py's default 3-second
delay) clients server-side since ~2026-02-25 (acknowledged by arXiv
staff on the API developer group), so client-side backoff alone cannot
fix this -- the run started failing every night from 2026-07-08 through
2026-07-24 (unbroken `-ERROR-topicNN` memos, confirmed in the 76.9
research brief).

**Fixed by this step (76.9):**
1. `RETRIEVER` reordered to `"semantic_scholar,arxiv,duckduckgo"` so a
   robust retriever occupies the fatal `retrievers[0]` planning slot;
   arxiv still contributes to the (already-tolerant) sub-query fan-out.
2. A network-weather classifier (`_is_network_weather()` in
   `run_memo.py`) narrowly detects arxiv-package `HTTPError` /
   connection- or timeout-class exceptions / "429"-"503"-"rate limit"
   in the exception chain, and on a match writes a **WARN** memo
   (`<date>-WARN-topicNN.md`, filename deliberately excludes
   `-ERROR-` so downstream failure counters do not count it) and
   returns 0 -- tolerating external retriever weather without paging.
   Every other exception keeps the original `-ERROR-` memo + `return 1`
   path byte-unchanged, so the 75.11 paging seam still fires on real
   faults (bad API key, missing dependency, code bugs).

### (c) ablation `.env` EOF crash (independent of autoresearch)

The ablation launchd job's original plist sourced `backend/.env`
directly (`. backend/.env` inside an inline bash `-c` ProgramArguments
string). `backend/.env` line 81 is a non-`KEY=value` orphan fragment
(`  ON"`, an unbalanced double-quote -- the hard-wrapped tail of line
80's comment that lost its leading `#`), which makes a raw POSIX
`.`-source die with `unexpected EOF while looking for matching '"'`.
This crashed the job before it ever ran `run_ablation.py`, ~37 nights
of zero ablation output, with 0-byte StandardOut (the job died before
logging a single line).

Verbatim `backend/.env` L80-81 (comment text only, no secret values;
this file is operator-gated -- NOT edited by 76.9):

```
L80: # phase-61.1 (2026-06-12): operator tokens "60.2 FLAG: ON" / "60.3 FLAG: ON" / "57.1 FLAG:
L81:   ON"
```

Repair recommendation for the operator: rejoin `  ON"` into L80's
comment (one logical line), or prefix L81 with `#`.

**Fix:** phase-75.11 (commit `07182b94`) added
`scripts/ops/run_ablation.sh`, which sources `backend/.env` through the
SAME sanitize used by `scripts/autoresearch/run_nightly.sh:19-27`
(phase-62.6's fix for the identical failure mode): grep-filter to
`^[A-Za-z_][A-Za-z0-9_]*=` lines only, then `set -a; . "$_envtmp"; set
+a`. Line 81 does not match that pattern and is dropped, so the
sanitized source succeeds. The live plist was pointed at
`run_ablation.sh` on 2026-07-24 (~08:52, operator-attended
`OPS-ROTATE-BOOTSTRAP` bootstrap); 76.9 proves the fix with a fixture
reproduction (the fixture first reproduces the raw-`.env` EOF failure,
then shows `run_ablation.sh` surviving the same malformed line) plus a
live sourcing-seam check against the real `backend/.env`.

### (d) Carry-forward note

This update satisfies 39.1's original `root_cause_documented` intent
for the causes that superseded its own (the `anthropic:` prefix fix
above remains the historical record for 39.1's own failure class).
39.1 itself is marked `superseded` by 76.9 in `.claude/masterplan.json`
because 39.1's verification command
(`ls handoff/autoresearch/ | grep -E '2026-05-(2[3-9]|3[01])-PASS'`)
is unsatisfiable (past dates; memos are never named `-PASS`).
