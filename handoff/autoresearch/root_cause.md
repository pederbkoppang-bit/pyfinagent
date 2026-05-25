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
