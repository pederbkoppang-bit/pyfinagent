"""phase-39.1 (OPEN-29) -- autoresearch cron exit 1 fix regression test.

Root cause: gpt-researcher's Config.parse_llm (.venv/.../gpt_researcher/
config/config.py:204-221) expects the `<llm_provider>:<llm_model>` format.
resolve_model in backend.config.model_tiers returns just the model id (no
provider prefix), so the split fails with ValueError.

Fix: scripts/autoresearch/run_memo.py prefixes FAST_LLM / SMART_LLM /
STRATEGIC_LLM with `anthropic:` at the caller boundary.

These tests lock the fix in so a future model_tiers refactor or
provider-change doesn't silently re-break the cron.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN_MEMO = REPO / "scripts" / "autoresearch" / "run_memo.py"


def test_phase_39_1_run_memo_prefixes_anthropic_on_llm_env_vars():
    text = RUN_MEMO.read_text(encoding="utf-8")
    # The 3 LLM env vars MUST be prefixed with `anthropic:` for
    # gpt-researcher's parse_llm to accept them.
    assert '"FAST_LLM": f"anthropic:{resolve_model(\'autoresearch_fast\')}"' in text, (
        "FAST_LLM must be set with anthropic: prefix"
    )
    assert '"SMART_LLM": f"anthropic:{resolve_model(\'autoresearch_smart\')}"' in text, (
        "SMART_LLM must be set with anthropic: prefix"
    )
    assert '"STRATEGIC_LLM": f"anthropic:{resolve_model(\'autoresearch_strategic\')}"' in text, (
        "STRATEGIC_LLM must be set with anthropic: prefix"
    )


def test_phase_39_1_gpt_researcher_parser_accepts_our_env_vars():
    """End-to-end: construct the env-var values our script would set,
    then run them through gpt-researcher's actual parser. If this fails,
    the cron will exit 1 in production."""
    import sys
    sys.path.insert(0, str(REPO))
    from backend.config.model_tiers import resolve_model
    from gpt_researcher.config.config import Config

    fast = f"anthropic:{resolve_model('autoresearch_fast')}"
    smart = f"anthropic:{resolve_model('autoresearch_smart')}"
    strat = f"anthropic:{resolve_model('autoresearch_strategic')}"

    # Parser raises ValueError on bad format; if these pass, the cron unblocks.
    fp, fm = Config.parse_llm(fast)
    sp, sm = Config.parse_llm(smart)
    tp, tm = Config.parse_llm(strat)

    assert fp == "anthropic" and fm  # truthy model id
    assert sp == "anthropic" and sm
    assert tp == "anthropic" and tm


def test_phase_39_1_model_ids_have_no_colon_so_anthropic_prefix_is_safe():
    """Mutation-resistance: if resolve_model ever starts returning an
    already-prefixed id (e.g. 'anthropic:claude-haiku-4-5'), the run_memo
    f-string would produce 'anthropic:anthropic:...'. Catch that."""
    import sys
    sys.path.insert(0, str(REPO))
    from backend.config.model_tiers import resolve_model
    for role in ("autoresearch_fast", "autoresearch_smart", "autoresearch_strategic"):
        model = resolve_model(role)
        assert ":" not in model, (
            f"resolve_model({role!r}) returned {model!r}; if this contains ':' the "
            f"anthropic: prefix in run_memo.py would double-up. Either remove the "
            f"prefix in run_memo.py or strip the colon prefix from model_tiers."
        )
