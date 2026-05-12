"""phase-25.A verifier -- decouple RiskJudge with independent LLM call in lite path.

Closes phase-24.4 audit F-1 (autonomous_loop.py:765 aliased
risk_assessment.reason = analysis["reason"]; one LLM call did both jobs).

Run: source .venv/bin/activate && python3 tests/verify_phase_25_A.py
"""
from __future__ import annotations

import asyncio
import re
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

REPO = Path(__file__).resolve().parents[1]
AUTOLOOP = REPO / "backend" / "services" / "autonomous_loop.py"
SIGNAL_ATTR = REPO / "backend" / "services" / "signal_attribution.py"


def _stub_anthropic_module(trader_text: str, risk_text: str):
    """Build a fake anthropic module with messages.create returning two
    distinct payloads -- first call (no system kwarg) is the trader, second
    call (with system kwarg) is the risk judge. Tests must import via
    sys.modules patching BEFORE calling the function under test."""

    state = {"call_count": 0}

    def _fake_create(**kwargs: Any):
        state["call_count"] += 1
        text = risk_text if kwargs.get("system") else trader_text
        msg = MagicMock()
        msg.text = text
        response = MagicMock()
        response.content = [msg]
        return response

    fake_client = MagicMock()
    fake_client.messages.create.side_effect = _fake_create

    fake_anthropic = types.ModuleType("anthropic")
    fake_anthropic.Anthropic = MagicMock(return_value=fake_client)
    return fake_anthropic, state


def _stub_yfinance_module():
    """Build a fake yfinance with deterministic info + history."""
    import pandas as pd

    fake_yf = types.ModuleType("yfinance")

    def _ticker(symbol):
        t = MagicMock()
        t.info = {
            "currentPrice": 100.0,
            "marketCap": 50_000_000_000,
            "trailingPE": 22.0,
            "sector": "Technology",
            "industry": "Software",
            "shortName": f"{symbol} Test Co",
        }
        idx = pd.date_range("2026-01-01", periods=90, freq="D")
        closes = [80 + i * 0.25 for i in range(90)]
        t.history.return_value = pd.DataFrame({"Close": closes}, index=idx)
        return t

    fake_yf.Ticker = _ticker
    return fake_yf


def _call_run_claude(monkey_anthropic, monkey_yf):
    """Reload autonomous_loop with the fake modules in place + run the
    coroutine. Returns the result dict.

    We pre-inject anthropic + yfinance into sys.modules so that the in-function
    `import anthropic` and `import yfinance as yf` pick up the fakes."""
    sys.modules["anthropic"] = monkey_anthropic
    sys.modules["yfinance"] = monkey_yf
    sys.modules.pop("backend.services.autonomous_loop", None)
    from backend.services import autonomous_loop  # type: ignore

    settings_mock = MagicMock()
    settings_mock.gemini_model = "claude-sonnet-4-6"
    settings_mock.anthropic_api_key = "sk-ant-test"

    return asyncio.run(autonomous_loop._run_claude_analysis("TEST", settings_mock))


def main() -> int:
    results: list[tuple[str, str, str]] = []

    if not AUTOLOOP.exists() or not SIGNAL_ATTR.exists():
        print(f"FAIL: required source files missing")
        return 1

    text = AUTOLOOP.read_text(encoding="utf-8")

    # ---- Claim 1: _LITE_RISK_JUDGE_SYSTEM mentions all three risk axes.
    sys_const = re.search(r"_LITE_RISK_JUDGE_SYSTEM\s*=", text)
    mentions_three_axes = all(
        a in text for a in ("VOLATILITY", "CONCENTRATION", "VALUATION")
    )
    results.append((
        "PASS" if sys_const and mentions_three_axes else "FAIL",
        "risk_judge_system_constant_present_with_three_axes",
        "_LITE_RISK_JUDGE_SYSTEM must exist and reference VOLATILITY/CONCENTRATION/VALUATION axes",
    ))

    # ---- Claim 2: _LITE_RISK_JUDGE_TEMPLATE present.
    tmpl_const = re.search(r"_LITE_RISK_JUDGE_TEMPLATE\s*=", text)
    results.append((
        "PASS" if tmpl_const else "FAIL",
        "risk_judge_template_constant_present",
        "_LITE_RISK_JUDGE_TEMPLATE must be defined as a module-level constant",
    ))

    # ---- Claim 3: >=2 client.messages.create calls in _run_claude_analysis.
    # Slice the function body and count.
    fn_match = re.search(
        r"async def _run_claude_analysis\(.*?\)(.*?)(?=\nasync def |\ndef |\Z)",
        text,
        re.DOTALL,
    )
    n_create_calls = 0
    if fn_match:
        n_create_calls = len(re.findall(r"client\.messages\.create\b", fn_match.group(1)))
    results.append((
        "PASS" if n_create_calls >= 2 else "FAIL",
        "second_llm_call_with_risk_specific_prompt_invoked",
        f"_run_claude_analysis must call client.messages.create >=2 times (found {n_create_calls})",
    ))

    # ---- Claim 4: re.DOTALL parse used for the risk JSON.
    dotall_match = re.search(
        r"re\.search\(\s*r?[\"']\\{\.\*\\}[\"']\s*,\s*risk_text\s*,\s*re\.(DOTALL|S)\b",
        text,
    )
    results.append((
        "PASS" if dotall_match else "FAIL",
        "risk_json_parse_uses_re_dotall",
        "risk-judge JSON parse must use re.search(r'\\{.*\\}', risk_text, re.DOTALL)",
    ))

    # ---- Claim 5: risk_assessment keys -- structural distinct dict.
    required_keys = (
        '"decision"',
        '"reasoning"',
        '"recommended_position_pct"',
        '"risk_level"',
        '"risk_limits"',
    )
    all_keys_present = all(k in text for k in required_keys)
    aliased_old = '"risk_assessment": {"reason": analysis["reason"]}'
    results.append((
        "PASS" if all_keys_present and aliased_old not in text else "FAIL",
        "risk_assessment_reasoning_distinct_from_analysis_reason",
        "risk_assessment must contain decision/reasoning/recommended_position_pct/risk_level/risk_limits and the old aliased line must be gone",
    ))

    # ---- Claim 6: BEHAVIORAL -- distinct trader vs risk text round-trip.
    behavior_ok = False
    behavior_err = ""
    n_calls = 0
    final_result: dict = {}
    try:
        sys.path.insert(0, str(REPO))
        fake_anthropic, state = _stub_anthropic_module(
            trader_text='{"action": "BUY", "confidence": 75, "score": 7, "reason": "Strong 60-day momentum drove the BUY"}',
            risk_text=(
                '{"decision": "APPROVE_REDUCED", "recommended_position_pct": 4.5, '
                '"risk_level": "MODERATE", "reasoning": "Valuation is rich at P/E 22, '
                'momentum elevated; sizing down.", '
                '"risk_limits": {"stop_loss_pct": 8, "max_drawdown_pct": 12}}'
            ),
        )
        fake_yf = _stub_yfinance_module()
        final_result = _call_run_claude(fake_anthropic, fake_yf)
        n_calls = state["call_count"]
        ra = final_result.get("risk_assessment", {})
        reasoning = ra.get("reasoning", "")
        trader_reason = (
            (final_result.get("full_report") or {}).get("analysis", {}).get("reason")
            or ""
        )
        pos_pct = float(ra.get("recommended_position_pct") or 0.0)

        if n_calls < 2:
            behavior_err = f"messages.create called {n_calls} times (expected >=2)"
        elif not reasoning:
            behavior_err = "risk_assessment.reasoning is empty"
        elif reasoning == trader_reason:
            behavior_err = f"reasoning matched trader reason: {reasoning!r}"
        elif pos_pct <= 0:
            behavior_err = f"recommended_position_pct={pos_pct} not > 0"
        else:
            behavior_ok = True
    except Exception as e:
        behavior_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if behavior_ok else "FAIL",
        "behavioral_distinct_trader_vs_risk_call_and_position_pct_positive",
        f"behavioral round-trip must show distinct reasoning + position_pct>0 + 2 LLM calls ({behavior_err})",
    ))

    # ---- Claim 7: risk_weight_greater_than_zero_for_lite_path.
    # Reuse the behavioral result -- pos_pct already computed above.
    pos_pct_seen = float((final_result.get("risk_assessment") or {}).get("recommended_position_pct") or 0.0)
    results.append((
        "PASS" if pos_pct_seen > 0 else "FAIL",
        "risk_weight_greater_than_zero_for_lite_path",
        f"recommended_position_pct must be > 0 in lite path (observed {pos_pct_seen})",
    ))

    # ---- Claim 8: BEHAVIORAL fallback -- malformed risk JSON -> default APPROVE_REDUCED.
    fallback_ok = False
    fallback_err = ""
    try:
        fake_anthropic2, _state2 = _stub_anthropic_module(
            trader_text='{"action": "HOLD", "confidence": 40, "score": 5, "reason": "Mixed signals"}',
            risk_text="not json at all, just some text",
        )
        result2 = _call_run_claude(fake_anthropic2, _stub_yfinance_module())
        ra2 = result2.get("risk_assessment", {})
        if ra2.get("decision") != "APPROVE_REDUCED":
            fallback_err = f"decision was {ra2.get('decision')!r}, expected APPROVE_REDUCED"
        elif float(ra2.get("recommended_position_pct") or 0) <= 0:
            fallback_err = f"position_pct={ra2.get('recommended_position_pct')} not > 0"
        else:
            fallback_ok = True
    except Exception as e:
        fallback_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if fallback_ok else "FAIL",
        "behavioral_malformed_risk_json_falls_back_to_safe_default",
        f"malformed risk JSON must fall back to APPROVE_REDUCED + position_pct>0 ({fallback_err})",
    ))

    # ---- Claim 9: signal_attribution consumer over the new shape produces
    # a RiskJudge row with weight > 0 and rationale != trader rationale.
    consumer_ok = False
    consumer_err = ""
    try:
        sys.modules.pop("backend.services.signal_attribution", None)
        from backend.services.signal_attribution import extract_signals_from_analysis  # type: ignore
        signals = extract_signals_from_analysis(final_result)
        risk_rows = [s for s in signals if s.get("agent") == "RiskJudge"]
        trader_rows = [s for s in signals if s.get("agent") == "Trader"]
        if not risk_rows:
            consumer_err = "no RiskJudge row in signal stack"
        elif risk_rows[0].get("weight", 0.0) <= 0:
            consumer_err = f"risk row weight={risk_rows[0].get('weight')} not > 0"
        elif not trader_rows:
            consumer_err = "no Trader row in signal stack"
        elif risk_rows[0].get("rationale") == trader_rows[0].get("rationale"):
            consumer_err = "risk + trader rationale identical"
        elif risk_rows[0].get("lite_path") is True:
            consumer_err = "risk row still flagged as lite_path duplicate after fix"
        else:
            consumer_ok = True
    except Exception as e:
        consumer_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if consumer_ok else "FAIL",
        "signal_attribution_consumer_emits_distinct_risk_row_with_weight",
        f"extract_signals_from_analysis must emit a RiskJudge row with weight>0 and rationale!=trader ({consumer_err})",
    ))

    # ---- Claim 10: independence directive present verbatim.
    independence_match = "NOT to validate the trader's recommendation" in text
    results.append((
        "PASS" if independence_match else "FAIL",
        "risk_judge_independence_directive_verbatim",
        "_LITE_RISK_JUDGE_SYSTEM must contain the literal phrase \"NOT to validate the trader's recommendation\"",
    ))

    # ---- Print results.
    n_pass = sum(1 for r in results if r[0] == "PASS")
    n_fail = len(results) - n_pass
    for verdict, claim, detail in results:
        print(f"{verdict}: {claim}")
        if verdict == "FAIL":
            print(f"      {detail}")

    print(f"\n{n_pass}/{len(results)} claims PASS, {n_fail} FAIL")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
