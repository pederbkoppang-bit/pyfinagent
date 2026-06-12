"""phase-62.8: away-mode digest sections -- fixtures, caps, byte-identity."""

import copy

from backend.slack_bot.formatters import (
    _aggregate_trades_by_market,
    format_away_compact_sections,
    format_away_digest_sections,
    format_evening_digest,
    format_morning_digest,
)

PORTFOLIO = {"total_nav": 23896.04, "starting_capital": 20000.0, "total_pnl_pct": 19.48}
TRADES = [
    {"ticker": "DELL", "action": "BUY", "price": 424.15, "total_value": 246.67},
    {"ticker": "DELL", "action": "SELL", "price": 425.08, "total_value": 247.2,
     "realized_pnl_pct": 0.2193},
    {"ticker": "000660.KS", "action": "SELL", "price": 2070000.0, "total_value": 610.7,
     "realized_pnl_pct": -9.9217},
    {"ticker": "SAP.DE", "action": "BUY", "price": 210.0, "total_value": 500.0},
]

POPULATED = {
    "trades_by_market": _aggregate_trades_by_market(TRADES),
    "system_state_line": "*Kill switch:* ACTIVE (daily +0.3%/4% | trail -1.2%/10%)",
    "commits_today": ["abc1234 phase-62.3: scheduled-session engine -- PASS"],
    "steps_flipped_today": ["62.3"],
    "pending_asks": [
        {"id": "SDK-CREDIT", "due": "2026-06-15", "age_days": 0,
         "reply_options": ["SDK CREDIT: STOP-ON-EXHAUSTION",
                           "SDK CREDIT: ENABLE USAGE CREDITS <cap>"]},
    ],
    "health": {"ok": True, "ts": "2026-06-12T10:00:00Z", "backend": "running",
               "frontend": "running", "slack_bot": "running",
               "last_cycle_age_h": 15.2, "restarts_performed": 0},
    "defect_counts": {"P0": 1, "P1": 3, "P2": 2, "fixed": 4},
    "am_session_result": "[am] END session result=rc0",
}


# ── aggregation ───────────────────────────────────────────────────────

def test_aggregate_by_market_suffix_derivation():
    agg = _aggregate_trades_by_market(TRADES)
    assert agg["US"]["trades"] == 2 and agg["US"]["buys"] == 1 and agg["US"]["sells"] == 1
    assert agg["KR"]["trades"] == 1 and agg["KR"]["sells"] == 1
    assert agg["EU"]["trades"] == 1 and agg["EU"]["buys"] == 1
    assert agg["KR"]["realized_pnl_usd"] < 0  # the -9.92% stop-out


def test_aggregate_tolerates_garbage():
    agg = _aggregate_trades_by_market([{}, {"ticker": None}, {"ticker": "X", "action": "SELL",
                                       "total_value": "bad", "realized_pnl_pct": "bad"}])
    assert sum(s["trades"] for s in agg.values()) == 3


# ── six sections, populated + empty ───────────────────────────────────

def _titles(blocks):
    return [b["text"]["text"].split("\n")[0] for b in blocks if b.get("type") == "section"]


def test_populated_renders_six_sections():
    blocks = format_away_digest_sections(POPULATED)
    titles = _titles(blocks)
    for t in ("*Trades by market (today)*", "*NAV and risk*", "*Shipped today*",
              "*Open operator asks*", "*System health*", "*Defect register*"):
        assert t in titles, titles
    body = str(blocks)
    assert "SDK CREDIT: STOP-ON-EXHAUSTION" in body  # exact reply string surfaces
    assert "62.3" in body                            # steps flipped


def test_empty_renders_six_sections_with_explicit_empty_states():
    blocks = format_away_digest_sections({})
    titles = _titles(blocks)
    assert len(titles) == 6
    body = str(blocks)
    assert "EU: 0 trades" in body or "No trades today" in body
    assert "not yet available" in body.lower() or "none" in body.lower()


def test_eu_zero_flagged_when_other_markets_traded():
    data = {"trades_by_market": {"US": {"trades": 2, "buys": 1, "sells": 1,
                                        "realized_pnl_usd": 1.0}}}
    body = str(format_away_digest_sections(data))
    assert "EU: 0 trades" in body and "65.4" in body


def test_none_input_safe():
    assert len(_titles(format_away_digest_sections(None))) == 6


# ── compact (morning) variant ─────────────────────────────────────────

def test_compact_keeps_only_asks_and_health():
    titles = _titles(format_away_compact_sections(POPULATED))
    assert titles == ["*Open operator asks*", "*System health*"]


# ── caps ──────────────────────────────────────────────────────────────

def test_block_caps_and_section_limits():
    away = format_away_digest_sections(POPULATED)
    evening = format_evening_digest(PORTFOLIO, TRADES, away_sections=away)
    assert len(evening) <= 50
    for b in evening:
        if b.get("type") == "section":
            assert len(b["text"]["text"]) <= 3000
        if b.get("type") == "header":
            assert len(b["text"]["text"]) <= 150


def test_long_ask_list_truncated():
    data = {"pending_asks": [{"id": f"ASK-{i}", "reply_options": ["X" * 300],
                              "age_days": i} for i in range(40)]}
    for b in format_away_digest_sections(data):
        if b.get("type") == "section":
            assert len(b["text"]["text"]) <= 3000


# ── OFF-path byte-identity (criterion 1) ──────────────────────────────

def test_evening_off_path_byte_identical():
    base = format_evening_digest(copy.deepcopy(PORTFOLIO), copy.deepcopy(TRADES))
    with_none = format_evening_digest(copy.deepcopy(PORTFOLIO), copy.deepcopy(TRADES),
                                      away_sections=None)
    assert base == with_none


def test_morning_off_path_byte_identical():
    base = format_morning_digest(copy.deepcopy(PORTFOLIO), [], cron_health=None,
                                 system_state=None)
    with_none = format_morning_digest(copy.deepcopy(PORTFOLIO), [], cron_health=None,
                                      system_state=None, away_sections=None)
    assert base == with_none


def test_on_path_appends_before_footer():
    away = format_away_digest_sections(POPULATED)
    evening = format_evening_digest(PORTFOLIO, TRADES, away_sections=away)
    assert evening[-1]["type"] == "context"  # footer still last
    assert "*Defect register*" in str(evening)


# ── steps-closed PASS filter (cycle-2: live read-back caught CONDITIONAL listed) ──

def test_steps_closed_counts_pass_only():
    from backend.slack_bot.scheduler import _steps_closed_from_log
    lines = [
        "## Cycle 56 -- 2026-06-12 -- phase=61.1 result=CONDITIONAL (...)\n",
        "## Cycle 57 -- 2026-06-12 -- phase=62.0 result=PASS (...)\n",
        "## Cycle 59 -- 2026-06-12 -- phase=62.3 result=PASS (cycle-2) (...)\n",
        "## Cycle 40 -- 2026-06-01 -- phase=50.1 result=PASS (...)\n",  # wrong day
        "regular text mentioning phase=99.9 result=PASS\n",             # not a header
    ]
    assert _steps_closed_from_log(lines, "2026-06-12") == ["62.0", "62.3"]
