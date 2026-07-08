"""
Block Kit message formatters for Slack bot responses.
Slack block text limit: 3000 chars per section.
"""

import math
from datetime import datetime


def _truncate(text: str, max_len: int = 2800) -> str:
    return text[:max_len] + "..." if len(text) > max_len else text


# ── phase-62.8 (goal-away-ops): away-mode digest sections ───────────────────

def _aggregate_trades_by_market(trades: list) -> dict:
    """Per-market trade counts + realized P&L sum from a trades list.

    Trades carry NO market column (8-day-audit finding); market derives from
    the ticker suffix via backend/backtest/markets.py::market_for_symbol.
    Pure function (unit-tested); tolerates malformed rows.
    """
    from backend.backtest.markets import market_for_symbol

    out: dict[str, dict] = {}
    for t in trades or []:
        try:
            mkt = market_for_symbol(t.get("ticker", "")) or "US"
        except Exception:
            mkt = "US"
        slot = out.setdefault(mkt, {"trades": 0, "buys": 0, "sells": 0, "realized_pnl_usd": 0.0})
        slot["trades"] += 1
        action = (t.get("action") or "").upper()
        if action == "BUY":
            slot["buys"] += 1
        elif action == "SELL":
            slot["sells"] += 1
            tv = t.get("total_value")
            rp = t.get("realized_pnl_pct")
            try:
                if tv is not None and rp is not None and float(rp) != 0:
                    tv, rp = float(tv), float(rp)
                    slot["realized_pnl_usd"] += tv - tv / (1 + rp / 100.0)
            except (TypeError, ValueError, ZeroDivisionError):
                pass
    return out


def format_away_digest_sections(away_data: dict | None) -> list[dict]:
    """The six away-mode evening-digest sections (goal-away-ops 62.8).

    Pure Block Kit builder over a pre-gathered away_data dict; every section
    renders an EXPLICIT empty state (incident.io practice: missing data must
    be distinguishable from a broken section). Caller gates on
    settings.away_mode_enabled -- this function never reads settings.
    """
    d = away_data or {}
    sections: list[dict] = [
        {"type": "header",
         "text": {"type": "plain_text", "text": "Away-mode report", "emoji": True}},
    ]

    def add(title: str, body: str):
        sections.append({"type": "divider"})
        sections.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": _truncate(f"*{title}*\n{body}")},
        })

    # 1. Trades by market (EU:0 flagged while the 65.4 proof is open)
    by_mkt = d.get("trades_by_market") or {}
    if by_mkt:
        lines = []
        for mkt in ("US", "KR", "EU"):
            s = by_mkt.get(mkt)
            if s:
                lines.append(f"{mkt}: {s['trades']} trades ({s['buys']}B/{s['sells']}S), "
                             f"realized {s['realized_pnl_usd']:+.2f} USD")
            elif mkt == "EU":
                lines.append("EU: 0 trades :red_circle: (65.4 proof pending)")
            else:
                lines.append(f"{mkt}: 0 trades")
        for mkt, s in sorted(by_mkt.items()):
            if mkt not in ("US", "KR", "EU"):
                lines.append(f"{mkt}: {s['trades']} trades")
        add("Trades by market (today)", "\n".join(lines))
    else:
        add("Trades by market (today)", "No trades today (or trades feed unavailable).")

    # 2. NAV + risk + kill-switch
    add("NAV and risk", d.get("system_state_line") or "Kill-switch/gate state unavailable.")

    # 3. Shipped today
    commits = d.get("commits_today") or []
    steps = d.get("steps_flipped_today") or []
    body = ""
    if commits:
        body += "\n".join(f"- {c}" for c in commits[:12])
    if steps:
        body += ("\n" if body else "") + "Steps closed: " + ", ".join(steps)
    add("Shipped today", body or "Nothing shipped today.")

    # 4. Open token asks (exact reply strings -- the operator's action surface)
    asks = d.get("pending_asks") or []
    if asks:
        lines = []
        for a in asks[:8]:
            reply = (a.get("reply_options") or ["?"])[0]
            due = f" (due {a['due']})" if a.get("due") else ""
            age = f" [{a['age_days']}d old]" if a.get("age_days") is not None else ""
            lines.append(f"- {a.get('id', '?')}{due}{age}: reply exactly `{reply}`"
                         + (f" or `{a['reply_options'][1]}`" if len(a.get("reply_options", [])) > 1 else ""))
        add("Open operator asks", "\n".join(lines))
    else:
        add("Open operator asks", "None.")

    # 5. System health (62.5 health.jsonl)
    h = d.get("health")
    if h:
        ok = "OK" if h.get("ok") else "DEGRADED"
        body = (f"{ok} at {h.get('ts', '?')} -- backend={h.get('backend', '?')} "
                f"frontend={h.get('frontend', '?')} slack_bot={h.get('slack_bot', '?')} "
                f"last_cycle_age_h={h.get('last_cycle_age_h', '?')} "
                f"restarts={h.get('restarts_performed', 0)}")
        if d.get("am_session_result"):
            body += f"\nAM session: {d['am_session_result']}"
        add("System health", body)
    else:
        add("System health", "health.jsonl not yet available (62.5 pending or watchdog quiet).")

    # 6. Defect-register delta (63.3)
    reg = d.get("defect_counts")
    if reg:
        add("Defect register", f"open P0={reg.get('P0', 0)} P1={reg.get('P1', 0)} "
                               f"P2={reg.get('P2', 0)} (fixed total={reg.get('fixed', 0)})")
    else:
        add("Defect register", "Not yet available (63.3 pending).")

    return sections


def format_away_compact_sections(away_data: dict | None) -> list[dict]:
    """Morning-digest compact variant: open asks + system health only."""
    full = format_away_digest_sections(away_data)
    # full = [header, (divider, section) x6]; asks = pair index 4 -> blocks 7-8,
    # health = pair 5 -> blocks 9-10. Select by title text instead of position
    # to stay robust to reordering.
    keep: list[dict] = []
    for i, b in enumerate(full):
        if b.get("type") == "section":
            txt = b.get("text", {}).get("text", "")
            if txt.startswith("*Open operator asks*") or txt.startswith("*System health*"):
                keep.extend([{"type": "divider"}, b])
    return keep


def _score_emoji(score: float) -> str:
    if score >= 8:
        return ":star:"
    if score >= 6:
        return ":white_check_mark:"
    if score >= 4:
        return ":large_yellow_circle:"
    return ":red_circle:"


def _rec_color(action: str) -> str:
    action_upper = action.upper() if action else ""
    if "STRONG_BUY" in action_upper or "STRONG BUY" in action_upper:
        return "#22c55e"
    if "BUY" in action_upper:
        return "#4ade80"
    if "SELL" in action_upper:
        return "#ef4444"
    return "#f59e0b"


def format_analysis_result(report: dict, ticker: str) -> list[dict]:
    """Format a completed analysis as Block Kit blocks."""
    score = report.get("final_weighted_score", 0)
    rec = report.get("recommendation", {})
    action = rec.get("action", "N/A") if isinstance(rec, dict) else str(rec)
    justification = rec.get("justification", "") if isinstance(rec, dict) else ""
    summary = report.get("final_summary", "")
    key_risks = report.get("key_risks", [])

    emoji = _score_emoji(score)

    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"{ticker} Analysis Complete", "emoji": True},
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Score:* {emoji} {score:.1f}/10"},
                {"type": "mrkdwn", "text": f"*Recommendation:* {action}"},
            ],
        },
    ]

    if justification:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Thesis:* {_truncate(justification, 500)}"},
        })

    if summary:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Summary:* {_truncate(summary, 800)}"},
        })

    if key_risks:
        risk_text = "\n".join(f"• {_truncate(r, 200)}" for r in key_risks[:5])
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Key Risks:*\n{risk_text}"},
        })

    # Debate consensus
    debate = report.get("debate_result", {})
    if debate:
        consensus = debate.get("consensus", "N/A")
        confidence = debate.get("consensus_confidence", 0)
        blocks.append({
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Debate Consensus:* {consensus}"},
                {"type": "mrkdwn", "text": f"*Confidence:* {confidence:.0%}"},
            ],
        })

    blocks.append({"type": "divider"})
    blocks.append({
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": f":robot_face: PyFinAgent | {datetime.now().strftime('%Y-%m-%d %H:%M')}"}],
    })

    return blocks


def format_portfolio_summary(data: dict) -> list[dict]:
    """Format portfolio performance as Block Kit blocks."""
    positions = data.get("positions", [])
    total_value = data.get("total_value", 0)
    total_pnl = data.get("total_pnl", 0)
    # phase-25.G: paper_trading endpoint returns total_pnl_pct, not total_return_pct
    total_return = data.get("total_pnl_pct", data.get("total_return_pct", 0))

    pnl_emoji = ":chart_with_upwards_trend:" if total_pnl >= 0 else ":chart_with_downwards_trend:"
    pnl_sign = "+" if total_pnl >= 0 else ""

    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "Portfolio Summary", "emoji": True},
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Total Value:* ${total_value:,.2f}"},
                {"type": "mrkdwn", "text": f"*P&L:* {pnl_emoji} {pnl_sign}${total_pnl:,.2f} ({pnl_sign}{total_return:.1f}%)"},
            ],
        },
    ]

    if positions:
        pos_lines = []
        for p in positions[:10]:
            ticker = p.get("ticker", "?")
            pnl = p.get("pnl", 0)
            ret = p.get("return_pct", 0)
            sign = "+" if pnl >= 0 else ""
            icon = ":green_circle:" if pnl >= 0 else ":red_circle:"
            pos_lines.append(f"{icon} *{ticker}*: {sign}${pnl:,.2f} ({sign}{ret:.1f}%)")

        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "\n".join(pos_lines)},
        })

    blocks.append({
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": f":robot_face: PyFinAgent | {datetime.now().strftime('%Y-%m-%d %H:%M')}"}],
    })

    return blocks


def _signal_emoji(action: str) -> str:
    """Green for BUY, red for SELL, yellow for HOLD. Matches _rec_color semantics."""
    action_upper = (action or "").upper()
    if "BUY" in action_upper:
        return ":green_circle:"
    if "SELL" in action_upper:
        return ":red_circle:"
    return ":large_yellow_circle:"


def format_signal_alert(signal: dict, trade: dict | None = None) -> list[dict]:
    """Format a published trading signal as Block Kit blocks.

    Called by `backend/agents/mcp_servers/signals_server.py::publish_signal`
    after the signal has been validated, risk-checked, and (for BUY/SELL)
    booked to the paper trader. Pure function; returns only the block list.
    Caller owns channel routing and the ASCII `text=` push-preview fallback.

    Layout follows the research-gate consensus (handoff/current/research.md
    sections 1 and 6): header (subject) -- section fields (conf/price/size/
    stop) -- section thesis -- divider -- context (signal_id, timestamp).
    Reuses _truncate and _signal_emoji to stay consistent with the other
    formatters in this module. Never raises; missing fields show "N/A".

    Args:
        signal: {"ticker", "signal" ("BUY"/"SELL"/"HOLD"), "confidence",
                 "date", "factors", "reason" (optional), "size_usd" (optional),
                 "stop_price" (optional), "signal_id" (optional)}
        trade:  Paper-trader trade record with "price", "total_value",
                "quantity", "trade_id" -- or None if stub/HOLD/dry-run

    Returns:
        Block Kit list[dict]. At minimum: header, fields section, context.
    """
    if not isinstance(signal, dict):
        signal = {}
    if trade is not None and not isinstance(trade, dict):
        trade = None

    ticker = str(signal.get("ticker", "UNKNOWN")).upper()
    action = str(signal.get("signal", "HOLD")).upper()
    try:
        confidence = float(signal.get("confidence", 0.0) or 0.0)
    except (ValueError, TypeError):
        confidence = 0.0
    reason_text = str(signal.get("reason", "") or "")
    sdate = str(signal.get("date", "") or "")
    signal_id = str(signal.get("signal_id", "") or "")

    emoji = _signal_emoji(action)

    # Price / size / stop resolution: explicit signal fields win, else trade
    # record, else "N/A". Formatting is paper-trader friendly -- USD only.
    price_str = "N/A"
    size_str = "N/A"
    stop_str = "N/A"
    if trade:
        try:
            tp = float(trade.get("price", 0.0) or 0.0)
            if tp > 0.0:
                price_str = f"${tp:,.2f}"
        except (ValueError, TypeError):
            pass
        try:
            tv = float(trade.get("total_value", 0.0) or 0.0)
            if tv > 0.0:
                size_str = f"${tv:,.2f}"
        except (ValueError, TypeError):
            pass
    try:
        sig_stop = float(signal.get("stop_price", 0.0) or 0.0)
        if sig_stop > 0.0:
            stop_str = f"${sig_stop:,.2f}"
    except (ValueError, TypeError):
        pass

    conf_str = f"{confidence:.2f}"

    blocks: list[dict] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{emoji} {ticker} {action}",
                "emoji": True,
            },
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Confidence:* {conf_str}"},
                {"type": "mrkdwn", "text": f"*Price:* {price_str}"},
                {"type": "mrkdwn", "text": f"*Size:* {size_str}"},
                {"type": "mrkdwn", "text": f"*Stop:* {stop_str}"},
            ],
        },
    ]

    if reason_text:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Thesis:* {_truncate(reason_text, 500)}",
            },
        })

    blocks.append({"type": "divider"})

    footer_parts = [":robot_face: PyFinAgent"]
    if sdate:
        footer_parts.append(sdate)
    if signal_id:
        footer_parts.append(f"signal_id: `{signal_id[:16]}`")
    footer_parts.append(datetime.now().strftime("%Y-%m-%d %H:%M"))
    blocks.append({
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": " | ".join(footer_parts)}],
    })

    return blocks


def format_report_card(data: dict, ticker: str) -> list[dict]:
    """Format a stored report as Block Kit blocks."""
    # phase-61.2: degraded rows carry present-but-None score/recommendation
    # (.get defaults do NOT fire on present-None; None crashed _score_emoji
    # comparisons and the :.1f format).
    score = data.get("final_score", 0) or 0
    rec = data.get("recommendation", "N/A") or "DEGRADED"
    summary = data.get("summary", "")
    date = data.get("analysis_date", "")

    emoji = _score_emoji(score)

    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"{ticker} — Latest Report", "emoji": True},
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Score:* {emoji} {score:.1f}/10"},
                {"type": "mrkdwn", "text": f"*Recommendation:* {rec}"},
                {"type": "mrkdwn", "text": f"*Date:* {date}"},
            ],
        },
    ]

    if summary:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Summary:*\n{_truncate(summary, 1500)}"},
        })

    blocks.append({
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": f":robot_face: PyFinAgent | Run `/analyze {ticker}` for a fresh analysis"}],
    })

    return blocks


def _portfolio_snapshot_date(p: dict) -> str | None:
    """phase-72 helper: extract the snapshot date the persisted portfolio
    row reflects. `paper_portfolio.updated_at` is the canonical write-time
    of the snapshot (set by `paper_trader.mark_to_market`). Format as
    YYYY-MM-DD so the digest reads naturally as "as of close 2026-05-22".
    Returns None if the row has no usable timestamp."""
    raw = p.get("updated_at") or p.get("snapshot_date") or p.get("last_updated")
    if not raw or not isinstance(raw, str):
        return None
    # Accept ISO timestamps (with or without timezone) and plain dates.
    return raw[:10]


def format_morning_digest(portfolio_data: dict, recent_reports: list, cron_health: str | None = None, system_state: str | None = None, away_sections: list | None = None) -> list[dict]:
    """Format the daily morning digest.

    phase-54.2: optional `cron_health` + `system_state` lines for the operator's
    remote week. The scheduler computes both from backend endpoints (fail-open) and
    passes them in; the formatter stays a pure template builder ($0, no I/O).
    `system_state` carries the kill-switch + go-live-gate state (the most
    decision-relevant away-week signal -- "is the system halted?"). When both are
    None (default) the digest is byte-identical to before -- purely additive.
    """
    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f":sunrise: Morning Digest — {datetime.now().strftime('%B %d, %Y')}", "emoji": True},
        },
    ]

    # Portfolio section
    if portfolio_data:
        # phase-71 cycle (2026-05-26): /api/paper-trading/portfolio returns a
        # nested envelope {"portfolio": {...}, "positions": [...],
        # "sector_breakdown": {...}}. Phase-25.G (commit 55241e3a) switched
        # the endpoint but missed both (a) the unwrap and (b) the fact that
        # the new endpoint has no `total_pnl` (dollar P&L) column -- only
        # `total_nav`, `starting_capital`, `total_pnl_pct`. Compute the
        # dollar P&L as nav - starting so the digest matches the cockpit.
        # Defensive: works whether caller passes the envelope or the inner dict.
        p = (portfolio_data.get("portfolio")
             if isinstance(portfolio_data.get("portfolio"), dict)
             else portfolio_data)
        total_nav = float(p.get("total_nav") or 0.0)
        starting = float(p.get("starting_capital") or 0.0)
        total_pnl = total_nav - starting
        total_return = float(
            p.get("total_pnl_pct") or p.get("total_return_pct") or 0.0
        )
        sign = "+" if total_pnl >= 0 else ""
        emoji = ":chart_with_upwards_trend:" if total_pnl >= 0 else ":chart_with_downwards_trend:"
        # phase-72 cycle (2026-05-26): operator-flagged that Slack values
        # ($23,184) disagreed with cockpit live NAV ($23,732). The Slack
        # digest correctly displays the persisted close-of-day snapshot
        # (digest cadence is fixed; no intraday accuracy needed) -- but
        # the message MUST be labeled "as of close YYYY-MM-DD" so the
        # operator isn't confused vs the live cockpit values.
        snap_date = _portfolio_snapshot_date(p)
        as_of = f" (as of close {snap_date})" if snap_date else ""

        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Portfolio:* {emoji} {sign}${total_pnl:,.2f} ({sign}{total_return:.1f}%){as_of}"},
        })

    # phase-54.2 cycle-2: kill-switch + go-live-gate state (criterion 3 -- the most
    # decision-relevant away-week signal). Provided by the scheduler (fail-open);
    # byte-identical when None.
    if system_state:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": system_state},
        })

    # Recent analyses
    if recent_reports:
        lines = []
        for r in recent_reports[:5]:
            t = r.get("ticker", "?")
            # phase-61.2: None-safe (degraded rows persist NULL score/rec).
            s = r.get("final_score", 0) or 0
            rec = r.get("recommendation", "N/A") or "DEGRADED"
            # phase-60.1 (AW-4): lite/full provenance marker. The away week
            # showed identical-looking 7.0/10 lines from the 2-call lite
            # wrapper; `[lite]` makes a degraded score visibly degraded.
            # Rows persisted before the tag existed have no analysis_path
            # and render unchanged.
            path = r.get("analysis_path")
            tag = f" `[{path}]`" if path in ("lite", "full") else ""
            lines.append(f"• *{t}*: {s:.1f}/10 — {rec}{tag}")

        blocks.append({"type": "divider"})
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Recent Analyses:*\n" + "\n".join(lines)},
        })

    # phase-54.2: cron-health line (above the footer divider). Only when provided.
    if cron_health:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": cron_health},
        })

    # phase-62.8: away-mode compact sections (54.2 idiom -- byte-identical when None).
    if away_sections:
        blocks.extend(away_sections)

    blocks.append({"type": "divider"})
    blocks.append({
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": ":robot_face: PyFinAgent | `/analyze TICKER` | `/portfolio` | `/report TICKER`"}],
    })

    return blocks


def format_evening_digest(portfolio_data: dict, trades_today: list, away_sections: list | None = None) -> list[dict]:
    """Format the daily evening digest with end-of-day summary."""
    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f":city_sunset: Evening Digest — {datetime.now().strftime('%B %d, %Y')}", "emoji": True},
        },
    ]

    if portfolio_data:
        # phase-71 cycle: same nested-envelope unwrap as format_morning_digest.
        # See that function's comment for full context.
        p = (portfolio_data.get("portfolio")
             if isinstance(portfolio_data.get("portfolio"), dict)
             else portfolio_data)
        total_nav = float(p.get("total_nav") or 0.0)
        starting = float(p.get("starting_capital") or 0.0)
        total_pnl = total_nav - starting
        total_return = float(
            p.get("total_pnl_pct") or p.get("total_return_pct") or 0.0
        )
        sign = "+" if total_pnl >= 0 else ""
        emoji = ":chart_with_upwards_trend:" if total_pnl >= 0 else ":chart_with_downwards_trend:"
        # phase-72: "as of close YYYY-MM-DD" snapshot timestamp.
        snap_date = _portfolio_snapshot_date(p)
        as_of = f" (as of close {snap_date})" if snap_date else ""

        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*End-of-Day Portfolio:* {emoji} {sign}${total_pnl:,.2f} ({sign}{total_return:.1f}%){as_of}"},
        })

    if trades_today:
        lines = []
        for t in trades_today[:10]:
            ticker = t.get("ticker", "?")
            action = t.get("action", "?")
            price = t.get("price", 0)
            lines.append(f"• *{ticker}*: {action} @ ${price:,.2f}")

        blocks.append({"type": "divider"})
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Today's Trades:*\n" + "\n".join(lines)},
        })
    else:
        blocks.append({"type": "divider"})
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*Today's Trades:* No trades executed today."},
        })

    # phase-62.8: away-mode sections (54.2 idiom -- byte-identical when None).
    if away_sections:
        blocks.extend(away_sections)

    blocks.append({"type": "divider"})
    blocks.append({
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": ":robot_face: PyFinAgent Evening Summary | `/portfolio` for details"}],
    })

    return blocks


def _pct(value, decimals: int = 1, signed: bool = False) -> str:
    """Render a numeric percent-scale value as a percent string. "N/A" on bad input."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "N/A"
    fmt = f"{{:+.{decimals}f}}" if signed else f"{{:.{decimals}f}}"
    return f"{fmt.format(v)}%"


def _coerce_int(d: dict, key: str) -> int:
    try:
        return int(d.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _coerce_float(d: dict, key: str) -> float:
    try:
        v = float(d.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    # Phase 4.2.3.1 SN1 fix: sanitize NaN / +Inf / -Inf at the display
    # boundary so upstream IEEE 754 non-finite values never render as
    # "nan%" or "inf%" in the Slack fields. See handoff/current/research.md.
    return v if math.isfinite(v) else 0.0


def format_accuracy_report(
    data: dict | None,
    window: tuple[str, str] | None = None,
) -> list[dict]:
    """Format a signal accuracy aggregate as Block Kit blocks for weekly reports.

    Pure-stdlib. Never raises. Input is the dict shape produced by the
    Phase 4.2.2 accuracy aggregator: total_count, scored_count, hits,
    misses, neutral, unscored, hit_rate (0..1), hit_rate_ci_low/high
    (0..1 Wilson), mean/median_forward_return_pct (percent scale),
    groups (dict[str, dict] of same shape).

    Research-gate lock-ins (handoff/current/research.md):
    - Block order: header, context(window), TL;DR section, divider, fields
      section, divider, per-group sections, trailing context.
    - Wilson CI inline only when scored_count >= 5. 1..4 -> "preliminary
      -- n=X". 0 -> hit-rate collapses to "Scoring pending" (no fake 0.00%).
    - Groups rendered as section mrkdwn (not fields), sorted by
      scored_count desc, hard-capped at 5 with overflow context block.
    - No emoji-as-label for a11y. Percents always carry "%".

    Args:
        data: Accuracy aggregate dict, or None.
        window: Optional (start_iso, end_iso) pair rendered as
            "YYYY-MM-DD to YYYY-MM-DD" in the early context block.

    Returns:
        Block Kit list[dict]. Always non-empty.
    """
    if isinstance(window, (tuple, list)) and len(window) == 2:
        try:
            win_text = f"{str(window[0])} to {str(window[1])}"
        except Exception:
            win_text = ""
    else:
        win_text = ""
    gen_ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    def _header_block() -> dict:
        return {
            "type": "header",
            "text": {"type": "plain_text", "text": "Weekly Accuracy Report", "emoji": True},
        }

    def _ctx_block(extra: str | None = None) -> dict:
        parts = [":robot_face: PyFinAgent"]
        if win_text:
            parts.append(win_text)
        if extra:
            parts.append(extra)
        parts.append(gen_ts)
        return {"type": "context", "elements": [{"type": "mrkdwn", "text": " | ".join(parts)}]}

    def _unavailable(reason: str) -> list[dict]:
        return [
            _header_block(),
            {"type": "section", "text": {"type": "mrkdwn",
                "text": _truncate(f"*Accuracy report unavailable* -- {reason}", 500)}},
            _ctx_block(),
        ]

    if data is None or not isinstance(data, dict):
        return _unavailable("input data missing")

    total_count = _coerce_int(data, "total_count")
    scored_count = _coerce_int(data, "scored_count")
    hits = _coerce_int(data, "hits")
    hit_rate = _coerce_float(data, "hit_rate")
    ci_low = _coerce_float(data, "hit_rate_ci_low")
    ci_high = _coerce_float(data, "hit_rate_ci_high")
    mean_ret = _coerce_float(data, "mean_forward_return_pct")
    median_ret = _coerce_float(data, "median_forward_return_pct")
    groups_raw = data.get("groups") if isinstance(data.get("groups"), dict) else {}

    if total_count <= 0:
        return _unavailable(
            f"No signals issued in {win_text}" if win_text else "No signals issued in this window"
        )

    hit_rate_pct_str = f"{hit_rate * 100.0:.1f}%"
    mean_str = _pct(mean_ret, decimals=2, signed=True)
    median_str = _pct(median_ret, decimals=2, signed=True)

    def _field(label: str, value: str) -> dict:
        return {"type": "mrkdwn", "text": _truncate(f"*{label}:* {value}", 180)}

    blocks: list[dict] = [_header_block()]

    early_parts = [":robot_face: PyFinAgent"]
    if win_text:
        early_parts.append(win_text)
    early_parts.append(f"Generated {gen_ts}")
    blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": " | ".join(early_parts)}]})

    if scored_count >= 1:
        tldr = (
            f"*TL;DR:* Hit rate *{hit_rate_pct_str}* on {scored_count:,} "
            f"scored signal{'s' if scored_count != 1 else ''} "
            f"({total_count:,} total this window)."
        )
    else:
        tldr = (
            f"*TL;DR:* {total_count:,} signal{'s' if total_count != 1 else ''} "
            f"issued; scoring pending (no forward-return data yet)."
        )
    blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": _truncate(tldr, 500)}})
    blocks.append({"type": "divider"})

    # Headline fields: always EVEN count, always <= 10.
    if scored_count <= 0:
        # Phase 4.2.3.1 SN2 fix: on n=0 samples, mean/median forward returns
        # have no data either -- collapse to the canonical "Scoring pending"
        # placeholder (CFA III(D) fair-presentation; do not render fake 0%).
        fields = [
            _field("Total signals", f"{total_count:,}"),
            _field("Hit rate", "Scoring pending"),
            _field("Mean forward return", "Scoring pending"),
            _field("Median forward return", "Scoring pending"),
        ]
    elif scored_count < 5:
        fields = [
            _field("Total signals", f"{total_count:,}"),
            _field("Scored", f"{scored_count:,} ({hits}/{scored_count})"),
            _field("Hit rate", hit_rate_pct_str),
            _field("Confidence", f"preliminary -- n={scored_count}"),
            _field("Mean forward return", mean_str),
            _field("Median forward return", median_str),
        ]
    else:
        # Clamp CI bounds to [0, 1] defensively; do not trust the producer.
        ci_low_c = max(0.0, min(1.0, ci_low))
        ci_high_c = max(0.0, min(1.0, ci_high))
        ci_str = f"[{ci_low_c:.2f}, {ci_high_c:.2f}]"
        fields = [
            _field("Total signals", f"{total_count:,}"),
            _field("Scored", f"{scored_count:,} ({hits}/{scored_count})"),
            _field("Hit rate", hit_rate_pct_str),
            _field("Wilson 95% CI", ci_str),
            _field("Mean forward return", mean_str),
            _field("Median forward return", median_str),
        ]

    if len(fields) % 2 == 1:
        fields = fields[:-1]
    if len(fields) > 10:
        fields = fields[:10]
    blocks.append({"type": "section", "fields": fields})

    # Per-group: sorted by scored_count desc, capped at 5, overflow in context.
    if groups_raw:
        def _grp_scored(item):
            _k, v = item
            return _coerce_int(v, "scored_count") if isinstance(v, dict) else 0

        sorted_groups = sorted(groups_raw.items(), key=_grp_scored, reverse=True)
        shown = sorted_groups[:5]
        overflow = max(0, len(sorted_groups) - len(shown))
        if shown:
            blocks.append({"type": "divider"})
            for label, gdata in shown:
                if not isinstance(gdata, dict):
                    continue
                g_scored = _coerce_int(gdata, "scored_count")
                g_hits = _coerce_int(gdata, "hits")
                g_hit_rate = _coerce_float(gdata, "hit_rate")
                g_total = _coerce_int(gdata, "total_count")
                g_mean = _coerce_float(gdata, "mean_forward_return_pct")
                g_label = _truncate(str(label), 40)
                if g_scored >= 1:
                    line = (
                        f"*{g_label}* -- {g_hit_rate * 100.0:.1f}% ({g_hits}/{g_scored})  "
                        f"mean {_pct(g_mean, decimals=2, signed=True)}  n={g_total}"
                    )
                else:
                    line = f"*{g_label}* -- scoring pending  n={g_total}"
                blocks.append({"type": "section",
                    "text": {"type": "mrkdwn", "text": _truncate(line, 500)}})
            if overflow > 0:
                blocks.append({"type": "context", "elements": [{
                    "type": "mrkdwn",
                    "text": f"+{overflow} more group{'s' if overflow != 1 else ''} -- see full report",
                }]})

    blocks.append({"type": "divider"})
    trailing_parts = [":robot_face: PyFinAgent | weekly accuracy summary"]
    if 0 < scored_count < 10:
        trailing_parts.append(f"small sample (n={scored_count})")
    blocks.append({"type": "context",
        "elements": [{"type": "mrkdwn", "text": " | ".join(trailing_parts)}]})

    return blocks


def format_trade_confirmation(trade: dict) -> list[dict]:
    """phase-25.J: format a paper-trade confirmation as Block Kit blocks.

    Receives the trade dict shape returned by `paper_trader.execute_buy/sell`:
        {trade_id, ticker, action, quantity, price, total_value,
         transaction_cost, reason, created_at, ...}

    Closes phase-24.5 audit F-5(a) — no send_trade_confirmation existed.
    Special-cases reason='stop_loss_trigger' with a rotating-light icon so
    operators immediately recognize 25.1-driven stop sells.
    """
    action = str(trade.get("action") or "TRADE").upper()
    ticker = str(trade.get("ticker") or "?")
    quantity = float(trade.get("quantity") or 0.0)
    price = float(trade.get("price") or 0.0)
    total_value = float(trade.get("total_value") or 0.0)
    reason = str(trade.get("reason") or "n/a")

    is_stop_loss = reason == "stop_loss_trigger"
    icon = ":rotating_light:" if is_stop_loss else (
        ":chart_with_upwards_trend:" if action == "BUY"
        else ":chart_with_downwards_trend:"
    )

    title = f"{icon} {action} {ticker}"
    if is_stop_loss:
        title = f"{icon} STOP-LOSS TRIGGERED: SELL {ticker}"

    blocks: list[dict] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": _truncate(title, 150), "emoji": True},
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Quantity:* {quantity:.4f}"},
                {"type": "mrkdwn", "text": f"*Price:* ${price:.4f}"},
                {"type": "mrkdwn", "text": f"*Total Value:* ${total_value:,.2f}"},
                {"type": "mrkdwn", "text": f"*Reason:* {_truncate(reason, 80)}"},
            ],
        },
    ]
    trade_id = trade.get("trade_id")
    if trade_id:
        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": f"trade_id: `{trade_id}`"}],
        })
    return blocks


def format_autoresearch_summary(results: dict) -> list[dict]:
    """phase-25.P: format a completed weekly meta-evolution / autoresearch cycle
    summary as Block Kit blocks.

    Closes audit bucket 24.5 F-5(g). Mirrors `format_cycle_summary` (25.N) but
    surfaces the fields that `run_meta_evolution_cycle` actually returns:
    cron_allocations count, provider_allocations count, archetype_count,
    duration, errors. Invoked by `backend/meta_evolution/cron.py` on Sunday
    completion.

    Returns Block Kit list[dict] (header + section + divider + context).
    """
    started_at = str(results.get("started_at") or "")
    finished_at = str(results.get("finished_at") or "")
    duration = results.get("duration_seconds")
    errors = results.get("errors") or []
    error_count = len(errors) if isinstance(errors, list) else 0
    cron_alloc = results.get("cron_allocations")
    prov_alloc = results.get("provider_allocations")
    archetype_count = results.get("archetype_count")

    def _int_or_na(v) -> str:
        if v is None:
            return "n/a"
        if isinstance(v, (list, dict)):
            return f"{len(v)}"
        try:
            return f"{int(v)}"
        except (TypeError, ValueError):
            return str(v)

    def _fmt_sec(v) -> str:
        if v is None:
            return "n/a"
        try:
            s = float(v)
            if s < 60:
                return f"{s:.0f}s"
            return f"{s / 60:.1f}m"
        except (TypeError, ValueError):
            return str(v)

    blocks: list[dict] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": ":microscope: Weekly Autoresearch Cycle Complete",
                "emoji": True,
            },
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Started:* {started_at or '(unknown)'}"},
                {"type": "mrkdwn", "text": f"*Finished:* {finished_at or '(unknown)'}"},
                {"type": "mrkdwn", "text": f"*Duration:* {_fmt_sec(duration)}"},
                {"type": "mrkdwn", "text": f"*Errors:* {error_count}"},
                {"type": "mrkdwn", "text": f"*Cron allocations:* {_int_or_na(cron_alloc)}"},
                {"type": "mrkdwn", "text": f"*Provider allocations:* {_int_or_na(prov_alloc)}"},
                {"type": "mrkdwn", "text": f"*Archetypes:* {_int_or_na(archetype_count)}"},
                {"type": "mrkdwn", "text": "*Cadence:* weekly (Sun)"},
            ],
        },
        {"type": "divider"},
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": ":robot_face: phase-25.P autoresearch digest * Closes bucket 24.5 F-5(g)",
                }
            ],
        },
    ]
    return blocks


def format_cycle_summary(summary: dict) -> list[dict]:
    """phase-25.N: format a completed autonomous-cycle summary as Block Kit blocks.

    Closes audit bucket 24.5 F-5(e): the loop currently posts only on failure;
    operators get no Slack signal on the happy path. This formatter is invoked
    by `autonomous_loop._final_status == "completed"`.

    Required summary keys: cycle_id, started_at, status. Optional: ended_at,
    duration_sec, trades_executed, stops_executed, mode (full/lite/dry_run),
    recommendations_count.

    Returns Block Kit list[dict] (header + section + fields + context).
    """
    cycle_id = str(summary.get("cycle_id") or "(unknown)")
    started_at = str(summary.get("started_at") or "")
    status = str(summary.get("status") or "unknown")
    duration = summary.get("duration_sec")
    trades = summary.get("trades_executed")
    stops = summary.get("stops_executed")
    mode = str(summary.get("mode") or "(unknown)")
    recs = summary.get("recommendations_count")

    def _fmt_int(v) -> str:
        if v is None:
            return "n/a"
        try:
            return f"{int(v)}"
        except (TypeError, ValueError):
            return str(v)

    def _fmt_sec(v) -> str:
        if v is None:
            return "n/a"
        try:
            s = float(v)
            if s < 60:
                return f"{s:.0f}s"
            return f"{s / 60:.1f}m"
        except (TypeError, ValueError):
            return str(v)

    blocks: list[dict] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f":bar_chart: Autonomous Cycle {status.title()}",
                "emoji": True,
            },
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Cycle ID:* `{cycle_id}`"},
                {"type": "mrkdwn", "text": f"*Started:* {started_at or '(unknown)'}"},
                {"type": "mrkdwn", "text": f"*Duration:* {_fmt_sec(duration)}"},
                {"type": "mrkdwn", "text": f"*Mode:* {mode}"},
                {"type": "mrkdwn", "text": f"*Trades:* {_fmt_int(trades)}"},
                {"type": "mrkdwn", "text": f"*Stops:* {_fmt_int(stops)}"},
                {"type": "mrkdwn", "text": f"*Recs:* {_fmt_int(recs)}"},
                {"type": "mrkdwn", "text": f"*Status:* {status}"},
            ],
        },
        {"type": "divider"},
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": ":robot_face: phase-25.N cycle-summary digest * Closes bucket 24.5 F-5(e)",
                }
            ],
        },
    ]
    return blocks


def format_strategy_switch(event: dict) -> list[dict]:
    """phase-25.R: format a Strategy Auto-Switch event as Block Kit blocks.

    Closes red-line goal-c (dynamically shift strategy to whichever is making
    the most money). Fires when `Promoter.write_to_registry` flips a new
    strategy to status="active" and supersedes the prior active row.

    Required event keys: new_strategy_id, prior_strategy_id (may be None),
    dsr, pbo, allocation_pct, switched_at (ISO), week_iso.

    Returns Block Kit list[dict]. P0 visual (rotating-light) + structured
    fields for at-a-glance diff between the new and superseded strategies.
    """
    new_id = str(event.get("new_strategy_id") or "(unknown)")
    prior_id_raw = event.get("prior_strategy_id")
    prior_id = str(prior_id_raw) if prior_id_raw else "(none -- first promotion)"
    dsr = event.get("dsr")
    pbo = event.get("pbo")
    allocation = event.get("allocation_pct")
    switched_at = str(event.get("switched_at") or "")
    week_iso = str(event.get("week_iso") or "")

    def _fmt_num(v, fmt: str = "{:.3f}") -> str:
        if v is None:
            return "n/a"
        try:
            return fmt.format(float(v))
        except (TypeError, ValueError):
            return str(v)

    blocks: list[dict] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": ":rotating_light: Strategy Auto-Switch (P0)",
                "emoji": True,
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*New active strategy:* `{new_id}` "
                    f"(week {week_iso}, switched {switched_at})"
                ),
            },
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Strategy ID:* {new_id}"},
                {"type": "mrkdwn", "text": f"*Week:* {week_iso or '(unknown)'}"},
                {"type": "mrkdwn", "text": f"*DSR:* {_fmt_num(dsr)}"},
                {"type": "mrkdwn", "text": f"*PBO:* {_fmt_num(pbo)}"},
                {"type": "mrkdwn", "text": f"*Allocation %:* {_fmt_num(allocation, '{:.2%}')}"},
                {"type": "mrkdwn", "text": f"*Switched at:* {switched_at or '(unknown)'}"},
            ],
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Superseded:* `{prior_id}`",
            },
        },
        {"type": "divider"},
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": ":robot_face: phase-25.R auto-switching policy * Closes red-line goal-c",
                }
            ],
        },
    ]
    return blocks


def format_escalation_alert(
    severity: str,
    title: str,
    details: dict,
    actions: list[str] | None = None,
) -> list[dict]:
    """Format a trading incident escalation as Block Kit blocks.

    Args:
        severity: "P0", "P1", or "P2".
        title: Short incident title (e.g. "Kill Switch Triggered").
        details: Dict of key-value pairs rendered as section fields.
        actions: Optional list of recommended next-step strings.

    Returns:
        Block Kit list[dict].
    """
    severity = str(severity or "P1").upper()
    title = str(title or "Incident")

    icon = ":rotating_light:" if severity == "P0" else ":warning:"
    sev_label = f"*{severity}*" if severity in ("P0", "P1", "P2") else f"*{severity}*"

    blocks: list[dict] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{icon} {title}",
                "emoji": True,
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"Severity: {sev_label} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            },
        },
    ]

    if details and isinstance(details, dict):
        fields = []
        for k, v in list(details.items())[:10]:
            fields.append({"type": "mrkdwn", "text": _truncate(f"*{k}:* {v}", 180)})
        if len(fields) % 2 == 1:
            fields.append({"type": "mrkdwn", "text": " "})
        blocks.append({"type": "section", "fields": fields[:10]})

    if actions:
        action_text = "\n".join(f"• {_truncate(str(a), 200)}" for a in actions[:5])
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Recommended actions:*\n{action_text}"},
        })

    blocks.append({"type": "divider"})
    blocks.append({
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": f":robot_face: PyFinAgent Escalation | {severity} | Immediate attention required"}],
    })

    return blocks
