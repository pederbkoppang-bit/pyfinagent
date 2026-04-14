"""
Block Kit message formatters for Slack bot responses.
Slack block text limit: 3000 chars per section.
"""

from datetime import datetime


def _truncate(text: str, max_len: int = 2800) -> str:
    return text[:max_len] + "..." if len(text) > max_len else text


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
    total_return = data.get("total_return_pct", 0)

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
    score = data.get("final_score", 0)
    rec = data.get("recommendation", "N/A")
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


def format_morning_digest(portfolio_data: dict, recent_reports: list) -> list[dict]:
    """Format the daily morning digest."""
    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f":sunrise: Morning Digest — {datetime.now().strftime('%B %d, %Y')}", "emoji": True},
        },
    ]

    # Portfolio section
    if portfolio_data:
        total_pnl = portfolio_data.get("total_pnl", 0)
        total_return = portfolio_data.get("total_return_pct", 0)
        sign = "+" if total_pnl >= 0 else ""
        emoji = ":chart_with_upwards_trend:" if total_pnl >= 0 else ":chart_with_downwards_trend:"

        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Portfolio:* {emoji} {sign}${total_pnl:,.2f} ({sign}{total_return:.1f}%)"},
        })

    # Recent analyses
    if recent_reports:
        lines = []
        for r in recent_reports[:5]:
            t = r.get("ticker", "?")
            s = r.get("final_score", 0)
            rec = r.get("recommendation", "N/A")
            lines.append(f"• *{t}*: {s:.1f}/10 — {rec}")

        blocks.append({"type": "divider"})
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Recent Analyses:*\n" + "\n".join(lines)},
        })

    blocks.append({"type": "divider"})
    blocks.append({
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": ":robot_face: PyFinAgent | `/analyze TICKER` | `/portfolio` | `/report TICKER`"}],
    })

    return blocks
