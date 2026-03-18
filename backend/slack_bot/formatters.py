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
