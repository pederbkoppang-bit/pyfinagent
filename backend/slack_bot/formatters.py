"""
Block Kit message formatters for Slack bot responses.
Slack block text limit: 3000 chars per section.
"""

import math
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


def format_evening_digest(portfolio_data: dict, trades_today: list) -> list[dict]:
    """Format the daily evening digest with end-of-day summary."""
    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f":city_sunset: Evening Digest — {datetime.now().strftime('%B %d, %Y')}", "emoji": True},
        },
    ]

    if portfolio_data:
        total_pnl = portfolio_data.get("total_pnl", 0)
        total_return = portfolio_data.get("total_return_pct", 0)
        sign = "+" if total_pnl >= 0 else ""
        emoji = ":chart_with_upwards_trend:" if total_pnl >= 0 else ":chart_with_downwards_trend:"

        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*End-of-Day Portfolio:* {emoji} {sign}${total_pnl:,.2f} ({sign}{total_return:.1f}%)"},
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
