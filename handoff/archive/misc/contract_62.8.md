# Contract -- phase-62.8: Away-mode digest sections (per-step file; rolling slots in use by 62.4)

Date: 2026-06-12. Goal: goal-away-ops. Research: research_brief_62.8.md (gate_passed,
6 in full, recency scan; GO).

## Research anchors

formatters.py: format_morning_digest(:323, params portfolio_data/recent_reports/
cron_health/system_state), format_evening_digest(:422, portfolio_data/trades_today);
helpers _truncate(:10, 2800), _pct(:483), _coerce_int/_coerce_float(:493/:500); _field is
NOT module-level (closure at :594). Block headroom: evening 6 -> +13 worst case = 19 of
50. Phase-54.2 optional-param-default-None idiom (:376-382) = the byte-identity pattern;
keep the formatter pure, gate the append in scheduler. Trades carry NO market column:
derive via backend/backtest/markets.py:142 market_for_symbol (ticker suffix); bump the
scheduler trades fetch limit (current limit=10 at :496 undercounts). Kill-switch/NAV:
extend _compute_system_state(:401-435). Shipped-today: subprocess git (commands.py
precedent); use --since-as-filter with midnight (git >=2.37; committer-date local-tz
pitfall). Steps-flipped: grep today's '^## Cycle' lines in harness_log.md (zero new
state; tolerate phase-less variant). Token asks: pending_tokens.json (live, schema
asks[]). health.jsonl + defect_register.md DO NOT EXIST yet (62.5/63.3): render "none/
not yet available" -- incident.io practice: render every section even when empty so
missing-data is distinguishable from broken-section. away_mode_enabled: settings Field
default False (env AWAY_MODE_ENABLED auto-maps); operator keystroke + bot restart at
62.7 (Fowler Ops Toggle, static-config-with-restart sanctioned; post-return removal
noted). Live trigger for criterion 3: one-shot script mirroring
scripts/ops/send_confirmation_digest.py (standalone WebClient + chat.getPermalink), NOT
the 23:00 cron, NOT /jobs/trigger. Inherited behavior flagged: digests ride the 51.3
trading-day gate (no weekend/holiday away sections). Slack returns 200 on invalid blocks
(silent drop) -> unit tests assert constraints OFFLINE (count<=50, header<=150,
section<=3000) + OFF-path byte-identity.

## Immutable success criteria (verbatim from masterplan 62.8)

1. "format_away_digest_sections() renders all six sections from fixture data, stays
   under the 50-block Slack cap, and is appended in the evening digest only when
   away_mode_enabled is true (OFF path byte-identical)"
2. "morning digest gains only the compact asks+health sections; unit tests cover
   empty-state and populated variants"
3. "one LIVE evening digest observed in Slack containing the new sections (permalink in
   live_check_62.8.md)"

verification.command (verbatim): cd /Users/ford/.openclaw/workspace/pyfinagent && source
.venv/bin/activate && python -m pytest backend/tests -k 'away_digest or 62_8' -q

## Plan

1. settings.py: away_mode_enabled Field(False) ops flag.
2. formatters.py: format_away_digest_sections(away_data: dict|None) -> list[dict] --
   six sections (trades-by-market w/ EU:0 red flag; NAV+DD vs caps+kill-switch;
   shipped-today commits + steps flipped; open token asks w/ exact reply strings + age;
   system health from health.jsonl last line; defect-register delta), every section
   renders an explicit empty state; _truncate(2800) per section; pure function over a
   pre-gathered away_data dict. Morning variant: format_away_compact_sections (asks +
   health only).
3. scheduler.py: _gather_away_data() (market derivation via market_for_symbol; trades
   fetch limit bump scoped to away-gathering; git log --since-as-filter midnight
   subprocess; harness_log grep; pending_tokens.json read; health.jsonl tail; register
   delta) + append blocks in _send_evening_digest/_send_morning_digest ONLY when
   settings.away_mode_enabled (OFF path byte-identical, 54.2 idiom).
4. scripts/away_ops/send_away_digest.py: one-shot live sender (standalone WebClient,
   evening|morning arg, prints permalink) for criterion 3 + future PM-session use.
5. backend/tests/test_phase_62_8_away_digest.py: fixtures empty/populated; block-count +
   header/section caps offline; OFF-path byte-identity (flag False => evening blocks
   identical to pre-62.8 output for same inputs); EU:0 flag presence; absent-file paths.
6. Bot restart (kickstart) AFTER tests; live one-shot send -> permalink ->
   live_check_62.8.md. Flag stays FALSE in .env (operator flips at 62.7); the one-shot
   sender takes --force-away to render sections for the live proof without the flag.
7. ONE fresh Q/A -> harness_log -> flip.

## Out of scope

health.jsonl writer (62.5), defect register (63.3), .env flag flip (operator, 62.7).
