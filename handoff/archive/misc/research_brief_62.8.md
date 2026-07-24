# Research Brief — phase-62.8 away-mode digest sections (tier: moderate)

Accessed 2026-06-12. Queries ran the 3-variant discipline: year-less ("Slack Block Kit blocks limit 50", "git log --since midnight day boundary"), recency-locked ("on-call handoff status report 2025", "Block Kit snapshot testing 2026"), plus direct canonical fetch (Fowler).

## Read in full (6 — counts toward gate)
| URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|
| https://docs.slack.dev/reference/block-kit/blocks/ | 2026-06-12 | official doc | WebFetch full | "up to 50 blocks in each message, and 100 blocks in modals or Home tabs" |
| https://git-scm.com/docs/git-log | 2026-06-12 | official doc | WebFetch full | --since filters by COMMITTER date, local timezone; plain --since STOPS traversal at first older commit; `--since-as-filter` (git 2.37+) visits all |
| https://martinfowler.com/articles/feature-toggles.html | 2026-06-12 | authoritative blog (Hodgson/Fowler) | WebFetch full | Ops Toggles = operational control category; test BOTH toggle states; plan removal (carrying cost); static config preferred when dynamism not needed |
| https://incident.io/blog/on-call-best-practices-guide-2026 | 2026-06-12 | industry (2026) | WebFetch full | shift report every shift EVEN QUIET ONES; cover active incidents + muted alerts/risky changes + specific URLs not "check Datadog" |
| https://dev.to/markturk87/the-definitive-guide-to-testing-slack-bots-and-why-its-been-so-hard-5feh | 2026-06-12 | community (high-signal) | WebFetch full | Slack returns 200 OK on INVALID blocks (silently dropped) → validate offline in unit tests: section ≤3000, header ≤150, count ≤50 |
| https://docs.slack.dev/changelog/2026/04/16/block-kit-new-blocks/ | 2026-06-12 | official changelog | WebFetch full | Apr 2026 added alert/card/carousel blocks (table block Aug 2025); 50-block cap UNCHANGED |

## Snippet-only (26; does not count)
docs.slack.dev/block-kit, api.slack.com/block-kit + /reference/block-kit, docs.slack.dev/reference/block-kit/block-elements, github bolt-js#2509 (~13.2K-char practical payload ceiling), python-slack-sdk#1188 (no official unit-test support), slack-testing-library, trailhead 50-item thread, workato, asu-codedevils, api.slack.com/tutorials/tags/testing, medium coveryourads, gist AustinWood; opensource.com git-log dates (since="day-1" inclusivity trap), git-book history, coderwall "git today" (`--since=midnight` idiom), 30secondsofcode; sre.google/workbook/on-call, upstat.io handoff guide, rootly blog + on-call-software, devops.com, oneuptime x3, smith.ai.

## Recency scan (2024–2026)
Performed. Findings: (a) Block Kit gained table (2025-08) + alert/card/carousel (2026-04) blocks — tempting for digests but NOT adopted here (slack_sdk version + mobile-render risk; sections suffice); 50-block cap unchanged. (b) As of Apr 2026 Bolt still ships no official testing support → pure-builder + fixture pattern remains correct. (c) incident.io 2026 guide: render every section even when empty ("none") so a missing section is distinguishable from a broken one.

## Internal code inventory
| File | Lines | Role |
|---|---|---|
| backend/slack_bot/formatters.py | 1054 total; :10 _truncate(2800); :323 format_morning_digest(portfolio_data, recent_reports, cron_health=None, system_state=None); :422 format_evening_digest(portfolio_data, trades_today); :483/:493/:500 _pct/_coerce_int/_coerce_float; :594 _field (closure inside format_accuracy_report, NOT module-level) | 54.2 optional-param None-default = the byte-identical additive idiom (:376-382, :406-411). Current worst-case blocks: morning 8, evening 6 → headroom 44 |
| backend/slack_bot/scheduler.py | :345 _is_us_trading_day_now gate; :358 _compute_cron_health (/api/jobs/all); :401 _compute_system_state — ALREADY parses /api/paper-trading/kill-switch payload (paused, pause_reason, breach.daily_loss_pct/trailing_dd_pct vs limits :407-422) + /gate; :438/:475 _send_morning/evening_digest gather via httpx; :496 trades?limit=10&since_today=true, envelope unwrap :503-504 | Gate the away append HERE (scheduler), keep formatter pure |
| backend/config/settings.py | :15 Settings(BaseSettings); :545 model_config env_file + extra=ignore; bool pattern :42 | `away_mode_enabled: bool = Field(False, ...)` auto-maps AWAY_MODE_ENABLED |
| backend/backtest/markets.py | :142 market_for_symbol: .KS/.KQ→KR; .DE/.PA/.AS/.F→EU; bare→US | trades carry NO market column (market is on pos_row only — paper_trader.py:312/:333) → derive market from ticker suffix; 50.3 stores suffixed ticker |
| backend/slack_bot/commands.py | :9 import subprocess; :60 git via check_output | precedent: subprocess git from bot process is accepted |
| backend/slack_bot/digest_test.py | :25-74 | plain-TEXT smoketest only (posts `--text`, verifies via conversations.history) — does NOT render digests; live_check needs an evening-mode extension or a one-shot mirroring scripts/ops/send_confirmation_digest.py (54.2 standalone WebClient, no Socket Mode) |
| handoff/away_ops/pending_tokens.json | live | schema: asks[]{id, raised_by, raised_at, due, ask, reply_options[], recommended}; age = today − raised_at; reply strings verbatim from reply_options |
| handoff/away_ops/health.jsonl + defect_register.md | ABSENT (62.5 / 63.3 ship them; masterplan :14695, :14806) | absent-file rendering is a MUST-TEST path |
| handoff/harness_log.md | :27018/:27029 `## Cycle N -- 2026-06-12 -- phase=62.3 result=PASS`; :27003 variant WITHOUT phase= | steps-flipped = grep today's `^## Cycle .* -- <date>` lines, tolerate missing phase=; ZERO new state |
| .claude/masterplan.json | :14748-14757 step 62.8 criteria | matches caller prompt verbatim |

## Key findings → application
1. **Shape**: new pure fn `format_away_digest_sections(data: dict) -> list[dict]` in formatters.py; scheduler builds the data dict (httpx + subprocess + file reads, all fail-open per :358/:401 idiom) and `blocks += format_away_digest_sections(...)` only `if get_settings().away_mode_enabled` — OFF path byte-identical by construction (54.2 precedent). Morning: compact asks+health only.
2. **Trades by market**: bump the away fetch to `limit=200&since_today=true` (limit=10 undercounts per-market totals); group by market_for_symbol(ticker); flag `EU: 0` (and any settings.paper_markets member at 0).
3. **NAV/DD/kill-switch**: reuse _compute_system_state's parse (:407-422) — extend or share a helper; NAV via portfolio envelope unwrap (formatters.py:350-357).
4. **Shipped-today**: `git -C <repo> log --since-as-filter=midnight --oneline --no-merges` (committer-date, Mac-local midnight = operator tz; plain --since may stop early on non-monotonic history). Fallback `--since=midnight` if git <2.37.
5. **Flag**: ops-toggle per Fowler taxonomy (operational, not trading); default False; operator types AWAY_MODE_ENABLED=true into backend/.env at 62.7 + bot restart (62.0 hook gates agent .env writes; operator keystroke bypasses — no conflict; sessions never flip it). Static-config-with-restart is Fowler-sanctioned when rapid runtime flips aren't needed. Plan removal after return (carrying cost).
6. **Block math**: worst case 1 away-header + 6×(divider+section) = 13 added; evening 6+13=19 ≤ 50; morning 8+2=10. Enforce per-section _truncate(2800) (3000 cap, file docstring :2).
7. **Tests**: fixture dicts, empty + populated, per existing test_phase_54_2_digest_cron_health.py / test_phase_slack_digest_71.py homes; assert offline constraints (count, header ≤150, section ≤3000) because Slack 200-OKs invalid blocks; assert OFF-path byte-identity (flag False ⇒ blocks equal pre-62.8 output).
8. **Live trigger**: do NOT wait for 23:00 cron and do NOT add a /jobs/trigger route (endpoint is paper_trading_daily-only); one-shot script (mirror scripts/ops/send_confirmation_digest.py) that runs the same gathering + chat_postMessage + chat.getPermalink for live_check_62.8.md.

## Consensus vs debate
Consensus: pure builders + offline fixture validation; 50-block cap; ops-flag classification. Debate: new 2026 alert/card blocks would prettify but add sdk/rendering risk — rejected; "render empty sections as 'none'" (incident.io) vs omit-to-save-blocks — adopt 'none' lines (headroom is ample, auditability wins).

## Pitfalls
- `--since` traversal stop + committer-vs-author date + local-tz midnight (git-scm).
- bolt-js#2509: practical ~13.2K-char total payload ceiling — keep sections truncated.
- Digest rides the 51.3 trading-day gate (:444/:481): no evening digest Sat/Sun/holidays → away sections gap those days (scheduled away sessions cover them); note in contract, don't "fix".
- harness_log Cycle-header has a phase-less variant (:27003) — regex must tolerate.
- Empty asks[] / empty-but-present health.jsonl ≠ absent file — three states to test.
- Slack emoji shortcodes are fine here (no-emoji rule is frontend-only; formatters.py already uses them).
- _field at :594 is a closure — don't import it; define locally.

## GO/NO-GO: **GO**
All six data sources exist or have specified absent-file behavior; flag pattern, additive byte-identity idiom, subprocess-git precedent, and one-shot send precedent are all in-repo.

## Research Gate Checklist
- [x] ≥5 sources read in full via WebFetch (6)
- [x] 10+ unique URLs (32)
- [x] Recency scan 2024–2026 performed + reported
- [x] Full pages read (not abstracts)
- [x] file:line anchors for every internal claim
- [x] Internal exploration covered all named modules
- [x] Contradictions/consensus noted; per-claim citations

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 26,
  "urls_collected": 32,
  "recency_scan_performed": true,
  "internal_files_inspected": 11,
  "report_md": "handoff/current/research_brief_62.8.md",
  "gate_passed": true
}
```
