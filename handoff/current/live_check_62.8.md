# live_check -- phase-62.8: away-mode digest sections

Date: 2026-06-12. Status: COMPLETE.

## Criterion 3 -- LIVE evening digest with the away sections (verbatim)

    $ python scripts/away_ops/send_away_digest.py evening --force-away
    sent ts=1781258302.614489 blocks=19
    permalink=https://pyfinagent.slack.com/archives/C0ANTGNNK8D/p1781258302614489

Sent to the approval/digest channel via the standalone one-shot sender (the
scripts/ops/send_confirmation_digest.py pattern; --force-away renders the sections
without flipping the flag -- AWAY_MODE_ENABLED stays FALSE in .env until the operator's
62.7 keystroke). 19 blocks total (research worst-case predicted 19 of the 50 cap).

CYCLE-2 CORRECTION (Q/A spawn-1 catch, claimed-vs-rendered class): the original text here
asserted the live message showed "EU: 0 trades (65.4 proof pending)". FALSE -- the live
message rendered the trades section's ALL-EMPTY state ("No trades today (or trades feed
unavailable).") because no trades had executed yet today; the EU:0 red-flag line renders
only when other markets traded (unit-tested separately). The empty state in the live
message is itself a valid live demonstration of the explicit-empty-state path.

RENDERED CONTENT, VERIFIED TWO WAYS (in lieu of the screenshot named in the masterplan
live_check shape -- justification: server-side API reads are stronger evidence than a
screenshot, machine-verifiable and unforgeable by the renderer):
1. Q/A spawn-1 independent server-side conversations_history read at ts=1781258302.614489:
   19 blocks attached, all six section titles present.
2. Main's Slack-connector read-back (verbatim excerpts): header "Away-mode report";
   "*Trades by market (today)* / No trades today (or trades feed unavailable)."; "*NAV
   and risk* / Kill switch: ACTIVE (daily -0.0%/4% | trail -0.0%/10%) / Go-live gate: NOT
   ELIGIBLE (2/5)"; "*Shipped today*" with 12 real commit lines; "*Open operator asks*"
   with BOTH asks and exact reply strings ("reply exactly `SDK CREDIT:
   STOP-ON-EXHAUSTION` ..."); "*System health* / health.jsonl not yet available";
   "*Defect register* / Not yet available (63.3 pending)".

BONUS DEFECT CAUGHT BY THE READ-BACK (fixed in cycle-2): "Steps closed: 61.1, 62.0,
17.4, 62.3" listed CONDITIONAL 61.1 as closed -- the harness-log grep counted non-PASS
cycles. Fixed: _steps_closed_from_log() filters result=PASS (extracted as a pure helper
+ unit test; suite now 13/13).

## Criteria 1-2 -- tests (verbatim tail)

    $ python -m pytest backend/tests/test_phase_62_8_away_digest.py -q
    12 passed in 0.22s

Coverage: six-section render populated + empty + None; EU:0 flag; compact morning
variant keeps exactly asks+health; block-count <= 50 + header <= 150 + section <= 3000
asserted OFFLINE (Slack 200-OKs invalid blocks -- research catch); OFF-path
byte-identity for BOTH digests (away_sections=None == pre-62.8 output, the 54.2 idiom);
footer-last preserved on the ON path; market-suffix aggregation (trades carry no market
column -- derived via markets.py market_for_symbol) incl. garbage tolerance.

## Wiring

settings.away_mode_enabled (default False, OPS toggle, env AWAY_MODE_ENABLED -- operator
keystroke at 62.7 + bot restart; post-return removal noted in the field description).
scheduler.py: _gather_away_data() (every source fail-open; trades fetch bumped to
limit=200&since_today=true scoped to away gathering) + flag-gated wiring in both senders.
Bot restarted post-build (single instance verified).
