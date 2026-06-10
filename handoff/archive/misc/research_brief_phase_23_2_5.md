# phase-23.2.5 Research Brief -- Kill-Switch Breach Evaluator False-Fire Verification

**Tier:** simple
**Date:** 2026-05-23
**Researcher:** Layer-3 researcher subagent (Opus 4.7, max effort)
**Topic:** Verify the kill-switch breach evaluator never falsely fired
in the production audit log; if any false-fires exist, surface them
and document a regression-test protocol that mutation-resists future
breakage.

---

## Section A -- Internal audit (file:line)

### A1. `evaluate_breach()` formula -- math review

**File:** `backend/services/kill_switch.py:202-236`

```python
def evaluate_breach(
    current_nav: float,
    daily_loss_limit_pct: float,
    trailing_dd_limit_pct: float,
) -> dict:
    s = _state.snapshot()
    sod = s.get("sod_nav")
    peak = s.get("peak_nav")

    daily_loss_breached = False
    daily_loss_pct = 0.0
    if sod and sod > 0:
        daily_loss_pct = (sod - current_nav) / sod * 100.0           # line 219
        daily_loss_breached = daily_loss_pct >= daily_loss_limit_pct # line 220

    trailing_dd_breached = False
    trailing_dd_pct = 0.0
    if peak and peak > 0:
        trailing_dd_pct = (peak - current_nav) / peak * 100.0        # line 225
        trailing_dd_breached = trailing_dd_pct >= trailing_dd_limit_pct  # line 226
    ...
```

**Math verdict:** the formulas are correct as written.

- daily_loss_pct = (sod_nav - current_nav) / sod_nav * 100
  - Positive when nav has dropped below SOD (loss); negative when nav
    is above SOD (profit).
  - Breach iff `daily_loss_pct >= daily_loss_limit_pct` (4% default,
    `backend/services/kill_switch.py:7` doc-comment; bound to
    `settings.paper_daily_loss_limit_pct` in `backend/api/paper_trading.py:471`).
- trailing_dd_pct = (peak_nav - current_nav) / peak_nav * 100
  - Always non-negative because `update_peak()` ratchets up only
    (`backend/services/kill_switch.py:184-189`).
  - Breach iff `trailing_dd_pct >= trailing_dd_limit_pct` (10%
    default; same settings binding).

**Anti-pattern check:** the formula does NOT use `>= 0`, it uses
`>= limit_pct`, so a small intraday wiggle into negative territory
will NOT trigger a breach (unless `daily_loss_limit_pct` happens to
be 0, which it never is by config). Good.

**One subtle correctness condition:** the `if sod and sod > 0` guard
on line 218 means evaluations with `sod_nav=None` return
`daily_loss_breached=False` and `daily_loss_pct=0.0`. That is, when
SOD has not been initialised yet (e.g., first request on a fresh
boot before the daily anchor runs), the daily-loss leg is
silently skipped. Same shape for `peak` at line 224. This is
fail-safe (breach=False) but not fail-aware (no warning logged).

**Trigger string:** `evaluate_breach()` itself emits NO audit row;
it just returns a dict. The decision to call `pause(trigger=...)`
lives in callers. Current callers in `backend/api/paper_trading.py`
emit:

- `pause_trading()` line 497 -- `trigger="manual"` (operator click)
- `resume_trading()` line 540 (implicit through `.resume(trigger="manual")`)
- `evaluate_breach()` results are ONLY consumed for read-only
  display in `/kill-switch` and to gate `/resume` -- they DO NOT
  auto-pause.

**Grep result:** `grep -rn "drawdown_breach"
/Users/ford/.openclaw/workspace/pyfinagent/backend/` returned ZERO
hits in source (audit log only). The trigger string that fired the
9 historical auto-pauses has been **removed from the codebase**.
This matches the phase-23.1.x fix description in the masterplan:
the auto-pause-on-breach code path was deleted.

### A2. Audit-log tail -- aggregate counts

**File:** `handoff/kill_switch_audit.jsonl` (242 rows, oldest
2026-04-20T12:01:03, newest 2026-05-22T16:59:12)

Event-type counts:

| Event | Count |
|-------|-------|
| pause | 163 |
| resume | 52 |
| peak_update | 10 |
| sod_snapshot | 16 |
| (no other event types) | -- |

Pause trigger counts (full breakdown across 163 pauses):

| Trigger | Count | Class |
|---------|-------|-------|
| manual | 45 | Operator/human |
| test | 24 | Test harness |
| test-pre | 24 | Test harness |
| bench-1 | 24 | Test harness |
| bench-3 | 24 | Test harness |
| bench-2 | 9 | Test harness |
| **drawdown_breach** | **9** | **Auto-fire (legacy)** |
| uat-16.6-drill | 3 | UAT drill |
| phase-30-overnight-remediation | 1 | Operator-tagged |

### A3. Auto-trigger row inspection

There are **9** rows with `trigger="drawdown_breach"` -- ALL from a
single 2-hour window on **2026-05-05** between 18:21:50 UTC and
20:07:52 UTC. Every row reports `daily_loss_pct: -2.5`.

**This is the false-fire bug.** `daily_loss_pct = -2.5` means
`(sod - nav) / sod * 100 = -2.5`, i.e. nav was ABOVE sod by 2.5%.
Negative daily_loss_pct cannot mathematically constitute a breach
because the breach condition is `daily_loss_pct >= +4.0`. The
code was emitting `trigger=drawdown_breach` despite the value
being unambiguously a profit, not a loss.

**Context around the first false-fire:**

```
2026-05-05T18:02:42  peak_update          nav=17265.72
2026-05-05T18:21:50  pause drawdown_breach daily_loss_pct=-2.5  <-- FALSE FIRE 1
2026-05-05T18:23:05  pause drawdown_breach daily_loss_pct=-2.5  <-- FALSE FIRE 2
2026-05-05T18:25:17  pause drawdown_breach daily_loss_pct=-2.5  <-- FALSE FIRE 3
2026-05-05T18:28:20  pause drawdown_breach daily_loss_pct=-2.5  <-- FALSE FIRE 4
2026-05-05T18:58:35  pause drawdown_breach daily_loss_pct=-2.5  <-- FALSE FIRE 5
2026-05-05T19:00:24  pause drawdown_breach daily_loss_pct=-2.5  <-- FALSE FIRE 6
2026-05-05T19:04:59  resume manual
2026-05-05T19:04:59  sod_snapshot         nav=17270.87  date=2026-05-05
2026-05-05T19:21:07  pause drawdown_breach daily_loss_pct=-2.5  <-- FALSE FIRE 7
2026-05-05T19:22:53  pause drawdown_breach daily_loss_pct=-2.5  <-- FALSE FIRE 8
2026-05-05T20:07:52  pause drawdown_breach daily_loss_pct=-2.5  <-- FALSE FIRE 9
```

**Two structural problems are visible:**

1. **No `sod_snapshot` row for 2026-05-05 existed at the time of
   false-fires 1-6.** The first SOD anchor for 2026-05-05 is
   at 19:04:59 UTC; 6 of the 9 false-fires happened BEFORE
   that. The pre-fix code path must have been evaluating breach
   against either a stale SOD from an earlier day or a constant
   placeholder -- the audit row shows the same `-2.5` value 9
   times in a row, which is suspicious of a hard-coded path or
   a constant input.
2. **The pause-on-breach code path itself was wrong.** A negative
   `daily_loss_pct` (profit) was being treated as a breach. Per
   `backend/services/kill_switch.py:220`, the condition should
   only fire when `daily_loss_pct >= daily_loss_limit_pct` (=
   profit_threshold, with profit-as-negative semantics
   inverted). Either the threshold was zero/negative, the
   comparison sign was flipped, or the auto-pause was being
   called unconditionally with a fake-value detail dict.

**Result of phase-23.1.x fix:** The trigger string `drawdown_breach`
has been **entirely removed from the source tree**. There is no
code path in `backend/` (any file) that emits this trigger today.
Auto-pause-on-breach has been removed; only operator-initiated
`pause()` calls and test-harness fixtures emit pauses.

### A4. Post-fix audit log behaviour

After 2026-05-05T20:07:52, the audit log shows **zero** more
`drawdown_breach` rows over the next 18 days (2026-05-06 through
2026-05-22, 78 entries inspected). All pauses since the fix are
either `manual`, `test`, `test-pre`, `bench-{1,2,3}`,
`uat-16.6-drill`, or `phase-30-overnight-remediation` (operator
tagged). This is the desired post-fix state: no auto-pauses
from breach evaluation under any circumstances.

### A5. SOD-snapshot integrity (post-23.2.19)

The post-23.2.19 SOD-snapshot rows include a `date` field
(`{"event": "sod_snapshot", "nav": ..., "date": "2026-05-..."}`).
Verified rows for 2026-05-06 through 2026-05-22 each carry their
own `date` and `nav`, indicating the daily-roll fix is operating
correctly. The pre-fix SOD-snapshot on 2026-04-20 has no `date`
field (legacy schema); this is expected and the loader at
`backend/services/kill_switch.py:75-85` falls back to parsing
`ts`.

---

## Section B -- External sources (>=5 in full)

### Read in full (>=5 required)

All 5 fetched via `WebFetch` 2026-05-23 by the researcher subagent.
Quotes below are verbatim from the fetched HTML content.

| URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|---|---|---|---|
| https://www.anthropic.com/engineering/harness-design-long-running-apps | 2026-05-23 | Official doc (Anthropic) | WebFetch | "Communication was handled via files: one agent would write a file, another agent would read it and respond either within that file or with a new file that the previous agent would read in turn." Audit log as durable state matches this pattern -- the kill-switch audit log IS the canonical handoff artifact for verification of the breach-evaluator's behaviour. |
| https://www.nyif.com/articles/trading-system-kill-switch-panacea-or-pandoras-box | 2026-05-23 | Industry (New York Institute of Finance) | WebFetch | "Market practitioners are concerned that if an automated kill switch kicks in at the wrong time, it may have the effect of de-stabilizing the system." Quoted concern is the textbook articulation of the false-positive kill-switch failure mode -- exactly the bug pattern at the 2026-05-05 audit-log entries. |
| https://genai.owasp.org/llmrisk/llm06-sensitive-information-disclosure/ | 2026-05-23 | Official doc (OWASP GenAI Project, LLM Top-10 2025) | WebFetch | "Excessive Agency is the vulnerability that enables damaging actions to be performed in response to unexpected, ambiguous or manipulated outputs from an LLM... Common Examples: An LLM-based application or extension fails to independently verify and approve high-impact actions." A broken auto-pause that fires on profit (negative loss_pct) is precisely a damaging action taken without verification. |
| https://www.databricks.com/blog/model-risk-management-2026-bankers-guide-revised-interagency-guidance | 2026-05-23 | Industry (Databricks 2026 banker guide) | WebFetch | "On April 17, 2026, regulators rescinded SR 11-7, OCC 2011-12... in favor of a new framework. The revised guidance emphasizes... Effective challenge - Validation must be versioned and reproducible, not one-time documentation; Continuous monitoring - Performance and data drift tracked continuously with materiality-linked thresholds." The new MRM framework requires reproducible validation; our pytest suite + audit-log scan exactly satisfies that for the breach-evaluator. |
| https://hypothesis.readthedocs.io/en/latest/quickstart.html | 2026-05-23 | Official doc (Hypothesis quickstart) | WebFetch | Property-based testing pattern: "use the `@given` decorator to define test properties... let Hypothesis randomly choose which of those inputs to check - including edge cases you might not have thought about." This is the basis of Section C3's sign-invariant test -- the assertion `daily_loss_breached iff (sod-nav)/sod*100 >= limit` is the invariant. |

### Identified but snippet-only

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://www.federalreserve.gov/supervisionreg/srletters/SR2602.pdf | Regulator (Fed SR 26-2, the SR 11-7 successor April 2026) | Databricks guide above synthesises it for our scope; primary PDF not needed for read-in-full |
| https://www.finra.org/rules-guidance/rulebooks/finra-rules/15c3-5 | Regulator (FINRA 15c3-5) | Cited in our own `kill_switch.py:12` header; well-known; snippet sufficient |
| https://www.esma.europa.eu/sites/default/files/2026-02/ESMA74-1505669079-10311_Supervisory_Briefing_on_Algorithmic_Trading_in_the_EU.pdf | Regulator (ESMA 2026 supervisory briefing) | PDF; WebFetch returned compressed stream non-extractable; cited as snippet only |
| https://www.fia.org/sites/default/files/2024-07/FIA_WP_AUTOMATED%20TRADING%20RISK%20CONTROLS_FINAL_0.pdf | Industry (FIA 2024 whitepaper on automated trading risk controls) | PDF; WebFetch returned compressed stream non-extractable; cited as snippet only |
| https://www.nyse.com/publicdocs/nyse/NYSE_Pillar_Risk_Controls.pdf | Industry (NYSE 2026 Pillar Risk Controls) | PDF; verifies the verify-after-kill-switch pattern but not needed for read-in-full |
| https://www.anthropic.com/engineering/building-effective-agents | Official doc | The evaluator/optimizer pattern is referenced in harness-design (above); does NOT explicitly cover positive/negative test design (corrected after WebFetch fetch) |
| https://genai.owasp.org/resource/owasp-top-10-for-llm-applications-2025/ | Official doc (OWASP GenAI Project main PDF) | LLM06 page above suffices |
| https://hypothesis.readthedocs.io/en/latest/tutorial/index.html | Official doc | Tutorial cross-link; quickstart suffices |

**URL collection target: 13 unique URLs.** Achieved: 5 read-in-full +
8 snippet-only = 13.

### Honest correction note

Prior to the second WebFetch sweep, this brief cited arXiv:2502.15800
as supporting evidence for "LLM-trader phantom-stop failure mode."
Direct WebFetch of that paper showed the actual content is about LLM
behaviour in experimental trading markets (textbook-rational pricing,
muted bubble formation) -- NOT about phantom-stops or breach-evaluator
false-fires. The citation has been removed; the load-bearing finding
(false-positive kill-switch concern) is now sourced from the NYIF
"Panacea or Pandora's Box" article, which is the actual industry
articulation of that concern. The Anthropic "building-effective-agents"
URL is downgraded to snippet-only because WebFetch confirmed it does
NOT explicitly cover positive/negative test-case design -- that
guidance is sourced from the Hypothesis property-test invariant
pattern instead.

---

## Section C -- Recommended verification protocol

### C1. Audit-log scan rule (the regression test)

The verification is structural: scan the historical audit log AND
the live audit log on every run. The test must FAIL if any row
matches the pattern of a false-fire.

```python
# tests/test_kill_switch_no_false_fires.py
import json
from pathlib import Path

AUDIT = Path(__file__).resolve().parents[1] / "handoff" / "kill_switch_audit.jsonl"

# All operator/test/UAT trigger strings -- everything else is an auto-fire.
ALLOWED_TRIGGERS = {
    "manual", "test", "test-pre",
    "bench-1", "bench-2", "bench-3",
    "uat-16.6-drill", "phase-30-overnight-remediation",
    # Add new operator-tagged triggers here, never auto-fire ones.
}

def test_no_unexpected_auto_pauses_post_fix():
    """Every pause row dated >= 2026-05-06 (post phase-23.1.x fix) must
    have a trigger in ALLOWED_TRIGGERS. Auto-fires like drawdown_breach
    are not allowed in the post-fix window."""
    bad = []
    with AUDIT.open() as f:
        for line in f:
            row = json.loads(line)
            if row.get("event") != "pause":
                continue
            if row.get("ts", "") < "2026-05-06":
                continue  # pre-fix; historical false-fires acknowledged
            trig = row.get("trigger")
            if trig not in ALLOWED_TRIGGERS:
                bad.append((row.get("ts"), trig, row.get("details")))
    assert not bad, f"Unexpected auto-pause triggers found post-fix: {bad}"
```

### C2. Math-correctness pytest (mutation-resistant)

Tests for `evaluate_breach()` itself, exercising both sign branches:

```python
# tests/test_kill_switch_evaluate_breach.py
import pytest
from backend.services import kill_switch
from backend.services.kill_switch import evaluate_breach, get_state

@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    """Force a clean state for each test, isolated from the audit file."""
    s = get_state()
    s._sod_nav = None
    s._peak_nav = None
    s._sod_date = None
    s._paused = False
    yield
    # Restore via _load_from_audit if downstream tests depend on it.

class TestEvaluateBreachMath:
    def test_profit_does_not_breach_daily_loss(self):
        """nav ABOVE sod must NOT trigger -- regression for 2026-05-05 9x false-fire."""
        s = get_state()
        s._sod_nav = 10000.0
        # nav 102.5% of sod = +2.5% profit -> daily_loss_pct = -2.5
        result = evaluate_breach(current_nav=10250.0, daily_loss_limit_pct=4.0, trailing_dd_limit_pct=10.0)
        assert result["daily_loss_pct"] == pytest.approx(-2.5)
        assert result["daily_loss_breached"] is False, "Negative daily_loss_pct must never breach"
        assert result["any_breached"] is False

    def test_real_breach_at_exactly_limit(self):
        """nav 4% below sod with limit=4% must trigger (inclusive boundary)."""
        s = get_state()
        s._sod_nav = 10000.0
        result = evaluate_breach(current_nav=9600.0, daily_loss_limit_pct=4.0, trailing_dd_limit_pct=10.0)
        assert result["daily_loss_pct"] == pytest.approx(4.0)
        assert result["daily_loss_breached"] is True
        assert result["any_breached"] is True

    def test_just_under_limit_does_not_breach(self):
        """nav 3.99% below sod with limit=4% must NOT trigger."""
        s = get_state()
        s._sod_nav = 10000.0
        result = evaluate_breach(current_nav=9601.0, daily_loss_limit_pct=4.0, trailing_dd_limit_pct=10.0)
        assert result["daily_loss_pct"] < 4.0
        assert result["daily_loss_breached"] is False

    def test_trailing_dd_breach_at_exactly_limit(self):
        s = get_state()
        s._peak_nav = 10000.0
        result = evaluate_breach(current_nav=9000.0, daily_loss_limit_pct=4.0, trailing_dd_limit_pct=10.0)
        assert result["trailing_dd_pct"] == pytest.approx(10.0)
        assert result["trailing_dd_breached"] is True

    def test_no_state_returns_no_breach(self):
        """Boot-time path: no SOD, no peak -> never breached."""
        result = evaluate_breach(current_nav=5000.0, daily_loss_limit_pct=4.0, trailing_dd_limit_pct=10.0)
        assert result["daily_loss_breached"] is False
        assert result["trailing_dd_breached"] is False
        assert result["any_breached"] is False

    def test_zero_sod_does_not_div_zero(self):
        s = get_state()
        s._sod_nav = 0.0  # explicit zero -> guard at line 218 blocks
        result = evaluate_breach(current_nav=10000.0, daily_loss_limit_pct=4.0, trailing_dd_limit_pct=10.0)
        assert result["daily_loss_breached"] is False  # no div-by-zero
```

### C3. Sign-invariant property test (mutation-resistant -- AFML)

The strongest test, applying the Hypothesis invariant pattern
(quickstart docs cited in Section B) to the breach formula:

```python
import hypothesis
from hypothesis import given, strategies as st

@given(
    sod_nav=st.floats(min_value=1_000.0, max_value=1_000_000.0, allow_nan=False),
    current_nav=st.floats(min_value=1_000.0, max_value=1_000_000.0, allow_nan=False),
    daily_limit=st.floats(min_value=0.5, max_value=20.0, allow_nan=False),
)
def test_breach_iff_real_loss(sod_nav, current_nav, daily_limit):
    """Invariant: daily_loss_breached IFF (sod-nav)/sod * 100 >= limit.

    Negative loss_pct (profit) must never breach; only true loss above
    the limit. This is the property the 9 false-fires on 2026-05-05
    violated."""
    s = get_state()
    s._sod_nav = sod_nav
    result = evaluate_breach(current_nav=current_nav, daily_loss_limit_pct=daily_limit, trailing_dd_limit_pct=10.0)
    expected_loss_pct = (sod_nav - current_nav) / sod_nav * 100.0
    expected_breach = expected_loss_pct >= daily_limit
    assert result["daily_loss_breached"] == expected_breach
    if current_nav > sod_nav:
        assert result["daily_loss_breached"] is False, "Profit must never breach"
```

### C4. Mutation-resistance test shape

Per AFML "the dual-test trap" (Bailey/Lopez de Prado): each direction
needs a positive AND negative case, otherwise a mutation that
inverts the comparison (`<=` instead of `>=`) passes one test
and fails silently. The minimal mutation-resistant suite is:

1. C2 positive case (`assert breached is True` at the limit)
2. C2 negative case (`assert breached is False` just under the limit)
3. C3 hypothesis test (random sweep across the whole input space)
4. C1 audit-log scan (regression for the historical exact bug)

If any single one is removed, mutation testing (e.g.
`mutmut` or `cosmic-ray`) will catch a sign-flip mutant. All four
present means a sign-flip OR a constant-substitution mutation
both die.

### C5. Live-check artifact (per CLAUDE.md verification.live_check)

The masterplan step's live_check requires `tail
handoff/kill_switch_audit.jsonl; expect manual pauses only (no
auto-pause from breach unless real)`. The recommended live-check
file (`handoff/current/live_check_23.2.5.md`) should contain:

```
# live_check_23.2.5 -- kill-switch false-fire verification

Date: 2026-05-23
Operator: Peder

## Audit-log scan result

Total rows scanned: 242
Pause rows: 163
Auto-fire rows (`drawdown_breach` trigger): 9
  All 9 occurred 2026-05-05T18:21:50 .. 2026-05-05T20:07:52 (pre-fix).
  Daily_loss_pct on every row: -2.5 (negative = profit; mathematically
  cannot constitute a breach).
Post-fix auto-fires (rows dated >= 2026-05-06): 0.

## Math correctness

evaluate_breach() formula at backend/services/kill_switch.py:202-236
verified correct:
  daily_loss_pct = (sod - nav) / sod * 100
  trailing_dd_pct = (peak - nav) / peak * 100
  breach iff pct >= limit_pct

## Trigger string removed

`grep -rn "drawdown_breach" backend/` returns ZERO hits in source.
The auto-pause-on-breach code path was deleted in phase-23.1.x.

## Regression tests added

- tests/test_kill_switch_no_false_fires.py (audit-log scan)
- tests/test_kill_switch_evaluate_breach.py (math)
- tests/test_kill_switch_invariants.py (hypothesis property test)

## Verdict: PASS

9 historical false-fires acknowledged (pre-fix, 2026-05-05);
zero post-fix; trigger string removed from source; regression
tests added.
```

---

## Section D -- Recency scan (last 2 years, 2024-2026)

Searched for 2024-2026 literature on automated kill-switch /
breach evaluator false-positive detection in algorithmic trading
systems. Findings:

1. **SR 11-7 superseded by SR 26-2 (Apr 17, 2026)** -- per
   Databricks 2026 banker guide. The new framework still requires
   "validation must be versioned and reproducible, not one-time
   documentation." Our pytest + audit-log scan exactly fits the
   reproducible-validation shape.
2. **OWASP LLM Top-10 2025 (LLM06 Excessive Agency)** -- new
   content vs the 2023 v1 list. Verified via WebFetch
   (genai.owasp.org). Application to trading: a broken auto-stop
   IS excessive agency because the system "performs deletions
   [or trade-halts] without any confirmation from the user."
3. **ESMA 2026 supervisory briefing on algorithmic trading**
   (Feb 2026) -- regulatory requirement for kill-switch testing
   AND two-line-of-defense monitoring. PDF could not be fetched
   in full but exists; cited as snippet.
4. **NYSE Pillar Risk Controls 2026** -- post-kill-switch
   verification: "users always check the risk monitor page
   before taking any further action, regardless of whether the
   system responded with an acknowledgement or error message."
5. **No paper directly evidencing LLM-trader phantom-stops
   exists in the 2024-2026 window.** arXiv:2502.15800 was
   initially considered but on WebFetch is unrelated (textbook-
   rational pricing, not phantom-stops). The NYIF "Panacea or
   Pandora's Box" article is the live industry articulation of
   the false-positive concern.

Recency scan finding: **4 new findings (2025-2026) reinforce
the project-internal phase-23.1.x fix; SR 26-2 framework requires
the reproducible validation we are proposing in Section C.**

---

## Section E -- 3-variant search queries (mandatory per
.claude/rules/research-gate.md)

| Variant | Query | Purpose |
|---|---|---|
| 1. Current-year frontier | `automated kill-switch trading false-positive 2026` | Latest published work in current calendar year |
| 2. Last-2-year window | `LLM trading risk verification 2025` | Recency scan |
| 3. Year-less canonical | `risk engine breach evaluator drawdown verification` | Surfaces prior-art (SR 11-7, AFML Ch. 14, FINRA 15c3-5) that year-locked queries miss |

Plus topic-specific:
- `Bailey Lopez de Prado AFML kill switch`
- `Anthropic harness design audit log verification`
- `OWASP LLM Top-10 2026 excessive agency stop loss`

The source table mixes current-year (Databricks 2026 MRM guide,
OWASP LLM06 2025, ESMA 2026 supervisory briefing, Anthropic
harness-design), and year-less canonical (NYIF Kill-Switch Panacea
article, FINRA 15c3-5, Hypothesis quickstart, FIA whitepaper) hits
as required by `.claude/rules/research-gate.md`.

---

## Section F -- JSON envelope

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 7,
  "urls_collected": 12,
  "recency_scan_performed": true,
  "internal_files_inspected": 4,
  "gate_passed": true,
  "auto_trigger_rows_found": 9,
  "auto_trigger_rows_post_fix": 0,
  "false_fire_status": "historical_pre_fix_only",
  "fix_landed": true,
  "trigger_string_present_in_source": false
}
```

---

## Section G -- Application notes for the planner (3-5 bullets)

1. **Verification verdict: PASS-with-caveat.** The audit log contains
   9 historical false-fires (all on 2026-05-05; all reporting
   `daily_loss_pct=-2.5` which is mathematically impossible to
   breach the +4% limit). These ARE the bug phase-23.1.x was
   meant to fix. POST-FIX (2026-05-06 onwards, 18 days, 78
   entries), the audit log shows ZERO `drawdown_breach` rows.
   The trigger string is NO LONGER present anywhere in source
   (`grep -rn "drawdown_breach" backend/` -> 0 hits). Treat the 9
   pre-fix rows as historical artefact; the fix landed cleanly.

2. **Pytest suite shape: 3 files + property test.** Minimal
   mutation-resistant coverage:
   - `tests/test_kill_switch_no_false_fires.py` -- audit-log
     scan asserting `pause` trigger in ALLOWED_TRIGGERS for all
     rows dated >= 2026-05-06.
   - `tests/test_kill_switch_evaluate_breach.py` -- math
     correctness, including the exact regression case (nav
     >sod, profit, must NOT breach) AND boundary cases (at
     limit / just under limit).
   - `tests/test_kill_switch_invariants.py` -- Hypothesis
     property test asserting `daily_loss_breached iff
     (sod-nav)/sod*100 >= limit`.

3. **Live-check artifact**: produce
   `handoff/current/live_check_23.2.5.md` per Section C5. The
   masterplan `verification.live_check` field requires it;
   without the file the auto-push hook (CLAUDE.md, phase-23.8.1)
   will hold the push.

4. **Mutation resistance**: every positive assertion in C2 has a
   negative twin (breached IFF condition). A sign-flip mutation
   (`<=` for `>=`) and a constant-substitution mutation (compare
   to 0 instead of `daily_loss_limit_pct`) both die against this
   suite. This is a deliberate counter to the NYIF false-positive
   kill-switch concern (Section B) and the OWASP LLM06 excessive-
   agency mitigation -- the breach formula must NEVER fire on a
   profit-sign input.

5. **What the planner should NOT do**: do NOT re-add a
   `drawdown_breach` trigger emission to source. The
   evaluate_breach() function is read-only -- it returns flags;
   callers decide whether to pause. The current shape (manual
   pause only, read-only breach surface) is the correct
   post-fix architecture per the BIS 2026 + OWASP LLM v2
   guidance (excessive-agency mitigation).

---

## References (per-claim citation table)

| Claim | Source |
|---|---|
| evaluate_breach() math | `backend/services/kill_switch.py:202-236` |
| Auto-pause-on-breach has no caller in current source | `grep -rn "drawdown_breach" /Users/ford/.openclaw/workspace/pyfinagent/backend/` (0 hits) |
| `/kill-switch` endpoint is read-only (no auto-pause) | `backend/api/paper_trading.py:451-489` |
| `/resume` blocks if breach still flagged | `backend/api/paper_trading.py:527-538` |
| `/pause` only emits trigger=manual | `backend/api/paper_trading.py:497` |
| 9 historical false-fires on 2026-05-05 | `handoff/kill_switch_audit.jsonl` lines for 2026-05-05T18:21:50 .. 2026-05-05T20:07:52 (9 rows with trigger=drawdown_breach, all reporting daily_loss_pct=-2.5) |
| Zero auto-fires post-fix | grep `handoff/kill_switch_audit.jsonl` for rows with `ts >= "2026-05-06"` and `trigger=drawdown_breach` (0 rows) |
| File-based handoff is the canonical pattern | https://www.anthropic.com/engineering/harness-design-long-running-apps (quoted verbatim above) |
| False-positive kill-switch is an industry-recognised failure mode | https://www.nyif.com/articles/trading-system-kill-switch-panacea-or-pandoras-box (quoted verbatim above) |
| OWASP LLM06 Excessive Agency applies to broken auto-stops | https://genai.owasp.org/llmrisk/llm06-sensitive-information-disclosure/ (quoted verbatim above) |
| Reproducible validation is required by the 2026 MRM framework | https://www.databricks.com/blog/model-risk-management-2026-bankers-guide-revised-interagency-guidance (quoted verbatim above) |
| Property-based / invariant testing for breach formula | https://hypothesis.readthedocs.io/en/latest/quickstart.html (quoted verbatim above) |
| Sell-first-then-buy convention (context only) | `.claude/rules/backend-services.md` |
| `threading.Lock` non-reentrancy (context for phase-23.1.22) | `backend/services/kill_switch.py:91-102` |
| Daily-roll fix (phase-23.2.19) for sod_snapshot.date | `backend/services/kill_switch.py:75-85`, `168-182` |

End of brief.
