# Experiment Results: Phase 4.2.3.1 Formatter Hardening

**Step:** Phase 4.2.3.1 -- SN1 + SN2 fixes to `format_accuracy_report`
**Date:** 2026-04-14
**Generator:** Ford Lead Opus 4.6 (in-session)
**Research gate:** PASSED (`handoff/current/research.md`, 20+ URLs / 7 categories)
**Contract:** `handoff/current/contract.md` (25 SCs)
**Target file:** `backend/slack_bot/formatters.py`

## Delta

```
backend/slack_bot/formatters.py | 14 +++++++++++---
 1 file changed, 11 insertions(+), 3 deletions(-)
```

**Diff budget**: 11 added / 3 deleted, well under the 20/5 contract cap.

## Changes

### SN1 fix -- `_coerce_float` sanitizes IEEE 754 non-finite values

```python
# Before
def _coerce_float(d: dict, key: str) -> float:
    try:
        return float(d.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0

# After
def _coerce_float(d: dict, key: str) -> float:
    try:
        v = float(d.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    # Phase 4.2.3.1 SN1 fix: sanitize NaN / +Inf / -Inf at the display
    # boundary so upstream IEEE 754 non-finite values never render as
    # "nan%" or "inf%" in the Slack fields. See handoff/current/research.md.
    return v if math.isfinite(v) else 0.0
```

Added `import math` at the top of the file (two-line import block:
`import math` + `from datetime import datetime`).

### SN2 fix -- n=0 branch collapses mean/median to "Scoring pending"

```python
# Before
if scored_count <= 0:
    fields = [
        _field("Total signals", f"{total_count:,}"),
        _field("Hit rate", "Scoring pending"),
        _field("Mean forward return", mean_str),
        _field("Median forward return", median_str),
    ]

# After
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
```

## Smoke Results (lead-self, 25 contract SCs)

| SC block | Count | Result |
|---|---|---|
| SC1-5: SN1 `_coerce_float` non-finite -> 0.0 | 5 | PASS |
| SC6-7: NaN/Inf sanitized in mean/median field render | 2 | PASS |
| SC8: NaN CI bound sanitized | 1 | PASS |
| SC9: group NaN sanitized | 1 | PASS |
| SC11-14: n=0 three metric fields "Scoring pending" | 4 | PASS |
| SC15: n=1 still renders real percents (gate) | 1 | PASS |
| SC16-17: 11 functions byte-identical to `eeea983` | 11 | PASS |
| SC20: imports = `import math` + `from datetime import datetime` | 1 | PASS |
| SC23: 10 adversarial inputs never raise | 10 | PASS |

Byte-identity-verified functions: `_truncate`, `_score_emoji`, `_rec_color`,
`format_analysis_result`, `format_portfolio_summary`, `_signal_emoji`,
`format_signal_alert`, `format_report_card`, `format_morning_digest`, `_pct`,
`_coerce_int`.

`_coerce_float` byte-identity is expected to diverge (SN1 change).
`format_accuracy_report` byte-identity expected to diverge inside the
`scored_count <= 0` branch only (SN2 change); all other branches unchanged
semantically, confirmed via SC15 smoke on n=1 fixture and SC6-7 smoke on
n=12 fixture with NaN/Inf inputs.

## Not run yet (QA-only scope)

- Fresh-process re-run of all 25 SCs by independent Opus qa-evaluator
- Adversarial probes designed by QA (target >= 10, unique from lead's)
- AST sub-branch byte-identity audit of `scored_count >= 1` branches
- Cross-server import scan
- Non-ASCII string literal audit in changed code

## Observations

- **`math.isfinite` import footprint is effectively free.** `math` is a
  C-extension stdlib module pre-loaded in any CPython process; the added
  import adds a single line and zero transitive dependencies.
- **"Scoring pending" repetition in n=0 branch is deliberate.** Four
  fields now have identical placeholder text. Considered collapsing to
  a single section block with "No data available yet", but that would
  violate the Phase 4.2.3 field-shape invariant (fields array must be
  EVEN, <= 10) and change the AST shape of the branch substantially,
  widening the blast radius beyond the SN2 fix.
- **Researcher subagent failure mode confirmed.** Spawned the researcher
  subagent per protocol; it hit "Stream idle timeout - partial response
  received" after ~5m / 47 tool uses, writing nothing to research.md.
  Fell back to in-session WebSearch (9 queries) and wrote research.md
  directly. Same failure mode Peder documented at 21:21 CEST.
  Per his advice: "default to in-session WebSearch for research gates;
  reserve subagents for QA-only (AST / fresh-process verification) work".
  Followed exactly.

## Next gate (EVALUATE)

Spawn independent Opus qa-evaluator subagent with anti-leniency brief:
1. Re-run all 25 contract SCs in a fresh Python subprocess.
2. Design >= 10 adversarial probes (unique from lead's smoke).
3. AST byte-identity audit against `origin/main` head `eeea983`.
4. Verify exactly one new import (`math`), no third-party additions.
5. Audit diff bound (11/3 vs 20/5 cap).
6. Write verdict to `handoff/current/evaluator_critique.md`.

Output JSON: `{ok, reason, checks_run, violated_criteria, soft_notes, scores}`.
