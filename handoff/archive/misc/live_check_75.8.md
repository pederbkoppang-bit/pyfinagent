# live_check -- Step 75.8 (gap6-01, gap6-10, gap3-02)

Date: 2026-07-23. All outputs below are verbatim captures from the live
repo; exit codes captured via `rc=$?` immediately after the command
(cycle-135 pipeline-trap rule).

## 1. Immutable verification command (exit 0)

```
$ cd /Users/ford/.openclaw/workspace/pyfinagent && .venv/bin/python -m pytest backend/tests/test_phase_75_promotion_gate.py -q
....................                                                     [100%]
20 passed in 0.10s
verification_exit=0
```

## 2. Change surface

```
$ git diff --stat HEAD -- backend/ scripts/ .claude/masterplan.json
 .claude/masterplan.json        |  20 ++++++++
 backend/main.py                |  12 ++++-
 scripts/risk/gauntlet.py       |  41 +++++++++++++++--
 scripts/risk/promotion_gate.py | 101 +++++++++++++++++++++++++++++++----------
 4 files changed, 144 insertions(+), 30 deletions(-)
```
New untracked code files: `backend/governance/divergence.py`,
`backend/tests/test_phase_75_promotion_gate.py`.
`backend/backtest/gauntlet/evaluator.py`, kill-switch code, DSR/PBO
constants, `backend/governance/limits.yaml`: NOT in the diff (criterion 5).

## 3. Live CLI refusal -- gauntlet non-dry-run fabricates nothing ($0)

```
$ .venv/bin/python scripts/risk/gauntlet.py --strategy baseline --seed 42
NotImplementedError: gauntlet live mode is not implemented -- the only data source is the dry-run stub. Re-run with --dry-run, or wire the real backtest engine before requesting a live report.
python_exit=1
```
`handoff/gauntlet/baseline/report.json` before AND after: mtime `10 jun. 15:50`
(unchanged); `git status --short handoff/gauntlet/` empty.

## 4. Live divergence measurement (current repo values, $0, pure read)

```
$ .venv/bin/python -c "from backend.governance.divergence import compute_divergence; import json; print(json.dumps(compute_divergence(), indent=1))"
[
 {"name": "daily_loss_kill_switch",  "settings_value_pct": 4.0,  "governed_value_pct": 2.0,  "divergent": true},
 {"name": "trailing_dd_kill_switch", "settings_value_pct": 10.0, "governed_value_pct": 10.0, "divergent": false}
]
```
(abridged to the four load-bearing keys per pair; full dicts include
settings_attr/governed_attr -- see test_divergence_flags_daily_loss_and_clears_trailing_dd)

## 5. Flag-gated / UI note

No flag-gated live-loop behavior is introduced (the lifespan WARNING is
unconditional, fail-open, log-only -- a $0 no-op for the trading loop by
construction; it first fires on the next operator restart). No UI surface
touched -> no Playwright capture required.
