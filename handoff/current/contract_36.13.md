# Contract — masterplan step 36.13

**[P0] `execute_buy` has no kill-switch gate, so the MCP signals path places orders while the switch
is PAUSED or DISARMED.** CWE-424 alternate path.

Step id: `36.13` · Phase: PLAN · Date: 2026-07-26 · HEAD at contract time: `902887f9`

## Immutable success criteria (copied VERBATIM from `.claude/masterplan.json`)

1. A test reproduces the bypass FIRST: with the kill switch PAUSED, the signals_server publish_signal BUY path still places a paper trade -- record that failing-intent output verbatim before fixing
2. A second test reproduces it for the DISARMED case (baselines unrestorable, armed False) -- also recorded before fixing
3. After the fix, both paths refuse, and the refusal is audited/observable rather than a silent no-op return
4. The call-site inventory is RE-DERIVED in the artifacts by the exact grep commands (not hand-counted), and every non-test caller is dispositioned explicitly as gated / deliberately-bypassing / out-of-scope
5. `scripts/go_live_drills/zero_orders_drill.py` and `scripts/smoketest_stages_5_through_13.py` still work as designed -- assert this explicitly; a drill that can no longer place its probe order is a regression, not a fix
6. The gate FAILS CLOSED: an unreadable or unknown kill-switch state blocks the order rather than allowing it -- a test proves it
7. Kill-switch thresholds, limits and the pause/resume API surface are byte-untouched -- diff must show no change to limit values
8. MUTATION-TEST every new guard, including the drill bypass if one is added (removing the bypass must fail the drill test; removing the gate must fail the bypass test)

Immutable verification command:
```
source .venv/bin/activate && python -m pytest backend/tests/ -q -k 'kill_switch or paper_trader or signals_server'
```
Immutable `live_check`: verbatim test output for the reproduce-then-fix pair on BOTH the paused and
the disarmed case, plus the re-derived call-site grep output showing every non-test `execute_buy`
caller and its disposition.

## Research-gate summary — it changed my design, which is why the gate exists

`handoff/current/research_brief_36.13.md` — `gate_passed: true`, **9 sources read in full** (floor 5),
28 URLs, recency scan performed, 11 internal files inspected.

I went in intending **(c)**: rename the primitive to `_execute_buy_unchecked` and add a guarded
`execute_buy` wrapper. The gate recommended **(a′)** — one gate *inside* `execute_buy`, **no bypass
parameter**, with the escape hatch provided by **dependency injection** — and it is right:

- **My escape-hatch premise was false.** Both "deliberate" callers are already fully stubbed and never
  touch the live book: `zero_orders_drill.py` builds `PaperTrader(settings=..., bq_client=StubBQ())`
  (verified at :92), and the smoketest passes `MagicMock()`. They do not need to bypass a safety
  control; they need to **supply their own state** — exactly what they already do for BigQuery.
- **CWE-424's parent is CWE-638 "Not Using Complete Mediation"**, whose remedy is stated directly:
  *"create and use a single interface that performs the access checks."* CWE-288: *"funnel all access
  through a single choke point."* That is (a′), not (b).
- **(c) costs more than it buys in Python.** The underscore is convention-only, and the rename is
  **19 call sites, not 4** (15 test sites across 8 files, including a stub `execute_buy` at
  `test_phase_70_3_atomic_swap.py:36`). A reachable-but-retired path is what Knight Capital cost
  $460M in 45 minutes.
- Flag-argument literature (Fowler) rules out the `force=True` / `skip_checks=True` shape outright.

**The trap the gate found, which I verified by execution rather than accepting:** `kill_switch._state`
is a module-level singleton whose `__init__` ends in `_load_from_audit()`, and the replay sets
`_paused = True` from any `pause` row. I ran it: a fresh process with one pause row in the trail
reports `is_paused() == True`. So a naive `get_state().is_paused()` gate would break both drills
**intermittently — only on days the book happens to be paused**, which is the worst possible failure
schedule. Injection is what makes the drills deterministic, not what weakens them.

**Disclosed gaps carried forward:** IEC 61511 clause numbers come from an industry summary (the
standard is paywalled); the researcher derived pause-durability by reading `kill_switch.py:254-268`
and I converted it to an executed measurement above.

## Re-derived call-site inventory (criterion 4 — commands, not hand-counts)

```
$ grep -rn '\.execute_buy(' --include='*.py' backend scripts | grep -v /tests/
backend/agents/mcp_servers/signals_server.py:444
backend/services/autonomous_loop.py:236
scripts/smoketest_stages_5_through_13.py:188
scripts/go_live_drills/zero_orders_drill.py:94

$ grep -rn 'is_paused()' --include='*.py' backend scripts | grep -v /tests/
backend/services/paper_trader.py:1197
backend/services/autonomous_loop.py:1316
```

Corrections to the step's own text, measured: `autonomous_loop.py:236` (step said `:207`),
`paper_trader.py:1197` / `autonomous_loop.py:1316` (step said `:1097` / `:1287`). `risk_server.py:91,181`
is a **third** paused-reader via the snapshot key rather than `is_paused()`. Note `--include=*.py`
needs quoting under zsh or the glob is eaten.

| Caller | Disposition |
|---|---|
| `signals_server.py:444` | **GATED** — the bypass this step closes |
| `autonomous_loop.py:236` | **GATED** — already behind `check_and_enforce_kill_switch`; the gate is now belt-and-braces |
| `zero_orders_drill.py:94` | **DELIBERATELY BYPASSING** via injected stub state, named and WARN-logged |
| `smoketest_stages_5_through_13.py:188` | **DELIBERATELY BYPASSING**, same mechanism |

## Hypothesis

The kill switch is enforced on the *cycle* rather than on the *act of buying*. Moving the check to the
single choke point every BUY must pass — `execute_buy`, one of only **two** `paper_trades` producers
(`:324` buy, `:510` sell; `portfolio_manager.py` never writes a trade) — makes BUY mediation complete
without touching a threshold.

## Plan

**The gate, inside `execute_buy`, at the top, before any order work.** Refuse when the book is
**paused**, and when the baselines are **genuinely lost**. Fail CLOSED: any exception reading the
state refuses the order (criterion 6).

**WHICH FLAG — and this is the lesson 36.9 just paid for.** The gate reads `baselines_present`, **not**
`armed`. Since 36.9, `armed` also means "can fire right now", which is False for an anchor that is
merely from yesterday — and gating BUYs on that would refuse every order each morning until the daily
roll, re-creating the exact regression 36.9's cycle-1 Q/A caught on the order path. Lost baselines are
a durable fault; an overnight anchor is not. Staleness must not block trading.

**`execute_sell` is deliberately NOT gated, and that asymmetry is load-bearing.**
`check_and_enforce_kill_switch` performs its flatten *through* `execute_sell` (`:736`, `:764`,
`:1029`), so gating sells on `paused` would deadlock the pause: the switch could never close the
positions it just decided to close. Selling is the safe direction. This is stated so a future reader
does not "fix" the asymmetry.

**Observability, not a silent return (criterion 3).** Reuse the existing refusal vocabulary rather
than inventing one: `self.buy_rejections.append({...})` (init `:107`, sole producer `:222-226`,
consumed by `autonomous_loop.py:1574-1582`), plus an ERROR log naming the cause. A refusal is thus
attributable at the cycle-summary layer exactly like the existing price-tolerance rejection.

**The injection seam.** `PaperTrader.__init__` gains `kill_switch_state=None`, resolved as
`kill_switch_state or get_state()` — the house idiom, matching `execution_router.py:268-269`
(`mode or _current_mode()`, whose docstring already cites Fowler on ops toggles). The drills pass
their own state object.

**Mitigating the counter-argument, which I accept rather than dismiss.** The gate's own strongest
objection to (a′) is that injection is a *silent* bypass, visible only at the construction site
instead of at the call site. Three mitigations, all mutation-tested, or the objection wins:
1. Any non-default `kill_switch_state` **WARN-logs at construction**, naming itself as a test seam.
2. A **source-scan guard** asserts no module under `backend/` passes `kill_switch_state=` — production
   code cannot inject, only tests and `scripts/`.
3. The refusal reason distinguishes `paused` from `baselines_lost` so the audit trail says which.

**Settings access must be defensive.** The smoketest constructs a 6-attribute `SimpleNamespace` as
settings, so any config read in the gate uses `getattr(self.settings, ..., default)` or it dies with
`AttributeError` on a caller this step is required to keep working (criterion 5).

## Out of scope → their own steps

- `signals_server.py` carries a **weaker duplicate** control: its own in-memory peak (`:88-89`, resets
  every restart) feeding a `drawdown_circuit_breaker` at `:950`. That is why this gap looked covered.
  Reconciling the duplicate with the real kill switch is a separate step.
- `scripts/go_live_drills/kill_switch_test.py` **does not test the kill switch** — its scenarios
  (`:72-129`) drive that signals_server duplicate. Its name asserts coverage the repo does not have.

## Do-no-harm

Paper trading only. No `.env` edits, no flag flips, `historical_macro` frozen, no optimizer runs.
Kill-switch **thresholds, limits and the pause/resume API surface byte-untouched** (criterion 7) — this
step adds a gate, it never changes when the switch trips. No peak reset.
`handoff/kill_switch_audit.jsonl` md5 `ce8fb93348bb9a3bbe26f2d91b1bc05e` verified before and after any
run that could write it. `:8000` GET-only. `:3000` never driven. The drills and smoketest are NOT run
against the live book — they place orders.

## References

- `handoff/current/research_brief_36.13.md` (envelope, source tables, recency scan)
- [CWE-424 Improper Protection of Alternate Path](https://cwe.mitre.org/data/definitions/424.html),
  [CWE-638 Not Using Complete Mediation](https://cwe.mitre.org/data/definitions/638.html),
  [CWE-288](https://cwe.mitre.org/data/definitions/288.html)
- Saltzer & Schroeder, *The Protection of Information in Computer Systems* (complete mediation)
- [Fowler — Flag Arguments](https://martinfowler.com/bliki/FlagArgument.html)
- Internal: `paper_trader.py` (`__init__` :94-107, `execute_buy` :123-260, `buy_rejections` :222-226,
  `paper_trades` producers :324/:510), `signals_server.py:444` + `:88-89` + `:950`,
  `autonomous_loop.py:236` + `:1316` + `:1574-1582`, `execution_router.py:268-269`,
  `kill_switch.py:254-268` (pause replay) + `:675` (singleton)
