# phase-86.12 -- CONTRACT

**Step:** P1 -- DETERMINE WHETHER THE KILL SWITCH EVALUATES DRAWDOWN AGAINST A
STALE NAV: kill-switch `current_nav` EQUALS `sod_nav` exactly, and disagrees
with the NAV the cockpit renders.

**Research gate:** PASSED -- `wf_b6b1e4e3-df2`, tier `moderate`, **7 sources
read in full**, 30 URLs collected, 23 snippet-only, recency scan performed, 8
internal files inspected. Brief: `handoff/current/research_brief_86.12.md`
(32,213 chars). The gate script independently re-read the brief and confirmed
all 7 claimed URLs appear in it.

**This is an INVESTIGATION step.** The deliverable is a measured answer, not a
behaviour change. Criterion 5 is explicit: if the behaviour is correct, say so
plainly and record the reasoning so the suspicion is not re-raised; if it is a
defect, **no threshold is changed as part of the diagnosis** -- the fix is its
own scope.

---

## 1. Research-gate summary (what the literature says the right shape is)

- **BCBS 239** principles P5 (timeliness) and P3(d): risk aggregation must
  produce data that is timely *and* the timeliness must itself be measurable.
  `https://www.bis.org/publ/bcbs239.pdf`
- **Lo & MacKinlay-lineage nonsynchronous-trading result (JFE 2004)**: forming
  a ratio from two quantities observed at different instants biases the
  measured variance **downward**. Applied here, a limit computed from a
  nonsynchronous `(sod_nav, current_nav)` pair fires **LATE**, not early --
  which is the dangerous direction for a loss limit.
  `http://web.mit.edu/Alo/www/Papers/JFE2004Pub.pdf`
- **dbt source freshness**: the standard pattern is to carry an `asof` with the
  value and assert a max age at the point of USE, not at the point of write.
  `https://docs.getdbt.com/docs/deploy/source-freshness`
- **Google SRE, Monitoring Distributed Systems**: staleness must be an
  explicit, alertable signal rather than an invisible property.
  `https://sre.google/sre-book/monitoring-distributed-systems/`
- **SEC 15c3-5** and **MiFID II RTS 6** mandate pre-trade risk limits but are
  **silent on mark freshness** -- so the regulatory floor does not settle this;
  the engineering literature does.
- Adversarial/recency note from the brief: several 2026 prop firms are
  *removing* fixed daily-loss limits. Recorded so this step does not treat "a
  daily-loss limit is obviously correct" as unexamined.

## 2. Hypothesis

`current_nav` on every kill-switch path is a **stored BigQuery figure**
(`paper_portfolio.total_nav`), never a live mark. The observed
`sod_nav == current_nav` equality is therefore the *expected* consequence of the
SOD roll anchoring to the same stored value that the breach then reads, and not
by itself a defect. The real finding, if any, is an **asymmetry**: the code
checks whether the *baseline* is stale but performs no freshness check at all on
`current_nav`, so the two halves of the ratio may be observed at different
instants with nothing detecting it.

## 3. Immutable success criteria -- copied VERBATIM from `.claude/masterplan.json`

1. the PROVENANCE of current_nav is traced to its source at file:line -- state
   exactly which call produces it and whether that call returns a live mark or a
   stored start-of-day figure; a description of the endpoint is not a trace
2. the sod_nav == current_nav equality is tested ACROSS MULTIPLE DAYS (the audit
   journal holds 8 sod_snapshot rows) rather than from one observation -- report
   whether it holds always, sometimes, or was a one-off
3. the $0.06 delta between the kill-switch current_nav and the cockpit-rendered
   NAV is EXPLAINED (rounding, FX timestamp, different endpoint, different asof)
   -- an unexplained delta between two numbers describing the same quantity is
   itself the finding
4. the safety question is answered directly: CAN the daily-loss leg fire on an
   intraday drawdown as currently wired? Demonstrate with a test driving
   evaluate_breach through the real production path, not by reasoning about the
   code
5. if the behaviour is CORRECT, the step closes with that stated plainly and the
   reasoning recorded, so the suspicion is not re-raised later; if it is a
   defect, no threshold is changed as part of the diagnosis -- the fix is its own
   scope
6. no guard weakened, no threshold touched, live handoff/kill_switch_audit.jsonl
   byte-identical across the investigation

**Immutable verification command** (unmodified):

```
bash -c 'curl -s -m 10 http://localhost:8000/api/paper-trading/kill-switch | python3 -c "..."'
```
(prints `sod_nav` and `current_nav`.)

## 4. Traps this step must not fall into

- **The step's own numbers are not to be trusted.** It says "8 sod_snapshot
  rows"; I have already measured **10**. Every figure gets re-derived, and where
  mine disagrees with the step text I report the disagreement rather than
  silently using mine.
- **The $0.06 delta may not reproduce.** First measurement tonight shows the
  kill-switch `current_nav` and `/api/paper-trading/performance` `nav` are
  **identical** (23833.94 both). A criterion that asks me to EXPLAIN a delta
  cannot be satisfied by inventing one; the honest answer may be "not
  reproducible now, and here is why it would appear transiently".
- **A "one-off" answer to criterion 2 must not be assumed from a single
  reading.** The equality has to be examined against the actual multi-day
  history and the conditions that produce it.
- **Criterion 4 says demonstrate, not reason.** A test must drive
  `evaluate_breach` through the real production path. Asserting on my reading of
  the code is exactly what the criterion forbids.
- **Do not fix anything.** If this is a defect, the fix is a separate step. The
  temptation to "just add the freshness check" is out of scope and criterion 5
  forbids it.
- **The live journal must be byte-identical afterwards** (criterion 6). The
  phase-86.6 preventer is now active and will refuse a stray write, but the
  digest is taken before and after regardless.

## 5. Plan

1. **Provenance (criterion 1)** -- trace every producer of `current_nav` to
   file:line and classify each as live-mark or stored. The research brief names
   five; verify each independently rather than transcribing.
2. **Multi-day equality (criterion 2)** -- read every `sod_snapshot` row from
   the live journal, pair each with what the portfolio NAV was at that time, and
   report always/sometimes/one-off with the conditions.
3. **The delta (criterion 3)** -- measure the kill-switch NAV against every
   endpoint that renders a NAV; if there is no delta now, explain the mechanism
   by which one appears and what would make it reappear.
4. **The safety answer (criterion 4)** -- build a test that drives
   `check_and_enforce_kill_switch` / `evaluate_breach` through the production
   path with a simulated intraday drawdown, and report whether the leg fires.
   Include the negative control (no drawdown -> no fire).
5. **State the verdict plainly (criterion 5)** -- correct or defect, with the
   reasoning recorded so it is not re-litigated.
6. **Digest the journal before and after (criterion 6).**
7. Q/A via the Workflow rail; transcribe the verdict verbatim; log; flip.

## 6. References

- `handoff/current/research_brief_86.12.md` (gate artifact, 7 sources in full)
- `backend/services/kill_switch.py`, `backend/services/paper_trader.py`,
  `backend/api/paper_trading.py`, `backend/agents/mcp_servers/risk_server.py`
- `handoff/kill_switch_audit.jsonl` (10 `sod_snapshot` rows, measured)
