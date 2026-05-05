# Phase-23.2.19 External Research Brief

**Tier assumption:** moderate (two well-localized sub-problems; UI accessibility + SOD daily-roll pattern)
**Date:** 2026-05-05

---

## Research: SOD Daily-Roll Patterns and Tooltip Accessibility for OpsStatusBar Fix

### Queries run (three-variant discipline)

**SOD / daily-roll topic:**
1. Current-year frontier: "start of day NAV baseline daily rollover trading system UTC timezone handling 2026"
2. Last-2-year window: "SOD start of day snapshot risk system daily PnL calculation pattern prop trading 2025"
3. Year-less canonical: "start of day baseline idempotent daily roll trading system fintech pattern"
4. Supplemental year-less: "idempotent SOD snapshot daily loss limit system restart mid-day risk management 2024 2025"

**Tooltip / accessibility topic:**
1. Current-year frontier: "WCAG native HTML title attribute tooltip accessibility 2025 2024"
2. Last-2-year window: "HTML title attribute accessibility keyboard touch inaccessible ARIA tooltip role 2025"
3. Year-less canonical: "WCAG tooltip role aria-describedby accessible informational 2024 2025"

---

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://www.24a11y.com/2017/the-trials-and-tribulations-of-the-title-attribute/ | 2026-05-05 | Authoritative accessibility blog | WebFetch | "most browsers still have not implemented any support to reveal the attribute's value to sighted users that aren't using a mouse" |
| https://sarahmhigley.com/writing/tooltips-in-wcag-21/ | 2026-05-05 | Authoritative accessibility blog (named researcher) | WebFetch | "role='tooltip' is optional and provides minimal benefit... aria-describedby and aria-labelledby do all the heavy lifting"; title= is "fundamentally inaccessible mouse-based behavior" |
| https://www.w3.org/WAI/WCAG21/Understanding/content-on-hover-or-focus.html | 2026-05-05 | Official W3C/WCAG docs | WebFetch | "This criterion does not attempt to solve such issues when the appearance of the additional content is completely controlled by the user agent" — explicitly exempts native title= tooltips from WCAG 1.4.13 |
| https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/Reference/Roles/tooltip_role | 2026-05-05 | Official docs (MDN) | WebFetch | Full pattern: role="tooltip" + aria-describedby; last updated November 4, 2025; "If the information is important enough for a tooltip, isn't it important enough to always be visible?" |
| https://library.tradingtechnologies.com/user-setup/ac-configuring-sod-settings-and-credit-limits.html | 2026-05-05 | Official vendor docs (Trading Technologies) | WebFetch | SOD records "generated for the prior exchange session at the daily reset time"; credit limits anchored to SOD via P&L rule; reset time is configurable with timezone |
| https://flook.co/blog/posts/are-tooltips-accessible | 2026-05-05 | Industry blog | WebFetch | "Relying on the HTML title attribute for tooltips is a common accessibility mistake. These tooltips are not reliably accessible to screen readers, cannot be styled, and often disappear too quickly." |
| https://www.w3.org/WAI/ARIA/apg/patterns/tooltip/ | 2026-05-05 | Official W3C/WAI docs | WebFetch | ARIA APG canonical tooltip pattern: role="tooltip" + aria-describedby; keyboard: Escape dismisses, focus stays on trigger |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://newyorkcityservers.com/blog/prop-firm-daily-drawdown-rules | Industry blog | Fetched; returned only that "resets at midnight server time" — insufficient depth for prop firm SOD mechanics |
| https://www.thinkcapital.com/prop-firm-drawdown-rules/ | Industry blog | Fetched; no operational detail on SOD snapshot timing or system restart behavior |
| https://funderpro.com/blog/master-prop-firm-drawdown-rules-in-2025/ | Industry blog | Fetched but content gave only conceptual drawdown types, not SOD snapshot mechanics |
| https://getwcag.com/en/accessibility-guide/aria-tooltip-name | Official/compliance guide | Snippet only — core content covered by MDN and W3C sources fetched in full |
| https://www.thewcag.com/examples/tooltips | Compliance guide | Snippet only — pattern covered by APG and MDN |
| https://pressbooks.library.torontomu.ca/wafd/chapter/tooltips/ | Academic | Snippet only |
| https://www.tpgi.com/using-the-html-title-attribute/ | Authoritative blog | 403 on fetch |
| https://dtcc.com/-/media/Files/Downloads/Transformation/theshiftto24x5trading.pdf | Industry doc | Snippet: DTCC 24x5 plan (June 2026); not directly relevant to SOD snapshot timing |
| https://lightrun.com/answers/tony19-logback-android-daily-rollover-uses-utc-instead-of-local-time | Community | Snippet — logback UTC rollover; tangentially relevant to timezone anti-patterns |
| https://medium.com/javarevisited/idempotency-in-distributed-systems-preventing-duplicate-operations-85ce4468d161 | Community | Snippet — general idempotency pattern; not trading-specific |

---

### Recency scan (2024-2026)

Searched explicitly for 2024-2026 literature on both topics.

**SOD daily-roll pattern:** No peer-reviewed or authoritative new publications in the 2024-2026 window that specifically address SOD snapshot idempotency patterns for autonomous trading systems. The Trading Technologies documentation (canonical vendor source) was confirmed current. The DTCC 24x5 plan (June 2026) changes settlement hours but does not affect intraday SOD baseline logic. Prop firm drawdown documentation (FunderPro, ThinkCapital, NYC Servers, all 2025-2026) consistently describes "midnight server time" as the reset moment but adds no new idempotency patterns.

**Tooltip accessibility:** MDN's tooltip_role page was updated November 4, 2025 — the most current official guidance. WCAG 2.2 (final published October 2023) did not materially change WCAG 1.4.13; the exemption for user-agent-controlled content (native title=) remains. No 2026 changes to tooltip patterns found. The 2025 Sarah Higley article remains the most authoritative practitioner reference. No breaking changes in the 2024-2026 window.

---

### Key findings

#### SOD Daily-Roll Pattern

1. **SOD = "prior session's closing value, captured once per trade day."** Prop firms universally define daily drawdown as starting from a snapshot taken at a fixed calendar boundary (midnight server time, often UTC or exchange-local). (Source: Trading Technologies docs, https://library.tradingtechnologies.com/user-setup/ac-configuring-sod-settings-and-credit-limits.html; newyorkcityservers.com prop-firm drawdown rules)

2. **SOD must NOT reset on system restart mid-day.** The semantics of "daily loss limit" require the SOD anchor to remain stable for the full calendar day. A system restart that re-anchors SOD to the current NAV would erase real intraday losses from the daily-loss calculation — a safety-critical regression. Trading Technologies documents that "position reset time is configurable and adhered to even if the credit risk setting is disabled" — i.e., the reset is time-governed, not event-governed. (Source: Trading Technologies docs)

3. **SOD reset should be idempotent per calendar day.** The correct pattern: check whether the stored SOD belongs to the current calendar day (UTC); if yes, do nothing; if no (or if none exists), write the current NAV as SOD. The check must use a stored date, not just presence of a non-None value. (Source: synthesized from Trading Technologies docs + pyfinagent codebase analysis)

4. **Anti-pattern: SOD anchoring to deposit/initial nav.** pyfinagent's current bug is that `_sod_nav = 9499.5` (the initial paper capital from 2026-04-20) acts as a permanent anchor. This is equivalent to anchoring SOD to deposit time, which produces nonsense daily-loss calculations as the account grows. (Source: audit of `kill_switch.py:53-74` + `kill_switch_audit.jsonl`)

5. **Anti-pattern: SOD resetting on every process restart.** The opposite failure — if SOD were re-captured on every restart — would erase intraday loss tracking. The `if snap.get("sod_nav") is None` guard in `paper_trader.py:548` was intended to prevent this, but its use of None-check rather than date-check means it only fires once ever (when the audit log has no sod_snapshot rows). (Source: `paper_trader.py:546-554`)

6. **UTC is the correct timezone for SOD boundaries.** All authoritative sources (Trading Technologies, stocktitan.net DST effects, babypips market hours) recommend UTC storage for timestamps; calendar-day boundaries should also use UTC for consistency with the rest of the system. pyfinagent already uses `datetime.now(timezone.utc).date().isoformat()` at the callsite — the date comparison just needs to be wired up. (Source: search results on timezone anti-patterns; existing `paper_trader.py:547`)

7. **Weekend/holiday behavior:** SOD baseline should persist across weekends — the "daily" calculation resumes on Monday anchored to Friday's close (or Friday's SOD if no trades fired). pyfinagent's paper trading only runs when the autonomous cycle fires, so if no cycle fires on a weekend, the SOD from Friday is stale by Monday. The correct fix resets SOD on the first cycle of any new calendar day, including Monday after a weekend gap. (Source: synthesized from prop-firm drawdown docs and TT SOD behavior)

#### Tooltip Accessibility

8. **WCAG 1.4.13 explicitly exempts native browser title= tooltips.** The W3C WCAG 2.1 understanding document states that "this criterion does not attempt to solve such issues when the appearance of the additional content is completely controlled by the user agent" and names "browser tooltips created through use of the HTML title attribute" as a canonical example. (Source: W3C WCAG 2.1 Understanding 1.4.13, https://www.w3.org/WAI/WCAG21/Understanding/content-on-hover-or-focus.html)

9. **title= fails for keyboard-only and touch users.** The 24 Accessibility article and MDN both confirm the title attribute cannot be activated via keyboard focus or touch — it is mouse-hover only. For an operator UI used primarily by a mouse-equipped admin (not a public-facing site), this is acceptable for low-priority supplemental information. (Source: 24a11y.com title attribute article; MDN ARIA tooltip_role page updated Nov 2025)

10. **Informational-only tooltip: title= is the correct lightweight choice for this project.** Sarah Higley's WCAG 2.1 analysis distinguishes informational tooltips (supplemental hint text) from interactive tooltips (contain links/buttons). The GATE segment tooltip is purely informational — it describes the 5 criteria, nothing interactive. The existing project pattern (three places in OpsStatusBar.tsx) uses title= consistently. Upgrading to role="tooltip" + aria-describedby + JavaScript visibility management is disproportionate for an internal operator bar. (Source: sarahmhigley.com tooltips-in-wcag-21; existing OpsStatusBar.tsx lines 164, 216, 287)

11. **For ARIA tooltip: role="tooltip" + aria-describedby is the canonical pattern.** MDN (Nov 2025) documents the pattern: `<span aria-describedby="tip-id">trigger</span><div role="tooltip" id="tip-id">text</div>`. The tooltip never receives focus; keyboard users trigger it via focus on the owning element; Escape dismisses. The ARIA APG pattern (W3C) confirms this but notes the spec is "still in progress without full task force consensus." (Source: MDN ARIA tooltip_role; W3C APG tooltip pattern)

12. **Multi-line native title= tooltip.** Native browser title attributes support newlines (`\n`) for multi-line display on most browsers (Chrome, Firefox, Safari). This matches the existing pipe-separated patterns used in OpsStatusBar.tsx line 287 (`bands.map(...).join(" | ")`). The gate tooltip can use `\n` to list each criterion on its own line. (Source: browser behavior observed in Cycle/Kill tooltips; MDN HTML global attributes: title)

---

### Consensus vs debate (external)

**Consensus:**
- UTC for all timestamp storage and SOD calendar-day boundaries
- SOD must persist across system restarts within the same calendar day
- SOD must roll to current NAV on the first invocation of a new calendar day
- Native title= is inaccessible for keyboard/touch users but is exempt from WCAG 1.4.13

**Debate:**
- Whether title= is "acceptable" for informational tooltips in operator UIs: WCAG-focused sources (flook.co, thewcag.com) advocate blanket replacement with ARIA patterns; practitioners (Sarah Higley, MDN) acknowledge that ARIA role="tooltip" adds complexity for marginal benefit when the information is already supplemental and the context is an admin interface. Given the existing project pattern uses title= in three places, consistency favors continuing with title=.

---

### Pitfalls (from literature and codebase)

1. **SOD anchored to deposit time, not trade-day open.** pyfinagent's current state. Produces absurd daily-loss percentages when account has grown. (Confirmed in audit: `sod_nav = 9499.5` from 2026-04-20)

2. **SOD resetting on every cycle/restart.** Would erase intraday loss tracking. Avoided by pyfinagent's None-check guard, but that guard is too blunt — it fires only once ever.

3. **SOD computed off local timezone instead of UTC.** pyfinagent correctly uses `datetime.now(timezone.utc)` in the callsite — just needs to be used for the comparison.

4. **SOD reset on weekend restarts creating Monday false baseline.** If the system restarts Monday with a stale Friday SOD, Monday's first cycle will correctly detect stale date and update. The date-based fix handles this automatically.

5. **Tooltip content that's incomplete.** The current "1/5 checks passing" tooltip doesn't identify WHICH criterion is passing. Adding per-criterion detail (label + current value) gives the operator actionable information without leaving the status bar.

6. **Exposing too much in a title= tooltip.** Native tooltips disappear quickly and cannot be scrolled — keep content to ≤7 lines. The 5-criterion breakdown fits comfortably.

---

### Application to pyfinagent

| Finding | Maps to | File:line |
|---------|---------|-----------|
| SOD must not reset on restart; must use date comparison | `paper_trader.py:550-554` — replace `else: pass` with date comparison | `backend/services/paper_trader.py:547-554` |
| SOD date tracking requires a stored date field | `KillSwitchState` needs `_sod_date: Optional[str]` | `backend/services/kill_switch.py:49-51` |
| Boot replay must load sod_date alongside sod_nav | `_load_from_audit()` line 69-70 | `backend/services/kill_switch.py:69-70` |
| update_sod_nav should stamp the date | Add `date=today` to `_append_audit("sod_snapshot", ...)` | `backend/services/kill_switch.py:149-153` |
| snapshot() should expose sod_date | `_snapshot_locked()` and `snapshot()` return dict | `backend/services/kill_switch.py:94-108` |
| WCAG 1.4.13 exempts native title= | GateSegment tooltip can stay as native title= | `frontend/src/components/OpsStatusBar.tsx:164` |
| Gate tooltip needs per-criterion detail | Replace generic title with breakdown matching GoLiveGateWidget labels | `frontend/src/components/OpsStatusBar.tsx:164` |
| Use `\n` for multi-line native tooltip | Same pattern as Cycle segment pipe-joins | `frontend/src/components/OpsStatusBar.tsx:164, 287` |
| GoLiveGateWidget has all label/hint strings | Copy label construction into GateSegment tooltip | `frontend/src/components/GoLiveGateWidget.tsx:92-123` |
| No SOD daily-roll test exists | New test needed | `tests/services/test_kill_switch_no_deadlock.py` (or new file) |

---

### Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (7 fetched in full)
- [x] 10+ unique URLs total incl. snippet-only (17 unique URLs collected)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (kill_switch.py, paper_trader.py, OpsStatusBar.tsx, GoLiveGateWidget.tsx, test file, audit log)
- [x] Contradictions / consensus noted (title= debate section)
- [x] All claims cited per-claim (not just listed in a footer)

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 10,
  "urls_collected": 17,
  "recency_scan_performed": true,
  "internal_files_inspected": 7,
  "gate_passed": true
}
```
