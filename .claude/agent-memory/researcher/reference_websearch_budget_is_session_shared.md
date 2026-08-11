---
name: websearch-budget-is-session-shared
description: WebSearch has a hard per-session cap (200) shared with everything spawned before you; it can be fully spent at spawn time, killing the three-variant search discipline while WebFetch still works
metadata:
  type: reference
---

`WebSearch` returns, in place of results:

> Web search was not performed: this session has used its web search budget
> (200 of 200 WebSearch calls). Continue with the information already gathered
> instead of issuing more searches. If more searches are genuinely needed, ask the
> user to raise `CLAUDE_CODE_MAX_WEB_SEARCHES_PER_SESSION`.

**Why it matters:** the cap is SESSION-level, not per-agent. A researcher spawned late
into a long session can find it already exhausted on its very first search -- measured
on step 86.33, 2026-08-11, where zero searches were available for the whole run.

**How to apply:**
1. `WebFetch` is a SEPARATE tool and still works when search is exhausted. The
   `>=5 read-in-full` and `>=10 URL` floors are still reachable by fetching canonical
   URLs from domain knowledge.
2. What is NOT reachable is `.claude/rules/research-gate.md`'s mandatory three-variant
   search discipline (current-year / last-2-year / year-less). Report that as an
   unchecked hard blocker and `gate_passed: false` -- do not quietly substitute
   "I fetched some 2025 papers" for "I searched the 2025 window". Direct fetching can
   only surface work you already knew of, so it cannot establish that nothing was
   missed.
3. Probe cheaply: issue the first search EARLY. Discovering the cap at tool call 10 is
   much worse than at tool call 3, because the brief's whole discovery strategy changes.
4. Remedy to hand back to Main: raise `CLAUDE_CODE_MAX_WEB_SEARCHES_PER_SESSION`, or
   re-spawn the gate in a fresh session.

**Fetch hosts that failed in the same run** (keep substitutes ready): `cap-lore.com`
-> "unable to verify the first certificate"; `erights.org` -> "connect ECONNREFUSED
...:443". Both are canonical capability-security primaries. Working substitutes:
the Wikipedia confused-deputy article, and the SELinux Notebook via
`raw.githubusercontent.com` (GitHub blob pages are JS-heavy; raw is plain markdown).
