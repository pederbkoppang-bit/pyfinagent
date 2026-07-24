# live_check — Step 63.3 (Verified defect register published)

**Status: INCOMPLETE — digest permalink OPEN (owed operator token). Step is PARKED, not done.**

The live_check spec requires: "live_check_63.3.md with the register header, row count, and digest permalink".

## Register header (verbatim first line of `handoff/away_ops/defect_register.md`)
`# Defect register — consolidated (phase-63.1 route-walk + phase-63.2 BQ cross-check)`

## Row count (immutable verification command output, 2026-07-18)
```
$ cd /Users/ford/.openclaw/workspace/pyfinagent && grep -cE '^\| DEF-[0-9]+ \|' handoff/away_ops/defect_register.md && grep -c 'SCREENSHOT-AREA' handoff/away_ops/defect_register.md
2
8
```
- DEF- rows = **2** (DEF-001 P1 reporting-broken; DEF-002 P2 cosmetic/console).
- SCREENSHOT-AREA lines = **8** (all 4 operator areas covered: reports+new-pages→DEF; positions/currency+dashboard-numbers→ALL-CLEAR).

## Digest permalink — ⛔ OPEN (owed operator action)
`<PENDING — Slack chat_getPermalink not yet obtained>`

Criterion 3 ("the register summary appeared in a Slack digest") requires an outward-facing `chat_postMessage` to
Slack + a `chat_getPermalink`, via `scripts/away_ops/send_away_digest.py:80,85` (62.8 formatter is DONE). This
unattended $0/paper drain does NOT auto-post to Slack. **Owed operator action:** post the phase-63.3 digest summary
(drafted in the `## Phase-63.3 consolidation` → "Digest summary" section of the register), then paste the resulting
Slack permalink into this field. Once the permalink is present, 63.3 criterion 3 is satisfied and the step can flip to
`done`. Until then it stays **pending/parked** — criteria 1 and 2 are built and independently verifiable.
