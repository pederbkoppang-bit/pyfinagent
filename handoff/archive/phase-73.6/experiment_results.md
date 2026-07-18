# Experiment Results — phase-73.6: D3 money runway (recommend-only)

Date: 2026-07-18. Session: Fable 5 + ultracode, effort MAX; RESEARCH + DESIGN ONLY ($0 metered).

## What was built

1. **Research gate** (`wf_9b114107-a7e`, opus/max, tier=simple, floor held): 6 sources read in full (broker fill-fidelity, model-risk governance, transition practice), 8 internal files incl. the full go-live gate; returned 3 ordered `runway_stages`. Brief: `research_brief_73.6.md`.
2. **`money_runway_73.md` finalized (6,742 chars, one page)** — three stages verbatim with prerequisites, evidence anchors, and **13 verbatim-actionable operator decision lines**: Stage 1 paper restoration (= the phase-72 ACT-NOW block + the P3 lever sequence, cross-referenced not duplicated); Stage 2 real-fill runway (the existing phase-68 chain with its two owed tokens — `EXEC-BACKEND: ALPACA_PAPER` cutover + Alpaca creds check — and the note that real fills upgrade three phase-73 designs); Stage 3 go-live (honest 5-boolean readout: **NOT eligible, clock un-started, 0 real round trips**; the 58.1 `LLM SPEND` token; `real_capital_enabled` stays False pending compliance; graduated sizing on eventual go-live aligned with 73.3's defer math).
3. **Honesty verdicts recorded**: the phase-69 'weaker-than-documented booleans' register note verified FIXED in 69.2; the '59 trades' figure disambiguated (raw fills vs ~30 round trips); SR 11-7 → SR 26-2 supersession noted as recommend-only doc-drift.
4. **No build steps appended by design** — the runway sequences the EXISTING phase-68/58.1 queue; adding parallel steps would duplicate the masterplan (criterion 2's consistency requirement).

## Verbatim verification output

```
$ bash -c 'test -f handoff/current/money_runway_73.md && grep -Eqi "real.?fill|go.?live" handoff/current/money_runway_73.md'
73.6 VERIFICATION COMMAND EXIT: 0 (PASS)
```

## File list

- `handoff/current/contract.md` (73.6; gate → contract → GENERATE; write-first skeleton disclosed, precedented)
- `handoff/current/research_brief_73.6.md`
- `handoff/current/money_runway_73.md`
- `.claude/masterplan.json` (73.6 in-progress flip only)

## Scope honesty

Recommend-only throughout: no spend, no flags, no code, nothing un-frozen, `real_capital_enabled` untouched and explicitly DEFERRED in the runway itself. The page's headline is the honest negative (go-live not eligible, clock not started) rather than a promotional framing.
