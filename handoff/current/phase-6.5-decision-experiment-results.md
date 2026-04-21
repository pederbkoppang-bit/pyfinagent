# Experiment Results — phase-6.5-decision (Path D applied)

**Step:** meta-decision, phase-6.5 path selection.
**Date:** 2026-04-19.
**Cycle:** 1.

## What was changed

`.claude/masterplan.json` only. No code, no tests.

## Exact masterplan diff

### phase-6.5 (phase-level)

- `path_decision` added: `{"selected": "D", "decided_at": ..., "contract": ..., "research_brief": ..., "summary": "Keep 4 of 9 steps..."}`

### Per-step status transitions

| Step | Before | After | Notes |
|---|---|---|---|
| 6.5.1 BigQuery intel schema | `pending` | `pending` (UNCHANGED) | Kept — infra required for 6.5.7 / 6.5.9 |
| 6.5.2 Source registry + scanner | `pending` | `pending` (UNCHANGED) | Kept — infra; scanner reusable if extractors ever added back |
| 6.5.3 Institutional extractors | `pending` | **dropped** | `superseded_by: phase-7.2`. Rationale in masterplan: paywalled/lagged/PR-filtered; scraping ToS risk; 13F captures positional signal. |
| 6.5.4 Academic extractors | `pending` | **dropped** | No successor. Rationale: McLean-Pontiff 2016 post-publication decay (50–60%). |
| 6.5.5 AI-frontier extractors | `pending` | **dropped** | No successor. Rationale: out-of-loop for alpha. Belongs in a model-upgrades or MCP-hygiene phase. |
| 6.5.6 Player-driven extractors | `pending` | **dropped** | `superseded_by: phase-7.5`. Rationale: peer-reviewed evidence of negative WSB long-term alpha (Springer 2023, ScienceDirect 2024). |
| 6.5.7 Novelty client + prompt-patch queue | `pending` | `pending` (UNCHANGED) | Kept — reusable everywhere; feeds phase-8.5 LLM proposer as soft-seed. |
| 6.5.8 Slack digest | `pending` | **dropped** | `superseded_by: phase-6.5.9`. Rationale: digest over empty 4-table schema is noise. |
| 6.5.9 End-to-end smoketest | `pending` | `pending` (UNCHANGED) | Kept — proves reduced schema/registry/novelty pipeline. |

## Verbatim verification output

### masterplan JSON structural check

```
$ python3 -c "
import json
with open('.claude/masterplan.json') as f:
    mp = json.load(f)
for phase in mp['phases']:
    if phase['id'] == 'phase-6.5':
        print('path_decision:', phase.get('path_decision', {}).get('selected'))
        for step in phase['steps']:
            print(f\"  {step['id']:15s} status={step['status']:10s} dropped_reason={'yes' if step.get('dropped_reason') else 'no':3s} superseded_by={step.get('superseded_by','-')}\")
"
path_decision: D
  phase-6.5.1     status=pending    dropped_reason=no  superseded_by=-
  phase-6.5.2     status=pending    dropped_reason=no  superseded_by=-
  phase-6.5.3     status=dropped    dropped_reason=yes superseded_by=phase-7.2
  phase-6.5.4     status=dropped    dropped_reason=yes superseded_by=-
  phase-6.5.5     status=dropped    dropped_reason=yes superseded_by=-
  phase-6.5.6     status=dropped    dropped_reason=yes superseded_by=phase-7.5
  phase-6.5.7     status=pending    dropped_reason=no  superseded_by=-
  phase-6.5.8     status=dropped    dropped_reason=yes superseded_by=phase-6.5.9
  phase-6.5.9     status=pending    dropped_reason=no  superseded_by=-
```

All dropped steps carry a `dropped_reason` string; 3 carry a `superseded_by` pointer. All kept steps retain their immutable verification criteria (authored 20:30 UTC) — not mutated.

### JSON validity

```
$ python3 -c "import json; json.load(open('.claude/masterplan.json')); print('valid')"
valid
```

## Contract criterion check

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | `decision_applied_to_masterplan` | PASS | 4 kept, 5 dropped — all 5 have `dropped_reason`; 3 have `superseded_by` pointer. |
| 2 | `drop_rationale_documented` | PASS | Each dropped step's `dropped_reason` is one sentence citing the research evidence or redirect target. |
| 3 | `q_a_independent_verdict` | PENDING | Next action is Q/A spawn. |
| 4 | `harness_log_appended_last` | PENDING | Appended only after Q/A PASS. |

## Pre-Q/A self-check

- Masterplan still parses as JSON.
- No `verification` field mutated on kept steps (authored 20:30 UTC, unchanged).
- No code or test changes (`git status --short` shows only handoff + masterplan).
- `path_decision.contract` points at the on-disk contract file; `path_decision.research_brief` points at the on-disk brief.
