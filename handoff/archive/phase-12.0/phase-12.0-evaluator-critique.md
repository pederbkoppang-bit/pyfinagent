---
phase: "12.0"
qa_id: "qa_120_v1"
date: "2026-04-19"
verdict: "CONDITIONAL"
cycle: 1
---

# Q/A Critique — phase-12.0

## Protocol audit (5/5)

1. Research brief present at `handoff/current/phase-12.0-research-brief.md`, `gate_passed: true`, 7 read-in-full + 10 snippet-only + recency scan — exceeds floor.
2. Contract pre-committed; "Unusual flow note" discloses researcher co-authored deliverable — acceptable given disclosure.
3. `phase-12.0-experiment-results.md` present.
4. `harness_log.md` last entry is phase-11.4 N+53 (PHASE-11 COMPLETE) — NOT 12.0. Correct (log is last).
5. Cycle-1, single Q/A. No verdict-shopping.

## Deterministic

- **A.** Immutable: `ok 7135` (>2000). PASS.
- **B.** 4 required sections: `Current Deploy Surface`=1, `Scope`=3, `palette`=3, `Rollback`=9, `SLO`=5. PASS.
- **C.** Scope: only `docs/RAINBOW_DEPLOY_PLAN.md` + phase-12.0 handoff files + `.claude/` churn pre-existing. PASS.
- **D.** Regression not re-run (phase-11.4 PASS 79p/1s; no code touched this cycle). Accepted.
- **E.** Content spot-check: components named match real pyfinagent (backend, frontend, harness, slack bot, MCP); IN (backend+slack bot) / OUT (frontend+harness+MCP) lists explicit; palette = 2 with expand path to 7; SLO = "under 30s; no pod restart". PASS.
- **F.** `dimcheff` appears 8x (URL + references + rationale). PASS.
- **G.** No new K8s YAML created. PASS.

## LLM content review

- `backend/Dockerfile` exists (19 lines): CORRECT.
- CI workflow count=5, `grep kubectl|kubernetes|deploy` over `.github/workflows/` = 0 hits: CORRECT.
- No K8s YAML in tree: CORRECT.
- **FACTUAL ERROR**: `frontend/Dockerfile` EXISTS (27 lines, `FROM node:20-alpine AS deps`). Plan says twice "NO Dockerfile" (table row 14 + scope-OUT row 36). Scope conclusion (frontend OUT because `npm run dev` dev pattern, not prod server) is still defensible, but the stated reason is factually wrong and phase-12.1 will inherit the mistake.
- Palette 2→7 justification: sound.
- SLO 30s vs Dimcheff 2-10s: realistic + generous.
- **Phase-12.4 scope-shift**: masterplan `12.4` has empty description + status=pending. Main's caveat #5 (12.4 needs new candidate now that phase-11 shipped sans Rainbow) is NOT captured in plan or masterplan. Non-blocking for 12.0 (audit-only) but should be recorded before 12.1.

## Violated criteria

| type | action | state | constraint |
|------|--------|-------|-----------|
| Contradiction | Plan claims frontend has no Dockerfile | `frontend/Dockerfile` exists (27 lines, node:20-alpine) | Content must match filesystem reality |

## checks_run

["protocol_audit", "immutable_verify", "section_grep", "scope_diff", "content_accuracy_fs", "ci_workflows_grep", "k8s_yaml_check", "masterplan_12.4_lookup"]

## Verdict: CONDITIONAL

Deliverable is 95% sound. One factual error (frontend Dockerfile exists). Fix: update the table row + scope-OUT rationale to "frontend has a Dockerfile but runs `npm run dev` in development; productionizing the Next.js container is a separate phase" or similar. Also record the phase-12.4 scope-shift note. Then re-spawn fresh Q/A on updated evidence.

certified_fallback: false

---

## Follow-up (2026-04-19 ~16:00 UTC, pre-respawn)

qa_120_v1 CONDITIONAL. Fixes applied:

1. **Factual error fixed in `docs/RAINBOW_DEPLOY_PLAN.md`**:
   - Row 14 (deploy surface table): `Frontend | NO Dockerfile` → `YES — frontend/Dockerfile exists (Node 20 alpine), but not yet wired into any prod runtime`.
   - Scope-OUT row 36: rewrote the reason to "Dockerfile exists but Next.js is served via `npm run dev` (dev server, not `next start`); productionizing is a prerequisite for Rainbow that's out of scope here; revisit when the backend Rainbow pattern has baked in."
2. **Phase-12.4 scope reassignment documented**:
   - New subsection under "Phase-12 Implementation Steps" in the plan doc explains that phase-11 shipped without Rainbow, so 12.4's "Vertex cutover" is no longer the candidate. Lists replacement candidates (next SDK bump OR dummy color-flip rehearsal).
   - `.claude/masterplan.json` phase-12 step 12.4 updated: `name` now reads "First Rainbow migration — candidate TBD after 12.3"; added `scope_reassigned_at` + `scope_reassignment_note` metadata fields. `verification` + `success_criteria` preserved untouched (anti-tamper).

3. Updated Known caveats in experiment_results in a way that isn't needed — the cycle-1 experiment_results already flagged the frontend/Dockerfile and phase-12.4 shift; the problem was the PLAN DOC said the opposite.

Cycle-2 evidence has changed: plan doc rows 14 + 36 now match reality, masterplan 12.4 scope-reassignment recorded. Respawning fresh Q/A for verification.
