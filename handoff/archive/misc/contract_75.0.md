# Contract — Step 75.0: Ultracode full-stack code-quality audit vs official docs + best practices

**Step id:** 75.0 (phase-75, installed this cycle)
**Date:** 2026-07-19
**Session:** Fable 5 + ultracode. Operator explicit override 2026-07-19: **Fable 5 on ALL agents** including Researcher + Q/A (session-scoped launches only; `.claude/agents/*.md` Opus pins untouched — no persistent repin, so no scheduled-revert step is owed; Max usage credits knowingly drawn post-free-window).
**Operator order (verbatim):** "use ultracode with fable 5 reasoning with all IT stack roles to audit our codebase againtst coding documenation and best practise. end goal should be adding new steps into our masterplan to improve our code on all levels" + correction "i told you to use ultracode with dynimic workflow where you use fable 5 on all agents". Goal draft: `handoff/current/goal_phase75_code_quality_audit_DRAFT.md`.

## Research-gate summary (gate PASSED before this contract)

Researcher ran via Workflow structured-output (run `wf_646a6e15-a94`, Fable 5 / effort max, agent-count 1, ~250K tokens): **24 official/peer-reviewed sources read in full**, ~200 URLs collected, 30 snippet-tier, 26 internal artifacts inspected, recency scan performed, audit-class coverage loop ran **10 rounds to 2 consecutive dry rounds (coverage.dry=true)**. Brief: `handoff/current/research_brief_75.0.md` (54KB) — per-role doc anchors + 3–8 checkable rules per role, adversarial methodology grounded in SWR-Bench (LLM-review precision crisis → verification gates mandatory), Refute-or-Promote (adversarial stage-gating), Agent Audit. Pre-verified seed facts: CVE-2025-66478 already patched (next@15.5.12); **no Python lockfile**; **no `maximum_bytes_billed` anywhere**; legacy `insertAll` streaming in 10+ modules vs Storage Write API recommendation; sync clients inside async routes in 10 files; inverted logging formatter in `backend/main.py`; `_PUBLIC_PATHS` drift vs security.md. Exclusion registers enumerated (phase-69: 50, phase-70: 17, phase-71: 17, phase-72/73/74 pending queues).

## Hypothesis

A role-partitioned multi-agent audit (14 IT-stack roles, read-only), where every finding requires verbatim file:line evidence plus a citable doc/best-practice basis and must survive independent adversarial verification (double-refuter at P0/P1), will surface real, non-duplicative engineering defects across all stack layers with low false-positive rate, and can be mechanically converted into an executor-tagged phase-75 remediation queue that cheaper sessions execute — without this step touching any product code.

## Immutable success criteria (copied verbatim into `.claude/masterplan.json` step 75.0)

1. research_brief_75.0.md exists with a gate_passed:true JSON envelope, >=5 external sources read in full, a recency-scan section, and audit-class coverage.dry==true, and it predates the contract and the audit run
2. The audit ran as a multi-agent Workflow with >=12 read-only role auditors plus adversarial verification: every CONFIRMED finding carries repo-relative file:line + verbatim evidence + a doc/best-practice basis and survived >=1 independent adversarial verifier; findings confirmed at P0/P1 severity survived a second independent refuter
3. handoff/current/audit_phase75/register.md and handoff/current/audit_phase75/confirmed_findings.json exist, containing workflow stats, the confirmed list, and the refuted/duplicate list with reasons
4. >=8 new remediation steps exist in phase-75 with status=pending, each with an [executor:] tag in the name, immutable testable success_criteria, and a non-interactive verification.command that exits 0 offline; an adversarial step-review ran and its verdict (approvals/revisions applied) is recorded in the register
5. This step changed NO product code and NO backend/.env: the step's git diff is confined to handoff/**, .claude/masterplan.json, and hook-managed CHANGELOG.md; kill-switch/stops/sector-caps/DSR/PBO gate files byte-untouched

## Plan

1. GENERATE — launch `phase75-fullstack-audit` Workflow (script staged in session scratchpad): 14 read-only role finders (`py-services`, `py-core`, `architecture`, `frontend-react`, `frontend-ts-contract`, `data-bq`, `security`, `sre-ops`, `qa-tests`, `llm-eng`, `perf`, `deps`, `api-design`, `docs-drift`; all `agentType: Explore`, `model: fable`, effort max; each reads the research brief for doc anchors, caps 15 findings with dropped-tail logging) → JS key-dedupe + fuzzy cluster agent → adversarial verify (pass-1 all findings, pass-2 refuter on P0/P1 survivors; duplicates vs phase-69..74 registers killed here) → completeness critic + up-to-6 targeted gap finders → synthesis into step candidates → independent adversarial step review.
2. Main writes `handoff/current/audit_phase75/{register.md, confirmed_findings.json}` + `experiment_results_75.0.md` from the captured workflow return, applies the step-review revisions editorially, and appends the approved remediation steps to phase-75 as `status: pending`.
3. EVALUATE — Q/A via session-local Fable variant of `qa-verdict.js` (structured output; verdict transcribed VERBATIM to `evaluator_critique_75.0.md`; errored/empty return = NO VERDICT, fall back to Agent-tool qa).
4. LOG — `live_check_75.0.md`, `harness_log.md` append, then status flip 75.0 → done (separate edit; auto-push hook).

## Boundaries (binding)

Audit + queue only: no product-code edits, no `backend/.env`, no flag flips, no optimizer runs (`historical_macro` FROZEN), paper-only; kill-switch/stops/sector-caps/DSR>=0.95/PBO<=0.5 files byte-untouched; no duplication of phase-72/73/74 pending queues; no re-litigation of phase-69/70/71 closed registers; Layer-3 harness internals out of scope (phase-71 just closed); harness stays exactly 3 agents (Main + Researcher + Q/A — the workflow fan-out is a launch mechanism inside GENERATE, not new harness members).

## References

- `handoff/current/research_brief_75.0.md` (24 full-read sources; envelope gate_passed:true, coverage.dry:true)
- FastAPI async/concurrency docs; OWASP API Security Top 10 (2023) + Secrets Management Cheat Sheet; BigQuery cost-control + Storage Write API docs; Python 3.14 What's New; Next.js server/client components + data-security guide + CVE-2025-66478 advisory; React 19 release notes; Anthropic Building Effective Agents / Writing Tools for Agents / Structured Outputs; SWR-Bench (arXiv 2509.01494); Refute-or-Promote (arXiv 2604.19049); Agent Audit (arXiv 2603.22853); uv pip-compile; pip-audit; pytest good practices; npm package-lock docs.
- Anthropic, "Harness design for long-running apps" (five-file protocol) — cycle mechanics.
