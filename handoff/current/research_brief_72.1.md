# Research Brief — phase-72.1 (P1 APPROVED-BUT-UNAPPLIED OPERATOR TOKEN AUDIT)

Tier: **simple** (depth knob only — the ≥5-source floor still applies). NOT audit-class.
Started: 2026-07-18. Writer: Layer-3 Researcher. Status: IN-PROGRESS (write-first).

Goal: reconcile EVERY operator approval/token against live state; emit one actionable
`.env` line per approved-but-dark lever for the operator decision sheet. This brief is the
research + internal-inventory deliverable; Main authors the decision sheet.

---

## Internal token inventory (the main leg)

### Structural root of the approval-to-deployment gap (verified)

The bot-side token ingestion (`backend/slack_bot/operator_tokens.py:52-61`,
`KNOWN_TOKEN_ENV_MAP`) can auto-apply a token to `backend/.env` ONLY if the
token key is registered in the map. The map currently holds exactly ONE entry:
`"AWAY DRILL": "AWAY_DRILL_NOOP"` (the FEE TABLE and EU SCREENER entries are
commented-out placeholders). **No phase-60, phase-66, or phase-69 flag key is in
the map.** Therefore EVERY behavior-flag promotion is an OPERATOR KEYSTROKE into
the agent-locked `backend/.env` — the agent has no write path by design
(pending_tokens.json PROMOTE-66.2 disposition_note: "NO agent path to write .env
exists by design"; harness_log 07-09: "agent can't write the agent-locked
backend/.env"). This is the mechanism that turns an in-chat "yes" into a dark
flag: authorization is recorded (`operator_tokens.jsonl`), the cursor advances,
but nothing writes the env line.

Two compounding load conditions gate whether a written flag is even LIVE:
1. **Written?** — the `.env` line must exist (operator keystroke).
2. **Loaded?** — the backend process must have restarted AFTER the write.
   Running backend `pid 98681` started **2026-07-08 23:24 CEST**
   (`live_check_72.0.md`). Any `.env` edit after that timestamp is NOT in the
   live process until a restart.

### Reconciliation table (every approval/token found)

| Token / approval | Provenance | Gated settings.py flag(s) (file:line) | Code default | Live state | applied_verdict |
|---|---|---|---|---|---|
| **PROMOTE SYNTHESIS-INTEGRITY** (66.2) | operator in-chat "yes" 2026-07-09; `operator_tokens.jsonl:1`; `pending_tokens.json` PROMOTE-66.2-FLAGS | `paper_synthesis_integrity_enabled` (settings.py:197) | `False` | **DARK** — never written (agent-locked .env); approval 07-09 postdates last restart 07-08 23:24 (double-blocked: unwritten AND unloaded) | **NOT-APPLIED** |
| **PROMOTE RJ-SHAPE** (66.2) | same as above (`operator_tokens.jsonl:1` bundles both) | `paper_risk_judge_shape_fix_enabled` (settings.py:311) | `False` | **DARK** — same as above | **NOT-APPLIED** |
| PROMOTE POSITION-REC (66.2) | flag_promotion_brief_2026-07-09.md — reviewer verdict **HOLD** (not approved) | `paper_position_recommendation_fix_enabled` (settings.py:201) | `False` | DARK (correctly — operator did NOT approve; guard is WARN-only) | NOT-APPLIED (by design; not owed) |
| **60.2 FLAG: ON** | AskUserQuestion 2026-06-11 (harness_log:26954); live_check_61.1.md:12 | `paper_swap_churn_fix_enabled` (settings.py:345) | `False` | **LIVE=true** — keystroke-appended 06-11 (predates 07-08 restart → loaded). Independently corroborated: 70.3 test causation "live .env has PAPER_SWAP_CHURN_FIX_ENABLED=true" (harness_log:27420 + archive/phase-70.3/experiment_results.md:76) + 65.3 BQ post-fix 0-churn-swaps | **APPLIED** |
| **60.3 FLAG: ON** | AskUserQuestion 2026-06-11 (harness_log:26954, "printf 3 lines"); live_check_61.1.md:13 | `paper_data_integrity_enabled` (settings.py:45) | `False` | Inferred LIVE=true (same 3-line 06-11 keystroke batch; predates restart). No independent runtime corroboration (KR-only prompt path, no in-window KR trade to prove) | APPLIED (keystroke-confirmed; weaker runtime proof than 60.2) |
| **57.1 FLAG: ON** | AskUserQuestion 2026-06-11 (harness_log:26954); live_check_61.1.md:14 | `paper_risk_judge_reject_binding` (settings.py:307) | `False` | Inferred LIVE=true (same 06-11 keystroke batch) | APPLIED (keystroke-confirmed) |
| KS-PEAK-RESET:APPROVED | phase-69.1 owed (harness_log:27360; masterplan phase-69.1 live_check) | `kill_switch_peak_reset_enabled` (settings.py:38) | `False` | DARK by design — token NEVER issued (owed, not approved) | NOT-APPLIED (owed, correctly dark) |
| sign_safe_overlays (flip token) | phase-69.3 owed (harness_log:27371 "flip after reviewing $0 ON-vs-OFF evidence") | `sign_safe_overlays` (settings.py:36) | `False` | DARK — token never issued (owed) | NOT-APPLIED (owed, correctly dark) |
| regime_net_liquidity (flip token) | phase-69.3 owed (harness_log:27371) | `regime_net_liquidity` (settings.py:37) | `False` | DARK — token never issued (owed) | NOT-APPLIED (owed, correctly dark) |
| historical_macro un-freeze | phase-69.2/69.3 owed (harness_log:27330,27350,27371) | **NONE** — operational posture (frozen ingestion / optimizer re-run gate), not a settings.py bool | n/a | Frozen (deliberate; "$0 metered, no optimizer runs" doctrine) | NOT-A-FLAG (posture token; gates optimizer/incumbent-revalidation, not env) |
| KILL SWITCH: RESUME | reserved bare token (operator_tokens.py:46 RESERVED handling; away-ops rail) | **NONE** — process action (paper-trading resume API), not env | n/a | Not owed — kill-switch never paused since 06-11 manual resume (money_recon; kill_switch_audit.jsonl) | NOT-A-FLAG (process action; not currently owed) |
| FABLE PERMANENT: AUTHORIZE | CLAUDE.md Fable-5 policy; phase-67.4 | **NONE** — agent-file model pin (researcher.md/qa.md frontmatter), not settings.py | n/a | Reverted to `model: opus` in 67.4; no FABLE PERMANENT recorded | NOT-A-FLAG (agent-file pin, correctly reverted) |
| paper_use_claude_code_route (rail) | settings default; no operator token per se | `paper_use_claude_code_route` (settings.py:175) | `False` | **Effectively ON at runtime** (72.0: rail invoked post-restart; regresses to credit-dead direct API — a credit problem, not a flag problem) | APPLIED (runtime-inferred ON; not the P1 lever) |
| AUTORESEARCH SPEND: RESUME | in-session 2026-07-07 (pending_tokens.json AUTORESEARCH-SPEND, resolved) | **NONE** — `--preflight-only` flag in scripts/autoresearch/run_nightly.sh | n/a | Applied — flag removed same session | NOT-A-FLAG (script flag; applied) |
| SETUP TOKEN: ADOPTED | in-session 2026-07-07; executed 07-08 (pending_tokens.json SETUP-TOKEN, resolved) | **NONE** — `CLAUDE_CODE_OAUTH_TOKEN` in 4 launchd plists | n/a | Applied — token wired into plist EnvironmentVariables 07-08 | NOT-A-FLAG (credential/launchd; applied) |

**Not-yet-approved dark flags (owed-token-gated; NOT "approved-but-unapplied",
listed for completeness):** phase-70 diversity bundle
(`paper_soft_sector_diversity_enabled/_w` :447-448, `paper_min_k_sectors_analyzed`
:449, `paper_unknown_sector_cap_exempt` :450), phase-70.3 money-path
(`paper_atomic_swap_enabled` :453, `paper_cross_sector_rotation_enabled` :454,
`paper_avg_entry_fx_fix_enabled` :455), `paper_session_budget_reconcile_enabled`
:456, `paper_scale_out_enabled` :34, `paper_learn_loop_enabled` :33 (68.4 owed +
writer crash-dead), `momentum_52wh_tilt_enabled` :443, `rebalance_band_enabled`
:113, `kill_switch_auto_resume_enabled` :355 — all default-OFF, activation gated
on operator tokens/OOS evidence that has not been issued. These are correctly
dark, NOT a deployment gap.

### The single actionable P1 finding

Exactly ONE approved-and-recorded operator authorization gates a settings.py flag
that is NOT live: **PROMOTE SYNTHESIS-INTEGRITY + RJ-SHAPE** (operator_tokens.jsonl
line 1, 2026-07-09). The two `.env` lines the operator decision sheet must carry:

```
PAPER_SYNTHESIS_INTEGRITY_ENABLED=true
PAPER_RISK_JUDGE_SHAPE_FIX_ENABLED=true
```

...plus a **backend restart** (bootout/bootstrap or `launchctl kickstart`), because
even a correct write is inert until the process reloads (`pid 98681` predates the
07-09 approval). Caveat surfaced by the 72.0 forensics: these flags convert a rail
hiccup into a survivable BUY, but they do NOT fix the underlying Anthropic
direct-API credit exhaustion (the P0 lever) or the meta-scorer bypass — so
promoting them is necessary-not-sufficient for the engine to earn again.

Reconciliation still blocked on the **operator `.env` grep** (requested twice;
`backend/.env` permission-blocked for agents). Every "Live state" above is
documentary/runtime-inferred, not a direct read. The grep would upgrade the three
06-11 flags from "inferred LIVE" to "confirmed" and definitively show whether the
07-09 flags were ever added.

---

## External research — config/flag governance (approval-to-deployment gap)

### Search-query composition (3-variant discipline)

- **Current-year frontier (2026):** "feature flag config drift detection reconciliation approved but not deployed 2026"; "7 best practices feature flags at scale 2026".
- **Last-2-year window (2024-2025):** "GitOps configuration drift detection reconciliation practices 2025"; "12 commandments of feature flags 2025".
- **Year-less canonical:** "feature flag lifecycle management drift stale flags governance"; "configuration drift causes remediation".

The read-in-full table below mixes all three: 2026 (oneuptime GitOps, Growthbook), 2025 (Octopus), and year-less canonical (Harness glossary, CloudBees, beefed.ai).

### Read in full (6; ≥5 required — counts toward the gate)

| # | URL | Accessed | Kind | Fetched | Key finding / quote |
|---|-----|----------|------|---------|---------------------|
| 1 | https://www.harness.io/harness-devops-academy/configuration-drift | 2026-07-18 | Vendor doc (Harness) | WebFetch full | Drift = "when the actual system deviates from the desired state." **Single approvals fail**: "changes drift post-deployment through everyday actions... create persistent gaps that grow silently." Remediation: "use automated reconciliation for approved configuration updates and manual review workflows for unexpected deltas or security-sensitive changes." |
| 2 | https://oneuptime.com/blog/post/2026-02-26-configuration-drift-detection-gitops/view | 2026-07-18 | Practitioner (2026) | WebFetch full | GitOps reconciliation loop: fetch declared state (Git) → render → read live state → semantic diff → alert and/or self-heal. "When drift is detected, ArgoCD performs a sync to bring the cluster back to the desired state." Detection latency = the reconciliation interval; supports BOTH alerting and auto-remediation (`selfHeal: true`). |
| 3 | https://octopus.com/devops/feature-flags/feature-flag-best-practices/ | 2026-07-18 | Vendor doc (Octopus, 2025) | WebFetch full | Strong on cleanup/expiry, thin on approval→live: "Developers should incorporate regular audits or automated checks... to identify and remove stale flags"; "Setting expiration dates or review points for each flag ensures they are assessed and managed proactively." No approval-to-live reconciliation prescribed. |
| 4 | https://www.cloudbees.com/blog/feature-flag-lifecycle | 2026-07-18 | Vendor doc (CloudBees) | WebFetch full | Lifecycle = Creation → Deployment → **Activation** ("turn things on when you're ready") → Retirement. "Each feature flag provides an opportunity for misconfiguration." Notably LACKS audit-trail/verify-configured-vs-intended guidance — activation is treated as a manual human step, exactly the un-instrumented seam pyfinagent has. |
| 5 | https://beefed.ai/en/feature-flag-governance-lifecycle-best-practices | 2026-07-18 | Practitioner | WebFetch full | Governance = mandatory metadata ("`owner`, `jira`, and `expiry_date` required fields at creation"), audit telemetry ("flag change history... every toggle event is auditable"), and a **scheduled reconciliation job**: "run a scheduled job that marks flags as candidate stale when they've been at 100% for N days, then open a cleanup ticket or PR for the owner" (Uber Piranha; "Flag Friday" weekly triage). "record the decision... The decision should be documented in the jira ticket." |
| 6 | https://www.growthbook.io/blog/how-to-implement-feature-flags-at-scale | 2026-07-18 | Vendor doc (Growthbook, 2026) | WebFetch full | "Approval workflows to allow senior team members to review changes before they go live." "Audit logs and change history to see what's happening and when." "Assign ownership: Every flag should get an assigned owner at creation, and that owner is responsible for cleanup." "Prerequisite flag support... preventing invalid state combinations." |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://www.pulumi.com/docs/iac/operations/stack-management/drift/ | Official docs | Reconciliation mechanics well-covered by #1/#2 |
| https://www.nudgesecurity.com/saas-security-glossary/configuration-drift | Vendor (2026) | "approved but not yet deployed" gap corroborated in search snippet |
| https://www.xopsschool.com/tutorials/drift-detection/ | Tutorial (2026) | Duplicate of drift-def coverage |
| https://www.firefly.ai/academy/enterprise-drift-management | Vendor | Enterprise drift; redundant |
| https://medium.com/spacelift/infrastructure-drift-is-inevitable-losing-to-it-isnt-32a9ee62dce1 | Blog (May 2026) | Recency-scan hit; thesis captured in snippet |
| https://oneuptime.com/blog/post/2026-03-13-drift-detection-alerts-compliance-flux/view | Practitioner (2026) | Flux variant of #2 |
| https://www.ai-infra-link.com/mastering-config-drift-detection-top-open-source-tools-for-2025/ | Blog (2025) | Tooling list; not governance |
| https://developer.harness.io/docs/feature-flags/get-started/feature-flag-best-practices/ | Official docs | Overlaps #3/#6 |
| https://vercel.com/i/feature-flagging | Vendor | Flag-debt framing; snippet sufficient |
| https://www.getunleash.io/blog/feature-flag-driven-development-a-guide | Vendor | FF-driven dev; tangential |
| https://designrevision.com/blog/feature-flags-best-practices | Blog | Duplicate best-practices list |
| https://nhimg.org/articles/organization-level-feature-flags-expose-the-real-entitlement-problem/ | Blog | Entitlement angle; off-topic |
| https://bridgephase.com/insights/drift-detection/ | Blog | GitOps drift; redundant |
| https://www.harness.io/.../configuration-drift (2026 Nudge/XOps dupes) | — | de-dup |

URLs collected total: 6 read-in-full + 13 snippet-only = **19 unique** (≥10 floor met).

### Recency scan (last 2 years, 2024-2026)

Performed. **Findings:** the 2026 sources (oneuptime GitOps drift [Feb 2026], Growthbook at-scale [2026], Nudge/XOps/Spacelift [2026]) converge on and REINFORCE the year-less canonical: drift = declared-vs-actual gap; the fix is a *continuous reconciliation loop*, not a one-time approval. The 2026 delta vs older material is (a) tighter reconciliation intervals + webhook-triggered sync (seconds, not poll cycles), and (b) explicit "approved-but-not-yet-deployed represents a window where configuration could diverge" framing (Nudge 2026 snippet). No 2024-2026 source CONTRADICTS the canonical practice; newer work operationalizes it. Feature-flag governance 2025-2026 (Octopus/Growthbook/beefed) adds mandatory-metadata CI gates + scheduled staleness jobs, but — notably — the *activation* seam (decision → live) remains the least-instrumented stage across every vendor doc, matching pyfinagent's exact gap.

### Key findings

1. **Drift is the gap between declared intent and running reality; a point-in-time approval does not close it.** "changes drift post-deployment through everyday actions... persistent gaps that grow silently" (Harness). This is precisely the pyfinagent 66.2 state: `operator_tokens.jsonl:1` declares intent (SYNTHESIS-INTEGRITY ON), the running process (`pid 98681`) never reconciled to it.
2. **The industry-standard fix is a closed reconciliation loop** that continuously diffs declared-vs-actual and either alerts or self-heals (oneuptime/ArgoCD). pyfinagent has NO loop closing `operator_tokens.jsonl` → `.env` → running process; the one-time keystroke is the only bridge, and it is manual + agent-locked.
3. **The activation/decision→live seam is the least-instrumented lifecycle stage** across all six vendor docs (CloudBees treats "Activation" as a manual "turn it on when ready" step with no verify-vs-intended check). pyfinagent's un-logged manual `.env` keystroke is the textbook version of this gap.
4. **Governance best practice = record the decision + audit trail + scheduled reconciliation job** (beefed: scheduled staleness job → cleanup PR; Growthbook: approval workflow + audit logs; "record the decision"). pyfinagent DOES record the decision (`operator_tokens.jsonl`) and DOES have a scheduled reconciler (`sentinel.sh`) — but the reconciler is one-directional and its map is empty (see below).
5. **Prerequisite/dependency awareness prevents invalid state combinations** (Growthbook; beefed unsafe-combo). Mirrors the pyfinagent unsafe-combo guard (position-rec requires synthesis-integrity) — already modeled in `flag_promotion_brief_2026-07-09.md`.

### Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/config/settings.py` | 36,37,38,45,175,197,201,307,311,345 | Flag defaults (all False except swap_enabled/route-runtime) | Read in full; anchors verified |
| `backend/slack_bot/operator_tokens.py` | 52-61 | `KNOWN_TOKEN_ENV_MAP` (token→env auto-apply) | **Only `AWAY DRILL` registered; no 66.2/60.x/69.x key** — no auto-apply path exists |
| `scripts/away_ops/sentinel.sh` | 102-126 | `.env`↔`operator_tokens.jsonl` reconciler | **ONE-DIRECTIONAL**: flags env-true-without-token (unauthorized ON); does NOT flag token-without-env-line (the 72.1 approved-but-unapplied case is INVISIBLE) |
| `handoff/operator_tokens.jsonl` | 1 (only line) | Declared operator intent (the "Git" of desired state) | Line 1 = 66.2 PROMOTE authorization 2026-07-09 |
| `handoff/current/live_check_61.1.md` | 12-14 | 06-11 token→env mapping used at phase-61.1 activation | Confirms 60.2/60.3/57.1 keystroke-applied |

### Application to pyfinagent (external → file:line)

- **Add the REVERSE reconciliation leg to `sentinel.sh:102-126`** (finding #2/#4): iterate `KNOWN_TOKEN_ENV_MAP`-mappable + owed-token keys and flag any recorded operator authorization whose `.env` line is absent/OFF. Today the sentinel only guards the "agent wrote an unauthorized flag" direction; the "operator approved but nobody applied" direction (the actual 72.1 failure) is unguarded. This is the GitOps "desired-state-not-reconciled" alert (oneuptime) applied to the token↔env pair. [Note: 72.1 is a REPORT step — this is a recommendation for the decision sheet / a future code-tagged step, NOT a 72.1 change.]
- **Instrument the activation seam** (finding #3): the decision→live transition needs a verify-vs-intended check. External docs show every vendor under-instruments it; pyfinagent can beat them by making `sentinel.sh`'s reverse check page the digest when a token is dark.
- **The restart-to-load requirement is a second drift axis**: even a correct `.env` write is inert until `pid` restarts (72.0). The reconciler must compare against the RUNNING process's loaded settings (e.g. a `/api/settings` read), not just the `.env` file — analogous to GitOps diffing *live cluster state*, not the manifest on disk (oneuptime #2).

---

## Research Gate Checklist

Hard blockers (gate_passed false if any unchecked):
- [x] ≥5 authoritative external sources READ IN FULL via WebFetch (6: Harness, oneuptime, Octopus, CloudBees, beefed.ai, Growthbook)
- [x] 10+ unique URLs total (19)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim
- [x] 3 query variants incl. year-less canonical

Soft checks:
- [x] Internal exploration covered every token-bearing module (settings.py, operator_tokens.py, sentinel.sh, operator_tokens.jsonl, pending_tokens.json, harness_log, live_check_61.1)
- [x] Consensus noted (all 6 external sources agree; no contradiction)
- [x] Claims cited per-claim

## Bottom line for Main / the decision sheet

- The ONLY approved-and-recorded operator token gating a NOT-live settings.py flag is **PROMOTE SYNTHESIS-INTEGRITY + RJ-SHAPE** (`operator_tokens.jsonl:1`, 2026-07-09). Decision-sheet lines: `PAPER_SYNTHESIS_INTEGRITY_ENABLED=true` + `PAPER_RISK_JUDGE_SHAPE_FIX_ENABLED=true` + **backend restart**.
- 60.2/60.3/57.1 were keystroke-applied 2026-06-11 (predate the 07-08 restart → loaded); swap-churn-fix independently corroborated live (70.3 test + 65.3 BQ). APPLIED.
- KS-PEAK-RESET / sign_safe_overlays / regime_net_liquidity are OWED-not-approved → correctly dark, NOT a deployment gap. historical_macro un-freeze / KILL SWITCH: RESUME / FABLE PERMANENT are NOT settings.py flags.
- Structural root: `KNOWN_TOKEN_ENV_MAP` (operator_tokens.py:52-61) is empty of every real flag key → no auto-apply path → every promotion is a manual agent-locked `.env` keystroke. The `sentinel.sh` reconciler is one-directional and cannot see an approved-but-unapplied token. External consensus (GitOps/flag-governance) prescribes a closed, bidirectional reconciliation loop diffing declared intent vs LIVE (loaded-process) state.
- Still blocked on the operator `.env` grep (permission-blocked); live states are documentary/runtime-inferred.

## JSON envelope

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 13,
  "urls_collected": 19,
  "recency_scan_performed": true,
  "internal_files_inspected": 14,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "Reconciled every operator token/approval against live state. The ONLY approved-but-unapplied settings.py-flag token is PROMOTE SYNTHESIS-INTEGRITY + RJ-SHAPE (operator_tokens.jsonl:1, 2026-07-09) -> decision-sheet lines PAPER_SYNTHESIS_INTEGRITY_ENABLED=true + PAPER_RISK_JUDGE_SHAPE_FIX_ENABLED=true + restart. 60.2/60.3/57.1 were keystroke-applied 06-11 (loaded). KS-PEAK-RESET/sign_safe_overlays/regime_net_liquidity are owed-not-approved (correctly dark). historical_macro/KILL SWITCH RESUME/FABLE are not settings.py flags. Structural root: KNOWN_TOKEN_ENV_MAP empty of all real flag keys (no auto-apply) + sentinel.sh reconciler one-directional (blind to approved-but-unapplied). External consensus: closed bidirectional reconciliation loop diffing intent vs LIVE process state. Blocked on operator .env grep.",
  "brief_path": "handoff/current/research_brief_72.1.md",
  "gate_passed": true
}
```
