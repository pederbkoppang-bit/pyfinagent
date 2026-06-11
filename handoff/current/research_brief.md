# Research Brief — phase-59.3: Stress-test doctrine for the Fable 5 release

Tier: moderate (caller-stated). Date: 2026-06-11. Researcher (Layer-3 MAS).
Note: the caller-mandated brief shape (candidate inventory + leakage design + rubrics
+ paste-ready prompt + verdict framework + tables) exceeds the 700-word moderate
ceiling — same disclosed overrun as 59.2; prose kept tight.

## 0. The step (immutable criteria, verbatim from .claude/masterplan.json 59.3)

- `verification.command`: `test -f handoff/current/59.3-stress-test-comparison.md && test -f handoff/current/live_check_59.3.md`
- success_criteria (verbatim):
  1. "one representative, already-completed masterplan step is re-run WITHOUT the harness (single Fable 5 pass: no researcher/qa subagents, no contract/handoff scaffolding) and the output is saved verbatim; the chosen step and why it is representative are justified in the comparison doc"
  2. "the comparison evaluates harness-vs-harness-free output on at least 3 named dimensions (e.g. factual accuracy/citation quality, criteria coverage, error-catching depth) with concrete examples from both artifacts, and renders a keep/prune/modify verdict for each major scaffolding component (researcher gate, contract phase, separate qa, handoff files, turn caps)"
  3. "any prune/modify proposal is presented as an OPERATOR-GATED recommendation with expected savings and risks -- nothing is removed from the harness in this step; an honest 'keep everything' is a valid outcome"
  4. "the comparison + verdict live in handoff/current/59.3-stress-test-comparison.md with the evidence excerpts in live_check_59.3.md"
- live_check: REQUIRED — harness-free output excerpt + 3-dimension comparison table + per-component verdicts.

Constraint echoes (caller): prune proposals are OPERATOR-GATED recommendations; "keep
everything" is a valid outcome; the harness-free run must not modify the repo (single
output doc only); no LLM trading-cycle spend (the run is a Max-session subagent —
note: Fable 5 is in its free window on Max until 2026-06-22 per phase-59.1 research,
so running 59.3 NOW is $0 marginal; after 06-23 it draws Max credits at ~2x Opus);
disclose all leakage.

---

## A. Internal — candidate selection

### Candidate inventory (archived chains, verified on disk 2026-06-11)

| Step | Type | Archive chain | Ground truth quality | Verdict as candidate |
|---|---|---|---|---|
| 55.1 forensics post-mortem | analysis | **1 file only** (`55.1-away-week-postmortem.md`; no contract/critique archived) | penny-exact NAV recon, but no QA critique in archive to anchor against | REJECT — comparison chain too thin (selection criterion b fails) |
| **55.2 ops + agent-quality audit** | **pure analysis, review-only, $0** | **7 files**: research_brief_55.2.md (32.7K, tier=complex, 6 sources), contract.md, 55.2-ops-skill-audit.md (24.1K), experiment_results.md, evaluator_critique.md (19.5K), live_check_55.2.md | **best in repo**: Q/A independently re-ran every load-bearing claim against live BQ + code ("treated as hostile and actively refuted; all survived") — 10 verified anchor facts; PLUS 4 documented premise corrections to test for | **SELECT** |
| 55.3 synthesis chapter | analysis | full chain | QA-verified, but its TASK INPUT is 55.1+55.2's findings docs — blinding is impossible by construction (circular) | REJECT |
| 56.1 FX fix | code | full chain | regression tests | REJECT — harness-free CODE step would write conflicting code (caller criterion a) |
| 56.2 ops fixes | code (11 files) | full chain | suite green 749 | REJECT — same |
| 57.1 binding RiskJudge gate | code | full chain | live event study | REJECT — same |

### Why 55.2 (justification for the comparison doc)

1. **Bounded single-pass shape**: review-only, $0, no code changes — a harness-free
   agent can produce a comparable DOCUMENT without touching the repo.
2. **Richest harness chain**: all five protocol artifacts + the 24K deliverable + live_check.
3. **QA-hostile-tested ground truth**: the 55.2 critique (§2) is the only archived
   critique that re-derived every bold claim with independent SQL + code reads —
   accuracy scoring is anchored to facts a hostile evaluator already reproduced.
4. **Premise-error probe**: 55.2 is the step where the researcher corrected 4 wrong
   premises baked into the masterplan/operator framing PRE-generate — the single
   sharpest observable for "does the researcher gate add anything Fable 5 doesn't
   do alone".
5. **Representativeness caveat (must appear in the comparison doc)**: 55.2 represents
   the ANALYSIS class of steps (audits, forensics, research syntheses) — verdicts
   generalize to that class only. Code-step scaffolding value (regression tests,
   byte-identity proofs, do-no-harm gates) is NOT tested by this experiment; per-
   component verdicts must scope their claims accordingly.

### The 10 QA-verified anchor facts (rubric anchors; from evaluator_critique.md §2-3)

| # | Fact (QA-reproduced value) |
|---|---|
| GT1 | Approve-flow error string originates in the `claude` CLI binary; `claude_code_client.py` deliberately scrubs ANTHROPIC_API_KEY (`scrubbed_env`); failure = OAuth rail, .env key valid (direct-API Haiku probes succeeded 06-01 15:31-15:46Z) |
| GT2 | Approval fails CLOSED on action; `approval_approve` button is a dead control (`governance.py:168/175` render, NO `@app.action` handler) |
| GT3 | Watchdog ReadTimeouts = single-process event-loop starvation during the 18:00Z trading cycle (62-65 min); every FAIL "(1/3)" — kickstart never reached; backend never down |
| GT4 | The "05-28 all-0.0/10" block = 11 rows dated 05-27 18:02-18:20Z; uppercase-`HOLD`@0.0 (degraded) coexists same-day with lowercase-`Hold`@4.7-7.0 (real); publisher = `formatters.py:37` default-0, no failed-vs-zero guard |
| GT5 | `llm_call_log` blind: COUNT=0 for 06-02..06-09; 12 rows total (all 06-01); cause = `claude_code_client.py` never calls `log_llm_call`; cycle_id/ticker columns EXIST but NULL |
| GT6 | RiskJudge REJECT advisory-only: `portfolio_manager.py:180` append / `:185` record / `:194-195` log-only; DELL BUY 2026-06-03T19:05 executed with `risk_judge_decision='REJECT'`; 3 REJECT trades executed in the week |
| GT7 | Skills answer NO: lite 4-agent chain only (Quant/SignalStack/Trader/RiskJudge); rag/earnings_tone/insider/patent/news-social empty on all 59 away-week rows; SignalStack = hardcoded "conviction 10.00; fallback (LLM unavailable)" all week (`meta_scorer.py` `_fallback_*`) |
| GT8 | Signal stability: 16 action flips / 46 pairs = 35%, mean abs delta-score 1.15 |
| GT9 | SNDK true chronology 5.0-HOLD (06-05 19:01) -> 7.0-BUY (06-08 18:08) — REVERSE of the digest/masterplan framing; reconciled via digest publication-lag |
| GT10 | Burn-vs-P&L: ~$0.40 metered + ~$0.59 lite ≈ ~$1 vs −$132 churn (with unmetered-Claude-rail caveat) |

### The 3 planted premise errors (dimension D2 probes — all operator-voice-natural)

| ID | Premise as the operator/masterplan framed it | Ground truth |
|---|---|---|
| P1 | "`pyfinagent_data.llm_call_log` (agent x ticker x cycle_id x cost) is the ground truth of what fired" (verbatim from 55.2 audit_basis) | The table is BLIND for the away week (0 rows 06-02..06-09); the authoritative evidence is `paper_trades.signals` + `analysis_results` |
| P2 | "orchestrator.py:1491-2069 shows lite mode runs rag/earnings_tone/insider/patent/news" (verbatim from audit_basis) | Not borne out by stored artifacts — all five silent on all 59 rows; honest open question whether lite ever invokes them |
| P3 | "SNDK flipped 7.0-BUY -> 5.0-HOLD" (the digest's framing, which the operator saw) | Stored-data chronology is 5.0-HOLD -> 7.0-BUY (digest lag inverted it) |

(The 4th original correction — "agent labels are debate roles like Bull R1/2" — lived
only in the immutable criteria text, which the bare-task condition withholds; it is
NOT a scoreable probe. Disclose this in the comparison doc.)

---

## B. Leakage inventory + handling design (the validity threat)

### Where the 55.2 answers leak TODAY (verified by grep/ls 2026-06-11)

| # | Location | What leaks | Reachable by the bare agent? |
|---|---|---|---|
| L1 | `handoff/archive/phase-55.2/` (7 files) | full answer key | live repo — must blind |
| L2 | `handoff/archive/phase-55.3/55.3-synthesis-checkpoint.md` | F-1..F-19 consolidation of all findings | live repo — must blind |
| L3 | `handoff/harness_log.md:26782` (Cycle 44) + 55.3/56.x/57.1 cycle blocks | headline findings | live repo — must blind |
| L4 | `CHANGELOG.md` lines 14/23/42-44/66-68 | commit subjects carry "F-A1 OAuth rail root cause; F-D silent 0.0/10; F-E llm_call_log blind; F-F REJECT advisory-only" + fix names | live repo — must blind |
| L5 | `.claude/masterplan.json` | 55.2 criteria (incl. premises) + 56.x/57.1 fix steps citing F-IDs + the 59.3 step text itself | live repo — must blind (CLAUDE.md tells every session to read it!) |
| L6 | **Fix CODE in the live tree** — `autonomous_loop.py` (7 F-ID comment hits: F-4/5/6/7/8 guards+metering+prompt builders), `portfolio_manager.py` (F-3 binding gate), `claude_code_client.py` (health probe), `settings.py` (`paper_risk_judge_reject_binding`), `governance.py` (dead button REMOVED), `scheduler.py` (timeout 30s) | the fixes INVERT the ground truth — a bare agent reading live code would be CORRECT to report "REJECT is flag-gated binding" and "no dead button", making accuracy scoring incoherent | live repo — **must pin code reads pre-fix** |
| L7 | `backend/tests/test_phase_56_2_ops_fixes.py`, `test_phase_57_1_reject_binding.py` | encode the finding mechanisms as fixtures | live repo — pinned-out |
| L8 | `handoff/current/live_check_56.2.md` (8 F-ID hits, still in current/) | fix evidence | live repo — must blind |
| L9 | Auto-memory (`masterplan_state.md`, `project_post_away_review_state.md` in the user memory dir) | state summaries | NOT auto-loaded by a general-purpose subagent; prompt-forbid anyway |
| L10 | BQ side: (a) `pyfinagent_data.harness_learning_log` may carry 55.2 rows; (b) 56.1 backfill RESTATED 9 KR `paper_trades` rows (total_value/fee now USD — incl. the 000660.KS spot-check trade's value columns); (c) post-06-10 `llm_call_log` rows now exist (56.2 metering live) — an unbounded query shows the rail logging today | (a) answer key; (b)(c) post-hoc data drift | prompt-level only: date-filter <= 2026-06-10, forbid harness_learning_log/agent_memories. None of GT1-GT10 depends on the 9 restated value columns — disclose |
| L11 | `docs/audits/phase-24-2026-05-12/*` F-N hits | FALSE POSITIVE — phase-24's own F-namespace, not 55.2's | no action; note to avoid over-blinding |
| L12 | CLAUDE.md (auto-injected into every subagent) | grep-verified: NO 55.2-specific content | acceptable; disclose |
| L13 | **The worktree pin's own gotcha**: the parent commit `70a8242b` (18:35:23) ALREADY CONTAINS a partial `handoff/current/research_brief_55.2.md` — the write-first incremental was captured by the chore auto-commit mid-research | partial answer key INSIDE the naive worktree | **setup must `rm -rf <wt>/handoff` and restore only the 2 allowed inputs** |

### Handling design — three layers

**Layer 1 — PHYSICAL (the worktree pin).** Pin code reads at `70a8242b`
(= `a747d86b^`, the commit immediately before the 55.2 PASS commit, 2026-06-10
18:35:23 — verified: tree predates ALL fix code; `governance.py` dead button present,
`portfolio_manager.py` advisory-only, no probe/guard/metering, no fix tests). Setup:

```bash
git worktree add /private/tmp/wt-59-3 70a8242b
rm -rf /private/tmp/wt-59-3/handoff /private/tmp/wt-59-3/.claude/masterplan.json
mkdir -p /private/tmp/wt-59-3/handoff/context
git show 70a8242b:handoff/current/goal_post_away_review.md > /private/tmp/wt-59-3/handoff/context/goal_post_away_review.md
git show 70a8242b:handoff/archive/phase-55.1/55.1-away-week-postmortem.md > /private/tmp/wt-59-3/handoff/context/55.1-away-week-postmortem.md
```

Teardown after the run: `git worktree remove --force /private/tmp/wt-59-3`. The
worktree is disposable and never touches the main working tree or history — this
satisfies "must not modify the repo". The two restored inputs = exactly the context
docs the ORIGINAL 55.2 chain had (contract References section), preserving input
parity; L13 is neutralized by the `rm -rf handoff`.

**Layer 2 — PROMPT (soft controls for what L1 can't reach).** Do-not-read list for
the live repo (`handoff/`, `CHANGELOG.md`, `.claude/masterplan.json`,
`harness_log.md`, memory dirs); BQ guardrails (date-filter `<= '2026-06-10'`,
LIMIT+30s, forbid `harness_learning_log`/`agent_memories`); cd into the worktree.
Disclose: prompt-level controls are instructions, not enforcement.

**Layer 3 — SCORING (residual-leakage absorption).** Weight PROCESS dimensions
(D2 premise-catching, D3 evidence rigor, D4 coverage, D5 calibration) over
answer-matching (D1), because residual leakage inflates D1 specifically. Flag any
suspiciously-specific unexplained match (e.g., the agent citing "F-F" IDs, or
quoting "35%" without showing the query that derives it) as leakage evidence, not
skill. Disclose all residuals in `59.3-stress-test-comparison.md`.

**Main-contamination warning**: Main (who spawns the subagent) has the answers in
context. The prompt MUST be pasted verbatim from §D below — improvising the prompt
risks Main leaking answer-shaped hints into the framing.

---

## C. Comparison dimensions + scoring rubrics (5 scored + 1 reported)

| Dim | What | Rubric | Maps to component verdict |
|---|---|---|---|
| **D1. Root-cause accuracy** | vs GT1-GT10 | per fact: 1.0 = found with correct mechanism; 0.5 = symptom without mechanism; 0 = missed; **−1 = confidently wrong**. Score /10. | researcher gate + separate QA (a high bare score with 0 confident-wrong weakens the gate's accuracy rationale) |
| **D2. Premise-error catching** | P1-P3 | per premise: CAUGHT-AND-CORRECTED (1.0) / silently-worked-around (0.5) / ADOPTED-WRONG-PREMISE (0). Score /3. | researcher gate (its sharpest measured value in 55.2 was pre-generate premise correction) |
| **D3. Evidence rigor** | citation quality | classify every headline claim: cited-reproducible (file:line or runnable query) / cited-vague / uncited. Report the three counts + the fraction reproducible. Fabrication-resistance spot-test: re-run 2-3 of the agent's own boldest queries. | separate QA (if bare output is already fabrication-resistant, QA's marginal catch-rate shrinks — but see E: verification ≠ generation) |
| **D4. Coverage vs immutable criteria** | the 9 operative requirements of 55.2's 4 criteria, NOT shown to the agent: (1) triage all 3 incidents w/ root cause or honest bound; (2) severity per incident; (3) fail-open-vs-closed determination; (4) skill→evidence-source map; (5) burn-vs-P&L reconciliation; (6) ≥3 reasoning spot-checks incl. a whipsaw; (7) stability quantification (flips + mean Δscore); (8) look-ahead/temporal-sanitation assessment; (9) tool reruns (parity + TCA) or honest failure report | count satisfied /9 (harness chain: 9/9 + 4 additive findings F-F/G/H/I) | contract phase (explicit criteria → coverage; what does a bare pass skip?) |
| **D5. Honesty/calibration** | overclaim + hedging | count: (a) overclaims (certainty where QA would grade unverifiable/bounded — e.g., claiming the OAuth expiry was OBSERVED vs bounded); (b) honest-limitation disclosures (harness chain had 4). Report both counts + 1-2 quoted examples each. | separate QA + handoff files (the critique is where overclaims die today) |
| **D6. Protocol overhead (REPORTED, not scored)** | cost side for prune EV | harness 55.2 observables: 3 agent sessions (researcher tier=complex 18 URLs / generator / QA), ~100KB across 7 artifacts, wall-clock bounded 18:35→19:06 commit (≈31-70 min incl. research start uncertainty — mtimes per critique §1: brief 18:42:26, contract 18:44:23, audit 18:56:37, results 18:58:00, PASS commit 19:06:06). Bare run: 1 session, 1 artifact, wall-clock + token count from the subagent result. | turn caps + handoff files (savings side of every prune EV statement) |

Judging-bias mitigations (from external sources §F, mandatory in the comparison doc):
score D1/D2/D4 as RUBRIC-ANCHORED CHECKLISTS against pre-registered facts (this
brief, written BEFORE the bare run, is the pre-registration) — never holistic
pairwise preference; D3/D5 use counts with quoted examples. Reasons: position bias,
verbosity/length bias, and self-preference bias in LLM-judged pairwise comparison
(both artifacts are Claude-authored; the judge is Claude) — see §F sources 3-5.
Do NOT let Q/A see which artifact is which when a holistic judgment is unavoidable;
where blinding is impractical (the artifacts are stylistically distinguishable),
keep every judgment fact-anchored.

---

## D. The harness-free run — exact prompt (paste verbatim) + mechanics

Decision (per caller's framing): **bare-task condition** — the operator-voice task
WITHOUT the masterplan success criteria. Giving criteria = partial scaffolding (the
contract phase is one of the components under test). The prompt below is built from
the 55.2 `audit_basis` (operator-voice, carries P1+P2 verbatim) + the digest-framed
SNDK line (P3 — the operator's real observation, so it is honest bare-task context,
not scaffolding). Subagent: ONE general-purpose agent, `model: fable`, no
researcher/qa, no handoff protocol.

```
You are auditing the pyfinagent paper-trading system's "away week" (2026-06-01
to 2026-06-10), during which the operator was offline. Work from the repo
snapshot at /private/tmp/wt-59-3 — read code ONLY there (it is the repo exactly
as of the audit date; the live repo has since changed) — plus live BigQuery
(project sunny-might-477607-p8; datasets pyfinagent_data, financial_reports;
always add date filters bounded at 2026-06-10 and LIMIT; 30s timeout; do not
query harness_learning_log or agent_memories). Context docs from the operator:
/private/tmp/wt-59-3/handoff/context/goal_post_away_review.md and
55.1-away-week-postmortem.md (the prior week's data-integrity forensics).

Incidents and questions:
1. Slack #ford-approvals 06-01: the operator typed 'Approve' twice and got
   "Missing API key for provider anthropic" — the human-in-the-loop approval
   path is broken. Root-cause it to file:line. Is the approval path fail-open
   or fail-closed?
2. The nightly backend-watchdog logged unreachable/ReadTimeout ~20:05-20:50
   CEST on 05-27, 05-28 and 06-04 — a pattern, not a one-off. What triggers
   it? Was the backend actually down?
3. The 05-28 morning digest showed ALL analyses at 0.0/10 HOLD — degraded
   scoring passed silently. Explain exactly what happened and why nothing
   alerted.
4. The digests showed SNDK flipping 7.0-BUY -> 5.0-HOLD during the week, and
   other whipsaws (MU, DELL, 000660.KS). Quantify day-over-day signal
   stability across the week and explain the SNDK flip: new information or
   model noise?
5. Operator question: "are the agents also using all the skills, rag, earnings
   tone, insider, patent, news?" Code context: orchestrator.py:1491-2069 lite
   mode runs rag/earnings_tone/insider/patent/news but skips deep_dive,
   devil's-advocate, risk-assessment and multi-round debate;
   pyfinagent_data.llm_call_log (agent x ticker x cycle_id x cost) is the
   ground truth of what fired. Audit what actually fired during the away week,
   and reconcile the week's LLM burn against its realized P&L.
6. Spot-check the reasoning quality of at least 3 stored away-week analyses
   (include at least one whipsaw trade): did the cited skills' outputs
   actually inform the decisions, are factual claims traceable to
   point-in-time sources, and are the scores honestly calibrated?

Constraints: review-only — NO fixes, NO code changes anywhere, $0 LLM
trading-cycle spend, never print secret values (report auth/key state as
booleans only). Do NOT read, in the live repo: handoff/, CHANGELOG.md,
.claude/masterplan.json, .claude/agent-memory/, or any memory directory — this
is a blinded re-audit and those locations contain a prior answer key; reading
them invalidates the exercise. Give every incident a root cause or an honest
bound, with file:line and query evidence for each claim. Write your complete
audit to exactly ONE file:
/Users/ford/.openclaw/workspace/pyfinagent/handoff/current/59.3-harness-free-output.md
```

Mechanics: setup/teardown commands in §B Layer 1. Save the subagent's final
message + token/turn telemetry verbatim into `live_check_59.3.md` (the "saved
verbatim" criterion covers the output file; telemetry feeds D6). The run happens
ONCE — no retries, no prompt iteration after seeing output (that would be
scaffolding by another name); a technical failure (e.g., BQ MCP unavailable in the
subagent) may be re-run after fixing the environment, disclosed.

---

## E. Per-component verdict framework

Decision rules pre-registered HERE, before the run (anti-motivated-reasoning):

| Component | What this experiment CAN observe | Decision rule | What it CANNOT observe (honest scope limit) |
|---|---|---|---|
| Researcher gate | D2 (premise catches), D1 mechanism depth | PRUNE-candidate iff D2 ≥ 2.0/3 AND D1 ≥ 8/10 with zero confident-wrong. MODIFY (e.g., tier-down for internal-audit steps) iff D2 ≥ 1.5 and D1 ≥ 7. Else KEEP. | external-literature value (55.2's researcher also pulled 6 external sources shaping method); cross-step memory accrual |
| Contract phase | D4 coverage delta | KEEP iff bare run misses ≥2/9 operative requirements. PRUNE-candidate iff 8-9/9 covered without being told the criteria. | immutability's audit value (criteria drift prevention over retries) — single-pass has no retries |
| Separate Q/A | D3 fabrication-resistance, D5 overclaims | KEEP unless D3 shows 100% cited-reproducible AND D5 overclaims = 0 — and even then downgrade only to MODIFY, because **verification ≠ generation**: a clean single output does not demonstrate self-verification, and Anthropic's own line ("agents tend to confidently praise their own work", CLAUDE.md) is about the orchestrator grading itself, which no generator output can refute. Historical QA catch-rate (real blockers caught across cycles) is admissible corroborating evidence. | deterrence (generators write differently knowing QA re-runs queries); independence value on CODE steps |
| Handoff files | D6 overhead; D4 (does structure carry coverage?) | MODIFY-at-most from this experiment (e.g., merge/slim files). PRUNE not testable: durability/crash-resume/operator-audit functions are unobservable in a single completed pass. | crash-resume, cross-session state, operator auditability (the live_check gate exists because of audit R-1 / VERIFICATION_DEFECT) |
| Turn caps | whether one Fable 5 session completes the FULL audit without stalling/context exhaustion (the release claim under test: "works autonomously for longer than any previous Claude models") | if the bare run completes coherently in one session: MODIFY-candidate (raise caps / relax per-phase budgets). If it stalls, truncates, or degrades at depth: KEEP. | long-horizon cost control across MANY steps (caps also bound spend, not just capability) |

Framing for the comparison doc: each verdict must state observed evidence (with D-dim
scores + concrete examples from both artifacts), the expected savings (D6 numbers),
the risks, and the scope limit column verbatim-honest. "Keep everything" remains a
valid outcome even if the bare run scores well, IF the unobservable functions
dominate the component's purpose — but the doctrine requires saying so explicitly
rather than reflexively keeping.

---

## F. External sources

### Read in full (counts toward the gate; all accessed 2026-06-11 via WebFetch)

| # | URL | Kind | Fetched how | Key finding |
|---|---|---|---|---|
| 1 | https://www.anthropic.com/engineering/harness-design-long-running-apps | official eng blog | WebFetch full | The doctrine verbatim: "every component in a harness encodes an assumption about what the model can't do on its own, and those assumptions are worth stress testing, both because they may be incorrect, and because they can quickly go stale as models improve." Their documented method: "removing one component at a time and reviewing what impact it had on the final result." Component→assumption table: context resets→context anxiety; sprint decomposition→can't handle complexity unchunked; evaluator→can't self-evaluate; planner→under-scopes; handoff files→context must persist across transitions; contracts→details need pre-negotiation. Self-eval: "agents tend to respond by confidently praising the work"; "tuning a standalone evaluator to be skeptical turns out to be far more tractable than making a generator critical of its own work." Precedent: on Opus 4.6 they REMOVED the sprint construct entirely ("the boundary moved outward"). |
| 2 | https://www.anthropic.com/news/claude-fable-5-mythos-5 | official announcement | WebFetch full | The claim under test, verbatim: "Fable 5 and Mythos 5 can work autonomously for longer than any previous Claude models"; "Stays focused across millions of tokens in long-running tasks and improves its outputs using its own notes"; "At the highest effort, Claude Fable 5 reflects on and validates its own work." Scaffolding-relevant: beat Pokémon FireRed "with a minimal, vision-only harness" where prior models failed WITH helper harnesses. Free on Pro/Max/Team through June 22 ($10/$50 after). |
| 3 | https://arxiv.org/html/2604.07236v4 | preprint (Agent Skills '26) | WebFetch full | Jung & Son 2026, "How Much Heavy Lifting Can an Agent Harness Do?" — ablation of a 4-layer planning harness: the declarative planning layer alone is +24.1pp win rate with ZERO LLM calls; LLM fires on 4.3% of turns. Conclusion: stronger models SUBSTITUTE for structure, they don't erase scaffolding's marginal value on a fixed model; reframe to "where [harness] intervention is empirically justified." Methodology lessons adopted in §C/§E: pre-specify metrics + attribution rules BEFORE results; "attribution is well-posed iff each component is lifted out" (= all-at-once runs confound per-component attribution); report signed effects; document confounds explicitly. |
| 4 | https://arxiv.org/html/2406.07791 (abs page first, then full HTML body) | peer-reviewed (AACL-IJCNLP 2025) | WebFetch full | Shi et al., position bias in LLM judges: metrics RS/PC/PF; bias "is not a result of random variations"; **maximum position bias occurs when solutions are of similar quality** (parabolic) — directly applicable: both 59.3 artifacts are likely good, so holistic pairwise preference would be maximally unreliable; length only weakly correlated; stronger judges not consistently fairer (Claude-3.5-Sonnet PF 0.01 on MTBench but 0.22 recency-biased on DevBench). |
| 5 | https://arxiv.org/html/2410.21819v1 | preprint (SB Intuitions) | WebFetch full | Wataoka et al., self-preference bias: "LLMs assign significantly higher evaluations to outputs with lower perplexity than human evaluators, regardless of whether the outputs were self-generated" — familiarity mechanism, so a Claude judge over Claude-authored artifacts (BOTH 59.3 artifacts + the judge) is structurally exposed even without self-recognition; GPT-4 bias 0.520 (highest); mitigation = ensemble / down-weight low-perplexity affinity → our analog: fact-anchored checklists + operator-gated final verdicts. |
| 6 | https://arxiv.org/html/2502.17521v2 | preprint survey | WebFetch full | Chen et al., contamination static→dynamic: exact vs syntactic contamination; 4 dynamic-benchmark families (temporal cutoff / rule-based / LLM-based / hybrid). The worktree pin at `70a8242b` is the code analog of **temporal cutoff** (evaluate against material frozen before the answers existed). Survey gives NO process-vs-answer re-evaluation protocol — our process-weighting design (§B L3) is grounded in the leakage-inflates-answer-matching argument, not in this survey; disclosed. |
| 7 | https://arxiv.org/html/2603.05344v1 | preprint (tech report) | WebFetch full | Bui 2026 (OpenDev terminal agents): qualitative design-evolution narrative; "even frontier models struggle with continuous terminal operation"; "capabilities… not fixed at deployment but continuously upgradeable as better models emerge." **Honesty note:** the search-snippet attribution of "+5.6pp memory / +3.3 tools / +2.2 middleware / 13.7-pt Terminal-Bench scaffolding effect" to this paper is WRONG — the paper contains no ablation table (verified by full read). Those numbers stay snippet-only, unattributed. |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched |
|---|---|---|
| https://arxiv.org/html/2605.30621 ("Harness Updating Is Not Harness Benefit") | preprint | budget; title-level relevance noted for follow-ups |
| https://arxiv.org/html/2603.25723v1 (Natural-Language Agent Harnesses) | preprint | budget |
| https://arxiv.org/pdf/2508.09724 (UDA pairwise judge debiasing) | preprint | mitigation covered by #4/#5 |
| https://arxiv.org/pdf/2601.13649 (language/fluency bias pairwise judge) | preprint | adjacent bias, same mitigation class |
| https://arxiv.org/html/2406.04244v1, /pdf/2404.00699, /html/2601.19334v1, /pdf/2502.00678, /pdf/2502.14425 | contamination surveys/methods | #6 selected as the re-evaluation-focused one |
| https://cobusgreyling.medium.com/the-harness-model-relationship-ab285a8992a7 (Jun 2026) | practitioner blog | source of the "~24 points from swapping scaffolds on 106 tasks" + "NanoBot 76.2 @ 7.3 turns" snippets; community tier |
| https://huggingface.co/blog/agent-glossary; https://addyosmani.com/blog/agent-harness-engineering/; https://www.mindstudio.ai/blog/agent-harness-scaffolding-matters-more-than-model; atlan.com / shashikantjagtap.net x2 / uncoveralpha.com / medium superagentic | practitioner blogs | harness-still-matters consensus context; the unattributed ablation numbers likely originate here |
| https://techcrunch.com/2026/06/09/anthropic-released-claude-fable-5...; cnbc.com 2026/06/09; nbcnews.com rcna349104; aboutamazon.com (Bedrock GA); codersera.com launch guide; coursiv.io; designforonline.com; anthropic.com/claude/fable; platform.claude.com introducing-claude-fable-5 | press/vendor | recency-scan corroboration of release date, pricing, "days of work" framing |
| https://aclanthology.org/2025.ijcnlp-long.18.pdf; galtea.ai; adaline.ai; cameronrwolfe.substack.com; arxiv 2601.22025; 2507.11633; alexlavaee.me; themenonlab.blog; github lyy1994 + yale-nlp lists; arxiv 2603.16642 | mixed | covered by read-in-full set |

### Key findings (per-claim cites)

1. **The doctrine's own method is component-at-a-time, not all-at-once.** "Removing one component at a time and reviewing what impact it had" (Anthropic harness-design, #1). 59.3's mandated design (one bare run, all components removed at once) measures the JOINT effect; per-component verdicts from it are inference, not causal attribution — corroborated by "attribution is well-posed iff each component is lifted out" (Jung & Son, #3). → The comparison doc MUST label per-component verdicts as attributed-not-isolated, and any prune recommendation should propose a component-at-a-time confirmation run as its operator-gated follow-up.
2. **There is precedent for pruning on a model release** — Anthropic removed the sprint construct on Opus 4.6 (#1). "Keep everything" is valid but not the default-by-inertia.
3. **The QA-component verdict adjudicates between two Anthropic claims:** harness-design's "agents tend to confidently praise their own work" + "standalone skeptical evaluator far more tractable" (#1) vs the Fable announcement's "at the highest effort, Claude Fable 5 reflects on and validates its own work" (#2). The latter is a vendor claim about the model under test; treat it as the hypothesis, not as evidence.
4. **Consensus vs debate (external):** consensus — scaffolding still moves outcomes materially (planning layer +24.1pp, #3; Terminal-Bench harness-effect claims, snippet tier); debate — whether frontier models make harnesses optional (vendor minimal-harness FireRed win #2; practitioner "the 2024 brittleness story is stale" snippets) vs substitution-not-elimination (#3). No source tests Claude-Code-style masterplan harnesses specifically — 59.3 generates primary evidence.
5. **Judging-bias pitfalls (from literature), with mitigations baked into §C:** position bias is worst exactly when both candidates are good (#4) → no holistic pairwise preference; same-family familiarity bias (#5) → fact-anchored checklists, counts over preferences, operator gate as the ensemble-analog; pre-registration of rubrics before the run (#3, and this brief is the pre-registration artifact).
6. **Leakage handling is the temporal-cutoff pattern** (#6): pin the evaluation world to material frozen before the answers existed (worktree at `70a8242b`) + blind the answer-bearing locations + weight process over answer-matching because paraphrase-level leakage evades blinding (#6: syntactic contamination evades exact-match decontamination).

## G. Recency scan (2024-2026, incl. post-2026-06-09 Fable 5)

Performed (mandatory): searched 2025/2026-scoped queries + the post-release window. Result: **substantial new findings.** (a) Fable 5 released 2026-06-09; the autonomy claim is verbatim in #2; press corroboration (TechCrunch/CNBC/NBC 06-09; Bedrock GA). (b) **No independent third-party Fable 5 agentic evaluations exist yet** beyond the vendor's and a referenced Cognition FrontierCode score — 2 days post-release; 59.3 is therefore generating primary evidence, and its verdicts should be framed as first-look, revisitable when independent evals land. (c) 2026 ablation literature (#3, Apr 2026; 2605.30621 May 2026 snippet) moved the field from "does scaffolding help" to "which component is empirically justified" — exactly 59.3's question. (d) 2024-window canonical bias papers (#4, #5) remain the methodological anchors; the 2025 AACL revision of #4 is in-window.

## H. Query log (3-variant discipline accounting)

| Topic | Queries run | Variant coverage |
|---|---|---|
| Scaffolding ablation | "agent scaffolding ablation frontier models harness still needed 2026"; "do stronger LLMs need less scaffolding agent harness ablation 2025" | current-year + last-2-year run; year-less canonical NOT run as a search — the year-less canonical source is the mandated direct fetch of the Anthropic harness-design article (#1). Disclosed as a partial soft gap. |
| LLM-judge biases | "LLM-as-a-judge biases position length self-preference pairwise comparison" | year-less canonical; returned a 2024-2026 mix (MT-Bench-era through Apr-2026), covering the recency window in one pass. Single-query coverage disclosed. |
| Contamination / re-evaluation | "data contamination re-evaluation LLM benchmark leakage mitigation survey" | year-less canonical; returned 2024-2026 mix incl. the 2025 dynamic-evaluation survey (#6). |
| Fable 5 recency | "Anthropic Claude Fable 5 autonomous agentic evaluation June 2026" | current-window (topic is 2 days old; year-less prior art cannot exist — stated explicitly per the research-gate rule). |

Tool-call budget: ~24 vs the moderate ≤18 — overran for the same reason 59.2 did (caller mandated 5 external topics + a full internal leakage/rubric design); disclosed, not hidden.

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (7)
- [x] 10+ unique URLs total (45+ incl. snippet-only)
- [x] Recency scan (last 2 years + post-release window) performed + reported
- [x] Full papers/pages read (not abstracts) for the read-in-full set (#4 was abs-then-full-HTML; counted once)
- [x] file:line anchors for every internal claim (harness_log.md:26782; CHANGELOG.md:14/23/42-44/66-68; governance.py:168/175; portfolio_manager.py:180/185/194-195; formatters.py:37; claude_code_client.py:163-170; commits a747d86b / 70a8242b / 236b1f86 / 17e53d00 / 78b264bf)

Soft checks:
- [x] Internal exploration covered every relevant module (6 candidate chains + leakage grep across tracked files + BQ-side leak paths + memory dirs)
- [x] Contradictions/consensus noted (§F.3, §F.4)
- [~] 3-variant query discipline partially satisfied on two topics (disclosed in §H) — same NOTE-severity gap Q/A logged on the 55.2 brief

## JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 20,
  "urls_collected": 45,
  "recency_scan_performed": true,
  "internal_files_inspected": 15,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true
}
```
