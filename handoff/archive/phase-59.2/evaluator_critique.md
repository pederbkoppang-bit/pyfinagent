# Evaluator Critique — Step 59.2 (EVALUATE)

**Step:** 59.2 — MCP audit + integration (Playwright full, Figma frontend workflow)
**Date:** 2026-06-11
**Agent:** Q/A (merged qa-evaluator + harness-verifier), FIRST spawn for 59.2
**Verdict:** **PASS**
**Evaluating snapshot:** OLD — this session's Q/A runs the pre-59.2 qa.md (no §1c in my loaded prompt), which is itself live confirmation that the restart caveat the step documents is real and necessary. I enforce 59.2's immutable criteria, not the new §1c gate retroactively.

---

## 0. Harness-compliance audit (5 items — all PASS)

1. **Researcher gate** — `handoff/current/research_brief.md:1` is the 59.2 brief; envelope `gate_passed: true`, `external_sources_read_in_full: 6`, `recency_scan_performed: true`, 17 URLs. Contract references it (tier=moderate, sources named). PASS.
2. **Contract pre-commit** — `contract.md` is for 59.2; python exact-compare against `.claude/masterplan.json` step 59.2: all 4 `success_criteria` **VERBATIM** (4/4) and the verification command verbatim (`cmd_verbatim: True`). mtime ordering: research_brief 06:29 < contract 06:31 < code edits 06:31:31–06:33:50 (.mcp.json, CLAUDE.md, qa.md, researcher.md, frontend.md) < live_check 06:35 < experiment_results 06:36. Contract precedes GENERATE. PASS.
3. **Results present** — `experiment_results.md` with 5-file change table, verbatim verification output, smoke evidence, and an honest-limitations section. PASS.
4. **Log-last / status** — `grep -c 'phase=59.2' handoff/harness_log.md` = 0; masterplan 59.2 `status: 'pending'`. Log + flip correctly deferred until after this verdict. PASS.
5. **First spawn / no verdict-shopping** — no prior 59.2 critique (the overwritten file was 59.1's PASS); `handoff/archive/phase-59.1/` exists, no `phase-59.2/`. Zero prior CONDITIONALs (3rd-CONDITIONAL rule N/A). PASS.

`retry_count: 0` → `certified_fallback: false`.

---

## 1. Deterministic checks (cannot hallucinate)

### Immutable verification command (run verbatim by Q/A)
```
$ source .venv/bin/activate && python -c "import json; cfg=json.load(open('.mcp.json')); assert 'alwaysLoad' in cfg['mcpServers']['playwright'], 'playwright alwaysLoad missing'" && grep -q 'Playwright' .claude/agents/qa.md && grep -qi 'figma' .claude/rules/frontend.md && test -f handoff/current/live_check_59.2.md; echo exit=$?
exit=0
```

### Content greps (file:line evidence)
- `.mcp.json` playwright block: `args` contains `@playwright/mcp@0.0.76` (HAS_0.0.76_in_args: True), `alwaysLoad: false`, all four flags intact (`--executable-path`, `--user-data-dir`, `--allowed-hosts localhost`, `--viewport-size 1440,900`).
- `CLAUDE.md:69` — playwright line in the alwaysLoad discipline list (`alwaysLoad: false` + episodic-server rationale + mid-session no-respawn caveat + 0.0.76 pin date); `CLAUDE.md:71` — Figma connector note (session connector, NOT pinned, advisory-only, never verification-load-bearing).
- `qa.md:102` — `### 1c. Live UI capture gate (BINDING -- REQUIRED if the step makes UI claims)`; `:106` — "**CANNOT receive PASS**"; CONDITIONAL + Missing_Assumption wording present (8 CONDITIONAL occurrences file-wide); `:117` — Figma awareness ("design-advisory and NEVER" satisfies the gate); `:119` — section-inline RESTART CAVEAT ("binds Q/A spawns from the session AFTER").
- `researcher.md:97-105` — MCP awareness block: Playwright (prefer live snapshot over code inference, ToolSearch path, frontend.md workflow pointer) + Figma (advisory-only, absent headless, "never make a brief's gate depend on it"). File-level RESTART CAVEAT at `researcher.md:19-20`.
- `frontend.md:73` — `## Live-UI verification (Playwright MCP + skip-auth :3100)` with `LIGHTHOUSE_SKIP_AUTH=1 npx next dev --port 3100` at `:82`; `frontend.md:104` — `## Figma MCP workflow — phase-59.2 (design-advisory ONLY)` with the headless-absence caveat at `:107` ("NOT in .mcp.json and is ABSENT in headless/cron sessions").

### Emoji scan
`git diff --unified=0` added lines across all 5 changed files + full `live_check_59.2.md`, perl scan over emoji ranges (U+1F300–1FAFF, U+2600–27BF, U+2B00–2BFF, U+FE0F): **zero hits**.

### Frontend lint + typecheck (mandated — diff touches `.claude/agents/qa.md`)
```
npx eslint .   -> 55 problems (0 errors, 55 warnings)  ESLINT_EXIT=0
npx tsc --noEmit                                       TSC_EXIT=0
```
Warnings are pre-existing project-wide (this diff touches zero `frontend/src` files); errors-only exit semantics → gate PASS.

### Pinned-artifact launch probe (Q/A-added, closes the smoke-version residual)
```
$ npx -y @playwright/mcp@0.0.76 --version
Version 0.0.76
PROBE_EXIT=0
```
The pinned 0.0.76 artifact resolves from the registry, installs, and executes on this machine.

---

## 2. Code-review heuristics (5 dimensions — no BLOCK/WARN)

- **Security:** config + docs diff only; no secrets in `.mcp.json` args (local paths + public package id); no injection paths; `supply-chain-dep-pin-removal` does NOT fire — the pin was bumped (0.0.75 → 0.0.76) with the reason documented in live_check §A, not removed. `--allowed-hosts localhost` preserved (least-privilege intact). Clean.
- **Trading-domain:** N/A — no execution path, kill_switch, stop-loss, perf_metrics, or signal code touched.
- **Code quality:** ASCII-clean additions; no Python/TS logic changed; eslint 0 errors / tsc 0 errors.
- **Anti-rubber-stamp:** no financial logic → behavioral-test heuristics N/A. **Stale-premise handling (demanded angle): HONEST.** The masterplan `audit_basis` assumed `alwaysLoad` was missing from the playwright entry; it already existed at `.mcp.json:91`. This is surfaced as "STALE AUDIT-BASIS FINDING" in `contract.md:10`, as an "Honest stale-premise note" in `live_check_59.2.md` §A, and in the `experiment_results.md` change table — the claimed new work is correctly scoped to the CLAUDE.md discipline-list entry (verified present at `CLAUDE.md:69`), NOT the pre-existing key. No overclaim.
- **LLM-evaluator anti-patterns:** first spawn, no prior verdict to flip; file:line + verbatim-output citations throughout.

---

## 3. LLM judgment vs the 4 immutable criteria

**Criterion 1 (Playwright version check + bump + alwaysLoad + smoke evidence)** — **MET (PASS-with-note).** Version checked (0.0.75/2026-05-07 vs npm latest 0.0.76/2026-06-10), delta documented (patch-level, 2 irrelevant devtools tools, ~10 bugfixes, all four pinned flags survive, zero breaking), pin bumped; `alwaysLoad: false` explicit with rationale consistent with the now-updated CLAUDE.md discipline list; live `browser_navigate` + `browser_snapshot` evidence (login-page elements, verbatim YAML excerpt) is in live_check §B — exactly the smoke-evidence shape the criterion itself defines. **The ruling on "smoke-tested if newer":** the capture ran on the session's already-connected 0.0.75 instance because editing `.mcp.json` cannot respawn a connected stdio server — an executable-this-session impossibility, not negligence. Mitigations: (i) disclosed twice in live_check (§A row + bold caveat paragraph) and in experiment_results limitations; (ii) permanently documented in `CLAUDE.md:69` + frontend.md; (iii) the delta is researcher-verified zero-breaking from release notes read in full; (iv) Q/A's own deterministic probe executed the pinned 0.0.76 artifact (`Version 0.0.76`, exit 0). Residual — the first in-session MCP connect + browser launch on 0.0.76 happens next session — is the same restart semantics this project already accepts for agent-file edits, with the caveat documented. A CONDITIONAL would demand an action impossible without restarting the operator's session; the disclosure + probe close everything executable. NOTE for next session: confirm `/mcp` shows playwright on 0.0.76.

**Criterion 2 (binding qa rule + researcher awareness + :3100 workflow)** — **MET.** `qa.md:102-119` §1c: UI-claims steps CANNOT receive PASS without a live capture referenced in the live_check; snapshot vs screenshot admissibility; absence → CONDITIONAL + `Missing_Assumption`; 55.1 precedent cited; Figma excluded from satisfying the gate; restart caveat inline. `researcher.md:97-101` Playwright awareness. `frontend.md:73-103` documents the full :3100 `LIGHTHOUSE_SKIP_AUTH=1` workflow (start/stop, operator :3000 untouched, disclosure template, capture relocation).

**Criterion 3 (Figma workflow + awareness + capability audit)** — **MET.** `frontend.md:104+` Figma MCP workflow (design-to-code for NEW views with navy/slate token reconciliation, code-to-design review, headless-absence caveat at `:107`); `researcher.md:102-105` + `qa.md:117` one-line awareness each; capability audit vs the Next.js cockpit in live_check §D — connector reality (`mcp__claude_ai_Figma__*`, session-only), what it CAN do (code-to-design capture of the live cockpit = first use; `get_design_context` React+Tailwind fit), what it CANNOT (not token-compliant by default, seat/pricing caveats, free-during-beta → LLM-cost approval rule when usage-based pricing lands).

**Criterion 4 (verification exit 0 + restart caveats + no emojis)** — **MET.** Command exit=0 (run verbatim by Q/A, §1 above). Restart caveats: `qa.md:19` file-level + `:119` section-inline; `researcher.md:19-20` file-level RESTART CAVEAT covers the awareness-block edit (NOTE: the new researcher.md block does not repeat the caveat inline — the live_check's "inline" wording is generous for that file, but the criterion's operative requirement, that the edited agent file carries the caveat, is satisfied at file level, and experiment_results separately discloses next-session enforcement). Emoji scan: zero hits in added lines + live_check.

---

## 4. Scope honesty (experiment_results disclosure)

Limitations honestly bound the claim: (i) §1c binds from the NEXT session's Q/A spawns (this spawn confirms — §1c is absent from my loaded snapshot); (ii) 0.0.76 pinned but not the connected instance (stdio no-respawn), first connect next session; (iii) Figma is docs/workflow only — no cockpit Figma file created (deferred as a natural follow-on). No overclaim detected.

**Required in the upcoming LOG step:** per CLAUDE.md separation-of-duties, the harness_log 59.2 entry must carry the Peder-review note for the `.claude/agents/` edits (qa.md §1c, researcher.md awareness), and next session should run `scripts/qa/verify_qa_roster_live.sh` + confirm playwright connects on 0.0.76.

---

## Verdict: PASS

All 4 immutable criteria met. Deterministic: immutable verification command exit=0; 4/4 criteria verbatim in contract; content greps land at file:line for every claimed edit; emoji scan clean; eslint 0 errors / tsc 0 errors; pinned 0.0.76 artifact executes (`Version 0.0.76`, exit 0). Stale audit-basis premise honestly surfaced in contract + live_check + results rather than claimed as new work. Smoke-version caveat (capture on connected 0.0.75, pin 0.0.76) disclosed twice, permanently documented, and closed to launch-level by the Q/A probe. No code-review BLOCK/WARN. First spawn, retry 0/3, no certified fallback.

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All 4 immutable criteria met. Verification cmd verbatim exit=0; criteria 4/4 VERBATIM in contract; .mcp.json pins @playwright/mcp@0.0.76 with alwaysLoad:false + all four flags; CLAUDE.md:69/:71 discipline entry + Figma note; qa.md:102 binding 1c gate (CANNOT receive PASS, CONDITIONAL+Missing_Assumption, restart caveat :119, Figma excluded); researcher.md:97-105 awareness; frontend.md:73/:104 both sections (LIGHTHOUSE_SKIP_AUTH :82, headless caveat :107); live_check has version-delta table, live navigate+snapshot excerpt, mid-session 0.0.75-capture disclosure, Figma capability audit. Criterion-1 ruling: smoke on connected 0.0.75 + pin 0.0.76 is PASS-with-note -- stdio no-respawn is unexecutable this session, disclosed twice + permanently documented, and Q/A's own probe executed the pinned artifact (npx @playwright/mcp@0.0.76 --version -> 'Version 0.0.76', exit 0). Stale alwaysLoad audit-basis premise honestly surfaced, not claimed as new work. eslint 0 errors / tsc 0 errors (qa.md-touch mandate). Emoji scan clean. Next session: verify_qa_roster_live.sh + /mcp shows 0.0.76; harness_log entry must carry the Peder-review note for agent-file edits.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": ["harness_compliance_audit", "verification_command", "criteria_verbatim_compare", "content_greps_5_files", "emoji_scan_diff_lines", "frontend_eslint", "frontend_tsc", "npx_0076_launch_probe", "mtime_ordering", "code_review_heuristics", "research_brief", "experiment_results", "live_check", "prior_critique_59.1"]
}
```
