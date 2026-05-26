# Contract -- Cycle 5: settings exposure + binary-path fix + live 27.6 closure

**Cycle:** 5 (production-readiness mode + testing-phase trading mandate)
**Date:** 2026-05-26
**Step targeted:** `27.6` "End-to-end smoke verify: full path on Claude" (P0). This cycle WILL flip 27.6 status to `done` once the in-flight live cycle completes with PASS evidence.
**Class:** operational closure of a P0 masterplan step. The two small code changes are infrastructure that enables the live verification; no trading-policy semantics changed (citation floor does NOT apply -- verification cycle).

**File-collision note (SEVENTH occurrence today):** `handoff/current/contract.md` clobbered by autonomous-loop sprint contract at 19:56, 20:36, 20:47, 22:47, 21:02 and twice more. Main re-wrote each time. Deconfliction stays on follow-up backlog.

## Research gate

Cycle 4 researcher `ab1987d4ec80af4dd` already gate_passed=true on the stdin pattern. Cycle 5's small additions (Pydantic field exposure + binary-path resolution) are mechanical extensions of cycle 3/4's surface; no new external research required. Researcher floor satisfied via the cycle 3 + cycle 4 briefs which both gate_passed=true.

## N* delta

- **B primary:** the autonomous-loop's next 13-ticker analysis pass produces ≥14/15 `analysis_results` rows persisted to `financial_reports.analysis_results` with `model=claude-sonnet-4-6`, zero "Full orchestrator failed" lines, lite_mode=False, and `rail=claude_code` per-ticker log lines. Step 27.6 closes PASS.
- **R secondary:** every change is gated behind a settings flag the operator can flip back. Binary-path resolution adds a `CLAUDE_CODE_BINARY` env override + `shutil.which()` + 3 well-known install locations -- defense in depth.

## What this cycle actually did

### Code changes (2 files)

1. `backend/api/settings_api.py` -- exposed `paper_use_claude_code_route` to the read + write API surface:
   - `FullSettings.paper_use_claude_code_route: bool = False` (response schema).
   - `SettingsUpdate.paper_use_claude_code_route: Optional[bool] = None` (write payload).
   - `_FIELD_TO_ENV["paper_use_claude_code_route"] = "PAPER_USE_CLAUDE_CODE_ROUTE"` (env-var persistence map).
   - `_settings_to_full` now copies the field into the response.
   Without this change, the PUT endpoint silently drops the field (cycle-3 ship only added the Pydantic field; the API allow-list is a separate hardcoded set).

2. `backend/agents/claude_code_client.py` -- added `_resolve_claude_binary()` resolver. The launchd-supervised backend doesn't inherit the operator's interactive-shell PATH so a bare `subprocess.run(["claude", ...])` failed with FileNotFoundError. Fix:
   - `_DEFAULT_SEARCH_PATHS` env-overridable list: `CLAUDE_CODE_BINARY` env var > `~/.local/bin/claude` > `/opt/homebrew/bin/claude` > `/usr/local/bin/claude`.
   - `_resolve_claude_binary(binary)` returns the first matching absolute path, falling back to the literal `binary` string so unit-test mocks of `subprocess.run` still see `"claude"`.
   - `claude_code_invoke` now passes `resolved_binary` to `args[0]`.

### Operational steps (executed live)

3. `launchctl kickstart -k gui/$(id -u)/com.pyfinagent.backend` -- reloaded backend twice (once after settings_api change, once after path-fix).
4. `PUT /api/settings/ {"paper_use_claude_code_route": true}` -- flag flipped successfully (verified via re-GET).
5. `PUT /api/settings/ {"gemini_model": "claude-sonnet-4-6"}` -- flipped from `claude-opus-4-7` to satisfy 27.6 success_criterion #1.
6. `POST /api/paper-trading/run-now` -- triggered the autonomous loop at 2026-05-26T23:39:44+0200 with the corrected configuration.

### In-flight verification

7. Backend.log shows `claude_code_invoke: success duration_ms=28373 input_tokens=6 output_tokens=941` lines -- the rail is operational. The cycle is mid-execution; ~50 LLM calls × ~30s/call = ~25-30 min total runtime expected (vs ~6 min on the credit-exhausted direct rail).

8. Cycle completion detection: poll `pyfinagent_data.strategy_decisions` for the cycle_id corresponding to the 23:39 trigger; poll `financial_reports.analysis_results` for `COUNT(*) WHERE DATE(analysis_date) = CURRENT_DATE() AND model = 'claude-sonnet-4-6'`.

## Immutable success criteria

1. `python -c "import ast; ast.parse(open('backend/api/settings_api.py').read())"` exit 0.
2. `python -c "import ast; ast.parse(open('backend/agents/claude_code_client.py').read())"` exit 0.
3. `pytest backend/tests/test_claude_code_client.py -v` -- all 12 pass.
4. `curl -s http://localhost:8000/api/settings/ | jq -r '.paper_use_claude_code_route'` returns `true`.
5. `curl -s http://localhost:8000/api/settings/ | jq -r '.gemini_model'` returns `claude-sonnet-4-6`.
6. `handoff/current/live_check_27.6.md` rewritten with verbatim PASS evidence (cycle_id of the 23:39 trigger; per-ticker `rail=claude_code` log lines; BQ COUNT >= 14 on analysis_results for today; zero "Full orchestrator failed" lines).
7. `.claude/masterplan.json` `27.6.status` flipped from `pending` to `done` AFTER live_check_27.6.md is updated AND harness_log is appended.
8. ZERO frontend changes.
9. ZERO new npm deps.
10. NO `npm run build`, NO `rm -rf .next/*`.
11. ZERO emojis introduced.

## /goal integration gates

1. pytest green. 2. AST parse green. 3. Live cycle reaches completion AND analyses_persist >= 14. 4. Log LAST. 5. masterplan flip happens AFTER all 5 handoff files are present + Q/A PASS. 6. No self-evaluation.
