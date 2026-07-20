# Contract — Step 75.2: Audit75 S2 — Slack control-plane authorization + dead-plane removal

- **Step id:** 75.2 (phase-75, P0, executor: opus-4.8/xhigh)
- **Date:** 2026-07-20
- **Boundary:** control-plane only, no trading logic, $0 metered unchanged.
- **Findings remediated:** gap1-01, gap1-03, gap1-04, gap1-05, gap1-07, gap1-08, gap1-09, gap1-10, gap1-11

## Research-gate summary

Gate PASSED (moderate tier, `handoff/current/research_brief_75.2.md`): 8 sources read in full (floor 5) — 1 preprint, 2 standards bodies, 4 official vendor docs, 1 engineering blog; zero community-tier in the read-in-full set. 32 URLs, 3-variant query discipline, recency scan with 3 findings, 18 internal files inspected with line anchors.

External warrant is unanimous: Bolt's `authorize` is installation-level and performs **no** per-user authorization (so the check is hand-rolled by definition); a matcher is a capability gate, not an authorization decision, so the check must repeat **at the sink** and fail closed on error (arXiv:2606.28679); OWASP LLM01:2025 requires privileged functions be handled in code rather than exposed to the model; Cloudflare's sliding window (`rate = prev * ((period-elapsed)/period) + cur`) needs two integers per user and no external store.

Internal findings that change the plan:
- **gap1-01 is worse than the register states** — `commands.py:328-355` has FIVE defects, not one: `event['user']` never read (any channel member's `white_check_mark` triggers `git push origin main`), `item.ts` never read, `subprocess.check_output` blocks the Socket-Mode loop up to 30s, no fail-closed branch when `slack_operator_user_id` is unset, and `say()` posts unthreaded.
- **The dead plane is genuinely dead and smaller than specced** — 2,306 lines (not ~2,900), zero live importers. `governance.py` doesn't even define the names `assistant_handler.py` imports from it, so that code would `ImportError` if reached: non-functional, not merely unused.
- **No push-approval poster exists** — the reaction handler was written against an imagined message. An empty `_pending_push_ts` therefore denies every reaction, which is the correct fail-closed day-one default.

## Hypothesis

Gating the reaction sink on operator identity + a tracked bot-posted `ts`, deleting the six dead modules, and installing pre-LLM deploy refusal + rate limiting + audit on the LIVE path removes every unauthenticated path to `git push` and every hallucinable deploy, without touching trading logic or metered spend.

## Immutable success criteria (verbatim from .claude/masterplan.json step 75.2)

1. A white_check_mark reaction from any user other than settings.slack_operator_user_id, or on any message whose ts is not a tracked bot-posted push-approval request, performs NO git push (fail-closed when the setting is unset); the push subprocess runs via asyncio.to_thread
2. The six dead modules are deleted and zero imports of them remain anywhere in backend/, scripts/, or tests (grep evidence in experiment_results.md); no deploy-capable code path exists without operator identity plumbing
3. A message containing a deploy verb reaches the refusal branch BEFORE any LLM/orchestrator call and the reply contains 'deploy commands are disabled'; the assistant can no longer answer deploy requests as if it deployed
4. Live assistant path has a per-user rate limit and appends one JSONL audit record per interaction to handoff/logs/assistant_audit.jsonl (gitignored logs dir); both are exercised by at least one unit test or scripted smoke recorded in experiment_results.md
5. App Home model-change actions reject non-operator users fail-closed and append an audit line; the UI labels the change process-local
6. append_operator_token refuses (returns None + warning log) records whose user or channel fail the operator/channel allowlist even when called by a future non-matcher caller

## OPERATOR ESCALATION — immutable-criteria collision (BLOCKER-1, do NOT silently resolve)

Deleting the six modules breaks three **already-`done`** steps' immutable verification commands. CLAUDE.md forbids amending immutable criteria, so this contract **does not touch them**; it records the collision for the operator:

| Dotted path | step_id | Breaks how |
|---|---|---|
| `phases[26].steps[4].verification.command` | 4.14.4 | `from backend.slack_bot import assistant_handler` → ImportError |
| `phases[26].steps[23].verification.command` | 4.14.24 | greps `assistant_handler.py`; missing file → count 0 → `awk` exit 1 |
| `phases[29].steps[8].verification` | 4.17.9 | names `scripts/go_live_drills/self_update_audit_test.py`, **which does not exist on disk today** — already unrunnable pre-change, independent of 75.2 |

Position taken: these are *historical* verifications of code this step intentionally retires; git history preserves the modules. Recorded verbatim in `experiment_results_75.2.md` + `live_check_75.2.md` with the dotted paths. No immutable field is edited.

## Plan steps (shapes per research brief §Application)

1. **(a) gap1-01 reaction sink** — module-level `_pending_push_ts: set[str]`; handler order: fail-closed when `slack_operator_user_id` unset → `event['user'] != operator` deny (NOT `item_user`) → channel check → `item.ts` must be in `_pending_push_ts` → single-use `discard` → `await asyncio.to_thread(subprocess.check_output, ...)` → threaded reply. `asyncio` (`:7`) and `subprocess` (`:9`) already imported. Expose a registration helper for the approval-request ts and document that an empty set denies everything.
2. **(b) deletion + pre-LLM refusal** — delete the six files; insert `_DEPLOY_VERBS` refusal in `streaming_integration.py` between `:99` and `:104` (after the empty-text return, before `get_orchestrator()`, well before `_classify_via_llm` at `:108`). Literal `deploy commands are disabled` kept byte-exact and lowercase. Verb list covers what the deleted `self_update.handle_deploy_command` matched. **[CORRECTION, cycle 2: the plan statement here was right; my first implementation did NOT meet it -- it used a 7-entry substring list built from memory rather than from the deleted code, and the cycle-1 Q/A proved bare `deploy` and 11 other surfaces bypassed the refusal. The detector is now derived from `git show HEAD:...self_update.py` and parity is measured at 21/21. Plan text left intact deliberately -- the failure was in execution, not in the plan.]** **BLOCKER-3: write no tombstone comment containing a dotted `slack_bot.<dead>` path or the literal `import <dead>`** — 75.2's own gate greps every `backend/slack_bot/*.py` for those substrings. Rationale goes in experiment_results, not the source.
3. **(b') BLOCKER-2** — retire/guard `scripts/go_live_drills/smoke_test_4_17_9.py:33-34` (`assert su.exists()` on `self_update.py`) in the same commit. `scripts/qa/sweep_ascii_logger_v3.py` needs no change (it skips missing files at `:54-56`); stale entries removed as tidiness only.
4. **(c) rate limit + audit** — Cloudflare two-int sliding window (60s / 20 msgs) per user; audit writer copies the `operator_tokens.py:111-132` idiom (`asyncio.Lock`, `mkdir(parents=True, exist_ok=True)`, one `json.dumps(..., ensure_ascii=False)` line, append). **Store `text_sha256`, not raw text** — preserves the privacy rationale for the gitignored path and makes future promotion to a tracked path safe.
5. **(d) App Home** — gate ONCE inside `_handle_model_change` (`app_home.py:378-395`), which covers all four registrations; `await ack()` stays FIRST (Slack's 3s rule — deny after ack, never by withholding it); audit line on denial; label the select "process-local, resets on restart". Add the `get_settings` import.
6. **(e) sink check** — factor the four checks at `operator_tokens.py:87-95` into a shared `_authorized(...)` helper used by BOTH the matcher and `append_operator_token`; the sink returns `None` + warning on mismatch. The matcher must keep returning `False` (not raise) so Bolt falls through to ticket ingestion (contract documented at `:84-86`). **BLOCKER-5: seven existing call sites** in `backend/tests/test_phase_62_2_operator_tokens.py` (`:81,95,97,104,106,113,121`) — any added defaults must fail CLOSED (empty operator id = deny).
7. **(BLOCKER-4) tests are required evidence** — the step's deterministic command only substring-checks five tokens; it does NOT verify the `ts` binding, fail-closed-when-unset, the rate limiter, or refusal-precedes-LLM. Criteria 1/3/4/6 therefore need unit tests. Pattern to copy: `backend/tests/test_phase_62_2_operator_tokens.py:75`.
8. **Verify** — run the immutable command (exit 0) + the new tests + full `backend/slack_bot` import smoke; write `experiment_results_75.2.md` + `live_check_75.2.md`; spawn Q/A; log-last; flip.

## References

- `handoff/current/research_brief_75.2.md` — 8 sources read in full, 5 BLOCKERS, drop-in code shapes for (a)–(e), 9 pitfalls (incl. `item_user` vs `user` inversion)
- `handoff/current/audit_phase75/register.md` — gap1-01/03/04/05/07/08/09/10/11
- `.claude/masterplan.json` step 75.2 (immutable criteria + verification command)
- Latent gap noted in passing, OUT OF SCOPE: `apply_leak_defenses` (`streaming_integration.py:500-517`) is defined but never called; `_stream_simple_response` streams raw text at `:184-185`.
