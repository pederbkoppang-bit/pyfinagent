# Claude Code Surfaces Compliance Audit (phase-4.15.13)

Date: 2026-04-18. Scope: CI workflows, Slack bot, cron budget, routines,
scheduling surfaces, devcontainer, LLM gateway, ultraplan/ultrareview.
AUDIT ONLY — no code changed. Supersedes phase-4.11 for the surfaces
covered here; all prior findings remain valid unless contradicted below.

---

## What changed since phase-4.11

The docs read for this audit confirm two new features now in research preview
that were not described in phase-4.11:

- **Routines** — Anthropic-managed cloud sessions with schedule, API, and
  GitHub-event triggers. Min interval: 1 hour. Run autonomously (no approval
  prompts). Accessible at `claude.ai/code/routines` or via `/schedule` CLI.
- **Ultraplan / Ultrareview** — now confirmed at v2.1.91+ (ultraplan) and
  with $15-25/run pricing for the managed Code Review service. Separate
  billing from plan usage.

Everything else (github-actions v1.0, code-review plugin, slack integration,
devcontainer, llm-gateway, /loop, checkpointing) is unchanged from phase-4.11.

---

## Pattern table (20 patterns)

### P-01 — claude.yml: no model pin (MF-17, MUST FIX)

**Evidence.** `claude.yml` uses `anthropics/claude-code-action@v1` with no
`claude_args`. The doc states: "Claude Code GitHub Actions default to Sonnet.
To use Opus 4.7, configure the model parameter to use `claude-opus-4-7`."

**Finding.** Every `@claude` invocation in issues and PRs runs on Sonnet 4.x
by default, not Opus 4.7 which the harness uses. The model used is
unpinned and will silently drift as defaults change.

**Fix.** Add `claude_args: '--model claude-opus-4-7'` under the `with:` block
of the `Run Claude Code` step.

---

### P-02 — claude.yml: no --allowedTools restriction (MF-17, MUST FIX)

**Evidence.** The commented line in `claude.yml` shows the default:
`# claude_args: '--allowed-tools Bash(gh pr *)'`. It is commented out.
Current state: all tools allowed by default (Bash unrestricted, Write,
Edit, etc.).

**Finding.** A malicious `@claude` invocation in an issue or PR comment can
run arbitrary bash on the GitHub runner — write secrets, exfiltrate env,
etc. This is the exact threat the example disallowed-tools pattern addresses.

**Fix.** Uncomment and tighten: at minimum `--allowedTools "Bash(gh pr
*),Bash(gh issue *),Read,Edit,Write"`. For trading-ops safety, also add
`--disallowedTools "Bash(curl *),Bash(pip *)"`.

---

### P-03 — claude.yml: permissions block is read-only

**Evidence.** `claude.yml` sets `contents: read`, `pull-requests: read`,
`issues: read`. The doc examples and the GitHub app grant `read+write` for
contents, pull-requests, and issues.

**Finding.** As currently configured, `@claude` cannot push commits, create
PRs, or post issue comments — it can only read. Triggers work, but any
action that writes back (PR creation, issue reply, code push) will fail with
a 403. This is inconsistent with what the trigger setup advertises.

**Status.** This may be intentional if the repo only wants Claude to answer
questions and not modify anything. If so, document the intent. If Claude is
expected to create PRs from `@claude implement`, the permissions block must
change.

---

### P-04 — claude-code-review.yml: plugin path vs managed service

**Evidence.** `claude-code-review.yml` uses `plugin_marketplaces`, `plugins:
'code-review@claude-code-plugins'`, and `/code-review:code-review ${{
github.repository }}/pull/${{ github.event.pull_request.number }}`. This is
the **self-hosted plugin path** that runs on GH runners.

The docs' managed Code Review is a separate surface: no workflow file,
inline comments posted automatically, $15-25/run, Team/Enterprise only,
incompatible with Zero Data Retention.

**Finding.** Our plugin path is doc-aligned for the plugin approach. The
managed service would be lower-maintenance but costs ~$15-25/run and requires
Team plan. Not a gap — a documented trade-off.

**No change required.** If we move to Team plan post go-live, re-evaluate.

---

### P-05 — claude-code-review.yml: no model pin

**Evidence.** `claude-code-review.yml` has no `claude_args`. Code review
plugin runs on the action's default model (Sonnet).

**Finding.** Same issue as P-01. A review on a complex Python backtest change
will run on Sonnet when Opus 4.7 would be more thorough. Low severity since
the managed Code Review service also has no per-repo model config.

**Fix.** Add `claude_args: '--model claude-opus-4-7'` if the workflow is to
remain self-hosted and cost permits.

---

### P-06 — cron_budget.yaml: disconnected from any scheduler (MF-19, MUST FIX)

**Evidence.** `.claude/cron_budget.yaml` defines 15 slots with `cadence`,
`priority`, and `alpha_velocity_eligible` fields. No code reads this file at
runtime. `scripts/harness/run_harness.py` does not import it. The Slack
bot's `scheduler.py` does not reference it. No GitHub Actions schedule
trigger fires these slots.

**Finding.** The file is a design doc masquerading as configuration. The
header states it is "owned by phase-10.7 Meta-Evolution Engine" — a phase
that does not yet exist. The 15-slot cap is a **self-imposed budget**, not
an Anthropic platform limit. Anthropic's routines page says the daily cap
is account-level and not quantified in docs; the 15-slot number is Peder's
conservative estimate.

**Fix (before phase-10.7).** Add a comment to the YAML header:
"This file is a planning document. It is not read by any scheduler.
Slots are mapped to implementation surface (routine / GH Actions cron /
Slack scheduler) in phase-10.7." Add `surface: unimplemented` to each slot.

**Fix (phase-10.7).** Choose a surface per slot: slots 6-15 → Anthropic
Routines (cloud, 1-hour min, no approval prompts); slots 1-5 (trading-ops)
→ cannot use Routines directly because routines run without permission
prompts. For slots 1-5 use a GH Actions `schedule:` trigger that invokes
`claude -p --bare` with an explicit `--allowedTools` list and requires
a human-approval job step before execution.

---

### P-07 — Routines: not used anywhere in the repo

**Evidence.** Grep for `routine`, `/schedule`, `claude.ai/code/routines`
across all YAMLs, Python, and Markdown returns no hits outside
`cron_budget.yaml`'s comments.

**Finding.** None of the 12 active cron_budget slots are wired to Anthropic
Routines. The morning/evening digest (slots 1-2), watchdog (3),
paper_trading_cycle (4), kill_switch_heartbeat (5), and all research slots
(6-13) run either manually, via the Slack bot scheduler, or via
`run_harness.py` driven by a human. At go-live (May 2026) this is
acceptable — the human-in-the-loop is intentional. Post go-live, Routines
are the correct primitive for unattended research slots (6-13, 15).

**Status.** Not a defect for go-live. Document the intent.

---

### P-08 — /loop and session-scoped scheduling: not a cron replacement

**Evidence.** Docs confirm: "Tasks are session-scoped: they live in the
current conversation and stop when you start a new one." Seven-day expiry.
Requires machine on and session open.

**Finding.** `cron_budget.yaml` slots cannot be satisfied by `/loop`. The
file's framing implies a real cron — this is the source of the aspirational
disconnect. `/loop` is suitable for interactive harness babysitting during a
session (e.g., "check CI every 5m while I'm working") but not for production
scheduled jobs.

**No action required** beyond the P-06 documentation fix.

---

### P-09 — Custom Slack bot: 16 modules, irreplaceable by native integration

**Evidence.** 16 Python modules in `backend/slack_bot/`: `app.py` (Bolt
Socket Mode), `mcp_tools.py`, `assistant_handler.py`, `streaming_handler.py`,
`scheduler.py`, `commands.py`, `governance.py`, `self_update.py`,
`direct_responder.py`, `formatters.py`, `context_management.py`,
`app_home.py`, `digest_test.py`, `assistant_lifecycle.py`,
`streaming_integration.py`, `__init__.py`.

The docs' "Claude Code in Slack" is a routing layer that sends `@Claude`
mentions to a Claude Code on the web session. Works only in channels, not
DMs. Per-user GitHub auth required. Cannot run scheduled digests, call
custom MCP tools, or integrate with the paper-trading ticket queue.

**Finding.** Zero functional overlap with our bot. Our bot is a domain
application; the native integration is a developer productivity tool.
No replacement is possible or desirable.

---

### P-10 — Slack bot: scheduler.py is the de-facto cron for slots 1-2

**Evidence.** `backend/slack_bot/scheduler.py` runs the morning/evening
digests (slots 1-2 in cron_budget). It is not wired to cron_budget.yaml.

**Finding.** The scheduler is a real implementation but is not
documented as the implementation surface for those slots. When
phase-10.7 Meta-Evolution tries to reallocate slots, it will not find
a machine-readable link between `slot_id: 1` and `scheduler.py`.

**Fix.** Add `implementation: slack_bot.scheduler` field to slots 1-2 in
`cron_budget.yaml` and a matching comment in `scheduler.py`.

---

### P-11 — No devcontainer present

**Evidence.** `ls .devcontainer/` returns "No such file or directory."
`find . -name 'devcontainer.json' ...` returns zero hits.

**Finding.** The Anthropic reference devcontainer (Node.js 20, custom
firewall `init-firewall.sh`, `--dangerously-skip-permissions` support) is
not used. For go-live, the harness runs on macOS localhost with `.venv`.
This is acceptable. The devcontainer becomes relevant if: (a) we
containerize the harness for cloud deployment, or (b) a second developer
joins and needs onboarding in < 10 minutes.

**Status.** Nice-to-have. Not required for May 2026 go-live.

---

### P-12 — No LLM gateway

**Evidence.** Grep for `LLM_GATEWAY`, `ANTHROPIC_BASE_URL`, `litellm`,
`llm.gateway` across all Python and YAML returns zero hits. `backend/llm_client.py`
is a multi-provider abstraction for app agents, not a gateway for the Claude
Code CLI.

**Finding.** These are orthogonal concerns. `llm_client.py` routes
analysis-pipeline Gemini/Anthropic/OpenAI calls. An LLM gateway would sit
between the Claude Code CLI and Anthropic's API for centralized cost
tracking, audit, and key management. A gateway is overkill while the harness
runs on one machine with one operator.

**Status.** No gap for go-live. Reconsider if harness moves to cloud or
multi-user.

---

### P-13 — No CLAUDE_CODE_USE_BEDROCK / CLAUDE_CODE_USE_VERTEX flags

**Evidence.** Grep across `.yml`, `.sh`, `.env` for Bedrock/Vertex env vars
returns zero hits.

**Finding.** We use the direct Anthropic API via `CLAUDE_CODE_OAUTH_TOKEN`
in GitHub Actions and local `ANTHROPIC_API_KEY`. Consistent. No Bedrock or
Vertex routing in use. Harness uses direct API; app agents use Vertex (Gemini)
via `llm_client.py`. These are separate credential chains and are correctly
separated.

**Status.** Aligned.

---

### P-14 — Ultraplan not referenced anywhere

**Evidence.** Grep for `ultraplan` across `.claude/`, `CLAUDE.md`, all
Python and YAML returns zero hits (confirmed by live grep).

**Finding.** Ultraplan requires Claude Code v2.1.91+ and Claude Code on the
web access. It is suitable for planning large phases at the web browser
review surface. Our harness uses `contract.md` + planner subagent pattern
which is functionally equivalent but keeps state in the repo. Ultraplan
would not feed `handoff/current/contract.md` automatically — it would
require a custom export step.

**Status.** No gap. Document as a known non-adoption in CLAUDE.md if
Ultraplan becomes a common question.

---

### P-15 — Ultrareview not referenced anywhere

**Evidence.** Same grep result. Zero hits.

**Finding.** Ultrareview (managed Code Review) costs $15-25/run billed as
extra usage. It runs a fleet of agents on Anthropic cloud and posts inline
comments, which our plugin-based `claude-code-review.yml` approximates at
GH runner cost. For the rare substantial PR (major backtest engine rewrite,
schema migration), `/ultrareview` or `@claude review once` would provide
deeper analysis. Not required for go-live.

**Status.** Nice-to-have. No change required.

---

### P-16 — frontend/chrome/: Playwright binary, not Chrome extension

**Evidence.** `ls frontend/chrome/mac_arm-147.0.7727.57/` confirms a
Chromium binary tree. Claude Code docs describe a "Chrome" feature as a
browser automation extension that lets Claude open tabs, read the DevTools
console, and fill forms during development workflows.

**Finding.** These share a name but are entirely different. Our chrome/
directory is downloaded by Playwright for frontend e2e tests. There is no
Claude Code Chrome extension installed or configured. No action required.
The directory name is not misleading to anyone who reads it.

---

### P-17 — governance-lint.yml: project-specific, no Claude Code equivalent

**Evidence.** Runs `scripts/governance/lint_limits_usage.py --strict` on
push to main and on PRs touching backend Python. No Claude Code action used.

**Finding.** This is a custom governance workflow with no Claude Code
equivalent in the docs. Correct approach. The workflow could optionally add
a Claude step that summarises violations for the PR author, but this is
cosmetic. No gap.

---

### P-18 — limits-tag-enforcement.yml: project-specific, no Claude Code equivalent

**Evidence.** Enforces GPG-signed `limits-rotation-YYYYMMDD` tags when
`backend/governance/limits.yaml` changes. Uses `git verify-tag` + secret
`ALLOWED_SIGNER_PUBKEYS`. No Claude Code action used.

**Finding.** Correct approach for cryptographic policy gate. No Claude Code
surface adds value here. The workflow was hardened in phase-4.9 (Cycle 89
dual-YAML fix). Status: aligned.

---

### P-19 — pip-audit.yml: weekly cron schedule wired correctly

**Evidence.** `pip-audit.yml` uses `schedule: - cron: "0 7 * * 1"` (Mondays
07:00 UTC). This IS a real GH Actions scheduler.

**Finding.** This is the only slot in the repo where a time-based cron is
actually wired to an execution surface. All 15 slots in `cron_budget.yaml`
should aspire to this pattern. Pip-audit is not in `cron_budget.yaml` at
all — it predates the budget concept and is treated as infrastructure, not
a Claude routine.

**Status.** No gap. Document pip-audit as a model for how other cron slots
should be wired when phase-10.7 lands.

---

### P-20 — Checkpointing vs handoff protocol: orthogonal

**Evidence.** Docs describe `/rewind` as in-session undo for file edits,
30-day retention, session-scoped. Our `handoff/current/` + `handoff/archive/`
protocol is step-level durable state across sessions.

**Finding.** Checkpointing provides session-level undo; the handoff protocol
provides harness-cycle-level audit and resume. They solve different problems
and are both needed. The handoff protocol is load-bearing for the qa-evaluator
→ harness-verifier cycle and feeds the Harness tab UI. `/rewind` is useful
for individual Claude sessions where a tool call went wrong. No conflict.

---

## Summary: MUST FIX vs NICE TO HAVE

### MUST FIX (before go-live)

| ID | File | Action |
|----|------|--------|
| P-01 | `.github/workflows/claude.yml` | Add `claude_args: '--model claude-opus-4-7'` |
| P-02 | `.github/workflows/claude.yml` | Add `--allowedTools` restriction; uncomment and tighten |
| P-06 | `.claude/cron_budget.yaml` | Add header comment clarifying aspirational status and `surface: unimplemented` per slot |

### SHOULD FIX (before phase-10.7)

| ID | File | Action |
|----|------|--------|
| P-03 | `.github/workflows/claude.yml` | Document intent: read-only permissions or upgrade to write for PR creation |
| P-05 | `.github/workflows/claude-code-review.yml` | Pin model to `claude-opus-4-7` if cost permits |
| P-10 | `cron_budget.yaml` + `slack_bot/scheduler.py` | Add cross-reference between slot_id and implementation module |

### NICE TO HAVE (post go-live)

- P-11: Add devcontainer for team onboarding and CI parity
- P-14/P-15: Mention ultraplan/ultrareview in CLAUDE.md as known non-adoption
- P-07: Wire research slots (6-13, 15) to Anthropic Routines post go-live

---

## Phase-4.11 findings: status check

All 12 findings from phase-4.11 remain valid. This audit adds:

- Confirmed Routines are now out of research preview (feature page live,
  API beta header `experimental-cc-routine-2026-04-01`).
- Confirmed `claude.yml` still lacks model pin and tool restriction
  (MF-17 not yet fixed).
- Confirmed `cron_budget.yaml` still aspirational (MF-19 not yet fixed).

---

## Sources

- `https://code.claude.com/docs/en/github-actions` — v1.0 action, model pin guidance
- `https://code.claude.com/docs/en/routines` — schedule/API/GitHub triggers, autonomy constraints
- `https://code.claude.com/docs/en/scheduled-tasks` — /loop session-scoped limits
- `https://code.claude.com/docs/en/slack` — native routing vs custom bot capabilities
- `https://code.claude.com/docs/en/code-review` — managed service vs plugin path
- `https://code.claude.com/docs/en/ultraplan` — v2.1.91+ requirement, web execution
- `https://code.claude.com/docs/en/llm-gateway` — gateway vs app-level abstraction
- `https://code.claude.com/docs/en/devcontainer` — reference setup, firewall model
- Local files: `.github/workflows/claude.yml`, `.github/workflows/claude-code-review.yml`,
  `.github/workflows/governance-lint.yml`, `.github/workflows/limits-tag-enforcement.yml`,
  `.github/workflows/pip-audit.yml`, `.claude/cron_budget.yaml`,
  `backend/slack_bot/` (16 modules), `frontend/chrome/mac_arm-147.0.7727.57/`
- `handoff/audit/phase-4.11/claude_code_surfaces.md` — prior audit baseline
