# Claude Code Surfaces Deep Audit (phase-4.11.5)

Date: 2026-04-18. Scope: all 33 Claude Code surface docs vs pyfinAgent's
current implementation. AUDIT ONLY — no code changed.

## URL coverage

All 33 URLs fetched in full and parsed (HTML stripped, text read
end-to-end, not abstracts):

CLI surface: cli-reference, commands, statusline, output-styles,
keybindings, terminal-config, voice-dictation, fullscreen, fast-mode,
interactive-mode, headless, checkpointing. Web surface: web-quickstart,
ultraplan, ultrareview, routines. Desktop: desktop, desktop-quickstart,
desktop-scheduled-tasks. Scheduling: scheduled-tasks. CI: github-actions,
gitlab-ci-cd, code-review. Chat: slack, channels, channels-reference.
Automation: remote-control, chrome. IDE: vs-code, jetbrains. Infra:
devcontainer, llm-gateway, network-config.

All pages were read from `/tmp/cc_docs/*.html` with nav/script stripped.

## Scheduled cron/routines vs our cron_budget concept

Our `.claude/cron_budget.yaml` encodes a hard cap of 15 "Claude
scheduled routine runs/day" with reserved trading-ops slots (1-5) and
research slots (6-15). This cap is a **self-imposed accounting
construct** — it does NOT map to any published Anthropic limit.

The docs describe three distinct scheduling surfaces:

| Surface | Runs on | Min interval | Survives machine off? | Used via |
|---|---|---|---|---|
| Routines (cloud) | Anthropic cloud | 1 hour | Yes | `/schedule` in CLI or claude.ai/code/routines |
| Desktop scheduled tasks | Local machine | 1 minute | Only while app open | Desktop app → Schedule |
| `/loop` | In-session | 1 minute | No (session-scoped, 7d expiry) | `/loop 5m <prompt>` |

Key findings:
- `scheduled-tasks` doc states "Tasks are session-scoped: they live in
  the current conversation and stop when you start a new one." This is
  **incompatible** with pyfinagent's model where cron_budget slots
  (morning_digest, paper_trading_cycle, watchdog etc.) must run
  independently of any open session.
- Routines are the correct fit for slots 1-12, BUT routines have their
  own "daily run allowance" tied to the claude.ai account (the doc
  says "count against your account's daily run allowance" but doesn't
  quantify it in the routines page itself). Our 15-slot/day cap is
  **not** a documented Anthropic limit; it appears to be Peder's
  conservative budget. Verify against the actual
  `usage-and-limits` section on claude.ai/code/routines before
  treating 15 as real.
- Routines run "autonomously as full Claude Code cloud sessions: there
  is no permission-mode picker and no approval prompts during a run."
  This conflicts with the cron_budget's `priority: reserved` +
  "human-in-loop approval required" rule for trading-ops. **If we
  migrate slots 1-5 to routines, we lose the approval gate** — need an
  external gate (GitHub PR, ticket queue) instead.
- Currently pyfinagent does NOT use routines at all. The 15-slot
  budget is enforced only by `scripts/harness/run_harness.py` and
  `backend/slack_bot/scheduler.py`. The file is aspirational for
  phase-10.7 Meta-Evolution. Status: **design doc, not wired**.
- Desktop scheduled tasks are irrelevant for our go-live (no Mac
  always-on server in the loop).
- `/loop` is only useful for the interactive harness driver while
  Claude is running; does not replace cron.

## GitHub/GitLab CI alignment (our 5 workflows)

Workflows present in `.github/workflows/`:

1. `claude.yml` — matches `github-actions.md` v1.0 setup exactly:
   `anthropics/claude-code-action@v1`, `@claude` mention trigger,
   `CLAUDE_CODE_OAUTH_TOKEN` secret, `additional_permissions:
   actions: read`. **Doc-aligned.** One gap: doc notes "Claude Code
   GitHub Actions default to Sonnet. To use Opus 4.7, configure the
   model parameter to use `claude-opus-4-7`." We do not pin the
   model — worth adding `claude_args: '--model claude-opus-4-7'` for
   consistency with our harness model choice.
2. `claude-code-review.yml` — installs the `code-review` plugin from
   `claude-code-plugins` marketplace and runs
   `/code-review:code-review` on every PR. This is the **plugin-based
   review**, NOT the managed Code Review service described in
   `code-review.md`. The docs' native Code Review is a research
   preview for Team/Enterprise only and posts inline comments
   automatically without a workflow file. Ours is functionally
   similar but self-hosted on GH runners — doc-aligned for the
   plugin path.
3. `governance-lint.yml`, `limits-tag-enforcement.yml`,
   `pip-audit.yml` — project-specific governance; no Claude Code
   equivalent in the docs. Out of scope.

No GitLab usage. `gitlab-ci-cd.md` irrelevant unless we migrate.

**Gaps:**
- We do not use `@claude review once` or the managed Code Review
  service. Could replace `claude-code-review.yml` with the managed
  service (lower maintenance), but it requires Team/Enterprise plan
  and is incompatible with ZDR. Nice-to-have, not required.
- Missing: the docs' suggested `claude_args: "--allowed-tools Bash(gh
  pr *)"` pattern on `claude.yml` to prevent action from running
  arbitrary bash. We currently allow defaults, which is broad.
  **Should tighten** for security.

## Slack/Code-review native vs our custom bot

Our Slack implementation is **substantial** (`backend/slack_bot/`,
~16 modules): `app.py` (Bolt Socket Mode entry, 71 LoC), `mcp_tools.py`
(247 LoC), `assistant_handler.py`, `streaming_handler.py`,
`scheduler.py`, `commands.py`, `governance.py`, `self_update.py`,
plus a `Dockerfile`. Uses the MAS orchestrator for domain-specific
work (paper trading, backtest summaries, digests, ticket queue).

The docs' "Claude Code in Slack" is a completely different beast:
- It's built on the existing **Claude for Slack** marketplace app
  + intelligent routing that spawns a **Claude Code on the web
  session** when it detects a coding request.
- Modes: "Code only" or "Code + Chat." Repository-scoped.
- Works only in channels (not DMs), requires a per-user claude.ai +
  GitHub link.
- Triggered by `@Claude` mentions.

**Does native Slack replace our custom bot? No.** The native
integration is a developer-productivity tool for repo work. Our bot
is a **domain application** (portfolio/trading/digest) with custom
MCP tools, scheduled digests, streaming responses, and ticket-queue
integration. Zero overlap.

One adjacent concept worth noting: the bot's `scheduler.py` currently
runs scheduled digests. If those digests ever moved to Claude's
infrastructure, **routines** (with a schedule trigger + Slack
connector posting to a channel) could replace that code. Not urgent.

## IDE integrations, headless, devcontainer, LLM gateway

**Headless / Agent SDK CLI (`-p` flag, `--bare`):**
`scripts/harness/run_harness.py` already spawns Claude sub-agents, but
it shells out via the Claude Code CLI, not the Agent SDK Python
package. The docs recommend `--bare` for CI/scripts to skip
auto-discovery of hooks/skills/MCP — our harness DOES want discovery
(it relies on project hooks and skills), so `--bare` is **not** a fit
for the main harness. It IS a fit for governance-lint and pip-audit
style invocations if we add Claude checks there. Current
implementation is doc-aligned given the harness needs full context.

**Devcontainer:** The docs' devcontainer pairs VS Code Dev Containers
+ a firewall `init-firewall.sh` + `--dangerously-skip-permissions` for
unattended runs. pyfinagent has no devcontainer. For go-live
(phase-4.x) we run on macOS/localhost with a `.venv` and systemd-style
scheduler. **Nice-to-have** for team onboarding, **not required** for
go-live. If we ever containerize the harness, the reference
devcontainer + firewall is a strong starting point (restrict egress
to Anthropic API, BigQuery, Slack, GitHub only).

**LLM gateway:** Docs describe a gateway sitting between Claude Code
and providers (LiteLLM pattern), useful for centralized auth, cost
tracking, audit, routing. We have `backend/llm_client.py` which is a
**multi-provider abstraction** (Vertex/Anthropic/OpenAI) for our app
code — NOT a gateway for the Claude Code CLI itself. These solve
different problems:
- Our `llm_client.py`: app → LLM routing for analysis-pipeline agents.
- CC gateway: CLI → provider for Claude Code sessions.

A gateway could be useful to centralize harness cost tracking if the
harness ever runs in multiple environments. Today the harness runs
one place (Peder's machine), so gateway is overkill. **Keep
`llm_client.py` as-is.**

**VS Code / JetBrains:** Not wired. Neither is used for the
autonomous harness. Out of scope.

**Chrome:** The docs' Chrome feature is a **browser automation
extension** (Claude opens tabs, reads console, fills forms) for
development workflows. Our `frontend/chrome/mac_arm-147.0.7727.57/` is
a **downloaded Chromium binary** used by Playwright for frontend
e2e tests. **Unrelated.** Same word, different things.

**Remote Control / Channels:** Could provide a lightweight chat
bridge into a running harness session (Telegram, Discord, iMessage)
without building a custom bot. Not a replacement for the domain
Slack bot, but could replace the "ops ping Peder on his phone"
direct-responder path. **Nice-to-have** for phase-4 go-live when
Peder is off-keyboard.

**Checkpointing:** Docs describe in-session `/rewind` for undoing
Claude file edits (session-scoped, 30d retention). Our "file-based
cycle state" in `handoff/current/` + `handoff/archive/phase-X.Y/` is
a **different concern** (step-level durable artifacts across sessions,
feed the next cycle, audit trail). Checkpointing does NOT win; they
are complementary. Keep the handoff protocol.

**Ultraplan / Ultrareview:** Both are cloud-delegated flows
(`/ultraplan`, `/ultrareview`). Ultrareview bills as extra usage
($5-20/run after 3 free). They do NOT obviate our Plan+Generate+
Evaluate harness because:
- Our harness is **domain-specific** (contract.md with immutable
  verification criteria copied from masterplan.json, dual-evaluator
  rule, harness-verifier runs the actual command).
- Ultraplan/Ultrareview are **general-purpose**. They don't know
  about our five-file protocol, harness_log.md, or the Alpha Velocity
  feedback loop.
- CLAUDE.md is explicit: self-evaluation is forbidden; we require
  qa-evaluator AND harness-verifier in parallel. Ultrareview's
  multi-agent verification IS similar in spirit but lives in
  Anthropic's cloud and doesn't feed our harness_log.md.

They could **augment** but not replace: consider
`/ultrareview` on the final PR before a merge for substantial
changes. Low priority.

**Fast mode (`/fast`):** Docs confirm `/fast` is real. It's a speed
toggle for **Opus 4.6** (NOT 4.7, which is what we use). Pricing:
$30/$150 MTok, billed as extra usage only. CLAUDE.md does not
currently mention `/fast` — the claim that "CLAUDE.md mentions /fast
toggle" in the audit prompt is **incorrect** (grep-checked: no
mention). Since we're on Opus 4.7 per this session's model banner,
`/fast` is **not available to us**.

## Findings

1. **cron_budget.yaml is aspirational.** Not wired to routines or any
   published Anthropic limit. The "15/day" cap is a Peder-imposed
   budget, not a platform ceiling.
2. **Routines are the right primitive** for slots 6-13, 15
   (research/backtest/autoresearch). Slots 1-5 (trading-ops) need an
   external approval gate because routines run without permission
   prompts.
3. **claude.yml workflow should pin the model** to
   `claude-opus-4-7` for consistency and pin `--allowed-tools` to
   reduce action privilege.
4. **claude-code-review.yml is doc-aligned** (plugin path); could
   simplify to managed Code Review if we move to Team plan and don't
   need ZDR.
5. **Custom Slack bot is irreplaceable** by native Slack integration.
   Different problem domains.
6. **frontend/chrome/ is unrelated** to the Chrome extension docs.
7. **Ultrareview could augment** PR review for substantial diffs; not
   a replacement for the harness.
8. **`/fast` is unavailable** to us (Opus 4.7, not 4.6).
9. **llm_client.py ≠ LLM gateway.** Keep as-is.
10. **No devcontainer** — nice-to-have for team/CI parity, not
    required for go-live.
11. **Checkpointing is orthogonal** to the handoff protocol. Neither
    replaces the other.
12. **Channels (Telegram/Discord/iMessage)** could provide a
    lightweight phone bridge into running harness sessions. Would
    reduce custom direct-responder code if we ever migrate.

## MUST FIX

1. Pin model in `.github/workflows/claude.yml`:
   `claude_args: '--model claude-opus-4-7 --allowed-tools
   "Bash(gh pr *),Read,Edit"'`. Current defaults are Sonnet + broad
   tool access.
2. Add a note to `.claude/cron_budget.yaml` header clarifying the
   15/day cap is self-imposed (NOT an Anthropic limit) and document
   which surface (routine vs desktop vs /loop) each slot maps to. As
   drafted today, the slots are surface-agnostic and therefore
   unimplementable.
3. Before phase-10.7 wires Meta-Evolution to actually schedule jobs,
   decide: routines (cloud, right fit, loses approval gate for
   slots 1-5) or a self-hosted cron that invokes
   `claude -p --bare ...`. Document the choice.

## NICE TO HAVE

- Add `/ultrareview` to pre-merge checklist for substantial PRs.
- Add a devcontainer with firewall rules for team onboarding and
  CI parity.
- Consider a Channels (iMessage) plugin as a cheaper phone bridge
  than the custom Slack direct-responder path.
- If moving to Team plan post-go-live, evaluate replacing
  `claude-code-review.yml` with managed Code Review.

## References

Primary docs (all under https://code.claude.com/docs/en/):
routines, scheduled-tasks, desktop-scheduled-tasks, github-actions,
gitlab-ci-cd, code-review, slack, chrome, ultraplan, ultrareview,
checkpointing, headless, devcontainer, llm-gateway, fast-mode,
remote-control, channels, channels-reference, vs-code, jetbrains,
cli-reference, commands, interactive-mode, network-config, statusline,
output-styles, keybindings, terminal-config, voice-dictation,
fullscreen, web-quickstart, desktop, desktop-quickstart.

Local files inspected:
- `/Users/ford/.openclaw/workspace/pyfinagent/.claude/cron_budget.yaml`
- `/Users/ford/.openclaw/workspace/pyfinagent/.github/workflows/claude.yml`
- `/Users/ford/.openclaw/workspace/pyfinagent/.github/workflows/claude-code-review.yml`
- `/Users/ford/.openclaw/workspace/pyfinagent/.github/workflows/governance-lint.yml`
- `/Users/ford/.openclaw/workspace/pyfinagent/.github/workflows/limits-tag-enforcement.yml`
- `/Users/ford/.openclaw/workspace/pyfinagent/.github/workflows/pip-audit.yml`
- `/Users/ford/.openclaw/workspace/pyfinagent/backend/slack_bot/` (16 modules, 71 LoC entry)
- `/Users/ford/.openclaw/workspace/pyfinagent/backend/llm_client.py` (referenced)
- `/Users/ford/.openclaw/workspace/pyfinagent/scripts/harness/run_harness.py` (1206 LoC)
- `/Users/ford/.openclaw/workspace/pyfinagent/frontend/chrome/mac_arm-147.0.7727.57/` (Playwright binary)
