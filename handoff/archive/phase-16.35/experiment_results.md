---
step: phase-16.35
cycle_date: 2026-04-25
forward_cycle: true
expected_verdict: PASS
deliverable: docs/architecture/claude-code-as-api-substitute.md
---

# Experiment Results -- phase-16.35

## What was done

Researcher subagent fetched 8 sources in full + 10 snippet sources (18 total) covering the Anthropic Agent SDK, Claude Code Remote, Max subscription terms, and the April 4, 2026 third-party-subscription policy change. Wrote a 528-line design doc at `docs/architecture/claude-code-as-api-substitute.md`. NO code shipped.

### Files touched

| Path | Action | Size |
|------|--------|------|
| `docs/architecture/claude-code-as-api-substitute.md` | CREATED (researcher) | 528 lines |
| `handoff/current/contract.md` | rewrite (rolling) | — |
| `handoff/current/experiment_results.md` | rewrite (this) | — |
| `handoff/current/phase-16.35-research-brief.md` | created (researcher) | — |

NO backend/frontend code modified.

## Verification (verbatim, immutable)

```
$ test -f docs/architecture/claude-code-as-api-substitute.md && grep -qE 'Recommendation|Upsides|Downsides|Implementation sketch' docs/architecture/claude-code-as-api-substitute.md && echo "verification PASS"
verification PASS

$ wc -l docs/architecture/claude-code-as-api-substitute.md
528
```

**Result: PASS.** File exists; all 4 required keywords present.

## Bottom-line answer to the user's question

**Claude Code / Claude Code Remote CANNOT substitute for the Anthropic API key.** The April 4, 2026 Anthropic policy change explicitly ended third-party subscription quota access. Max subscription pays for the official Anthropic apps (Claude.ai, Claude Code CLI for personal use), NOT for programmatic SDK calls from pyfinagent.

### Key findings (cited in the doc)

1. **Agent SDK Python (`claude-agent-sdk`) REQUIRES `ANTHROPIC_API_KEY=sk-ant-api03-*`.** Officially documented; SDK refuses to use OAuth subscription tokens for programmatic calls.

2. **"Claude Code Remote" is a mobile-sync feature, NOT a cloud API.** It connects a phone/browser to a Claude Code session running on your Mac. No programmatic endpoint exposed.

3. **Setting `ANTHROPIC_API_KEY` env var silently switches Claude Code CLI from Max-subscription to API-token billing.** Real risk: if Peder adds the key to `backend/.env` and any shell-sourcing leaks it (e.g., `source backend/.env` in iTerm), the next `claude` invocation bills the API instead of Max.

4. **Subprocess overhead.** If we did try to use `claude -p` as a subprocess (grey-area subscription path), it adds 500-1500ms per call + complex async integration. Zero cost benefit at pyfinagent volumes (~$3-10/mo at native-API rates per phase-16.27 trading-MAS doc).

### Upsides of using Claude Code subprocess (none decisive)
- IF the grey-area unenforced personal-use path stays open: $0 marginal cost (Max already paid)
- IF Claude Code Remote API ever exists (no current plans Anthropic has announced): could be cleaner for cloud-hosted pyfinagent
- Forces standardization on Claude Code's CLAUDE.md harness model

### Downsides of using Claude Code subprocess (decisive)
- **VIOLATES Anthropic's April 4, 2026 policy** for any third-party tool use
- **NO SLA / NO guarantee** — Anthropic can revoke at any time without notice
- **+500-1500ms latency** per call (subprocess startup, JSON-RPC, CLAUDE.md discovery)
- **ANTHROPIC_API_KEY env-var leakage** silently switches Claude Code to per-token billing
- **`_call_agent_with_tools` uses bespoke thinking params** (Opus 4.7 adaptive vs 4.6 enabled) that don't translate to Agent SDK abstractions
- **Async-only SDK** raises `RuntimeError: This event loop is already running` from FastAPI handlers without thread-executor scaffolding
- **Without `--bare`, `claude -p` loads pyfinagent's full CLAUDE.md** (harness protocol, ~10K+ tokens) on every call, adding noise and bills

### Doc structure (10 sections, per researcher)

§1 Current state — sk-ant-oat-* in `backend/.env`; 3 call sites blocked
§2 Option A — Claude Agent SDK (Python subprocess)
§3 Option B — Claude Code Remote (cloud-hosted)
§4 Upsides per option
§5 Downsides per option (with citations)
§6 Implementation sketch (which files would change + subprocess invocation patterns)
§7 Cost analysis — Max $200/mo flat vs API ~$3-10/mo at pyfinagent volumes
§8 Reliability + risk — what breaks if Anthropic changes Max terms (and they JUST DID)
§9 **Recommendation: paste the sk-ant-api03-* key (existing #21 path).** Don't try to substitute Claude Code.
§10 What this CLOSES — does it replace #21? **No.** It confirms #21 is the only path.

## Success criteria assessment

| # | Criterion | Result | Evidence |
|---|-----------|--------|----------|
| 1 | doc_exists | PASS | 528 lines at canonical path |
| 2 | claude_code_sdk_evaluated | PASS | §2 Option A (Agent SDK Python) |
| 3 | claude_code_remote_evaluated | PASS | §3 Option B (Claude Code Remote = mobile sync, not cloud API) |
| 4 | cost_comparison_present | PASS | §7 — Max $200/mo flat vs API ~$3-10/mo |
| 5 | implementation_sketch_present | PASS | §6 — file:line anchors for each call site |
| 6 | honest_recommendation | PASS | §9 — explicit "no, paste the API key" with rationale |

## Honest disclosures

1. **Recommendation is "no Claude Code substitute" — that's the honest answer.** The doc doesn't sugarcoat. Claude Max is a subscription product for Anthropic's first-party apps; pyfinagent is a third-party tool.

2. **Grey area documented, not endorsed.** The doc notes that `claude -p` without `ANTHROPIC_API_KEY` MIGHT use subscription quota for personal use, but explicitly flags this as policy-prohibited and unenforced — not a reliable engineering decision.

3. **Env-var leakage caveat is critical.** If Peder adds the API key to `backend/.env` and runs `set -a; . backend/.env; set +a` in any shell (which the autoresearch script does), the resulting shell-env `ANTHROPIC_API_KEY` will switch Claude Code CLI to API billing. The doc surfaces this; the mitigation is to use a separate env-loading mechanism for backend that doesn't leak to user's shell.

4. **No code shipped this cycle.** The doc is reading material for Peder's decision.

5. **Does NOT close #21.** Confirms the only path is paying for the API key.

6. **April 4, 2026 policy change** is the decisive evidence. Researcher cites PYMNTS.com source; Q/A may verify by curl.

## No-regressions

`git diff --stat`:
- `docs/architecture/claude-code-as-api-substitute.md` (NEW, 528 lines)
- handoff/* (rolling)

NO backend/frontend code touched.

## Closes

- masterplan step **phase-16.35** (research deliverable)
- Does NOT close #21 (the user-action item remains pending — but now the user knows the alternatives are not viable)

## Next

Spawn Q/A to audit doc honesty + completeness. If PASS → log + flip → continue with **phase-10.7.3** (Algorithm Discovery archetype seed library).
