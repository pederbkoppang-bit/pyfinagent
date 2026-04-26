---
step: phase-16.58
title: Close pending Anthropic key swap (sk-ant-oat-* -> sk-ant-api03-*) -- task #21 closure
cycle_date: 2026-04-26
harness_required: true
verification: 'source .venv/bin/activate && python -c "from dotenv import dotenv_values; e = dotenv_values(''backend/.env''); k = e.get(''ANTHROPIC_API_KEY'',''''); assert k.startswith(''sk-ant-api03''), f''wrong prefix: {k[:12]}''; assert len(k) == 108, f''wrong length: {len(k)}''; print(''ok'')"'
research_brief: handoff/current/phase-16.58-research-brief.md
---

# Contract -- phase-16.58

## Step ID

`phase-16.58` -- "Close pending Anthropic key swap (sk-ant-oat-* ->
sk-ant-api03-*) -- task #21 closure". Operator-requested closure
following successful smoke test of the new sk-ant-api03 key.

## Research-gate summary

Spawned researcher (simple tier, internal-only). Brief at
`handoff/current/phase-16.58-research-brief.md` (10KB, on disk
verified). gate_passed=true.

Decisive findings:
- **2 prefix-guard files** correctly gate on `startswith("sk-ant-api")`:
  `directive_rewriter.py:173` + `directive_review.py:132`. These already
  reject OAT tokens by design.
- **3 log-scrub regex files** already cover both key formats:
  `mcp_capabilities.py:179`, `streaming_integration.py:394`,
  `signal_attribution.py:30`.
- **backend/.env has TWO `ANTHROPIC_API_KEY=` lines:**
  - Line 15: `sk-ant-oat01-*` (DEAD; harmless because python-dotenv
    last-wins makes line 57 active)
  - Line 57: `sk-ant-api03-*` (CORRECT; what the smoke test confirmed)
- **No code path REJECTS the new key format.** All checks accept `sk-ant-api`.
- **Recommended cleanup:** delete line 15 from backend/.env (hygiene only;
  not load-bearing). No code changes needed.

## Hypothesis

The key swap is FUNCTIONALLY complete (smoke test PASSED with
`claude-sonnet-4-6` returning `"ok"`). Closing task #21 just requires:
1. Verifying the active key has the right prefix + length (the
   immutable verification command does this)
2. Optional cleanup: removing the dead OAT line from `backend/.env`
3. Logging the closure + flipping the masterplan

If the OAT-line removal succeeds via Bash sandbox, do it. If denied,
document for operator manual cleanup and proceed (the duplicate is
harmless under last-wins dotenv semantics).

## Immutable success criteria

```
source .venv/bin/activate && python -c "from dotenv import dotenv_values; e = dotenv_values('backend/.env'); k = e.get('ANTHROPIC_API_KEY',''); assert k.startswith('sk-ant-api03'), f'wrong prefix: {k[:12]}'; assert len(k) == 108, f'wrong length: {len(k)}'; print('ok')"
```

Confirms the active key (last value wins per dotenv spec) is the
108-char `sk-ant-api03-*` Console key, NOT the dead OAT token.

## Plan steps

1. Try removing line 15 (OAT token) from `backend/.env` via python script
   using `dotenv_values` + filtered rewrite. If sandbox blocks, fall back
   to documenting for operator manual cleanup.
2. Run immutable verification command -- confirms active key shape is correct.
3. Optional: live-call smoke test (already passed in prior turn -- can re-run).

## References

- `handoff/current/phase-16.58-research-brief.md`
- `backend/meta_evolution/directive_rewriter.py:173` (prefix guard)
- `backend/meta_evolution/directive_review.py:132` (prefix guard)
- Task #21 (pending since 16.20)
- Phase-19.0 decision document at `docs/architecture/claude-remote-handoff-feasibility.md`

## Out of scope

- Changes to the prefix-guard code (correctly designed; rejecting OAT is intentional)
- Changes to log-scrub regex (already covers both formats)
- Phase-19.1 spike implementation (separate cycle)
- Mid-session key rotation testing (16.31 already verified `reset_anthropic_client()`)
