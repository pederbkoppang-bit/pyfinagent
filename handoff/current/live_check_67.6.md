# live_check 67.6 -- builder-level request-payload dumps + SDK pin, verbatim

Required shape (masterplan 67.6): "verbatim builder-level request-payload dumps for a
fable-5 and a sonnet-5 shape (no temperature, no budget_tokens, effort in
output_config) plus pip show anthropic output matching the reconciled pin".

## Request-payload dumps (venv, 2026-07-10; SDK boundary faked, entire real
## request-construction path exercised; system/messages bulk elided for size)

```
--- claude-fable-5 request kwargs ---
{
  "model": "claude-fable-5",
  "max_tokens": 2048,
  "system": "<19355 chars omitted>",
  "messages": "<1 user message omitted>",
  "output_config": {
    "effort": "xhigh"
  }
}
--- claude-sonnet-5 request kwargs ---
{
  "model": "claude-sonnet-5",
  "max_tokens": 2048,
  "system": "<19355 chars omitted>",
  "messages": "<1 user message omitted>",
  "thinking": {
    "type": "adaptive"
  },
  "output_config": {
    "effort": "high"
  }
}
```

Note: NO `temperature`, NO `top_p`/`top_k`, NO `budget_tokens` in either dump; the
fable-5 dump has NO `thinking` key at all (always-on family); effort rides in
output_config -- xhigh surviving on fable proves the spurious-downgrade fix.
Config used: fable `{"thinking": {"budget_tokens": 1024}, "effort": "xhigh"}` (the
trap input that previously produced enabled+budget_tokens + temperature), sonnet-5
`{"thinking": {"budget_tokens": 1024}}` (effort from the new fallback row).

## SDK pin reconciliation

```
$ pip show anthropic | head -2
Name: anthropic
Version: 0.96.0
$ grep -c "anthropic==0.96.0" backend/requirements.txt
1
```

## Gates over this diff (2026-07-10)

```
=== lint gate rerun ===
All checks passed!
exit=0
=== full regression rerun ===
25 passed in 0.28s
=== import smoke ===
orchestrator import OK
```

## Fresh Q/A verdict JSON

Returned by qa-67-6 2026-07-10: `verdict: PASS, ok: true, violated_criteria: [],
certified_fallback: false`, 14 checks_run -- incl. its OWN fable-5 payload
reproduction at the SDK boundary (thinking key: False / temperature: False /
effort xhigh -- matching the dumps above), its own 67.1 gate runs (lint exit=0,
imports OK), a budget_tokens-producer reachability grep (no bypass path), and a
consumer-contract check on the removed imports. Full JSON:
evaluator_critique_67.6.md.
