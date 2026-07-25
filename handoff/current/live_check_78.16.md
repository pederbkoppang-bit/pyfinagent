# live_check — phase-78.16

**Required (immutable, from `.claude/masterplan.json`):**
> `handoff/current/live_check_78.16.md`: captured pre/post wire kwargs for one
> service on the flag-OFF path, and the mutation.

Date: 2026-07-25 · Cycle 164 · Machine: operator's Mac · venv: `.venv` (Python 3.14)

---

## 1. Captured pre/post wire kwargs — meta_scorer, flag-OFF path

Captured by driving the **real** path — `settings.meta_scorer_model` →
`make_client` → `ClaudeClient` → request assembly — with only the Anthropic SDK
boundary faked, so nothing left the machine. `paper_use_claude_code_route=False`
throughout: this is the state an operator lands in after the documented one-flag
revert.

Script: `scratchpad/live_capture_78_16.py`. Verbatim stdout:

```
### PRE  -- make_client called WITHOUT the caching argument (exactly what 78.1 shipped)
  service            : meta_scorer
  settings.meta_scorer_model : claude-haiku-4-5
  paper_use_claude_code_route: False   (the documented one-flag revert)
  client type        : ClaudeClient  (metered / direct Anthropic)
  type(system)       : list
  system blocks      : 1
  block[0].type      : text
  len(block[0].text) : 19075
  block[0].cache_control : {"type": "ephemeral", "ttl": "1h"}
  other kwargs       : ['max_tokens', 'messages', 'model', 'output_config', 'temperature']
  model / max_tokens / temperature : claude-haiku-4-5 / 512 / 0.0

### POST -- make_client called the way meta_scorer.py:242 now calls it (enable_prompt_caching=False)
  service            : meta_scorer
  settings.meta_scorer_model : claude-haiku-4-5
  paper_use_claude_code_route: False   (the documented one-flag revert)
  client type        : ClaudeClient  (metered / direct Anthropic)
  type(system)       : str
  len(system)        : 19075
  cache_control      : ABSENT
  other kwargs       : ['max_tokens', 'messages', 'model', 'output_config', 'temperature']
  model / max_tokens / temperature : claude-haiku-4-5 / 512 / 0.0

### DIFF (the only field that changes)
  system: list -> str
  every other kwarg identical: True

### PRE-78.1 EQUIVALENCE
  the pre-78.1 construction was: ClaudeClient(model_name='claude-haiku-4-5', api_key=..., enable_prompt_caching=False)
  that object's enable_prompt_caching : False
  post-78.16 make_client(...) yields  : False
  MATCH: True
```

**Reading of this capture.** PRE is the defect: on the revert path the `system`
field is a 1-block list carrying `cache_control {"type":"ephemeral","ttl":"1h"}`.
POST is the pre-78.1 shape: a plain 19,075-char `str`, no `cache_control`. Every
other kwarg is byte-identical between the two (`every other kwarg identical:
True`), so the change is scoped to exactly the field the 78.1 Q/A flagged and
nothing else moved. The final block closes the loop against the *actual*
pre-78.1 construction, not a description of it.

---

## 2. Mutation matrix

Immutable verification command used as the probe for every case:

```
.venv/bin/python -m pytest backend/tests/ -q -k 'llm_client or make_client or prompt_caching'
```

Protocol per case: apply → purge `__pycache__` (per step 78.14's stale-bytecode
finding) → run → restore from a byte-copy backup of the working tree → verify the
restore by SHA-256. Restore is from backup, **not** `git checkout`: the 78.16
edits are uncommitted, so `git checkout` would have destroyed the work rather
than undone the mutation.

Script: `scratchpad/mutate_78_16.sh`.

| # | Mutation | Expected | Observed | Revert |
|---|----------|----------|----------|--------|
| baseline | none | GREEN | `19 passed, 2016 deselected` | — |
| M1 | `meta_scorer` drops `enable_prompt_caching=False` from its `make_client` call (the 78.1 regression, re-injected) | RED | `1 failed, 18 passed` — `test_revert_path_restores_pre_78_1_request_shape[meta_scorer-meta_scorer_model]` | sha match |
| M2 | `make_client` accepts the parameter but drops it on the way to `ClaudeClient` (the exact defect 78.1 shipped) | RED | `6 failed, 13 passed` — all six `test_revert_path_restores_pre_78_1_request_shape[…]` | sha match |
| M3 | default flips `None` → `False` (would change behaviour for the 7 callers this step was never scoped to touch) | RED | `1 failed, 18 passed` — `test_make_client_default_leaves_class_default_untouched` | sha match |
| M4 | mutate the **stub**: SDK fake overwrites `system` with a plain str | RED | `2 failed, 17 passed` — `test_make_client_forwards_caching_true`, `test_make_client_default_leaves_class_default_untouched` | sha match |

Final state after the matrix: all three touched files SHA-identical to their
pre-matrix state, `19 passed`.

**M4 deserves a sentence the table does not give it** (raised as finding N1 by
the Q/A, and it is the most informative result in the matrix). Under the
str-forcing stub, all six `test_revert_path_restores_pre_78_1_request_shape`
cases stay **GREEN** — because a lying fixture produces exactly the plain `str`
they assert. Taken alone those six therefore cannot distinguish *"the code
correctly emits a str"* from *"the fixture lied"*. What closes that hole is the
pair that M4 does kill: `test_make_client_forwards_caching_true` and
`test_make_client_default_leaves_class_default_untouched`, which assert the
**cached** shape and so cannot be satisfied by a str-forcing stub. Both live in
the same file and run under the same immutable command, so the suite as a whole
is non-vacuous — but only as a whole.

### M1's first run was GREEN — and the *mutation* was the bug, not the guard

Disclosed rather than quietly re-run, because a green mutation is exactly the
signal this project treats as a vacuity finding.

First attempt applied `t.replace(", enable_prompt_caching=False)", ")", 1)` to
`meta_scorer.py`. That matched the **first** occurrence in the file — which is
inside the phase-78.16 explanatory comment (`… constructed
ClaudeClient(..., enable_prompt_caching=False)`), not the call on line 242. So
nothing behavioural changed and the suite correctly stayed green.

Verified by re-applying M1 against the full call-line text and re-running:

```
M1 applied to the CALL line
242:    client = make_client(getattr(settings, "meta_scorer_model", "claude-haiku-4-5"), None, settings)
FAILED backend/tests/test_phase_78_16_prompt_caching_intent.py::test_revert_path_restores_pre_78_1_request_shape[meta_scorer-meta_scorer_model]
1 failed, 18 passed, 2016 deselected, 1 warning in 6.27s
```

Then restored and re-confirmed green (`19 passed`), sha
`ea4a2f87b7a697c266e03f65a2028a2101fc8cca56bc87907c67a6337db92b2b`.

This is `feedback_executor_sees_mutation_transients` in its other form: diff the
observed strings against the mutation you *intended* before calling it an
incident. The lesson worth keeping is narrower and reusable: **when the code
under mutation is documented by a comment that quotes the code, a
`replace(..., 1)` mutation will hit the comment.** Target the full call-line, or
count occurrences first.

---

## 3. What this live_check does NOT prove (stated so nobody over-reads it)

- **Not proven: whether prompt caching would even engage on Haiku 4.5.** The
  block is 19,075 chars; Anthropic's documented minimum for Haiku 4.5 is 4,096
  **tokens**, and the three heuristics in the research brief straddle it
  (3,877 / 4,551 / 4,769). The authoritative check is
  `cache_creation_input_tokens > 0` on a real Haiku response, which needs live
  direct-API credits (owed operator action 79.3). This step does not depend on
  that number — it restores the prior shape either way — but the *follow-up*
  question of flipping the six to `True` does, and is queued for that reason.
- **Not proven: any live production call.** `model='claude-haiku-4-5'` has zero
  `provider='anthropic'` rows in `llm_call_log` over 60 days (BigQuery MCP,
  bounded query), so the metered path for these six is not currently exercised
  in production at all. The divergence this step fixes is **latent today** and
  goes live when credits return. That is an argument for fixing it now, not for
  deferring it — but it does mean no production row can witness the fix yet.
- The SDK boundary was faked in both captures. That is deliberate (no credits, no
  spend, hermetic) and is the same seam `test_claude_request_shapes.py` uses.
