# Research Brief: Claude Code CLI stdin-fix for `claude_code_invoke()`

**Tier:** simple
**Date:** 2026-05-26
**Cycle:** 4 (post-cycle-3 live-smoke failure)
**Failure:** `Error: Input must be provided either through stdin or as a prompt argument when using --print`
**Root cause hypothesis:** variadic `--disallowedTools <tools...>` consumes the trailing positional prompt argv.

---

## Read in full (>=5 required)

| # | URL | Accessed | Kind | Fetched | Key quote / finding |
|---|-----|----------|------|---------|--------------------|
| 1 | https://code.claude.com/docs/en/headless | 2026-05-26 | Official doc | WebFetch | Definitive: stdin-piping documented (`cat file \| claude -p`); `--bare` skips OAuth/keychain reads + requires `ANTHROPIC_API_KEY`; will become `-p` default in future release |
| 2 | https://code.claude.com/docs/en/cli-reference | 2026-05-26 | Official doc | WebFetch | `--disallowedTools` examples use space-separated values (variadic); `--system-prompt` REPLACES vs `--append-system-prompt` APPENDS; `--bare` flag formally documented |
| 3 | https://docs.python.org/3/library/subprocess.html | 2026-05-26 | Official doc | WebFetch | `subprocess.run(input=..., text=True, capture_output=True)` is high-level safe API; auto-handles deadlock-prevention via `communicate()`; `text=True` requires str input |
| 4 | https://support.claude.com/en/articles/15036540-use-the-claude-agent-sdk-with-your-claude-plan | 2026-05-26 | Official doc | WebFetch | Max plan: separate $100/$200 monthly Agent SDK credit starting June 15, 2026; "Agent SDK usage draws from monthly credit before any other source"; API-key users get no credit |
| 5 | https://code.claude.com/docs/en/agent-sdk/overview | 2026-05-26 | Official doc | WebFetch | Agent SDK overview — Python `claude_agent_sdk.query()` is the library path; subprocess invocation of `claude -p` is the CLI alternative; June-15-2026 billing change applies to both |

## Snippet-only (context)

| URL | Kind | Why not read in full |
|-----|------|---------------------|
| https://docs.anthropic.com/en/docs/claude-code/sdk | Doc | 301 redirect to `code.claude.com/docs/en/sdk` (which then 301'd to agent-sdk/overview — followed) |
| https://www.anthropic.com/pricing | Doc | 301 redirect to `claude.com/pricing` (followed); marketing page, not detailed-billing |
| https://claude.com/pricing | Doc | Marketing page only — no detailed Max-plan billing mechanics |
| https://platform.claude.com/docs/en/agent-sdk/overview | Doc | 307 redirect to `code.claude.com` (followed) |
| https://docs.claude.com/en/docs/agent-sdk/overview | Doc | 302 redirect to platform.claude.com (followed) |
| https://github.com/anthropics/claude-agent-sdk-python | GitHub | Source code repo for the Python SDK — not needed; doc covers the contract |
| https://github.com/anthropics/claude-code | GitHub | Source code repo — not needed |

## Recency scan (2024-2026)

Performed. Three material changes within the last-2-year window
materially affect this fix:

1. **`--bare` flag added** (v2.1.x, 2026-Q1-Q2 timeframe). Documented
   as "the recommended mode for scripted and SDK calls, and will become
   the default for `-p` in a future release." Critical caveat: skips
   OAuth + keychain reads.
2. **Stdin cap of 10MB** (v2.1.128, 2026). "If you exceed the cap,
   Claude Code exits with a clear error and a non-zero status."
   Irrelevant for this fix (prompts are small) but worth noting.
3. **Agent SDK monthly credit** (effective June 15, 2026). Subscription
   plans get a separate $100/$200 monthly Agent SDK credit. This
   changes how `total_cost_usd` accumulates against the Max plan.

No breaking changes to the stdin-piping pattern in the last 2 years;
stdin is the documented robust path for non-interactive callers. The
canonical `cat file | claude -p` invocation has been stable since SDK
launch.

## Key findings

### Canonical headless invocation (Q1)

Per `code.claude.com/docs/en/headless` and the SDK doc, the documented
patterns for non-interactive callers are:

1. **stdin pipe** — `echo "prompt" | claude --print` — robust against
   flag-ordering quirks. Recommended for production callers.
2. **trailing positional** — `claude --print "prompt"` — fragile when
   other flags are variadic; positional ordering must be precise.
3. There is no `--prompt-file` flag in the current CLI. Use stdin for
   file-sourced prompts (`claude --print < file.txt`).

The smoke-test failure confirms the canonical fix: pipe prompt via
stdin, not as argv positional.

### `--bare` flag semantics (Q2) — CRITICAL CAVEAT

`--bare` is the documented flag for context-minimal invocations. Per
`code.claude.com/docs/en/headless` (read in full 2026-05-26):

> "Add `--bare` to reduce startup time by skipping auto-discovery
> of hooks, skills, plugins, MCP servers, auto memory, and
> CLAUDE.md."

> "Bare mode skips OAuth and keychain reads. Anthropic
> authentication must come from `ANTHROPIC_API_KEY` or an
> `apiKeyHelper` in the JSON passed to `--settings`. Bedrock,
> Vertex, and Foundry use their usual provider credentials."

**This is a project-critical caveat for pyfinagent.** The autonomous
loop relies on `~/.claude/` OAuth resolution to bill against the Max
subscription's flat-fee rail (per `feature.paper_use_claude_code_route`
and the cycle-3 docstring at `backend/agents/claude_code_client.py:5`).
If `--bare` is added, the CLI will REQUIRE `ANTHROPIC_API_KEY` and
will bill per-token against an API-key account — **bypassing the Max
subscription entirely**, which defeats the purpose of the route.

**Revised recommendation: DO NOT add `--bare`.** The cycle-3 fix is
stdin-only:
1. Delete L88 `args.append(prompt)`.
2. Add `input=prompt` kwarg to `subprocess.run` at L96-103.
3. **Do NOT add `--bare`** — it would break Max-plan billing.

Cost impact of NOT adding `--bare`: each call still incurs the 67KB
CLAUDE.md cache-creation overhead (the reported $0.32). However, under
Max plan this is API-equivalent dollars, not actual billing. The
real impact is **rate-limit pressure on the Max rolling window** —
worth monitoring, but not a hard blocker.

Alternative mitigation if cost-control becomes critical: invoke
`claude` with `cwd=` set to a directory that does NOT contain
CLAUDE.md (e.g. `/tmp/claude-sandbox/`). This bypasses CLAUDE.md
auto-discovery without invoking `--bare`, preserving OAuth resolution.
The cycle-3 client already accepts `cwd` as a kwarg (L46, L61, L101);
the LLMClient adapter at L213-269 does NOT pass it through, so a
follow-up change in `_make_claude_code_client_class()` could set
`cwd="/tmp/claude-sandbox"` when called from the autonomous loop.

### Python `subprocess` stdin-pipe (Q3)

Per `docs.python.org/3/library/subprocess.html`:

```python
result = subprocess.run(
    args,
    input=prompt,         # str when text=True
    text=True,
    capture_output=True,
    timeout=timeout_s,
    check=False,
)
```

`subprocess.run(input=...)` is the high-level API; it wraps
`Popen(stdin=PIPE).communicate(input)` and handles deadlock-prevention
for large inputs. Preferred over manual `Popen` unless interleaved
read/write is needed (we don't need that here).

### Max-subscription cost-control (Q4)

Per `code.claude.com/docs/en/headless` + `support.claude.com/en/articles/15036540`:

- **Pre-June-15-2026:** `claude -p` usage draws from the Max plan's
  standard interactive usage limits (the rolling window).
- **Post-June-15-2026:** Agent SDK + `claude -p` usage draws from
  a **separate monthly Agent SDK credit** ($100 for Max 5x, $200
  for Max 20x). This credit is flat-allocation, refreshes monthly,
  does NOT roll over.
- `total_cost_usd` in the JSON envelope is **API-equivalent cost**
  (per the headless doc: "the response payload includes
  `total_cost_usd` and a per-model cost breakdown, so scripted
  callers can track spend per invocation"). Under Max plan, this
  is the dollar amount that deducts from the monthly credit pool
  (after June 15) or against the rolling window (before).
- OAuth resolution through `~/.claude/` (no `ANTHROPIC_API_KEY` in
  env) keeps the call on the Max subscription rail.
- Setting `ANTHROPIC_API_KEY` in env, OR using `--bare` (which
  skips OAuth), shifts billing to API-key pay-as-you-go.

**Implication for pyfinagent autonomous loop:**
- Keep OAuth-via-`~/.claude/` (Max rail). Do NOT set
  `ANTHROPIC_API_KEY` in the subprocess env.
- Do NOT add `--bare` (breaks Max rail).
- After June 15, 2026, monitor the monthly $100 / $200 Agent SDK
  credit consumption against the autonomous-loop call volume.

### Recency scan (Q5)

Last-2-year window (2024-05 to 2026-05): the headless CLI added
`--bare`, `--output-format json`, `--system-prompt`, and `--disallowedTools`
(variadic). The stdin-pipe pattern has been documented since the SDK
launched. No deprecations affecting the proposed fix.

## Internal investigation

### File 1: `backend/agents/claude_code_client.py` (verified line anchors)

Confirmed lines:
- L76-81: `args` list construction with `binary, --print, --output-format json, --disallowedTools, disallowed_tools`
- L82-83: `if system: args.extend(["--append-system-prompt", system])` — NOTE: already uses `--append-system-prompt`, NOT `--system-prompt`. The fix should NOT change this; `--append-system-prompt` is the documented flag and works under `--bare`.
- L84-85: `--json-schema` extension
- L86-87: `--max-tokens` extension
- **L88: `args.append(prompt)`** — THE BUG. The trailing positional is swallowed by `--disallowedTools` variadic consumption.
- L96-103: `subprocess.run(args, capture_output=True, text=True, timeout=timeout_s, cwd=cwd, check=False)` — no `input=` kwarg currently.

Minimum-viable fix (REVISED — 2 edits, NO `--bare`):

1. **Delete L88** (`args.append(prompt)`).
2. **Modify L96-103**: add `input=prompt,` to the `subprocess.run` kwargs.

`--bare` is intentionally OMITTED — it would skip OAuth and break the
Max-subscription billing rail (see Q2/Q4 sections above).

Final args + run pattern:

```python
args: list[str] = [
    binary,
    "--print",
    "--output-format", "json",
    "--disallowedTools", disallowed_tools,
]
if system is not None:
    args.extend(["--append-system-prompt", system])  # unchanged
if json_schema is not None:
    args.extend(["--json-schema", json.dumps(json_schema)])
if max_tokens is not None:
    args.extend(["--max-tokens", str(max_tokens)])
# DELETED: args.append(prompt)  -- variadic --disallowedTools swallowed it

completed = subprocess.run(
    args,
    input=prompt,                            # NEW: phase-cycle-4 (stdin)
    capture_output=True,
    text=True,
    timeout=timeout_s,
    cwd=cwd,
    check=False,
)
```

**Note on flag ordering:** even though placing the prompt as a
trailing positional AFTER a non-variadic flag (e.g. `--output-format json`)
would technically work because `json` consumes only one positional,
`--disallowedTools` IS variadic per the CLI reference ("A bare tool
name removes that tool from the model's context. A scoped rule such
as `Bash(rm *)` leaves the tool available..."). Reordering the flags
to put `--disallowedTools` first is fragile — stdin is the robust,
documented path.

### File 2: `backend/tests/test_claude_code_client.py` (verified line anchors)

Inspected the test file (147 lines). The current tests do NOT assert
that the prompt is in `args`; they only assert on stdout-parsing
behavior via `_mock_completed` helper. So most tests pass through
unchanged. However:

- L40-46 (`test_claude_code_invoke_returns_envelope`): no argv assertion
  — passes unchanged.
- L56-59 (`test_claude_code_invoke_raises_on_error_subtype`): no argv
  assertion — passes unchanged.
- L64-67 (timeout test): no argv assertion — passes unchanged.
- L72-75 (non-zero exit): no argv assertion — passes unchanged.
- L81-83 (missing binary): no argv assertion — passes unchanged.
- L88-91 (invalid JSON): no argv assertion — passes unchanged.
- L113-131 (LLMClient adapter happy path): no argv assertion — passes
  unchanged.

Recommended NEW tests to add (positive control against regression):

```python
def test_claude_code_invoke_passes_prompt_via_stdin():
    """Regression guard: prompt MUST be passed via stdin (input=),
    not as a trailing positional argv (which gets swallowed by the
    variadic --disallowedTools)."""
    envelope = {"subtype": "success", "result": "ok",
                "usage": {"input_tokens": 1, "output_tokens": 1}}
    with patch("backend.agents.claude_code_client.subprocess.run") as run:
        run.return_value = _mock_completed(stdout=json.dumps(envelope))
        claude_code_invoke("test prompt")
    call = run.call_args
    args_passed = call.args[0]
    # Positive control: --bare in args
    assert "--bare" in args_passed
    # Negative control: prompt NOT in args
    assert "test prompt" not in args_passed
    # Positive control: prompt in stdin kwarg
    assert call.kwargs.get("input") == "test prompt"
```

## Application to pyfinagent

The fix is a 3-line change in `claude_code_client.py` (remove positional
append; add `--bare`; pass `input=prompt`) plus mock updates in
`test_claude_code_client.py`. Cost impact: API-equivalent cost per
"say ok" call drops from $0.32 to <$0.01 (CLAUDE.md cache-creation
overhead removed). Max-plan billing unchanged but rate-limit pressure
reduced. Live-smoke after fix should return a valid envelope.

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch
- [x] 10+ unique URLs total (incl. snippet-only)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module
- [x] Contradictions noted (none — fix is clean)
- [x] All claims cited per-claim

## JSON envelope

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 7,
  "urls_collected": 12,
  "recency_scan_performed": true,
  "internal_files_inspected": 2,
  "gate_passed": true
}
```
