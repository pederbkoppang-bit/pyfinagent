# Evaluator Critique -- Cycle 4: claude_code_invoke stdin-pipe bugfix (2026-05-26)

Single-agent Q/A pass. Bugfix on cycle-3 ship. No SSOT change. No
masterplan flip. NO citation floor (bugfix, not trading-policy change).
Prior critique (cycle 3 routing PASS) is overwritten in full; this is
the first cycle-4 Q/A spawn (no verdict-shopping).

## Harness-compliance audit (5 items)

| # | Item | Evidence | Result |
|---|------|----------|--------|
| 1 | Researcher spawn | `handoff/current/research_brief_phase_claude_code_stdin_fix.md` exists (14149 B, mtime 23:25). `contract.md` cites `Researcher ab1987d4ec80af4dd, tier=simple, 5 sources read in full, 7 snippet-only, 12 URLs, recency scan performed, internal_files_inspected=2, gate_passed=true` (line 12). | PASS |
| 2 | Contract pre-commit | `contract.md` mtime 1779830788 PRECEDES `claude_code_client.py` 1779830808 and `test_claude_code_client.py` 1779830826. SIXTH-occurrence preamble present (`contract.md:8` -- "SIXTH occurrence today"). | PASS |
| 3 | experiment_results.md | Present (3758 B, mtime 23:29). Documents 1 modified file, 1 new test, 1 live-evidence artifact. Verbatim verification command output included (`experiment_results.md:26-67`). | PASS |
| 4 | harness_log absence | `grep "Cycle 4 -- 2026-05-26" handoff/harness_log.md` returns exit=1 (empty). Append is correctly held until after Q/A PASS, per the "log-LAST" rule. | PASS |
| 5 | No verdict-shopping | Prior critique was cycle-3 Claude Code routing PASS. This is the first cycle-4 Q/A spawn. OVERWRITE is correct. Per `code-review-trading-domain` simultaneous-presentation rule: evidence (the diff) IS new, so verdict reflects the fix, not unchanged-evidence shopping. | PASS |

## Deterministic checks

```
$ pytest backend/tests/test_claude_code_client.py -v 2>&1 | tail -5
backend/tests/test_claude_code_client.py::test_claude_code_client_class_returns_empty_on_error PASSED [100%]
============================== 12 passed in 0.21s ==============================
```

```
$ pytest backend/tests/ -k "llm_client or autonomous_loop or claude_code" 2>&1 | tail -3
================ 34 passed, 597 deselected, 1 warning in 2.41s =================
```

```
$ python -c "import ast; ast.parse(open('backend/agents/claude_code_client.py').read())" && echo client_syntax_ok
client_syntax_ok
$ python -c "import ast; ast.parse(open('backend/tests/test_claude_code_client.py').read())" && echo test_syntax_ok
test_syntax_ok
```

```
$ grep -c "args.append(prompt)" backend/agents/claude_code_client.py
0
$ grep -c "input=prompt" backend/agents/claude_code_client.py
1
$ grep -c -- "--bare" backend/agents/claude_code_client.py
2     # see below: these are in COMMENTS (lines 83-84), not in args[]
$ grep -c "input=prompt" backend/tests/test_claude_code_client.py
0     # see below: test asserts via kwargs.get("input")=="analyze TSLA" -- equivalent guard
$ grep -c -- "--bare" backend/tests/test_claude_code_client.py
3     # 1 comment + 2 assertion lines that PROHIBIT --bare in cmd_args
```

```
$ test -f handoff/current/live_check_cycle_4_stdin_fix.md && echo present
present
$ grep -c "SUBTYPE: success" handoff/current/live_check_cycle_4_stdin_fix.md
1
$ grep -c "RESULT: ok" handoff/current/live_check_cycle_4_stdin_fix.md
1
```

```
$ git diff --stat HEAD -- frontend/
(empty)
$ git diff HEAD -- frontend/package.json
(empty)
```

**Resolution of nominal grep mismatches** (the literal-grep heuristics
in the prompt overshoot the actual intent of the contract):

- `grep -c "--bare" claude_code_client.py = 2`: both hits are in the
  defensive negative-instruction comment block at
  `claude_code_client.py:83-84` ("Do NOT add `--bare` ...
  --bare rejects OAuth + keychain reads"). No `--bare` appears in
  the built `args[]` list at `claude_code_client.py:86-97`. The
  intent of contract item 7 is "no `--bare` in argv"; the comment
  documenting WHY enforces the rule. Verified by reading `args[]`
  directly.
- `grep -c "input=prompt" test_claude_code_client.py = 0`: the
  regression test on `test_claude_code_client.py:48-75` asserts the
  stdin pattern via `kwargs.get("input") == "analyze TSLA"`
  (line 66) and `"analyze TSLA" not in cmd_args` (line 69) and
  `"--bare" not in cmd_args` (line 73). The kwargs-based assertion
  is the canonical mock-guard pattern; matching the literal source
  string `input=prompt` is not necessary. Intent of contract item 8
  ("the new test passes a stdin-input assertion") is satisfied --
  the assertion reads the recorded kwargs dict.

## LLM judgment (A-K)

| Item | Check | Evidence | Result |
|------|-------|----------|--------|
| A | Stdin pattern correct | `claude_code_client.py:105-113`: `subprocess.run(args, input=prompt, capture_output=True, text=True, timeout=timeout_s, cwd=cwd, check=False)`. `args.append(prompt)` is removed (grep=0). | PASS |
| B | No --bare in args | `args[]` built at `claude_code_client.py:86-97` contains only `binary, --print, --output-format, json, --disallowedTools, <list>, [--append-system-prompt, system], [--json-schema, <schema>], [--max-tokens, N]`. No `--bare`. | PASS |
| C | Regression test guards both invariants | `test_claude_code_client.py:48-75` (`test_claude_code_invoke_passes_prompt_via_stdin_not_argv`) asserts: (i) `kwargs["input"] == "analyze TSLA"` (stdin); (ii) `"analyze TSLA" not in cmd_args` (no argv leak); (iii) `"--bare" not in cmd_args` (no auth break). Three invariants, one test. | PASS |
| D | Live smoke evidence real | `live_check_cycle_4_stdin_fix.md:24-29`: `SUBTYPE: success`, `RESULT: ok`, `OUTPUT_TOKENS: 6`, `DURATION_MS: 30013`. Extended block at lines 32-44 includes `session_id: dc9a1440-9a15-4a10-a608-a549daa79b53`, `usage.cache_read_input_tokens: 8682`, `usage.cache_creation_input_tokens: 25727`. These values are mutually consistent (output_tokens matches RESULT length, cache_read>0 implies ~/.claude session reuse, duration 30s matches an Opus-Anthropic round-trip) and could not be plausibly synthesized. | PASS |
| E | All 12 unit tests pass | `12 passed in 0.21s` -- cycle-3's 11 tests plus the new `test_claude_code_invoke_passes_prompt_via_stdin_not_argv`. | PASS |
| F | Neighboring suites pass | `pytest -k "llm_client or autonomous_loop or claude_code"` -> `34 passed`. No regressions. | PASS |
| G | No --bare in modified files | Verified above (B). Comment-only references at `claude_code_client.py:83-84` are defensive negative-instruction; test references at `test_claude_code_client.py:72-74` are the prohibition assertion. | PASS |
| H | Contract.md cycle-4 content | `contract.md` cites `Researcher ab1987d4ec80af4dd` (line 12, grep=1); SIXTH-occurrence preamble at line 8 ("SIXTH occurrence today: ... 19:56, 20:36, 20:47, 22:47, 21:02, and likely one more"). Note: prompt mentioned both FIFTH and SIXTH; contract explicitly says SIXTH (more recent, matches the prompt's authoritative "SIXTH occurrence" claim). | PASS |
| I | Zero frontend changes | `git diff --stat HEAD -- frontend/` empty. No files under `frontend/` in modified-set. | PASS |
| J | Zero new npm deps | `git diff HEAD -- frontend/package.json` empty. No `frontend/package-lock.json` change. | PASS |
| K | Zero emojis introduced | `grep -P '[^\x00-\x7F]' <all 5 cycle-4 files>` returns 0 lines. ASCII-only across the diff. | PASS |

## Code-review heuristic dispatch (5 dimensions)

| Dimension | Findings | Severity |
|-----------|----------|----------|
| 1. Security audit | The fix MOVES untrusted prompt from argv to stdin -- an IMPROVEMENT against command-injection surface (argv concat was never an injection risk here because `args` is a list with `shell=False` (implicit), but stdin is strictly safer). No new secrets, no new write capabilities, no new tool grants. `--disallowedTools` correctly blocks all side-effect tools. | NONE |
| 2. Trading-domain correctness | Diff is in the LLM-routing layer (`backend/agents/claude_code_client.py`), NOT in `paper_trader.py`, `kill_switch.py`, `risk_engine.py`, `perf_metrics.py`, or `backtest_engine.py`. No kill-switch / stop-loss / position-sizing surface touched. No crypto re-enablement. No BQ schema migration. | NONE |
| 3. Code quality | No new `print()`, no new broad `except`, no new global state, no Unicode in logger calls (`claude_code_client.py:99-102, 114-118` all ASCII). Type hints present on the public function. | NONE |
| 4. Anti-rubber-stamp on financial logic | Diff touches the LLM-rail abstraction, NOT financial-formula code. The "behavioral test required" guard fires only for `perf_metrics.py / risk_engine.py / backtest_engine.py / backtest_trader.py`. The new test IS a behavioral test of the new code path (exercises mock kwargs, asserts recorded invocation shape). No tautological assertion. | NONE |
| 5. LLM-evaluator anti-patterns | Q/A self-audit: this verdict is based on (i) direct reads of the modified Python sources, (ii) the pytest tail showing `12 passed`, (iii) the verbatim live-smoke envelope, (iv) the file mtimes, (v) the contract cross-reference. file:line citations are present in every A-K item. No sycophancy (prior verdict was PASS on different scope; no flip-under-rebuttal). No second-opinion-shopping (first cycle-4 spawn). No criteria-erosion. No 3rd-CONDITIONAL escalation needed (cycle 4 is the first cycle on this bugfix scope). | NONE |

`checks_run` appended: `code_review_heuristics`. No findings to
record in `violated_criteria`.

## Final Verdict

**PASS**

## Violated criteria

None.

## Summary (200 words)

Cycle 4 is a clean bugfix on cycle 3's Claude Code routing layer.
The empirical bug was that `--disallowedTools <tools...>` is variadic
per the CLI parser, so the trailing positional prompt was getting
swallowed by the tool-list at runtime. Cycle 3's unit tests mocked
`subprocess.run` and never exercised the real parser, so the bug
shipped. Cycle 4 routes the prompt via `subprocess.run(input=prompt,
...)` -- the canonical headless-SDK pattern -- and adds a regression
test that asserts three invariants in one mock-inspection: (i) prompt
in `kwargs["input"]`, (ii) prompt NOT in `cmd_args`, (iii) `--bare`
NOT in `cmd_args`. The `--bare` prohibition is researcher
`ab1987d4ec80af4dd`'s Section-2 finding (it rejects OAuth and forces
ANTHROPIC_API_KEY billing, which would break the Max rail).
Live-smoke evidence is verbatim Python-side: `subtype=success`,
`result=ok`, output 6 tokens, 30s duration, session_id present, cache
tokens consistent with real Anthropic backend. 12 of 12 unit tests
pass, 34 of 34 neighboring tests pass. Zero frontend / npm / emoji
delta. Contract cites researcher + SIXTH-collision preamble. Five
files written before harness_log append (log-last preserved). All
five harness-audit items PASS; all A-K judgment items PASS. Verdict
PASS. Main may now append `handoff/harness_log.md` for Cycle 4 and
hand the routing rail back to the operator for the flag flip that
unblocks step 27.6.

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "Cycle 4 stdin-pipe bugfix is correct. 12/12 unit tests pass (was 11 in cycle 3; +1 stdin-guard regression test). 34/34 neighboring suite tests pass. Verbatim live-smoke envelope shows subtype=success, result=ok, output_tokens=6, duration_ms=30013, session_id present. --bare correctly absent from args (researcher Section 2). All 5 harness-audit items PASS. All A-K LLM-judgment items PASS. No code-review heuristic findings across all 5 dimensions.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": ["syntax", "verification_command", "pytest_dedicated", "pytest_neighboring", "grep_invariants", "live_evidence_artifact", "git_diff_frontend", "code_review_heuristics", "evaluator_critique"]
}
```
