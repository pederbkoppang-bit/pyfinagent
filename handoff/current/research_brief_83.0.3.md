# Research Brief -- step 83.0.3 (tier=simple)

Researcher: Layer-3 researcher, accessed/measured 2026-08-07.
Status: internal half COMPLETE + measured; external half in progress.
Topic: prove the PBO false-pass guard in `risk_server.pbo_check` is
behaviourally load-bearing (test file + mutation test + raw-call-site
DELTA + gate.py bypass disclosure + 2 live MCP captures).

---

## 0. Premise check (Main's context verified independently)

Main's 2026-08-04 measurement is CONFIRMED on the working tree:

| Claim | Verified? | Anchor |
|---|---|---|
| `pbo_check` already routes through `compute_pbo_checked` | YES | `backend/agents/mcp_servers/risk_server.py:149-150` |
| returns explicit refusal (ok=False, `pbo_refused:`, isError=True) | YES | `risk_server.py:157-167`; measured payload in S2 below |
| `compute_pbo_checked` reports gate_grade/column_corr_mean/columns_diverse | YES | `backend/backtest/analytics.py:264-296` |
| `min_pbo_trials` floor bypassed when `pbo_n_trials` absent | YES | `backend/autoresearch/gate.py:44-45` (verbatim in S5) |
| test file `backend/tests/test_phase_83_0_3_pbo_false_pass.py` missing | YES | `ls` -> no such file |
| shipped under e5bb9f25 (phase-82.27) | YES | `git show --stat e5bb9f25` |

**So 83.0.3 is a PROOF step, not a fix step.** The code change landed;
what is missing is the evidence that it is load-bearing. Contract should
say this plainly -- writing it as a fix step would produce a diff that
re-does e5bb9f25.

### One correction to the step text

The step name says "MEASURED 2026-08-04: `risk_server.py:142-143` imports
and calls RAW `compute_pbo`". That is **no longer true at those lines** --
:142-143 is now *comment prose* describing the old defect, and the live
call is at :149-150 via the checked wrapper. The step text describes the
PRE-e5bb9f25 tree. Criteria are immutable and remain satisfiable; the
contract must note the line numbers moved so Q/A does not read the stale
anchors as a live defect.

---

## 1. Internal code inventory

| File | Lines | Role | Status |
|---|---|---|---|
| `backend/agents/mcp_servers/risk_server.py` | 271 | FastMCP server factory; 6 tools | FIXED at :148-185 |
| -- `create_risk_server()` | :41-266 | factory; all tools are **closures** inside it | not module-level importable |
| -- `pbo_check` | :131-185 | the subject | routes to checked wrapper :149-150 |
| -- `evaluate_candidate` | :189-262 | composite gate; calls `pbo_check` :215 | reads `ok` before `vetoed` :229 |
| `backend/backtest/analytics.py` | -- | PBO maths | -- |
| -- `compute_pbo_checked` | :208-296 | refusing wrapper; returns dict | the guard |
| -- `compute_pbo` | :276+ | RAW; `return 0.0` at N<2 or T<S*2 | the hazard |
| -- `PBO_CEILING_LIVE` / `_CANONICAL` / `PBO_MIN_TRIALS_GATE_GRADE` | :196-206 | 0.20 / 0.50 / 10 | constants |
| `backend/autoresearch/gate.py` | -- | `PromotionGate.evaluate` | bypass at :43-45 |
| `backend/tests/test_phase_82_23_pbo_in_gate.py` | 195 | 82.23 suite | GREEN |
| `backend/tests/test_phase_82_27_pbo_sweep_producer.py` | ~380 | 82.27 suite, 19 tests | GREEN |
| `backend/tests/test_phase_83_0_3_pbo_false_pass.py` | -- | **DOES NOT EXIST** | the deliverable |

Verification run (read-only, allowed):
```
$ python -m pytest backend/tests/test_phase_82_23_pbo_in_gate.py \
      backend/tests/test_phase_82_27_pbo_sweep_producer.py -q
40 passed in 7.80s
```

---

## 2. THE critical test-design fact: tools are closures, not importables

`pbo_check` is defined **inside** `create_risk_server()` and decorated
`@mcp.tool` (`risk_server.py:131`). It is therefore **not importable** as
`from ...risk_server import pbo_check`. Installed runtime measured:
`fastmcp 3.2.4`, `python 3.14.4`.

Measured resolution (this is the idiom the repo ALREADY uses at
`test_phase_82_27_pbo_sweep_producer.py:210`, so reuse it -- do not invent
a new one):

```python
tool = asyncio.run(create_risk_server().get_tool("pbo_check"))
r = tool.fn(pnl_matrix=..., S=16)
```

Measured shapes:
- `mcp.get_tool(name)` is a **coroutine** -> `asyncio.run(...)`.
- it returns `fastmcp.tools.function_tool.FunctionTool`.
- `.fn` is the **undecorated Python function** (`<class 'function'>`), so
  it returns the raw `dict`, not an MCP `CallToolResult`. This is why the
  existing tests can assert `r["ok"] is False` directly.
- `mcp._tool_manager` is **None** in 3.2.4 -- do NOT reach for it (it was
  the FastMCP 2.x idiom and is now dead). `mcp.call_tool` exists but wraps
  the result in MCP content blocks; `.fn` is the cheaper assertion surface.

No running server, no transport, no network is required.

---

## 3. Measured payloads (the two fixtures the criteria name)

### Criterion 2 fixture -- T=10, N=4, S=16 (T < S*2 = 32)

Verbatim, from `.fn(pnl_matrix=rng.normal(size=(10,4)).tolist(), S=16)`:

```json
{"ok": false, "vetoed": false, "pbo": null, "threshold": 0.5,
 "n_trials": 4, "n_obs": 10,
 "reason": "pbo_refused:T=10 < S*2=32; compute_pbo would return a false-good 0.0 that PASSES the ceiling",
 "isError": true}
```
RAW `compute_pbo` on the SAME matrix returns `0.0` -- measured. That is
the false PASS the step exists to prevent, and it makes a perfect
discriminator.

**Note for the contract:** on the refusal path the payload does NOT carry
`gate_grade` / `columns_diverse` (`risk_server.py:157-167` omits them).
So criterion 4 must be exercised on a NON-refused matrix (T >= 32).

### Criterion 4 fixture -- 8 near-identical columns

Measured sweep over the perturbation size (T=600, 8 columns,
`base + N(0, eps)`):

| eps | mean pairwise corr | min pairwise corr | `columns_diverse` | `gate_grade` |
|---|---|---|---|---|
| 0.0005 | 0.99742 | 0.99711 | **False** | False |
| 0.001 | 0.98960 | 0.98866 | True | False |
| 0.002 | 0.95981 | 0.95575 | True | False |
| independent | 0.01749 | -- | True | False |

**The fixture boundary is TIGHT** -- eps must be <= ~0.0007 or the
correlation drops under the 0.99 threshold and `columns_diverse` flips to
True. A seed change could silently flip this. Mitigation for the contract:
the test must **assert the fixture precondition first** (min pairwise corr
> 0.99) before asserting `columns_diverse is False`, so a fixture that
drifts fails as a fixture, not as a false verdict on the guard.

`gate_grade` is **False for BOTH** the near-identical and the independent
8-column matrix (N=8 < `PBO_MIN_TRIALS_GATE_GRADE`=10, `analytics.py:206`).
So `gate_grade` does NOT discriminate diversity -- the criterion-4 test
needs the independent-columns **mirror guard** (`columns_diverse is True`)
or "always False" would satisfy it.

---

## 4. Criterion 3 -- the mutation test, PROTOTYPED AND PROVEN

Mechanism: textual mutant of `risk_server.py`, `ast.parse`-validated,
`exec`'d into a fresh module, criterion-2 assertions re-applied.

Target (exists **exactly once**; counted, per the "a no-match replace
looks like success" rule):
```python
            from backend.backtest.analytics import compute_pbo_checked
            checked = compute_pbo_checked(pnl_matrix, S=S)
```
replaced by the raw-call revert:
```python
            from backend.backtest.analytics import compute_pbo
            checked = {"pbo": float(compute_pbo(pnl_matrix, S=S)),
                       "n_trials": None, "n_obs": None}
```

Measured result on the T=10/S=16 matrix:
```
MUTANT -> {"ok": true, "pbo": 0.0, "vetoed": false, "gate_grade": null,
           "columns_diverse": null, "reason": "pbo_within_bounds", "isError": false}
REAL   -> {"ok": false, "pbo": null, "reason": "pbo_refused:T=10 < S*2=32; ...", "isError": true}
MUTANT KILLED: criterion-2 assertions fail on the reverted call.
REAL passes the same assertions.
```
The mutant reproduces the **original defect exactly** (`ok:true, pbo:0.0,
vetoed:false` -- a clean PASS on an unevaluable matrix), which is the
strongest possible evidence the guard is load-bearing. This is
behavioural discrimination, not a source-text grep -- the 82.23 Q/A
explicitly rejected `inspect.getsource` token scans as
"satisfiable by a comment" (`test_phase_82_27_pbo_sweep_producer.py:16`).

Prototype script (scratchpad, not for commit):
`/private/tmp/claude-501/-Users-ford--openclaw-workspace-pyfinagent/cc942179-c9ad-44ed-a4e0-9ea34b301ce6/scratchpad/probe3.py`

---

## 5. Criterion 6 -- gate.py bypass, verbatim source

`backend/autoresearch/gate.py:43-45`, verbatim:

```python
        n_trials = trial.get("pbo_n_trials")
        if n_trials is not None:
            try:
```
with the declared intent at `gate.py:25-30`:
```python
    # A trial that does not report N at all is UNCHANGED in behaviour (see below), so
    # this is additive for every existing producer.
    min_pbo_trials: int = 10
```
and at `:41-43`:
```python
        # phase-82.23: when the producer DOES report its trial count, refuse an
        # undersized one. Absent => unchanged legacy behaviour, so no existing
        # producer starts failing on a field it never emitted.
```

Reading: the `if n_trials is not None:` guard means a trial dict that
simply **omits** `pbo_n_trials` skips the floor entirely and is evaluated
on DSR/PBO alone. This is a deliberate, documented legacy carve-out --
NOT a bug to fix in 83.0.3 (the step scopes it as a disclosure only). The
actionable consequence for 83.5: **any producer feeding the gate must
always emit `pbo_n_trials`**, otherwise the floor is inert for it.

Already proven live by the existing suite
(`test_phase_82_27_pbo_sweep_producer.py:258-262`): stripping the key
flips `promoted` from False back to True.

---

## 6. Criterion 5 -- the raw-call-site DELTA census

`grep` alone over `compute_pbo` returns 40+ lines that are overwhelmingly
**comments and docstrings**, not call sites. A raw count off that grep
would be a fabricated number. Two-command approach (both committable):

**Command 1 -- the grep of record (word-boundary, excludes the wrapper):**
```bash
grep -rnP '\bcompute_pbo\b(?!_checked)' --include='*.py' . | grep -v '/\.venv/' | sort
```
(Note: `grep -rnE` with `(?!...)` FAILS on macOS/BSD grep -- `-P` is
required. Verified this session.)

**Command 2 -- the AST census that actually answers the criterion**
(counts `Call` nodes, so comments/docstrings cannot inflate it):
```bash
python - <<'PY'
import ast, pathlib
for p in sorted(pathlib.Path('.').rglob('*.py')):
    if '.venv' in p.parts: continue
    try: t = ast.parse(p.read_text())
    except SyntaxError: continue
    for n in ast.walk(t):
        if isinstance(n, ast.Call) and getattr(n.func, 'id', None) == 'compute_pbo':
            print(f"{p}:{n.lineno}")
PY
```

Measured TRUE raw-`compute_pbo` **call sites** (AST-grade, this session,
pre-change baseline for 83.0.3):

| Site | Kind | Keep or migrate? |
|---|---|---|
| `backend/backtest/analytics.py:240` | inside `compute_pbo_checked` | **KEEP** -- this is the wrapper's own legitimate delegation |
| `scripts/harness/run_82_3_candidate_backtests.py:206` | one-off harness script | already guarded at :195/:203 |
| `backend/tests/test_phase_82_23_pbo_in_gate.py:37` | test asserting the raw 0.0 | **KEEP** -- deliberately pins the hazard |
| `tests/autoresearch/test_phase_48_2_backtest_adapter.py:106` | test | harmless |

Plus 3 raw **imports**: `run_82_3_candidate_backtests.py:129`,
`test_phase_82_23_pbo_in_gate.py:18`, `test_phase_48_2_backtest_adapter.py:15`.

**Production (non-test, non-script) raw call sites reaching `compute_pbo`
outside the wrapper: 0.** The correct DELTA to record is therefore
`before=0, after=0` on the production surface with the four sites
enumerated and classified -- NOT "we removed N". Recording it as a
before/after pair with the command that produced it is exactly what the
criterion asks for ("recorded as a measured DELTA ... rather than
asserted to be zero"). Re-run both commands before AND after the test-file
add and commit both outputs.

---

## 7. Live-check capture plan

The live_check asks for a **BEFORE** capture showing `pbo 0.0 with no
veto`. That state **no longer exists on the tree** (e5bb9f25 fixed it), so
it cannot be captured from the live MCP honestly. Two admissible routes,
both must be disclosed as such:
1. capture BEFORE from the **mutant** (Section 4) -- it is the byte-level
   revert of the one call, and its measured output is exactly
   `{"ok": true, "pbo": 0.0, "vetoed": false}`; label it "reconstructed
   from an ast-validated mutant, not from git history";
2. capture AFTER from the **real** `mcp__pyfinagent-risk__pbo_check` tool
   over the live MCP surface (`.mcp.json` pins it `alwaysLoad: true`).

Do NOT present the mutant output as a historical capture. The honest
framing is "the pre-fix behaviour, reproduced".

---

## 8. External research

### Search queries run (3-variant discipline)

| Topic | Current-year (2026) | Last-2-year (2025) | Year-less canonical |
|---|---|---|---|
| MCP/FastMCP testing | "FastMCP testing tools pytest in-memory client 2026" | "MCP server tool testing patterns 2025 conformance regression" | "FastMCP testing mcp.tool decorated function pytest" |
| Mutation testing | "mutation testing 2026 mutation score test suite effectiveness empirical" | (covered by ICST-2025 hits in variant 3) | "mutation testing equivalent mutants guard clause verification" |
| PBO/CSCV | (covered below) | "backtest overfitting PBO deflated Sharpe 2025 2026 practice critique" | "probability of backtest overfitting CSCV number of trials N correlated configurations Bailey" |

### Read in full (7; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|
| https://gofastmcp.com/servers/testing | 2026-08-07 | official doc | WebFetch | Canonical pattern is `async with Client(transport=mcp)`; requires `pytest-asyncio` with `asyncio_mode = "auto"`. **Does NOT document `.fn` or `get_tool`** -- the repo's idiom is undocumented-but-real API. |
| https://jlowin.dev/blog/stop-vibe-testing-mcp-servers | 2026-08-07 | authoritative blog (FastMCP creator) | WebFetch | "You're testing your actual server logic: No mocks or simplified protocol implementations." Advocates asserting through the client layer: the assertion "validates the *protocol response format* (TextContent object), not just the mathematical result." |
| https://agentcat.com/guides/writing-unit-tests-mcp-servers/ | 2026-08-07 | practitioner guide | WebFetch (301 from mcpcat.io) | "The best way to unit-test an MCP tool is to connect a real client to your real server over an in-memory transport." **"Your handler returns `8`, but the client doesn't hand you back `8`."** Errors: "the framework catches the exception and returns a normal result with `isError` set". |
| https://modelcontextprotocol.io/specification/2025-06-18/server/tools | 2026-08-07 | official spec | WebFetch | Two error mechanisms: **Protocol Errors** (JSON-RPC) vs **Tool Execution Errors** "Reported in tool results with `isError: true`: API failures, Invalid input data, **Business logic errors**". |
| https://codex.danielvaughan.com/2026/05/30/mcp-server-testing-frameworks-unit-integration-conformance-validation/ | 2026-08-07 | practitioner (2026-05-30, upd. 2026-07-05) | WebFetch | "The foundational layer skips transport entirely and calls handler functions directly through an in-memory client-server binding." Guard testing via `@pytest.mark.parametrize("days,should_pass", [...])` boundary pairs. |
| https://arxiv.org/html/2408.01760 | 2026-08-07 | preprint | WebFetch | "the rate of equivalent mutants in real-world development scenarios ranges from 4% to 39%"; "the presence of equivalent mutants makes it impossible to achieve a score of 100 percent." |
| https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf | 2026-08-07 | peer-reviewed (Bailey/Borwein/Lopez de Prado/Zhu) | WebFetch failed (binary) -> **pdfplumber**, 63,988 chars | See quotes below. |

### Identified but snippet-only (18; context, does NOT count)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://github.com/jlowin/fastmcp/issues/1019 | issue | duplicate of official doc content |
| https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/tests.py | code | superseded by direct runtime measurement |
| https://dev.to/klement_gunndu/your-mcp-server-has-no-tests-here-are-4-patterns-to-fix-that-2k59 | blog | community tier |
| https://dev.to/aws-heroes/testing-mcp-servers-the-five-gates-between-demo-and-production-2inf | blog | five-gates model captured via Codex article |
| https://mcpcat.io/guides/integration-tests-mcp-flows/ | guide | E2E, out of scope |
| https://testomat.io/blog/mcp-server-testing-tools/ | vendor | vendor tier |
| https://www.kaigritun.com/mcp/testing-mcp-servers | blog | community tier |
| https://realpython.com/python-mcp/ | tutorial | introductory |
| https://mcpmarket.com/tools/skills/fastmcp-python-testing | listing | listing only |
| https://github.com/YawLabs/mcp-compliance | tool | conformance, not unit-guard |
| https://github.com/r-huijts/mcp-server-tester | tool | WIP |
| https://arxiv.org/pdf/2506.11019 | paper | telemetry, off-topic |
| https://conf.researchr.org/details/icst-2025/mutation-2025-papers/1/Equivalent-Mutants-Deductive-Verification-to-the-Rescue | paper (ICST 2025) | abstract page; thesis captured via 2408.01760 |
| https://arxiv.org/pdf/2607.00511 | preprint | extension of 2408.01760 |
| https://arxiv.org/pdf/2404.09241 | preprint | manual equivalent mutants |
| https://arxiv.org/html/2607.22880 | preprint | LLM-generated suites, off-topic |
| https://link.springer.com/article/10.1007/s10664-022-10149-y | journal | paywalled |
| https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253 | paper | SSRN landing; full text obtained via davidhbailey.com mirror |
| https://www.sciencedirect.com/science/article/abs/pii/S0950705124011110 | journal | paywalled |
| https://medium.com/balaena-quant-insights/the-probability-of-backtest-overfitting-pbo-9ba0ac7fb456 | blog | secondary |

**URLs collected: 27.**

### Recency scan (2024-2026) -- PERFORMED

Searched the 2024-2026 window on all three topics. Result: **3 new
findings that COMPLEMENT (none supersede) the canonical sources.**

1. **MCP testing has consolidated on in-memory transport (2026).** The
   Codex article (2026-05-30, updated 2026-07-05) and the agentcat guide
   both now state the in-memory client-server binding as the default unit
   layer, and formalise a "Five Gates" ladder (Smoke -> Conformance ->
   Scenarios -> Load -> Pentest). 83.0.3 sits at gate 1-2 only; no change
   to the plan.
2. **Equivalent-mutant rate is materially high (2024-2026).** 4%-39%
   (arXiv:2408.01760). Relevant because criterion 3 rests on ONE mutant --
   see Pitfalls.
3. **PBO/DSR practice unchanged in the window.** The 2025-2026 material
   (CPCV comparisons, practitioner "gauntlet" pipelines) re-affirms
   Bailey et al. rather than revising it. No finding supersedes SSRN
   2326253. The repo's implementation needs no re-derivation.

### Key findings (external)

1. **The MCP spec explicitly classes a business-logic refusal as a tool
   execution error, reportable via `isError`** -- "Tool Execution Errors:
   Reported in tool results with `isError: true`: API failures, Invalid
   input data, Business logic errors" (MCP spec 2025-06-18,
   modelcontextprotocol.io, accessed 2026-08-07). A PBO refusal is
   squarely a business-logic error, so the design intent in
   `risk_server.py:184` is spec-aligned -- but see the defect in S9.
2. **Framework-level results are wrapped; the handler's return value is
   not what a client receives** -- "Your handler returns `8`, but the
   client doesn't hand you back `8`" (agentcat.com, accessed
   2026-08-07). This is why a `.fn` test and a `Client` test assert
   different things.
3. **The FastMCP author advocates protocol-layer assertions** -- the
   assertion "validates the *protocol response format* (TextContent
   object), not just the mathematical result" (jlowin.dev, accessed
   2026-08-07).
4. **A single mutant is weak evidence in general** -- equivalent mutants
   run "4% to 39%" (arXiv:2408.01760, accessed 2026-08-07). Mitigated
   here because our mutant is demonstrably NON-equivalent (measured
   behavioural divergence, S4).
5. **Bailey et al. directly validate the criterion-4 fixture.** Verbatim
   (pdfplumber extraction, accessed 2026-08-07): *"although a high PBO
   indicates overfitting in the group of N tested strategies, skillful
   strategies can still exist in these N strategies. For example, it is
   entirely possible that all the N strategies have high but similar
   Sharpe ratios. Since none of the strategies is clearly better than the
   rest, PBO will be high."* -> near-identical columns produce a
   **spuriously HIGH** PBO. Measured exactly: near-identical 8 columns
   -> PBO 0.703 (vetoes); independent 8 columns -> PBO 0.340.
   `columns_diverse` is therefore the flag that distinguishes "genuinely
   overfit" from "degenerate column set", which is precisely Sec 5.2's
   caveat.
6. **N >> 10 is the paper's own floor** -- *"if the investor is sensitive
   to values of phi < 1/10 ... the range of values that the logits can
   adopt must be greater than 10, and so N >> 10 is required."* Confirms
   `PBO_MIN_TRIALS_GATE_GRADE = 10` (`analytics.py:206`) and explains why
   `gate_grade` is False at N=8.
7. **The file-drawer warning the repo already cites is verbatim
   correct** -- *"Hiding trials will lead to an underestimation of the
   overfit, because each logit will be evaluated under a biased relative
   rank."*
8. **S=16 is the paper's convention and implies a T floor** -- *"we
   partition M across rows, into an even number S of disjoint submatrices
   ... of order (T/S x N)"*; *"if S = 16, we will form 12,780
   combinations."* The repo's `T < S*2` refusal is a defensible (stricter
   than strictly necessary) floor: at T<32 with S=16 the submatrices
   would have <2 rows.

### Consensus vs debate (external)

- **Consensus:** in-memory, in-process testing (no subprocess, no
  network) is the correct unit layer for MCP tools. All four MCP sources
  agree.
- **Debate -- and it bears on this step:** *protocol-layer* (`Client`)
  vs *direct-function* (`.fn`) assertions. jlowin, agentcat and the MCP
  spec favour going through the client so you test serialization,
  schema validation and the real error channel. The Codex 2026 article
  explicitly endorses skipping transport. **Recommendation for 83.0.3:
  do BOTH** (cheap -- measured this session) because they catch
  different failures, and because the divergence is itself a finding
  (S9).

### Pitfalls (from literature, mapped to this step)

1. **One mutant is not a mutation score.** With a 4-39% equivalent-mutant
   base rate (arXiv:2408.01760), a surviving mutant would be ambiguous.
   Ours is proven non-equivalent by measured behavioural divergence, so
   state that explicitly rather than claiming "mutation tested".
2. **Asserting the handler's dict when a client sees a wrapper.** A test
   that only uses `.fn` cannot detect a serialization or output-schema
   regression (agentcat).
3. **A near-identical fixture produces a HIGH PBO, not a low one**
   (Bailey Sec 5.1/5.2). A test author expecting "degenerate => PBO
   looks good" would write a backwards assertion. Measured: 0.703,
   `vetoed=True`.
4. **Fixture drift.** The eps window for corr>0.99 is narrow (S3).
   Assert the precondition.

---

## 9. DISCOVERED DEFECT (out of scope -- queue as its own step)

**The payload's `isError` is NOT the MCP protocol's `isError` flag.**

Measured this session over the real in-memory client:

```
protocol is_error attr: False
structured_content: {"ok": false, ..., "reason": "pbo_refused:T=10 < S*2=32; ...", "isError": true}
```

`risk_server.py:184` comments the field as the "MCP-native veto signal",
but FastMCP sets the protocol-level `isError` only when the handler
RAISES. A returned dict key named `isError` is just data. Per the MCP
spec a business-logic refusal SHOULD surface as `isError: true` at the
protocol level -- so an MCP client (or an agent) branching on the
protocol flag reads a PBO refusal as a **successful** call. The JSON
payload is honest; the protocol envelope is not.

This is the "guard stops one seam short" shape. It is **outside 83.0.3's
stated scope** (route through the wrapper / surface the fields / disclose
the gate.py loophole), so per the standing rule it must be **queued as its
own research-gated masterplan step**, not folded in here. Suggested
framing: "risk_server tools report refusals in the payload only; the MCP
protocol envelope reports success -- decide whether to raise `ToolError`
and re-verify every consumer."

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL (7: 6 via WebFetch, 1 via pdfplumber per the PDF fallback rule)
- [x] 10+ unique URLs total (27)
- [x] Recency scan (2024-2026) performed + reported (S8)
- [x] Full papers/pages read, not abstracts (Bailey read as 63,988 extracted chars)
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (risk_server, analytics, gate, both 82.x suites, census)
- [x] Contradictions / consensus noted (protocol-layer vs .fn debate, S8)
- [x] All claims cited per-claim
- [ ] Brief exceeds the `simple` 300-word target -- DELIBERATE: the caller
      specified five distinct contract-ready deliverables. Depth of
      external analysis is held at `simple`.

---

## JSON envelope

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 20,
  "urls_collected": 27,
  "recency_scan_performed": true,
  "internal_files_inspected": 9,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "83.0.3 is a PROOF step, not a fix step: e5bb9f25 already landed the routing. Measured the two fixtures live -- T=10/N=4/S=16 returns the refusal payload while raw compute_pbo returns 0.0; 8 near-identical columns (eps<=0.0005, corr_mean 0.99742) give columns_diverse=False. Prototyped and PROVED the criterion-3 mutant: reverting the one call yields {ok:true, pbo:0.0, vetoed:false} -- the original false PASS. MCP tools are closures, reachable via asyncio.run(create_risk_server().get_tool(name)).fn (idiom already at test_phase_82_27:210; _tool_manager is None in fastmcp 3.2.4). Production raw-compute_pbo call sites outside the wrapper: 0 (AST census). Bailey verbatim confirms near-identical columns give a spuriously HIGH PBO and N>>10. DISCOVERED, out of scope: payload isError=true while protocol is_error=False -- queue as its own step.",
  "brief_path": "handoff/current/research_brief_83.0.3.md",
  "gate_passed": true
}
```
