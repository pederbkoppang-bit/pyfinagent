---
name: project-pbo-mcp-proof-83-0-3
description: Step 83.0.3 findings -- the fix already shipped (proof step, not fix step); FastMCP 3.2.4 tool-access idiom; payload isError != protocol is_error; near-identical PBO columns give a HIGH not low PBO
metadata:
  type: project
---

Step 83.0.3 (PBO false-pass on the MCP surface), researched 2026-08-07.

**The step's own premise was stale.** The code fix landed under e5bb9f25
(phase-82.27). The step name cites `risk_server.py:142-143` as "imports and
calls RAW compute_pbo" -- those lines are now *comment prose* about the old
defect; the live call is `compute_pbo_checked` at :149-150. So 83.0.3 is a
**PROOF step, not a fix step**: what's missing is the test file, the mutation
proof, the DELTA census, the gate.py disclosure and the live captures.

**Why:** the masterplan text was written 2026-08-04 against the pre-e5bb9f25
tree and was never re-measured after the commit landed the same day.

**How to apply:** before contracting ANY defect step written more than a day
before execution, re-measure the cited line numbers. A step whose premise has
already been fixed produces a diff that re-does the prior commit.

## FastMCP 3.2.4 -- how to reach an @mcp.tool function from pytest

Tools are **closures defined inside `create_*_server()`**, so they are NOT
importable. Measured idiom (already used at
`backend/tests/test_phase_82_27_pbo_sweep_producer.py:210`):

```python
tool = asyncio.run(create_risk_server().get_tool("pbo_check"))
r = tool.fn(pnl_matrix=..., S=16)     # .fn is the undecorated function
```

- `get_tool()` is a **coroutine**; returns `fastmcp.tools.function_tool.FunctionTool`.
- `mcp._tool_manager` is **None** in 3.2.4 -- that was the FastMCP 2.x idiom
  and is dead. Do not reach for it.
- Official docs (gofastmcp.com/servers/testing) document only
  `async with Client(transport=mcp)`; `.fn` / `get_tool` are undocumented
  but real. Both layers assert different things -- the `Client` layer sees a
  wrapped `CallToolResult`, not the handler's dict.

## Payload `isError` is NOT the MCP protocol `isError` (queued defect)

Measured over a real in-memory `Client`: `result.is_error` is **False** while
`structured_content` carries `"isError": true`. FastMCP sets the protocol flag
only when the handler RAISES; a returned dict key named `isError` is just
data. `risk_server.py:184` calls it the "MCP-native veto signal" -- that
comment is wrong. An agent branching on the protocol flag reads a PBO refusal
as a successful call.

**Why:** classic "guard stops one seam short" -- the JSON payload is honest,
the protocol envelope is not.

**How to apply:** whenever a tool signals refusal, assert BOTH the payload
field and `result.is_error` over a `Client` round-trip. Out of scope for
83.0.3; queue as its own step.

## Near-identical CSCV columns produce a HIGH PBO, not a low one

Counterintuitive and easy to assert backwards. Measured: 8 near-identical
columns -> PBO **0.703** (vetoes); 8 independent columns -> PBO 0.340.
Bailey/Borwein/Lopez de Prado/Zhu say so explicitly: *"it is entirely
possible that all the N strategies have high but similar Sharpe ratios. Since
none of the strategies is clearly better than the rest, PBO will be high."*
So `columns_diverse` distinguishes "genuinely overfit" from "degenerate
column set" -- it is not cosmetic.

Fixture caveat: the perturbation window for mean pairwise corr > 0.99 is
NARROW (eps <= ~0.0005 at T=600; eps=0.001 already gives 0.9896 and flips
`columns_diverse` to True). Assert the fixture precondition before asserting
the consequence, or a seed change silently inverts the test.

See [[project_pbo_single_strategy_cpcv]], [[project_psr_dsr_formulas]],
[[project_phase82_strategy_pack]].
