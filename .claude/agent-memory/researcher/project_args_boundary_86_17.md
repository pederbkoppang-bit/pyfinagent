---
name: args-boundary-86-17
description: Workflow-script args parsing (86.17) -- the empty catch is LOAD-BEARING for the checker's own import; double-encoded JSON defeats try/catch entirely; a THIRD in-repo variant already fails loud; EviBound's 0% needs BOTH gates, not just the post-hoc one
metadata:
  type: project
---

Researched 2026-08-09 for step 86.17 (input-boundary integrity of the Layer-3 Workflow gates).
Four findings that are NOT derivable by reading the code, and that a future fix will get wrong
without them.

**1. The empty `catch (_e) { a = {} }` is load-bearing for the CHECKER's own import path.**
`scripts/qa/verify_research_gate_workflow.mjs` loads `research-gate.js` by slicing everything
before `phase('Research')`, appending an export line, and dynamic-`import()`ing it. In that temp
module `args` is an **unbound identifier**. `typeof args === 'string'` is safe, but the
`else if (args && ...)` branch dereferences it and throws `ReferenceError: args is not defined`
-- swallowed by the catch. **Measured:** delete the catch naively and all 40 green checks die at
module load. Any fix must discriminate with `typeof args === 'undefined'`, never a bare
`args === undefined`. Same reason the checker cannot vary a module-scope free variable across
input shapes: ESM caches by URL, so each shape needs a fresh temp file -- extract a pure
`parseArgs()` instead.

**Why:** the fix looks like "just remove the catch and throw", and that specific move is the one
that breaks the only existing test coverage of the file.

**How to apply:** before touching any Workflow args block, run the checker first and check
whether the block executes inside the imported slice.

**2. Double-encoded JSON defeats the try/catch entirely -- there is no error to catch.**
`JSON.parse('"{\\"step_id\\":...}"')` **succeeds** and returns a *string*. `a` is then a string
and `a.step_id` is `undefined`. Hardening only the catch does not cover it; the **post-parse
type** must be re-asserted (`typeof x === 'object' && x !== null && !Array.isArray(x)`).
Likewise `typeof [] === 'object'`, so an array passes the guard -- the same file guards exactly
this twice inside `enforceGate` with an explicit comment, but not in the args block.

**3. A THIRD in-repo args variant already fails loud, and nobody has flagged it as broken.**
`.claude/workflows/harness-self-audit.js:23` has **no try/catch** -- it throws an uncaught
`SyntaxError` on malformed JSON, while staying safe on the unbound case because both branches
are `typeof`-guarded and short-circuit. So the repo already demonstrates that fail-loud is
operationally tolerable on this runtime. Reframes any such fix from "introduce a risky throw" to
"converge three divergent idioms on the one that already behaves correctly".

**4. CORRECTION to the citation in `research-gate.js:20-24`.** It says EviBound false-completion
falls "to 0% ONLY with a post-hoc gate that queries the artifact store." The paper
(arXiv 2511.05524, verified at source) measures the **post-hoc Verification Gate alone at 25%**;
0% requires it *plus* a **pre-execution Approval Gate** that rejects underspecified contracts and
placeholder run_ids. That pre-execution leg is the args-validation analogue and the comment omits
it. Prompt-level self-reflection alone = **100%** false completion (8/8) -- which is the exact
mechanism `qa-verdict.js:25-29` relies on when it says "the prompt tells the agent to
self-recover".

Related: [[guard-from-instance-not-class]], [[structured-output-schema-floors-not-enforceable]].
