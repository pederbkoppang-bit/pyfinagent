---
name: enum-membership-guard-passes-every-wrong-value
description: A guard asserting `x in {A,B,C}` on an enum-ish field cannot fail on a wrong-but-valid value; the author's matrix mutates it to "" (the one value outside the set) and scores a KILL. Found on 86.108's rail stamp, where the field was also a GLOBAL FLAG posing as a per-event attribution.
metadata:
  type: feedback
---

When a new record carries an enum-ish field (`rail`, `kind`, `status`, `source`),
check what the guard actually asserts. If it is **set membership**
(`assert rec["rail"] in {"claude_code","gemini_or_direct","unknown"}`), then every
*wrong* answer is still a *member* — the guard can only catch ABSENCE, never
MIS-ATTRIBUTION. The author's own mutation cell will look convincing precisely
because it mutates the field to `""` or `None`, i.e. the one value outside the
vocabulary, which is the only value that assertion can catch.

**Why:** measured on step 86.108 (2026-08-17). Control green (20 passed). I ran an
in-memory mutant identical to `current_rail()` except the attribution was
INVERTED, with the except-path behaviour preserved so the "settings explode ->
unknown" test was unaffected: **20 passed, rc=0, SURVIVED.** Author cell M12
mutated the same field to `""` and scored KILLED. Both facts are true; only one
is about correctness.

**The second half, which is the more dangerous shape:** the field was populated by
`current_rail()`, a **zero-argument** function reading one global boolean
(`paper_use_claude_code_route`), while the transport it claimed to name is chosen
per call by `make_client` on `model_name.startswith("claude-") AND the flag`. So a
Gemini-served event was stamped `rail=claude_code`. **The signature alone proves
it**: a function with no model/client parameter cannot report a per-call
transport, and `record_parse_failure(agent, kind, *, site, detail, ticker)` never
receives one either. `inspect.signature` is a 5-second structural proof; reach for
it before reading any prose. The irony worth remembering: the step's whole point
was "attribution measured, not inherited", and its prospective fix inherited the
attribution from a flag.

**How to apply:**
1. For every new enum-ish field, name the WRONG-BUT-VALID value and mutate to it,
   not to empty. Use `pytest.main(argv, plugins=[Plug()])` to patch in memory —
   no disk write, so the write-guard is respected.
2. Preserve the mutant's unrelated behaviour. My first, cruder mutant replaced the
   whole function and went red via an exception-path test — a mis-attributed kill
   by my own probe. Redo it precisely before believing either colour.
3. Then ask whether the value is DERIVED FROM THE EVENT or READ FROM A GLOBAL.
   Grep the real chooser (`make_client`, the router, the dispatcher) for the
   condition, and check whether the recorder can even see the inputs that
   condition reads.
4. Cross-check with data: the flag was ON while live BigQuery showed 823
   gemini-provider calls against 40 claude-code-tagged ones in the same window —
   the step's own table already contained the refutation of its own stamp.

Related: [[driven-guard-asserts-the-key-not-the-value]],
[[mutate-the-flag-read-not-just-the-guard]],
[[check-the-attribution-not-just-the-count]],
[[a-control-built-from-your-own-pattern-tests-nothing]].
