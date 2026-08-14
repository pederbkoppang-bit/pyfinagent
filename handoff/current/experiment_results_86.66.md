# Experiment results — step 86.66 (PARTIAL — criteria 3/4/5 blocked by the step's own no-metered-spend rule)


> **TIMESTAMP CORRECTION (2026-08-14 04:35 CEST).** Wall-clock times in this file were
> **narrated, not measured** — I read the clock once at session start and invented a
> progression from it. The real session spans **08-13 23:10 → 08-14 04:26** (~5h), not the
> 16+ hours the original times implied. Times below are now the **git commit timestamps**
> of this artifact, which are ground truth. Durations and orderings derived from the old
> figures should be disregarded; the measurements themselves are unaffected.
**Step:** 86.66 — autoresearch crashes with `AttributeError: 'str' object has no attribute 'append'`
**Date:** 2026-08-14 ~03:03 CEST (git)
**Immutable command:** `ls handoff/autoresearch/ >/dev/null && echo autoresearch-dir-present` → **present, exit 0**

> **HEADLINE: the step names ONE bug; there are TWO, and the one it names has no traceback
> anywhere.** Criterion 4 is a stop condition — *"no metered spend is incurred: if
> reproducing requires a paid API call, say so and stop rather than spending"* — and it
> binds. Metered spend is also a standing NON-NEGOTIABLE in the goal.

---

## C1 — the traceback: ONE bug has a full one, THE NAMED ONE DOES NOT

The step's title, and the only artifact for it, say `'str' object has no attribute
**'append'**`. The full traceback I found says something else:

```
Traceback (most recent call last):
  File ".../gpt_researcher/actions/agent_creator.py", line 58, in choose_agent
    agent_dict = json.loads(response)
  ...
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File ".../gpt_researcher/actions/agent_creator.py", line 80, in handle_json_error
    if agent_dict.get("server") and agent_dict.get("agent_role_prompt"):
       ^^^^^^^^^^^^^^
AttributeError: 'str' object has no attribute 'get'
```

**`'get'`, at `agent_creator.py:80`, not `'append'`.** Dated:

| | |
|---|---|
| `autoresearch-v5.log` mtime | **2026-07-24**, covering only 07-24/07-25 |
| the `'append'` ERROR file | **2026-08-13T00:01:35Z** |

**Three weeks apart. Two distinct defects.**

```
logs containing "no attribute 'append'" : 0  (of 9 autoresearch logs)
logs containing "no attribute 'get'"    : 6  (autoresearch-v5.log only)   <- POSITIVE CONTROL
ERROR files containing "Traceback"      : 0  of 63
```

The zero for `'append'` is a **measured** zero: the identical probe returns 6 for `'get'`,
so it is live.

**The offending assignment for the NAMED bug cannot be identified**, and one candidate is
positively excluded: `run_memo.py:58` `topics.append(line)` is the script's **only**
`.append`, and `topics` is declared `topics: list[str] = []` at `:53` two lines above — it
**cannot** be a `str`. So the named error originates inside `gpt-researcher 0.14.8`, which
has **84 `.append` call sites**; without a traceback none can be singled out.

**`run_memo.py:207` does call `traceback.print_exc()`** — the traceback goes to stderr. But
every `autoresearch.launchd*.log` is **0 bytes**, so on the current wiring the stderr is
discarded. That is why a one-line summary is all that survives.

## C2 — the share of historical failures: 1 of 63 by count, but it is the ONLY current one

**Classification command:** `grep -m1 '^Error:' <file> | sed 's/^Error: *//' | sed 's/:.*//'`
over all 63 `handoff/autoresearch/*ERROR*.md`.

| exception | count |
|---|---:|
| `ValueError` | 25 |
| `HTTPError` | 15 |
| `BadRequestError` | 15 |
| `ModuleNotFoundError` | 7 |
| **`AttributeError`** | **1** |

> **My first census was wrong and I caught it by inspection, not by luck.** It reported
> all 63 as bare `Error`, because `grep -oE '[A-Za-z_]*(Error|Exception)'` takes the
> **first** match and every file begins `Error: <Class>: …`. A uniform result across 63
> files is a probe smell, not a finding.

**By count the AttributeError is 1.6% — by date it is the whole live signal:**

```
2026-08-07 .. 2026-08-12   ERROR=0  success=1 each   <- six consecutive clean days
2026-08-13                 ERROR=1  (AttributeError) success=0
2026-08-14                 ERROR=0  success=1        <- IT RECOVERED
```

**The job succeeded again on 08-14.** So the named defect is, on the evidence, **transient
and non-recurring** — n=1, one failure between two clean runs. Every other class died in
May–August. *(This corrects the framing in the goal, which recorded 08-13 as the live break
without the 08-14 recovery, since that file did not exist yet.)*

## C3 / C4 / C5 — STOPPED, by the step's own criterion 4

- **C3** requires *"driving the real code path to completion"* — an autoresearch run is a
  paid Anthropic call.
- **C4** says: *"no metered spend is incurred: if reproducing requires a paid API call, say
  so and stop rather than spending."* **Saying so, and stopping.**
- **C5** mutation-tests *"the fix"*; there is no fix to mutate, because there is no
  identified defect site.

**A zero-spend path exists but does not reach the defect:** `run_memo.py` supports
`--preflight-only` (phase-62.6, *"stop BEFORE any LLM call"*). It verifies imports and the
embedding preflight — it cannot reach `choose_agent`, which is where both errors live.

## The cheap fix this analysis actually justifies (NOT done — no criterion owns it)

The diagnosis cost here was caused by **discarded stderr**, not by the bug: every
`autoresearch.launchd*.log` is 0 bytes while `run_memo.py:207` faithfully prints the
traceback. Writing the formatted traceback **into the ERROR memo** at `:224` would make the
next failure diagnosable for free. That is a one-function change to a non-trade-path
script, it incurs no spend, and **no criterion of this step authorises it**, so it is
queued rather than slipped in.

## Scope honesty

- **No code changed. No metered spend. No cycle run.**
- **The named bug is NOT diagnosed** — only bounded: not in our script, somewhere in 84
  library call sites, no traceback in existence.
- The `'get'` traceback **is** diagnosed to file and line, but it is a **different, older**
  defect the step does not name.
- **No Q/A has graded this**, and the step is not flipped.
