# live_check 68.0 -- design-pack sections + brief envelope, verbatim

Required shape (masterplan 68.0): "quoting the design doc's config-precedence and
rollback sections plus the brief's envelope; fresh Q/A verdict JSON".

## Design doc -- config precedence (opening, verbatim)

> Resolution chain (first hit wins), resolved AT CONSTRUCTOR TIME
> (execution_router.py:65-71 reads os.getenv per-construction ...):
> 1. `os.environ["EXECUTION_BACKEND"]` -- the launchd plist EnvironmentVariables
>    block (precedent: the 2026-07-08 setup-token wiring ...)
> 2. NEW `settings.execution_backend` pydantic field ... the router consults
>    settings as the second link, NOT by exporting env.
> 3. Default: `bq_sim` -- byte-identical behavior when nothing is set.

## Design doc -- rollback (verbatim)

> Set `EXECUTION_BACKEND=bq_sim` (or remove the key) in the plist + `launchctl
> kickstart -k` the backend; the next scheduled cycle runs fully on bq_sim. bq_sim
> is never removed -- it stays the compiled-in default forever. The 68.3 drill:
> flip back, one clean bq_sim cycle, flip forward; all three states evidenced.

## Brief envelope (verbatim from research_brief_68.0.md)

```
{external_sources_read_in_full: 13, snippet_only: 35, urls: 48, recency: true,
 internal_files: 19, gate_passed: true}
```

## Immutable command

```
IMMUTABLE VERIFICATION 68.0: exit=0 PASS
```

## Fresh Q/A verdict JSON

Returned by qa-68-0 2026-07-10: `verdict: PASS, ok: true, violated_criteria: [],
certified_fallback: false`, 14 checks_run -- every code anchor spot-checked against
live code, the price overturn INDEPENDENTLY corroborated (evaluator's own yfinance
fetch + the fills verbatim in backend.log:124393/124398), masterplan 68.5 verified
byte-unchanged (criteria immutability honored). Full JSON: evaluator_critique_68.0.md.
