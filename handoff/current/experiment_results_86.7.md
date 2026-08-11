# Experiment results -- step 86.7

**Step**: `86.7` (phase-86, **P1**) | **Phase**: GENERATE (**PARTIAL**)
**Date**: 2026-08-11 | **Driver**: Main (`pyfinagent-06`)

**STATUS: PARTIAL. Three of six criteria require an action reserved for the
operator.** No verdict is claimed. Details in §5.

**§3 CONTAINS A CORRECTION OF MY OWN OVERSTATED FINDING** -- I claimed nothing
alerts on a dead rail; a breaker does. Read §3 before §3b.

## 1. The immutable command is GREEN and it does NOT satisfy criterion 1

```
$ bash -c 'source .venv/bin/activate && python -c "...pop(CLAUDE_CODE_OAUTH_TOKEN)...
           ClaudeCodeClient(claude-opus-4-8).generate_content(Reply with exactly: OK)"'
LLMResponse(text='OK', ..., degraded=False)                                exit=0
```

**Recording this as a criterion-1 pass would be false**, for two reasons the
criterion itself anticipates:

1. **My shell is ATTENDED** — it runs inside the operator's logged-in GUI
   session. Criterion 1 says in terms: *"a claim that 'the keychain works' based
   only on the attended backend is rejected."*
2. **The `pop()` was a NO-OP.** `CLAUDE_CODE_OAUTH_TOKEN` was **not in the
   environment to begin with** (measured). So the command proves the client works
   without that env var — it does **not** isolate the keychain as the credential
   source, and it does not exercise a launchd context at all.

## 2. THE 08-08 OUTAGE SIGNATURE, REPRODUCED

Re-running the same probe under `env -i` (only `HOME` and `PATH` preserved):

```
claude_code_invoke: non-zero exit code=1
stdout={"is_error":true,"duration_api_ms":0,"num_turns":1,...,"total_cost_usd":0,
        "usage":{"input_tokens":0,...,"output_tokens":0}}
ClaudeCodeClient: generate_content failed (...); returning empty LLMResponse
  text=''   degraded=False
```

`is_error: true` with **`duration_api_ms: 0`** is precisely the signature of the
2026-08-08 outage in which **20 of 20 rail calls failed**. So the rail's success
is **environment-sensitive**, and a minimal environment reproduces the historical
failure exactly.

**I am NOT concluding which variable matters.** `env -i` strips a great deal, and
the credential store is reached through the Security framework, which depends on
session context as well as environment. The defensible claim is: *the rail
succeeds in my shell and fails under a minimal environment, with the 08-08
signature.* Naming a cause would be the generalise-from-one-instance error I made
on the drought this morning and had to retract.

## 3. CORRECTED -- I OVERSTATED THIS. THERE **IS** AN ALERTING PATH.

**The section below originally concluded that nothing notices a dead rail and
that this is "how the rail ran dark for a week". That conclusion is WRONG and I
am correcting it rather than editing it away.**

What I missed sits three lines below the code I was reading:
`_rail_guard_record_failure()` (`claude_code_client.py:170-212`) counts
consecutive failures, trips a breaker at a threshold (default **20**, settings key
`claude_rail_breaker_threshold`), and on the closed->open **transition** pages via
`raise_cron_alert_sync(source="claude_code_rail", severity="P1")`. It even names
the operator action and the runbook. Alert-on-transition is the correct pattern
(Fowler / PagerDuty), and the paging is fail-open so it can never break the rail.

**How I got it wrong:** I enumerated consumers of `.degraded`, found none, and
concluded nothing notices. But the detection does not go through `.degraded` at
all — it goes through the breaker. **I enumerated the wrong set**, which is the
same failure as the two sort-key errors earlier today: a correct query against
the wrong subject.

**WHAT SURVIVES, and it is narrower but still real:**

- `LLMResponse.degraded` **defaults to `False`**, the failure path never sets it,
  and **no consumer reads it**. So the field is dead weight that reads as a
  health signal. A caller inspecting it is misled; it should be set or removed.
- A caller of a single failed call still gets `text='', degraded=False` and
  cannot distinguish a dead rail from an empty reply. The **breaker** notices at
  20; an **individual caller** never does.
- **NOT ESTABLISHED, and I will not assert it:** whether the breaker actually
  paged during the 08-08 outage. 20/20 failures should have crossed the
  threshold of 20 exactly. Whether the page fired, and whether it was delivered,
  is a question for the away-ops alert records — not something to infer from the
  code path.

The original three-layer analysis is retained below for the audit trail.

## 3b. ORIGINAL (OVERSTATED) ANALYSIS -- retained, superseded by 3

Criterion 2 requires that *"an away session that cannot authenticate must ALERT,
not silently produce degraded analyses as the rail did for a week."* **It does
not, and here is the mechanism — three independent layers, each measured:**

| layer | measured |
|---|---|
| 1 | `LLMResponse.degraded` **defaults to `False`** (`inspect.signature`) |
| 2 | the `except ClaudeCodeError` path at `claude_code_client.py:761-769` returns an **empty** `LLMResponse` and logs `ok=False` — but **never sets `degraded`** |
| 3 | **NO consumer reads `.degraded` anywhere** in `backend/agents/` or `backend/services/` (grep returns nothing) |

So a caller receiving a totally dead rail gets `text='' , degraded=False` —
**indistinguishable from a model that simply replied with nothing.** The flag
that exists to signal "the rail is dead" is never set *and* never read.

**This is how the rail ran dark for a week without anyone noticing**, which is
the exact history criterion 2 cites. It is a finding, not a fix: correcting it
changes live analysis-path behaviour and belongs in GENERATE with its own
mutation test, not in a probe.

## 4. Criterion 5: STILL-OPEN, both -- see the contract

62.1.1 and 85.3.3 are **not** closed. The premise that the plaintext token "no
longer exists in any plist" is false: 12 plist-like files carry 3 distinct
tokens, and a 92-char value (`sha256[:16] 32fd305146379e49`) sits in 5
**git-tracked** `handoff/away_ops/session_*.json` files, 08-08T20:00Z..08-10T20:00Z,
**bounded and stopped** (08-11T05:30Z is clean). Verified by hash; no values
printed. Rotation is **operator ASK #2**.

## 5. THREE OF SIX CRITERIA NEED AN OPERATOR-RESERVED ACTION

This is the structural finding of the step and it should shape whoever picks it up:

| criterion | what it needs | why I did not do it |
|---|---|---|
| 1 | the away-session launchd context | the real entrypoint writes session state, runs `git rebase`, and can POST to Slack (`run_away_session.sh:195`); the alternative is loading a LaunchAgent, and `bootstrap`/`bootout` are reserved by away-ops rail 9. `launchctl asuser` needs root. |
| 2 | keychain unavailable | locking the login keychain risks the live rail ~8h before the 20:00 cycle. The criterion permits *"or run in a context without it"* — §2 is the start of that leg. The alerting half is NOT answered: a breaker pages at 20 consecutive failures (§3), so the open question is whether that threshold and its delivery were adequate on 08-08, which needs the away-ops alert records. |
| 6 | a plist copy actually loaded | `bootout`/`bootstrap` (rail 9); `kickstart -k` does **not** re-read `EnvironmentVariables` (measured 2026-08-09). The cell can prove the CHECK reacts, never that launchd would. |

**None of these is a reason to weaken a criterion.** They are reasons the step is
partly operator-gated by construction, and that should be said plainly rather
than worked around.

## 6. What I cannot verify

- **No Q/A has run. No verdict is claimed.**
- **Criterion 1 is NOT satisfied**, despite the immutable command being green.
  Stated here so a future reader does not mistake `exit=0` for the criterion.
- **The cause of the `env -i` failure is unidentified** (§2) — deliberately.
- **Criteria 3 and 4 are untouched** in this pass.
- The research gate's **WebSearch budget was exhausted session-wide**, so the
  external survey's currency is weaker than tier `moderate` implies.
