# Operator asks -- 2026-08-11 (session `pyfinagent-51`)

> ## ✅ ANSWERED 2026-08-14 -- asks `06-2` / `51-4` (and `#20`) are CLOSED
>
> The operator revoked the credential at `claude.ai/settings/claude-code` and
> re-authenticated. **The revocation was then CONFIRMED BY DIRECT PROBE, not taken on
> report** -- the leaked value, including the 108-char reconstruction rejoined across
> the newline, returns **401** while the operator's live credential returns **400** on
> the same endpoint in the same minute. Evidence:
> `handoff/current/INCIDENT_2026-08-14_credential_exposure.md` §9.
>
> **`06-2`, `51-4` and `#20` were always ONE credential and are now ONE closure.** Per
> the disambiguation below, `51-4` was never a separate decision.
>
> **⚠ The rotation instruction recorded across these files was WRONG** and is corrected
> in the incident record: `claude setup-token` is **mint-only** and `claude /login`
> refreshes a *different* credential -- neither revokes. Revocation is the
> `claude.ai/settings/claude-code` web UI only; there is no CLI revoke
> (anthropics/claude-code#57400, closed as not planned).

> ## READ FIRST -- THE ASK NUMBERS COLLIDE ACROSS TWO SESSIONS
>
> Two sessions worked this repo today and numbered asks independently, so a bare
> "ASK #2" is ambiguous. **The numbers below are NOT renumbered** -- they are
> already cited in commit messages, step texts and
> `handoff/current/goal_next_2026-08-12.md`, and silently renumbering them would
> break those references. Disambiguate by SESSION:
>
> | qualified id | this file | the other session's file |
> |---|---|---|
> | **51-1** | ratify 86.37's reused gate | -- |
> | **51-2** | classify the Vertex 429 | -- |
> | **51-3** | subagent token budget | -- |
> | **51-4** | credential EXPOSURE TIMING | **same issue as their `06-2`** |
>
> `pyfinagent-06` numbers its own asks **06-2** (credential rotation, in
> `experiment_results_86.7.md`), **06-5** (86.5's frozen-red immutable command),
> **06-6** (qa-write-guard fail-closed direction) and **06-7** (a live Slack bot
> token inlined in the crontab, `ask_07_slack_token_in_crontab.md`). Their file is
> `operator_asks_2026-08-11_pyfinagent-06.md`.
>
> **`51-4` IS NOT A SEPARATE ASK FROM `06-2`.** They are the same credential.
> `06-2` asks whether to rotate; `51-4` supplies the exposure timing that bears on
> that decision -- namely that the last affected file reached `origin/main`
> **today at 06:42Z via my own step-closure commit**, not on 2026-08-10 as the file
> dates suggest. Answer `06-2`; read `51-4` as evidence for it, not as a second
> decision.
>
> ### The collision is NARROWER than this header first said
>
> Corrected 2026-08-11 after `pyfinagent-06` narrowed their own claim and I
> re-measured. **Only `#2` has two cross-session referents.** `#5`, `#6` and `#7`
> have one definition each and do NOT collide -- my first version implied a wider
> clash than exists, and a wider-than-true claim in a disambiguation note is
> self-defeating.
>
> My scan (heading-anchored, over `handoff/current/*.md`) found `#1` and `#2` with
> two headings each -- **but in both cases both headings are MINE**, the same ask
> restated in its step's `experiment_results` and in this file. That is duplication,
> not ambiguity.
>
> **STATED LIMIT OF THAT SCAN:** it matches only heading lines, so it CANNOT see
> `06-2`, which is defined in prose inside `experiment_results_86.7.md`
> ("Rotation is **operator ASK #2**"). It also returns zero for `#5` and `#6`. So
> my scan neither confirms nor refutes the peer's count -- I am taking their
> narrower claim on their measurement, and recording that mine could not check it
> rather than implying it did.
>
> **Separately, `06-7` is a DIFFERENT credential** (a Slack bot token in the
> crontab, reportedly in zero git-tracked files). Do not conflate it with the
> `away_ops` one.


Three asks outstanding. None blocks work already committed; all three are
decisions I cannot take under the standing constraints.

---

## ASK #1 -- ratify phase-86.37's REUSED research gate, or direct a fresh one

**ANSWERED 2026-08-17 (attended session, AskUserQuestion): "Ratify the reuse
(Recommended)" -- the disclosed, twice-re-verified 86.31 brief stands as
86.37's gate. Recorded by Main; the ruling is quoted in
experiment_results_86.37.md cycle 5 and evaluator_critique_86.37.md.**

86.37 fixed the researcher rail. Its own research gate was **REUSED, not re-run**
-- and the awkwardness is structural: *the rail being fixed is the rail that runs
the gate, and it had just dropped.* Re-running the gate on the rail under repair
would have been evidence of nothing until the fix was in; not re-running it
leaves the step's gate older than its code.

**The step cannot close without a ruling.** Either:
- **(a) RATIFY the reuse** -- the artifact exists for the step id, the contract
  cites it, and its claims were re-verified against source; or
- **(b) DIRECT A FRESH GATE** now that the rail is repaired, accepting ~180k
  tokens for it.

I have not chosen, because choosing would be me ruling on the adequacy of my own
step's gate.

---

## ASK #2 -- classify the Vertex 429, or accept lite-on-quota-exhaustion

Full detail in `handoff/current/experiment_results_86.38.md` section 8.

Short version: the 429 body is complete and carries **no discriminator** by
design; classification requires reading
`serviceruntime.googleapis.com/quota/rate/net_usage` in the GCP console, which is
outside this step and adjacent to spend decisions.

**Recommended: option A (read the metric, free, ~5 min), then B (accept the
fallback as designed).** NOT option C (Provisioned Throughput / paid tier) --
it is metered spend, and the per-cycle evidence shows degradation is not causing
the trade drought, so it would be money spent on the wrong subsystem.

---

## ASK #3 -- the subagent token budget, and whether today's rate is acceptable

**I have no read on the weekly Max headroom and cannot get one.** What I can
measure, this session only:

| | |
|---|---|
| Workflow runs launched | **13** (all terminal) |
| returned a verdict/envelope | 8 |
| **dropped, returning nothing** | **5 (38.5%)** |
| tokens in dropped runs | **~887k** |
| cumulative subagent tokens | **~3.1M** |

*(These figures were re-derived from the per-run `journal.jsonl` files. Two
earlier drafts of this table were wrong -- "5 of 11 / 45%" quoted from memory,
then "4 of 12 / 36%" from a tally whose age heuristic mis-classified a run that
had just dropped as still running. The figures above enumerate every run by
whether its journal contains a `result` line, which is the only reliable test.)*

**One step alone accounts for four of the thirteen**: 86.38 took 4 spawns and
~702k tokens for ONE completed verdict, and was parked rather than given a fifth.

The peer session reports >8.7M for 2026-08-10 alone. The standing rule is that
**50% of the weekly Max allowance is a hard ceiling**, past which usage moves to
metered credits and breaks the `$0 metered` constraint.

**What I want:** either a headroom figure I can budget against, or an instruction
to cap spawns per day. I have been self-limiting -- parking steps at the
escalation boundary rather than spending another ~180k on a likely-FAIL, and
declining to start P3 steps that would need a full ~400k cycle -- but that is my
judgement substituting for a number.

**Worth noting on the other side:** every dropped run's write-first record was
recovered, and two of them carried findings I acted on. The drops were not
entirely wasted, but ~700k for partial records is a poor rate.


---

## ASK #4 -- CREDENTIAL EXPOSURE: one affected file reached `origin/main` via MY push today

**Raised by the peer session as their ASK #2; this entry adds a fact about
exposure timing that they could not have known, and it is mine.**

The peer reports a live 92-char credential (`sha256[:16] 32fd305146379e49`) in
five TRACKED `handoff/away_ops/session_*.json` files spanning
2026-08-08T20:00Z..2026-08-10T20:00Z, with 2026-08-11T05:30Z clean. **I have not
opened, read, printed or copied any of those files** -- everything below is from
commit metadata only, and I am deliberately not verifying the credential's
presence myself because doing so adds handling without adding information.

**What I established, by `git log --diff-filter=A` on paths only:**

```
session_am_20260808T053009Z.json   first added 4c17f06a  2026-08-08 (earlier session)
session_pm_20260808T200008Z.json   first added 8aa3f52e  2026-08-08 (earlier session)
session_am_20260809T053008Z.json   first added 5d0e462c  2026-08-09 (earlier session)
session_pm_20260809T200008Z.json   first added 6763f10f  2026-08-09 (earlier session)
session_am_20260810T053009Z.json   first added cad38647  2026-08-10 (earlier session)
session_pm_20260810T200010Z.json   first added 630fa95b  2026-08-11 08:42  <-- MINE
session_am_20260811T053009Z.json   first added 630fa95b  2026-08-11 08:42  (the CLEAN one)
```

**`630fa95b` is my phase-86.25 step-closure commit, and it is on `origin/main`.**
The last file in the affected window therefore reached the remote **today, at
08:42 local, through my push** -- not on 2026-08-10 when it was written.

**How it happened, and why it is a repeat rather than a novelty.** The
`auto-commit-and-push` hook runs `git add -A` on a masterplan status flip, so a
step closure sweeps every dirty path in the tree under that step's name. I have a
standing note about exactly this (`audit-the-commit-not-your-diff`) and I ran
`git add -An` before my *manual* commits all day -- but the automated closure
path does not consult me, and I did not check what it had swept afterwards.
**The discipline I applied to my own commits did not extend to the hook's.**

**What I did NOT do, deliberately:** no history rewrite, no file deletion, no
`git rm`. Rewriting published history is operator-gated and the peer has already
put rotation in front of you.

**What this changes for the rotation decision:** if the timeline mattered to
whether rotation is needed, the answer is that the exposure window on the remote
is newer than the file dates suggest. Treat the credential as having been on
`origin/main` since **2026-08-11T06:42Z**, not since 2026-08-10.
