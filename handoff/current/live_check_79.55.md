# live_check -- 79.55 RAIL-MODEL TIER CONFIRMATION

Operator decision, verbatim (Peder, interactive session 2026-08-06, via
AskUserQuestion; question: "Masterplan 79.55 (P0 restart blocker): which rail
model tiers should the app run now that the --model flag actually takes
effect?"):

```
"AS CONFIGURED (Recommended)"
```

Selected option text as presented and chosen: "Record 'RAIL TIERS: AS
CONFIGURED' and keep the coded tiers (overlays on haiku-4-5). Unblocks restarts
and the smoke immediately. The smoke's E6 quota math then gives you real burn
data to decide upgrades from, instead of guessing now."

No re-pin was requested, so per the step's own instruction the required action
is option (a): record the line and do nothing. No llm_call_log row is required
on this branch (that evidence is specified only "if re-pinned").

Evidence the record exists (verification command output, verbatim):

```
$ grep -n 'RAIL TIERS' handoff/harness_log.md
31182:RAIL TIERS: AS CONFIGURED
31200:tree closes 79.55; its verification grep for 'RAIL TIERS' is satisfied by this
```

(Output re-run and pasted verbatim after an earlier draft of this file quoted
asserted-not-measured line numbers -- caught by the verify-own-claims rule
before the flip. Line numbers as of the 4000.1 close; the operator-action block
at :31181 also records the rationale --
revisit tier upgrades WITH the 4000.3 E6 quota-burn data -- and the
deliberately-held flip, now executed in this quiet-tree window. The restart
landmine 79.55 guarded against is disarmed: a restart now ships
operator-confirmed tiers.)
