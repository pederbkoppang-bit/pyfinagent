# Operator asks -- 2026-08-11, session `pyfinagent-06`

**READ THIS FIRST: "ASK #2" IS AMBIGUOUS TODAY.** Two sessions numbered asks
independently and both reached `#2`. I am not renumbering anyone's file; I am
naming the collision so you are not asked to act on the wrong one.

| id as written | referent | raised by | file |
|---|---|---|---|
| **ASK #2** | classify the Vertex 429, or accept lite-on-quota-exhaustion | `pyfinagent-51` | `operator_asks_2026-08-11.md` §2 |
| **ASK #2** | **a credential is published on `origin/main`** | `pyfinagent-06` (me) | `experiment_results_86.7.md:119` |

Measured, not assumed: `grep` for definition-shaped `ASK #n --` lines across
`handoff/current/` returns **two** distinct referents for `#2`, and **exactly one
each** for `#5`, `#6`, `#7`. So the ambiguity is confined to `#2`. Separately,
ask numbers **1-30** are referenced across `handoff/current/` from a longer-running
series (`#23`/`#24`/`#25` are the rail/ops asks used below), which is why fresh
low numbers were a poor choice by both sessions.

**Below, every ask of mine is prefixed `06-` to make it unambiguous.** Where an ask
already has its own file, that file is the detail and this is the index.

---

## 06-#2 -- CREDENTIAL PUBLISHED ON `origin/main`. The only time-sensitive item here.

> **`06-#2` AND `51-#4` ARE ONE ASK, NOT TWO -- ANSWER THIS ONE.** The peer session
> raised the point and it is a real hazard: presented as two numbered items, you
> could reasonably answer one and believe the other still open. **Same credential.**
> `51-#4` is not a second decision; it is *evidence bearing on this one*.

A 92-char `sk-ant-*` token (`sha256[:16] = 32fd305146379e49`) sits in five TRACKED
`handoff/away_ops/session_*.json` files. `51-#4` supplies the timing fact that
matters: the last affected file reached the remote **2026-08-11 at 06:42Z, through a
step-closure push**, not on 2026-08-10 as the filenames suggest.

**I have not opened, printed or copied those files**, and I have not rewritten
history — rewriting published history is yours.

**NEEDS: rotate, or rule that rotation is unnecessary.** Everything else in this
file can wait; this is the one where waiting has a cost.

## 06-#5 -- 86.5 is structurally uncloseable without an immutability exception

`86.5`'s immutable verification command was frozen while **already red for 128
unrelated reasons**, so no amount of correct work turns it green. The step is
**PARKED** after two graded cycles.

**NEEDS: authorise an exception to repair that one step's verification command**
(criteria untouched), or direct that 86.5 stay parked. Project rule is that
verification criteria are immutable, so I will not touch it on my own judgement.

The general lesson is already recorded: run the verification command **before**
freezing it.

## 06-#6 -- choose the direction for the qa-write guard. All three options cost something.

`.claude/hooks/qa-write-guard.sh` keys on `agent_type`, which **the caller chooses**
-- so it is a convention check, not a boundary. Measured on real traffic: the hook
payload carries exactly one caller-chosen role field plus one opaque instance id;
nothing distinguishes subagent TYPE from spawn NAME. `general-purpose` already
evades the guard and has written `evaluator_critique` files.

1. **Fail-closed on unknown `agent_type`** -- strongest of the three that I can
   implement, but it will silently break new legitimate spawn shapes, and the
   researcher rail is what preserved a 76KB brief and three Q/A verdicts today.
2. **Platform `disallowedTools` on the qa agent** -- the real fix, but `qa.md`'s
   `memory: project` re-injects `Write` by design, so it fights the agent
   definition. Also an agent-definition change I am separated-of-duties from.
3. **Accept it as a convention check** -- document it, keep it fail-open, rely on
   it for accident-prevention only. That is a downgrade of a stated security
   property, which is why it is yours and not mine.

**I am not choosing.** Detail: `experiment_results_86.33.md` §6.

## 06-#7 -- a live Slack bot token is inlined in the crontab (LOCAL, not published)

Full detail: **`handoff/current/ask_07_slack_token_in_crontab.md`**. No value is
printed there or here.

`export SLACK_BOT_TOKEN='...' && script` puts the secret on the spawned shell's
**command line**, visible in `ps` to any process on this machine, **720 times a
day**. `backend/.env` already carries the same key, so the inlining looks
unnecessary -- but "looks equivalent" is not "verified equivalent" for a job running
next to a live book, and the standing goal forbids `.env` writes.

**Deliberately NOT conflated with 06-#2**: this token appears in **zero**
git-tracked files (I checked -- 35 files match `xoxb-`, but they hold 5 distinct
placeholder values of 15-33 chars, none matching this token's hash). This one is
local; 06-#2 is published.

## 06-#24 -- rail timeout 150s -> 210s. NOW RECOMMENDED, on stated-provenance grounds.

**Recommended, with the provenance said plainly rather than buried**: the decisive
figures (p90 **134s**, longest SUCCESS **145s**, against a **150s** cap -- a
censored distribution by definition) are **PRE-FIX**, from
`research_brief_85.4.md:321`. They **cannot** be re-derived from the post-fix
window: the measurement script prints `agent latency : None` for the only post-fix
cycle.

**And the post-fix datum that does exist argues against urgency**: 1 timeout in 152
calls (**0.66%**). The honest case is not "the last cycle was bad" -- it is that the
rate is **highly variable across comparable cycles** and the cap sits 5s above the
longest observed success, so on a bad night it censors work that would have finished.

> **CORRECTED before you act on this.** An earlier version of this paragraph said
> *"five other measured cycles ran 9.9%-23.4%"*. **It is FOUR** (#1 23.4%, #3 18.1%,
> #2 14.9%, #6 9.9%); two further cycles ran 0.0% but are not comparable -- 340s and
> 322s wall with 20 and 33 rail calls, against 124-177 in the others. I had adopted
> the figure from a reviewer's critique without re-deriving it, and it overstated
> prevalence in the direction supporting this very recommendation.
>
> **One more thing you should weigh, because it cuts against me**: the cycle with the
> **highest** rail-timeout rate in the whole set (#1, 23.4%) **did not overrun** --
> 6/6 tickers finished. So a high rail-timeout rate is not sufficient to cause an
> overrun. #24 is still worth doing on censored-distribution grounds, but the causal
> story is weaker than my earlier framing implied.

This is the **endorsed** remedy in the literature (raise a per-ITEM cap against a
censored distribution). The already-shipped budget raise was the **rejected** one
(raising a global batch deadline).

**NEEDS: approve the change to the rail timeout.** It is a timeout on the live
analysis rail, so I will not touch it on a recommendation alone.

## 06-#25 -- merged dispatch: NOT recommended now, NOT withdrawn

Stated unambiguously because "deferred" is a third value against a criterion worded
*"recommended or withdrawn"*. Effective parallelism is **1.85** against a cap of 3,
so headroom exists -- but the measured binding constraint is the rail timeout rate,
and changing dispatch shape and the rail cap together would make neither
attributable. **Revisit after 06-#24 lands.** No decision needed today.

---

## Not asks -- just so they are not lost

- **`86.54` filed today**: the effective cycle budget is logged **only on the timeout
  path** (`autonomous_loop.py:1896`), never at cycle start.
  **CORRECTED before you read this**: an earlier version of this bullet said the value
  was "never logged" and that 86.9 "could not establish" which budget the 2026-08-10
  cycle ran under. **Both were wrong**, and the 86.9 cycle-2 Q/A refuted them -- the
  process is identifiable (`Started server process [43839]`, 2026-08-09 22:11:55) and
  started 6h21m after the `.env` write, so it read the new value on construction. The
  real defect is narrower: establishing the in-force budget takes a multi-step
  inference across startup lines and backup timestamps, and that only works because no
  restart happened to intervene.
- **`86.53` filed**: one cycle-budget concept with **three different defaults**,
  derived rather than counted -- **7200.0** (`settings.py:33`, `settings_api.py:123`
  and `:383`, `cycle_lock.py:63`), **1800.0** (`autonomous_loop.py:507`), and the live
  **10800.0**; plus a validation range of `300.0`-`21600.0` at `settings_api.py:171`,
  which is a bound and not a fourth default. The hazard is the **1800.0** consumer
  fallback: a missing attribute silently yields a 30-minute budget -- a sixth of the
  authorised value -- with no error or alert.
