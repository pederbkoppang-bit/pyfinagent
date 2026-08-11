---
name: timeout-raise-86-9
description: Deadline-terminated runs are RIGHT-CENSORED so their max is the budget not the work; the endpoint that "doesn't exist" existed; lru_cache+cache_clear means no startup snapshot
metadata:
  type: project
---

# Timeout-raise research (step 86.9, 2026-08-11)

Four class-level findings from researching "is raising a global timeout ever the right fix".

## 1. A deadline-terminated run is RIGHT-CENSORED -- its duration is the budget, not the work

**Why:** pyfinagent's cycle table had two entries of `7200.117s` and `7200.077s`. Those look
like measurements. They are not: they are the deadline. The uncensored values (from the
diagnostic's own `PROJECTED` column) were 8554s and 8529s.

**How to apply:** whenever a duration column contains values equal to a configured cap,
say so explicitly and refuse to compute percentiles from that column -- they are biased LOW.
The max is not an observation. Related: [[normalization-rule-must-be-stated-with-the-ratio]].
Also: with n samples the largest observation sits near n/(n+1), so **n=7 cannot express a p95
at all** (7/8 = 0.875). Reporting one would be fabrication -- say "not computable at this n".

## 2. The discriminator between "needs more time" and "a hung dependency" is PER-ITEM, never batch wall-clock

**Why:** batch wall-clock cannot separate the two -- both look like "it overran". The tests
that do separate them are (a) the **censoring test** (count successes landing within epsilon
of the per-item cap; a pile-up means the cap is truncating real successes), (b) the
**bimodality test** (a spike exactly AT the cap = hung dependency; a whole-distribution shift
= genuinely slower work), and (c) the **yield test** (timeouts x cap; if the wasted time
exceeds the overrun, the overrun is an artifact).

**How to apply:** run the yield arithmetic before endorsing any budget raise. On 86.9 it was
decisive: 32 timeouts x 150s = 4800s wasted against a 1329s overrun.

## 3. "Is raising a timeout ever right?" -- YES for a PER-ITEM cap, NO for a global per-BATCH deadline

**Why:** the literature is NOT "always shorter". Azure pairs "set a shorter time-out" with
"But ensure that the time-out is long enough for the operation to succeed most of the time",
and Google concedes picking one "can be something of an art". The legitimate raise is a
per-item cap sitting BELOW the success distribution's tail. The rejected raise is the global
batch deadline, which buys time for whatever is wasting it.

**How to apply:** when a step proposes a timeout raise, first ask **which layer**. Per-item +
measured censoring = endorsed. Per-batch = the anti-pattern ("deadlines several orders of
magnitude longer than the mean request latency is usually bad", Google SRE ch.22).

## 4. Check the API surface before accepting a "no endpoint exposes it" premise

**Why:** 86.9's spawn prompt stated as ESTABLISHED that no endpoint exposed the cycle budget
and that criterion 1 was therefore the hard one. `GET /api/settings/` had exposed it since
step 38.12, returned the live value on the first curl, and had **already been used as live
evidence** in `handoff/archive/misc/live_check_27.6.md:93`. One grep of the API module would
have found it.

**How to apply:** a caller's "ALREADY ESTABLISHED" list is a hypothesis, not a fact. Grep the
API/router modules for the field name before building a brief around its absence. Sibling of
[[suspect-the-clean-check]] -- suspect the clean *premise* too.

## 5. `lru_cache`d settings + a runtime `cache_clear()` means there is NO startup snapshot

**Why:** the standing rule [[committed-is-not-in-force]] says a running process holds the
pre-fix module, so a `.env` edit needs a restart. The **converse trap** also exists: if any
hot path calls `get_settings.cache_clear()` (pyfinagent does it **per ticker**, to cure a
cross-worker desync), the process re-reads `.env` from disk continuously and the edit is live
WITHOUT a restart.

**How to apply:** before claiming either "needs a restart" or "already in force", grep for
`cache_clear` on the accessor. And note that a value captured ONCE into a local (e.g. a
deadline captured at cycle start) does not track later re-reads -- so an endpoint can report
a different number than the in-flight job is actually running under.
