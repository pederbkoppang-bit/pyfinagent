# experiment_results — phase-86.94

## What changed

| file | change |
|---|---|
| `scripts/qa/verify_no_sliding_windows_86_94.py` | **NEW.** The regression guard: known-member recall, source-derived enumeration, criterion-4 disclosure check, mutation cells. |
| `scripts/qa/replay_changelog_rule_86_68.py` | `CORPUS_SINCE` → `2026-08-11T00:00:00Z`. **One character.** Figures unchanged; the window is now TZ-invariant. |
| `scripts/qa/verify_changelog_flip_86_91.py` | Comment only — discloses that its window assertion is a *substring* test and therefore a pin-presence check, not a TZ check. |
| `handoff/current/experiment_results_86.91.md`, `live_check_86.91.md` | The "reproducible figures" claim qualified **in place** (criterion 5). |

Full measured evidence: `live_check_86.94.md` §A–G.

---

## The finding

**Two defects, one class.** A window is reproducible only if it names an
*instant*. Two ways to fail that, and phase-86.91 fixed one and believed it had
fixed both:

1. **A bare date is not midnight.** `git rev-parse --since=2026-08-11` resolved
   to `2026-08-11 23:09:03 CEST` — today's clock time carried onto the target
   date. `--since=today` likewise resolves to *now*, not midnight (0 commits vs
   64), contradicting `git-log(1)`.
2. **A pinned timestamp is still TZ-local.** `2026-08-11T00:00:00` with *both
   ends pinned* measured **707** under Oslo/UTC/New York and **787** under
   Seoul. An 80-commit spread decided by `$TZ`.

Defect 2 is the one that matters for the harness, because 86.91's remediation
*was* defect 2 and it looked complete. The fix is one character (`Z`), the
published figures are unchanged (707 / 251 / 9 / 11, exit gate green), and the
corpus is now regenerable off this laptop.

**Measured, ≥1h apart:** the bare-date count went **DOWN while the repo GREW**
(376 → 360 across 22:50:20 → 23:51:09) while the pinned form went **up by exactly
the four commits that landed** (424 → 428). The arithmetic closes with no
residual: 376 + 4 added − 20 slid out = 360, and 424 + 4 = 428. A corpus that
shrinks as history grows is a reading of the clock, not a corpus.

**The drift is intermittently invisible, which is what makes it dangerous.** The
obvious boundary date, `2026-08-11`, would have shown *no* change tonight: its
last commit is 22:36:46, already behind the 22:50 cutoff, so its drift window is
exhausted. Two runs an hour apart would have agreed — a true result that reads
like a refutation. The date had to be chosen by measuring where commits actually
sit relative to the sliding band.

---

## The guard

`scripts/qa/verify_no_sliding_windows_86_94.py`, 77 assertions, exit 0.

- **Known-member recall is a hard gate.** The rule must find the pre-86.91 form
  of the replay, recovered from git at `06c3265f`, and classify it SLIDING. If
  the blob becomes unreachable the section **fails** rather than skipping.
- **The rule is written down in source and is wider than "bare date"** — a
  bare-date-only rule would have declared 86.91's TZ-naive pin clean.
- **It is an allowlist, not a ban.** One member is legitimately relative:
  `backend/slack_bot/scheduler.py` builds the Slack "shipped today" digest, and a
  report about today must move with today. A blanket prohibition would break
  correct code and then be switched off.
- **Criterion 4 is enforced as a BOUND MEASUREMENT** — `quoted_as_evidence` must
  equal what a figure-probe search over the tracked corpus actually finds, so the
  judgement can be contradicted (see below).

### Two defects my own rule had first, both of this step's classes

1. **It matched its own documentation.** The replay *documents* this defect in a
   comment quoting `` `--since=2026-08-11` ``; the scanner reported that prose as
   two SLIDING sites. Comments are now stripped, with a control in **both**
   directions — a stripper that quietly does nothing looks identical to a
   correct one.
2. **It failed OPEN on indirection.** `--since={CORPUS_SINCE}` was classified
   REPRODUCIBLE because the value is "decided at the call site". That is exactly
   how the TZ-naive constant stayed invisible. The resolver now reads the literal
   through one level of indirection and **fails closed** when it cannot. This is
   the leg that found the real defect — without it the guard would have shipped
   green over the very thing it was written for.
3. **It flagged ITSELF, and only once committed.** `git ls-files` cannot see an
   untracked file, so the checker was invisible to its own scan until the moment
   it shipped — which is exactly when a self-blind guard is worst. Its section-[4]
   fixtures are deliberately-sliding literals and became 14 false findings. Now
   self-excluded, with an assertion that the exclusion is **exactly one file** so
   it cannot grow into an escape hatch, and the residual stated: a real sliding
   window in this checker is not caught by this checker.
4. **Docstrings are a third comment form.** `is_prose` knew only `#` lines, so
   the module docstring — which quotes a bare-date window while *explaining* the
   defect — was reported as findings. `strip_docstrings` blanks triple-quoted
   blocks while preserving line numbering, with its own control pair.
5. **The rule covers `.sh`; every cell mutated `.py`.** A guard demonstrated on
   one language is demonstrated on half its scope, and the hooks are shell.
   Added a shell kill, a shell negative control and a shell-comment case.

---

## Criterion 4 — the judgement, stated

| member | class | judgement (measured, cycle 4) |
|---|---|---|
| `scheduler.py:503` `midnight` | LEGITIMATELY RELATIVE, figures QUOTED | The window stays (a report about today must move with today), but the judgement was **FALSE through cycle 4 and is corrected**: `handoff/archive/misc/live_check_62.8.md:31` (tracked) quotes `"*Shipped today*" with 12 real commit lines` as read-back evidence. Quoted, unreproducible, inert. `quoted_as_evidence: True`. *(Cycle 5 also cited `Steps closed: …` here; cycle 6 withdrew it — not a quotation, and not this window's figure. See `live_check_86.94.md` §J8.)* |
| `frontend_route_inventory.py:73` `30.days` | SLIDING, left | Its figures HAVE been quoted as evidence — 3 tracked files, 5 hits (`"usage_source": "git_activity_30d"`, `12/12 integer opens_30d`, `opens_30d=0` in `handoff/archive/phase-4.7.0/`). Quoted, unreproducible, inert. `quoted_as_evidence: True`. |
| `verify_decision_log_86_97.py:360` `{first_stamp}` | runtime-derived, allowed | Figures **are** quoted (`commits=51  decision lines=26  gap=25`) — always with the clock time they were taken at, and the checker asserts a *relationship*, not a number. `quoted_as_evidence: True`. |
| `replay_changelog_rule_86_68.py:114` `{CORPUS_SINCE}` | was SLIDING → **FIXED** | The TZ-naive pin. |

**The bool is now bound to a measurement, and the previous binding is gone rather
than annotated.** Cycles 2-3 bound it to `mentions_reviewed`, a pinned count of
files containing the member's *filename*. Measured: flipping
`frontend_route_inventory` True→False and `scheduler` False→True each left the
guard's FAIL set byte-identical — **a factually wrong judgement shipped green in
both directions.** It also counted over the working tree, 89.5% of which is
gitignored, so it was a number about a machine in the very class this step
closes; it went red inside the commit that recorded it green, because that commit
added a park note that merely *names* the scripts.

`figure_probes` replace it: patterns for a figure *produced by that member's
window*, each derived from the emitting expression in the member's own source,
matched against the **git-tracked** corpus, with the check asserting
`quoted_as_evidence == bool(hits)`. A wrong bool now fails in both directions
(M-D, M-E), and drift on the relevant corpus still re-opens the judgement (M-G),
while prose that merely names a file is inert. This step's own artifacts remain
excluded, and the exclusion is stated: they necessarily name every member, which
would guarantee a hit for each and make the check vacuous.

---

## Criterion 5 — corrected in place, not annotated

The figure found TZ-dependent was the replay's corpus and the
"reproducible figures are 707 / 251 / 9 / 11" claim built on it. Both 86.91
artifacts asserting that claim now carry the bound **in the same sentence**, and
the enumeration was driven by the *claim* (the window string and the figures),
not by my own phrasing — the 86.97 FAIL earlier tonight was caused by sweeping
for my own wording, and I was not going to repeat it in the step whose criterion
3 is a recall test.

`verify_changelog_flip_86_91.py:440` asserts the window by **substring**, so it
matched the `Z` form and stayed green through the fix — but it would equally stay
green if the `Z` were removed again. That bound is now disclosed in its own
comment, and the TZ property is asserted by the new checker, which resolves the
constant and classifies its *value*.

---

## Verification

```
$ bash -c 'source .venv/bin/activate && python scripts/qa/verify_changelog_flip_86_91.py > /dev/null && echo green'
green
```

**Disclosed:** the immutable command runs the *86.91* checker, which is green
today and would have stayed green through every defect this step is about. It
cannot fail on the class. The real evidence is in `live_check_86.94.md`.

```
verify_no_sliding_windows_86_94.py   ALL GREEN: 77 passed, 0 failed   (exit 0)
verify_changelog_flip_86_91.py       ALL GREEN: 42 passed, 0 failed
verify_workflow_args_boundary.mjs    ALL GREEN: 96 passed, 0 failed
ruff (default ruleset, new file)     All checks passed!
```

---

## Scope honesty — what I did NOT do

- Did **not** ban relative windows. One member is correctly relative; a ban would
  break it and would then be disabled.
- Did **not** change `frontend_route_inventory.py`. Classified SLIDING, judged
  acceptable, and the judgement is recorded with the measurement behind it.
- Did **not** re-run or amend 86.68's or 86.91's closed criteria. 86.91's
  *claim* was qualified; its criteria were not touched.
- The resolver sees through **one** level of indirection (a module-level string
  literal in the same file). A value assembled at runtime, or imported from
  another module, resolves to nothing and is failed **closed** — correct, but it
  means such a site needs an allowlist entry rather than being proven safe. The
  `{first_stamp}` entry is exactly that case and says so.
- The criterion-4 mention check uses **name presence** as its proxy. It cannot
  tell a quoted count from a descriptive mention; that is why it enforces a
  stated judgement instead of asserting absence, and why the mention sites are
  printed for an auditor rather than swallowed.

---

## Cycle-3 remediation (after a second FAIL)

The cycle-2 Q/A returned FAIL again, and its central finding was that my
**correction had accompanied rather than replaced** — the exact criterion I was
enforcing, committed inside the step written to enforce it. Measured and
confirmed by me before fixing:

- `experiment_results_86.91.md:141` still asserted, in the present tense,
  `CORPUS_SINCE = "2026-08-11T00:00:00"` while the shipped constant was
  `...00Z`. My parenthetical sat below it; the false sentence remained.
- `:146`, `live_check_86.91.md:90-91` and `harness_log.md:35558` carried the
  naive window the same way.

**All four are now REPLACED**, not annotated. The one verbatim capture among
them (`experiment_results_86.91.md:77`) was **regenerated** rather than edited —
the script now prints the `Z` form, so the capture is a fresh reading, not a
retouched one.

**Criterion 4 was inverted in the artifact that matters.** The corrected
judgement landed in the source allowlist, while `live_check_86.94.md:262` — the
file the masterplan's `live_check` field names — still carried the falsified
"Mentioned in **0** files … never quoted as evidence". And §E's `[3b]` capture
still showed the cycle-1 counts. My cycle-2 claim that "§C/§E/§G were
regenerated" was true for two of three. Both are now replaced from the shipped
run (`282 / 6 / 49`).

**Criterion 6 — the argv-list form.** `subprocess.run(["git","log","--since","2026-08-11"])`
is this repo's *dominant* git idiom, and the option pattern required `=` or
whitespace immediately after the option name. With a quote there the line never
matched at all, so the site was **invisible** and the fail-closed `<unparsed>`
path never fired. Both argv spellings are cells now. **RETRACTED (phase-86.94 cycle-3 verdict, and I re-measured it myself).** An earlier revision of this paragraph claimed the widened rule *"immediately found a live site the old one missed"*. It measures **zero**: reverting only the `WINDOW_RE` widening leaves the live-site enumeration byte-identical, and the cycle-2 blob enumerates the same four sites. Every live git window in this repo uses the `=` spelling. What I mistook for a find was `census_qa_write_guard_log_86_31.py:64` — an `argparse` flag for a non-git tool, i.e. a FALSE POSITIVE that I then excluded with the git-proximity rule. The widening's real effect is confined to the mutation cells: it closes a future-introduction gap, which is what criterion 6 governs, and it changes nothing about today's tree.

That widening had a consequence, disclosed rather than smoothed: it also matched
`argparse` definitions (`ap.add_argument("--before", default=None)`), which are
CLI flags for non-git tools. A window site now additionally requires `git` in
view — same line or the three above, which covers a multi-line argv list without
swallowing an argparse block. **Residual stated in source:** a git argv list that
puts the word `git` more than three lines above its window option is not matched.

**The criterion-4 predicate was token-satisfiable, and my first replacement was
worse.** Checking for the word "quoted" passed for the true entry, the false one
it replaced, *and* the sentence "never quoted as evidence". My first fix — a
deny-list of phrases — then fired on the entry's own **rejection** of one of
them ("…not 'never quoted'"), i.e. the probe matched its own correction. The
judgement is now **data**: `quoted_as_evidence` is an explicit bool bound to
`figure_probes` — patterns for a figure the member's window actually emits,
matched against the git-tracked corpus. A wrong bool fails in both directions and
a change in the quoted figures re-opens the judgement, while prose that merely
names the file is inert. (Cycle 3 bound it to `mentions_reviewed`, a filename
count over the working tree; that binding is removed, not annotated — see
`live_check_86.94.md` §J.) A bool cannot be satisfied by vocabulary.

**37 → 45 → 68 → 74 → 77 assertions.**

---

## Cycle-4 remediation (after the overnight PARK at the 3-attempt cap)

**Starting state: the guard was RED, not "ALL GREEN 45/0" as the park note
records.** `42 passed, 3 failed` — all three the `mentions_reviewed` tripwire
firing because last night's own park note and day report *name* the guarded
scripts. Provenance: `handoff/current/day_halt.md`.

| file | change |
|---|---|
| `scripts/qa/verify_no_sliding_windows_86_94.py` | `mentions_reviewed` → `figure_probes`; corpus restricted to `git ls-files`; `quoted_as_evidence == bool(hits)`; 4 fail-closed `<unparsed>` cells; per-cell mechanism assertions; the stale space-form comment replaced. **45 → 68 assertions.** |
| `handoff/current/live_check_86.94.md` | §E replaced (not annotated); §J added with the full cycle-4 evidence. |
| `handoff/current/experiment_results_86.94.md` | criterion-4 section replaced; stale `45` counts corrected. |
| `.claude/masterplan.json` | 86.94 note corrected — the design it described no longer exists. |

**The three findings the cycle-3 evaluator named, each measured before and after:**

| # | finding | before | after |
|---|---|---|---|
| (a) | `quoted_as_evidence` only `isinstance`-checked, so a wrong bool stays green | M-D **SURVIVED**, M-E **SURVIVED** | both **KILLED** |
| (b) | the `<unparsed>` fail-closed branch has no mutation cell | M-A **SURVIVED** | **KILLED** |
| (c) | the argv cells may be credited to the wrong leg | M-B **SURVIVED** (value-parse leg uncovered); M-C kills exactly the 2 argv cells | M-B **KILLED**; attribution corrected to the visibility leg |

The two uncovered mechanisms in (b) and (c) were **masking each other**: with
`VALUE_ARGV_RE` neutralised, argv sites fell through to the fail-closed branch and
were still flagged, so neither leg could be tested while the other was intact.

**One finding the guard was hiding, surfaced by the research gate:** the
`mentions_reviewed` corpus walked the working tree, of which **89.5%
(43,927 of 49,094 `.md` under `handoff/`) is gitignored**. 45 of 50 hits for
`frontend_route_inventory` were in the ignored quarantine, and the allowlist's own
smoking-gun citation was itself gitignored — so the evidence for
`quoted_as_evidence: True` was absent on a fresh clone. The corpus is now the
tracked set, and the True judgements were re-verified against tracked carriers.

**Mutation matrix: `killed=7 survived=0 unscorable=0`, control observed GREEN
(68/0) before any cell was scored.** Full transcript in `live_check_86.94.md` §J.
M-A/B/D/E moving from SURVIVED to KILLED is the evidence that this replaced a
weaker check with a stronger one rather than deleting a red check to get green.

**Verification, re-run after every edit:**

```
$ bash -c 'source .venv/bin/activate && python scripts/qa/verify_changelog_flip_86_91.py > /dev/null && echo green'
green

$ python scripts/qa/verify_no_sliding_windows_86_94.py
ALL GREEN: 77 passed, 0 failed
```

**Still disclosed, unchanged:** the immutable command runs the *86.91* checker and
cannot fail on any defect in this step's class; it proves only that this work did
not break 86.91.

---

## Cycle-5 remediation (after a CONDITIONAL on evidence integrity)

The cycle-4 Q/A found two defects. Both reproduce, and the first is this step's
own recurring class landing on me one more time.

**1. The `scheduler.py` criterion-4 judgement was FALSE, and my probe was blind to
the counterexample.** `handoff/archive/misc/live_check_62.8.md:31` is tracked and
quotes `"*Shipped today*" with 12 real commit lines` — a count of exactly what the
midnight window emitted, quoted as verification evidence. *(This paragraph
originally cited a second figure, `Steps closed: …`, and called both "counts of
exactly what the midnight window emitted". Cycle 6 withdrew it on both grounds —
it was not a quotation but my own regex truncation, and it is not this window's
output. See the cycle-6 section below.)* My
cycle-4 probe was documented as derived from `formatters.py:102-109` but could not
match that code's output, because rendering goes through `add()` at
`formatters.py:71-76` which emits `*{title}*\n{body}` — an asterisk between "today"
and the newline. **I had read the call site, not the renderer.** Fixed by
EXECUTING `format_away_digest_sections` and scoring probes against its real output;
`scheduler.py` is now `quoted_as_evidence: True` — quoted, unreproducible, inert.

**2. Every probe now carries POSITIVE CONTROLS.** The structural hole was that for
a `False` claim the check passes *precisely when the probes match nothing*, so a
dead probe set is indistinguishable from a measured absence — the Q/A proved it by
substituting a never-matching literal and getting a clean 68/0. `probe_fixtures`
are sampled from the real emitted text, and a probe matching none of its own
fixtures now FAILS whatever the bool says (cells M-H, M-I).

**3. The correction sweep is rewritten to reproduce.** Class A is 8 carriers, not
6; Class B has 2 coincidental hits, not 9. Cycle 4 quoted one pattern and reported
counts from a wider one it never showed — the same "figure that cannot be
regenerated" defect the step exists to close. Every class now quotes its
enumeration command, and `day_report_2026-08-17.md:49` ("Guard ships green at
45/0", which `day_halt.md` itself calls false) is corrected in place.

**Matrix: killed=9 survived=0 unscorable=0, control GREEN at 74/0 first.** Guard
68 → 74 assertions.

---

## Cycle-6 remediation (after a FAIL on criterion 5 + evidence integrity)

The cycle-5 Q/A returned **FAIL**. Four findings, all mine, all reproduced by me
before accepting.

**1. The correction accompanied instead of replacing — inside the step whose
criterion 5 is that rule.** §E of `live_check_86.94.md` carried two output blocks.
My cycle-5 edit rewrote the *wrong* one, leaving the stale cycle-4 output — showing
`QUOTED in 0` and `quoted_as_evidence=False`, the exact FALSE judgement this step
had just corrected — sitting under the label **"The current run prints:"**, while
the real output was labelled *superseded*. **Both stale blocks are deleted**;
§E now carries one regenerated block, so there is nothing left to mislabel.

**2. A misquote presented as a quotation.** I cited `Steps closed: 6` as text from
`live_check_62.8.md:31`. That file's only `Steps closed` line is `:36`, reading
`"Steps closed: 61.1, 62.0, 17.4, 62.3"`. My probe `Steps closed:\s*\S` matched
and stopped at the `6` of `61.1`, and I printed the fragment in quote marks — the
*paraphrase-inside-quote-marks* defect this document corrects for another member
three sections earlier. Removed from all five carriers.

**3. False provenance.** That same probe was bound to `d["steps_flipped_today"]`,
which `_steps_closed_from_log()` reads from `handoff/harness_log.md`
(`scheduler.py:511-513`) — **not** the midnight window. The Q/A showed the
consequence: keeping only that probe left the bool `True` at 74/0, so the
judgement was sustainable on evidence the allowlisted window never emitted. Probe
removed; the judgement now rests on the genuine commit-line evidence alone.

**4. An overclaim, withdrawn, and then actually fixed.** Cycle 5 said the
positive-control check "protects any future member claimed `False`". The Q/A
falsified it by execution: a probe and fixture **co-written from one misreading**
passed at a clean 74/0, because provenance was a source comment and nothing
enforced it. Fixtures are now `{text, source}` pairs and the guard asserts the
text is really present in the **tracked** file named. The scheduler control is
`scripts/qa/fixtures/shipped_today_render_86_94.txt`, generated by *executing* the
formatter (`scripts/qa/gen_shipped_today_fixture_86_94.py`; re-running it
reproduces the committed file byte-for-byte). **That mutant is now cell M-J and it
is KILLED.**

**Matrix: killed=10 survived=0 unscorable=0, control GREEN at 77/0 first.** When
an anchor failed to apply, the cell scored **UNSCORABLE** and the run reported
`MATRIX NOT CLEAN` rather than counting it — twice, and both times I fixed the
anchor rather than the score.

**Filed, not fixed here: masterplan `86.104`** — section `[1]` re-implements the
scan inline rather than calling `scan_text`, so `[1]` and `[2]` can disagree about
the same line and the known-member gate cannot tell the defective pre-86.91 blob
from the corrected one. A WARN-level finding from the same evaluator; criterion 3
is literally satisfied, so this is a separate defect rather than a reopening.
