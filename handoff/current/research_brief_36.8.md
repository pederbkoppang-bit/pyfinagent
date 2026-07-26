# Research Brief — masterplan step 36.8

**Tier:** moderate. `coverage.audit_class = false`. **Date:** 2026-07-26.
**Topic:** the kill-switch archive **merge** lets a frozen archived `peak_update`
beat a fresh, intentional, LOWER live re-anchor → `trailing_dd_pct` computes
against a stale high-water mark → flatten+pause on the first cycle after
restart; `reset_peak` is DARK and `update_peak` ratchets only ⇒ permanent
engine stop.

**READ-ONLY session.** `handoff/kill_switch_audit.jsonl` md5 verified
`ce8fb93348bb9a3bbe26f2d91b1bc05e` before and after every command, including
the one that imported `backend.services.kill_switch` (which builds the
module-level singleton at import). Unchanged. No production file was written.

---

## Read in full (8; gate floor is 5)

| URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|---|---|---|---|
| https://martinfowler.com/eaaDev/RetroactiveEvent.html | 2026-07-26 | canonical pattern catalogue | WebFetch, full | "For rejected events we reverse the event and mark it as rejected." / "**Events marked as rejected are ignored by all further processing, so they aren't re-processed during replay** or reversed in future rewinds." Corrections work by MARKING, not by recency. |
| https://learn.microsoft.com/en-us/azure/architecture/patterns/event-sourcing | 2026-07-26 | official docs (page updated 2026-04-20) | WebFetch, full | "The only way to update an entity or undo a change is to add a compensating event." / "if a bug produces incorrect events, those events persist in the store … you might also need compensating events **or upcasters to handle the bad data during replay**." / "Snapshots are an optimization, not a replacement for the eventstream." / in-place rewrite "breaks immutability and should be a last resort". |
| https://docs.confluent.io/kafka/design/log_compaction.html | 2026-07-26 | official docs | WebFetch, full | "Records with the same primary key are selectively removed when there is a more recent update." / "Any consumer progressing from the start of the log will see **at least the final state** of all records in the order they were written." / "Ordering of messages is always maintained." Precedence is per-KEY latest-wins, and only because a compactable log has a key — a `max()` over an unkeyed stream has no such rule. |
| https://cwiki.apache.org/confluence/display/KAFKA/KIP-101+-+Alter+Replication+Protocol+to+use+Leader+Epoch+rather+than+High+Watermark+for+Truncation | 2026-07-26 | official design doc | WebFetch, full | **[ADVERSARIAL to "just use the watermark"]** "the replicas can diverge, with different message lineage in different replicas" — the High Watermark is *insufficient* to decide truncation; the fix is a generation marker: "the follower gets the appropriate LeaderEpoch from the leader's vector of past LeaderEpochs and uses this to truncate **only messages that do not exist in the leader's log**." |
| https://www.postgresql.org/docs/current/continuous-archiving.html | 2026-07-26 | official docs | WebFetch, full | **[ADVERSARIAL to "live file wins"]** "WAL segments that cannot be found in the archive will be sought in `pg_wal/`… **However, segments that are available from the archive will be used in preference to files in `pg_wal/`.**" Also the epoch analogue: "The default behavior of recovery is to recover to the latest timeline found in the archive… you need to specify `current` or the target timeline ID in `recovery_target_timeline`." |
| https://martin.kleppmann.com/2016/02/08/how-to-do-distributed-locking.html | 2026-07-26 | authoritative blog | WebFetch, full | "a fencing token is simply a number that increases … every time a client acquires the lock" / "The storage server remembers that it has already processed a write with a higher token number (34), and so it **rejects the request with token 33**." Monotonic marker ⇒ stale actors are refused. |
| https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-lock.html | 2026-07-26 | official docs | WebFetch, full | "Object Lock uses a *write-once-read-many* (WORM) model" / "**A legal hold provides the same protection as a retention period, but it has no expiration date.** Instead, a legal hold remains in place until you explicitly remove it." / compliance mode: "can't be overwritten or deleted by any user, including the root user". The named pattern for "this file must not be pruned" is an explicit, per-object hold — a MARKER ON THE DATA, not a policy in a wiki. |
| https://propfirmapp.com/learn/trailing-drawdown | 2026-07-26 | industry practitioner | WebFetch, full | "**It can never move down, only up.**" / "When you withdraw profits, your account balance drops but your drawdown level stays exactly where it was, locked to your historic peak balance." / "The drawdown level only rises until it reaches your original starting balance. Once it locks there, the trailing stops." |

## Identified but snippet-only (11 — context, does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://help.tradeify.co/en/articles/10495897-rules-trailing-max-drawdowns | broker help centre | **HTTP 403** (Intercom blocks WebFetch) |
| https://the5ers.com/prop-firm-drawdown-rules-explained-daily-max-and-trailing-limits-in-2026/ | industry | recency-scan hit; superseded by propfirmapp read-in-full |
| https://funderpro.com/blog/master-prop-firm-drawdown-rules-in-2025/ | industry | recency-scan hit, same content class |
| https://www.thinkcapital.com/prop-firm-drawdown-rules/ | industry | duplicate of read-in-full content |
| https://maventrading.com/blog/trailing-drawdown-prop-trading | industry | already cited in `kill_switch.py:10` docstring |
| https://martinfowler.com/eaaDev/EventSourcing.html | canonical | parent article of the read-in-full Retroactive Event page |
| https://developer.confluent.io/courses/architecture/compaction/ | official course | same material as the read-in-full compaction doc |
| https://nvlpubs.nist.gov/nistpubs/legacy/sp/nistspecialpublication800-92.pdf | standard (2006) | binary PDF; and it is 20 years stale on immutability — NIST is mid-revision |
| https://csrc.nist.gov/pubs/sp/800/92/final | standard landing page | metadata only |
| https://martendb.io/events/learning.html | library docs | snapshot/replay mechanics, no authority-boundary rule |
| https://quality.arc42.org/approaches/event-sourcing | quality model | overview only |

**URLs collected: 30 unique.** Query variants run: year-less canonical
("event sourcing compensating event … replay authority", "Kafka log compaction
tombstone … precedence recovery", "high water mark trailing drawdown reset after
deposit withdrawal prop firm"), current-year 2026 ("event sourcing 2026 …
epoch marker snapshot as authority boundary"), and last-2-year 2025
("trailing drawdown high water mark recalculated after withdrawal 2025 …",
"audit log retention … NIST 800-92 2025").

## Recency scan (2024-2026) — 3 findings, one of them decision-changing

1. **DECISION-CHANGING.** As of 2025-2026 the futures-prop industry has
   converged on *withdrawal-triggered* HWM re-anchoring: FundedNext resets the
   max-loss limit "to your exact initial account balance" after the first
   payout; Tradeify/Apex "payouts trigger the drawdown lock mechanism … floor
   will permanently lock at your starting balance + $100"; Topstep moved
   trailing DD to end-of-day balance through 2025. So a **downward re-anchor
   after a capital withdrawal is standard, documented, and AUTHORIZED BY A
   DISCRETE EVENT (the payout), never by continuous recomputation** — which is
   exactly the shape 36.8 needs, and it argues against any "whichever file is
   fresher wins" heuristic. (the5ers 2026 guide, funderpro 2025, propfirmapp.)
2. The Azure Event Sourcing page was **substantively revised 2026-04-20** and
   now names *upcasters* alongside compensating events as the sanctioned way to
   "handle the bad data during replay" — i.e. a replay-time transform is a
   first-class, current-doctrine option, not a hack.
3. **Null result, stated honestly:** no 2024-2026 source found addresses the
   specific tension "a monotonic aggregate (`max`) must nonetheless be
   correctable". The literature's answer is uniformly *don't correct the
   aggregate — mark the stream* (Fowler's rejected-events, Kafka's leader
   epoch, PostgreSQL timelines). That absence is itself the finding: there is no
   clever `max()` variant to reach for; the fix must be a boundary marker.

---

## Key findings

**F1 — `max()` is commutative, so it structurally cannot honour a correction;
every authority in the literature solves this with a MARKER on the stream, not
with recency.** Fowler: rejected events "are ignored by all further processing,
so they aren't re-processed during replay". Kafka KIP-101 exists *precisely*
because a high-watermark number could not tell divergent lineages apart, and
the fix was a generation marker (leader epoch) that says which history is
authoritative. PostgreSQL's answer to the same question is the **timeline ID**:
after a PITR re-anchor, subsequent WAL belongs to a *child timeline*, and
recovery must be told which lineage to follow. Kleppmann's fencing token is the
same idea reduced to one integer. **Read-across: 36.8 needs an authority
BOUNDARY row in the audit stream, not a rule about which file it came from.**

**F2 — "the live file wins" is refuted by the closest production analogue.**
PostgreSQL explicitly prefers the ARCHIVE over the live directory: "segments
that are available from the archive will be used in preference to files in
`pg_wal/`" — because the live segment is the one that can be torn, truncated or
partially written. That is precisely the pyfinagent situation: `handoff/kill_switch_audit.jsonl`
is the file that *did* get moved twice (`kill_switch.py:48-65`), and TODAY it
contains **zero baseline rows** (measured below). A live-file-precedence rule
would therefore be data-dependent and unpredictable — and it would regress
36.7, whose entire point is that the archives hold the true mark.

**F3 — the correct minimal boundary already half-exists in this codebase:
`peak_reset` is an authority boundary.** `kill_switch.py:262-269` documents it
as an "AUTHORITATIVE downward move (assignment, not max) … later `peak_update`
rows ratchet up from the reset value because the rows are replayed in `ts`
order." That is a per-lineage `max()` — a boundary that resets the fold — which
is exactly Kafka's leader-epoch semantics expressed in JSONL. **The gap is that
the *accidental* form of a re-anchor is invisible**, and `kill_switch.py:412-434`
(36.12's own docstring) says so verbatim: `update_peak`'s row "carries only
`nav`, so an anchor-from-None is forensically identical to a legitimate upward
ratchet". An unmarked anchor cannot be honoured by any replay, no matter how the
files are ordered.

**F4 — the mechanism the step text asserts is only reachable through a narrower
door than "a book re-anchors lower", and the contract must say so.** With the
merge in place, `update_peak` (`:405-410`) reads the ALREADY-MERGED in-memory
peak, so **it will refuse to write a lower `peak_update`**. A fresh *lower*
live `peak_update` can therefore only be written when the in-memory peak was
`None`/lower at write time, i.e. when that boot's merge did not see the higher
archived peak. Reachable paths, all real: (a) `_audit_source_paths` swallows any
archive-scan exception (`:104-105`) → a transient unreadable `handoff/audit/`
yields peak=`None` → the cycle writes the current, lower NAV as a `peak_update`
(`paper_trader.py:1133`) → the next boot, archives readable again, restores the
older higher peak; (b) `handoff/audit/` absent at that boot (fresh clone /
pre-sweep) and populated later; (c) an archive file *restored* later (git
checkout, backup) retroactively RAISING the peak. Conversely, once
`kill_switch_peak_reset_enabled` is flipped, a `peak_reset` written today sorts
AFTER every archived row (corpus max ts = 2026-07-24) and already wins by
ts-order — **so the DARK flag is what makes this reachable at all, and turning
it on shrinks, not grows, the hole.** The fixture in criterion 1 is legitimate
and constructible; the contract should record that its production door is the
swallow/absent-archive asymmetry rather than a plain withdrawal.

**F5 — a cap or pruning policy would be actively harmful here, and that is
MEASURED, not argued.** The true high-water mark 24666.57 lives in the OLDEST
archive file (`handoff/audit/kill_switch_audit.jsonl`, 2026-06-03). Any
"keep the N newest files" cap drops exactly the file that determines the live
threshold: peak would fall to 24124.77 (−2.2pp of trailing-DD headroom), and
dropping all four disarms the trailing leg entirely. The literature's mechanism
for "data whose deletion changes a control's behaviour" is a **hold marker on
the data**, not a retention schedule: S3 Object Lock's legal hold "has no
expiration date … remains in place until you explicitly remove it". The
pyfinagent-native equivalents already exist in two shapes — the housekeeping
`HANDOFF_ROOT_KEEP` allowlist (`scripts/housekeeping/backfill_handoff_archive.py:58`,
`verify_handoff_layout.py:49`, pinned byte-identical by
`test_phase_36_7_kill_switch_allowlists_agree_between_the_two_scripts`) and git:
**all five audit files are tracked** (`git ls-files` — measured), so a deletion
is recoverable and shows up in review. Criterion 3's honest discharge is
therefore: measure the cost (done, below), REFUSE a cap with the measured
reason, and extend the existing allowlist idiom to
`handoff/audit/kill_switch_audit*.jsonl` with a test.

---

## Internal code inventory (HEAD, post-36.7 + post-36.12)

| File | Lines | Role | Status |
|---|---|---|---|
| `backend/services/kill_switch.py` | 760 | whole defect surface | read in full |
| `kill_switch.py:91-108` | `_audit_source_paths` | unconditional MERGE (`paths.extend(sorted(archive.glob(...)))`, live file appended LAST); exception on the scan is swallowed to a warning (`:104-105`) | **THE DEFECT + F4's door** |
| `kill_switch.py:183-211` | `_read_audit_rows` | sorts `(ts, src_idx, line_idx)`; ts-less rows collate FIRST; per-file read errors warn and continue | 36.7's hinge — file order is NOT authority |
| `kill_switch.py:249-261` | `peak_update` replay | RATCHET `nav > self._peak_nav` (36.7 fix) | the `max()` an archive can win |
| `kill_switch.py:262-275` | `peak_reset` replay | bare ASSIGN `_coerce_nav(row.get("new_peak"))`, **no None guard**; clears `_baseline_provenance` | already an authority boundary (F3); 36.15's defect |
| `kill_switch.py:276-281` | `baseline_anchor_on_lost_history` replay | provenance FLAG only, assigns no baseline | 36.12; deliberately replay-inert |
| `kill_switch.py:405-410` | `update_peak` | ratchets UP only; row carries only `nav` | cannot express an anchor (F3/F4) |
| `kill_switch.py:412-444` | `record_lost_history_anchor` | docstring says "Step 36.8 … should consume this row rather than introduce a second one" | see recommendation |
| `kill_switch.py:446-477` | `reset_peak` | DARK unless `settings.kill_switch_peak_reset_enabled` (`settings.py:39`) | criterion 4: must stay untouched |
| `kill_switch.py:111-138` | `_coerce_nav` | None on absent / unparseable / non-finite / <=0 | shared by both peak branches |
| `kill_switch.py:488-514` | `baseline_history_exists` + `_BASELINE_EVENTS` (`:485`) | re-reads the merged stream; set = {sod_snapshot, peak_update, peak_reset} | **a NEW event name must be added here** or 36.12's probe goes blind to it |
| `kill_switch.py:480` | `_state = KillSwitchState()` | replay at MODULE IMPORT | why an exec'd module wrote 54 rows into the live trail |
| `backend/services/paper_trader.py:1130-1133` | `first_ever_boot` + `state.update_peak(nav)` | the only live `update_peak` caller; runs every cycle before the SOD roll | where an anchor-vs-ratchet marker would be stamped |
| `backend/services/paper_trader.py:1196` | `record_lost_history_anchor(...)` | 36.12 block-not-pause path | the accidental re-anchor site |
| `backend/tests/test_phase_36_7_kill_switch_rotation_rearm.py` | 1017 | `ks_tmp_audit` (`:108`, patches `_AUDIT_PATH` only — archive dir is derived), `isolated_state` (`:133`), **autouse live-file write-protect** (`:176-193`) | a 36.8 test MUST reuse these |
| `backend/tests/test_phase_36_12_kill_switch_trading_path_block.py` | 840 | includes `test_phase_36_12_the_new_event_is_replay_inert` | pins 36.12's event as NON-authoritative |
| other test files touching the peak API | — | `test_dod4_tier1_coverage_investment.py`, `test_phase_38_1_kill_switch_auto_resume.py`, `test_64_3_kill_switch_machine.py`, `test_book_safety_69.py` | regression surface for any replay change |

### Measured archive corpus (READ-ONLY)

| File | rows | ts range | events | max `peak_update` |
|---|---|---|---|---|
| `handoff/audit/kill_switch_audit-v2.jsonl` | 49 | 06-10T17:59 → 06-11T10:48 | pause 24, resume 24, sod 1 | — |
| `handoff/audit/kill_switch_audit-v3.jsonl` | 387 | 06-11T11:40 → 07-24T05:48 | pause 178, resume 178, peak_update 5, sod 26 | 24124.77 @ 06-22T18:10 |
| `handoff/audit/kill_switch_audit-v4.jsonl` | 1 | 07-24T18:36 | sod 1 | — |
| `handoff/audit/kill_switch_audit.jsonl` | 452 | 04-20T12:01 → 06-10T13:40 | pause 261, resume 149, peak_update 15, sod 26, cleanup 1 | **24666.57 @ 06-03T19:04** |
| `handoff/kill_switch_audit.jsonl` (**LIVE**) | 8 | 07-25T11:35 → 07-25T11:37 | pause 4, resume 4 | **— none** |
| **total** | **897** | 2026-04-20 → 2026-07-25 | `peak_update`=20, **`peak_reset`=0**, `sod_snapshot`=54, `baseline_anchor_on_lost_history`=0 | **24666.57** |

Read-only restore, real corpus: `KillSwitchState().snapshot()` →
`{'paused': False, 'sod_nav': 23838.19, 'sod_date': '2026-07-24',
'peak_nav': 24666.57, 'baseline_provenance': None}`.
**100% of the live baselines come from the ARCHIVES today** — `peak_nav` from the
oldest file, `sod_nav` from `-v4`. The archives are not a hypothetical
determinant of a live safety threshold; they are the *only* determinant.
Zero `peak_reset` rows exist, so the authoritative-downward path has never run
in production.

### Boot cost (criterion 3, measured not feared)

`_read_audit_rows()`: **897 rows in 0.88 ms**. Ten successive
`KillSwitchState()` constructions: mean **0.95 ms**, min 0.88, max 1.31
(a linear per-line `json.loads` scan ⇒ **≈1.06 µs/row**). Extrapolated on the
same slope: 10k rows ≈ 11 ms, 100k ≈ 106 ms, 1M ≈ 1.1 s. **Boot cost is a
non-issue at any plausible growth rate.** The real hazard in criterion 3 is
semantic (a pruned file silently moves a live threshold — F5), not performance;
say that explicitly rather than adding a cap.

---

## Consensus vs debate (external)

**Consensus:** corrections in an append-only stream are expressed by APPENDING a
marked/boundary event, never by editing history and never by a recency
heuristic (Fowler; Azure; Kafka KIP-101; PostgreSQL timelines; Kleppmann). A
monotonic fold is reset at the boundary, not made non-monotonic.

**Debate / adversarial:** *which* segment is authoritative is genuinely
contested and the two production answers point OPPOSITE ways. Kafka log
compaction: most-recent-per-key wins. PostgreSQL recovery: **archive wins over
the live directory**, because the live segment is the one that can be torn.
pyfinagent's live file is exactly PostgreSQL's `pg_wal/` — rotatable, and right
now empty of baselines. **The debate resolves against file-based precedence in
either direction and in favour of an in-stream boundary marker.**

**Pitfalls from the literature:** (i) Azure — "if a bug produces incorrect
events, those events persist in the store"; the 20 unmarked legacy
`peak_update` rows can never be retro-marked, so the replay must default
UNMARKED ⇒ RATCHET (this is what preserves 36.7). (ii) Azure — "Snapshots are
an optimization, not a replacement for the eventstream"; do not convert the
archives into a snapshot-and-discard. (iii) Kafka — compaction "will never
reorder messages"; do not touch the `(ts, src, line)` sort. (iv) Kleppmann —
a boundary marker only works if it is monotonic and the reader REFUSES anything
older; a marker the replay merely notes is worthless (that is 36.12's
deliberately inert event).

---

## Application to pyfinagent — recommended fix shape

**Position: an in-stream AUTHORITY BOUNDARY (a marked anchor row), unmarked
rows defaulting to ratchet. NOT live-file-precedence. NOT boot-time-only
scoping. NOT read-once-then-discard.**

1. **Mark the anchor at the writer.** `update_peak` (`kill_switch.py:405-410`)
   currently writes `{nav}` whether it is a ratchet or an anchor-from-`None`.
   Stamp the distinction on the row (e.g. `prior_peak: null` / `anchor: true`)
   when `self._peak_nav is None`. This closes the forensic gap that
   `kill_switch.py:412-434` already names in prose, and it is the *only* place
   the two cases are still distinguishable.
2. **Make the boundary reset the fold in the replay.** In `_load_from_audit`,
   a marked anchor row ASSIGNS (like `peak_reset` at `:262-275`) instead of
   ratcheting; rows with ts after it ratchet up from it; rows before it are
   superseded because the sort is by `ts`. This is Kafka's leader-epoch /
   PostgreSQL's timeline semantics in one branch, and it satisfies criterion 1.
3. **Default unmarked ⇒ ratchet.** All 20 existing `peak_update` rows carry no
   marker, so 36.7's behaviour (restore 24666.57 across the merge) is
   byte-preserved — criterion 2, asserted against the real corpus and against
   the existing `test_phase_36_7_kill_switch_peak_replay_ratchets_never_assigns`.
4. **Route BOTH assignment branches through one guarded helper**
   (`_apply_authoritative_peak`): ignore + log loudly when the value does not
   coerce to a positive finite float. 36.8 must not ship a second
   assignment-semantics branch while the first one is unguarded.
5. **Do NOT touch `reset_peak`'s DARK gate** (criterion 4) and do not widen
   `_BASELINE_EVENTS` semantics; if a new event NAME is introduced rather than a
   field on `peak_update`, it MUST be added to `_BASELINE_EVENTS`
   (`kill_switch.py:485`) or 36.12's `baseline_history_exists` probe goes blind
   to it. **Preferring a FIELD on `peak_update` over a new event avoids that
   coupling entirely — recommended.**
6. **Criterion 3:** refuse a cap (F5: the cap would delete the file holding the
   true peak — measured), record the measured 1.06 µs/row boot cost, and extend
   the existing allowlist idiom (`HANDOFF_ROOT_KEEP` in both housekeeping
   scripts) to name `handoff/audit/kill_switch_audit*.jsonl` as
   safety-relevant-do-not-prune, pinned by a test in the style of
   `test_phase_36_7_kill_switch_allowlists_agree_between_the_two_scripts`. Note
   in the artifacts that all five files are git-TRACKED (measured), which is the
   existing recoverability backstop.
7. **On consuming 36.12's `baseline_anchor_on_lost_history` (the step asks):
   read it, do NOT make it authoritative.** Making it assign a baseline would
   break `test_phase_36_12_the_new_event_is_replay_inert` AND is semantically
   backwards — a lost-history anchor is an ACCIDENT (36.12 exists to flag it as
   a fiction), so granting it authority to lower a real high-water mark is the
   under-conservative direction. The *authorized* re-anchor is `peak_reset`
   (token-gated), which matches the measured industry practice: withdrawal-
   triggered resets are authorized by a discrete payout event, never automatic.
   36.8's new marker should live on `update_peak`'s row (the accidental-anchor
   case), and it should ASSIGN only forward in ts-order, never retroactively.

**Also flag for the contract (F4):** state the reachable production door (the
swallowed archive-scan exception at `:104-105` / an absent-then-present
`handoff/audit/`) rather than "a book re-anchors lower", because
`update_peak` cannot write a lower peak while the merged peak is higher.
Consider whether the swallow at `:104-105` should distinguish "no archive dir"
from "archive dir unreadable" — the second is not a safe reason to anchor a new
peak. That is arguably its own step, not 36.8's scope.

## Overlap check (mandatory)

- **36.15 (P1) — SEPARATE, but sequence-coupled. Do NOT absorb.** Same lines
  (`kill_switch.py:249-275`) and same verification command (`-k kill_switch`),
  so a merge collision is certain if both run blind. 36.15 has scope 36.8 does
  not: the two-direction reproduce (None-stays-None vs heals-DOWNWARD to
  20000.0), the "valid `peak_reset` still assigns downward" fixture, and an
  explicit decision about `_append_audit` refusing a non-positive write.
  **Recommendation:** 36.8 lands the shared guarded helper (item 4 above) and
  says so in its artifacts; 36.15 then becomes a small, independently
  mutation-testable follow-on that routes the `peak_reset` branch through it and
  keeps all six of its criteria. Run 36.8 first.
- **36.9 (P0) — no functional collision.** 36.9 edits `evaluate_breach`
  (`:524-616`) — stale `sod_date`, `nav_invalid`/`armed` consistency,
  `sod_nav=0.0`. Disjoint from the replay branches. Only shared surface is the
  `-k kill_switch` test selector and the `isolated_state` fixture. Safe in
  parallel; whoever lands second reruns the selector.
- **36.16 (P1) — frontend only, zero code collision.** Relevant only as a
  constraint: if 36.8 adds any operator-visible string (e.g. "peak anchored,
  not ratcheted"), it must go to `OpsStatusBar.tsx`, which IS mounted —
  `KillSwitchPanel.tsx` is unmounted dead code, so a badge added there reaches
  nobody.

## Prior art in the repo — what carries over

- `handoff/current/research_brief_36.7_80.40.md` — **finding #4 carries over and
  is the hinge**: the `max()` ratchet is sound ONLY because `_read_audit_rows`
  sorts by `(ts, src, line)` and `peak_reset` assigns rather than ratchets. F3
  above builds directly on it: that assign-then-ratchet pair IS already an
  authority boundary, so 36.8 generalises an existing mechanism rather than
  inventing one. Its finding #5 (no external guidance exists on adversarial
  review of AI-authored safety-critical financial code; the project's own
  discipline is ahead of the literature) also carries over and is re-confirmed
  by this session's null result (recency finding 3).
- `research_brief_36.12.md` / `experiment_results_36.12.md` — the
  provenance-marker prior art (Galera `grastate.dat` / `safe_to_bootstrap` /
  `seqno: -1`, cited at `kill_switch.py:496-500`) carries over and reinforces
  the marker-on-the-data recommendation. What does **NOT** carry over is the
  suggestion in `record_lost_history_anchor`'s docstring that 36.8 should
  consume that event as its re-anchor carrier — see recommendation item 7;
  it must stay replay-inert.
- `experiment_results_36.7.md` — the fixture contract (`ks_tmp_audit`,
  `isolated_state`, autouse live-file write-protect) carries over verbatim and
  is mandatory for any 36.8 test.

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (**8**)
- [x] 10+ unique URLs total (**30**)
- [x] Recency scan (last 2 years) performed + reported (3 findings, one
      decision-changing, one honest null)
- [x] Full pages read, not abstracts
- [x] file:line anchors for every internal claim

Soft:
- [x] Internal exploration covered every module named in the spawn prompt
- [x] Contradictions noted (PostgreSQL archive-precedence vs Kafka
      latest-wins — an explicit ADVERSARIAL pair)
- [x] Claims cited per-claim
- [x] Overlap check for 36.9 / 36.15 / 36.16 with a stated position

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 11,
  "urls_collected": 30,
  "recency_scan_performed": true,
  "internal_files_inspected": 9,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "The archive merge is unconditional (kill_switch.py:91-108) and max() cannot honour a correction. Every authority (Fowler rejected-events, Kafka KIP-101 leader epoch, PostgreSQL timelines, Kleppmann fencing tokens) solves this with an in-stream boundary MARKER, not with file recency -- and PostgreSQL explicitly prefers the ARCHIVE over the live dir, refuting live-file-precedence. peak_reset (:262-275) is already such a boundary; the gap is that update_peak's row cannot distinguish an anchor-from-None from a ratchet (kill_switch.py:412-434 says so). Recommend: stamp the anchor on update_peak's row, have the replay ASSIGN at a marked anchor and ratchet elsewhere, default unmarked=ratchet (preserves 36.7 and all 20 legacy rows), route both assignment branches through one guarded helper. Measured: 897 rows / 5 files, 100% of today's live baselines come from the archives (peak 24666.57 from the OLDEST file), zero peak_reset rows ever, boot 0.95 ms (1.06 us/row) -- so refuse a cap (it would delete the file holding the true peak) and extend the housekeeping allowlist instead. 36.15 stays separate but sequence-coupled; 36.9 and 36.16 do not collide.",
  "brief_path": "handoff/current/research_brief_36.8.md",
  "gate_passed": true
}
```
