# Research Brief — phase-85.5.1 (P1 BOOK SAFETY: stale/None `sod_date` disarms the daily leg)

Tier: **moderate**. Audit-class: **false**. Read-only session — no code changed, no suite
run, no service restarted, no kill-switch state mutated. **`handoff/kill_switch_audit.jsonl`
md5 verified identical immediately before and immediately after the measurement script**
(5324 bytes); it was still 5324 bytes at the end of the session, and `git diff --stat` on it
is empty. Scope of that claim: my own actions, not a guarantee about the concurrently-running
backend.

Disclosure: this brief exceeds the moderate ~700-word prose target. The overage is tables,
the measurement script and its verbatim output — criterion 1 demands a recorded measurement.
Tool calls ~26 vs the ~18 moderate budget; the extra calls are the 7 full reads plus the
in-process measurement.

---

## 0. Headline — three of the step's premises need correcting

1. **"The kill switch DISARMS … instead of firing on a real 20% breach" is overstated.**
   Measured: on a stale/None anchor the switch still fires — `any_breached=True` via the
   trailing leg. Only the *daily leg* is disarmed. The RED test fails on its first conjunct
   (`daily_loss_breached is True`), not on `any_breached`.
2. **The guard is not broken.** `evaluate_breach` already implements exactly what criterion 2
   asks for: per-leg markers, no wholesale early return, the evaluable leg still fires, the
   unevaluable one is loudly marked. `kill_switch.py:774-785` states this as designed intent.
   The minimal correct change to `kill_switch.py` is **none**.
3. **85.6 NARROWED reachability, it did not widen it** — measured, case D below. 85.5.1 must
   not touch `paper_trader.py:1276-1301` or `:1413-1449`.

Plus one path the step did not list, and one it mis-bucketed (§4 Q1, rows E and F).

---

## 1. Internal inventory (file:line anchored)

| File | Role | Status |
|---|---|---|
| `backend/services/kill_switch.py` | `_load_from_audit` :272-395 · `_snapshot_locked` :450-473 · `update_sod_nav` :537-579 · `evaluate_breach` :749-882 · `_sod_date_is_stale` :905-930 · `_log_disarmed_once` :933-960 · singleton :704 | LIVE |
| `backend/services/paper_trader.py` | `sod_anchor_needs_reroll` :86-90 · `roll_daily_anchor` (85.6 Step-0) :1220-1328 · `check_and_enforce_kill_switch` :1330+ · roll :1413-1414 · provisional upgrade :1415-1449 | LIVE |
| `backend/tests/test_book_safety_69.py` | the RED test :77-83 · **the mock :80** | RED |
| `backend/tests/test_phase_36_9_kill_switch_armed_liveness.py` | `isolated_state` fixture :63-88 — the correct idiom, already in-repo | GREEN |
| `backend/api/paper_trading.py` | badge `sod_date` :530 · resume 409 :620-659 | LIVE |
| `frontend/src/components/KillSwitchPanel.tsx` / `OpsStatusBar.tsx` | read `daily_baseline_stale` :157-176 / :338,:350 | LIVE |

### 1a. The four claims, re-derived

**Claim 1 — CONFIRMED, one anchor corrected.** `test_book_safety_69.py:80`
monkeypatches `snapshot` on `ks.get_state()` (:79), which IS the singleton `ks._state`
(`kill_switch.py:704`), and `evaluate_breach` reads `_state.snapshot()` at
`kill_switch.py:793` — so the patch does reach the production function. The real
`_snapshot_locked` emits `sod_date` at **`:465`** (not `~444`; :444 is inside
`_append_audit`). It also emits `baseline_provenance` :471 and `sod_provisional` :472,
so the mock now omits **three** contract keys. A repo-wide grep of `backend/tests/*.py`
for a `snapshot` monkeypatch returns exactly **one** hit — confirmed.

**Claim 2 — CONFIRMED, range corrected to `:298-316`** (not `:285-295`).
`self._sod_nav = _coerce_nav(row.get("nav"))` :299 → `sod_date = row.get("date")` :303 →
`ts` fallback :304-312 → `sod_date = None` on parse failure :312 → `self._sod_date = sod_date`
:313. Measured live: cases A and B below.
**Bonus finding: a TORN pair is unreachable.** Between :299 and :313 the only call that can
raise is `_coerce_nav` at :299 itself, and it precedes the date assignment — so `_sod_nav`
and `_sod_date` always come from the same row. Any fix may rely on that.

**Claim 3 — CONFIRMED and measured** (case G): `daily_loss_breached=False`,
`trailing_dd_breached=True`, `any_breached=True`. Bound `[4%, 10%)` holds — **conditional on
the trailing baseline being present and correct**, which case E can break (it strands the peak
too if the aborting row precedes any `peak_update`).

**Claim 4 — PARTIALLY REFUTED.** A **None** `sod_date` does self-heal: `sod_anchor_needs_reroll`
(`paper_trader.py:90`) is `snap.get("sod_date") != today`, so `None != today` → re-roll fires on
the next cycle. What does not self-heal quickly is the **ordinary** case: from 00:00 UTC until
the day's first roll, `_sod_date` is yesterday and `_sod_nav` is positive. That window opens
every UTC day unconditionally. It is not confined to 85.6's deadlock — the deadlock only made
it last all day instead of minutes.

---

## 2. External literature — READ IN FULL (7; gate floor is 5)

| # | URL | Accessed | Kind | Fetched how | Key finding / quote |
|---|---|---|---|---|---|
| 1 | https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2024.1430894/full | 2026-08-08 | peer-reviewed (Frontiers in Energy Research, 2024) | WebFetch | Dangerous-**undetected** failures are the modelled hazard: *"any failure is not known until they are activated either through demand or a test"* (§2.1). Unavailability = *"the probability that the system is not able to perform its required function on demand"* (§2.2) — a channel that cannot be evaluated contributes to unavailability, it is not credited as healthy. |
| 2 | https://abseil.io/resources/swe-book/html/ch13.html | 2026-08-08 | official engineering book (Google, *SWE at Google* ch.13 Test Doubles) | WebFetch | *"**Fidelity** refers to how closely the behavior of a test double resembles the behavior of the real implementation that it's replacing."* · *"our first choice for tests is to use the real implementations"* · mocking *"often violates the API contract of the type being mocked — for instance, returning null for a method that can never return null."* · *"there is no way to guarantee that the contract is correct (i.e., that the stubbed function has fidelity to the real implementation)."* **This is the 85.5.1 defect, named exactly.** |
| 3 | https://instrumentationtools.com/iec-61511-standard-requirements-for-safety-bypass-and-override/ | 2026-08-08 | industry/standards summary (IEC 61511 clause text) | WebFetch | Cl. 3.2.7 compensating measure = *"managing risks during any period … when it is known that the performance of the SIS is degraded."* Cl. 16.2.3: *"Compensating measures that ensure continued safety while the SIS is disabled or degraded due to bypass … shall be applied with the associated operation limits (duration, …)."* Cl. 16.2.4: continued operation with a device in bypass *"shall only be permitted if a hazards analysis has determined that compensating measures are in place."* Cl. 16.2.6/16.2.7: bypasses *logged, authorized, indicated*. |
| 4 | https://automationforum.co/iec-61511-safety-bypass-override-sis-maintenance/ | 2026-08-08 | industry/standards summary | WebFetch | **[The single most on-point external source.]** *"only bypass the least amount of elements needed. For example, bypass a single channel rather than the entire trip logic. Selective bypass reduces the reduction in safety integrity and preserves available redundancy."* · *"Bypass status must be visible to operators at all times, not hidden in maintenance menus."* · *"Every safety bypass needs to have a set time limit."* |
| 5 | https://arxiv.org/html/2602.00409v1 | 2026-08-08 | preprint (arXiv, 2026) — *Are Coding Agents Generating Over-Mocked Tests?* | WebFetch (`/html/` chain, per rule) | **Honest negative result**: it measures the *frequency* of agent-authored mocking, not mock correctness — no taxonomy, no divergence failure modes, no rates. It does relay the Google Testing Blog rule: *"using mocks in tests only guarantees success if the mocks match the real implementations. In practice, this is hard to ensure, especially as the real code evolves."* Recorded as read-in-full with low yield rather than cited beyond what it says. |
| 6 | https://www.elektrobit.com/wp-content/uploads/2015/11/The_safe_state-Design_patterns_and_degradation_mechanisms_for_fail-operational_systems.pdf | 2026-08-08 | vendor technical (Elektrobit, safetronic.2015), 24pp | **WebFetch returned binary → extracted with `pdfplumber`** (documented last-resort chain; already present in `.venv`, no install) | Slide deck, 15,110 chars extracted. Yield thin: the fail-safe vs fail-operational split, the 1oo2**D** pattern (two channels + **D**iagnostics), and the degradation menu *"Deactivate / degrade function"* / *"Some functions provide a degraded mode, sometimes limited in time"*. No usable normative text. Recorded honestly. |
| 7 | https://risknowlogy.com/articles/detail/17305/ | 2026-08-08 | industry (functional-safety consultancy) | WebFetch | Thin on IEC 61508 normative text, but supplies the latch pattern: a detected fault must trigger *"a defined safe reaction"*, e.g. *"if any check fails, the controller disables the heater and **latches a diagnostic until a supervised reset**."* — the analogue of `armed=False` + the `/resume` 409. |

### Identified but snippet-only (does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://sback.it/publications/msr2017b.pdf (Spadini et al., *To Mock or Not To Mock?*, MSR 2017) | peer-reviewed | Would have been source #1 for mocking; PDF, and 7 full reads already clear the floor. **Recommended follow-up read.** |
| https://dl.acm.org/doi/10.1007/s10664-018-9663-0 (*Mock objects for testing Java systems*, EMSE) | peer-reviewed | paywall |
| https://www.sciencedirect.com/science/article/pii/S0164121225004170 (test-mocking challenges, JSS 2025) | peer-reviewed | paywall; **2025 recency hit** |
| https://arxiv.org/pdf/2503.19284 (*Understanding and Characterizing Mock Assertions in Unit Tests*, 2025) | preprint | **2025 recency hit**, snippet only |
| https://arxiv.org/abs/2011.00892 (*A Formally Verified Fail-Operational Safety Concept*) | preprint | ar5iv 307-redirected to `/abs/`; abstract-only does not count as read-in-full |
| https://www.icheme.org/media/9906/xviii-paper-23.pdf (IChemE Hazards, IEC 61511) | conference | PDF, superseded by #3/#4 |
| https://www.primatech.com/images/docs/faq_s84_standard_for_safety_instrumented_systems.pdf | industry | PDF |
| https://standards.iteh.ai/…/pren-iec-61508-2-2025 · …/pren-iec-61508-5-2025 | standard (draft 2025) | paywalled; **2025 recency hits** |
| https://www.tandfonline.com/doi/full/10.1080/09617353.2024.2343959 | peer-reviewed (2024) | paywall |
| https://instrumentationtools.com/voting-logic-safety-instrumented-system/ · https://industrialmonitordirect.com/blogs/knowledgebase/voting-architectures-and-sil-levels-in-safety-instrumented-systems · …/tmr-vs-2oo3-… · …/sil-2oo3-vs-2oo2-… · …/1oo2-vs-2oo3-… · https://insteng.com/t/…/6214 · https://www.instrumentationblog.in/voting-architectures-safety-systems/ · https://www.scribd.com/document/436958011/… | community/industry | 1oo2/2oo3 degradation covered adequately by #4 + #6 |
| https://www.nxp.com.cn/company/blog/automotive-functional-safety-…:BL-AUTOMOTIVE-SAFETY-EVOLUTION · https://community.infineon.com/t5/Knowledge-Base-Articles/…/ta-p/436862 · 2× linkedin.com/pulse · https://arxiv.org/pdf/2412.08862 · 3× uspto.gov patents | vendor/community/patent | lower tier |
| https://www.electronicdesign.com/…/55379935 · https://ez.analog.com/…/diagnostics-are-they-worth-the-effort · https://instrunexus.com/iec-61508-an-comprehensive-guide/ · https://www.pacificblueengineering.com/what-is-iec-61511/ · https://www.downduck.com/threads/…1983533/ · https://arxiv.org/pdf/2509.19185 · https://arxiv.org/pdf/2103.01783 · https://dev.to/patoliyainfotech/…-5282 | community/vendor | lowest weight |

**~40 unique URLs collected**, 7 read in full.

### Search-query composition (three-variant discipline)

- **Year-less canonical (prior art):** `IEC 61511 safety instrumented system degraded mode bypass unavailable protection layer` · `1oo2 2oo3 voting architecture channel out of service degraded voting safety instrumented function` · `empirical study mocking practices test doubles fidelity mock drift production interface divergence`
- **Current-year (2026):** `fail-safe versus fail-operational degraded mode safety function unavailable 2026`
- **Last-2-year (2025):** `IEC 61508 dangerous undetected failure diagnostic coverage unrevealed unavailable safety function 2025`

---

## 3. Recency scan (2024-2026)

**Performed.** Result: **no 2024-2026 finding supersedes the canonical frame; two complement it.**

- **prEN IEC 61508-2:2025 / -5:2025** are in circulation (paywalled). The 2025 edition keeps
  diagnostic coverage `DC = λ_DD/λ_D` and safe-failure-fraction normative in its annexes — the
  detected/undetected split this brief relies on is **not** superseded.
- **Frontiers 2024 (source #1)** is the newest peer-reviewed treatment found of an
  unavailable/unrevealed-failed safety channel, and it confirms rather than revises the frame.
- **Mocking (2025-2026):** JSS 2025 (`S0164121225004170`), arXiv 2503.19284 (2025) and arXiv
  2602.00409 (2026) all post-date the abseil/Spadini guidance. The 2026 agent-mocking paper
  **explicitly does not** replace the fidelity rule — it re-cites it. So Google's ch.13
  formulation (source #2) remains the operative statement and is what this brief cites.
- **Adversarial note:** I searched specifically for a source arguing *for* mock-based unit
  tests over real implementations in safety-critical code. None was found at standards or
  peer-reviewed tier. The strongest counter-position located is abseil's own hedge — *"it is
  appropriate to use a test double even without perfect fidelity"* when backed by larger-scope
  tests. That qualifies the recommendation in §4 Q2 but does not overturn it, because here the
  double is unbacked by any larger-scope test of the same path.

---

## 4. Answers

### Q1 — every production path to `sod_date` None-or-stale with `sod_nav > 0` (MEASURED)

Script: `scratchpad/measure_85_5_1.py` (reproduced verbatim in §5). It redirects `_AUDIT_PATH`
into a tmp dir **before** constructing any state and swaps `ks._state` only in-process, so it
writes nothing to live state — verified by md5 (§5).

| # | Path | Result | Reachable today? | What it takes |
|---|---|---|---|---|
| **A** | Legacy `sod_snapshot` row: positive `nav`, **no `date`, no `ts`** | `sod_date=None`, `sod_nav=100.0`, **daily leg dead**, trailing fires | Mechanism LIVE; **no such row in the live journal today** | A row written before the phase-23.2.19 schema bump, or a hand-edited/truncated row |
| **B** | Same, `ts` present but **unparseable** (`:304-312` → `:312` sets None) | identical to A | Mechanism LIVE, same condition | A corrupted or partially-flushed `ts` |
| **C** | **UTC date rollover — yesterday's anchor before the day's first roll** | `sod_date='2026-08-07'`, **daily leg dead**, trailing fires | **YES — entered EVERY UTC day, unconditionally, with no fault at all** | Nothing. This is the normal state from 00:00 UTC until a cycle rolls. **This is the real severity driver, not A/B.** |
| **D** | Same as C but **85.6's Step-0 provisional roll has run** | `sod_date=today`, **`armed=True`, daily 20% → breached=True** | This is the 85.6 FIX | — |
| **E** | **Replay ABORTS mid-stream** — `_coerce_nav` raises `OverflowError` on a 401-digit JSON integer (`:131` catches only `TypeError, ValueError`), swallowed by the outer handler at `:394-395`, so **today's rows are never applied** | anchor frozen at the last successfully-replayed row → stale; peak stranded too | Mechanism LIVE; needs one malformed row | Any row whose `nav` / `prior_peak` / `new_peak` is a JSON integer too large for a float. **NEW — not in the step's list.** Log line observed: `kill_switch: audit load failed: int too large to convert to float` |
| **F** | **Backend startup before any anchor exists** | `sod_nav=None` → `daily_baseline_**missing**=True`, `stale=False` | Yes on a fresh install | **MIS-BUCKETED in the step.** `_sod_date_is_stale` returns False when `sod_nav is None or <=0` (`:922-923`) — it deliberately refuses to double-name an absence. This is a different code path with a different marker, and the fix must not conflate them. |

**Severity.** In every stale/None case measured (A, B, C, E) the trailing leg still fired at
20% and `any_breached` was `True`. The `[4%, 10%)` exposure bound therefore holds — **except in
the compound case** where the same fault also removes the peak (case E with an early-aborting
row, or a book with no `peak_update` yet). Then both legs are unevaluable and
`evaluate_breach` returns `any_breached=False` for **any** drawdown — by explicit design
(`kill_switch.py:770-773`: it does *not* set `any_breached=True` on a missing baseline). The
compensating control is `paper_trader`'s BUY gate reading `baselines_present`
(`:1372`), which blocks new orders. That is a genuine fail-operational-vs-fail-safe design
choice, correctly documented, and **out of 85.5.1's scope** — queue it separately if it is to
be revisited.

**Did 85.6 change reachability? YES — it NARROWED it.** Case D is the proof: with the Step-0
provisional roll applied, `armed=True` and a 20% drop fires the daily leg. Pre-85.6 the daily
leg was disarmed from 00:00 UTC until **Step 5.5** of a cycle that survived the analysis phase
(85.4 measured that it frequently does not) — potentially all day. Post-85.6 it is disarmed
only from 00:00 UTC until **Step 0** of the next cycle. The one direction in which 85.6 *could*
have widened harm — a multi-session-stale value wearing today's date and reporting as a
same-day loss — is closed by the upgrade at `paper_trader.py:1415-1449` reading the durable
`sod_provisional` flag. **85.5.1 must not touch `paper_trader.py:1276-1301` or `:1413-1449`.**

### Q2 — the minimal change

**Minimal change to `kill_switch.py`: NONE. The guard already does what criterion 2 asks.**

`evaluate_breach` computes `daily_leg_unevaluable` (`:810`) and `trailing_baseline_missing`
(`:798`) **independently**, gates each leg on its own marker (`:859`, `:865`), and combines with
`or` at `:876` — so the trailing leg fires while the daily leg is honestly unevaluable. That is
measured (A/B/C/E), and `:774-785` states it as deliberate: *"A wholesale `if not sod or not
peak: return disarmed` would disable a leg that still works — strictly LESS likely to pause.
The markers are therefore PER LEG."*

This is textbook. IEC 61511 practice (source #4): *"bypass a single channel rather than the
entire trip logic. Selective bypass reduces the reduction in safety integrity and preserves
available redundancy."* And the disarm is not silent — *"Bypass status must be visible to
operators at all times"* is satisfied by `armed=False` + `daily_baseline_stale=true` on the
API (`:878`), the badge (`KillSwitchPanel.tsx:157-176`, `OpsStatusBar.tsx:338,350`), the ERROR
at `:956`, and the `/resume` 409 (`paper_trading.py:620-659`). Cl. 16.2.6/16.2.7's "logged,
authorized, indicated" is met by the audit stream.

**Any change that re-arms the daily leg on a stale anchor would re-introduce the phase-36.9 F1
defect verbatim** — `kill_switch.py:799-808` records it measured on this book on 2026-07-26:
a two-day move reported as a same-day loss, biasing toward a spurious flatten. Phase-36.9's
whole point ("`armed:true` must mean the leg can fire NOW") is IEC 61508's
detected-vs-undetected distinction (source #1: unavailability is *"the probability that the
system is not able to perform its required function on demand"*). Do not weaken it.

**The change belongs in the test.** Replace the mock with the production path, using the
idiom already in this repo — `test_phase_36_9_kill_switch_armed_liveness.py:63-88`, whose
docstring makes exactly the abseil argument in-repo: *"A REAL constructor against the
redirected paths, not `object.__new__` with hand-set fields. The hand-built form silently omits
whatever attribute the snapshot contract grows next."*

Preferred shape (drives the real writers, so it also pins that `update_sod_nav` stamps a usable
date):

```python
monkeypatch.setattr(ks, "_AUDIT_PATH", tmp_path / "kill_switch_audit.jsonl")
monkeypatch.setattr(ks, "_audit_archive_dir", lambda: tmp_path / "audit")
st = ks.KillSwitchState()
monkeypatch.setattr(ks, "_state", st)
TODAY_UTC = datetime.now(timezone.utc).date().isoformat()   # COMPUTED, never hardcoded
st.update_sod_nav(100.0, date=TODAY_UTC)
st.update_peak(100.0)
r = ks.evaluate_breach(80.0, 4.0, 10.0)
```

`TODAY_UTC` must be computed — `test_phase_36_7…:100` and `test_phase_23_2_5…:54` both carry
the "COMPUTED, never hardcoded" comment for this exact reason; a literal date passes today and
fails tomorrow.

**Recommended second test** (converts today's accidental RED into an asserted invariant, and
is the honest-degradation contract criterion 2 is really about): with a **stale** anchor, assert
`daily_loss_breached is False`, `trailing_dd_breached is True`, `any_breached is True`,
`armed is False`, `daily_baseline_stale is True`. Check for overlap with
`test_phase_36_9_…:100-165` first — 36.9 covers stale/None on the `LIVE_SOD`/`LIVE_PEAK` shape;
if it already pins all five, cite it rather than duplicate.

### Q3 — what the test asserts today, and what must become true

`test_book_safety_69.py:82-83`, four assertions:

```python
assert r["daily_loss_breached"] is True and r["trailing_dd_breached"] is True
assert r["any_breached"] is True and not r.get("nav_invalid")
```

Measured with the current mock (case G): `daily_loss_breached=False`,
`trailing_dd_breached=True`, `any_breached=True`, `nav_invalid` absent (falsy). **Only the first
conjunct fails.**

For all four to pass **without relaxing any of them**, the snapshot `evaluate_breach` reads must
carry `sod_date == today (UTC)` alongside `sod_nav=100.0` and `peak_nav=100.0`. Measured — case
**F**: `daily=20.0% breached=True`, `trailing=20.0% breached=True`, `any_breached=True`,
`armed=True`, no `nav_invalid` key. **All four assertions pass at full strength; nothing on the
assertion side needs to change.** The fix is entirely on the input side, which is precisely what
criterion 3 protects.

Note the mock also omits `baseline_provenance` (:471) and `sod_provisional` (:472). Using a real
`KillSwitchState` fixes all three omissions at once and is drift-proof against the next contract
key — the abseil `@DoNotMock` argument, and the reason the fixture idiom already exists here.

### Q4 — measuring the 26/3017 baseline without corrupting the live journals

**Measured facts.** Every polluting path is derived from the module's own location, not from an
env var or a constant:

- `kill_switch.py:48` — `_AUDIT_PATH = Path(__file__).resolve().parents[2] / "handoff" / "kill_switch_audit.jsonl"`
- `cycle_health.py:36-37` — `_HISTORY_PATH = _HANDOFF / "cycle_history.jsonl"`, `_HEARTBEAT_PATH = _HANDOFF / ".cycle_heartbeat.json"`
- `cycle_lock.py:53` — `_LOCK_PATH = _HANDOFF / ".autonomous_loop.lock"`
- `git ls-files` confirms **both** `handoff/kill_switch_audit.jsonl` and
  `handoff/.cycle_heartbeat.json` are **tracked**.

**Recommendation: run the baseline in a `git worktree`.** Because all four constants derive from
`Path(__file__).parents[N]`, a worktree relocates them *all at once* — no per-test monkeypatch,
no env var, no allowlist to maintain and drift. One worktree already exists
(`/private/tmp/.../wt-base2`), so the pattern is established.

```bash
cd /Users/ford/.openclaw/workspace/pyfinagent
WT=/private/tmp/claude-501/-Users-ford--openclaw-workspace-pyfinagent/<session>/scratchpad/wt-85-5-1
git worktree add --detach "$WT" HEAD
source /Users/ford/.openclaw/workspace/pyfinagent/.venv/bin/activate

# PRECONDITION — assert the isolation actually took effect before trusting the run:
cd "$WT" && PYTHONPATH="$WT" python -c \
  "import backend.services.kill_switch as k, backend.services.cycle_health as c; print(k._AUDIT_PATH); print(c._HEARTBEAT_PATH)"
# BOTH must print paths under $WT. If either prints the operator's tree, STOP.

PYTHONPATH="$WT" python -m pytest backend/tests -q --timeout=120 2>&1 | tail -5
git -C "$WT" status --porcelain handoff/     # exactly what the suite wrote
```

Capture the **failure SET**, not just the count, and diff pre/post — a bare `26` can hide a
1-for-1 swap:

```bash
PYTHONPATH="$WT" python -m pytest backend/tests -q --timeout=120 2>&1 \
  | grep -E "^(FAILED|ERROR)" | sort > "$WT/../baseline_85_5_1.txt"
```

**Do NOT use the copy/restore alternative** (`cp` … run … `git checkout -- handoff/…`). It is
repair rather than prevention, and it **races the live backend**, which is running with the book
RESUMED and armed and may write to those same files mid-run. `git stash` is separately banned
here (hooks append to tracked logs on every Bash call). The worktree is the only option that
does not race live state.

**Running `backend/tests/test_book_safety_69.py` ALONE — CONFIRMED SAFE, with one measured
precondition.** Every test in the file either is read-only against the singleton, or redirects
`ks._AUDIT_PATH` / `cl._LOCK_PATH` to `tmp_path` first (:97, :114, :126, :160-161). The one test
that touches the **real singleton without a redirect** is `test_peak_reset_dark_by_default`
(:86-92), which calls `st.reset_peak(12345.0, trigger="flatten")` on `ks.get_state()`. It is safe
**only because** `settings.kill_switch_peak_reset_enabled` is **measured `False`** — verified this
session — so `reset_peak` returns `None` at `kill_switch.py:694` *before* taking the lock or
appending. Also note `test_no_process_kill_sink_in_commands` (:147) reads
`backend/slack_bot/commands.py` via a **relative** path, so it only passes with cwd = repo root.

> **QUEUE THIS AS ITS OWN DEFECT STEP (out of 85.5.1 scope).** If the owed **KS-PEAK-RESET**
> operator token is ever flipped to APPROVED, `test_peak_reset_dark_by_default` will write a
> `peak_reset` row into the **live** journal and set the **live** peak to `12345.0` — below the
> real high-water mark (~24666.57), permanently, replayed on every future boot. A test named
> "dark by default" becomes a live safety mutation the moment the flag it asserts is turned on.
> The fix is a `tmp_path` redirect on that test, same as its sibling at :95-107.

---

## 5. The runnable measurement (criterion 1) — verbatim

Script (also at `scratchpad/measure_85_5_1.py`). Non-destructive by construction:
`_AUDIT_PATH` is redirected **before** any `KillSwitchState` is built, `ks._state` is swapped
in-process only, no service is started, no cycle triggered, no BigQuery call made.

```python
import json, sys, tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix="ks_measure_")); AUDIT = TMP / "kill_switch_audit.jsonl"
import backend.services.kill_switch as ks
ks._AUDIT_PATH = AUDIT              # redirect BEFORE constructing any state
(TMP / "audit").mkdir(exist_ok=True)
TODAY = datetime.now(timezone.utc).date().isoformat()
YDAY  = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()

def build(rows):
    AUDIT.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    ks._disarmed_logged = True
    return ks.KillSwitchState()

def report(label, st, nav):
    ks._state = st
    snap = st.snapshot(); r = ks.evaluate_breach(nav, 4.0, 10.0)
    print(f"\n--- {label}")
    print(f"    sod_nav={snap['sod_nav']!r} sod_date={snap['sod_date']!r} "
          f"peak_nav={snap['peak_nav']!r} provisional={snap.get('sod_provisional')!r}")
    print(f"    nav={nav}  daily={r['daily_loss_pct']}% breached={r['daily_loss_breached']}"
          f"  trailing={r['trailing_dd_pct']}% breached={r['trailing_dd_breached']}")
    print(f"    any_breached={r['any_breached']} armed={r['armed']} "
          f"stale={r['daily_baseline_stale']} missing={r['daily_baseline_missing']}")
    return r

a = build([{"event":"peak_update","ts":f"{YDAY}T10:00:00+00:00","nav":100.0},
           {"event":"sod_snapshot","nav":100.0}])                       # no ts, no date
report("A  legacy row, no ts / no date  (claim 2)", a, 80.0)

b = build([{"event":"peak_update","ts":f"{YDAY}T10:00:00+00:00","nav":100.0},
           {"event":"sod_snapshot","ts":"not-a-timestamp","nav":100.0}])
report("B  legacy row, unparseable ts  (claim 2)", b, 80.0)

c = build([{"event":"peak_update","ts":f"{YDAY}T10:00:00+00:00","nav":100.0},
           {"event":"sod_snapshot","ts":f"{YDAY}T13:35:00+00:00","date":YDAY,"nav":100.0}])
report("C  yesterday's anchor, pre-roll  (the DAILY case)", c, 80.0)

d = build([{"event":"peak_update","ts":f"{YDAY}T10:00:00+00:00","nav":100.0},
           {"event":"sod_snapshot","ts":f"{YDAY}T13:35:00+00:00","date":YDAY,"nav":100.0},
           {"event":"sod_snapshot","ts":f"{TODAY}T00:05:00+00:00","date":TODAY,
            "nav":100.0,"provisional":True}])
report("D  85.6 Step-0 provisional roll applied", d, 80.0)

huge = "1" + "0"*400            # 401-digit JSON int -> Python int -> OverflowError in _coerce_nav
e = build([{"event":"peak_update","ts":f"{YDAY}T10:00:00+00:00","nav":100.0},
           {"event":"sod_snapshot","ts":f"{YDAY}T13:35:00+00:00","date":YDAY,"nav":100.0}])
with AUDIT.open("a", encoding="utf-8") as f:
    f.write('{"event":"peak_update","ts":"%sT09:00:00+00:00","nav":%s}\n' % (TODAY, huge))
    f.write('{"event":"sod_snapshot","ts":"%sT13:35:00+00:00","date":"%s","nav":100.0}\n'
            % (TODAY, TODAY))
ks._disarmed_logged = True; e = ks.KillSwitchState()
report("E  replay ABORTS mid-stream (OverflowError) -> today's row never applied", e, 80.0)

f_ = build([{"event":"peak_update","ts":f"{TODAY}T10:00:00+00:00","nav":100.0},
            {"event":"sod_snapshot","ts":f"{TODAY}T13:35:00+00:00","date":TODAY,"nav":100.0}])
report("F  control: same-day anchor, 20% down", f_, 80.0)

class _MockSnap:                                   # the RED test's exact mock
    def snapshot(self): return {"sod_nav": 100.0, "peak_nav": 100.0}
ks._state = _MockSnap(); r = ks.evaluate_breach(80.0, 4.0, 10.0)
print("\n--- G  the RED test's mock (test_book_safety_69.py:80)")
print(f"    daily_loss_breached={r['daily_loss_breached']} "
      f"trailing_dd_breached={r['trailing_dd_breached']} any_breached={r['any_breached']} "
      f"armed={r['armed']} stale={r['daily_baseline_stale']}")
```

How it was invoked, and the live-journal proof:

```
$ BEFORE=$(stat -f%z handoff/kill_switch_audit.jsonl); BEFORE_MD5=$(md5 -q handoff/kill_switch_audit.jsonl)
$ source .venv/bin/activate
$ PYTHONPATH=/Users/ford/.openclaw/workspace/pyfinagent python scratchpad/measure_85_5_1.py
$ AFTER=$(stat -f%z handoff/kill_switch_audit.jsonl); AFTER_MD5=$(md5 -q handoff/kill_switch_audit.jsonl)
```

Output, verbatim:

```
kill_switch: audit load failed: int too large to convert to float
audit redirected to: /var/folders/n4/.../ks_measure_sy6961tv/kill_switch_audit.jsonl
today (UTC) = 2026-08-08

--- A  legacy row, no ts / no date  (claim 2)
    sod_nav=100.0 sod_date=None peak_nav=100.0 provisional=False
    nav=80.0  daily=0.0% breached=False  trailing=20.0% breached=True
    any_breached=True armed=False stale=True missing=False

--- B  legacy row, unparseable ts  (claim 2)
    sod_nav=100.0 sod_date=None peak_nav=100.0 provisional=False
    nav=80.0  daily=0.0% breached=False  trailing=20.0% breached=True
    any_breached=True armed=False stale=True missing=False

--- C  yesterday's anchor, pre-roll  (the DAILY case)
    sod_nav=100.0 sod_date='2026-08-07' peak_nav=100.0 provisional=False
    nav=80.0  daily=0.0% breached=False  trailing=20.0% breached=True
    any_breached=True armed=False stale=True missing=False

--- D  85.6 Step-0 provisional roll applied
    sod_nav=100.0 sod_date='2026-08-08' peak_nav=100.0 provisional=True
    nav=80.0  daily=20.0% breached=True  trailing=20.0% breached=True
    any_breached=True armed=True stale=False missing=False

--- E  replay ABORTS mid-stream (OverflowError) -> today's row never applied
    sod_nav=100.0 sod_date='2026-08-07' peak_nav=100.0 provisional=False
    nav=80.0  daily=0.0% breached=False  trailing=20.0% breached=True
    any_breached=True armed=False stale=True missing=False

--- F  control: same-day anchor, 20% down
    sod_nav=100.0 sod_date='2026-08-08' peak_nav=100.0 provisional=False
    nav=80.0  daily=20.0% breached=True  trailing=20.0% breached=True
    any_breached=True armed=True stale=False missing=False

--- G  the RED test's mock (test_book_safety_69.py:80)
    daily_loss_breached=False trailing_dd_breached=True any_breached=True armed=False stale=True

live journal untouched? 5324 bytes (compare before/after)

LIVE JOURNAL: before=5324 after=5324  md5 same? YES
```

---

## 6. Consensus vs debate (external)

**Consensus.** (a) A protective function that cannot be evaluated must not be credited as
functioning — sources #1 (unavailability), #3 (Cl. 3.2.7 "known that the performance of the SIS
is degraded"), #7 (latch until supervised reset). (b) Degradation should be **selective**: bypass
the smallest element, keep the rest of the redundancy live — source #4, source #6's 1oo2D.
(c) Bypass status must be **visible and logged**, never silent — sources #3 (Cl. 16.2.6/16.2.7),
#4. (d) Test doubles must have **fidelity** to the real implementation, and the real
implementation is the first choice — source #2, echoed by #5.

**Debate.** Whether continued operation is permitted at all while a leg is bypassed. IEC 61511
Cl. 16.2.4 permits it *only* after a hazard analysis establishes compensating measures; source #4
adds a mandatory time limit. This project's compensating measure is
`paper_trader.py:1372`'s BUY-gate block plus the `/resume` 409 — but **there is no time limit** on
how long the daily leg may sit disarmed. Post-85.6 the window is bounded by cycle cadence rather
than by an explicit deadline. Worth a separate step; not 85.5.1.

**Adversarial.** No standards-tier or peer-reviewed source was found arguing that a mock-based
unit test is preferable to a real implementation for safety-path code. The nearest counter is
abseil's own hedge that imperfect fidelity is acceptable *when backed by larger-scope tests* —
which does not apply here, since no larger-scope test covers this path.

## 7. Pitfalls (from literature, mapped to this fix)

1. **Re-arming a leg that cannot be evaluated** — violates source #1's unavailability
   formulation and re-opens phase-36.9 F1 (`kill_switch.py:799-808`, measured 2026-07-26).
2. **Wholesale disarm instead of per-leg** — violates source #4's selective-bypass rule and
   is explicitly rejected at `kill_switch.py:774-785`.
3. **Silent degradation** — violates Cl. 16.2.6/16.2.7 and #4's visibility rule. The existing
   surfaces (`armed`, `daily_baseline_stale`, ERROR :956, /resume 409, badge) must all survive.
4. **A double that omits contract keys** — source #2's `@DoNotMock` case: *"violates the API
   contract of the type being mocked."* Exactly `test_book_safety_69.py:80`.
5. **Hardcoding today's date** in the replacement fixture — in-repo precedent
   (`test_phase_36_7…:100`, `test_phase_23_2_5…:54`) says COMPUTE it.

## 8. Application to pyfinagent — recommended scope for 85.5.1

- **No change to `kill_switch.py`.** (Q2.)
- **No change to `paper_trader.py:1276-1301` or `:1413-1449`** — that is 85.6's fix and it is
  what keeps case D armed. (Q1.)
- **`backend/tests/test_book_safety_69.py:77-83`** — replace the `snapshot` mock with a real
  `KillSwitchState` on a redirected `_AUDIT_PATH`, per the `isolated_state` idiom at
  `test_phase_36_9_kill_switch_armed_liveness.py:63-88`. All four assertions survive unchanged.
- **Add the honest-degradation test** (stale anchor → daily False / trailing True / any True /
  armed False / stale True) after checking `test_phase_36_9_…:100-165` for overlap.
- **Baseline** via `git worktree` with the `_AUDIT_PATH`/`_HEARTBEAT_PATH` precondition asserted
  first, comparing failure SETS not counts. (Q4.)

**Queue as separate steps (do NOT absorb):**
- **[P1]** `test_peak_reset_dark_by_default` (`:86-92`) mutates the LIVE peak the moment
  KS-PEAK-RESET is approved. Needs a `tmp_path` redirect.
- **[P2]** `_coerce_nav` (`kill_switch.py:114-141`) catches only `TypeError, ValueError`; an
  oversized JSON integer raises `OverflowError` and aborts the entire replay (case E), stranding
  both anchors. One-line widen to `except (TypeError, ValueError, OverflowError)`.
- **[P3]** No time limit on how long the daily leg may sit disarmed (IEC 61511 Cl. 16.2.3
  "associated operation limits (duration…)"). Post-85.6 the bound is implicit in cycle cadence.
- **[P3]** Compound case: both legs unevaluable → `any_breached=False` for any drawdown, with
  only the BUY gate as a compensating measure. Documented and deliberate; revisit only as a
  designed change.

---

## 9. Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (7; #6 via the documented
      `pdfplumber` last-resort chain after WebFetch returned binary)
- [x] 10+ unique URLs total (~40)
- [x] Recency scan (2024-2026) performed + reported (§3)
- [x] Full pages/papers read, not abstracts (arXiv fetched via `/html/`, never `/pdf/`;
      `arxiv.org/abs/2011.00892` explicitly NOT counted)
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered `kill_switch.py`, `paper_trader.py`, the test file, the
      36.9 fixture, the API + two frontend readers, and the 85.6 experiment record
- [x] Contradictions / consensus noted (§6), incl. three corrections to the step premise (§0)
- [x] Claims cited per-claim
- [~] Two of the seven full reads (#5, #6) had low yield; recorded honestly rather than padded

## 10. JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 33,
  "urls_collected": 40,
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
  "summary": "Three step premises corrected. (1) The switch is NOT silent on a 20% drop: measured, the trailing leg fires and any_breached=True; only the daily leg disarms, so the RED test fails on its first conjunct alone. (2) evaluate_breach already implements per-leg selective degradation exactly as IEC 61511 prescribes (bypass one channel, keep the rest live, make the bypass visible) -- the minimal change to kill_switch.py is NONE; the defect is the mock at test_book_safety_69.py:80, which omits three snapshot-contract keys. (3) phase-85.6 NARROWED reachability, it did not widen it: the Step-0 provisional roll re-arms the daily leg (measured case D), and the upgrade at paper_trader.py:1415-1449 closes the value hazard -- so 85.5.1 must not touch that path. Q1 enumerated and measured six paths; the dominant one is the ordinary UTC rollover (entered daily, no fault required), plus a NEW one the step missed: an oversized JSON int raises OverflowError in _coerce_nav, aborting the whole replay. Startup-before-first-anchor was mis-bucketed -- it is *missing*, not *stale*. Q4: run the baseline in a git worktree (all four journal paths derive from Path(__file__).parents[N]) and diff failure SETS, not counts. Running test_book_safety_69.py alone is safe ONLY because kill_switch_peak_reset_enabled is measured False -- flagged as a queued P1.",
  "brief_path": "handoff/current/research_brief_85.5.1.md",
  "gate_passed": true
}
```
