# Research gate — step 86.75 (REPAIR RUN) — **PASSED**

**Run:** `wf_c1b10b08-07c` | 2 agents, 0 errors, **301,129 tokens**, 85 tool calls, 1,292s
**Brief:** `handoff/current/research_brief_86.75.md` (79,363 chars)

> **This gate should have run BEFORE the work.** It is a repair for the breach recorded in
> `PROTOCOL_BREACH_86.65.md`: 86.75's GENERATE happened with no gate and no contract. The
> prior work was handed to the researcher as **input to be challenged**, not as evidence,
> and I asked it explicitly to hunt for proof that change (2) was WRONG — because I authored
> that change and then relied on it all session.

## Recomputed result

| check | value |
|---|---|
| `sources_floor_ok` | **26 ≥ 5** |
| `urls_floor_ok` | **66 ≥ 10** |
| `urls_collected_corroborated` | 66 ≤ **69** distinct in brief |
| `all_26_claimed_sources_present_in_brief` | 0 missing |
| **`audit_class_dry_ok`** | **18 rounds, 2 dry, `dry: true`** |
| `self_report_disagreed` | **false** |
| `rail_dropped` | `null` |

---

## It challenged the work, as asked. Four findings.

### 1. Change (2) is directionally right — but I used the WEAKER argument

Deleting the absolute *"do NOT override"* **is** supported: arXiv 2606.19544 finds *"the most
reproducible judges are among the least valid"*; harmful self-preference at **86%**; and the
law-of-the-case doctrine is itself *"more complicated than that simple phrase."*

**But I justified it with self-attribution**, and arXiv 2603.04582 measures that as **NOT
fired by explicit labelling**. The real hazard is **authority anchoring (−14.95 pp)** — a
different mechanism than the one I cited.

### 2. The deletion shipped NEITHER safeguard the doctrine pairs with it

Law-of-the-case allows revisiting, but with (a) **burden on the party seeking the change**
and (b) **the override recorded**. I shipped neither.

**And (b) has no schema field — which is the exact unenforceability argument I used to
justify change (3)**, the rubric deletion. I applied one standard to a rule I removed and
its opposite to a rule I kept.

### 3. DECISIVE, and it indicts a change I made and have been relying on

**arXiv 2604.15224: consequence-framing makes judges LENIENT in 58 of 72 cells
(p < 0.001), ERRJ = 0.000, and it is INVISIBLE TO CHAIN-OF-THOUGHT.**

`qa.md` now **requires the Q/A to derive and state its own attempt number** — I added that.
**That is consequence-framing, built into the judge's own instructions.** And I amplified it
in spawn prompts, most explicitly on 86.64 cycle 3: *"ESCALATION BOUNDARY — READ THIS FIRST …
if you would issue a THIRD consecutive CONDITIONAL you MUST return FAIL instead."*

**Prescribed fix: compute the counter OUTSIDE the judge.** The Q/A should receive a verdict
request, not a running tally of its own consequences.

*(Note the direction: the bias is toward LENIENCY. So the 86.64 FAIL is not explained by
this — a leniency bias makes that verdict harder to reach, not easier. The exposure runs the
other way, on every PASS and CONDITIONAL I have collected.)*

### 4. The counter question, settled — and the independence gap measured

**Attempts bound the budget; verdicts bound the sequence.** They are different quantities and
neither substitutes: **5 of 35** ledger rows are `NO_VERDICT`, invisible to a verdict-keyed
count.

**`35/35` ledger rows are `recorded_by: main`** — the counter's input is authored entirely by
the party the rule constrains, which is 86.21 criterion 4's objection, now measured.

**Change (1) is already superseded by 86.21** — independent confirmation of the duplication
that 86.76 exists to fix. **Changes (3), (4), (5) stand.**

---

## What this does NOT license

- **It does not close 86.75.** The breach is the ORDER, and a gate run afterwards cannot
  repair ordering — it can only stop the next cycle from compounding it. A contract must
  still be written before any further GENERATE on this step.
- **No code or prompt was changed by this gate.** Finding 3 is filed, not applied:
  `qa.md` edits are already under an owed separation-of-duties review (three of mine).
- `coverage.dry` is `true` here on a genuine 18-round loop, unlike the non-audit gates
  earlier today where it was informational only.
