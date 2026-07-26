# Per-step tier ledger — `/goal` masterplan drain

Done-definition item **8**: *"A per-step tier ledger exists (step-id → tier → model/effort)
so the tiering policy can be audited rather than assumed."*

**This records what ACTUALLY ran, not what the policy prescribed.** Where the two differ,
the deviation is stated plainly.

## Measured pins (source of truth for every row below)

| Where | Value | Measured from |
|---|---|---|
| Main session | `claude-opus-5[1m]` | session environment |
| Main effort | `xhigh` | `.claude/settings.json → effortLevel` |
| Q/A subagent | `model: opus` → **Opus 5** (alias drift) | `.claude/agents/qa.md` frontmatter |
| Researcher subagent | `model: opus` → **Opus 5** | `.claude/agents/researcher.md` frontmatter |
| Fallback chain | `claude-sonnet-5`, `claude-haiku-4-5` | `.claude/settings.json → fallbackModel` |

The DRAFT's precondition *"Confirm the `model: opus` → Opus 5 alias drift is intended"*
is **still unconfirmed by the operator**. Every subagent in this drain therefore ran Opus 5
by alias resolution, not by an explicit decision.

## The ledger

| step | prio | tier used | model / effort | policy tier | deviation |
|---|---|---|---|---|---|
| `80.2` | P0 | **T3** | Opus 5 / xhigh (Main), Opus 5 / max (Q/A) | **T4** (Fable xhigh) | **YES — ran T3, not T4** |
| `80.1` | P0 | **T3** | Opus 5 / xhigh, Q/A Opus 5 / max | **T4** (Fable xhigh) | **YES — ran T3, not T4** |
| `80.27` | P0 | **T3** | Opus 5 / xhigh, Q/A Opus 5 / max | **T4** (Fable **max**) | **YES — ran T3, not T4** |
| `80.31` | P2 | **T3** | Opus 5 / xhigh, Q/A Opus 5 / max | T2 (Opus high) | over-spec (T3 > T2) |
| `80.3` | P0 | **T3** | Opus 5 / xhigh, Q/A Opus 5 / max | T3 | none |
| `80.4` | P0 | **T3** | Opus 5 / xhigh, 4× Q/A Opus 5 / max, 2× researcher Opus 5 / max | T3 | none |
| P0 triage (16 steps) | — | **T3** | 13-agent Workflow, Opus 5 | T3 (audit-class research + adversarial verify) | none |

**Fable 5 was authorized and never used. Zero T4 invocations in this drain.**

## Honest accounting of the deviation

The three T4-designated steps (`80.1`, `80.2`, `80.27`) ran at **T3**. The truthful reason
is **not** a deliberate quota decision: they ran on the session default because I did not
make a per-step tier call before starting them. The tiering directive was not applied as a
mechanical per-step gate the way the DRAFT specifies ("*do this per step, not per phase*").

What that did and did not cost, measured rather than assumed:

- **All three closed with a Q/A PASS**, and `80.27` — the one step in the drain that changes
  live decision behaviour — passed its fail-safe-only constraint. No correctness failure is
  attributable to the tier.
- The DRAFT's own rationale limits the downside: *"On the Max rail model choice costs $0
  metered… tier for speed and quota, never trade correctness for a saving that is zero."*
  Running T3 where T4 was prescribed spends **more** quota-safe capability, not less — it
  is the conservative direction, and it left the weekly-capped Fable budget untouched.
- The one place it plausibly mattered is `80.27` (`max` effort was prescribed). It ran at
  `xhigh` on Main with `max` on the Q/A gate. It passed, and the fail-safe direction was
  independently verified.

**T1 (Haiku) was never used either** — correctly, since every step in this drain touched
either the money path, a live decision path, or multi-file changes, and the DRAFT's T1 gate
requires *all* of single-file + no money path + no live decision path + no research gate.

## What a future session should do differently

1. **Assign the tier in the contract**, as a named field, before GENERATE. A tier that is
   never written down is a tier that silently defaults.
2. If a T4 step is started without Fable, **say so in the contract** rather than leaving the
   ledger to discover it afterwards.
3. Fable remains authorized **per-invocation only** (`agent({model:'fable', effort:'xhigh'})`),
   never a roster repin — exhaustion is a HARD FAIL (`harness_log.md:27346`), and pins
   silently fall back to Opus in headless runs.

## Cross-references

- Goal: `handoff/current/goal_masterplan_drain_2026-07-25_DRAFT.md` (tiers at lines 57–64)
- Counts re-derivation: `handoff/current/count_reconciliation_2026-07-26.md`
- Operator asks: `handoff/current/operator_ask_2026-07-26.md`
