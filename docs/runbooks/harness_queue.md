# Harness queue -- 2026-08-21

The ordered list of steps that must land before the harness is cheap enough to run the
money phases. Written by the operator-attended session on 2026-08-21 from a full read of the
masterplan (480 open steps; 128 in the harness phases 81/84/86/87/89/90). Every step below
carries a `sequencing_note` in `.claude/masterplan.json` pointing back here.

**Who runs this:** the ONE session that flips masterplan steps
(`project_concurrent_claude_sessions`). The other session does research/code/evidence and
hands over.

**The standing rule this queue exists to serve (operator, 2026-08-21):** do NOT spend Q/A
cycles on a step whose evidence apparatus has a known, filed defect -- every cycle re-buys the
defect (90.1 + 90.2 = 10 spawns, ~1.13M tokens, both FAIL with the product verified). Land the
apparatus fixes in group A first, then everything else costs what it should.

## State legend

| tag | meaning |
|---|---|
| `grade-built` | code + artifacts are on disk and committed; needs ONE Q/A (or one `--operator-extend` + one Q/A where the budget is at 5/5). Do not rebuild. |
| `build` | research gate -> contract -> GENERATE -> one Q/A. |
| `operator` | needs a decision the operator has to type; do not spend a cycle on it. |

## Collision groups -- never two in flight from the same group

- `qa-verdict.js`: 90.15, 90.16, 90.2, 90.14, 90.13, 86.98
- `attempt_gate.py`: 90.1, 90.3, 90.6, 89.3
- `qa-write-guard.sh`: 84.4, 86.64
- `verdict_gate.py`: 81.0, 81.3
- `hooks` (archive-handoff.sh / auto-commit-and-push.sh): 86.29, 86.15, 86.19, 90.7

## The queue

### A. Loop termination (the token burn) -- 13 steps, 7 already built

| # | step | state | what to do |
|---|---|---|---|
| 1 | 90.15 | grade-built (1/5 used) | LIVE hole on every Q/A spawn; gate + build landed in d4ff4d57. One Q/A. |
| 2 | 90.16 | build | residual of 90.15 (mutation seam). Same file -- do right after #1. |
| 3 | 90.2 | grade-built (5/5) | `python3 scripts/harness/attempt_gate.py --operator-extend 90.2` then ONE Q/A on the tree as it stands. 90.14 finished its criterion 6. |
| 4 | 90.14 | grade-built | shares 90.2's immutable command; grade right after #3. |
| 5 | 90.13 | build (small) | positional-claim detector false negative; same file, cheap. |
| 6 | 90.1 | grade-built (5/5) | `--operator-extend 90.1`; 90.12 closed the blocker that failed cycle 5 (commit f4188124). ONE Q/A. |
| 7 | 90.12 | grade-built | ERROR discriminator now reads the type (36ca16e1); retro gate found UnboundLocalError and it was added. ONE Q/A. |
| 8 | 90.5 | build | rebuild the verdict ledger from run records; prereq for 90.3 (absorbs 87.3). |
| 9 | 90.3 | built-partial | gate PASSED; contract/results/live_check + mutation matrix committed (ae29f25a, "DEFAULT-OFF, ungraded"). Finish + ONE Q/A. After 90.1 and 90.5. |
| 10 | 90.6 | build | research-gate launches must not consume the Q/A attempt ceiling. After 90.1. |
| 11 | 89.3 | build | Agent-tool spawns get a correlation contract so the attempt gate sees them. |
| 12 | 90.9 | grade-built | criterion-shape classifier built and self-verified (4165e67d). A Q/A was IN FLIGHT at 12:00 on 2026-08-21 (verdict_wip_90.9__20260821T095922Z.md) -- read its outcome before spawning another. |
| 13 | 86.98 | build + operator | verdict as a function of the criteria; criterion 7 needs an operator sign-off -- ask for it in the contract. |

### B. Diagnostic before any 87.x remediation -- 1 step

| 14 | 87.6 | build (cheap) | frozen-anchor replay of today's qa.md over >=8 pre-08-09 PASS steps. Descriptive only. |

### C. Evaluator-independence and gate holes -- 5 steps

| 15 | 84.4 | build | qa-write-guard matches only the literal name `qa`; every `qa-<step>-c<n>` spawn is ungated. |
| 16 | 86.64 | build | same guard, the Bash write channel. After #15. |
| 17 | 81.0 | grade-built | all four changes verified on disk 2026-08-21. ONE Q/A. |
| 18 | 81.3 | build | subprocess-level test of the verdict-gate CLI seam (M10/M11 must go red). |
| 19 | 75.5.10 | build | live_check_gate.py must READ the artifact, not stat it. |

### D. Stop re-discovering filed work -- 3 steps

| 20 | 86.76 | build | research gate gets a prior-art leg over the masterplan's own defect register. |
| 21 | 87.2 | build | wire `scripts/qa/pre_spawn_gate.py` before every Q/A launch (read-only, safe to wire). |
| 22 | 87.1 | build | evidence sentence generated FROM the capture. |

### E. Unattended-run stoppers -- 2 steps

| 23 | 86.57 | build | session-stop gate demanded an unsatisfiable condition 7x and read a cached goal. |
| 24 | 86.8 | measure | crossSessionInbound parks a bypassPermissions session -- measure, then decide. |

### F. Two-session safety -- 6 steps

| 25 | 86.43 | build | write-first at a step-scoped path lets session B truncate session A's live brief. |
| 26 | 86.15 | build | auto-commit `git add -A` cross-attribution (absorbs 78.15). |
| 27 | 86.18 | build | masterplan has no canonical serialization -> whole-file rewrites + lost updates. The audit measured it: `json.dumps(indent=2, ensure_ascii=True) + "\n"` is byte-exact today. |
| 28 | 86.29 | build | archive-handoff.sh snapshots the wrong step's files (absorbs 36.29). |
| 29-30 | 86.19 + 90.7 | build, as a pair | duplicate step ids (4 exact + 5 prefix-normalised) and the two consumers. |

### G. Test-suite trust (Q/A runs the suite on every money step) -- 8 steps

| 31 | 86.118 | build | 8 red tests remain (was 19); the suite cannot detect a regression until it is green. |
| 32 | 86.125 | build | hermetic settings (absorbs 36.30, 86.48). Traps are in its notes. |
| 33 | 86.119 | build | install pytest-randomly; WITH or AFTER 86.118. |
| 34 | 86.124 | build | the four judgement-blocked reds. |
| 35 | 86.126 | build -- MONEY | swap SELL 1-vs-2 (absorbs 86.51). Establish which count is right first. |
| 36 | 86.123 | build | outcome_write_schema two tests (absorbs 86.52). |
| 37 | 36.28 | build | kill-switch injection seam class (absorbs 82.36). Misfiled in phase-83; it is suite work. |
| 38 | 78.19 | build | eslint ignores for `.next-*` (absorbs 75.6.1, 61.3.3). |

Minimum if time is short: #1-#13 plus #15, #11, #20, #21 (the 14 that directly stop the burn).

## Operator decisions queued (type the token; do not spend cycles)

- `--operator-extend` for 90.1 and 90.2 (one Q/A each; both products verified).
- 86.98 criterion 7 sign-off.
- 72.0.2 vs 91.5 policy conflict: fail-forward to a tier that can produce a GENUINE analysis
  is allowed; if none can, SKIP -- never persist a fabricated row. Confirm or amend; record in
  91.5's contract.
- Operator-close candidates, product shipped and criteria MET per their last Q/A, parked on
  evidence findings: **86.9, 86.44, 86.91, 86.97**. Close via the escalation file or grant
  one extend each.
- 85.1 migrate-or-retire for the nightly autoresearch. Seven steps are annotated
  `GATED ON 85.1` (86.66, 86.80, 86.121, 76.9.1, 76.9.4, 76.9.5, 82.49) and are moot on retire.

## What the audit already did (2026-08-21, commit on main)

23 supersessions, each with the measurement in `superseded_reason`:
86.51->86.126, 86.52->86.123, 86.56->86.110, 36.29->86.29, 75.11.3->86.50, 36.30->86.125,
86.48->86.118+86.125, 86.112->86.118, 82.36->36.28, 75.6.1->78.19, 61.3.3->78.19,
78.15->86.15, 4000.6->86.39, 73.2.1->86.35, 86.106->89.2, 86.113->75.5.6, 80.44->78.18,
87.3->90.5, 88.4->86.69, 86.5->(its triage output), 86.47->86.69, 4000.10->86.41,
86.45->86.85+86.78.

Verified before superseding: 86.112 and 86.48's tests pass (16/16); the quant crash site
carries the 86.41 guard; `qa-verdict.js:638` skips NO_VERDICT; 81.0's four changes are on disk.

## Not in the queue, on purpose

- **Phase-86 money-path steps** (37 of its 90): 86.69 -> 86.74 -> 86.108 -> 86.59/86.60/86.116/86.117,
  then 86.120/86.122/86.114/86.87/86.88, 86.4, 86.9/86.53/86.54, 86.35, 86.38, 86.40, 86.62,
  86.63, 86.67, 86.111, 86.7. They are the money queue; run them on the harness AFTER group A.
- **Evidence hygiene** (P3, zero verdict bearing): 86.23, 86.30, 86.42, 86.46, 86.55, 86.89,
  86.93, 86.95, 86.99, 86.100, 86.101, 86.102, 86.107, 86.115, 86.83, 87.4, 87.5, 87.7, 87.8,
  87.11, 89.2, 89.4, 89.5, 89.6, 89.7, 90.4, 90.8, 90.10, 90.11, 84.2, 84.3, 84.5, 84.1.1,
  86.65, 86.16, 86.73, 86.82, 86.70, 75.5.7, 75.5.9, 75.5.14, 75.11.2, 78.14, 78.18, 86.50,
  62.1.3, 82.30, 86.13, 86.10, 86.11.
- **Changelog-hook cluster** (86.77, 86.91, 86.97, 86.103, 86.104, 86.94): six steps on a
  version-bump hook. Last.

## Families (read the siblings before building any one of them)

- Fabricated fallback rows: 86.69 (anchor), 61.2 item 1, 91.5, 86.122, 88.2, 91.30, 86.114,
  86.87, 86.88. Precedent for a class guard: 86.63.
- Rail exhaustion / degraded visibility: 86.120, 91.5, 91.6, 91.2, 86.14, 91.3, 86.38(c).
- Brittle pins: 78.18 (absorbs 80.44), 86.50, 75.5.7 -- one pass.
