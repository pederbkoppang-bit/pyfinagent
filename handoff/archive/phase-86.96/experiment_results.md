# experiment_results -- step 86.96 (GENERATE, 2026-08-17)

Contract: `contract_86.96.md`. Research gate: PASSED (`wf_9e9ef2b7-70b`,
recomputed; brief COMPLETE, 8 sources in full, 30 URLs). Everything below was
executed this session; commands sit beside their outputs in
`live_check_86.96.md`.

## 1. Reproduction (criterion 1) -- deterministic, verbatim

The two production kills are on disk with their verbatim error
(`error` field of run records `wf_1f6b0398-020` and `wf_88302c2a-d20`,
2026-08-16T09:14/09:15Z): `qa-verdict: args are PRESENT but not parseable as
JSON (typeof=string ...) -- pass a plain object (or valid JSON) carrying
step_id, or omit args entirely for a dry run.` Reproduced by execution: the
shipped `classifyArgs` sliced from `.claude/workflows/qa-verdict.js:75-99` and
driven on both stored payloads THROWS on every run (2 runs x 2 payloads, 4/4 --
deterministic, not intermittent). Cross-parser control: `python3 json.loads`
fails the same payloads at pos 4939 / 5536 ("Expecting ',' delimiter"); node
`JSON.parse` fails with "Expected ',' or ']' after array element" -- node's
message already names the ARRAY, consistent with the bisection.

## 2. Bisection (criterion 2) -- the minimal failing input is ONE character

At the exact failure offsets, the char is `}` in the context `..."},"known`:

- SUBSTITUTION `}` -> `]` at pos 4939 / 5536: **both payloads PARSE**.
- INSERTION of `]` before the `}`: **both FAIL** ("Unexpected non-whitespace
  character after JSON") -- the bracket is WRONG, not missing, which rules out
  truncation/boundary-shear for these two.
- **SIZE tested and ruled out**: an equal-length (6,088-char) valid payload
  parses.
- **ESCAPED QUOTES tested and ruled out**: four named production string
  payloads containing `\"` all parse (`wf_f7d084d8-76c`, `wf_091e2312-0d8`,
  `wf_ea569c91-52a`, `wf_4575b02b-eb0`; escaped-quote counts 2-8).

Both rulings answer the criterion's cite of the cycle-2 payloads that parsed.
Mechanism (research brief section 6): idiom priming -- the dict-valued field
immediately before `extra.judge_these_specifically` closes with `"},` and the
list reused that close; identical key and direction on two independent spawns.

## 3. Localisation (criterion 3) -- evidence PER layer, not by elimination

| Layer | Evidence | Verdict |
|---|---|---|
| Caller serialisation | the one-char wrong bracket at a hand-composed position; identical shape on two independent spawns; the class census (section 5) shows failures ONLY among string args | **GUILTY** |
| Workflow marshalling | run-record `args` sha1-identical to the delivered payloads (brief section 2: `f678987c...` / `151e86ef...`); my census reads the same bytes from the records | **INNOCENT** |
| Script `JSON.parse` | node and python independently refuse the same bytes at the same positions; the refusal is the documented fail-fast, and 390 production string payloads parse through the identical path | **INNOCENT (and correct)** |

## 4. Byte-verbatim round-trip (criterion 4) -- executed on BOTH shapes

`verify_prompt_render_86_90.mjs` section `[7]` (new): an adversarial criteria
array -- backticks, double quotes, a quoted `bash -c` command, raw newlines,
non-ASCII (`æ ø å ü ß`, `評価基準`, `naïve café`), and the `"},` bracket idiom
-- driven through the WHOLE shipped script with the runtime stubbed, as an
OBJECT and as a JSON STRING. Every criterion arrives byte-verbatim, in order,
and **the two shapes render IDENTICAL prompt bytes**. 18 new checks, all green.

## 5. Census (criterion 5) -- re-derived at execution time

Population rule: every parseable `*/workflows/wf_*.json` under the project's
`~/.claude/projects` tree; args classified OBJECT / STRING-parses /
STRING-fails / ABSENT by `json.loads`. Measured 2026-08-17 over **585
records: 95 objects, 390 strings-that-parse, 4 string failures, 96 absent** --
strings are 394/489 = 80.6% of arg-carrying launches (the brief measured 81.4%
on a 580-record corpus; the corpus moves daily, both figures carry their rule).
The failure class is FOUR events, all enumerated with run ids in the
live_check: the two 2026-08-16 bracket kills plus two earlier research-gate
string failures -- one "Unterminated string" (pos 123 / len 3,201) and one
failing at pos == len 4,911, i.e. a TRUNCATED payload. Zero failures among 95
object launches.

## 6. Regression guard (criterion 6) -- mutation-tested, control green first

Section `[7]`'s guard would go RED if: a verbatim-critical payload silently
fails to round-trip (per-criterion byte assertions + object/string identity);
or a malformed payload stops dying loud (the minimal bracket-defect fixture
must throw `not parseable as JSON` and spawn NOTHING). Two mutation cells, each
CONTROL-clean first, both KILLED: `7-classifyArgs-repairs-instead-of-refusing`
(the catch silently substitutes a recovered object -- the repair anti-pattern)
and `7-render-mangles-quotes-in-transit` (strings altered in transit -- the
step name's recorded workaround harm, as a permanent cell).

## 7. Verdict semantics (criterion 7)

No behavioural change to either workflow script in this step. The only planned
edit to `qa-verdict.js` is a COMMENT at `classifyArgs` recording the four-event
class, the idiom-priming mechanism, and the object-first launch contract --
staged for the closure commit (deferred while two evaluators were mid-flight on
the file) and verified comment-only by `git diff` there. The immutable command
(`node --check` both scripts) and the full checker family stay green:
prompt-render 113/113, args-boundary 96/96, research-gate-workflow 124/124.

## Honest limits

- The idiom-priming MECHANISM is an inference from two identically-shaped
  instances plus the literature; the bisection facts (one wrong bracket,
  substitution-repairs) are measured. Stated as such.
- Caller behaviour (an LLM composing a string) cannot be mechanically
  constrained from inside this repo. What IS pinned: the object path and the
  valid-string path both transmit byte-verbatim (so no workaround is ever
  needed), and any future silent-repair or in-transit mangle turns the guard
  RED. The object-first guidance lands as the classifyArgs comment.
- The 4 historical failures cost one retry each (fail-fast at dispatch, 0
  tokens); the pre-hardening silent event (brief section 5: `audit_class:true`
  dropped by a blind run) predates the 86.17/86.37 hardening that now makes
  blind runs loud.
