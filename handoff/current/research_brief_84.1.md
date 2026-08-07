# Research Brief -- Step 84.1 (tier=moderate)

**Topic:** `scripts/housekeeping/audit_memory.py` resolves `[[wikilinks]]` by
FILENAME ONLY while memory-writing agents link by frontmatter `name:` slugs
(hyphens vs underscores) -> false breakage. Fix = resolve by basename-stem OR
frontmatter-name after normalization, three named report classes, exit-code
semantics per criterion 3.

**Status:** internal audit COMPLETE; external research in progress.
**Access date for all sources:** 2026-08-07.

---

## Part 1 -- INTERNAL AUDIT (re-measured TODAY, 2026-08-07)

### 1.1 The auditor, in full (69 lines)

`scripts/housekeeping/audit_memory.py` -- entire file read.

| Anchor | Behaviour |
|---|---|
| `:23-25` | `DEFAULT = ~/.claude/projects/-Users-ford--openclaw-workspace-pyfinagent/memory` -- the MAIN corpus, **outside the repo, 0 files git-tracked** (measured: `git ls-files` -> 0; 76 `.md` on disk) |
| `:30` | `if not idx_path.is_dir() and not idx_path.exists():` -- **latent bug**: `idx_path` is `root/MEMORY.md`, so the `is_dir()` leg is dead for a file and, were `MEMORY.md` ever a directory, the guard would PASS and `:35` would raise `IsADirectoryError`. Intent was almost certainly `root.is_dir()`. Also: a **missing `--dir`** falls into the same branch and prints `FAIL: no MEMORY.md in <root>` -> exit 1 (no distinct "corpus not found" signal). |
| `:33-34` | `files` = `root.glob("*.md")` minus `MEMORY.md`; `names` = **filenames only** (`p.name`, i.e. `foo.md`). Non-recursive. |
| `:36` | index pointers parsed as markdown links `](...md)`, NOT wikilinks |
| `:41-42` | `DANGLING POINTER` = index links a file that is gone |
| `:45-46` | `NO POINTER` = file exists, index never links it. **This is a set-difference on `names - linked`, i.e. exact filename** -- it is NOT affected by the wikilink normalization change. |
| `:50-51` | `MALFORMED FRONTMATTER` = naive `text.split("---")[1]` + `"type:" in` substring. No YAML parse. Passes both the top-level `type:` and the nested `metadata:/type:` schemas. |
| `:52-54` | **THE DEFECT**: `if f"{link}.md" not in names` -- exact-filename-only resolution. Regex is `\[\[([^\]]+)\]\]`. |
| `:56-63` | prints `memory files: N   pointers: M`, then `N PROBLEM(S):` + `  - <line>` each; returns 1 if any problem else 0 |
| `:66-69` | `sys.exit(audit(ap.parse_args().dir))` -- the `__main__` exit plumbing criterion 4 targets |

### 1.2 The convention clash is CONFIRMED at its source

The step's root-cause claim holds. The auto-memory system-prompt section that
every memory-writing agent receives says, verbatim:

> "In the body, link to related memories with `[[name]]`, where `name` is the
> other memory's `name:` slug."

and the frontmatter template it prescribes is `name: {{short-kebab-case-slug}}`
-- **kebab-case**, while files are written `snake_case.md`. Measured today,
frontmatter `name:` differs from the filename stem in **19/75 main, 36/37 qa,
76/90 researcher** files. There is **no `.claude/` doc** describing the
convention (grepped `.claude/rules`, `.claude/context`, `docs/runbooks`): the
only statement of it is the injected system prompt. So the criterion-1 fix is
correct and the convention itself is out of scope (correctly).

### 1.3 Today's re-measurement -- THE STEP'S FIGURES ARE STALE

Verbatim exits, `2026-08-07`, all three via `python3 scripts/housekeeping/audit_memory.py [--dir ...]`:

| Corpus | Step said (2026-08-06) | **Measured TODAY** | files | pointers | exit |
|---|---|---|---|---|---|
| main (`~/.claude/.../memory`) | 7 problems | **13 problems** | 75 | 75 | 1 |
| `.claude/agent-memory/qa` | 54 problems | **61 problems** | 37 | 37 | 1 |
| `.claude/agent-memory/researcher` | 71 problems (3 NO POINTER) | **105 problems (3 NO POINTER)** | **90** (was 71) | 87 | 1 |

The researcher corpus gained **19 files in one day** (4 untracked: `project_auth_latch_85_3`,
`project_credential_free_ci_lane_85_2`, `project_decision_input_integrity_61_2`,
`project_rail_failforward_72_0_2`). **Do not quote 7/54/71 in the contract.**
This is exactly why criterion 6 forbids count assertions on the real corpora.

### 1.4 Link taxonomy re-derived TODAY (denominator stated)

Classifier: `scratchpad/classify.py`. Rule = casefold, `-`->`_`, then strip ONE
leading prefix from `{feedback_, project_, reference_, user_}`; alias index =
`{norm(stem)} U {norm(frontmatter name)}` per file; cross-corpus = resolves in
one of the other two directories.

**Under the auditor's CURRENT regex, all three corpora, N = 406 links:**

| Class | Count | % of ALL 406 | % of the 176 NON-exact |
|---|---|---|---|
| exact same-dir | 230 | 56.7% | -- |
| same-dir normalized-only | 127 | 31.3% | **72.2%** |
| cross-corpus | 37 | 9.1% | **21.0%** |
| unresolvable | 12 | 3.0% | **6.8%** |

Step text (2026-08-06, N=320): 191 / 107 / 16 / 6 -> 82.9% / 12.4% / 4.7% of
129 failures. **The "83% same-directory" figure has decayed to 72.2% in one
day, and cross-corpus has nearly doubled its share (12.4% -> 21.0%)** because
the researcher corpus's newest files link outward to main-corpus feedback
memories. The step's *direction* (same-dir normalization is the dominant fix)
still holds; its *ratio* does not. State the denominator when quoting either.

### 1.5 NORMALIZATION COLLISIONS: measured ZERO today

Across all three corpora, **0 normalized keys map to more than one file**
(89 / 39 / 99 distinct alias keys for 75 / 37 / 90 files). So the tie-break
policy is currently unexercised -- but the criteria are silent on ties and the
alias index doubles the key population, so a policy must still be specified
(see Part 3).

### 1.6 FALSE-POSITIVE LINK CLASSES the criteria do not anticipate

The regex `\[\[([^\]]+)\]\]` at `:52` has no newline bound and no code-awareness.
Four distinct non-link artefacts are being counted as links **today**:

1. **Multi-line greedy matches (2).** `masterplan_state.md` and
   `project_phase84_memory_graph.md` each yield one match spanning many lines
   (`[[' and never mentions links in any form -- ...`). Killed by bounding the
   character class (`[^\[\]\n|#]{1,120}`).
2. **Prose mentions of the syntax itself (2).** `masterplan_state.md` contains
   the literal text `audit_memory.py resolves a [[link]] target by FILENAME ONLY`
   and `Classifying all 320 [[wikilinks]] across the three corpora` -- **the
   phase-84 step description, mirrored into memory**. Verified: the backticks
   present in `masterplan.json` are NOT present in the mirror, so a code-span
   strip does **not** remove them. `masterplan_state.md` is regenerated from
   `.claude/masterplan.json` by `.claude/hooks/masterplan-memory-sync.sh:41-42`
   on every masterplan write -- **this contaminant regenerates and cannot be
   fixed by editing a memory file** (which criterion 9 forbids anyway).
   **LIVE PROOF, observed during this session:** the mtime of
   `masterplan_state.md` moved from `1786113565` to `1786114067` (+502 s)
   between my first and last snapshot, with no write by me -- a concurrent
   masterplan edit regenerated the mirror mid-audit. Every other file in all
   three corpora was byte/mtime-identical across the session.
3. **A C++ attribute (1).** `project_macro_preload_refusal_82_13.md` contains
   `` `[[nodiscard]]` `` in a code span. Killed by stripping inline code.
4. **A masterplan step-id (1).** `project_phase4000_cc_rail_goal.md:45,58`
   references `[[4000.10]]` -- a step id, not a memory. Not killed by anything;
   it is a genuine unresolvable under the criteria's taxonomy.

### 1.7 FRONTMATTER PARSING HAZARD -- `yaml.safe_load` RAISES on real files

Measured: `yaml.safe_load` on the frontmatter block raises
`mapping values are not allowed here` on **2 of 75 main-corpus files** --
`feedback_log_last.md` (`description: Per-step ordering: research -> ...`) and
`project_phase75_state.md` (`description: ... 39.1 superseded; next: 75.20.1 ...`).
Cause: an unquoted scalar containing `: `. Both files then have **no extractable
`name:`** (2 more files with no `name:` at all). PyYAML 6.0.3 IS in `.venv`.

**Design consequence:** a naive `yaml.safe_load` implementation of the alias
lookup **crashes the auditor on the live main corpus**, directly violating
criterion 6 ("the run completes without raising"). Extract `name:` with a
line-anchored regex over the frontmatter block, or wrap the YAML parse in
`try/except Exception` and fall back to the regex.

Also measured: frontmatter `name:` is not always a slug --
`feedback_research_gate_min_three_sources.md` has
`name: Research gate requires >=3 sources fetched + read in full -- regardless of tier`
(a full sentence with unicode). Normalizing it is harmless (it simply never
matches a link) but the index must not assume slug shape.

### 1.8 CRITERION 8 -- read-only ALREADY TRUE today (verified, not assumed)

Snapshotted `(mtime, path)` for all 205 `.md` files across the three corpora,
ran the auditor 3x (once per corpus), re-snapshotted: **0 diff lines in all
three**. The auditor performs no write today (only `read_text` at `:35` and
`:49`). Criterion 8 is therefore a **regression guard**, not a repair.

### 1.9 CONSUMERS whose exit-code contract could change

`grep -rn "audit_memory"` over the whole repo (excluding `handoff/archive`):

| Consumer | Anchor | Contract |
|---|---|---|
| Masterplan step **84.2** | `.claude/masterplan.json:23060` | `verification.command` = `.venv/bin/python scripts/housekeeping/audit_memory.py --dir .claude/agent-memory/researcher`, criterion "exits 0". **Depends on 84.1 loosening the exit rule.** |
| Masterplan P2 memory-hygiene step | `.claude/masterplan.json:12409` | `verification.command` = bare `python scripts/housekeeping/audit_memory.py` (main corpus). Its criteria ask for NEW checks (index-line length, schema drift) -- not touched by 84.1. |
| Masterplan step at `:12395` | `.claude/masterplan.json:12395` | criterion "audit_memory.py is extended to detect a SECOND consumer pointing at a different MEMORY.md" -- future work, not 84.1. |
| **cron / launchd / hooks / CI** | -- | **NONE.** `crontab -l` has no memory/audit entry; no `~/Library/LaunchAgents/*` plist references it; no `.claude/hooks/*` invokes it; no GitHub workflow. The tool is **operator-invoked only**. |

So the criterion-3 exit-code change has exactly **one** live downstream
dependant (84.2) and it is *helped*, not broken, by the loosening.

### 1.10 POST-FIX EXIT PROJECTION (what the real corpora will do after 84.1)

Simulated the proposed resolver over today's corpora. Under the **current**
regex / under a **bounded** regex:

| Corpus | exact | normalized | cross-corpus | **UNRESOLVABLE** | NO POINTER | exit |
|---|---|---|---|---|---|---|
| main | 164 / 165 | 7 / 7 | 1 / 1 | **5 / 3** | 0 | 1 |
| qa | 10 / 10 | 55 / 55 | 4 / 4 | **2 / 2** | 0 | 1 |
| researcher | 56 / 56 | 65 / 65 | 32 / 32 | **5 / 5** | 3 | 1 |

**All three still exit 1 after 84.1.** That is CORRECT and consistent with the
criteria: criterion 3 is proven *by fixture* and criterion 6 forbids asserting
counts on the real corpora. 84.1's own verification command is the pytest file,
not an auditor run. 84.2 then has a bounded, achievable job on researcher
(3 NO POINTER + 5 unresolvable = 8 problems, down from 105).

Residual unresolvables after 84.1 (today's snapshot):
- main: `[[link]]`, `[[wikilinks]]` (mirror prose), `[[4000.10]]` (step id) [+2 multi-line artefacts if the regex is not bounded]
- qa: `[[measure-dont-assert]]` (main has `measure_dont_assert_claims` -- the `_claims` suffix defeats the fold), `[[killed-mutant-needs-behavioural-differential]]` (file is `feedback_killed_mutant_needs_differential_too.md`)
- researcher: `[[qa-guards-stop-one-seam-short]]`, `[[measure-dont-assert]]` x2, `[[gemini-lifecycle-pipeline-restoration-60-1]]`, `[[nodiscard]]`

Note the residue is *real* link rot (wrong slug remembered), which is exactly
what the tool should surface -- the signal is no longer buried under 127 false
positives.

---

## Part 2 -- EXTERNAL RESEARCH

### 2.0 Queries run (three-variant discipline)

| Variant | Query | Topic |
|---|---|---|
| year-less canonical | `Obsidian wikilink resolution shortest path aliases ambiguous link same filename` | 1 |
| year-less canonical | `Zettelkasten link rot maintaining note links renaming notes best practice` | 1 |
| current-year | `slug normalization case folding hyphen underscore collision pitfalls identifier 2026` | 2 |
| last-2-year | `agent memory graph audit broken links LLM memory rot 2025 2026` | 3 |

### 2.1 Read in full (9 -- counts toward the gate; all accessed 2026-08-07)

| URL | Kind | Fetched how | Key finding / verbatim |
|---|---|---|---|
| https://gist.github.com/dhpwd/9bb86c53b69cb63e09ccca42e3bf924c | doc (rules spec) | WebFetch | Resolution order, verbatim: *"exact filename (ignore `.md`, case-insensitive) -> normalised (treat spaces/`-`/`_` as equivalent) -> path if provided."* and *"If the front matter `aliases` or `title` fields exist, consider them for matching."* On unresolved: *"Do **not** create files... Report what is missing and ask for instruction."* **No tie-break rule specified for true ambiguity.** |
| https://obsidian.md/help/links | official doc | WebFetch (301 from help.obsidian.md) | Confirms `[[Note]]` and `[[Folder/Note]]` forms; *"A string which contains the following characters may not work as a link: `# \| ^ : %% [[ ]]`"* -> justifies bounding the link regex. Does **not** document duplicate-name resolution (gap). |
| https://community.obsidian.md/plugins/alias-linker | official plugin doc | WebFetch | **The precedence rule 84.1 needs**, verbatim: *"Keeps normal Obsidian behavior when `[[link]]` already matches a real file name"*; *"Obsidian tries its normal file-path resolution first, and only if that fails does Alias Linker look for notes whose aliases match the link text."* Ambiguity tie-break: *"the closest note (by folder distance from the source note) is chosen first"* + *"prefer explicit links like `[[file\|alias]]` whenever you need deterministic targeting."* |
| https://forum.obsidian.md/t/add-settings-to-control-link-resolution-mode/69560 | community (dev-adjacent) | WebFetch | Ambiguous basenames resolve **absolute-path-first**; users report it differs from VS Code/Typora which resolve same-folder-first. Confirms ambiguity is resolved by a **documented deterministic preference**, never by erroring. |
| https://devtoys.pro/blog/slugify-guide | industry doc | WebFetch | *"Two posts titled 'My Great Article' produce the same slug. Your database unique constraint will catch this -- but your application needs a strategy for resolving it before inserting."* Underscores -> hyphens; *"hyphens are the universal convention."* Collisions are **expected**, not exceptional. |
| https://zettelkasten.de/posts/the-hidden-problem-with-note-titles-as-links-and-how-to-fix-it/ | authoritative blog (canonical ZK) | WebFetch | Titles-as-identifiers cause rot: links *"feel strange or stop working if you change the title."* Fix = separate ID from title (stable ID + alias). Exactly our failure mode: the file stem is the ID, the frontmatter `name:` is the title, and agents link the title. |
| https://arxiv.org/html/2605.23723 (MemAudit) | preprint | WebFetch | Post-hoc audit of agent memory as a graph: *"edges reflect semantic relatedness and logical consistency"*; anomaly threshold *"CAS(m) > mu + 2sigma"*. Limitations: *"may produce false positives and remove useful memories, which could degrade performance."* -> **argues for report-don't-repair**, which 84.1 already adopts. |
| https://arxiv.org/html/2606.06036v1 (Graph Memory for LLM Agents) | preprint | WebFetch | **Negative finding**: the paper defers relational reasoning to retrieval and provides *"no quantified analysis of edge quality, missing bridge detection, or noise amplification"*. No prior art for our exact problem. |
| https://mem0.ai/blog/state-of-ai-agent-memory-2026 | industry report, **published 2026-08-07** | WebFetch | Staleness is *"a harder, open problem"*; the report explicitly does **not** provide memory-graph integrity metrics, link/edge quality measurements, or auditing tooling. |

### 2.2 Snippet-only (26 URLs; context, does NOT count toward the gate)

Obsidian/ZK: `obsibrain.com/blog/obsidian-linking...`, `forum.obsidian.md/t/.../28575`,
`forum.obsidian.md/t/link-notes-but-how/58831`, `forum.zettelkasten.de/discussion/1233`,
`forum.zettelkasten.de/discussion/3426`, `github.com/pvojtechovsky/obsidian-link-with-alias`,
`github.com/banisterious/obsidian-charted-roots/issues/538`,
`github.com/bitbonsai/mcpvault/pull/101`, `github.com/akosbalasko/zoottelkeeper`,
`github.com/cdaven/katalorg`, `neovim.substack.com/p/using-neovim-as-a-zettelkasten`,
`atlasworkspace.ai/blog/zettelkasten-method-guide`, `adobe.com/acrobat/resources/zettelkasten-method`,
`huggingface.co/spaces/anpigon/obsidian-qa-bot/...Import Zettelkasten notes.md`.
Slug: `devtoolhub.net/blog/url-slug-best-practices/`, `seo-architecture.com/.../slug-normalization-strategies/`,
`dev-smiths.org/text/slugify`, `utill.net/en/util/slug-generator/`,
`richdevtools.com/articles/web/url-slug-best-practices`, `grigora.co/tools/url-slug-generator/`,
`mbrenndoerfer.com/writing/text-normalization-unicode-nlp` (**attempted, HTTP 403**).
Agent memory: `openreview.net/pdf/061241ca...`, `openreview.net/forum?id=YPoHy6lgKP`,
`github.com/tfatykhov/awesome-agent-memory`, `arxiv.org/html/2607.19359v1`,
`kunalganglani.com/blog/ai-agent-memory-state-management`.

### 2.3 Recency scan (2024-2026) -- PERFORMED

Searched `agent memory graph audit broken links LLM memory rot 2025 2026` and
`slug normalization ... 2026`. **Result: 3 new findings, none superseding the
canonical Obsidian/Zettelkasten prior art, and one that strengthens 84.1's
design.** (a) MemAudit (arXiv 2605.23723, 2026) is the first post-hoc
agent-memory *graph* auditor in the literature -- but it audits for *poisoning*
via semantic contradiction, not for *referential* integrity, and its own
Limitations section warns it *"may produce false positives and remove useful
memories"*. (b) The mem0 State-of-AI-Agent-Memory report, **published today
(2026-08-07)**, states memory staleness is *"a harder, open problem"* and offers
no memory-graph integrity metric or auditing tooling. (c) arXiv 2606.06036v1
explicitly defers relational reasoning to retrieval and offers no edge-quality
analysis. **Conclusion: there is no 2024-2026 standard for referential-integrity
auditing of an agent memory graph.** The mature prior art is the note-taking
world (Obsidian resolution order + Zettelkasten stable-ID doctrine), which is
year-less canonical and still authoritative. 84.1 should copy Obsidian's
resolution ladder rather than invent one.

### 2.4 Consensus vs debate

**Consensus (4 independent sources).** Link resolution is a documented
**ladder**, not a single match: exact filename first (case-insensitive), then a
normalized form treating `-`/`_`/space as equivalent, then aliases, then path.
The gist spec, the Alias Linker plugin, and Obsidian's own behaviour all put
**filename before alias**. -> criterion 1's "EITHER stem OR frontmatter name" must
be an *ordered* ladder, not an unordered union, so the reported class is
deterministic.

**Consensus (2 sources).** An unresolved link is **reported, never
auto-repaired** (gist: *"Do not create files... Report what is missing"*;
MemAudit's false-positive limitation). -> the step's "detect, do not repair"
split is well-founded.

**Debate.** Nothing in the literature agrees on an ambiguity tie-break.
Obsidian uses absolute-path-first; Alias Linker uses folder-distance; VS
Code/Typora use same-folder-first; the gist spec is silent; slugify guidance says
the *application* must own a strategy. -> the criteria's silence on ties is a
real gap 84.1 must close with an explicit, documented rule.

### 2.5 Pitfalls from the literature (mapped to our anchors)

1. **Titles-as-IDs rot** (zettelkasten.de). Our stem is the ID, `name:` is the
   title, and agents link the title -> `audit_memory.py:52-54`.
2. **Normalization creates collisions** (devtoys). Folding case + `-`/`_` +
   *stripping the type prefix* means `feedback_x` and `project_x` collide.
   Measured today: **0 collisions** -- but the policy must exist before the
   first one appears.
3. **Auto-repair produces false positives** (MemAudit Limitations) -> stay
   read-only (criterion 8).
4. **Invalid link characters** (Obsidian docs: ``# | ^ : %% [[ ]]``) -> the
   unbounded `[^\]]+` regex at `:52` is why prose and C++ attributes are being
   counted as links (Part 1.6).

---

## Part 3 -- APPLICATION TO PYFINAGENT (the design)

See `contract_ready_notes` in the returned envelope for the full spec. Summary
of the load-bearing decisions, each traced to a source:

1. **Ordered ladder, not a union** (gist + Alias Linker, read in full):
   exact filename -> same-dir normalized (stem, then frontmatter name) ->
   cross-corpus normalized -> unresolvable.
2. **Ambiguity = report + name all candidates + DO NOT FAIL.** Literature
   offers no consensus tie-break; the one thing all four sources agree on is
   that ambiguity never *errors*. Failing on it would re-create exactly the
   defect being fixed (a link whose target exists reported as breakage).
3. **`name:` extraction by line-regex, not `yaml.safe_load`** -- measured: YAML
   raises on 2 real main-corpus files (Part 1.7), which would violate
   criterion 6.
4. **Bound the link regex** (`[^\[\]\n|#]{1,120}`) + strip fenced/inline code --
   removes 3 of today's 12 false unresolvables (Part 1.6). Justified by
   Obsidian's own invalid-character list.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (**9**)
- [x] 10+ unique URLs total (**35**)
- [x] Recency scan (2024-2026) performed + reported (S2.3)
- [x] Full pages read, not abstracts (all 9 fetched and content extracted; 2
      returned honest negative findings, recorded as such)
- [x] file:line anchors for every internal claim (Part 1)

Soft checks:
- [x] Internal exploration covered every relevant module (auditor, all 3
      corpora, mirror generator, every consumer, test prior art)
- [x] Contradictions / consensus noted (S2.4 -- incl. the tie-break debate)
- [x] All claims cited per-claim
- [!] Brief exceeds the `moderate` 700-word guidance. Deliberate: the caller
      specified the internal audit as "the bulk" and asked for a re-measurement
      plus a full design. Depth of *external* analysis is held at moderate.

---

## JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 9,
  "snippet_only_sources": 26,
  "urls_collected": 35,
  "recency_scan_performed": true,
  "internal_files_inspected": 12,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "Root cause confirmed at audit_memory.py:52-54 (exact-filename-only) against the system-prompt convention 'link by the other memory's name: slug' (kebab-case vs snake_case filenames). Re-measured today: main 13 problems / qa 61 / researcher 105 -- the step's 7/54/71 are STALE (researcher gained 19 files in a day). Re-derived taxonomy over N=406 links: 230 exact / 127 same-dir-normalized / 37 cross-corpus / 12 unresolvable, so the '83% same-directory' ratio has decayed to 72.2% of the 176 non-exact links and cross-corpus nearly doubled to 21.0%. Zero normalization collisions today, so the tie-break rule is a guard. Four hazards the criteria do not anticipate: yaml.safe_load RAISES on 2 real main-corpus files; the unbounded regex counts prose, a C++ [[nodiscard]] and a step-id as links; masterplan_state.md regenerates its own contamination from masterplan.json; and criterion 3's 'if and only if' literally disarms the DANGLING POINTER check the tool was built for. Read-only already verified (0 mtime diffs over 205 files). Only consumer is masterplan step 84.2 -- no cron, hook or CI.",
  "brief_path": "handoff/current/research_brief_84.1.md",
  "gate_passed": true
}
```

