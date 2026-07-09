# Research Gate -- How-to (phase-4.16)

This is the HOW-TO guide for every researcher spawn. The REFERENCE
record lives in `ARCHITECTURE.md::Research Gate Discipline (phase-4.16)`.
The agent-facing prompt lives in `.claude/agents/researcher.md`.
Do not duplicate rules across the three files -- cross-link.

## Floor: at least 5 sources read in full

Every researcher spawn, at every tier, must fetch and read IN FULL
at least 5 sources via `WebFetch`. Search snippets do NOT count.
This applies to `simple`, `moderate`, and `complex` tiers alike;
the tier only sets the depth of analysis and the length of the
brief, not the 5 sources floor.

If fewer than 5 sources were fetched in full, the researcher MUST
return `gate_passed: false` and list what was attempted. Padding a
brief to mask an under-fetch is a protocol breach.

## Last-2-year recency scan (mandatory)

Every brief must include a dedicated "Recency scan (last 2 years)"
section reporting either:

- N new findings from the last-2-year window that complement or
  supersede older sources, OR
- No relevant new findings in the window.

The section must be present even when empty. An older canonical
source is still valuable; newer work just needs to be evaluated
against it.

## Search-query composition (mandatory)

Every research session MUST run at least three search-query variants
per topic to cover both the current frontier and the canonical prior
art. A single year-locked query is a protocol breach.

1. **Current-year frontier** -- append `2026` to the topic. Example:
   `"agent skill optimization 2026"`. Catches the latest published
   work in the current calendar year.
2. **Last-2-year window** -- append `2025` (and optionally `2024`).
   Used alongside #1 for the "Recency scan" section. Do NOT rely on
   this alone; see #3.
3. **Year-less canonical** -- the bare topic with NO year suffix.
   Example: `"agent skill optimization"`. This surfaces well-known
   prior-art (textbooks, the founding paper, classic blog posts) that
   a year-locked query will miss because search engines heavily bias
   to recent results when any year is present.

The brief must make the three-variant discipline visible: either by
listing the queries run in a short subsection, or by ensuring the
source table has a mix of current-year, last-2-year, and year-less
hits. If the topic is genuinely too new for year-less prior-art
(e.g., "phase-4.14.27 Anthropic effort param"), say so explicitly;
don't silently skip the year-less query.

Snippet-only hits from the year-less canonical search are valuable
evidence of "what prior art exists" even when not read in full, and
belong in the snippet-only table.

## Source quality hierarchy

1. Peer-reviewed (arXiv, ACM, IEEE, Journal of Finance)
2. Official docs (Anthropic, Google, IETF, NIST engineering blogs)
3. Authoritative blogs (OpenAI, DeepMind, named academic researchers)
4. Industry practitioner (Two Sigma, AQR, quant firms)
5. Community (StackOverflow, forums) -- lowest weight

Enforce this hierarchy in the fetched-in-full set. A brief with 5
community-tier URLs read in full does NOT clear the gate.

## PDF and arXiv paper fetching strategy (phase-29.7)

When a research source is an arXiv paper, do NOT call `WebFetch` on
the `/pdf/<id>` URL -- it returns binary data with no extractable
text and triggers the "Binary PDF, no text extracted" skip pattern.
Follow this priority chain:

### Step 1: arXiv native HTML (primary)

Try `https://arxiv.org/html/<arxiv_id>` first. This works for papers
submitted in TeX/LaTeX on or after December 1, 2023. Approximately
74% convert fully and 97% are at least partially readable per arXiv's
own statistics (source: `arxiv.org/html/2402.08954v1`). Partial
renders (TikZ errors, red markup) are still readable; proceed with
what renders rather than auto-falling back.

URL form: `https://arxiv.org/html/<arxiv_id>v<N>` (e.g.,
`https://arxiv.org/html/2402.08954v1`). If the version is unknown,
omit the version suffix and check the abstract page for the HTML
link below the PDF link.

### Step 2: ar5iv fallback (pre-Dec 2023 papers)

If the paper predates December 2023 and `/html/` returns 404 or a
red unsupported-package error, use
`https://ar5iv.labs.arxiv.org/html/<arxiv_id>`. ar5iv is an arXivLabs
project providing LaTeXML-rendered HTML for the full pre-snapshot
arXiv corpus. Confirmed working live via `arxiv:2312.01700` test
fetch on 2026-05-19 (renders equations, sections, references).

Note: ar5iv is a **snapshot** through end of April 2026, NOT a live
service. Papers submitted after April 2026 are not in ar5iv -- use
only native `arxiv.org/html/` for those.

### Step 3: pdfplumber (last resort for binary PDFs)

If both `/html/` and `ar5iv` fail, download the PDF and extract text
with `pdfplumber`:

1. Install in the researcher environment (NOT in
   `backend/requirements.txt` -- pdfplumber is a research-time
   convenience, not a project dependency):
   `pip install pdfplumber`  -- v0.11.9 as of Jan 5 2026; CPU-only;
   MIT license; deps `pdfminer.six` + `pillow` + `pypdfium2`.
2. Download: `curl -sL https://arxiv.org/pdf/<id> -o paper.pdf`.
3. Extract:
   ```python
   import pdfplumber
   with pdfplumber.open("paper.pdf") as pdf:
       text = "\n".join(p.extract_text() or "" for p in pdf.pages)
   ```

**Limitations:**
- Cannot handle scanned (image-only) PDFs -- older working papers
  pre-digital return empty text.
- Finance text F1 = 0.9568 (strong; benchmark
  `arXiv:2410.09871`).
- Scientific text F1 = 0.7644 (weaker; equations render
  symbolically, not semantically).
- Table extraction is weak (low recall) -- prefer HTML for
  table-heavy papers.

### Alternatives to pdfplumber (CPU-only / free only)

- **`pypdf`** -- pure-Python, no C deps, already an indirect dep via
  `paper-search-mcp`. Simpler than pdfplumber; adequate for plain
  finance papers; may produce spacing artifacts.
- **`PyMuPDF`** (`fitz`) -- faster, better scientific layout, but
  requires a C extension. Not in project deps.
- **`Nougat` / `marker-pdf`** -- best for equation-dense papers but
  GPU-required. Violates the CPU-only / free constraint; do not use.

### Never do

- Call `WebFetch` on `https://arxiv.org/pdf/<id>` as the primary
  attempt and then skip the paper because "Binary PDF, no text
  extracted." That is a protocol breach; the `/html/` chain above
  must be attempted first.
- Count a paper as "read in full" if only the abstract page was
  fetched and no HTML or PDF text was extracted.
- Add `pdfplumber` (or any PDF-extraction library) to
  `backend/requirements.txt` without owner sign-off -- the research
  use case does not justify shipping the dep in production builds.

## URL collection

Collect 10+ unique candidate URLs before pruning to the 5 sources
read-in-full set. The snippet-only set is the remaining URLs --
recorded in its own table so auditors can see what was evaluated
vs what was read.

## JSON envelope (always emit)

Every brief ends with this envelope, even when the caller does not
ask for it:

```json
{
  "tier": "simple|moderate|complex",
  "external_sources_read_in_full": 0,
  "snippet_only_sources": 0,
  "urls_collected": 0,
  "recency_scan_performed": false,
  "internal_files_inspected": 0,
  "gate_passed": false
}
```

`gate_passed: true` iff `external_sources_read_in_full >= 5` AND
`recency_scan_performed == true` AND every hard-blocker checklist
item is satisfied.

## Write-first discipline

The brief must be created early and written incrementally as sources are
read (never a single end-of-session flush); a session that cannot clear
the gate still leaves a partial brief + an honest `gate_passed: false`
envelope. The agent-facing directive lives in
`.claude/agents/researcher.md` ("Write-first (non-negotiable)") -- do not
duplicate the wording here.

## Handoff folder convention

The `handoff/` tree is strictly partitioned:

| Directory | Purpose | Writer |
|-----------|---------|--------|
| `handoff/current/` | In-flight step's files + `_templates/` | Main (cycle in progress) |
| `handoff/archive/phase-<sid>/` | Completed-step snapshots | `.claude/hooks/archive-handoff.sh` on masterplan status flip |
| `handoff/audit/` | Append-only JSONL audit streams | PreToolUse / ConfigChange / InstructionsLoaded hooks |
| `handoff/logs/` | Runtime process logs | Gitignored |

Invariants (verified by `scripts/housekeeping/verify_handoff_layout.py`):

- `handoff/current/` contains NO files belonging to `status=done`
  steps. Rolling top-level files (`contract.md`,
  `experiment_results.md`, `evaluator_critique.md`,
  `research_brief.md`) are allowed; they snapshot into the archive
  dir on each step close.
- No `*_audit.json*` at `handoff/` root (they live in `handoff/audit/`).
- No `*.log` at `handoff/` root (they live in `handoff/logs/`).

Backfill script: `scripts/housekeeping/backfill_handoff_archive.py`
(idempotent; safe to re-run).

## Cross-references

- `.claude/agents/researcher.md` -- agent prompt (system message).
- `ARCHITECTURE.md::Research Gate Discipline (phase-4.16)` -- MADR
  record.
- `CLAUDE.md` -- harness protocol (cycle-2 flow, session restart,
  stress-test doctrine). Authoritative for cycle semantics; defers
  to this file for research-gate mechanics.
- `docs/runbooks/per-step-protocol.md` -- operator runbook.
