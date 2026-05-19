# Contract — phase-29.7 (arXiv-HTML precedence + pdfplumber rules)

**Step ID:** phase-29.7
**Date:** 2026-05-19
**Author:** Main (overnight execution)
**Tier:** complex

---

## Research-gate summary

| Metric | Value |
|---|---|
| Sources read in full | 8 |
| Snippet-only | 12 |
| URLs collected | 20 |
| Recency scan | DONE — pdfplumber 0.11.9 Jan 5 2026; ar5iv corpus April 2026 cutoff |
| `gate_passed` | true |

**Brief:** `handoff/current/research_brief.md` (this cycle).

**Headline findings:**
1. `arxiv.org/html/<id>v<N>` works for ~74% (full) / ~97% (partial) of TeX-source papers submitted ≥ Dec 1 2023.
2. `ar5iv.labs.arxiv.org/html/<id>` is the snapshot fallback through April 2026 (covers older papers).
3. pdfplumber 0.11.9 (Jan 5 2026) is the CPU-only last-resort PDF→text extractor. Finance F1=0.9568, Scientific F1=0.7644. **NOT in `backend/requirements.txt`** — must be installed researcher-environment-only.
4. Existing `.claude/rules/research-gate.md` (134 lines) has no PDF/HTML-fetch section. Insertion point: after "Source quality hierarchy", before "URL collection".

---

## Audit-basis (from phase-29.0)

phase-29.0 audit §1.3-1.4: researcher reflexively WebFetches `/pdf/<id>` and skips on "Binary PDF, no text extracted". Need arXiv-HTML precedence rule + pdfplumber CPU-only fallback. This cycle delivers both.

---

## Verbatim immutable success criteria

1. `research_gate_md_contains_arxiv_html_precedence_rule` — new `## PDF and arXiv paper fetching strategy` section present in `.claude/rules/research-gate.md`.
2. `arxiv_org_html_url_documented` — section names the `https://arxiv.org/html/<id>v<N>` URL pattern.
3. `ar5iv_fallback_documented` — section names the `https://ar5iv.labs.arxiv.org/html/<id>` fallback for pre-Dec-2023 papers.
4. `pdfplumber_last_resort_documented` — section gives the `pip install pdfplumber` + `pdfplumber.open(...).pages[n].extract_text()` recipe.
5. `pdfplumber_researcher_only_not_project_dep` — section explicitly states pdfplumber is a researcher-environment install, NOT a project dependency.
6. `never_do_block_present` — section ends with a "Never do" block forbidding raw `/pdf/<id>` WebFetch as the primary attempt.
7. `internal_research_gate_md_cross_link_updated` — the existing "Cross-references" section at the bottom of research-gate.md updated to reflect this new section (optional cross-link to CLAUDE.md harness protocol).

**Verification command:**
```bash
grep -q '## PDF and arXiv paper fetching strategy' .claude/rules/research-gate.md && \
grep -q 'arxiv.org/html' .claude/rules/research-gate.md && \
grep -q 'ar5iv.labs.arxiv.org' .claude/rules/research-gate.md && \
grep -q 'pdfplumber' .claude/rules/research-gate.md && \
grep -q 'NOT a project dependency' .claude/rules/research-gate.md && \
grep -q 'Never do' .claude/rules/research-gate.md
```

**`verification.live_check`** (R-1 gate): `"live_check_29.7.md captures (a) the pre-insertion vs post-insertion line counts of research-gate.md, (b) verbatim arxiv.org/html and ar5iv.labs URL examples that work today (one each tested via WebFetch), (c) pdfplumber smoke test (pip install + 5-line script extract a known PDF)."`

---

## Plan

1. DONE — Spawn researcher complex.
2. DONE — Write contract.
3. NEXT — GENERATE:
   - EDIT 1: Insert new section `## PDF and arXiv paper fetching strategy` into `.claude/rules/research-gate.md` after the "Source quality hierarchy" section.
   - EDIT 2: Update masterplan.json 29.7 entry's verification.command + success_criteria + live_check.
   - EDIT 3: Write experiment_results.md (verbatim diff section + verification output).
   - EDIT 4: Write live_check_29.7.md.
4. Spawn `qa` once. Circuit breaker: 2 fresh-qa.
5. Append log → flip masterplan → commit.

---

## Out of scope

- Adding pdfplumber to `backend/requirements.txt` (research findings explicitly say NOT to).
- Editing `backend/.env.example` (permission-blocked).
- Implementing the rule in researcher.md frontmatter (the rule is reference content; researcher.md already imports rules/research-gate.md by convention).
