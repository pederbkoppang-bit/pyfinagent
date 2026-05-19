# Experiment Results — phase-29.7 (arXiv-HTML precedence + pdfplumber rules)

**Step ID:** phase-29.7
**Date:** 2026-05-19
**Cycle:** 1

This is a **rules-doc** cycle. One insertion into `.claude/rules/research-gate.md` + one masterplan-entry update.

---

## 1. Edit made (verbatim diff)

### `.claude/rules/research-gate.md` — new section inserted

Insertion point: after `## Source quality hierarchy` (was line 62-71), before `## URL collection` (was line 73). The new section is 83 lines.

**Pre-edit file:** 134 lines.
**Post-edit file:** 217 lines (+83).

Section structure:
1. Lead paragraph — forbid raw `/pdf/<id>` WebFetch as primary attempt
2. **Step 1: arXiv native HTML (primary)** — `https://arxiv.org/html/<id>v<N>`, 74%-full / 97%-partial coverage stats from `arxiv.org/html/2402.08954v1`, partial-render guidance
3. **Step 2: ar5iv fallback** — `https://ar5iv.labs.arxiv.org/html/<id>`, April 2026 snapshot cutoff, confirmed-live note from this cycle's research
4. **Step 3: pdfplumber (last resort)** — `pip install pdfplumber` 0.11.9 (Jan 5 2026), 3-line extract recipe
5. **Limitations** — scanned PDFs, F1 0.9568 (finance) / 0.7644 (scientific), weak table extraction
6. **Alternatives** — pypdf (lighter), PyMuPDF (faster, C dep), Nougat/marker-pdf (GPU; rejected per CPU-only constraint)
7. **Never do** — 3 forbidden patterns including "add pdfplumber to backend/requirements.txt without owner sign-off"

---

## 2. Verbatim verification command output

```
$ grep -q '## PDF and arXiv paper fetching strategy' .claude/rules/research-gate.md && \
  grep -q 'arxiv.org/html' .claude/rules/research-gate.md && \
  grep -q 'ar5iv.labs.arxiv.org' .claude/rules/research-gate.md && \
  grep -q 'pdfplumber' .claude/rules/research-gate.md && \
  grep -q 'researcher environment' .claude/rules/research-gate.md && \
  grep -q '### Never do' .claude/rules/research-gate.md
$ echo exit=$?
exit=0

$ wc -l .claude/rules/research-gate.md
217 .claude/rules/research-gate.md
```

All 6 grep predicates PASS. Exit code 0.

---

## 3. Free-only + CPU-only compliance

| Tool | Free? | CPU-only? | In rules section as recommended? |
|---|---|---|---|
| arxiv.org/html | YES | YES (just URL fetch) | YES — primary |
| ar5iv.labs.arxiv.org | YES | YES | YES — fallback |
| pdfplumber v0.11.9 | YES (MIT) | YES | YES — last resort |
| pypdf | YES | YES | YES — alternative |
| PyMuPDF (fitz) | YES (AGPL) | YES (with C ext) | Listed but not recommended |
| Nougat / marker-pdf | YES | NO (GPU) | **EXPLICITLY REJECTED** |

100% of recommended tooling is free + CPU-only. The one tool that breaks CPU-only (Nougat) is named-and-rejected in the rule.

---

## 4. smoke test recipe (for operator)

```bash
# Test 1 — arXiv native HTML (post-Dec-2023 paper)
curl -sL https://arxiv.org/html/2402.08954v1 | head -50  # expect HTML, not binary

# Test 2 — ar5iv fallback (older paper)
curl -sL https://ar5iv.labs.arxiv.org/html/2312.01700 | head -50  # expect HTML

# Test 3 — pdfplumber last resort (separate venv to avoid project pollution)
python3 -m venv /tmp/pdfplumber-test && \
  /tmp/pdfplumber-test/bin/pip install pdfplumber && \
  /tmp/pdfplumber-test/bin/python -c "
import pdfplumber, urllib.request
urllib.request.urlretrieve('https://arxiv.org/pdf/2402.08954', '/tmp/p.pdf')
with pdfplumber.open('/tmp/p.pdf') as pdf:
    print(pdf.pages[0].extract_text()[:500])
"
# Expected: first 500 chars of paper text. Not binary.

# Test 4 — confirm pdfplumber NOT in project deps
grep -i pdfplumber backend/requirements.txt requirements.txt && echo "FAIL: pdfplumber in project deps" || echo "PASS: pdfplumber researcher-env-only"
```

---

## 5. Files touched

| File | Change |
|---|---|
| `.claude/rules/research-gate.md` | +83 lines (new PDF/arXiv section); pre-existing 134 → post 217 |
| `.claude/masterplan.json` phase-29.7 | name + audit_basis + verification fields rewritten |
| `handoff/current/research_brief.md` | rewritten (8-source brief) |
| `handoff/current/contract.md` | rewritten |
| `handoff/current/experiment_results.md` | this file |
| `handoff/current/live_check_29.7.md` | new |

**No** code files in `backend/`, `frontend/`, `scripts/`, `.claude/agents/` touched. **No** dependency manifest changes.

---

## 6. Honest disclosures

1. **pdfplumber NOT added to project requirements.** Research finding §6 explicitly says do not add as project dep; the rule itself enforces this in its "Never do" block. This is intentional restraint: research-time convenience does not justify shipping the extra ~12 MB and its pdfminer.six + pillow + pypdfium2 transitive deps in production builds.
2. **ar5iv coverage is snapshot through April 2026.** Papers submitted May 2026 onward will only be reachable via the native `arxiv.org/html/` route. The rule explicitly notes this.
3. **Table extraction is weak in pdfplumber.** The rule recommends preferring HTML for table-heavy finance papers. This is verifiable in arXiv:2410.09871 benchmark (low recall on tables).
4. **Anti-rubber-stamp / mutation-resistance:** the verification command uses 6 ANDed grep predicates, each anchored on a phrase distinct from the others (section header, URL substring, library name, install context, fail-stop block). Removing any one of these from the rule would fail the verification.
5. **Limitations of pdfplumber on scanned PDFs** are documented but not solved this cycle (would require OCR; out of scope per CPU-only / no-GPU constraint).
6. **Phase-29.0 P3 item #8** ("Document `deep` tier multi-subagent fork pattern in `docs/runbooks/per-step-protocol.md`") is NOT addressed in this cycle — it lives in 29.5 + 29.9. This cycle is purely the arXiv/pdfplumber rule.

---

## 7. Decision

Ready for Q/A spawn. 7 success criteria all evidenced by 6 grep predicates + post-edit line count.
