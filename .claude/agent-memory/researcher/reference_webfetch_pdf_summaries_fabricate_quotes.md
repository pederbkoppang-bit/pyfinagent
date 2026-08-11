---
name: webfetch-pdf-summaries-fabricate-quotes
description: A WebFetch of a large PDF returns a SUMMARY that can invent quotation-marked text; measured twice (83.1.1 and 86.29) -- always re-extract and regex-verify before a quote enters a brief or contract
metadata:
  type: reference
---

`WebFetch` on a large PDF does not hand you the document; it hands you a small
model's summary of it, and that summary **can emit fabricated text inside
quotation marks**. This is now a CLASS, not an incident -- measured twice:

- step 83.1.1: a WebFetch PDF summary fabricated content of the paper.
- step 86.29 (2026-08-10): fetching CCSDS 650.0-M-3 (OAIS) returned, quoted,
  *"All changes to the AIP must be documented to maintain the ability to
  establish authenticity."* Re-extracting the real 150-page PDF with `pypdf`
  (374,127 chars) gives **0 hits** for that string. The genuine definitions
  (`Provenance Information`, `Transformation`, `Transformational Information
  Property`) exist, read differently, and are stronger than the invention.

**Why to care:** the fabricated line was plausible, on-topic, and would have
been copied into a contract as a standards citation. Fail-plausible, not
fail-obvious.

**How to apply.** When a WebFetch response carries the marker
`[Binary content (application/pdf, N MB) also saved to <path>]`, that local file
is the ground truth -- use it:

```python
from pypdf import PdfReader          # already an indirect dep via paper-search-mcp
txt = "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
```

then `re.finditer` every string you intend to put inside quotation marks. Report
the hit count. Zero hits = the quote does not exist; drop it and quote what the
extraction actually contains. Paraphrases from the summary are still usable as
*leads*, never as citations.

This is cheap (one Bash call) and it upgrades the source from "summarised" to
genuinely read-in-full -- so it strengthens the gate rather than costing it.
Same discipline as [[measure-dont-assert]]: a quotation is a claim about a set
(the document's strings) whose membership you have not checked.
