# Research Brief -- step 91.9

**Topic:** Leaking internal development phase-tracking labels into user-facing UI
copy; separating internal implementation metadata from end-user-facing strings in
production web apps.

**Tier:** simple (caller-specified; NOT self-declared).
**Audit-class:** NO (caller-specified). `coverage` reported for information only;
`coverage.dry` is not required for this step.
**Started:** 2026-08-20.

---

## ENVELOPE (born inert -- phase-86.37)

```json
{
  "brief_status": "COMPLETE",
  "tier": "simple",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 18,
  "urls_collected": 25,
  "recency_scan_performed": true,
  "internal_files_inspected": 7,
  "coverage": {
    "audit_class": false,
    "rounds": 2,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 2,
    "dry": false
  },
  "gate_passed": true
}
```

**Count provenance (re-derived from this file on disk, not carried from a running
tally).** `urls_collected` = 25 unique `https?://` URLs extracted from this brief
(33 naive occurrences, 25 after de-dup; the LOWER figure is claimed).
7 read-in-full + 18 snippet/attempted = 25, and all 7 claimed read-in-full URLs
were verified to appear literally in this file.
`internal_files_inspected: 7` counts files whose CONTENT I read and cite
(`observability/page.tsx` in full, `settings/page.tsx` excerpt,
`sovereign/page.tsx` line context, `.claude/agents/researcher.md`,
`.claude/rules/research-gate.md`, `.claude/rules/frontend.md`,
`.claude/rules/frontend-layout.md`). Separately, **174** files under
`frontend/src` were machine-scanned for the label pattern -- that is a scan
denominator, deliberately not claimed as "inspected".
`coverage` is informational: this step is NOT audit-class, so `coverage.dry` does
not gate.

---

## Status log (write-first, incremental)

- [t0] Brief created; envelope written born-inert. Read `.claude/agents/researcher.md`
  and `.claude/rules/research-gate.md` in full as instructed.
- [t1] Search round 1 (three-variant discipline) + internal grep pass done.
- [t2] Internal exploration COMPLETE (see below). Source 1 read in full (OWASP WSTG-INFO-05).

---

## Search queries run (three-variant discipline, `.claude/rules/research-gate.md`)

| # | Variant | Query |
|---|---|---|
| 1 | current-year frontier (2026) | `internal implementation metadata leaking into user-facing UI copy production web app 2026` |
| 2 | current-year frontier (2026) | `lint rule prevent hardcoded internal version phase identifiers in JSX text CI guard 2026` |
| 3 | last-2-year window (2025/2026) | `UI copy style guide avoid internal jargon developer terminology user-facing microcopy 2025` |
| 4 | last-2-year window (2025/2026) | `"internal" build or sprint identifiers accidentally shipped in user interface text incident 2025 2026` |
| 5 | year-less canonical | `separating internal developer identifiers from user-facing strings localization content design` |
| 6 | year-less canonical | `plain language jargon user interface guidance` (via NN/g + Microsoft style-guide entry points) |

## Read in full (counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|---|---|---|---|---|
| 1 | https://owasp.org/www-project-web-security-testing-guide/latest/4-Web_Application_Security_Testing/01-Information_Gathering/05-Review_Web_Page_Content_for_Information_Leakage | 2026-08-20 | official standard (OWASP WSTG-INFO-05) | WebFetch, full page | *"comments and metadata included in the HTML code might reveal internal information that should not be available to potential attackers"*; *"Sometimes, they forget about the comments and they leave them in production environments."* Objective explicitly includes reviewing page **content**, not just comments. Notably the page gives **no remediation section** and **assigns no severity** -- detection methodology only. |
| 2 | https://mozilla-l10n.github.io/documentation/localization/dev_best_practices.html | 2026-08-20 | official docs (Mozilla L10n) | WebFetch, full page | String IDs are *internal* identifiers paired with translatable text; the ID *"should always be descriptive of the message and its role in the interface (button label, title, etc.)"*. Translator/author context belongs in **localization notes attached to the string**, never inside the string. *"If you are changing a string such that its meaning has changed, you must update the string ID."* |
| 3 | https://www.nngroup.com/articles/plain-language-experts/ | 2026-08-20 | authoritative UX research (NN/g) | WebFetch, full page | Usability study with **domain experts**: even *"highly educated online readers crave succinct information that is easy to scan, just like everyone else."* An IT manager on a jargon-dense page: *"This is probably not a page I'd look into more. It's got a whole lot of jargon and technical terms I'm not aware of."* Kills the "our users are technical, so internal tags are fine" defence. |
| 4 | https://learn.microsoft.com/en-us/style-guide/word-choice/avoid-jargon | 2026-08-20 | official docs (Microsoft Writing Style Guide) | WebFetch, full page | *"Don't use jargon if: ... The term is familiar to only a small segment of your readers."* Testing checklist: *"If you think a term is jargon, it probably is."* / *"If it's an acronym or abbreviation, it may be jargon."* Page `ms.date` 2018, `updated_at` **2026-07-06** -- i.e. still-maintained current guidance. |
| 5 | https://eslint.org/docs/latest/rules/no-restricted-syntax | 2026-08-20 | official docs (ESLint) | WebFetch, full page | Accepts *"a list of strings, where each string is an AST selector"* or objects with `selector` + `message`; *"If a custom message is specified with the `message` property, ESLint will use that message when reporting occurrences"*. Precise selectors supported (`BinaryExpression[operator='in']`, `CallExpression[callee.name='setTimeout'][arguments.length!=2]`). **Limitation: syntax-only, no runtime-value logic** -- see Pitfall 2. |
| 6 | https://guidance.publishing.service.gov.uk/writing-to-gov-uk-standards/writing-guidelines/clear-language/ | 2026-08-20 | official standard (UK Government Digital Service) | WebFetch, full page (reached after a 301 + a redirect stub) | *"Avoid government 'buzzwords' and jargon. Often, these words are too vague and can lead to misinterpretation or empty, meaningless text."* *"Use the language that users: will be familiar with [and] use themselves."* *"Plain English is mandatory for all of GOV.UK."* And the escape hatch, which is the operative one here: *"Where you need to use specialist terms, you can... you just need to explain what they mean the first time you use them."* |
| 7 | https://dev.to/rmi_b83569184f2a7c0522ad/stop-shipping-hard-coded-strings-meet-i18nguard-an-i18n-linter-for-jsts-i18next-react-intl-4m8a | 2026-08-20 | community / tooling writeup (2025-09-22) | WebFetch, full page | Recency-window datapoint. *"A single hard-coded string slips into a PR... and users get a half-translated UI."* Teams rely on *"ad-hoc linters or per-framework scripts"*. Ships a CI gate: `npx i18nguard scan --format sarif --output i18n.sarif --fail-on-error`, plus a per-PR budget `maxNewHardCodedPerPR: 0`. **Does not discuss false positives** -- a notable omission given Pitfall 2. |

## Identified but snippet-only / attempted-and-failed (does NOT count toward the gate)

| URL | Kind | Why not read in full |
|---|---|---|
| https://developer.apple.com/design/human-interface-guidelines/writing | official docs | **Attempted WebFetch, FAILED** -- JS-rendered page, fetcher returned no content. Same failure mode as the recorded `feedback_gcloud_docs_fetch` memory. |
| https://www.gov.uk/guidance/content-design/writing-for-gov-uk | official standard | **Attempted, 301** -> redirected to guidance.publishing.service.gov.uk |
| https://guidance.publishing.service.gov.uk/writing-to-gov-uk-standards/tone-of-voice/ | official standard | **Attempted, redirect stub only** ("Redirecting...") |
| https://guidance.publishing.service.gov.uk/writing-to-gov-uk-standards/writing-guidelines/ | official standard | **Attempted, navigation shell only** -- no body text returned |
| https://learn.microsoft.com/en-us/style-guide/word-choice/ | official docs | Fetched but is a **78-word table of contents**, not substantive content; the substantive child page (`avoid-jargon`) was read in full instead and is counted there |
| https://github.com/OWASP/www-project-web-security-testing-guide/blob/master/v41/4-Web_Application_Security_Testing/01-Information_Gathering/05-Review_Webpage_Comments_and_Metadata_for_Information_Leakage.md | official standard (older v4.1 revision) | Superseded by the `latest` page read in full |
| https://kb.1ci.com/1C_Enterprise_Platform/Guides/Developer_Guides/1C_Enterprise_Development_Standards/Localization_requirements/Localization_guidelines_____Interface_strings_in_modules/ | vendor standard | Snippet carries the single most on-point rule found (see Key finding 2) but is tier-4 vendor doc; not spent a full-read slot |
| https://www.gridly.com/blog/what-are-string-identifiers-and-how-to-use-them-in-content-localization/ | industry blog | Tier-4; duplicative of Mozilla source |
| https://phrase.com/blog/posts/10-common-mistakes-in-software-localization/ | industry blog | Tier-4; duplicative |
| https://simplelocalize.io/blog/posts/internationalization-guide-software-localization/ | industry blog | Tier-4; duplicative |
| https://mozilla-l10n.github.io/documentation/localization/making_string_changes.html | official docs | Sibling of source 2; same doctrine |
| https://developer.apple.com/library/archive/documentation/MacOSX/Conceptual/BPInternational/Glossary/Glossary.html | official docs (archived) | Archived glossary; low value |
| https://www.parallelhq.com/blog/ux-writing-best-practices | industry blog | Tier-4; duplicative of NN/g |
| https://github.com/jsx-eslint/eslint-plugin-react | official plugin repo | `react/jsx-no-literals` is the adjacent rule; superseded here by the core ESLint rule read in full |
| https://blog.logrocket.com/12-essential-eslint-rules-react/ | industry blog | Tier-4 |
| https://docs.deno.com/lint/ | official docs | Wrong runtime for this repo (Next.js/ESLint) |
| https://www.cobalt.io/blog/excessive-data-exposure-how-apis-leak-sensitive-data | industry blog | API-side leakage, adjacent not on-point |
| https://cycode.com/blog/application-security-vulnerabilities/ | industry blog | Tier-4 listicle |

---

## Internal code inventory (the Explore half)

### Method

`grep -rnE 'phase-[0-9]+(\.[0-9]+)*' frontend/src` returns ~60 raw hits, but almost
all are `//` line comments or `{/* ... */}` JSX comments, which are **not rendered
text** (JSX comments are stripped at compile; `//` comments never reach the bundle
as visible copy). A raw grep therefore massively over-reports. To answer the real
question -- *which phase tags reach the user's eyes* -- I ran a comment-stripping
scan over all 174 source files under `frontend/src` (blank out `/* ... */` first,
then `//...$`, then match on what remains):

```
scanned_files=174
total_non_comment_hits=38
```

Of those 38, **35 are `describe(...)` strings in `*.test.tsx` / `*.test.ts` files**
(Vitest suite names -- never shipped to a browser). That leaves **3** non-comment
hits in production files.

### The 3 non-comment hits in production code

| File:line | Text | Rendered to a user? | Verdict |
|---|---|---|---|
| `frontend/src/app/observability/page.tsx:115` | `Per-table age + SLA bands across the warehouse (phase-25.C7)` | **YES** -- JSX text node inside the Tier-1 page-header `<p className="text-sm text-slate-500">` | **THE DEFECT** |
| `frontend/src/app/sovereign/page.tsx:61` | `console.error("phase-25.B12: RedLine fetch failed:", err)` | No -- devtools console only | Out of scope for "UI copy"; see Pitfall 3 |
| `frontend/src/app/settings/page.tsx:961` | `...inference (cycles 1+2+3+5; cycle 4 sector calendars is pure data-pull, zero LLM cost).` | **YES** -- rendered help text in the cost/privacy banner | Adjacent, *different* class: internal shorthand, not a dev-phase tag (see below) |

### Exact defect site

`frontend/src/app/observability/page.tsx:113-116`:

```tsx
<h2 className="text-2xl font-bold text-slate-100">Data Freshness</h2>
<p className="text-sm text-slate-500">
  Per-table age + SLA bands across the warehouse (phase-25.C7)
</p>
```

Per `.claude/rules/frontend-layout.md` §3 ("Page header patterns"), that `<p>` is the
**page subtitle** -- the highest-visibility descriptive string on the route, always
in the fixed (never-scrolling) header zone. `phase-25.C7` is a masterplan step id
from `.claude/masterplan.json`. It has no referent for any user; it is provenance
for the *author*, misfiled into the *reader's* channel.

### Structural facts about the file (196 lines total, read in full)

- It is the **only** copy of this string: no i18n catalog, no constants module, no
  shared `PAGE_META` map. Every page in `frontend/src/app/**` hardcodes its own
  `<h2>`/`<p>` header pair inline (confirmed against the template in
  `.claude/rules/frontend-layout.md` "New Page Template", which itself shows the
  literal-inline pattern). So there is exactly one edit site and no catalog to keep
  in sync.
- The file's own conventions are already correct elsewhere: the display-label map
  `BAND_LABEL` at `:22-27` maps internal enum values (`green`/`amber`/`red`/
  `unknown`) to user-facing words (`Fresh`/`Lagging`/`Stale`/`Unknown`). **The
  separation pattern this step wants already exists in this very file, 90 lines
  above the defect** -- the subtitle simply bypasses it.
- The genuine internal-provenance comment for the same page sits at `:9-12`
  (`// phase-49.3: this client page fetches live data...`) -- i.e. the codebase's
  *own* idiom is "phase tag goes in a comment". `:115` is the deviation, not the norm.
- `Computed at {data.computed_at}` (`:187-190`) renders a raw ISO timestamp -- a
  second, milder instance of internal-format-in-user-channel, but out of this
  step's scope.

### The `settings/page.tsx:961` adjacent case

`cycles 1+2+3+5; cycle 4` there refers to the **alpha-overlay feature cycles**, a
product concept, not a masterplan phase id. It does not match the phase-tag pattern
and is not the same defect. It is still internal shorthand a user cannot resolve, so
it is worth **noting**, but folding it into this step would widen scope beyond the
caller's stated boundary. Recommend queueing separately rather than bundling.

### Dead / duplicate code found

None relevant. No duplicate copy of the subtitle string exists anywhere
(`grep -rn "Per-table age"` -> single hit). No dead code in the file.

---

## Recency scan (last 2 years, 2024-2026) -- MANDATORY SECTION

Four searches were scoped to the 2024-2026 window (queries 1-4 in the table
above), covering the defect class from three angles: security/leakage framing,
UX-writing framing, and tooling/CI-enforcement framing.

**Result: ONE new finding in the window; ZERO findings that supersede the
canonical guidance.**

- **New (2025-09-22):** `i18nGuard`, an i18n linter for JS/TS that ships an
  explicit CI gate for hard-coded user-facing strings
  (`--fail-on-error`, `maxNewHardCodedPerPR: 0`). This is a *mechanism*
  contribution -- the 2024-2026 window's movement is in **enforcement tooling**,
  not in doctrine.
- **Nothing supersedes.** The doctrinal sources are stable and currently
  maintained rather than stale: the Microsoft "Avoid jargon" page carries
  `ms.date: 2018` but `updated_at: 2026-07-06`; the OWASP WSTG page fetched was
  the `latest` revision; GOV.UK clear-language is live standing guidance. No
  2024-2026 source argues *for* internal identifiers in user-facing copy, and no
  2024-2026 source revises the rule.
- **Explicit negative result:** a targeted search for a *documented incident* of
  internal build/sprint/phase identifiers shipping in production UI
  (query 4) returned **nothing on point**. There is no cited case study to lean
  on; this class is treated in the literature as an obvious style violation, not
  as an incident category. That absence is itself a finding -- it caps how
  strongly the step may frame the severity (see Consensus vs debate).

---

## Key findings

1. **The rule exists in near-identical form in four independent traditions, and
   it is a channel-separation rule, not a vocabulary rule.** Mozilla's model is
   the cleanest statement of it: a string has an *internal* ID and a *user-facing*
   text, and author/translator context lives in a **localization note attached to
   the string, never inside it** (Mozilla L10n,
   https://mozilla-l10n.github.io/documentation/localization/dev_best_practices.html,
   accessed 2026-08-20). The vendor-standard snippet found via search states the
   consequence bluntly -- *"Metadata object names and internal identifiers used in
   the code must not appear in the user interface, user messages, or reference
   information"* (1ci.com localization guidelines, snippet-only,
   https://kb.1ci.com/1C_Enterprise_Platform/Guides/Developer_Guides/1C_Enterprise_Development_Standards/Localization_requirements/Localization_guidelines_____Interface_strings_in_modules/).

2. **"Our users are technical, so a phase tag is harmless" is empirically
   refuted.** NN/g ran the study on *domain experts* and found they behave like
   everyone else: *"highly educated online readers crave succinct information that
   is easy to scan, just like everyone else"*, with an IT manager reacting to a
   jargon-dense page with *"This is probably not a page I'd look into more. It's
   got a whole lot of jargon and technical terms I'm not aware of."* (NN/g,
   https://www.nngroup.com/articles/plain-language-experts/, accessed 2026-08-20).
   pyfinagent's UI has exactly one operator, who *is* technical -- this finding is
   what closes off the obvious "it doesn't matter here" defence.

3. **Microsoft's jargon test resolves this case mechanically.** *"Don't use jargon
   if: ... The term is familiar to only a small segment of your readers"*, plus
   *"If you think a term is jargon, it probably is"* and *"If it's an acronym or
   abbreviation, it may be jargon"* (Microsoft Writing Style Guide,
   https://learn.microsoft.com/en-us/style-guide/word-choice/avoid-jargon,
   accessed 2026-08-20). `phase-25.C7` is familiar to a segment of size one -- the
   author -- and resolves only against `.claude/masterplan.json`, a file no user
   can see.

4. **The only legitimate escape hatch does not apply.** GOV.UK permits specialist
   terms conditionally: *"Where you need to use specialist terms, you can... you
   just need to explain what they mean the first time you use them"*, against a
   backdrop where *"Plain English is mandatory"* and you should *"Use the language
   that users: will be familiar with [and] use themselves"*
   (https://guidance.publishing.service.gov.uk/writing-to-gov-uk-standards/writing-guidelines/clear-language/,
   accessed 2026-08-20). `(phase-25.C7)` is unexplained, and cannot be usefully
   explained to a user, so the exemption is unavailable.

5. **The security framing is real but MILD, and must not be over-claimed.** OWASP
   WSTG-INFO-05 covers exactly this territory -- *"comments and metadata included
   in the HTML code might reveal internal information that should not be available
   to potential attackers"*, *"Sometimes, they forget about the comments and they
   leave them in production environments"* -- and its stated objective includes
   reviewing page **content**, not only comments
   (https://owasp.org/www-project-web-security-testing-guide/latest/4-Web_Application_Security_Testing/01-Information_Gathering/05-Review_Web_Page_Content_for_Information_Leakage,
   accessed 2026-08-20). **But** that page assigns **no severity rating** and
   offers **no remediation section** -- it is detection methodology only. Its
   worked examples are leaked DB passwords and SQL, which is a different order of
   magnitude from a step id. Treat this as *supporting* context, not as a
   vulnerability claim.

6. **A one-time grep is not a guard; the 2024-2026 tooling movement is toward a
   CI gate with a zero-new budget.** i18nGuard's pattern is
   `npx i18nguard scan --fail-on-error` plus `maxNewHardCodedPerPR: 0`
   (https://dev.to/rmi_b83569184f2a7c0522ad/stop-shipping-hard-coded-strings-meet-i18nguard-an-i18n-linter-for-jsts-i18next-react-intl-4m8a,
   2025-09-22, accessed 2026-08-20). The proportionate in-repo analogue is ESLint
   `no-restricted-syntax`, which takes *"a list of strings, where each string is
   an AST selector"* and supports a per-selector `message`
   (https://eslint.org/docs/latest/rules/no-restricted-syntax, accessed
   2026-08-20).

---

## Consensus vs debate (external)

**Consensus (unanimous, no dissenting source found):** internal implementation
identifiers do not belong in user-facing strings. Agreed across UX research
(NN/g), a vendor style guide (Microsoft), a government content standard (GOV.UK),
and localization engineering practice (Mozilla). No source in any tier argued the
other way -- the closest thing to a counterweight is Microsoft's own concession
that *"In the right context, for a particular audience, jargon serves as shorthand
for well-understood concepts"*, which does not rescue `phase-25.C7` because it is
not well-understood by any audience but the author.

**Genuine debate 1 -- is this a security finding?** OWASP catalogues
internal-information-in-page-content under WSTG-INFO-05, which invites framing
this as an information-leakage defect. Against that: OWASP assigns no severity,
provides no remediation, and its examples are credentials and SQL. And the
targeted 2024-2026 incident search found **no documented case** of phase/sprint
identifiers in UI causing harm. **Recommendation: frame the step as a
content/correctness defect with information-hygiene as secondary support. Do not
write a criterion that asserts a security impact** -- the evidence will not carry
it, and an unsupportable severity claim is exactly what a Q/A pass should catch.

**Genuine debate 2 -- how heavy should the guard be?** The i18n-linter school
bans *all* literal strings in JSX and forces a message catalog. That is
disproportionate here: pyfinagent's frontend is single-locale English with no i18n
layer, and every page in `frontend/src/app/**` hardcodes its header pair inline by
design (`.claude/rules/frontend-layout.md` "New Page Template"). Adopting a
catalog to fix one subtitle would be a large architectural change smuggled in as a
copy fix. The targeted-selector school (ESLint `no-restricted-syntax` matching
only the phase-tag shape) is the proportionate option.

---

## Pitfalls (from literature + measured here)

1. **A naive grep over-reports this defect roughly 20:1, and a criterion written
   against a naive grep is unsatisfiable.** Measured above: `grep -rnE 'phase-[0-9]'`
   over `frontend/src` returns ~60 hits; after comment-stripping, 38 remain; after
   excluding `*.test.*`, **3** remain; after excluding the console string and the
   product-cycle string, **1** remains. The ~57 excluded hits are `//` and
   `{/* */}` provenance comments that the repo's own conventions *require*
   (`observability/page.tsx:9-12` is one). **A criterion of the form "grep for
   `phase-` in `frontend/src` returns zero hits" would demand deleting legitimate
   provenance and would fail forever.** The verification command must
   comment-strip and test-exclude, or scope to rendered JSX text nodes.

2. **ESLint `no-restricted-syntax` is syntax-only.** Its documented limitation is
   that it *"only restricts syntax; it doesn't support conditional logic based on
   runtime values"*. Matching a *substring pattern inside a `JSXText` node* is at
   the edge of what esquery attribute selectors do. Whether a selector such as
   `JSXText[value=/phase-\d/]` actually works **must be executed and verified
   before it is written into an immutable criterion** -- do not assert it from the
   docs. (This brief does not claim it works; it was not run.)

3. **Deleting the tag loses the provenance; the doctrine says relocate it.**
   Mozilla's model keeps author context *attached to* the string as a note. The
   correct fix is to move `phase-25.C7` into a JSX comment adjacent to the
   subtitle (matching the file's own idiom at `:9-12`), not to erase the trace.
   Related standing guidance: `feedback_provenance_is_only_where_a_reader_looks`.

4. **Adjacent channels must be decided explicitly, not silently.** The phase tag
   at `sovereign/page.tsx:61` lives in a `console.error` string -- user-observable
   in devtools, but not UI copy. The caller's scope says *rendered JSX text*, so it
   is **out of scope**; say so in the contract rather than leaving a reader to
   wonder whether the sweep missed it. Same for `settings/page.tsx:961`
   (`cycles 1+2+3+5`), which is internal shorthand of a *different* class.

5. **The replacement copy can import a fresh violation.** The remaining subtitle
   would read *"Per-table age + SLA bands across the warehouse"*. By Microsoft's
   own checklist (*"If it's an acronym or abbreviation, it may be jargon. Spell it
   out"*), **SLA** is borderline. But `SLA Interval` is already a rendered column
   header at `observability/page.tsx:165`, so "fixing" SLA would widen scope from
   one string to several and change a table header. Recommend: leave SLA, note it.

---

## Application to pyfinagent (external findings -> file:line anchors)

| Finding | Anchor | Implication for the contract |
|---|---|---|
| Internal ids must not appear in UI text (Mozilla / 1ci / Microsoft / GOV.UK) | `frontend/src/app/observability/page.tsx:115` | Single edit site. Remove `(phase-25.C7)` from the rendered subtitle. |
| Author context attaches to the string, it is not discarded (Mozilla) | `frontend/src/app/observability/page.tsx:9-12` shows the file's own correct idiom | Relocate the tag to a JSX comment; do not simply delete it. |
| Separation pattern already exists in-file | `frontend/src/app/observability/page.tsx:22-27` (`BAND_LABEL` maps internal enum -> user words) | No new abstraction is needed; the fix is a one-line copy edit, not an i18n layer. |
| Expert users still reject jargon (NN/g) | n/a (whole-UI claim) | Forecloses the "single technical operator, so it's fine" objection in the contract's hypothesis. |
| Naive grep over-reports 20:1 (measured) | ~57 comment hits across `frontend/src`, incl. `middleware.ts:6`, `globals.css:21`, `layout.tsx:22`, `agents/page.tsx:381` | **The immutable verification command must comment-strip and exclude `*.test.*`.** This is the highest-risk item for a green-able criterion. |
| Durable guard beats a one-time sweep (i18nGuard / ESLint) | no ESLint rule of this shape exists in the repo today | Optional hardening; if adopted, the selector must be *executed* first (Pitfall 2). |
| Adjacent-but-out-of-scope instances | `sovereign/page.tsx:61`, `settings/page.tsx:961` | Name them in the contract as explicitly out of scope; queue separately per `feedback_queue_discovered_defects_in_masterplan`. |

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **7** read in full
- [x] 10+ unique URLs total (incl. snippet-only) -- **25** unique URLs in this brief
- [x] Recency scan (last 2 years) performed + reported -- 4 window-scoped queries; 1 new finding, 0 superseding
- [x] Full pages read (not abstracts) for the read-in-full set -- all 7 fetched as full pages; 5 failed/thin fetches are recorded as attempts in the snippet table, not counted
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module -- all 174 source files under `frontend/src` scanned, not just the named file
- [x] Contradictions / consensus noted -- two genuine debates recorded (security framing; guard weight)
- [x] All claims cited per-claim with URL + access date
- Gap disclosed: Apple HIG "Writing" could not be fetched (JS-rendered); GOV.UK required three hops and two of them returned no body. Source-quality hierarchy is still satisfied without them (2 official standards + 2 official docs + 1 authoritative UX research in the read-in-full set).

