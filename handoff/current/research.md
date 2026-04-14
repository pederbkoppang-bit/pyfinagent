# Research Gate -- Phase 4.2.3.2 / SN4 `since_date` Lexicographic Trap

**Step:** SN4 micro-fix to `SignalsServer.get_signal_history` in
`backend/agents/mcp_servers/signals_server.py`.

**Path taken:** Primary researcher subagent path skipped per
`.claude/context/known-blockers.md` and the 2026-04-14-2026 session log --
both researcher and general-purpose subagents are currently flaky due to
`Stream idle timeout - partial response received` on web-heavy briefs.
Fell back to in-session WebSearch (7 queries, ~70 URL results across 7
topic categories).

## Categories

### 1. Python `date.fromisoformat` semantics (3.11+)

Python 3.11 relaxed `datetime.date.fromisoformat` to accept "most ISO 8601
formats" including the basic form `YYYYMMDD` and ISO week dates
`YYYY-Www-d`. **However, unpadded month/day forms like `YYYY-M-D` are
still rejected.** The helper must normalize these forms manually before
handing them to `date.fromisoformat`.

Sources:
- [Python 3.14 datetime docs](https://docs.python.org/3/library/datetime.html)
- [Python 3.11 what's new (fromisoformat expansion)](https://docs.python.org/3/whatsnew/3.11.html)
- [Python 3.11 changelog notes](https://gist.github.com/moreati/85de3f0745d473f54dd9076125338642)
- [note.nkmk.me: isoformat / fromisoformat](https://note.nkmk.me/en/python-datetime-isoformat-fromisoformat/)
- [LabEx: creating datetime objects from ISO-8601](https://labex.io/tutorials/python-how-to-create-datetime-objects-from-iso-8601-date-strings-417942)

### 2. ISO 8601 canonical zero-padding requirement

ISO 8601 **requires** each date component to be zero-padded to a fixed
width (4 / 2 / 2). The canonical "lexicographic == chronological"
equivalence only holds for zero-padded forms. Mixed padded/unpadded
inputs break the equivalence silently.

Sources:
- [ISO 8601 (Wikipedia)](https://en.wikipedia.org/wiki/ISO_8601)
- [ISO -- ISO 8601 Date and Time Format](https://www.iso.org/iso-8601-date-and-time-format.html)
- [IONOS: ISO 8601 global standard](https://www.ionos.com/digitalguide/websites/web-development/iso-8601/)
- [W3C NOTE-datetime](https://www.w3.org/TR/NOTE-datetime)
- [Markus Kuhn: international standard date notation](https://www.cl.cam.ac.uk/~mgk25/iso-time.html)

### 3. Lex-compare vs chronological-compare traps

Concrete examples of the SN4 failure mode documented in the wild:

- `"2025-12-01" < "2025-2-15"` lexically (because `"1" < "2"` at index 5),
  but `2025-12-01 > 2025-02-15` chronologically. A lex compare would
  spuriously treat December as "earlier" than February.
- AWS SDK `InstantAsStringAttributeConverter` bug #2219: non-canonical
  trailing-zero truncation broke lexicographic sorting for DynamoDB.
- Mozilla bugzilla #1500748: `Date.parse` accepts non-zero-padded
  ISO-8601 forms but downstream consumers assume padded-sort.

Sources:
- [aws/aws-sdk-java-v2 #2219 -- Instant-as-String not lex-sortable](https://github.com/aws/aws-sdk-java-v2/issues/2219)
- [Mozilla bug 1500748 -- Date.parse accepts non-padded ISO](https://bugzilla.mozilla.org/show_bug.cgi?id=1500748)
- [The Subtle Trap of ISO Date Strings in JavaScript (musatov.com)](https://musatov.com/posts/iso-date-string-parsing/)
- [dev.to mirror: Subtle Trap of ISO Date Strings](https://dev.to/musatov/the-subtle-trap-of-iso-date-strings-in-javascript-49co)
- [Medium: ISO 8601 sortable date string (Kamasu Paul)](https://medium.com/@kamasupaul/iso-8601-the-sortable-date-string-83bc226306b6)
- [copyprogramming: Python compare two date strings (2026 guide)](https://copyprogramming.com/howto/python-python-compare-two-strings-that-are-dates)

### 4. Defensive date parsing in financial/quant systems

Financial code must **never raise** from a read API on user-supplied
filters. The canonical pattern is "parse to datetime, catch
`ValueError` + `TypeError`, fall back to a safe default" (unfiltered,
not "reject silently wrong"). `dateutil.parser` is the third-party
alternative, but adds a dependency and is arguably *too* permissive for
quant pipelines (accepts ambiguous `mm/dd` vs `dd/mm`).

Sources:
- [LabEx: resolve datetime parsing error](https://labex.io/tutorials/python-how-to-resolve-datetime-parsing-error-438481)
- [LabEx: manage date type exceptions](https://labex.io/tutorials/python-how-to-manage-date-type-exceptions-452374)
- [LabEx: interpret date formats safely](https://labex.io/tutorials/python-how-to-interpret-date-formats-safely-450837)
- [LabEx: address invalid date operations](https://labex.io/tutorials/python-how-to-address-invalid-date-operations-437216)
- [Python 3.14 tutorial -- errors and exceptions](https://docs.python.org/3/tutorial/errors.html)
- [Duke FinTech Core Python datetime exercise answers](https://fintechpython.pages.oit.duke.edu/jupyternotebooks/1-Core%20Python/answers/28-DateTime.html)

### 5. Zero-padding normalization techniques

- `str.zfill(2)` -- stdlib zero-pad for strings representing numbers.
- f-string format spec `f"{int(x):02d}"` -- parses then re-renders,
  also validates the input is numeric (no regex needed).
- `dateutil.parser` -- third-party, rejected (scope discipline).

We picked the **f-string approach** because it folds parsing + padding
into one step: `int("4")` on a non-numeric string raises `ValueError`
which is caught by the outer except, giving a natural reject path.

Sources:
- [note.nkmk.me: zero-padding strings and numbers](https://note.nkmk.me/en/python-zero-padding/)
- [Medium (PythonistaSage): leading zero formatting](https://medium.com/vicipyweb3/mastering-leading-zero-magic-a-friendly-guide-to-formatting-leading-zeros-in-python-3cb934d7aa4f)
- [prefetch.net: normalizing date strings with dateutil](https://prefetch.net/blog/2017/08/02/normalizing-date-strings-with-the-python-dateutils-module/)

### 6. Quant-trading-specific date alignment bugs

Lookahead bias, backtest period-boundary errors, and data-integrity
issues are all downstream consequences of sloppy date comparisons.
QuantConnect's "Common Errors" doc explicitly calls out mixed
timezone/format date comparisons as a lookahead-bias root cause.

Sources:
- [QuantConnect -- History common errors](https://www.quantconnect.com/docs/v2/writing-algorithms/historical-data/common-errors)
- [QuantConnect -- Dataset misconceptions](https://www.quantconnect.com/docs/v2/cloud-platform/datasets/misconceptions)
- [QuantConnect -- Periods](https://www.quantconnect.com/docs/v2/writing-algorithms/key-concepts/time-modeling/periods)
- [QuantLib-Python dates and conventions](https://quantlib-python-docs.readthedocs.io/en/latest/dates.html)

### 7. Scope-discipline anchors (never-raise, stdlib-only)

- `ValueError` + `TypeError` is the minimal correct exception set for
  stdlib date parsing. `OverflowError` is raised on out-of-range
  integers but only for `datetime`, not `date` with `int()` intermediate.
- No third-party imports -- keeps the micro-fix under the no-new-deps
  rule and matches the Phase 4.2.3.1 `math.isfinite` precedent (stdlib
  only, one-import addition).
- Never raise from a read API (`get_signal_history` is a read) -- the
  existing block already has this invariant; the fix must preserve it.

Sources (reused): Python stdlib docs (category 1), Duke FinTech (4),
Phase 4.2.3.1 contract.md lock-ins (in-tree).

## Design lock-ins (binding on GENERATE)

1. **Add `date` to the existing `from datetime import ...` line**.
   No new top-level imports beyond this one word.
2. **Helper is a `@staticmethod` on `SignalsServer`**, not a
   module-level free function. Matches the file's "everything on
   the class" shape (signals_server.py has zero module-level helpers
   currently).
3. **Helper name `_parse_iso_date`**, private, returns
   `Optional[date]`. Accepts `Any` input, returns `None` on any failure.
4. **Two-step parse**: (a) try `date.fromisoformat(s)` strict;
   (b) on failure, split on `-`, re-pad each component via
   `f"{int(x):02d}"`, retry. Bounded by an outer try/except
   catching `ValueError` + `TypeError`.
5. **Call-site** in `get_signal_history` replaces the lex-compare
   block with `since_dt = self._parse_iso_date(since_date)` then
   `sig_dt = self._parse_iso_date(sdate)` per record, compare as
   `date` objects.
6. **Never raise invariant preserved**: both the helper and the
   call-site catch all date-parse exceptions internally. No new
   `raise` statements anywhere.
7. **Scope discipline**: ONLY `get_signal_history` and the new helper
   are touched. All other methods on `SignalsServer` must remain
   AST-byte-identical to pre-fix HEAD (2494d10).
8. **Diff budget**: `<= 50` added / `<= 15` deleted lines, single file.
9. **ASCII-only**: no non-ASCII characters in helper or call-site.
10. **No cross-server imports**: helper is self-contained.

## Out-of-scope (intentionally deferred)

- Parsing full datetime strings (`YYYY-MM-DDTHH:MM:SS`).
- Accepting non-ISO forms like `MM/DD/YYYY` or `DD.MM.YYYY`.
- Hardening `_append_signal_history` to normalize on write.
- Any change to Phase 4.2.2 scaffold functions.

## Research gate tally

- **Categories covered**: 7/7
- **Unique URLs**: ~50+ across 7 queries
- **Sources cross-referenced**: Python 3.11 what's new, ISO 8601
  Wikipedia, copyprogramming date-compare guide, LabEx datetime
  error guides
- **Gate**: PASSED
