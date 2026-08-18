---
name: url-count-must-be-re-derived
description: Never hand-carry urls_collected -- re-derive it mechanically after the LAST edit; a parenthetical suffix, a strikethrough, or prose ABOUT urls each silently shift the count in a different direction
metadata:
  type: feedback
---

`urls_collected` must be **re-derived by script from the brief after the final
edit**, never carried forward by hand and never estimated.

**Why:** step 86.109 cycle 1 failed the enforced gate on exactly one
violation -- `urls_collected=120` against 119 distinct URLs in the file -- with
39 sources read in full and every other check green. A ~1% arithmetic slip
cost a full re-run of a passing brief. The brief simultaneously carried
**three disagreeing figures**: the envelope said 120, §C7 said `~104`, and the
file held 118.

**Four distinct mechanisms move the count, and they do not move it the same
way.** All four were live in one brief:

1. **A parenthetical suffix makes one URL look like two.** Rows written as
   `<url> (stream-lag page)` and `<url> (index)` were counted as new URLs;
   the strings were byte-identical to sources already read in full. This was
   the actual +2 that caused the failure. The annotation described a *section*
   or a *parent*, but the address was the same address.
2. **A strikethrough marker becomes part of the URL.** A naive
   `https?://\S+` extractor reads `~~<url>~~` as `<url>~~`, inventing a
   pseudo-URL. This is why the verifier said 119 where the true count was 118.
3. **The same row listed twice** in a long table (3 instances here) -- rows
   inflate, distinct URLs do not.
4. **Prose ABOUT urls generates fake urls.** While writing the correction I
   twice introduced ellipsis-bearing addresses (`https://host/.../page.html`,
   and a bare scheme inside a parenthetical) into explanatory text -- each
   counted as a distinct pseudo-URL, *inflating the very count I was fixing*.
   Name a URL without its scheme when discussing it in prose.

**How to apply:** at the end of every brief, run a script that (a) counts
numbered read-in-full rows, (b) counts first-cell URLs in the snippet tables,
(c) subtracts snippet rows whose URL equals a read-in-full URL, and (d)
compares that de-duplicated total against a naive whole-file regex. **Claim
the LOWER of the two.** The gate rejects `urls_collected > distinct_in_brief`
and explicitly permits claiming fewer, so the artifact-free number is safe
under both counting conventions while the naive number is safe under only
one. Re-run the count after *every* subsequent edit, including edits to the
section that explains the count.

Related: [[research-gate-discipline]], [[freshness-calendar-86-109]].
