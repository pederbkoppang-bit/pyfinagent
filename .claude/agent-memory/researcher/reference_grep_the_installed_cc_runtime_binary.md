---
name: grep-the-installed-cc-runtime-binary
description: When Claude Code docs are silent on runtime behaviour, decompile the installed binary — it answered three questions the docs and GitHub could not (step 86.84)
metadata:
  type: reference
---

The installed Claude Code binary is a greppable source of truth and outranks the
docs when they are silent. Path: `~/.local/share/claude/versions/<version>`
(2.1.232 = 306 MB, a bun-compiled native binary, NOT a `cli.js`).

**Technique that works** (a naive `grep -o -E '.{0,90}FOO.{0,90}'` over the whole
binary FAILS — ugrep rejects it with "exceeds complexity limits" on UTF-8 byte
classes, and a full scan takes ~2 min):

1. `LC_ALL=C grep -abo -m1 -F '<a string unique to the code path>'` to get a byte
   offset. Pick a string with trailing syntax (e.g. `...");`) so you match the CODE
   occurrence, not the string table.
2. `dd if=$BIN bs=1M skip=<MB> count=16` to carve a region file.
3. Search the region in **python3**, not grep — `data.find(needle)` in a loop with
   a +/-110-byte window. Instant, no regex-engine limits.

**What it answered on step 86.84** that no doc or issue did: the exact live error
string (and that its nudge count had changed from the one in the public issue), that
the workflow `agent()` opts has no per-call turn budget, that `max_turns_reached`
just logs and `break`s, and that the schema branch throws where the non-schema
branch returns accumulated text.

Related: [[project-86-84-workflow-turn-cap-drops]]. Beware the inverse trap —
[[a-probe-can-match-its-own-documentation]] applies here too: the binary's string
table and its code both contain the same literal.
