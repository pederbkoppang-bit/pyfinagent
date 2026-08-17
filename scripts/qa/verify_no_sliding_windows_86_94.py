#!/usr/bin/env python3
"""phase-86.94 -- corpora and windows defined relative to `now` are not reproducible.

THE CLASS, AND WHY A BAN IS THE WRONG SHAPE
-------------------------------------------
`git log --since=<bare date>` does NOT mean midnight. Git resolves it with the
CURRENT TIME OF DAY, so the window slides forward as the clock advances and
commits on the boundary day silently leave the corpus. Proven from git's own
resolver rather than inferred from a count:

    now: 2026-08-16 23:09:03 CEST
    $ git rev-parse --since=2026-08-11
    --max-age=1786482543   ->  2026-08-11 23:09:03 CEST

The bare date carried TODAY's clock time onto the target date. `--since=today`
is likewise NOT midnight (0 commits vs 64 for a real midnight pin) even though
git-log(1) and git-rev-list(1) both say it is.

A NAIVE PIN IS NOT ENOUGH EITHER, and this is the part phase-86.91 missed when
it "fixed" the replay by pinning `2026-08-11T00:00:00`: that timestamp is
TZ-LOCAL. Measured on this repo, both ends pinned, varying only `$TZ`:

    over the WHOLE history:      Oslo 766   Seoul 846   NY 766   UTC 766
    over the replay's own corpus: Oslo 707   Seoul 787   NY 707   UTC 707
    ... with the Z form, every timezone agrees (766 and 707 respectively)

An 80-commit spread decided by the machine's timezone -- the same 80 either way,
because it is one contiguous band of commits that Seoul's midnight includes and
the others' excludes. Two corpora are quoted because the replay bounds its upper
end at 8dc70502 while the raw history does not; both are measured, and neither
figure means anything without the bound it was taken under. The asymmetry is data,
not theory -- New York and Oslo coincide because the band they straddle is
EMPTY (0 commits in [2026-08-10T22:00Z, 2026-08-11T04:00Z)), while Seoul's is
not (80 commits in [2026-08-10T15:00Z, 2026-08-11T00:00Z)). So a TZ-naive pin
can look perfectly stable for months and then move by 80 when someone runs it
elsewhere. Silence is not evidence of safety.

WHY AN ALLOWLIST, NOT A PROHIBITION. One member is legitimately relative:
`backend/slack_bot/scheduler.py` reports "shipped today" to Slack, and a report
about today MUST be relative to today. A blanket ban would break correct code
and would then be switched off. Every relative use is therefore allowed only
with a RECORDED REASON, and an unlisted one fails.

Run:  python scripts/qa/verify_no_sliding_windows_86_94.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

_pass = 0
_failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> bool:
    global _pass
    if cond:
        _pass += 1
        print(f"  ok   {name}")
    else:
        _failures.append(f"{name}{' -- ' + detail if detail else ''}")
        print(f"  FAIL {name}{' -- ' + detail if detail else ''}")
    return bool(cond)


# ── THE ENUMERATION RULE (criterion 2: written down, not hand-listed) ────────
#
#   A WINDOW SITE is any line in a tracked *.py or *.sh file under scripts/,
#   .claude/hooks/ or backend/ that passes a `--since` / `--until` /
#   `--since-as-filter` / `--until-as-filter` argument to git.
#
#   A window site is SLIDING when its value is not an instant that resolves
#   identically on every machine and at every clock time. Concretely, a value is
#   REPRODUCIBLE only if it is
#     (a) UTC-qualified   -- ends in Z or +0000/-0000, or is an @epoch, or
#     (b) an operator-supplied parameter with no hardcoded default, or
#     (c) interpolated from a value read out of a pinned artifact.
#   Everything else -- a bare date, `today`, `midnight`, `N.days`, `now`
#   arithmetic, or a TZ-NAIVE timestamp -- is SLIDING.
#
# The rule is deliberately WIDER than "bare date": phase-86.91 pinned a naive
# timestamp and believed it had closed this, so a rule that only caught bare
# dates would have called that fix clean.
# `--after` and `--before` are EXACT synonyms of `--since` / `--until`
# (git-log(1)); measured: `git rev-parse --after=2026-08-13` returns the same
# --max-age as `--since=2026-08-13`. The first version of this rule named only
# since/until, so both synonyms walked straight past it -- and both are named
# verbatim in this step's own audit_basis, which makes the omission a recall
# failure rather than a scope choice.
WINDOW_OPTS = r"--(?:since|until|after|before)(?:-as-filter)?"
# `[=\s]` alone misses the repo's DOMINANT idiom:
#     subprocess.run(["git", "log", "--since", "2026-08-11"])
# There the option is followed by a QUOTE, not `=` or whitespace, so the
# line never matched at all -- the site was INVISIBLE, and the fail-closed
# <unparsed> path never fired. The subject script builds its own args this
# way (replay_changelog_rule_86_68.py:114) and so does this checker.
WINDOW_RE = re.compile(WINDOW_OPTS + r"([=\s]|[\"']\s*,)")
# Two spellings: `--since=VALUE` and `--since VALUE` (separate argv element or
# separate token). Both are captured; anything the pattern cannot parse is
# reported as UNPARSED and FAILS, never skipped -- see scan_text.
VALUE_EQ_RE = re.compile(WINDOW_OPTS + r"=(?P<v>[^\"'\s,)\]]+)")
# THE SPACE FORM IS AMBIGUOUS WITH ENGLISH and that is not hypothetical:
# `print("... a bare --since date slides with the clock ...")` is executable
# code, not a comment, so comment-stripping cannot save it, and an
# unconditional space pattern captured the word "date" as a window value in
# TWO tracked files. The `=` form has no such ambiguity and stays
# unconditional. The space form therefore additionally requires the value to
# LOOK like a window value.
#
# RESIDUAL, STATED: a space-separated window whose value matches none of the
# known shapes is NOT detected. That is a deliberate trade against flooding
# the gate with English false positives, which would get it switched off. The
# `=` form -- the spelling every site in this repo actually uses -- remains
# fail-closed on an unparsed value.
VALUE_SP_RE = re.compile(WINDOW_OPTS + r"\s+[\"']?(?P<v>[^\"'\s,)\]]+)")
# argv-list: "--since", "2026-08-11"  /  '--after' , '30.days'
VALUE_ARGV_RE = re.compile(WINDOW_OPTS + r"[\"']\s*,\s*[\"'](?P<v>[^\"']+)[\"']")
PLAUSIBLE_VALUE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}:\d{2}\S*)?|@\d+|today|midnight|yesterday|now"
    r"|\d+\.\w+|[{$]\{?[A-Za-z_][A-Za-z0-9_]*\}?)$")


def window_value(line: str):
    """(value, parsed_ok). `=` form is unconditional; space form is filtered."""
    m = VALUE_EQ_RE.search(line)
    if m:
        return m.group("v"), True
    m = VALUE_ARGV_RE.search(line)
    if m:
        return m.group("v"), True
    m = VALUE_SP_RE.search(line)
    if m and PLAUSIBLE_VALUE.match(m.group("v")):
        return m.group("v"), True
    if m:
        return None, None          # space form, English-looking -> prose
    return "", False               # option present, value unparseable -> FAIL CLOSED

UTC_QUALIFIED = re.compile(r"(Z$|[+-]00:?00$|^@\d+$)")
NAIVE_TS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")
BARE_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
RELATIVE_WORD = re.compile(r"^(today|midnight|yesterday|now|\d+\.\w+|\d+\s+\w+\s+ago)$")

SEARCH_ROOTS = ["scripts", ".claude/hooks", "backend"]

# THIS FILE EXCLUDES ITSELF, AND THAT IS A BOUND TO STATE, NOT A CONVENIENCE.
# Its section-[4] mutation cells are deliberately-sliding literals whose whole
# job is to prove the rule fires; scanned as production code they are 14 false
# findings. The exclusion was NOT needed until the file was COMMITTED -- an
# untracked file is invisible to `git ls-files` -- so the defect appeared the
# moment the guard shipped, which is exactly when a self-blind guard is worst.
#
# RESIDUAL, stated rather than hidden: a REAL sliding window introduced into
# this checker would not be caught by this checker. The mitigation is that its
# own fixtures are asserted in BOTH directions in [4], so a rule that stopped
# working fails there instead. Section [2] also asserts that the exclusion is
# exactly ONE file, so it cannot quietly grow into a general escape hatch.
SELF_REL = "scripts/qa/verify_no_sliding_windows_86_94.py"

# ── THE ALLOWLIST (criterion 4: the judgement is STATED, never silent) ───────
#
# Keyed on (path suffix, value). A relative window is permitted ONLY with a
# reason, and the reason has to say why a REPRODUCIBLE form would be wrong --
# not merely that the current form is convenient.
ALLOWLIST_REASONS = {
    ("backend/slack_bot/scheduler.py", "midnight"): (
        "LEGITIMATELY RELATIVE, AND ITS FIGURES HAVE BEEN QUOTED AS EVIDENCE -- "
        "corrected in phase-86.94 cycle 5. This builds the Slack 'shipped today' "
        "digest. A report about TODAY must move with today; pinning it would "
        "freeze the digest to one date and is straightforwardly wrong, so the "
        "WINDOW stays. But the criterion-4 judgement attached to it was FALSE. "
        "Cycles 2-4 asserted 'no COUNT produced by this window is quoted as "
        "evidence'. MEASURED, and the counterexample is a TRACKED file inside "
        "the very corpus [3b] scans: handoff/archive/misc/live_check_62.8.md:31 "
        "records a Slack read-back as verification evidence and quotes "
        "'\"*Shipped today*\" with 12 real commit lines' -- a count of exactly "
        "what _git_today() emitted through this window. "
        "CYCLE-6 CORRECTION: cycle 5 also cited 'Steps closed: 6' from that file "
        "as a second quoted figure. BOTH halves of that were wrong and both are "
        "removed rather than annotated. (i) It is not a quotation: the file's "
        "only 'Steps closed' text is at :36 and reads 'Steps closed: 61.1, 62.0, "
        "17.4, 62.3'; 'Steps closed: 6' was the regex `Steps closed:\\\\s*\\\\S` "
        "truncating at the first character of a step id, which I then printed "
        "inside quote marks -- the same paraphrase-as-quote defect corrected for "
        "frontend_route_inventory below. (ii) It is not this window's figure "
        "anyway: d['steps_flipped_today'] comes from _steps_closed_from_log() "
        "reading handoff/harness_log.md (scheduler.py:511-513), not from the "
        "--since-as-filter=midnight window at :501-507. The judgement stands on "
        "the '12 real commit lines' evidence alone. "
        "So the correct statement is the same one frontend_route_inventory "
        "carries: QUOTED, UNREPRODUCIBLE (the window has slid months past that "
        "digest and it can never be regenerated), and INERT (a closed step's "
        "read-back; nothing live depends on it). "
        "WHY THE EARLIER JUDGEMENT SURVIVED FOUR CYCLES: the cycle-4 probe was "
        "documented as derived from formatters.py:102-109 and could not match "
        "what that code emits. add() at formatters.py:71-76 renders "
        "_truncate(f'*{title}*\\n{body}'), so an asterisk sits between 'today' "
        "and the newline, and a probe anchored on 'Shipped today\\\\s*\\\\n' matches "
        "nothing. I had read the CALL SITE and not the RENDERER. The probes "
        "below were built by EXECUTING format_away_digest_sections and scoring "
        "against its real output as well as against the tracked artifact above."),
    ("scripts/qa/verify_decision_log_86_97.py", "{first_stamp}"): (
        "RUNTIME-DERIVED FROM A PINNED ARTIFACT, and judged acceptable. "
        "`first_stamp` is read from the FIRST LINE of "
        "handoff/logs/changelog-decisions.log, which is a full UTC "
        "`...Z` stamp, so the lower bound is an instant and not a clock "
        "expression. The rule cannot resolve it statically and therefore fails "
        "closed -- correctly -- so the judgement is recorded here instead. "
        "Criterion 4 disclosure: figures ARE derived from this window and ARE "
        "quoted (live_check_86.97.md quotes commits/lines/gap three times), but "
        "each is quoted WITH the clock time it was taken at, and the checker "
        "asserts a RELATIONSHIP (gap tracks the recursion-guard count) rather "
        "than any pinned number. The upper bound floats with HEAD by design: "
        "the point is that the relationship holds as the repo grows."),
    ("scripts/harness/frontend_route_inventory.py", "30.days"): (
        "SLIDING, AND ITS FIGURES HAVE BEEN QUOTED AS EVIDENCE -- corrected in "
        "phase-86.94 cycle 2. A rolling 30 days is the intended semantics for a "
        "'which routes changed recently' inventory, so the WINDOW is left as it "
        "is. But my cycle-1 judgement -- that no figure from it was ever quoted "
        "-- was FALSE, and it was false because the scan covered handoff/current "
        "only. Measured over the whole handoff tree: 49 files mention it (the same "
        "figure ALLOWLIST_CLAIMS pins as mentions_reviewed, so prose and instrument "
        "cannot drift apart), and "
        "handoff/archive/_quarantine_2026-04-21/phase-3.7.5-v22/experiment_results.md "
        "quotes figures derived from this exact window AS SUCCESS-CRITERIA "
        "EVIDENCE ('usage_source: git_activity_30d', '/portfolio 2 /login 1', "
        "'every_route_has_usage_count | PASS (12/12 integer opens_30d)'). Those "
        "figures are NOT regenerable -- the window has slid months past them. "
        "They are in an ARCHIVED, closed step and nothing live depends on them, "
        "which is why the window is still left rather than pinned; but the "
        "correct statement is 'quoted, unreproducible, and inert', not 'never "
        "quoted'."),
}

# THE JUDGEMENT IS A STRUCTURED COMMITMENT, NOT A SENTENCE.
#
# Cycle 2 checked criterion 4 by looking for the word "quoted" in the entry.
# That predicate passed for the TRUE entry, for the FALSE entry it replaced, and
# for the sentence "never quoted as evidence" -- it proved a sentence had been
# written, never that it was true, which is how a falsified claim survived into
# the live_check. Cycle 3's first replacement was worse: a deny-list of phrases
# fired on the entry's own REJECTION of one ("...not 'never quoted'"), i.e. the
# probe matched its own correction.
#
# So the claim is DATA. `quoted_as_evidence` is an explicit bool. A bool cannot
# be satisfied by vocabulary.
#
# CYCLE 4 REPLACED WHAT THE BOOL IS BOUND TO, and the previous binding is gone
# rather than annotated -- a correction must replace, not accompany.
#
# Cycle 3 bound the bool to `mentions_reviewed`, a pinned count of files whose
# text contains the member's FILENAME. Two things were wrong with it, both
# measured rather than argued:
#
#   1. WRONG PROPERTY. Criterion 4 asks whether a FIGURE DERIVED FROM the window
#      was quoted as evidence. `name in text` answers a different question, so
#      the bool it guarded stayed green when FALSIFIED. Measured 2026-08-17:
#      flipping frontend_route_inventory True->False, and scheduler False->True,
#      each left the FAIL set byte-identical. A wrong judgement shipped green in
#      BOTH directions -- the defect the cycle-3 evaluator named.
#   2. UNREPRODUCIBLE CORPUS -- the same class this whole step exists to close.
#      The count walked the WORKING TREE: 49,094 `.md` under handoff/, of which
#      only 5,167 are tracked; 43,927 (89.5%) are gitignored via `.gitignore:80`
#      (`handoff/archive/_quarantine_*/`). 45 of frontend_route_inventory's 50
#      hits were in the ignored quarantine, and the allowlist's own smoking-gun
#      citation below is ITSELF gitignored -- so on a fresh clone the number
#      differed and the evidence for the bool was absent. A count over "whatever
#      .md happens to be on this disk" is a number about a MACHINE exactly as
#      `--since=<bare date>` is a number about a CLOCK.
#      In practice it behaved as a pure change-detector: it went RED inside the
#      very commit that recorded it green, because that commit added a park note
#      that merely NAMES the scripts, and RED again when the next session wrote
#      its own report. Neither quoted a figure. Google's SWE-book calls this
#      exact shape brittle -- a failure "in the face of an unrelated change ...
#      that does not introduce any real bugs" (abseil.io/resources/swe-book ch12).
#
# The binding is now the property itself. `figure_probes` are patterns for a
# figure PRODUCED BY THAT MEMBER'S WINDOW, each derived from the emitting
# expression in the member's own source -- never from an author's phrasing, which
# is the recall trap criterion 2 forbids for the enumeration. The check asserts
# `quoted_as_evidence == bool(hits)`, so a wrong bool now fails in both
# directions, while prose that merely names a file is inert. in-toto v1 states
# the general form: a claim binds to its subject BY DIGEST, and "Subjects are
# assumed to be immutable" -- a name over a mutable corpus is the binding that
# spec explicitly rejects.
#
# Drift detection is PRESERVED, not dropped: if a quoted figure appears or
# disappears, `bool(hits)` changes and the judgement RE-OPENS. What no longer
# re-opens it is someone writing a sentence.
ALLOWLIST_CLAIMS = {
    # backend/slack_bot/scheduler.py:501-507 `_git_today()` -> d["commits_today"];
    # rendered at formatters.py:102-109 THROUGH add() at :71-76, which wraps the
    # title as `*{title}*\n{body}`. These probes were scored against the output of
    # a real format_away_digest_sections() call, not against a reading of the
    # source -- the cycle-4 set was read off the call site alone and matched
    # neither the render nor the artifact that quotes it.
    #
    # NOT INCLUDED, deliberately: a bare `\*Shipped today\*`. It matches the same
    # single file, so it would not change the bool, but the section HEADER is not
    # a figure. Criterion 4 asks whether a FIGURE DERIVED FROM the window was
    # quoted; admitting the header would slide this back toward the name-mention
    # proxy cycle 4 removed.
    "backend/slack_bot/scheduler.py": {
        "quoted_as_evidence": True,
        # TWO PROBES WERE REMOVED IN CYCLE 6, both for the same reason: they were
        # not bound to a figure THIS WINDOW produces.
        #   `Steps closed:\s*\S` matched d["steps_flipped_today"], which comes
        #   from _steps_closed_from_log() reading handoff/harness_log.md
        #   (scheduler.py:511-513) -- NOT from the --since-as-filter=midnight
        #   window at :501-507. Keeping it meant the criterion-4 judgement for
        #   this member was sustainable on evidence the allowlisted window never
        #   emitted. Worse, the text I quoted for it, "Steps closed: 6", was the
        #   regex truncating "Steps closed: 61.1, 62.0, 17.4, 62.3" at the first
        #   character of a step id, and I printed that fragment inside quote
        #   marks as if it were a quotation -- the exact "paraphrase inside quote
        #   marks is not a quote" defect this file corrects elsewhere.
        #   `commits_today[...]` matched nothing in the render or the corpus.
        "figure_probes": [
            r'\d+\s+real commit lines',
            r'\*Shipped today\*[^\n]*\n\s*[-*]\s+[0-9a-f]{7,}',
        ],
        # Controls. The first two are VERBATIM output of a real
        # format_away_digest_sections() call; the third is the sentence in
        # handoff/archive/misc/live_check_62.8.md:31 that quotes this window's
        # counts as verification evidence.
        "probe_fixtures": [
            {"text": '"*Shipped today*" with 12 real commit lines',
             "source": "handoff/archive/misc/live_check_62.8.md"},
            {"text": "*Shipped today*\n- 8853e74c chore: auto-changelog hook entry",
             "source": "scripts/qa/fixtures/shipped_today_render_86_94.txt"},
        ],
    },
    # scripts/qa/verify_decision_log_86_97.py -- window `--since={first_stamp}`.
    # The emitted figure is the "commits=N decision lines=N gap=N" triple and the
    # recursion-guard count printed beside it.
    "scripts/qa/verify_decision_log_86_97.py": {
        "quoted_as_evidence": True,
        "figure_probes": [
            r'commits=\d+\s+decision lines=\d+\s+gap=\d+',
            r'commits matching the recursion guard=\d+',
        ],
        # Controls: verbatim lines this checker prints (live_check_86.97.md:71).
        "probe_fixtures": [
            {"text": "commits=51  decision lines=26  gap=25",
             "source": "handoff/current/live_check_86.97.md"},
            {"text": "commits matching the recursion guard=26",
             "source": "handoff/current/live_check_86.97.md"},
        ],
    },
    # scripts/harness/frontend_route_inventory.py -- window `--since=30.days`.
    # Emitted figures are the per-route `opens_30d` counts and the `usage_source`
    # tag naming the window that produced them.
    "scripts/harness/frontend_route_inventory.py": {
        "quoted_as_evidence": True,
        "figure_probes": [
            r'"usage_source":\s*"git_activity_30d"',
            r'\d+/\d+ integer opens_30d',
            r'opens_30d=\d+',
        ],
        # Controls: verbatim strings from handoff/archive/phase-4.7.0/.
        "probe_fixtures": [
            {"text": '"usage_source": "git_activity_30d"',
             "source": "handoff/archive/phase-4.7.0/experiment_results.md"},
            {"text": "every_route_has_usage_count | PASS (12/12 integer opens_30d)",
             "source": "handoff/archive/phase-4.7.0/experiment_results.md"},
            {"text": "No route has opens_30d=0 in this window",
             "source": "handoff/archive/phase-4.7.0/experiment_results.md"},
        ],
    },
}
ALLOWLIST = ALLOWLIST_REASONS

# ── KNOWN MEMBER (criterion 3): a scan that cannot find this FAILS ───────────
# The pre-86.91 form of the replay is recoverable from git and is the canonical
# positive control: it is the exact defect this step exists to close, so a scan
# blind to it is a scan that proves nothing.
KNOWN_MEMBER_REF = "06c3265f"
KNOWN_MEMBER_PATH = "scripts/qa/replay_changelog_rule_86_68.py"


def tracked_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files", *SEARCH_ROOTS], cwd=REPO,
                         capture_output=True, text=True, check=False).stdout
    return [REPO / p for p in out.splitlines()
            if p.endswith((".py", ".sh")) and p.strip()
            and not p.endswith(SELF_REL)]


CONST_RE_T = "^\\s*{name}\\s*=\\s*[\"'](?P<lit>[^\"']+)[\"']"


def resolve(value: str, text: str) -> tuple[str, str | None]:
    """Resolve `{NAME}` / `$NAME` to a module-level string literal in the same file.

    FAILS CLOSED. The first version of this checker returned REPRODUCIBLE for
    anything interpolated, on the reasoning that the value is "decided at the
    call site". That is a fail-OPEN default and it hid the very defect this step
    was filed for: `--since={CORPUS_SINCE}` looked clean while CORPUS_SINCE was
    the TZ-naive `2026-08-11T00:00:00` that measures 766 in Oslo and 846 in
    Seoul. A rule that cannot see through one level of indirection is not a rule
    about windows, it is a rule about spelling.
    """
    m = re.match(r"^[{$]\{?([A-Za-z_][A-Za-z0-9_]*)\}?$", value)
    if not m:
        return value, None
    name = m.group(1)
    cm = re.search(CONST_RE_T.format(name=re.escape(name)), text, re.MULTILINE)
    if not cm:
        return value, None
    return cm.group("lit"), name


def classify(value: str) -> tuple[str, str]:
    """(verdict, reason) for one window value. Verdict in REPRODUCIBLE/SLIDING."""
    if "{" in value or value.startswith("$"):
        # Unresolvable indirection -- fail CLOSED, never open.
        return "SLIDING", (f"indirection {value!r} could not be resolved to a literal "
                           "in the same file -- failing closed rather than assuming "
                           "the call site got it right")
    if UTC_QUALIFIED.search(value):
        return "REPRODUCIBLE", "UTC-qualified instant (Z / +0000 / @epoch)"
    if NAIVE_TS.match(value):
        return "SLIDING", ("TZ-NAIVE timestamp -- pinned to the clock but not to a "
                           "timezone; measured 766 (Oslo/NY/UTC) vs 846 (Seoul)")
    if BARE_DATE.match(value):
        return "SLIDING", "bare date -- git applies the CURRENT TIME OF DAY"
    if RELATIVE_WORD.match(value):
        return "SLIDING", f"relative expression {value!r} -- resolves against now"
    return "SLIDING", f"unrecognised form {value!r} -- fails closed"


DOC_DELIMS = ('\"\"\"', "\'\'\'")


def strip_docstrings(text: str) -> str:
    """Blank out triple-quoted blocks, preserving line numbering.

    `is_prose` only knew about `#` lines. Module and function docstrings are
    a THIRD comment form, and this file's own docstring quotes a bare-date
    window while EXPLAINING the defect -- so an unstripped scan reported its
    own explanation as findings. Lines are blanked rather than removed so
    reported line numbers still point at the real source.
    """
    out, in_block, delim = [], False, ''
    for line in text.splitlines():
        if not in_block:
            hit = next((d for d in DOC_DELIMS if d in line), None)
            # An ASSIGNED triple-quoted string is DATA, not a docstring, and it
            # can be handed straight to subprocess. Blanking it created a false
            # NEGATIVE that the H2 docstring fix itself opened: `CMD = """git log
            # --since=<bare date>"""` disappeared from the scan. A docstring is a
            # bare expression statement, so a line with `=` before the delimiter
            # is not one.
            if hit is not None and '=' in line.split(hit)[0]:
                hit = None
            if hit is not None:
                if line.count(hit) < 2:
                    in_block, delim = True, hit
                line = ''
            out.append(line)
        else:
            if delim in line:
                in_block = False
            out.append('')
    return '\n'.join(out)


def is_prose(line: str) -> bool:
    """A comment line -- PROSE, not a window site.

    THIS FILE IS THE WORST CASE FOR AN UNANCHORED SCAN AND SO IS ITS SUBJECT.
    `replay_changelog_rule_86_68.py` documents this very defect in a comment
    block that quotes `--since=2026-08-11` in backticks; the first version of
    this scanner dutifully flagged that PROSE as two SLIDING sites. A checker
    that matches its own documentation reports defects that do not exist and
    trains its reader to ignore it.

    Stripping is therefore mandatory -- and, because a stripper that quietly
    does nothing looks identical to a correct one, section [4] proves this
    function is live in BOTH directions before any result above is believed.
    """
    s = line.lstrip()
    return s.startswith(("#", "*", "//"))


def scan_text(rel: str, text: str) -> list[tuple[str, int, str, str, str]]:
    """The scan, over TEXT rather than a path, so section [4] can feed it mutants
    through the SAME code path the shipped enumeration uses."""
    found = []
    if True:
        _lines = strip_docstrings(text).splitlines()
        for i, line in enumerate(_lines, 1):
            if is_prose(line) or not WINDOW_RE.search(line):
                continue
            # THE CLASS IS *GIT* REVISION-RANGE WINDOWS. Widening the option
            # pattern to the argv-list form made it match `argparse`
            # definitions too -- `ap.add_argument("--before", default=None)` is a
            # CLI flag for a non-git tool, not a window. A window site must
            # therefore have `git` in view: same line, or within the 3 lines
            # above, which covers a multi-line argv list without swallowing an
            # argparse block.
            # RESIDUAL, STATED: a git argv list that spreads the word "git" more
            # than 3 lines above its window option is not matched.
            _ctx = "\n".join(_lines[max(0, i - 4):i])
            if "git" not in _ctx:
                continue
            raw, _parsed = window_value(line)
            if raw is None:
                continue      # space form with an English-looking value: prose
            if not raw:
                # FAIL CLOSED. The first version `continue`d here, which meant a
                # window whose value the pattern could not parse was SILENTLY
                # SKIPPED -- a fail-OPEN inside the module whose central claim is
                # that it fails closed.
                #
                # THE ORIGINAL EXAMPLE FOR THIS COMMENT IS NO LONGER TRUE and is
                # replaced rather than left standing beside a correction. It read:
                # "Measured: `--since 2026-08-11` (space form) matched WINDOW_RE
                # but not VALUE_RE, so it vanished." That stopped being the case
                # when PLAUSIBLE_VALUE landed -- window_value() now returns
                # ('2026-08-11', True) for the space form, which takes the normal
                # value path. Re-measured 2026-08-17, the shapes that actually
                # reach this branch are: an argv list whose value is a VARIABLE
                # (`["git","log","--since", win]`), the f-string-element form,
                # `--since=` with an empty value, and `--after` + variable. The
                # argv-with-variable form is a realistic idiom, so this is an
                # uncovered branch and not a corner case; section [4] now has
                # cells for it.
                found.append((rel, i, "<unparsed>", "SLIDING",
                              ("a window option was found but its value could not be "
                               "parsed -- failing closed rather than skipping")))
                continue
            value, via = resolve(raw, text)
            verdict, reason = classify(value)
            if via:
                reason = f"via {via} = {value!r}; {reason}"
            found.append((rel, i, raw, verdict, reason))
    return found


def scan(files: list[Path]) -> list[tuple[str, int, str, str, str]]:
    """(relpath, lineno, value, verdict, reason) for every window site found."""
    out = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        out.extend(scan_text(str(f.relative_to(REPO)), text))
    return out


# ── [1] THE RULE FINDS ITS OWN KNOWN MEMBER (criterion 3) ────────────────────
print("\n[1] KNOWN-MEMBER RECALL -- the pre-86.91 form, recovered from git (criterion 3)\n")

blob = subprocess.run(["git", "show", f"{KNOWN_MEMBER_REF}:{KNOWN_MEMBER_PATH}"],
                      cwd=REPO, capture_output=True, text=True, check=False)
check("[1] the pre-86.91 blob is recoverable from git", blob.returncode == 0,
      f"git show {KNOWN_MEMBER_REF}:{KNOWN_MEMBER_PATH} failed -- the control is gone, "
      "so recall cannot be demonstrated and this gate FAILS rather than skipping")

if blob.returncode == 0:
    hits = []
    for i, line in enumerate(strip_docstrings(blob.stdout).splitlines(), 1):
        if is_prose(line) or not WINDOW_RE.search(line):
            continue
        val, _parsed = window_value(line)
        if val:
            hits.append((i, val, *classify(val)))
    check("[1] the rule FINDS a window site in the pre-86.91 blob", bool(hits),
          "the scan is blind to the very defect this step exists to close")
    sliding = [h for h in hits if h[2] == "SLIDING"]
    check("[1] and classifies it SLIDING", bool(sliding),
          f"classified {[(h[1], h[2]) for h in hits]}")
    for ln, val, verdict, reason in hits:
        print(f"       {KNOWN_MEMBER_REF}:{ln}  {val!r}  -> {verdict}: {reason}")

# ── [2] ENUMERATE THE LIVE CLASS (criterion 2) ──────────────────────────────
print("\n[2] ENUMERATION of live window sites, by the written-down rule (criterion 2)\n")

FILES = tracked_files()
_all_py_sh = [p for p in subprocess.run(["git", "ls-files", *SEARCH_ROOTS],
                                        cwd=REPO, capture_output=True, text=True,
                                        check=False).stdout.splitlines()
              if p.endswith((".py", ".sh"))]
check("[2] the self-exclusion covers exactly ONE file (this checker), so it "
      "cannot grow into a general escape hatch",
      len(_all_py_sh) - len(FILES) == 1,
      f"tracked={len(_all_py_sh)} scanned={len(FILES)} -- excluded "
      f"{len(_all_py_sh) - len(FILES)} files, expected exactly 1")
check("[2] the file set is non-empty (a scan over nothing is not a clean bill "
      "of health)", len(FILES) > 0, "git ls-files returned nothing")

SITES = scan(FILES)
check("[2] the rule finds at least one live window site", bool(SITES),
      "the rule matched NOTHING across the whole tree -- it is broken, not the tree")

for rel, ln, val, verdict, reason in sorted(SITES):
    key = next((k for k in ALLOWLIST if rel.endswith(k[0]) and k[1] == val), None)
    tag = "ALLOWED" if key else verdict
    print(f"       {rel}:{ln}  {val!r}  -> {tag}")
    if verdict == "SLIDING" and not key:
        print(f"           reason: {reason}")

unlisted = [(rel, ln, val, reason) for (rel, ln, val, verdict, reason) in SITES
            if verdict == "SLIDING"
            and not any(rel.endswith(k[0]) and k[1] == val for k in ALLOWLIST)]

check("[2] every SLIDING site is either fixed or carries a RECORDED REASON in "
      "the allowlist", not unlisted,
      "; ".join(f"{r}:{ln} {v!r} -- {why}" for r, ln, v, why in unlisted)
      + " || classify the new member deliberately; do not widen the rule")

# ── [3] THE ALLOWLIST IS NOT A DUMPING GROUND ───────────────────────────────
print("\n[3] ALLOWLIST HYGIENE\n")

live_keys = {(k[0], k[1]) for k in ALLOWLIST
             for (rel, _ln, val, _v, _r) in SITES
             if rel.endswith(k[0]) and k[1] == val}
stale = [k for k in ALLOWLIST if k not in live_keys]
check("[3] no allowlist entry is stale (each still matches a real site)",
      not stale, f"stale entries: {stale} -- an allowlist that outlives its site "
                 "quietly permits a future re-introduction at the same path")
for k, why in ALLOWLIST.items():
    check(f"[3] the entry for {k[0]} states a reason", len(why) > 80,
          "an allowlist entry without a stated reason is a silent exemption")

# ── [3b] CRITERION 4 -- was any figure from a SLIDING member ever QUOTED? ────
#
# The allowlist entries ASSERT that no figure derived from them is quoted as
# evidence. An assertion in a comment is not a check, and shipping one that
# nothing executes is how a claim rots. So the grep those entries refer to is
# RUN HERE, against the masterplan, the handoff tree and the CHANGELOG.
print("\n[3b] CRITERION 4 -- do any quoted figures derive from a SLIDING member?\n")

QUOTE_SURFACES = [".claude/masterplan.json", "CHANGELOG.md"]
# THE WHOLE handoff TREE, not just current/. The first version scanned
# handoff/current only, and on that evidence I recorded a judgement that was
# measurably FALSE: an ARCHIVED experiment_results quotes figures from the
# 30-day window as success-criteria evidence. A criterion-4 judgement is only
# as good as the corpus it was taken over.
QUOTE_DIRS = ["handoff"]

# THE CORPUS IS THE TRACKED SET, and that is a correctness requirement, not tidying.
# Walking the working tree made this section's own figures unreproducible in
# exactly the class the step exists to close: 89.5% of the `.md` under handoff/
# is gitignored (`.gitignore:80`), so the same command answered differently on a
# fresh clone than on this laptop. `git ls-files` is the enumeration rule, it is
# the same rule `tracked_files()` already uses for the scan itself, and it is
# stated here rather than left implicit.
#
# CONSEQUENCE, STATED: the quarantined archive is no longer evidence. The cycle-2
# citation for frontend_route_inventory pointed at
# handoff/archive/_quarantine_2026-04-21/phase-3.7.5-v22/experiment_results.md,
# which is gitignored. Its figures are ALSO quoted in tracked archives
# (handoff/archive/phase-4.7.0/, phase-4.7.1/), so the True judgement stands on
# tracked evidence -- measured, not assumed: 5 tracked hits across 3 files.
def _tracked_set() -> set[str]:
    out = subprocess.run(["git", "ls-files"], cwd=REPO,
                         capture_output=True, text=True, check=False).stdout
    return {ln for ln in out.splitlines() if ln.strip()}


_TRACKED = _tracked_set()

MENTIONS = {}
for _rel in QUOTE_SURFACES:
    _p = REPO / _rel
    if _p.exists() and _rel in _TRACKED:
        MENTIONS[_rel] = _p.read_text(encoding="utf-8", errors="replace")
for _d in QUOTE_DIRS:
    for _p in sorted((REPO / _d).rglob("*.md")):
        _r = str(_p.relative_to(REPO))
        if _r in _TRACKED:
            MENTIONS[_r] = _p.read_text(encoding="utf-8", errors="replace")

check("[3b] the quote corpus is the TRACKED set (a working-tree walk is a number "
      "about a machine, the same defect class this step closes)",
      bool(_TRACKED) and all(r in _TRACKED for r in MENTIONS),
      "the corpus contains untracked files, so this section is not reproducible "
      "on a fresh clone")

check("[3b] the quote corpus is non-empty (an empty grep proves nothing)",
      sum(len(v) for v in MENTIONS.values()) > 10000,
      f"only {sum(len(v) for v in MENTIONS.values())} chars gathered")

# THIS STEP'S OWN ARTIFACTS ARE EXCLUDED, and the exclusion is STATED rather
# than quietly applied: 86.94's contract, brief, live_check and
# experiment_results necessarily discuss every member by name, so counting them
# would guarantee a hit for every member and make the check meaningless. What
# criterion 4 asks is whether a figure was quoted AS EVIDENCE somewhere else.
#
# The check does NOT assert absence. An earlier version did, using
# "is the script name mentioned anywhere" as a proxy, and it immediately
# falsified two of my own allowlist claims -- correctly as to the proxy, and
# misleadingly as to the question, because every hit was descriptive prose
# rather than a quoted count. Criterion 4 asks for a JUDGEMENT to be STATED, so
# the check surfaces the mention sites for audit and requires the entry to have
# stated one.
SELF = "86.94"


def figure_sites(claim: dict) -> list[tuple[str, str, str]]:
    """(relpath, pattern, matched text) for every QUOTED FIGURE of this member.

    The predicate criterion 4 actually asks about. Compare `name in text`, which
    answers "was this file talked about" and is satisfied by any sentence.
    """
    out = []
    for rel, text in MENTIONS.items():
        if SELF in rel:
            continue
        for pat in claim.get("figure_probes", []):
            m = re.search(pat, text)
            if m:
                out.append((rel, pat, m.group(0)[:70].replace("\n", "\\n")))
    return out


for (_path_suffix, _val), _entry in ALLOWLIST.items():
    _name = Path(_path_suffix).name
    _claim = ALLOWLIST_CLAIMS.get(_path_suffix)
    _named = [rel for rel, text in MENTIONS.items() if _name in text and SELF not in rel]

    check(f"[3b] {_name}: the criterion-4 judgement is a STRUCTURED claim, not a "
          "sentence (quoted_as_evidence is an explicit bool)",
          isinstance(_claim, dict) and isinstance(_claim.get("quoted_as_evidence"), bool),
          "no machine-readable claim -- a prose predicate is satisfiable by vocabulary "
          "and cannot be contradicted by the measurement")

    if isinstance(_claim, dict):
        # A claim with no probe cannot be contradicted by anything. An
        # unfalsifiable entry is the defect, so it fails rather than passing
        # vacuously.
        check(f"[3b] {_name}: the claim carries FIGURE PROBES, so it can be "
              "contradicted by a measurement",
              bool(_claim.get("figure_probes")),
              "quoted_as_evidence with no probe is unfalsifiable -- exactly the "
              "isinstance-only state cycle 3 shipped")

        # EVERY PROBE MUST BE DEMONSTRABLY LIVE, and this is the cycle-5 fix for
        # the hole that let a FALSE judgement survive four cycles.
        #
        # `quoted_as_evidence == bool(hits)` binds the bool to a measurement, but
        # for a FALSE claim it is satisfied precisely when the probes match
        # NOTHING -- so a probe set that is silently dead is byte-indistinguishable
        # from a correct measurement. That is exactly what happened: the cycle-4
        # scheduler probes could not match the digest their own comment cited, the
        # claim was False, and the check went green over a tracked counterexample.
        # Substituting a never-matching literal for those probes ALSO left the
        # guard at 68/0.
        #
        # So each probe carries POSITIVE CONTROLS: fixtures taken from the real
        # emitted text (for scheduler, the output of an actual
        # format_away_digest_sections() call, plus the sentence in the artifact
        # that quotes it). A probe matching none of its own fixtures is dead and
        # FAILS here, whatever the bool says and whichever way the corpus happens
        # to fall.
        _fixtures = _claim.get("probe_fixtures", [])
        check(f"[3b] {_name}: the probe set carries POSITIVE CONTROLS",
              bool(_fixtures),
              "without fixtures a dead probe set is indistinguishable from a "
              "measured absence")

        # THE CONTROL MUST BE PROVENANCED, NOT AUTHORED -- cycle 6.
        #
        # Cycle 5 shipped fixtures as bare strings and claimed the check
        # "protects any future member claimed False". The cycle-5 Q/A falsified
        # that by execution: a probe and a fixture CO-WRITTEN from the same
        # misreading (`QQ_SELF_WRITTEN_TOKEN_\d+` matched only by the invented
        # fixture `QQ_SELF_WRITTEN_TOKEN_7`) left the guard at a clean 74/0. The
        # provenance was stated in a source comment and enforced by NOTHING, so
        # the control could be manufactured to fit the probe -- which is the same
        # shape as the defect it was added to catch.
        #
        # Each fixture now names a TRACKED FILE and the text must actually be in
        # it, so a control cannot be invented. Fixture sources are deliberately
        # allowed to sit OUTSIDE the [3b] quote corpus (e.g. a generated render
        # under scripts/qa/fixtures/): the fixture proves the probe recognises
        # the figure SHAPE, while the corpus search answers the different
        # question of whether that figure was quoted AS EVIDENCE. Conflating the
        # two would force every claim to True by construction.
        _badsrc = []
        for _f in _fixtures:
            if not isinstance(_f, dict) or not _f.get("source") or not _f.get("text"):
                _badsrc.append(f"{_f!r}: not a {{text, source}} pair")
                continue
            _sp = _f["source"]
            if _sp not in _TRACKED:
                _badsrc.append(f"{_sp}: NOT TRACKED")
            elif _f["text"] not in (REPO / _sp).read_text(encoding="utf-8", errors="replace"):
                _badsrc.append(f"{_sp}: does not contain the fixture text")
        check(f"[3b] {_name}: every positive control is PROVENANCED -- its text is "
              "actually present in the tracked file it names",
              not _badsrc,
              f"unprovenanced control(s): {_badsrc} -- a fixture that cannot be "
              "traced to real text can be manufactured to fit the probe, which is "
              "the defect this control exists to prevent")

        _ftexts = [_f["text"] for _f in _fixtures
                   if isinstance(_f, dict) and _f.get("text")]
        _dead = [p for p in _claim.get("figure_probes", [])
                 if not any(re.search(p, f) for f in _ftexts)]
        check(f"[3b] {_name}: every figure probe matches at least one control, so "
              "none of them is silently dead",
              not _dead,
              f"probe(s) match NOTHING even in their own fixtures: {_dead} -- a "
              "probe that cannot fire cannot contradict the claim it guards")

        _figs = figure_sites(_claim)
        _files = sorted({r for r, _, _ in _figs})
        print(f"       {_name}: NAMED in {len(_named)} tracked file(s); "
              f"a FIGURE it produced is QUOTED in {len(_files)}"
              + (": " + ", ".join(_files[:3]) if _files else ""))
        for _r, _p, _t in _figs[:3]:
            print(f"           {_r}  ~{_p}  -> {_t!r}")

        # THE BOOL IS NOW BOUND TO THE MEASUREMENT, in both directions. Flipping
        # it either way fails. Cycle 3's binding could not do this: with only an
        # isinstance check, True->False and False->True both shipped GREEN.
        check(f"[3b] {_name}: quoted_as_evidence={_claim.get('quoted_as_evidence')} "
              "matches the measured figure evidence",
              _claim.get("quoted_as_evidence") == bool(_figs),
              f"claim says {_claim.get('quoted_as_evidence')} but "
              f"{len(_figs)} quoted figure(s) were measured in {len(_files)} tracked "
              f"file(s) -- re-review and re-state the judgement. Sites: "
              f"{_files[:4]}")

    check(f"[3b] {_name}: the entry carries a stated REASON",
          len(_entry) > 80,
          "an allowlist entry without a stated reason is a silent exemption")

# ── [4] MUTATION -- the guard must go RED on a NEW sliding window (criterion 6)
#
# CONTROL FIRST. Every cell below is a differential against a clean tree, so if
# the tree were already dirty the "kill" would be meaningless.
print("\n[4] MUTATION -- a new sliding window must turn this guard RED (criterion 6)\n")

CLEAN = REPO / "scripts" / "qa" / "replay_changelog_rule_86_68.py"
CLEAN_TEXT = CLEAN.read_text(encoding="utf-8")
CLEAN_REL = "scripts/qa/replay_changelog_rule_86_68.py"

_clean_sliding = [h for h in scan_text(CLEAN_REL, CLEAN_TEXT) if h[3] == "SLIDING"]
check("[4] CONTROL: the tree's own replay has NO unlisted sliding window",
      not _clean_sliding,
      f"control is already dirty: {_clean_sliding} -- no kill below can be believed")

INJECTIONS = [
    # THE FIRST FIVE WERE FOUND SURVIVING BY THE CYCLE-1 Q/A, and three of them
    # (`--after`, `--before`, now-relative arithmetic) are named VERBATIM in this
    # step's own audit_basis -- so their absence was a recall failure against the
    # step's own filing, not a scope choice. They are cells now.
    ("argv-list-form", 'subprocess.run(["git", "log", "--since", "2026-08-11"])',
     ("the ARGV-LIST spelling -- the repo's dominant git idiom, which the "
      "option regex missed entirely because a QUOTE follows the option name, "
      "so the site was INVISIBLE and the fail-closed path never fired")),
    ("argv-list-after", 'subprocess.run(["git", "log", "--after", "30.days"])',
     "the argv-list spelling of the --after synonym"),
    ("after-synonym", 'sh("git", "log", "--after=2026-08-11")',
     "`--after`, an EXACT synonym of --since (measured: same --max-age)"),
    ("before-synonym", 'sh("git", "log", "--before=2026-08-11")',
     "`--before`, the synonym of --until"),
    ("space-separated", 'sh("git", "log", "--since 2026-08-11")',
     ("the SPACE form, which the first rule matched as a line but not as a value "
      "and then silently skipped -- a fail-OPEN inside the fail-closed module")),
    ("now-relative-feeding-git",
     'w = (datetime.utcnow() - timedelta(days=30)).isoformat()\nsh("git", "log", f"--since={w}")',
     "now-relative arithmetic reaching a git window through an f-string"),
    ("bare-date-in-executed-string",
     'CMD = """git log --since=2026-08-11"""\nsubprocess.run(CMD, shell=True)',
     ("a window inside an ASSIGNED triple-quoted string -- data that gets executed, "
      "a false negative the docstring fix itself opened")),

    ("bare-date", 'sh("git", "log", "--since=2026-08-11")',
     "a bare date -- the original defect"),
    ("relative-days", 'sh("git", "log", "--since=30.days")',
     "a rolling N-day window"),
    ("today-word", 'sh("git", "log", "--since=today")',
     "`today`, which git does NOT resolve to midnight"),
    ("tz-naive-pin", 'sh("git", "log", "--since=2026-08-11T00:00:00")',
     "a pinned but TZ-NAIVE timestamp -- the shape phase-86.91 believed was a fix"),
]
# A KILL MUST NAME THE MECHANISM THAT MADE IT, or the cell cannot tell a real
# detection from a coincidental one. Every cell used to assert only
# `bool(hits)` filtered on h[3] == "SLIDING", never the VALUE field h[2] -- so a
# cell was green whether classify() fired on a parsed value or the fail-closed
# <unparsed> branch caught it, and the two are different claims about the rule.
# The mutation literature is explicit that this distinction is the whole game:
# assertion kills "imply that the test oracles actually capture the correct
# program behaviour", while other kills "may only show coincidental impacts of
# the mutation" (arXiv:2306.02319, corroborated arXiv:2511.11999).
#
# MEASURED 2026-08-17, all 11 injections differenced against the clean control:
# every one is killed by classify() on a PARSED value; none reaches the
# fail-closed branch. So each asserts value != "<unparsed>". This also corrects
# an attribution recorded in cycle 3: neutralising VALUE_ARGV_RE (the argv
# VALUE-PARSE leg) leaves every cell GREEN, because argv sites then fall through
# to the fail-closed branch and are still flagged. What actually kills the two
# argv cells is WINDOW_RE's argv alternative -- the VISIBILITY leg. Neutralising
# that turns exactly `argv-list-form` and `argv-list-after` red, and nothing else.
for mid, injected, why in INJECTIONS:
    mutated = CLEAN_TEXT + "\n" + injected + "\n"
    hits = [h for h in scan_text(CLEAN_REL, mutated) if h[3] == "SLIDING"]
    check(f"[4] {mid}: KILLED -- introducing {why} is flagged SLIDING", bool(hits),
          "the guard stayed green after a sliding window was introduced")
    check(f"[4] {mid}: the kill is attributable -- classify() fired on a PARSED "
          "value, not the fail-closed catch-all",
          bool(hits) and all(h[2] != "<unparsed>" for h in hits),
          f"reported value(s) {[h[2] for h in hits]} -- a cell that cannot name "
          "its mechanism cannot distinguish a real detection from a coincidental one")

# ── FAIL-CLOSED BRANCH (scan_text): its own cells ───────────────────────────
#
# The module's central claim is that an unparseable window FAILS CLOSED rather
# than being skipped. Until now that branch had ZERO cells: restoring the
# fail-OPEN `continue` in its place left all assertions green, so the claim was
# asserted by a comment and executed by nothing. Each cell below asserts the
# reported value IS "<unparsed>", which is the only signal that distinguishes
# this branch from classify(); without it the cells would be satisfiable by the
# ordinary value path and would prove nothing about fail-closing.
FAILCLOSED_INJECTIONS = [
    ("fc-argv-variable", 'subprocess.run(["git", "log", "--since", win])',
     "an argv list whose window value is a VARIABLE -- the repo's dominant git "
     "idiom with a runtime-computed bound, which no static value pattern can read"),
    ("fc-argv-fstring", 'subprocess.run(["git", "log", "--since", f"{lo}"])',
     "the f-string-element form of the same argv idiom"),
    ("fc-empty-equals", 'sh("git", "log", "--since=" + WINDOW)',
     "`--since=` built by concatenation, so the value is absent at the option"),
    ("fc-after-variable", 'subprocess.run(["git", "log", "--after", cutoff])',
     "the --after synonym with a variable value"),
]
for mid, injected, why in FAILCLOSED_INJECTIONS:
    mutated = CLEAN_TEXT + "\n" + injected + "\n"
    hits = [h for h in scan_text(CLEAN_REL, mutated) if h[3] == "SLIDING"]
    check(f"[4] {mid}: KILLED -- {why} is flagged SLIDING", bool(hits),
          "an unparseable window was SKIPPED -- the module fails OPEN")
    check(f"[4] {mid}: it is the FAIL-CLOSED branch that fired (value == '<unparsed>')",
          any(h[2] == "<unparsed>" for h in hits),
          f"reported value(s) {[h[2] for h in hits]} -- this cell is being satisfied "
          "by the ordinary value path, so it does not cover the fail-closed branch "
          "and restoring the fail-OPEN `continue` would leave it green")

# A REPRODUCIBLE form must NOT be flagged, or the guard is just noise and will
# be switched off. A detector that flags everything discriminates nothing.
_ok = [h for h in scan_text(CLEAN_REL, CLEAN_TEXT + '\nsh("git","log","--after=2026-08-11T00:00:00Z")\n')
       if h[3] == "SLIDING"]
check("[4] NEGATIVE CONTROL: a UTC-qualified window is NOT flagged", not _ok,
      f"flagged a reproducible form: {_ok} -- the rule cannot discriminate")

# SHELL COVERAGE. The rule's file set is *.py AND *.sh (the hooks are shell), but
# every cell above mutates Python. A guard demonstrated on one language only is
# demonstrated on half its scope, and `#` comments plus bare `git log` calls look
# different enough in shell to be worth an explicit cell rather than an argument.
_sh_bad = 'git log --since=2026-08-11 --format=%H\n'
_sh_ok = 'git log --since=2026-08-11T00:00:00Z --format=%H\n'
_sh_prose = '# we used to run: git log --since=2026-08-11\n'
check("[4] SHELL: a sliding window in a .sh body is flagged",
      any(h[3] == "SLIDING" for h in scan_text("x.sh", _sh_bad)),
      "the rule is Python-only in practice -- the hooks are shell and would be uncovered")
check("[4] SHELL NEGATIVE CONTROL: a UTC-qualified shell window is NOT flagged",
      not any(h[3] == "SLIDING" for h in scan_text("x.sh", _sh_ok)),
      "flags a reproducible shell form -- cannot discriminate")
check("[4] SHELL: a window in a `#` comment is not reported as a site",
      not scan_text("x.sh", _sh_prose),
      "shell prose is being matched")

# The comment-stripper, in BOTH directions. Without this the scan could be
# silently matching nothing, or silently matching prose.
# The fixture must mention `git`, because a window site now requires git in
# view (see scan_text). A fixture that cannot satisfy the rule's own
# precondition tests nothing -- caught when the proximity rule landed.
_prose = '# the defect was `git log --since=2026-08-11`, a bare date\n'
check("[4] STRIPPER: a window quoted in PROSE is not reported as a site",
      not scan_text("x.py", _prose),
      "the scanner matches its own documentation -- it will report defects that "
      "do not exist")
# The DOCSTRING stripper needs its own pair: `#` and triple-quoted blocks are
# different code paths, and this file's own module docstring is what exposed
# the gap -- it quotes a bare-date window while explaining the defect.
_doc_prose = '"""\ndocs: git log --since=2026-08-11\n"""\n'
# ... and the sibling case the cycle-1 Q/A found: an ASSIGNED triple-quoted
# string is DATA that can be executed, so it must still be scanned.
_doc_assigned = 'CMD = """git log --since=2026-08-11"""\n'
_doc_code = 'sh("git","log","--since=2026-08-11")\n'
check("[4] DOCSTRING STRIPPER: a window inside a triple-quoted block is not a site",
      not scan_text("x.py", _doc_prose),
      "docstrings are not stripped -- the checker reports its own explanation")
check("[4] DOCSTRING STRIPPER CONTROL: the same window as CODE *is* reported",
      bool(scan_text("x.py", _doc_code)),
      "the control never landed, so the docstring check above proves nothing")
check("[4] EXECUTED triple-quoted string IS scanned (an assigned block is data, "
      "not a docstring)",
      any(h[3] == "SLIDING" for h in scan_text("x.py", _doc_assigned)),
      "blanking assigned triple-quoted strings creates a false negative -- the "
      "H2 docstring fix opened exactly this hole")

check("[4] STRIPPER CONTROL: the same text as CODE *is* reported",
      bool(scan_text("x.py", _prose.lstrip("# "))),
      "the control never landed, so the stripper check above proves nothing")

# The indirection resolver, in both directions -- this is the leg that found the
# real defect, so it must be shown live rather than assumed.
_ind_bad = 'CONST = "2026-08-11T00:00:00"\nsh("git","log",f"--since={CONST}")\n'
_ind_ok = 'CONST = "2026-08-11T00:00:00Z"\nsh("git","log",f"--since={CONST}")\n'
check("[4] RESOLVER: sees THROUGH indirection to a TZ-naive literal",
      any(h[3] == "SLIDING" for h in scan_text("x.py", _ind_bad)),
      "the resolver failed open on an interpolated value -- the exact miss that "
      "let the phase-86.91 'fix' look clean")
check("[4] RESOLVER CONTROL: a UTC-qualified literal behind the same indirection "
      "is NOT flagged",
      not any(h[3] == "SLIDING" for h in scan_text("x.py", _ind_ok)),
      "the resolver flags regardless of the value -- it is not reading the literal")

# ── [5] SCOPE DISCLOSURE -- the class this gate does NOT cover ──────────────
#
# The rule's declared scope is GIT REVISION-RANGE windows, which is what this
# step's title, audit_basis and every measured defect are about. A now-relative
# expression that never reaches a git window -- a SQL `CURRENT_DATE()`, a pandas
# date filter, a bare `datetime.now() - timedelta(...)` -- is OUT of that scope
# and is NOT gated.
#
# That bound is REPORTED rather than left silent, because "the guard is green"
# would otherwise be read as "the repo has no sliding measurement windows", and
# the cycle-1 Q/A measured that it does. Report-only on purpose: gating this
# surface would flood on legitimate uses (schedulers, digests, TTLs) and the gate
# would be switched off, which is the same reasoning that made the git rule an
# allowlist rather than a ban.
print("\n[5] SCOPE DISCLOSURE -- now-relative windows OUTSIDE the git surface\n")

NONGIT_RE = re.compile(
    r"(datetime\.(now|utcnow)\(\)|date\.today\(\)|CURRENT_DATE\(\)|timedelta\s*\()")
_nongit = []
for _f in FILES:
    try:
        _t = strip_docstrings(_f.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError):
        continue
    for _i, _l in enumerate(_t.splitlines(), 1):
        if is_prose(_l) or not NONGIT_RE.search(_l):
            continue
        _nongit.append((str(_f.relative_to(REPO)), _i))
_files = sorted({r for r, _ in _nongit})
print(f"       {len(_nongit)} now-relative expression(s) in {len(_files)} file(s), "
      f"NOT gated -- outside the git revision-range scope")
for _r in _files[:8]:
    print(f"         {_r}")
if len(_files) > 8:
    print(f"         ... and {len(_files) - 8} more")
check("[5] the scope bound is REPORTED, not silently omitted (the census ran and "
      "found a non-empty surface, so 'guard green' cannot be read as 'no sliding "
      "windows anywhere')", len(_nongit) > 0,
      "the census found nothing, which would mean either the repo is clean or the "
      "census is broken -- and those look identical, so it fails rather than passes")

print()
if _failures:
    print(f"FAILED: {_pass} passed, {len(_failures)} failed")
    for f in _failures:
        print(f"  - {f}")
    sys.exit(1)
print(f"ALL GREEN: {_pass} passed, 0 failed")
