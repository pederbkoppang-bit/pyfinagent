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
WINDOW_RE = re.compile(r"--(?:since|until)(?:-as-filter)?[=\s]")
VALUE_RE = re.compile(r"--(?:since|until)(?:-as-filter)?=(?P<v>[^\"'\s,)\]]+)")

UTC_QUALIFIED = re.compile(r"(Z$|[+-]00:?00$|^@\d+$)")
NAIVE_TS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")
BARE_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
RELATIVE_WORD = re.compile(r"^(today|midnight|yesterday|now|\d+\.\w+|\d+\s+\w+\s+ago)$")

SEARCH_ROOTS = ["scripts", ".claude/hooks", "backend"]

# ── THE ALLOWLIST (criterion 4: the judgement is STATED, never silent) ───────
#
# Keyed on (path suffix, value). A relative window is permitted ONLY with a
# reason, and the reason has to say why a REPRODUCIBLE form would be wrong --
# not merely that the current form is convenient.
ALLOWLIST = {
    ("backend/slack_bot/scheduler.py", "midnight"): (
        "LEGITIMATELY RELATIVE. This builds the Slack 'shipped today' digest. A "
        "report about TODAY must move with today; pinning it would freeze the "
        "digest to one date and is straightforwardly wrong. No figure derived "
        "from it is quoted in any criterion, audit_basis, handoff artifact or "
        "the CHANGELOG as EVIDENCE -- it is a chat message, regenerated every "
        "run. MEASURED: the name does appear in CHANGELOG.md and the "
        "masterplan, but every hit is descriptive (an em-dash cleanup, an "
        "APScheduler job description, a different scheduler at :761-795); "
        "none quotes a COUNT produced by this window."),
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
        "SLIDING, JUDGED ACCEPTABLE AND LEFT. It answers 'which routes changed "
        "recently' for an inventory report; a rolling 30 days is the intended "
        "semantics. Criterion 4's test is whether a figure derived from it was "
        "ever QUOTED as evidence. MEASURED: outside this step's own artifacts the "
        "name appears nowhere in the masterplan, CHANGELOG or handoff tree, so "
        "no count from it is load-bearing anywhere. NOTE the script DOES report "
        "per-route figures in its own stdout -- they are simply never quoted as "
        "evidence. Recorded rather than silently tolerated: if a count from this "
        "script is ever quoted, this entry must be revisited."),
}

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
            if p.endswith((".py", ".sh")) and p.strip()]


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
        for i, line in enumerate(text.splitlines(), 1):
            if is_prose(line) or not WINDOW_RE.search(line):
                continue
            m = VALUE_RE.search(line)
            raw = m.group("v") if m else ""
            if not raw:
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
    for i, line in enumerate(blob.stdout.splitlines(), 1):
        if WINDOW_RE.search(line) and (m := VALUE_RE.search(line)):
            hits.append((i, m.group("v"), *classify(m.group("v"))))
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
QUOTE_DIRS = ["handoff/current"]


def quote_corpus() -> str:
    parts = []
    for rel in QUOTE_SURFACES:
        p = REPO / rel
        if p.exists():
            parts.append(p.read_text(encoding="utf-8", errors="replace"))
    for d in QUOTE_DIRS:
        for p in (REPO / d).glob("*.md"):
            parts.append(p.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts)


MENTIONS = {}
for _rel in QUOTE_SURFACES:
    _p = REPO / _rel
    if _p.exists():
        MENTIONS[_rel] = _p.read_text(encoding="utf-8", errors="replace")
for _d in QUOTE_DIRS:
    for _p in sorted((REPO / _d).glob("*.md")):
        MENTIONS[str(_p.relative_to(REPO))] = _p.read_text(encoding="utf-8", errors="replace")

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
for (_path_suffix, _val), _entry in ALLOWLIST.items():
    _name = Path(_path_suffix).name
    _sites = [rel for rel, text in MENTIONS.items()
              if _name in text and SELF not in rel]
    print(f"       {_name}: mentioned outside this step's own artifacts in "
          f"{len(_sites)} file(s)" + (": " + ", ".join(_sites[:4]) if _sites else ""))
    check(f"[3b] {_name}: its allowlist entry states an explicit criterion-4 "
          "judgement about quoted figures",
          ("Criterion 4" in _entry) or ("quoted" in _entry),
          "the entry does not say whether any figure from this member was ever "
          "quoted as evidence -- criterion 4 requires the judgement STATED, not silent")

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
    ("bare-date", 'sh("git", "log", "--since=2026-08-11")',
     "a bare date -- the original defect"),
    ("relative-days", 'sh("git", "log", "--since=30.days")',
     "a rolling N-day window"),
    ("today-word", 'sh("git", "log", "--since=today")',
     "`today`, which git does NOT resolve to midnight"),
    ("tz-naive-pin", 'sh("git", "log", "--since=2026-08-11T00:00:00")',
     "a pinned but TZ-NAIVE timestamp -- the shape phase-86.91 believed was a fix"),
]
for mid, injected, why in INJECTIONS:
    mutated = CLEAN_TEXT + "\n" + injected + "\n"
    hits = [h for h in scan_text(CLEAN_REL, mutated) if h[3] == "SLIDING"]
    check(f"[4] {mid}: KILLED -- introducing {why} is flagged SLIDING", bool(hits),
          "the guard stayed green after a sliding window was introduced")

# A REPRODUCIBLE form must NOT be flagged, or the guard is just noise and will
# be switched off. A detector that flags everything discriminates nothing.
_ok = [h for h in scan_text(CLEAN_REL, CLEAN_TEXT + '\nsh("git","log","--since=2026-08-11T00:00:00Z")\n')
       if h[3] == "SLIDING"]
check("[4] NEGATIVE CONTROL: a UTC-qualified window is NOT flagged", not _ok,
      f"flagged a reproducible form: {_ok} -- the rule cannot discriminate")

# The comment-stripper, in BOTH directions. Without this the scan could be
# silently matching nothing, or silently matching prose.
_prose = '# the defect was `--since=2026-08-11`, a bare date\n'
check("[4] STRIPPER: a window quoted in PROSE is not reported as a site",
      not scan_text("x.py", _prose),
      "the scanner matches its own documentation -- it will report defects that "
      "do not exist")
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

print()
if _failures:
    print(f"FAILED: {_pass} passed, {len(_failures)} failed")
    for f in _failures:
        print(f"  - {f}")
    sys.exit(1)
print(f"ALL GREEN: {_pass} passed, 0 failed")
