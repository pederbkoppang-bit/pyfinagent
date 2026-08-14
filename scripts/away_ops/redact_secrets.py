#!/usr/bin/env python3
"""Redact credential-shaped strings from a session JSON before it is written to
a git-tracked path.

    python3 scripts/away_ops/redact_secrets.py <raw.json> > out.json
    python3 scripts/away_ops/redact_secrets.py --self-test

WHY THIS EXISTS
---------------
`run_away_session.sh` wrote the Claude CLI's raw `--output-format json` straight
to `handoff/away_ops/session_*.json`, which is TRACKED and PUSHED. On
2026-08-08..08-10 that published a live `sk-ant-oat01...` OAuth token to a public
remote for six days. The vector was an API error that echoes the header:

    "API Error: Header 'Authorization' has invalid value: 'Bearer <TOKEN>..."

The incident record inferred the leak was fixed because six later files were
clean. THAT INFERENCE WAS WRONG: there was no redaction in the producer at all,
so the channel was DORMANT, not closed. This closes it.

TWO DESIGN RULES, both learned from the incident:

1. **STRUCTURE IS PRESERVED.** `run_away_session.sh` greps the output for
   `"api_error_status": *401` to drive the credential-death latch. Redacting the
   file must not break that, so only string VALUES are rewritten and keys,
   numbers and booleans are untouched.

2. **THE SCAN MUST CROSS A HYPHEN.** The original scan that reported these files
   CLEAN used `sk-[A-Za-z0-9]{20,}`, which cannot match `sk-ant-oat01-...`
   because the character class excludes `-`. A false negative on a live
   credential is the worst possible outcome here, so every pattern below is
   hyphen-aware and the self-test pins that specific miss.
"""
from __future__ import annotations

import json
import re
import sys

#: Ordered, and each is hyphen-aware. `_` and `-` are inside every class.
PATTERNS = [
    # Anthropic: API keys AND OAuth tokens (sk-ant-api..., sk-ant-oat01-...).
    (re.compile(r"sk-ant-[A-Za-z0-9_-]{10,}"), "sk-ant-<REDACTED>"),
    # Generic bearer values -- catches a header echo even under a vendor prefix
    # this file has never seen.
    (re.compile(r"(?i)(Bearer\s+)[A-Za-z0-9._~+/=-]{20,}"), r"\1<REDACTED>"),
    # Slack bot/user/app tokens.
    (re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"), "xox<REDACTED>"),
    # OpenAI-style and other `sk-` keys NOT already caught above.
    (re.compile(r"sk-(?!ant-)[A-Za-z0-9_-]{20,}"), "sk-<REDACTED>"),
    # Google API keys.
    (re.compile(r"AIza[A-Za-z0-9_-]{30,}"), "AIza<REDACTED>"),
    # GitHub tokens.
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"), "gh_<REDACTED>"),
]


#: A credential can be WRAPPED across a newline, and every pattern above uses a
#: character class that cannot cross whitespace. MEASURED on the real leaked file:
#: `result` is 185 chars and contains a newline; the primary pattern matched 92
#: chars and left a **29-character continuation of the token** in place. The
#: naive check ("does the pattern still match? no") called that CLEAN.
#:
#: So after the primary pass, any credential-shaped run that immediately follows a
#: redaction marker is swallowed too. Bounded deliberately: >=12 chars, no spaces,
#: and at least TWO character classes (upper/lower/digit) -- ordinary prose words
#: do not look like that, which is what the negative controls pin.
_CONTINUATION = re.compile(r"(<REDACTED>)((?:\s*[A-Za-z0-9_+/=-]{12,})+)")


def _looks_like_secret(run: str) -> bool:
    core = re.sub(r"\s+", "", run)
    if len(core) < 12:
        return False
    classes = sum(bool(re.search(p, core)) for p in (r"[a-z]", r"[A-Z]", r"[0-9]"))
    return classes >= 2


def redact_text(s: str) -> str:
    for rx, repl in PATTERNS:
        s = rx.sub(repl, s)
    # Swallow wrapped continuations. Repeat until stable: a token can wrap twice.
    for _ in range(5):
        new = _CONTINUATION.sub(
            lambda m: m.group(1) if _looks_like_secret(m.group(2)) else m.group(0), s)
        if new == s:
            break
        s = new
    return s


def redact(obj):
    """Walk the JSON, rewriting string VALUES only. Keys are left alone so the
    401 detector's `"api_error_status": *401` grep keeps working."""
    if isinstance(obj, str):
        return redact_text(obj)
    if isinstance(obj, list):
        return [redact(v) for v in obj]
    if isinstance(obj, dict):
        return {k: redact(v) for k, v in obj.items()}
    return obj


def _self_test() -> int:
    ok, fail = 0, 0

    def check(name, cond, detail=""):
        nonlocal ok, fail
        if cond:
            ok += 1
            print(f"  ok   {name}")
        else:
            fail += 1
            print(f"  FAIL {name}" + (f" -- {detail}" if detail else ""))

    # THE EXACT SHAPE THAT LEAKED, and the exact scan that missed it.
    leaked = ("API Error: Header 'Authorization' has invalid value: "
              "'Bearer sk-ant-oat01-AbCdEf0123456789_-xyz'")
    old_scan = re.compile(r"sk-[A-Za-z0-9]{20,}")
    check("the ORIGINAL scan really did miss it (this is why we are here)",
          old_scan.search(leaked) is None,
          "if this ever passes, the false-negative premise is wrong")
    out = redact_text(leaked)
    check("the real leaked shape IS redacted", "sk-ant-oat01" not in out, out)
    check("...and no raw token body survives", "AbCdEf0123456789" not in out, out)
    check("the surrounding message is preserved (still diagnosable)",
          "API Error" in out and "Authorization" in out, out)

    # Structure preservation -- the 401 latch depends on it.
    doc = {"api_error_status": 401, "result": leaked, "total_cost_usd": 1.25,
           "nested": [{"k": "Bearer abcdefghijklmnopqrstuvwxyz012345"}]}
    red = redact(doc)
    check("numeric api_error_status is UNTOUCHED (the 401 latch still fires)",
          red["api_error_status"] == 401)
    check("non-string values are untouched", red["total_cost_usd"] == 1.25)
    check("nested strings are redacted too",
          "<REDACTED>" in red["nested"][0]["k"], red["nested"][0]["k"])
    check("keys are never rewritten", "api_error_status" in red)
    check("the grep the shell uses still matches",
          re.search(r'"api_error_status": *401', json.dumps(red)) is not None)

    # THE WRAPPED-TOKEN CASE. This is the one the first version of this file MISSED:
    # the real leaked `result` wraps the token across a newline, so a line-bounded
    # class matched 92 chars and left 29 chars of the secret behind -- and the check
    # "does the pattern still match?" reported CLEAN.
    wrapped = ("API Error: Header 'Authorization' has invalid value: "
               "'Bearer sk-ant-oat01-AbCdEf0123456789_-xyzAAAAAAAAAAAA\n"
               " uzVSs8NJsEWUe6LFc8xw-p2DdmQAA'")
    wout = redact_text(wrapped)
    check("the WRAPPED continuation is swallowed too",
          "uzVSs8NJsEWUe6L" not in wout, wout)
    check("...and the prefix is gone as well", "sk-ant-oat01" not in wout, wout)
    check("...leaving no long credential-shaped run at all",
          not re.search(r"[A-Za-z0-9_+/=-]{20,}", wout.replace("<REDACTED>", "")), wout)

    # NEGATIVE CONTROL -- must not redact ordinary prose, or the file is useless.
    benign = "the session completed in 42s and wrote 3 files to handoff/current/"
    check("negative control: benign prose is unchanged",
          redact_text(benign) == benign, redact_text(benign))
    check("negative control: a short sk- word is not clobbered",
          redact_text("sk-abc") == "sk-abc")
    # The continuation pass must not eat prose that merely FOLLOWS a redaction.
    prose = redact_text("Bearer abcdefghijklmnopqrstuvwxyz012345 and then we retried")
    check("negative control: prose after a redaction survives",
          "and then we retried" in prose, prose)

    print(f"\n  {ok} passed, {fail} failed")
    return 0 if fail == 0 else 1


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        return _self_test()
    if len(argv) < 2:
        print("usage: redact_secrets.py <raw.json> | --self-test", file=sys.stderr)
        return 2
    raw = open(argv[1], encoding="utf-8", errors="replace").read()
    try:
        doc = json.loads(raw)
    except Exception:
        # Not valid JSON (a crash, a truncated write). Still redact it as TEXT
        # rather than passing it through -- an unparseable file is exactly when
        # an error message full of credentials is most likely.
        sys.stdout.write(redact_text(raw))
        return 0
    json.dump(redact(doc), sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
