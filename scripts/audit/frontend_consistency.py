"""phase-4.7 step 4.7.5: cross-page consistency audit.

Lint and build handle the JS/TS-level rules. This audit catches the
four contract criteria that would silently slip past ESLint:

1. `no_emoji_in_ui` -- any high-codepoint character in page/component
   TSX that isn't a known-safe code-visual character (em-dash, arrow,
   bullet, fraction, heart in aria-label on an unrelated feature).
2. `phosphor_icons_only` -- page/component TSX must not import icons
   from a non-Phosphor library (e.g. react-icons, heroicons, lucide).
3. `ops_status_bar_pattern_applied` -- cockpit homepage
   (frontend/src/app/page.tsx) must render <OpsStatusBar />.
4. (auxiliary) login page must not re-import removed NavAnalyze-style
   emojis.

Writes handoff/frontend_consistency.json. `--check` exits 1 on any
criterion failing.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FRONTEND_SRC = REPO / "frontend" / "src"
OUT = REPO / "handoff" / "frontend_consistency.json"


# Unicode ranges that indicate emoji (Extended_Pictographic). We don't
# pull in regex packages -- keep the check dependency-free.
EMOJI_RANGES = [
    (0x1F300, 0x1F5FF),  # Misc Symbols & Pictographs
    (0x1F600, 0x1F64F),  # Emoticons
    (0x1F680, 0x1F6FF),  # Transport & Map
    (0x1F700, 0x1F77F),  # Alchemical
    (0x1F780, 0x1F7FF),  # Geometric Shapes Extended
    (0x1F800, 0x1F8FF),  # Supplemental Arrows-C
    (0x1F900, 0x1F9FF),  # Supplemental Symbols & Pictographs
    (0x1FA00, 0x1FA6F),  # Chess Symbols
    (0x1FA70, 0x1FAFF),  # Symbols & Pictographs Extended-A
    (0x2600, 0x26FF),    # Misc Symbols (sun, umbrella, etc.)
    (0x2700, 0x27BF),    # Dingbats
]

# Explicit allow-list for non-emoji high-codepoint characters that
# legitimately appear in UI strings or comments (em-dash, en-dash,
# non-breaking space, arrows used in prose). These do not violate the
# "no emoji" rule.
ALLOWED_HIGH_CP = {
    0x2014,  # em-dash --
    0x2013,  # en-dash -
    0x2026,  # horizontal ellipsis ...
    0x00A0,  # non-breaking space
    0x2192,  # right arrow
    0x2190,  # left arrow
    0x2022,  # bullet
    0x00B0,  # degree
    0x00B1,  # plus-minus
    0x00D7,  # multiplication
    0x00F7,  # division
    0x2248,  # approximately equal
    0x221E,  # infinity
    0x2264,  # <=
    0x2265,  # >=
    0x2260,  # not-equal
}

FORBIDDEN_ICON_IMPORTS = (
    "react-icons",
    "@heroicons",
    "@iconify",
    "lucide-react",
    "react-feather",
)


def _is_emoji(cp: int) -> bool:
    if cp in ALLOWED_HIGH_CP:
        return False
    for lo, hi in EMOJI_RANGES:
        if lo <= cp <= hi:
            return True
    return False


def _iter_tsx(root: Path):
    for p in root.rglob("*.tsx"):
        if "node_modules" in p.parts or ".next" in p.parts:
            continue
        yield p
    for p in root.rglob("*.ts"):
        if "node_modules" in p.parts or ".next" in p.parts:
            continue
        yield p


def check_emoji() -> list[dict]:
    hits: list[dict] = []
    for path in _iter_tsx(FRONTEND_SRC):
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            for ch in line:
                cp = ord(ch)
                if cp >= 0x2500 and _is_emoji(cp):
                    hits.append({
                        "file": str(path.relative_to(REPO)),
                        "line": line_no,
                        "codepoint_hex": hex(cp),
                        "char": ch,
                    })
                    break  # one hit per line is enough evidence
    return hits


def check_icon_imports() -> list[dict]:
    hits: list[dict] = []
    for path in _iter_tsx(FRONTEND_SRC):
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            m = re.search(r"""from\s+["']([^"']+)["']""", line)
            if not m:
                continue
            src = m.group(1)
            for bad in FORBIDDEN_ICON_IMPORTS:
                if src.startswith(bad):
                    hits.append({
                        "file": str(path.relative_to(REPO)),
                        "line": line_no,
                        "imports_from": src,
                    })
    return hits


def check_ops_status_bar_on_cockpit() -> tuple[bool, str]:
    page = FRONTEND_SRC / "app" / "page.tsx"
    if not page.exists():
        return False, "homepage page.tsx missing"
    text = page.read_text(encoding="utf-8")
    if "<OpsStatusBar" not in text:
        return False, "homepage does not render <OpsStatusBar />"
    if "import { OpsStatusBar }" not in text and "OpsStatusBar," not in text and "OpsStatusBar }" not in text:
        return False, "homepage renders OpsStatusBar without an import"
    return True, "ok"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    emoji_hits = check_emoji()
    icon_hits = check_icon_imports()
    ops_ok, ops_reason = check_ops_status_bar_on_cockpit()

    verdict = "PASS" if (not emoji_hits and not icon_hits and ops_ok) else "FAIL"
    result = {
        "step": "4.7.5",
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "no_emoji_in_ui": not emoji_hits,
        "phosphor_icons_only": not icon_hits,
        "ops_status_bar_pattern_applied": ops_ok,
        "emoji_hits": emoji_hits,
        "non_phosphor_icon_imports": icon_hits,
        "ops_status_bar_reason": ops_reason,
        "verdict": verdict,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "wrote": str(OUT),
        "verdict": verdict,
        "emoji_hits": len(emoji_hits),
        "icon_hits": len(icon_hits),
        "ops_ok": ops_ok,
    }))
    if args.check and verdict != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
