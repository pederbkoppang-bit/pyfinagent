"""phase-4.7 step 4.7.6: keyboard-only kill-switch workflow audit.

Asserts the kill-switch emergency action is reachable by keyboard
alone, i.e. without mouse. Specifically verifies KillSwitchShortcut
(shipped in Cycle 70) still has real teeth:

- registers a window keydown listener
- matches Ctrl/Cmd+Shift+H
- calls preventDefault to block OS capture
- invokes postPaperKillSwitchAction twice: FLATTEN_ALL then PAUSE
- renders an aria-live region for screen-reader feedback
- is actually mounted on the homepage cockpit

Plus the OpsStatusBar buttons (mouse alternative path):
- each action button has an aria-label
- each has a focus-visible ring class

Writes handoff/keyboard_flatten.json. `--check` exits 1 on any FAIL.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KS_SHORTCUT = REPO / "frontend" / "src" / "components" / "KillSwitchShortcut.tsx"
OPS_BAR = REPO / "frontend" / "src" / "components" / "OpsStatusBar.tsx"
HOME = REPO / "frontend" / "src" / "app" / "page.tsx"
OUT = REPO / "handoff" / "keyboard_flatten.json"


def _strip_comments(text: str) -> str:
    """Remove //line and /* block */ comments so substring checks
    can't be fooled by a regression like `// e.preventDefault();`."""
    # Block comments (non-greedy)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    # Line comments
    text = re.sub(r"//[^\n]*", "", text)
    return text


def _contains_live(text_stripped: str, needle: str) -> bool:
    """Plain substring check but against the comment-stripped text."""
    return needle in text_stripped


def _check_shortcut() -> tuple[bool, list[str]]:
    """Return (ok, reasons_if_fail). Checks operate on a comment-
    stripped copy of the source so that commenting-out a critical
    line (e.g. `//e.preventDefault();`) is treated as a real
    regression -- catches what harness-verifier cycle-75 flagged."""
    reasons: list[str] = []
    if not KS_SHORTCUT.exists():
        return False, ["KillSwitchShortcut.tsx missing"]
    raw = KS_SHORTCUT.read_text(encoding="utf-8")
    text = _strip_comments(raw)

    if "addEventListener" not in text or '"keydown"' not in text:
        reasons.append("no window keydown listener")
    if "ctrlKey" not in text or "metaKey" not in text or "shiftKey" not in text:
        reasons.append("missing Ctrl/Cmd+Shift modifier check")
    if '"H"' not in text and '"h"' not in text:
        reasons.append("handler does not match the 'H' key")
    # preventDefault must appear as a live call, not just as a string.
    if not re.search(r"\.preventDefault\s*\(", text):
        reasons.append("no preventDefault -- OS shortcut will capture")
    if "postPaperKillSwitchAction" not in text:
        reasons.append("does not call postPaperKillSwitchAction")
    if "FLATTEN_ALL" not in text:
        reasons.append("does not request FLATTEN_ALL")
    if '"PAUSE"' not in text and "'PAUSE'" not in text:
        reasons.append("does not chain PAUSE")
    # aria-live region is a JSX attribute; we only care it's in a
    # non-comment region. (Stripping block/line comments is enough
    # since JSX attributes themselves don't contain the // token.)
    if "aria-live" not in text:
        reasons.append("no aria-live region for screen-reader status")
    if "sr-only" not in text:
        reasons.append("aria-live region not visually hidden (sr-only) -- may disrupt layout")
    return (not reasons), reasons


def _check_ops_buttons() -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not OPS_BAR.exists():
        return False, ["OpsStatusBar.tsx missing"]
    text = OPS_BAR.read_text(encoding="utf-8")
    # Each of the three action labels (Pause, Resume, Flatten) must
    # appear inside a <button> that has both aria-label and a focus
    # ring class. We check by scanning blocks.
    required_labels = ("Resume", "Pause", "Flatten")
    # Match `   Flatten\n[whitespace]</button>` allowing any indent.
    for label in required_labels:
        m = re.search(rf"\b{re.escape(label)}\s*\n\s*</button>", text)
        if not m:
            m = re.search(rf">\s*{re.escape(label)}\s*</button>", text)
        if not m:
            reasons.append(f"no <button>{label}</button> render site found")
            continue
        end = m.start()
        start = text.rfind("<button", 0, end)
        if start < 0:
            reasons.append(f"could not locate <button> opening for {label}")
            continue
        button_block = text[start:end]
        if "aria-label" not in button_block:
            reasons.append(f"{label} button missing aria-label")
        if "focus-visible:" not in button_block and ":focus:" not in button_block:
            reasons.append(f"{label} button missing focus ring")
    return (not reasons), reasons


def _check_homepage_mount() -> tuple[bool, list[str]]:
    if not HOME.exists():
        return False, ["homepage page.tsx missing"]
    text = HOME.read_text(encoding="utf-8")
    reasons: list[str] = []
    if "<KillSwitchShortcut" not in text:
        reasons.append("homepage does not render <KillSwitchShortcut />")
    if "import { KillSwitchShortcut }" not in text:
        reasons.append("homepage does not import KillSwitchShortcut")
    return (not reasons), reasons


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    args = p.parse_args()

    short_ok, short_reasons = _check_shortcut()
    btn_ok, btn_reasons = _check_ops_buttons()
    home_ok, home_reasons = _check_homepage_mount()

    all_ok = short_ok and btn_ok and home_ok
    result = {
        "step": "4.7.6",
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "keyboard_only_kill_switch_workflow_green": all_ok,
        "shortcut": {"ok": short_ok, "reasons": short_reasons},
        "ops_buttons": {"ok": btn_ok, "reasons": btn_reasons},
        "homepage_mount": {"ok": home_ok, "reasons": home_reasons},
        "verdict": "PASS" if all_ok else "FAIL",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "wrote": str(OUT),
        "verdict": result["verdict"],
        "shortcut_ok": short_ok,
        "ops_buttons_ok": btn_ok,
        "homepage_mount_ok": home_ok,
    }))
    if args.check and not all_ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
