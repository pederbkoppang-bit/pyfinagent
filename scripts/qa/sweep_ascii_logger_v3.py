#!/usr/bin/env python3
"""phase-38.5.1 v3: remediate the 24 catch-all `?` substitutions from v2.

Per Q/A cycle-1 critique: the catch-all `?` was used 24 times (19%);
Q/A flagged this as semantic-loss WARN. Fix: scan swept files for
`logger.*("? ...")` patterns + strip the leading `"? `. Downstream message
is self-explanatory; the `?` adds nothing.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Match logger.<method>(f"? <content>") and variants (f, r, rf, fr, etc.)
PATTERN = re.compile(
    r'(logger\.\w+\(\s*(?:f|F|r|R|rf|Rf|rF|RF|fr|fR|Fr|FR)?(["\']))\?\s+',
)


def main() -> int:
    candidate_files = [
        REPO_ROOT / "backend" / "autonomous_loop.py",
        REPO_ROOT / "backend" / "services" / "ticket_queue_processor.py",
        REPO_ROOT / "backend" / "services" / "sla_monitor.py",
        REPO_ROOT / "backend" / "services" / "stuck_task_reaper.py",
        REPO_ROOT / "backend" / "services" / "response_delivery.py",
        REPO_ROOT / "backend" / "services" / "queue_notification.py",
        REPO_ROOT / "backend" / "services" / "ticket_ingestion.py",
        REPO_ROOT / "backend" / "services" / "slack_ticket_webhook.py",
        REPO_ROOT / "backend" / "slack_bot" / "app.py",
        REPO_ROOT / "backend" / "slack_bot" / "app_home.py",
        REPO_ROOT / "backend" / "slack_bot" / "assistant_handler.py",
        REPO_ROOT / "backend" / "slack_bot" / "assistant_lifecycle.py",
        REPO_ROOT / "backend" / "slack_bot" / "commands.py",
        REPO_ROOT / "backend" / "slack_bot" / "self_update.py",
        REPO_ROOT / "backend" / "slack_bot" / "streaming_integration.py",
        REPO_ROOT / "backend" / "db" / "tickets_db.py",
        REPO_ROOT / "backend" / "api" / "mas_events.py",
        REPO_ROOT / "backend" / "agents" / "openclaw_client.py",
        REPO_ROOT / "scripts" / "harness" / "run_autonomous_loop.py",
        REPO_ROOT / "scripts" / "harness" / "run_harness.py",
        REPO_ROOT / "scripts" / "migrations" / "add_phase27_columns.py",
        REPO_ROOT / "scripts" / "repair_phase_23_1_17.py",
    ]

    total_files = 0
    total_substitutions = 0

    for f in candidate_files:
        if not f.exists():
            continue
        text = f.read_text(encoding="utf-8")
        new_text, n = PATTERN.subn(r"\1", text)
        if n == 0:
            continue
        try:
            ast.parse(new_text, filename=str(f))
        except SyntaxError as e:
            print(f"SYNTAX BREAK {f.relative_to(REPO_ROOT)}: {e}", file=sys.stderr)
            continue
        f.write_text(new_text, encoding="utf-8")
        total_files += 1
        total_substitutions += n
        print(f"  remediated {f.relative_to(REPO_ROOT)}: {n} substitution(s)", file=sys.stderr)

    print(f"v3: {total_files} files, {total_substitutions} substitutions", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
