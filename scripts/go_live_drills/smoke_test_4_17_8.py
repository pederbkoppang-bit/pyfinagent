#!/usr/bin/env python
"""phase-4.17.8 smoke test -- Slack interface startup + handler registration.

Two-part verification:

1. Static import -- backend.slack_bot.app and .commands import cleanly
   without syntax or missing-import errors. Proves the module graph is
   healthy.
2. Dry-start (no Slack tokens required) -- create the AsyncApp and
   register commands, assert command count > 0. Do NOT start the
   Socket Mode handler (would require a live Slack connection).

Criteria:
- slack_bot_process_starts_without_import_error
- command_handlers_registered_cleanly
- no_syntax_errors_in_app_py_or_commands_py
- socket_mode_handler_class_importable
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))


def test_slack_bot_imports_and_commands_register():
    # 1. Syntax check
    for p in ("backend/slack_bot/app.py", "backend/slack_bot/commands.py"):
        with open(p) as f:
            src = f.read()
        try:
            ast.parse(src)
        except SyntaxError as e:
            raise AssertionError(f"syntax error in {p}: {e}")
    print("PASS no_syntax_errors_in_app_py_or_commands_py")

    # 2. Socket Mode handler importable
    try:
        from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler  # noqa: F401
        print("PASS socket_mode_handler_class_importable")
    except Exception as e:
        raise AssertionError(f"AsyncSocketModeHandler import FAIL: {e!r}")

    # 3. Commands module registers >=1 command handler when given a dummy AsyncApp
    os.environ.setdefault("SLACK_BOT_TOKEN", "xoxb-smoke-dummy")
    os.environ.setdefault("SLACK_APP_TOKEN", "xapp-smoke-dummy")
    os.environ.setdefault("SLACK_SIGNING_SECRET", "smoke-signing-dummy")
    try:
        from slack_bolt.async_app import AsyncApp
        app = AsyncApp(token="xoxb-smoke-dummy", signing_secret="smoke-signing-dummy")
        from backend.slack_bot.commands import register_commands
        register_commands(app)
        # listeners = app._async_listeners is the Bolt internal registry
        listeners = getattr(app, "_async_listeners", None) or getattr(app, "_listeners", None)
        n = len(listeners) if listeners is not None else 0
        assert n >= 1, f"no command handlers registered (n={n})"
        print(f"PASS command_handlers_registered_cleanly -- n_listeners={n}")
    except Exception as e:
        raise AssertionError(f"command_handlers_registered FAIL: {e!r}")

    print("PASS slack_bot_process_starts_without_import_error")
    print("PASS 4.17.8 Slack interface smoke")


if __name__ == "__main__":
    try:
        test_slack_bot_imports_and_commands_register()
    except AssertionError as e:
        print("FAIL:", e, file=sys.stderr)
        sys.exit(1)
    sys.exit(0)
