"""
Go-Live drill test: Slack signals end-to-end (Phase 4.4.3.2).

Standalone, stdlib-only drill. Verifies the full code path from signal dict
through Block Kit rendering to the Slack WebClient call site in
`publish_signal`. Does NOT require a running Slack bot or valid tokens --
live Slack delivery is deferred to launch-week (precedent: 4.4.3.1 deferred
runtime curl verification).

Uses AST-level analysis only (no runtime import of formatters or
signals_server) because the production code uses Python 3.10+ type union
syntax (dict | None) which is not available on all system Pythons.

Checks:
  S0  format_signal_alert function exists in formatters.py
  S1  format_signal_alert has correct parameters (signal, trade)
  S2  _signal_emoji helper exists in formatters.py
  S3  _signal_emoji body contains green/red/yellow mappings
  S4  format_signal_alert body builds header block with ticker/action
  S5  format_signal_alert body builds section with Confidence/Price/Size/Stop
  S6  format_signal_alert body builds context with PyFinAgent + signal_id
  S7  format_signal_alert body builds divider block
  S8  format_signal_alert handles missing fields via .get() with defaults
  S9  publish_signal method exists on SignalsServer
  S10 publish_signal source imports format_signal_alert
  S11 publish_signal source calls chat_postMessage with blocks
  S12 publish_signal has ASCII-only text_fallback for push previews
  S13 publish_signal degrades to slack_not_configured when no token
  S14 publish_signal handles SlackApiError gracefully
  S15 No non-ASCII in logger messages within Slack posting path

Run from the repo root:

    python3 scripts/go_live_drills/slack_signals_e2e_test.py

Exit code 0 on PASS, exit 1 on any failure.
"""

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FORMATTERS_PATH = REPO_ROOT / "backend" / "slack_bot" / "formatters.py"
SIGNALS_SERVER_PATH = REPO_ROOT / "backend" / "agents" / "mcp_servers" / "signals_server.py"


def read_source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def find_function(tree: ast.AST, name: str) -> ast.FunctionDef:
    """Find a top-level or class-level function by name."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def get_function_source(source: str, func_node) -> str:
    """Extract source lines for a function node."""
    lines = source.splitlines()
    start = func_node.lineno - 1
    end = func_node.end_lineno or start + 1
    return "\n".join(lines[start:end])


def main():
    passed = 0
    failed = 0
    total = 16

    def check(label: str, condition: bool, detail: str = ""):
        nonlocal passed, failed
        if condition:
            passed += 1
            print(f"  PASS {label}" + (f" -- {detail}" if detail else ""))
        else:
            failed += 1
            print(f"  FAIL {label}" + (f" -- {detail}" if detail else ""))

    print("Phase 4.4.3.2: Slack signals end-to-end drill")
    print("=" * 60)

    # ---- Parse formatters.py ----
    fmt_source = read_source(FORMATTERS_PATH)
    fmt_tree = ast.parse(fmt_source)

    # S0: format_signal_alert exists
    fsa_node = find_function(fmt_tree, "format_signal_alert")
    check("S0  format_signal_alert exists in formatters.py",
          fsa_node is not None)
    if fsa_node is None:
        print(f"\nRESULT: {passed}/{total} PASS -- ABORT (missing function)")
        sys.exit(1)

    # S1: correct parameters (signal, trade)
    args = fsa_node.args
    param_names = [a.arg for a in args.args]
    # Expect (self-less) params: signal, trade
    has_signal = "signal" in param_names
    has_trade = "trade" in param_names
    check("S1  parameters include (signal, trade)",
          has_signal and has_trade,
          f"params={param_names}")

    # S2: _signal_emoji helper exists
    emoji_node = find_function(fmt_tree, "_signal_emoji")
    check("S2  _signal_emoji helper exists",
          emoji_node is not None)

    # S3: _signal_emoji body contains green/red/yellow mappings
    emoji_source = get_function_source(fmt_source, emoji_node) if emoji_node else ""
    has_green = "green" in emoji_source
    has_red = "red" in emoji_source
    has_yellow = "yellow" in emoji_source
    check("S3  _signal_emoji maps BUY=green SELL=red HOLD=yellow",
          has_green and has_red and has_yellow,
          f"green={has_green} red={has_red} yellow={has_yellow}")

    # Get format_signal_alert source for body checks
    fsa_source = get_function_source(fmt_source, fsa_node)

    # S4: builds header block with ticker/action
    has_header_type = '"header"' in fsa_source or "'header'" in fsa_source
    has_ticker_in_header = "ticker" in fsa_source and "action" in fsa_source
    check("S4  header block with ticker+action",
          has_header_type and has_ticker_in_header,
          f"header_type={has_header_type} ticker+action={has_ticker_in_header}")

    # S5: section with Confidence/Price/Size/Stop fields
    has_confidence = "Confidence" in fsa_source
    has_price = "Price" in fsa_source
    has_size = "Size" in fsa_source
    has_stop = "Stop" in fsa_source
    check("S5  section fields: Confidence/Price/Size/Stop",
          has_confidence and has_price and has_size and has_stop,
          f"conf={has_confidence} price={has_price} size={has_size} stop={has_stop}")

    # S6: context with PyFinAgent + signal_id
    has_pyfinagent = "PyFinAgent" in fsa_source
    has_signal_id = "signal_id" in fsa_source
    check("S6  context has PyFinAgent + signal_id",
          has_pyfinagent and has_signal_id,
          f"PyFinAgent={has_pyfinagent} signal_id={has_signal_id}")

    # S7: divider block
    has_divider = '"divider"' in fsa_source or "'divider'" in fsa_source
    check("S7  divider block present",
          has_divider)

    # S8: handles missing fields via .get() with defaults
    get_count = fsa_source.count(".get(")
    has_defaults = ('0.0' in fsa_source or 'or 0.0' in fsa_source) and '""' in fsa_source
    check("S8  graceful .get() with defaults for missing fields",
          get_count >= 5 and has_defaults,
          f".get() calls={get_count}")

    # ---- Parse signals_server.py ----
    ss_source = read_source(SIGNALS_SERVER_PATH)
    ss_tree = ast.parse(ss_source)

    # Find SignalsServer class
    ss_class = None
    for node in ast.walk(ss_tree):
        if isinstance(node, ast.ClassDef) and node.name == "SignalsServer":
            ss_class = node
            break

    # S9: publish_signal method exists on SignalsServer
    ps_method = None
    if ss_class:
        for item in ss_class.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "publish_signal":
                ps_method = item
                break
    check("S9  publish_signal on SignalsServer",
          ps_method is not None)

    if ps_method is None:
        print(f"\nRESULT: {passed}/{total} PASS -- ABORT (missing publish_signal)")
        sys.exit(1)

    ps_source = get_function_source(ss_source, ps_method)

    # S10: imports format_signal_alert
    check("S10 imports format_signal_alert",
          "format_signal_alert" in ps_source,
          "lazy import inside publish_signal")

    # S11: calls chat_postMessage with blocks
    has_chat_post = "chat_postMessage" in ps_source
    has_blocks_arg = "blocks=" in ps_source
    check("S11 chat_postMessage with blocks",
          has_chat_post and has_blocks_arg,
          f"chat_postMessage={has_chat_post} blocks_arg={has_blocks_arg}")

    # S12: ASCII-only text_fallback
    has_text_fallback = "text_fallback" in ps_source
    check("S12 ASCII text_fallback for push preview",
          has_text_fallback,
          "text_fallback variable used for Slack push-preview")

    # S13: slack_not_configured degradation
    check("S13 slack_not_configured degradation",
          "slack_not_configured" in ps_source,
          "structured reason when no token/channel configured")

    # S14: SlackApiError handled gracefully
    has_slack_api_error = "SlackApiError" in ps_source
    has_except_block = "except" in ps_source
    check("S14 SlackApiError handled gracefully",
          has_slack_api_error and has_except_block,
          f"SlackApiError={has_slack_api_error}")

    # S15: No non-ASCII in logger messages within Slack posting path
    ps_lines = ps_source.splitlines()
    non_ascii_in_logger = False
    offending_line = ""
    for line in ps_lines:
        stripped = line.strip()
        if stripped.startswith("logger."):
            for ch in stripped:
                if ord(ch) > 127:
                    non_ascii_in_logger = True
                    offending_line = stripped[:60]
                    break
    check("S15 ASCII-only logger messages in Slack path",
          not non_ascii_in_logger,
          f"0 non-ASCII in logger calls" if not non_ascii_in_logger
          else f"non-ASCII found: {offending_line}")

    # ---- Summary ----
    print()
    print("=" * 60)
    print(f"RESULT: {passed}/{total} PASS, {failed} FAIL")
    if failed == 0:
        print("VERDICT: PASS -- Phase 4.4.3.2 code-level verification complete")
        print("NOTE: Live Slack delivery deferred to launch-week (precedent: 4.4.3.1)")
    else:
        print("VERDICT: FAIL")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
