"""phase-75.11: SRE hardening -- log rotation, single service authority,
unattended-wrapper timeouts, pkill guard, log-formatter fix.

Six immutable criteria (verbatim from .claude/masterplan.json step 75.11),
one test group per criterion. Criteria 4 and 6 are BEHAVIORAL (drive the
real pre-tool-use-danger.sh hook / the real setup_logging() code path);
the rest are text-marker assertions against the shipped scripts/templates,
each paired with a named breaking mutation in the mutation matrix recorded
in handoff/current/experiment_results_75.11_draft.md.

1. Rotation plist template + rotation script under scripts/ops/ covering
   the four named logs with cp+truncate, a watchdog-liveness (health.jsonl
   mtime) alarm seam, and the runbook + OPS-ROTATE-BOOTSTRAP token drafted.
2. start_services.sh: launchctl kickstart for backend+frontend, no bare
   'pkill -9 uvicorn' / 'pkill -9 "next dev"' anywhere, legacy branch uses
   scoped SIGTERM pkill only, no '> backend.log' truncation.
3. Frontend plist template running 'next start' via a pre-start build
   wrapper; ablation wrapper using the sanitized-sourcing block (no raw
   '. backend/.env') and logging FAIL rc with a paging seam.
4. pre-tool-use-danger.sh blocks pkill/killall targeting
   python|uvicorn|next|slack_bot, with the CLAUDE_ALLOW_DANGER escape.
5. run_away_session.sh git pull is gtimeout-wrapped; slack_mention_checker
   curl carries -m 15; run_cycle.sh claude call is gtimeout-capped.
6. main.py setup_logging: debug -> CompactFormatter, default ->
   JsonFormatter (behavioral, incl. redaction survival); plus the
   no-secrets-in-templates and no-executed-bootstrap-in-scripts guards
   that back up "executor edits no .env and bootstraps no machine agents".
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HOOK = REPO_ROOT / ".claude" / "hooks" / "pre-tool-use-danger.sh"
OPS_DIR = REPO_ROOT / "scripts" / "ops"


def _read(relpath: str) -> str:
    return (REPO_ROOT / relpath).read_text(encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────
# Criterion 1 -- log rotation agent (sre-ops-01)
# ─────────────────────────────────────────────────────────────────────

def test_c1_rotation_script_covers_four_logs_with_cp_truncate():
    text = _read("scripts/ops/rotate_logs.sh")
    for log_ref in (
        '"$REPO/backend.log"',
        '"$REPO/frontend.log"',
        '"$LOGDIR/slack_bot.log"',
        '"$LOGDIR/auto-push.log"',
    ):
        assert log_ref in text, f"missing rotation target: {log_ref}"
    # cp+truncate idiom, modeled verbatim on healthcheck.sh:246-255.
    assert 'cp "$src" "$archive"' in text
    assert ': > "$src"' in text
    assert "gzip" in text


def test_c1_rotation_script_has_watchdog_liveness_alarm_seam():
    text = _read("scripts/ops/rotate_logs.sh")
    assert "health.jsonl" in text or "HEALTH=" in text
    assert "STALE_THRESHOLD_S" in text
    assert "7200" in text  # 2h, per contract
    assert "page_bot_token" in text


def test_c1_logrotate_plist_template_exists_and_points_at_script():
    plist = _read("scripts/ops/com.pyfinagent.logrotate.plist.template")
    assert "rotate_logs.sh" in plist
    assert "StartInterval" in plist


def test_c1_runbook_and_operator_token_drafted():
    runbook = _read("handoff/current/ops_rotate_runbook_75.11.md")
    assert "OPS-ROTATE-BOOTSTRAP" in runbook


# ─────────────────────────────────────────────────────────────────────
# Criterion 2 -- start_services.sh single service authority (sre-ops-02)
# ─────────────────────────────────────────────────────────────────────

def test_c2_kickstart_backend_and_frontend():
    text = _read("scripts/start_services.sh")
    assert "launchctl kickstart -k" in text
    assert "com.pyfinagent.backend" in text
    assert "com.pyfinagent.frontend" in text


def test_c2_no_bare_pkill9_uvicorn_or_next_dev():
    text = _read("scripts/start_services.sh")
    assert "pkill -9 uvicorn" not in text
    assert 'pkill -9 "next dev"' not in text


def test_c2_legacy_branch_uses_scoped_sigterm_pkill():
    text = _read("scripts/start_services.sh")
    assert "pkill -f 'uvicorn backend.main'" in text
    # must live INSIDE the LEGACY_DIRECT-gated branch, after the flag check.
    assert text.index("LEGACY_DIRECT") < text.index("pkill -f 'uvicorn backend.main'")


def test_c2_no_backend_log_truncation():
    text = _read("scripts/start_services.sh")
    # A bare '>' immediately before backend.log (not preceded by a second
    # '>', i.e. not the append form '>>') would be a truncation.
    assert re.search(r"(?<!>)>\s*backend\.log", text) is None


# ─────────────────────────────────────────────────────────────────────
# Criterion 3 -- frontend single authority + ablation wrapper (sre-ops-09/04)
# ─────────────────────────────────────────────────────────────────────

def test_c3_frontend_plist_template_wrapper_runs_next_start():
    plist = _read("scripts/ops/com.pyfinagent.frontend.plist.template")
    assert "frontend_start.sh" in plist
    wrapper = _read("scripts/ops/frontend_start.sh")
    # Check the ACTUAL executable line, not just any substring match --
    # narrative comments/echo lines elsewhere in the file could still say
    # "next start" even if the real `exec` line reverted to `next dev`.
    assert re.search(r"^\s*exec\s+npx\s+next start\b", wrapper, re.M)
    assert not re.search(r"^\s*exec\s+npx\s+next dev\b", wrapper, re.M)


def test_c3_ablation_wrapper_no_raw_dot_env_uses_sanitized_block():
    text = _read("scripts/ops/run_ablation.sh")
    assert re.search(r"grep -E '\^\[A-Za-z_\]\[A-Za-z0-9_\]\*='", text)
    assert "mktemp" in text
    # No line does a raw dot-source of backend/.env itself (only the
    # sanitized tmp file, `. "$_envtmp"`, is sourced).
    for line in text.splitlines():
        stripped = line.strip()
        assert not re.match(r'^\.\s+"?\$REPO/backend/\.env"?\s*$', stripped)
        assert not re.match(r"^\.\s+backend/\.env\s*$", stripped)


def test_c3_ablation_wrapper_logs_fail_rc_with_paging_seam():
    text = _read("scripts/ops/run_ablation.sh")
    assert "FAIL rc=" in text
    assert "consecutive_fails" in text
    assert "chat.postMessage" in text


def test_c3_ablation_plist_template_points_at_wrapper():
    plist = _read("scripts/ops/com.pyfinagent.ablation.plist.template")
    assert "run_ablation.sh" in plist


# ─────────────────────────────────────────────────────────────────────
# Criterion 4 -- pre-tool-use-danger.sh pkill/killall rail (sre-ops-05)
# BEHAVIORAL: drives the REAL hook script via env vars. Zero real kills --
# the hook only matches the command STRING and never executes it.
# ─────────────────────────────────────────────────────────────────────

def _run_hook(command: str, allow_danger: bool = False) -> int:
    env = dict(os.environ)
    env["CLAUDE_TOOL_NAME"] = "Bash"
    env["CLAUDE_TOOL_INPUT"] = json.dumps({"command": command})
    if allow_danger:
        env["CLAUDE_ALLOW_DANGER"] = "1"
    else:
        env.pop("CLAUDE_ALLOW_DANGER", None)
    proc = subprocess.run(
        ["bash", str(HOOK)], env=env, capture_output=True, text=True, timeout=10,
    )
    return proc.returncode


def test_c4_pkill_uvicorn_blocked():
    assert _run_hook("pkill -9 uvicorn") == 2


def test_c4_killall_next_blocked():
    assert _run_hook("killall next") == 2


def test_c4_pkill_f_uvicorn_backend_main_blocked():
    # Matches "uvicorn" -- correctly blocked at the top-level Bash-tool
    # layer even though this exact scoped form is what the LEGACY branch
    # runs INSIDE a script (a subprocess, not a Bash tool call, so it is
    # unaffected there; see test_c2_legacy_branch_uses_scoped_sigterm_pkill).
    assert _run_hook('pkill -f "uvicorn backend.main"') == 2


def test_c4_pkill_unrelated_allowed():
    assert _run_hook("pkill SomeUnrelatedTool") == 0


def test_c4_allow_danger_escape_hatch_overrides():
    assert _run_hook("pkill -9 uvicorn", allow_danger=True) == 0


# ─────────────────────────────────────────────────────────────────────
# Criterion 5 -- unattended-wrapper timeouts (sre-ops-07)
# ─────────────────────────────────────────────────────────────────────

def test_c5_run_away_session_git_pull_gtimeout_wrapped():
    text = _read("scripts/away_ops/run_away_session.sh")
    assert re.search(
        r'"\$GTIMEOUT"\s+-k\s+10\s+120\s+git pull --rebase origin main', text,
    )
    # additive-only: the existing offline-mode branch is unchanged (no new
    # branch was introduced for rc=124 -- it already routes any nonzero).
    assert "OFFLINE MODE" in text


def test_c5_slack_mention_checker_curl_m15():
    text = _read("scripts/slack_mention_checker.sh")
    assert re.search(
        r"curl -s -m 15 -X POST https://slack\.com/api/conversations\.history", text,
    )


def test_c5_run_cycle_claude_call_gtimeout_capped():
    text = _read("scripts/mas_harness/run_cycle.sh")
    assert re.search(r'"\$GTIMEOUT"\s+-k\s+60\s+3600', text)
    idx_gtimeout = text.index('"$GTIMEOUT" -k 60 3600')
    idx_claude = text.index('"$CLAUDE_BIN" \\', idx_gtimeout)
    # the gtimeout wraps THIS SAME invocation (adjacent, not a stray no-op).
    assert 0 < (idx_claude - idx_gtimeout) < 200


# ─────────────────────────────────────────────────────────────────────
# Criterion 6 -- main.py formatter fix (pysvc-05) + no-.env/no-bootstrap
# ─────────────────────────────────────────────────────────────────────

class _FakeLogSettings:
    def __init__(self, debug: bool):
        self.debug = debug
        self.log_level = "INFO"


class _FakeStderr:
    """Buffer-backed stand-in for sys.stderr, used only for the duration of
    one setup_logging() call. setup_logging() wraps sys.stderr.buffer in a
    FRESH io.TextIOWrapper on every call; when the previous handler (and
    its TextIOWrapper) is dropped by root.handlers.clear(), CPython's GC
    closing that wrapper also closes the underlying buffer it wraps --
    which would be the REAL sys.stderr.buffer if we called setup_logging()
    against the live stderr more than once in the same process (exactly
    what this test file does, 3x). Swapping in a throwaway buffer for the
    duration of each call keeps that GC side effect off the real stderr."""

    def __init__(self):
        import io
        self.buffer = io.BytesIO()


def _run_setup_logging_and_capture(debug: bool, message: str):
    """Call the REAL backend.main.setup_logging() under a monkeypatched
    settings, emit one record through the real handler chain (including
    SecretRedactionFilter), and return (formatter_class, emitted_text).

    setup_logging() does `root.handlers.clear()` -- in a full-suite run
    that WIPES OUT pytest's own log-capture handler (installed once at
    session start), which corrupted later tests' log capture and, once,
    crashed the interpreter's stderr entirely (observed empirically: the
    isolated file passed, the full suite did not). So beyond swapping in
    a throwaway stderr for setup_logging()'s own TextIOWrapper, this also
    snapshots and restores the root logger's PRE-CALL handlers/level so
    the rest of the suite sees no trace of this call ever happening.
    """
    import io
    import logging
    import sys

    import backend.main as main_mod

    root = logging.getLogger()
    orig_handlers = list(root.handlers)
    orig_level = root.level
    orig_get_settings = main_mod.get_settings
    orig_stderr = sys.stderr
    main_mod.get_settings = lambda: _FakeLogSettings(debug=debug)
    sys.stderr = _FakeStderr()
    try:
        main_mod.setup_logging()
        handler = root.handlers[0]
        buf = io.StringIO()
        handler.stream = buf
        logging.getLogger("test_phase_75_sre_ops").info(message)
        return type(handler.formatter), buf.getvalue()
    finally:
        for h in root.handlers:
            if h not in orig_handlers:
                try:
                    h.close()
                except Exception:
                    pass
        root.handlers = orig_handlers
        root.setLevel(orig_level)
        main_mod.get_settings = orig_get_settings
        sys.stderr = orig_stderr


def test_c6_debug_true_uses_compact_formatter():
    from backend.main import CompactFormatter

    fmt_cls, _ = _run_setup_logging_and_capture(True, "hello-debug-compact")
    assert fmt_cls is CompactFormatter


def test_c6_debug_false_uses_json_formatter():
    import json as _json

    from backend.main import JsonFormatter

    fmt_cls, emitted = _run_setup_logging_and_capture(False, "hello-json-default")
    assert fmt_cls is JsonFormatter
    payload = _json.loads(emitted.strip())
    assert payload["message"] == "hello-json-default"


def test_c6_redaction_survives_json_branch():
    import json as _json

    _, emitted = _run_setup_logging_and_capture(
        False, "api_key=SECRETVALUE123 fetching prices",
    )
    assert "SECRETVALUE123" not in emitted
    assert "REDACTED" in emitted
    # Still valid JSON -- proves redaction ran BEFORE json.dumps (handler
    # filters run before format()), not that the JSON string happened to
    # survive an ad-hoc string replace.
    payload = _json.loads(emitted.strip())
    assert "REDACTED" in payload["message"]


_SECRET_LIKE_RE = re.compile(r"[A-Za-z0-9+/_=-]{30,}")
_KNOWN_SECRET_VARS = (
    "AUTH_SECRET", "AUTH_GOOGLE_SECRET", "CLAUDE_CODE_OAUTH_TOKEN", "SLACK_BOT_TOKEN",
)


def test_c6_no_plaintext_secrets_in_templates():
    template_files = sorted(OPS_DIR.glob("*.plist.template"))
    assert len(template_files) >= 3, "expected >=3 .plist.template files under scripts/ops/"
    for f in template_files:
        text = f.read_text(encoding="utf-8")
        long_tokens = _SECRET_LIKE_RE.findall(text)
        # Paths (contain '/') and the __REPO_ROOT__ placeholder (contains
        # '__') are not secret-shaped; anything else 30+ chars is suspect.
        suspicious = [t for t in long_tokens if "/" not in t and "__" not in t]
        assert suspicious == [], f"{f.name} contains a secret-shaped literal: {suspicious}"
        # Prose MAY discuss these var names (documenting that the LIVE plist
        # embeds them is exactly the finding this step reacts to); what must
        # never appear is one DEFINED as a plist <key> -- i.e. sourced with
        # a literal value in this template's own EnvironmentVariables dict.
        for var in _KNOWN_SECRET_VARS:
            assert f"<key>{var}</key>" not in text, (
                f"{f.name} defines {var} as a plist key (must be prose-only, never a <key>)"
            )


def test_c6_no_launchctl_bootstrap_executed_in_ops_scripts():
    # 'launchctl bootstrap' may appear in prose (runbook / plist-template
    # XML comments) but must never appear as an executed line in a .sh file.
    for f in sorted(OPS_DIR.glob("*.sh")):
        for line in f.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert "launchctl bootstrap" not in stripped, f"{f.name}: {stripped!r}"
