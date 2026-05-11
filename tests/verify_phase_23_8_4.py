"""phase-23.8.4 verifier — 11-claim source-level + behavioral assertion.

Run: source .venv/bin/activate && python3 tests/verify_phase_23_8_4.py

Exits 0 on PASS, 1 on FAIL. Each criterion is a single source grep, a
JSON-shape assertion, a syntax check, or a behavioral run of the hook.

Audit basis: cycle 40 / phase-23.8.3 experiment_results.md:148 +
contract.md:208 — auto-commit-and-push hook did not auto-fire on Edit
calls in cycles 38/39/40; operator manually triggered.

Safety: claims 7 + 10 invoke the actual hook script as a subprocess.
The hook is fail-open + idempotent: if `newly_done` (the set of
masterplan steps newly flipped to done since HEAD) is empty, the
script logs INVOKED then exits silently at the existing
`[ -z "$FLIPPED_STEP" ]` guard before any `git add` / `git commit` /
`git push` work. The verifier ASSERTS newly_done is empty before
invoking, refusing to run the behavioral claims otherwise — prevents
the verifier from accidentally triggering a real auto-commit.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS: list[tuple[str, str, str]] = []


def _check(name: str, ok: bool, detail: str = ""):
    flag = "PASS" if ok else "FAIL"
    RESULTS.append((flag, name, detail))
    return ok


def _grep_in_file(path: Path, needle: str) -> bool:
    if not path.exists():
        return False
    return needle in path.read_text(encoding="utf-8")


def _count_invoked_lines(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    return sum(
        1 for line in log_path.read_text(encoding="utf-8").splitlines()
        if "INVOKED auto-commit-and-push" in line
    )


def _newly_done_is_empty() -> tuple[bool, str]:
    """Return (True, '') if newly_done is empty so behavioral claims
    are safe to run. (False, reason) otherwise."""
    mp = REPO / ".claude" / "masterplan.json"
    if not mp.exists():
        return False, "masterplan.json missing"
    try:
        curr = mp.read_text(encoding="utf-8")
        prev_r = subprocess.run(
            ["git", "show", "HEAD:.claude/masterplan.json"],
            capture_output=True, text=True, timeout=10, cwd=str(REPO),
        )
        prev = prev_r.stdout if prev_r.returncode == 0 else ""
    except Exception as exc:
        return False, f"git read failed: {exc}"

    def done_ids(blob: str) -> set[str]:
        if not blob.strip():
            return set()
        try:
            d = json.loads(blob)
        except json.JSONDecodeError:
            return set()
        out: set[str] = set()

        def walk(n):
            if isinstance(n, dict):
                if n.get("status") == "done" and "id" in n:
                    out.add(str(n["id"]))
                for v in n.values():
                    walk(v)
            elif isinstance(n, list):
                for v in n:
                    walk(v)

        walk(d)
        return out

    p = done_ids(prev)
    c = done_ids(curr)
    newly = c - p
    if newly:
        return False, f"newly_done is non-empty: {sorted(newly)}"
    return True, ""


def main() -> int:
    hook = REPO / ".claude" / "hooks" / "auto-commit-and-push.sh"
    settings = REPO / ".claude" / "settings.json"
    log_path = REPO / "handoff" / "logs" / "auto-push.log"
    harness_log = REPO / "handoff" / "harness_log.md"

    # --- 1. settings.json parses ---
    settings_obj = None
    settings_ok = False
    try:
        settings_obj = json.loads(settings.read_text(encoding="utf-8"))
        settings_ok = True
    except Exception:
        settings_ok = False
    _check(
        "1. settings_json_valid",
        settings_ok,
        ".claude/settings.json must parse as JSON",
    )

    # --- 2. Edit matcher `if` predicate preserved ---
    edit_if_ok = False
    if settings_ok:
        post_blocks = settings_obj.get("hooks", {}).get("PostToolUse", [])
        for blk in post_blocks:
            if blk.get("matcher") == "Edit" and blk.get("if") == "Edit(.claude/masterplan.json)":
                edit_if_ok = True
                break
    _check(
        "2. edit_matcher_if_predicate_preserved",
        edit_if_ok,
        "PostToolUse Edit matcher must STILL have `if: 'Edit(.claude/masterplan.json)'` (observability-first; we did NOT drop it)",
    )

    # --- 3. Write matcher `if` predicate preserved ---
    write_if_ok = False
    if settings_ok:
        post_blocks = settings_obj.get("hooks", {}).get("PostToolUse", [])
        for blk in post_blocks:
            if blk.get("matcher") == "Write" and blk.get("if") == "Write(.claude/masterplan.json)":
                write_if_ok = True
                break
    _check(
        "3. write_matcher_if_predicate_preserved",
        write_if_ok,
        "PostToolUse Write matcher must STILL have `if: 'Write(.claude/masterplan.json)'`",
    )

    # --- 4. Hook contains INVOKED log call near the top ---
    hook_text = hook.read_text(encoding="utf-8") if hook.exists() else ""
    has_invoked = 'log "INVOKED auto-commit-and-push pid=$$"' in hook_text
    # "Near the top" means before the masterplan-exists guard. Find the
    # line index of both markers; INVOKED must be ABOVE the guard.
    lines = hook_text.splitlines()
    invoked_line = next((i for i, l in enumerate(lines) if 'log "INVOKED auto-commit-and-push' in l), -1)
    guard_line = next((i for i, l in enumerate(lines) if 'if [ ! -f "$MASTERPLAN" ]' in l), -1)
    is_at_top = (invoked_line >= 0 and guard_line >= 0 and invoked_line < guard_line)
    _check(
        "4. auto_commit_hook_has_invoked_log_at_top",
        has_invoked and is_at_top,
        f"hook must contain `log \"INVOKED auto-commit-and-push pid=$$\"` ABOVE the masterplan-exists guard (invoked_line={invoked_line}, guard_line={guard_line})",
    )

    # --- 5. Hook bash syntax valid ---
    try:
        r = subprocess.run(
            ["bash", "-n", str(hook)],
            capture_output=True, text=True, timeout=10,
        )
        bash_ok = (r.returncode == 0)
        bash_detail = r.stderr.strip() if r.stderr else ""
    except Exception as exc:
        bash_ok = False
        bash_detail = str(exc)
    _check(
        "5. auto_commit_hook_bash_syntax_valid",
        bash_ok,
        f"bash -n must accept the hook; stderr: {bash_detail}",
    )

    # --- 6. Hook still filters by newly_done (regression check) ---
    # The newly_done detection lives in a Python heredoc; assert the
    # marker substrings are present so a future edit that breaks the
    # filter is caught.
    newly_done_ok = (
        "newly_done" in hook_text
        and 'subprocess.run' in hook_text
        and 'HEAD:.claude/masterplan.json' in hook_text
        and 'FLIPPED_STEP' in hook_text
        and 'if [ -z "$FLIPPED_STEP" ]' in hook_text
    )
    _check(
        "6. auto_commit_hook_still_filters_by_newly_done",
        newly_done_ok,
        "hook's `newly_done` Python heredoc + bash FLIPPED_STEP guard must still be present (regression check — the INVOKED log MUST NOT replace the filter)",
    )

    # --- 7. Behavioral: invoking the hook writes an INVOKED line ---
    safe, reason = _newly_done_is_empty()
    if not safe:
        _check(
            "7. invocation_writes_invoked_line_to_auto_push_log",
            False,
            f"BEHAVIORAL CLAIM SKIPPED — newly_done is non-empty ({reason}); refusing to invoke hook to avoid triggering a real auto-commit during verification",
        )
        _check(
            "8. invoked_line_includes_timestamp_marker_and_hook_name",
            False,
            "skipped (claim 7 skipped)",
        )
        _check(
            "10. mutation_resistance_removing_invoked_line_breaks_behavioral_claim",
            False,
            "skipped (claim 7 skipped)",
        )
    else:
        before = _count_invoked_lines(log_path)
        try:
            subprocess.run(
                ["bash", str(hook)],
                capture_output=True, text=True, timeout=15, cwd=str(REPO),
            )
            invocation_ok = True
        except Exception as exc:
            invocation_ok = False
            print(f"  (invocation failed: {exc})", file=sys.stderr)
        after = _count_invoked_lines(log_path)
        delta = after - before
        _check(
            "7. invocation_writes_invoked_line_to_auto_push_log",
            invocation_ok and delta >= 1,
            f"invoking the hook must append >=1 INVOKED line (before={before}, after={after}, delta={delta})",
        )

        # --- 8. INVOKED line format check ---
        tail = log_path.read_text(encoding="utf-8").splitlines()[-1] if log_path.exists() else ""
        fmt_re = re.compile(
            r"\[20\d\d-\d\d-\d\dT\d\d:\d\d:\d\dZ\] INVOKED auto-commit-and-push pid=\d+$"
        )
        fmt_ok = bool(fmt_re.match(tail))
        _check(
            "8. invoked_line_includes_timestamp_marker_and_hook_name",
            fmt_ok,
            f"tail line must match `[ISO-8601] INVOKED auto-commit-and-push pid=N`; got: {tail!r}",
        )

        # --- 10. Mutation-resistance: copy script, remove INVOKED line, invoke, expect no new INVOKED ---
        # (claim numbering keeps 9 = sibling-hook syntax for ordering)
        with tempfile.TemporaryDirectory() as tmpd:
            tmp_hook = Path(tmpd) / "auto-commit-and-push-MUTATED.sh"
            # Remove the INVOKED log line entirely. Use regex to match
            # exactly the literal line.
            mutated_text = re.sub(
                r'log "INVOKED auto-commit-and-push pid=\$\$"\n',
                '',
                hook_text,
            )
            tmp_hook.write_text(mutated_text, encoding="utf-8")
            mutated_has_invoked = "INVOKED auto-commit-and-push" in tmp_hook.read_text(encoding="utf-8")
            if mutated_has_invoked:
                _check(
                    "10. mutation_resistance_removing_invoked_line_breaks_behavioral_claim",
                    False,
                    "mutation step failed to strip the INVOKED line from the script copy",
                )
            else:
                before_m = _count_invoked_lines(log_path)
                try:
                    subprocess.run(
                        ["bash", str(tmp_hook)],
                        capture_output=True, text=True, timeout=15, cwd=str(REPO),
                    )
                    mut_invocation_ok = True
                except Exception:
                    mut_invocation_ok = False
                after_m = _count_invoked_lines(log_path)
                delta_m = after_m - before_m
                # Mutated copy should NOT add any INVOKED lines.
                _check(
                    "10. mutation_resistance_removing_invoked_line_breaks_behavioral_claim",
                    mut_invocation_ok and delta_m == 0,
                    f"mutated copy (INVOKED line removed) must NOT append INVOKED to log; before={before_m}, after={after_m}, delta={delta_m}",
                )

    # --- 9. No regressions: sibling hooks bash syntax valid ---
    siblings = [
        REPO / ".claude" / "hooks" / "masterplan-memory-sync.sh",
        REPO / ".claude" / "hooks" / "archive-handoff.sh",
        REPO / ".claude" / "hooks" / "commit-reminder.sh",
        REPO / ".claude" / "hooks" / "post-commit-changelog.sh",
        REPO / ".claude" / "hooks" / "pre-tool-use-danger.sh",
        REPO / ".claude" / "hooks" / "config-change-audit.sh",
        REPO / ".claude" / "hooks" / "instructions-loaded-research-gate.sh",
        REPO / ".claude" / "hooks" / "teammate-idle-check.sh",
    ]
    all_ok = True
    failures: list[str] = []
    for s in siblings:
        if not s.exists():
            continue
        try:
            r = subprocess.run(
                ["bash", "-n", str(s)],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode != 0:
                all_ok = False
                failures.append(f"{s.name}: {r.stderr.strip()}")
        except Exception as exc:
            all_ok = False
            failures.append(f"{s.name}: {exc}")
    _check(
        "9. no_regressions_other_hooks_bash_syntax_valid",
        all_ok,
        f"all sibling hooks must pass `bash -n`; failures: {failures}",
    )

    # --- 11. harness_log has Cycle 41 entry ---
    # This is the log-last protocol check. At Q/A spawn time this MUST
    # FAIL (log is appended last); after the LOG phase it MUST PASS.
    log_has_41 = False
    if harness_log.exists():
        text = harness_log.read_text(encoding="utf-8")
        log_has_41 = (
            "Cycle 41" in text
            and "phase=23.8.4" in text
        )
    _check(
        "11. harness_log_has_cycle_41_entry",
        log_has_41,
        "harness_log.md must contain `## Cycle 41 -- ... -- phase=23.8.4` entry (will FAIL at Q/A time per log-last protocol; PASSes after LOG phase)",
    )

    # --- Print + exit ---
    print("=== phase-23.8.4 verifier ===")
    fail_count = 0
    # Sort results by their claim number prefix to keep them in order.
    def sort_key(t: tuple[str, str, str]) -> tuple[int, str]:
        m = re.match(r"(\d+)\.", t[1])
        return (int(m.group(1)) if m else 99, t[1])

    for flag, name, detail in sorted(RESULTS, key=sort_key):
        prefix = "[PASS]" if flag == "PASS" else "[FAIL]"
        print(f"  {prefix} {name}")
        if flag == "FAIL" and detail:
            print(f"         -> {detail}")
            fail_count += 1
    total = len(RESULTS)
    pass_count = total - fail_count
    summary = "PASS" if fail_count == 0 else "FAIL"
    print(f"{summary} ({pass_count}/{total}) EXIT={0 if fail_count == 0 else 1}")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
