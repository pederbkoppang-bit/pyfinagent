"""Revert-hygiene drill (BLOCKER-2 verification).

Proves the hardened mas-harness run_cycle.sh:
  - Refuses to run when the working tree is dirty (exit 0 but ABORT message).
  - Does NOT create a new stash.
  - Does NOT silently drop the uncommitted edit.

Also verifies:
  - cycle_prompt.md carries the hardened wording.
  - .claude/hooks/pre-tool-use-danger.sh has the new case arms.
  - scripts/mas_harness/run_cycle.sh no longer contains `git stash`.

Prints PASS / FAIL. Exits 0/1.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SENTINEL_PATH = ROOT / "backend" / "services" / "_revert_hygiene_sentinel.py"
SENTINEL_BODY = "# revert-hygiene-drill sentinel\n# intentionally left dirty by zero_orders drill\n"


def run(cmd: list[str], cwd: Path = ROOT) -> tuple[int, str]:
    res = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    return res.returncode, (res.stdout or "") + (res.stderr or "")


def count_stashes() -> int:
    code, out = run(["git", "stash", "list"])
    if code != 0:
        return -1
    return len([l for l in out.splitlines() if l.strip()])


def cleanup(sentinel_was_tracked: bool) -> None:
    if SENTINEL_PATH.exists():
        SENTINEL_PATH.unlink()
    # drop any git-index trace of the sentinel (in case a prior run staged it)
    run(["git", "reset", "HEAD", str(SENTINEL_PATH)])


def assert_(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
    return ok


def main() -> int:
    failures = 0

    # Static checks on hardened files (do not depend on running the cycle).
    run_cycle = (ROOT / "scripts" / "mas_harness" / "run_cycle.sh").read_text()
    prompt = (ROOT / "scripts" / "mas_harness" / "cycle_prompt.md").read_text()
    hook = (ROOT / ".claude" / "hooks" / "pre-tool-use-danger.sh").read_text()

    print("Static checks:")
    if not assert_("run_cycle.sh contains 'ABORT dirty tree'", "ABORT dirty tree" in run_cycle):
        failures += 1
    if not assert_("run_cycle.sh does NOT contain 'git stash'", "git stash" not in run_cycle):
        failures += 1
    if not assert_("cycle_prompt.md contains 'do NOT use' hardened rule",
                   "do NOT use" in prompt.lower() or "do not use" in prompt.lower()):
        failures += 1
    if not assert_("pre-tool-use-danger.sh guards file-level git checkout",
                   "git[[:space:]]+checkout" in hook and "(--|HEAD)" in hook):
        failures += 1
    if not assert_("pre-tool-use-danger.sh guards git restore",
                   "git[[:space:]]+restore" in hook):
        failures += 1

    if failures:
        print("FAIL (static checks)")
        return 1

    print("Dynamic checks:")
    # Pre-state
    stashes_before = count_stashes()
    if not assert_("can read stash list", stashes_before >= 0):
        return 1

    # Create an uncommitted sentinel edit. Use a net-new file so we never
    # touch a tracked path that could matter.
    SENTINEL_PATH.write_text(SENTINEL_BODY)
    try:
        # The sentinel file is UNTRACKED; `git status --porcelain` lists it
        # with '??'. That counts as dirty for our guard.
        code, out = run(["git", "status", "--porcelain", str(SENTINEL_PATH)])
        if not assert_("sentinel is visible in git status", SENTINEL_PATH.name in out):
            return 1

        # Drive the harness script directly.
        cycle_code, cycle_out = run(["bash", str(ROOT / "scripts" / "mas_harness" / "run_cycle.sh")])
        # The hardened script logs to handoff/mas-harness.log rather than
        # stdout; grab the tail.
        log_path = ROOT / "handoff" / "mas-harness.log"
        log_tail = log_path.read_text().splitlines()[-60:] if log_path.exists() else []
        log_tail_s = "\n".join(log_tail)

        if not assert_("run_cycle.sh exited 0 (skip is not an error)", cycle_code == 0,
                       detail=f"exit={cycle_code}"):
            failures += 1
        if not assert_("log contains 'ABORT dirty tree'", "ABORT dirty tree" in log_tail_s):
            failures += 1
        if not assert_("sentinel edit still present on disk", SENTINEL_PATH.exists()):
            failures += 1
        if SENTINEL_PATH.exists():
            if not assert_("sentinel content unchanged", SENTINEL_PATH.read_text() == SENTINEL_BODY):
                failures += 1

        stashes_after = count_stashes()
        if not assert_("stash list did not grow", stashes_after == stashes_before,
                       detail=f"before={stashes_before} after={stashes_after}"):
            failures += 1

    finally:
        cleanup(sentinel_was_tracked=False)

    if failures:
        print(f"FAIL ({failures} check(s) failed)")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
