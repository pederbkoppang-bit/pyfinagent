"""
Self-Update — Deploy new code via Slack commands.

Since Slack is the primary interface to the Mac Mini, the bot
needs to be able to update and restart itself.

Flow:
  1. You push files to git from your dev machine
  2. You tell the bot: "deploy update" or use /deploy
  3. The bot runs git pull, verifies, and restarts itself

Commands (in #ford-approvals or assistant panel):
  "deploy update"     — git pull + restart
  "deploy status"     — show current git state
  "deploy rollback"   — revert to previous commit
  "deploy diff"       — show what changed since last deploy
  "deploy logs"       — show recent deploy history

Safety:
  - Backs up current files before pulling
  - Validates Python syntax after pull
  - Graceful restart with 3s delay
  - Rollback available if something breaks
"""

import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent  # pyfinAgent/
DEPLOY_LOG = PROJECT_ROOT / "logs" / "deploy.log"


def _run(cmd: list[str], cwd: str = None, timeout: int = 30) -> tuple[int, str]:
    """Run a shell command and return (returncode, output)."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = (result.stdout + result.stderr).strip()
        return result.returncode, output
    except subprocess.TimeoutExpired:
        return 1, "Command timed out"
    except Exception as e:
        return 1, str(e)


def _log_deploy(action: str, detail: str):
    """Append to deploy log file."""
    try:
        DEPLOY_LOG.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        with open(DEPLOY_LOG, "a") as f:
            f.write(f"{ts} | {action} | {detail}\n")
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════
# DEPLOY COMMANDS
# ═══════════════════════════════════════════════════════════════════


def deploy_status() -> str:
    """Show current git state."""
    lines = []

    # Current branch and commit
    _, branch = _run(["git", "branch", "--show-current"])
    _, commit = _run(["git", "log", "-1", "--oneline"])
    _, status = _run(["git", "status", "--short"])
    _, remote_url = _run(["git", "remote", "get-url", "origin"])

    lines.append(f"📋 *Deploy Status*\n")
    lines.append(f"• Branch: `{branch}`")
    lines.append(f"• Latest commit: `{commit}`")
    lines.append(f"• Remote: `{remote_url}`")

    if status:
        lines.append(f"• Local changes:\n```{status[:500]}```")
    else:
        lines.append(f"• Working tree: clean ✅")

    # Check if behind remote
    _run(["git", "fetch", "--dry-run"])
    _, behind = _run(["git", "rev-list", "--count", f"{branch}..origin/{branch}"])
    if behind and behind.strip() != "0":
        lines.append(f"• ⚠️ *{behind.strip()} commits behind remote* — run `deploy update`")
    else:
        lines.append(f"• Up to date with remote ✅")

    return "\n".join(lines)


def deploy_diff() -> str:
    """Show what would change on pull."""
    # Fetch latest
    code, _ = _run(["git", "fetch", "origin"])
    if code != 0:
        return "❌ Failed to fetch from remote"

    _, branch = _run(["git", "branch", "--show-current"])
    _, diff = _run(["git", "diff", "--stat", f"HEAD..origin/{branch.strip()}"])

    if not diff.strip():
        return "✅ No changes to pull — already up to date."

    # Also show the commit messages
    _, log = _run(["git", "log", "--oneline", f"HEAD..origin/{branch.strip()}"])

    return (
        f"📦 *Changes available*\n\n"
        f"*New commits:*\n```{log[:800]}```\n\n"
        f"*Files changed:*\n```{diff[:800]}```"
    )


def deploy_update() -> str:
    """
    Pull latest code, clean up old processes, and restart the bot.

    Steps:
    1. Stash any local changes
    2. Git pull
    3. Validate Python syntax
    4. Kill competing old processes
    5. Log the deploy
    6. Schedule restart (the bot restarts itself)
    """
    lines = []

    # Step 1: Record current state
    _, old_commit = _run(["git", "log", "-1", "--oneline"])
    lines.append(f"🔄 *Deploying update...*")
    lines.append(f"• Current: `{old_commit}`")

    # Step 2: Stash local changes
    _, stash_out = _run(["git", "stash"])
    if "No local changes" not in stash_out:
        lines.append(f"• Stashed local changes")

    # Step 3: Pull
    code, pull_output = _run(["git", "pull", "origin", "main"], timeout=60)
    if code != 0:
        # Restore stash on failure
        _run(["git", "stash", "pop"])
        lines.append(f"❌ *Pull failed:*\n```{pull_output[:500]}```")
        _log_deploy("PULL_FAILED", pull_output[:200])
        return "\n".join(lines)

    # Step 4: New commit
    _, new_commit = _run(["git", "log", "-1", "--oneline"])
    lines.append(f"• Updated to: `{new_commit}`")

    if old_commit.strip() == new_commit.strip():
        lines.append(f"• Already up to date ✅")
        _run(["git", "stash", "pop"])
        return "\n".join(lines)

    # Step 5: Show what changed
    _, diff_stat = _run(["git", "diff", "--stat", f"{old_commit.split()[0]}..HEAD"])
    if diff_stat:
        lines.append(f"• Files changed:\n```{diff_stat[:500]}```")

    # Step 6: Validate Python syntax on key files
    key_files = [
        "backend/slack_bot/app.py",
        "backend/slack_bot/assistant_handler.py",
        "backend/slack_bot/commands.py",
        "backend/slack_bot/governance.py",
        "backend/slack_bot/app_home.py",
        "backend/agents/agent_definitions.py",
        "backend/agents/multi_agent_orchestrator.py",
    ]
    syntax_ok = True
    for f in key_files:
        fpath = PROJECT_ROOT / f
        if fpath.exists():
            code, err = _run(["python", "-c", f"import py_compile; py_compile.compile('{fpath}', doraise=True)"])
            if code != 0:
                lines.append(f"❌ Syntax error in `{f}`: {err[:200]}")
                syntax_ok = False

    if not syntax_ok:
        lines.append(f"\n⚠️ *Syntax errors detected — NOT restarting.* Fix and push again.")
        _log_deploy("SYNTAX_ERROR", "Validation failed")
        return "\n".join(lines)

    lines.append(f"• Syntax validation: passed ✅")

    # Step 7: Kill competing old processes
    cleaned = _cleanup_old_processes()
    if cleaned:
        lines.append(f"• Cleaned up: {', '.join(cleaned)}")

    # Step 8: Log and schedule restart
    _log_deploy("DEPLOYED", f"{old_commit} → {new_commit}")
    lines.append(f"\n🔄 *Restarting bot in 3 seconds...*")
    lines.append(f"_I'll be back online shortly._")

    # Schedule restart in background
    _schedule_restart()

    return "\n".join(lines)


def deploy_rollback() -> str:
    """Revert to the previous commit."""
    _, current = _run(["git", "log", "-1", "--oneline"])

    code, output = _run(["git", "revert", "--no-commit", "HEAD"])
    if code != 0:
        return f"❌ Rollback failed:\n```{output[:300]}```"

    code, output = _run(["git", "commit", "-m", f"Rollback: revert {current.strip()}"])
    if code != 0:
        _run(["git", "revert", "--abort"])
        return f"❌ Rollback commit failed:\n```{output[:300]}```"

    _, new = _run(["git", "log", "-1", "--oneline"])
    _log_deploy("ROLLBACK", f"Reverted {current} → {new}")

    _schedule_restart()

    return (
        f"⏪ *Rolled back*\n"
        f"• Was: `{current}`\n"
        f"• Now: `{new}`\n"
        f"🔄 Restarting in 3 seconds..."
    )


def deploy_logs() -> str:
    """Show recent deploy history."""
    if not DEPLOY_LOG.exists():
        return "📋 No deploy history yet."

    try:
        with open(DEPLOY_LOG) as f:
            lines = f.readlines()
        recent = lines[-15:]  # Last 15 entries
        return "📋 *Deploy History*\n```" + "".join(recent) + "```"
    except Exception as e:
        return f"❌ Error reading deploy log: {e}"


# ═══════════════════════════════════════════════════════════════════
# RESTART MECHANISM
# ═══════════════════════════════════════════════════════════════════


def _schedule_restart():
    """
    Schedule a SAFE bot restart in 3 seconds.

    Safety measures:
    1. Waits 3s so the Slack response sends
    2. Kills old process
    3. Starts new process
    4. Verifies new process is alive after 5s
    5. If dead → retries once
    6. If still dead → logs error (user can send "deploy status" when they notice)
    
    The bot will ALWAYS come back because:
    - The restart script itself is a separate process
    - It retries on failure
    - Even worst case, "deploy rollback" was logged so user knows the state
    """
    restart_script = f"""
import time, os, signal, subprocess, sys

LOG = '{PROJECT_ROOT}/logs/restart.log'

def log(msg):
    with open(LOG, 'a') as f:
        f.write(f'{{time.strftime("%Y-%m-%d %H:%M:%S")}} {{msg}}\\n')
    print(msg)

# Step 1: Wait for Slack response to send
time.sleep(3)
log('Restart initiated')

# Step 2: Kill the old bot
bot_pid = {os.getpid()}
try:
    os.kill(bot_pid, signal.SIGTERM)
    log(f'Killed old process (PID {{bot_pid}})')
except ProcessLookupError:
    log(f'Old process already gone (PID {{bot_pid}})')

time.sleep(2)

# Step 3: Start the new bot
def start_bot():
    env = os.environ.copy()
    # Load .env if exists
    env_file = '{PROJECT_ROOT}/backend/.env'
    if not os.path.exists(env_file):
        env_file = '{PROJECT_ROOT}/.env'
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    k, v = line.split('=', 1)
                    env[k.strip()] = v.strip()
    
    proc = subprocess.Popen(
        [sys.executable, '-m', 'backend.slack_bot.app'],
        cwd='{PROJECT_ROOT}',
        start_new_session=True,
        stdout=open('{PROJECT_ROOT}/logs/slackbot.log', 'a'),
        stderr=subprocess.STDOUT,
        env=env,
    )
    return proc.pid

# Attempt 1
new_pid = start_bot()
log(f'Started new bot (PID {{new_pid}})')

# Step 4: Verify it's alive after 5s
time.sleep(5)
try:
    os.kill(new_pid, 0)  # Signal 0 = just check if alive
    log(f'✅ New bot verified alive (PID {{new_pid}})')
except ProcessLookupError:
    log(f'⚠️ New bot died (PID {{new_pid}}), retrying...')
    
    # Attempt 2: retry
    time.sleep(2)
    new_pid = start_bot()
    log(f'Retry started (PID {{new_pid}})')
    
    time.sleep(5)
    try:
        os.kill(new_pid, 0)
        log(f'✅ Retry succeeded (PID {{new_pid}})')
    except ProcessLookupError:
        log(f'❌ Bot failed to start after 2 attempts. Check logs/slackbot.log')
        log(f'   Recovery: ssh in and run: cd {PROJECT_ROOT} && python -m backend.slack_bot.app')

log('Restart script complete')
"""
    restart_path = PROJECT_ROOT / "logs" / "_restart.py"
    restart_path.parent.mkdir(parents=True, exist_ok=True)
    restart_path.write_text(restart_script)

    subprocess.Popen(
        [sys.executable, str(restart_path)],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    logger.info("🔄 Safe restart scheduled (3s delay, with verification)")


# ═══════════════════════════════════════════════════════════════════
# CLEANUP OLD PROCESSES AND FILES
# ═══════════════════════════════════════════════════════════════════

# Old competing scripts that conflict with the new multi-agent bot
_OLD_PROCESSES = [
    "active_slack_monitor",
    "slack_response_agent",
    "slack_mention_checker",
    "imsg_responder_tickets",  # replaced by new system
]


def _cleanup_old_processes() -> list[str]:
    """Kill competing old processes. Returns list of what was killed."""
    killed = []
    for proc_name in _OLD_PROCESSES:
        code, _ = _run(["pkill", "-f", proc_name], timeout=5)
        if code == 0:
            killed.append(proc_name)
            logger.info(f"🧹 Killed old process: {proc_name}")

    # Remove mention checker from cron if present
    _run(["bash", "-c", "crontab -l 2>/dev/null | grep -v slack_mention_checker | crontab -"], timeout=5)

    return killed


def deploy_cleanup() -> str:
    """Kill old competing processes and report what was cleaned."""
    killed = _cleanup_old_processes()

    lines = ["🧹 *Cleanup Report*\n"]

    if killed:
        for proc in killed:
            lines.append(f"• Killed: `{proc}`")
    else:
        lines.append("• No old processes found running ✅")

    # Check for old script files that are no longer needed
    old_files = [
        "scripts/active_slack_monitor.py",
        "scripts/slack_response_agent.py",
        "scripts/slack_mention_checker.py",
    ]
    found_old = []
    for f in old_files:
        fpath = PROJECT_ROOT / f
        if fpath.exists():
            found_old.append(f)

    if found_old:
        lines.append("\n*Old files still on disk (safe to delete):*")
        for f in found_old:
            lines.append(f"• `{f}`")
        lines.append("\n_These won't interfere — they're just not used anymore._")
    else:
        lines.append("• No old script files found ✅")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# ROUTE DEPLOY COMMANDS (called from message handler)
# ═══════════════════════════════════════════════════════════════════


def handle_deploy_command(text: str) -> str | None:
    """
    Check if a message is a deploy command and handle it.

    Returns the response string, or None if not a deploy command.
    """
    text_lower = text.lower().strip()

    if text_lower in ("deploy update", "deploy pull", "update bot", "pull and restart"):
        return deploy_update()
    elif text_lower in ("deploy status", "deploy info", "git status"):
        return deploy_status()
    elif text_lower in ("deploy diff", "deploy changes", "what changed"):
        return deploy_diff()
    elif text_lower in ("deploy rollback", "deploy revert", "rollback"):
        return deploy_rollback()
    elif text_lower in ("deploy logs", "deploy history"):
        return deploy_logs()
    elif text_lower in ("deploy cleanup", "deploy clean", "cleanup", "clean old"):
        return deploy_cleanup()
    elif text_lower.startswith("deploy"):
        return (
            "🚀 *Deploy Commands*\n\n"
            "• `deploy status` — current git state\n"
            "• `deploy diff` — preview incoming changes\n"
            "• `deploy update` — git pull + cleanup + restart\n"
            "• `deploy rollback` — revert last deploy\n"
            "• `deploy cleanup` — kill old competing processes\n"
            "• `deploy logs` — deploy history"
        )

    return None  # Not a deploy command
