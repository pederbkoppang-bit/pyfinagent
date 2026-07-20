"""
Slash command handlers: /analyze, /portfolio, /report
Message handlers: status command, push approval reactions
Ticket ingestion: all messages in #ford-approvals are persisted as tickets
"""

import asyncio
import logging
import subprocess
import time
from pathlib import Path

import httpx
from slack_bolt.async_app import AsyncApp

from backend.config.settings import get_settings
from backend.slack_bot.formatters import format_analysis_result, format_portfolio_summary, format_report_card
from backend.services.ticket_ingestion import get_ingestion_service

logger = logging.getLogger(__name__)

# Backend base URL (internal Docker network or localhost)
_BACKEND_URL = "http://localhost:8000"

# Ford approval channel
_APPROVAL_CHANNEL = "C0ANTGNNK8D"

# Project root
_PROJECT_ROOT = Path(__file__).parent.parent.parent

# phase-75.2 (gap1-01) / phase-75.2.1: pending push approvals, keyed by the ts
# of the bot-posted request message. A reaction only authorizes a push when it
# lands on one of these, and each is single-use. Process-local: it resets on
# restart, and an empty mapping denies every reaction -- the fail-closed default.
#
# phase-75.2.1 binds the approval to WHAT WAS SHOWN, not merely to the message:
# the value carries the HEAD sha displayed in the request plus an expiry. A ts
# alone records THAT an approval was requested, not what the operator actually
# reviewed, so a commit landing between request and reaction would ride an
# approval it was never part of (OWASP Transaction Authorization 2.6/2.8 -- the
# approved data must be re-validated at execution). The TTL is 2.9.
_APPROVAL_TTL_SECONDS = 600.0

# ts -> (head_sha_shown, expires_at_monotonic)
_pending_push_ts: dict[str, tuple[str, float]] = {}

# phase-75.2.1: module-scope so tests exercise the PRODUCTION regex objects.
# Re-declaring copies in a test cannot detect drift in these.
import re as _re_mod

TOKEN_KEYWORD_RE = _re_mod.compile(
    r"^(?:[0-9][0-9.]*\s+)?[A-Z][A-Z0-9 _-]+:\s*.+$|^(?:HALT-DEV|RESUME-DEV)$"
)
PUSH_REQUEST_KEYWORD_RE = _re_mod.compile(r"^\s*PUSH\s*$")


def register_push_approval_request(
    ts: str, *, head_sha: str, ttl_seconds: float = _APPROVAL_TTL_SECONDS
) -> None:
    """Record a bot-posted push-approval request as approvable.

    `head_sha` is the commit the request message displayed, and it is REQUIRED
    (keyword-only, no default): the reaction handler skips its TOCTOU
    re-validation when the stored sha is empty, so a default would let a future
    caller silently fail open on a git-push authorization path.
    """
    if ts:
        _pending_push_ts[ts] = (head_sha, time.monotonic() + ttl_seconds)


def _pending_push_payload() -> tuple[str, str]:
    """Return (head_sha, commit_list) for commits not yet on origin/main.

    Blocking: callers must dispatch this via asyncio.to_thread so the
    Socket-Mode event loop is never held. Both git invocations carry their own
    timeout, which is the real bound -- cancelling the awaiting coroutine does
    not kill the worker thread.

    NOTE origin/main is a LOCAL ref. No fetch is performed here (it would add
    network to this path and widen the window between what is displayed and
    what is pushed), so the comparison is against the last-known origin/main.
    Callers must say so rather than imply freshness.
    """
    head_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=str(_PROJECT_ROOT), text=True, timeout=5,
    ).strip()
    commits = subprocess.check_output(
        ["git", "log", "origin/main..HEAD", "--oneline"],
        cwd=str(_PROJECT_ROOT), text=True, timeout=5,
    ).strip()
    return head_sha, commits


def _resolve_head_sha() -> str:
    """Current HEAD sha. Blocking -- dispatch via asyncio.to_thread."""
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=str(_PROJECT_ROOT), text=True, timeout=5,
    ).strip()


def _read_status() -> str:
    """Read current plan status from memory and plan files."""
    lines = []

    # Read today's memory
    from datetime import date
    today = date.today().isoformat()
    mem_path = Path.home() / ".openclaw" / "workspace" / "memory" / f"{today}.md"
    if mem_path.exists():
        content = mem_path.read_text()
        # Extract the last section (most recent work)
        sections = content.split("## ")
        if len(sections) > 1:
            last = sections[-1][:500]
            lines.append(f"*Today's work:*\n{last}")

    # Read plan status
    plan_path = _PROJECT_ROOT / "PLAN.md"
    if plan_path.exists():
        content = plan_path.read_text()
        # Count checked vs unchecked items
        checked = content.count("- [x]")
        unchecked = content.count("- [ ]")
        total = checked + unchecked
        pct = int(checked / total * 100) if total > 0 else 0
        lines.append(f"*Plan progress:* {checked}/{total} items ({pct}%)")

    # Git status
    try:
        local_commits = subprocess.check_output(
            ["git", "log", "origin/main..HEAD", "--oneline"],
            cwd=str(_PROJECT_ROOT), text=True, timeout=5
        ).strip()
        if local_commits:
            count = len(local_commits.splitlines())
            lines.append(f"*Local commits waiting for push:* {count}\n```{local_commits}```")
        else:
            lines.append("*Git:* All pushed, up to date.")
    except Exception:
        pass

    # Backtest status
    try:
        import urllib.request, json
        req = urllib.request.Request(f"{_BACKEND_URL}/api/backtest/status")
        resp = json.loads(urllib.request.urlopen(req, timeout=5).read())
        if resp.get("status") == "running":
            p = resp.get("progress", {})
            lines.append(f"*Backtest running:* Window {p.get('window', '?')}/{p.get('total_windows', '?')} ({p.get('elapsed_seconds', 0):.0f}s)")
        elif resp.get("status") == "completed" and resp.get("has_result"):
            lines.append("*Backtest:* Completed ✅")
    except Exception:
        pass

    return "\n\n".join(lines) if lines else "No status available."


def register_commands(app: AsyncApp):
    # ── phase-62.2: operator-token handler (MUST register before the
    # catch-all @app.message below -- Bolt dispatch is first-match-wins in
    # registration order; register_commands is the first registrar). The
    # allowlist lives in the MATCHER so non-matches fall through to ticket
    # ingestion instead of being swallowed.

    from backend.slack_bot.operator_tokens import (
        _authorized,
        append_operator_token,
        is_operator_token_message,
    )

    _settings = get_settings()
    _token_channels = {
        c for c in (_settings.slack_channel_id, _APPROVAL_CHANNEL) if c
    }

    async def _operator_token_matcher(message) -> bool:
        return is_operator_token_message(
            message, _settings.slack_operator_user_id, _token_channels
        )

    _TOKEN_KEYWORD = TOKEN_KEYWORD_RE

    # phase-75.2.1: operator-gated push-approval REQUEST path. Without this,
    # _pending_push_ts stays empty forever and the checkmark-to-push flow that
    # 75.2 secured is inert.
    #
    # Trigger is deliberately COLON-LESS and anchored. `PUSH REQUEST: main`
    # matches TOKEN_KEYWORD_RE, so an unanchored trigger would make these two
    # handlers ambiguous. NOTE the ordering: this handler registers FIRST
    # (index 0, before the operator-token handler and the catch-all), and Bolt
    # dispatch is first-match-wins -- so the hazard runs THIS way: widening
    # PUSH_REQUEST_KEYWORD_RE (e.g. to ^PUSH\b) would make this path swallow
    # operator TOKENS. The two regexes are disjoint today and
    # test_trigger_does_not_collide_with_the_operator_token_grammar pins that
    # against the production objects.
    _PUSH_REQUEST_KEYWORD = PUSH_REQUEST_KEYWORD_RE

    async def _push_request_matcher(message) -> bool:
        # Capability gate. Authorization is re-checked at the sink below --
        # a matcher is not an authorization decision.
        return _authorized(
            user=message.get("user"),
            channel=message.get("channel"),
            operator_user_id=_settings.slack_operator_user_id,
            allowed_channels=_token_channels,
            bot_id=message.get("bot_id"),
        )

    @app.message(_PUSH_REQUEST_KEYWORD, matchers=[_push_request_matcher])
    async def handle_push_request(message, say, logger):
        """Post a push-approval request the operator can react to."""
        # Sink-side re-check: fail closed if the matcher was bypassed or the
        # operator id is unset (an unset id must never authorize anything).
        if not _authorized(
            user=message.get("user"),
            channel=message.get("channel"),
            operator_user_id=_settings.slack_operator_user_id,
            allowed_channels=_token_channels,
            bot_id=message.get("bot_id"),
        ):
            logger.warning(
                "push request refused: unauthorized user=%s channel=%s",
                message.get("user"), message.get("channel"),
            )
            return

        thread_ts = message.get("thread_ts") or message.get("ts")
        try:
            head_sha, commits = await asyncio.to_thread(_pending_push_payload)
        except Exception as e:
            logger.error(f"push request: git inspection failed: {e}")
            await say(text=f"Could not inspect the repo: {str(e)[:200]}",
                      thread_ts=thread_ts)
            return

        if not commits:
            # Nothing to approve -- register nothing, so no ts becomes live.
            await say(text="Nothing to push -- HEAD matches the last-known origin/main.",
                      thread_ts=thread_ts)
            return

        count = len(commits.splitlines())
        # Show exactly what is being signed for, in the approval channel --
        # the reaction gate only accepts that channel, so a request posted
        # anywhere else would be un-approvable by construction.
        resp = await say(
            channel=_APPROVAL_CHANNEL,
            text=(
                f"*Push approval requested* -- {count} commit(s) to `origin/main`\n"
                f"HEAD `{head_sha[:12]}`\n"
                f"```{commits}```\n"
                "React :white_check_mark: to push, :x: to reject. "
                f"Valid for {int(_APPROVAL_TTL_SECONDS // 60)} minutes, single use. "
                "Commit list is against the last-known origin/main (no fetch performed); "
                "the push is refused if HEAD moves before you approve."
            ),
        )

        # Bolt's AsyncSay returns AsyncSlackResponse, which exposes .get()
        # but is NOT a dict subclass (MRO is [AsyncSlackResponse, object]).
        # Duck-type it: an isinstance(dict) check silently registers nothing
        # and leaves this whole path inert.
        posted_ts = None
        if resp is not None and hasattr(resp, "get"):
            try:
                posted_ts = resp.get("ts")
            except Exception:  # pragma: no cover - defensive
                posted_ts = None
        if not posted_ts:
            logger.error("push request: no ts returned; nothing registered")
            return

        # Register the BOT's message ts, never the operator's own -- binding to
        # the operator's message would be self-approval.
        register_push_approval_request(posted_ts, head_sha=head_sha)
        logger.info("push approval requested: ts=%s head=%s", posted_ts, head_sha[:12])

    @app.message(_TOKEN_KEYWORD, matchers=[_operator_token_matcher])
    async def handle_operator_token(message, say, body, logger):
        """Record a verbatim operator decision token (goal-away-ops)."""
        thread_ts = message.get("thread_ts") or message.get("ts")
        result = await append_operator_token(
            text=message.get("text", ""),
            user=message.get("user", ""),
            channel=message.get("channel", ""),
            ts=message.get("ts", ""),
            # phase-75.2 (gap1-11): the sink re-checks identity itself.
            operator_user_id=_settings.slack_operator_user_id,
            allowed_channels=_token_channels,
            event_id=body.get("event_id"),
        )
        if result is None:
            await say(
                text="Token already recorded (duplicate delivery) -- no new line written.",
                thread_ts=thread_ts,
            )
            return
        line_no, record = result
        await say(
            text=(
                f"OPERATOR TOKEN RECORDED (operator_tokens.jsonl line {line_no}): "
                f"`{record['raw']}` -- the next away session acts on it. "
                f"Open asks live in handoff/away_ops/pending_tokens.json."
            ),
            thread_ts=thread_ts,
        )

    """Register all slash command handlers."""

    @app.command("/analyze")
    async def handle_analyze(ack, respond, command):
        """Start analysis for a ticker and post result when complete."""
        await ack()

        ticker = (command.get("text") or "").strip().upper()
        if not ticker or not ticker.isalpha() or len(ticker) > 5:
            await respond("Usage: `/analyze AAPL` — provide a valid ticker symbol")
            return

        await respond(f":hourglass_flowing_sand: Analysis started for **{ticker}**... This takes 2-5 minutes.")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Start analysis
                start_res = await client.post(f"{_BACKEND_URL}/api/analysis/", json={"ticker": ticker})
                start_res.raise_for_status()
                analysis_id = start_res.json()["analysis_id"]

                # Poll until complete (max 10 minutes)
                for _ in range(120):
                    await asyncio.sleep(5)
                    status_res = await client.get(f"{_BACKEND_URL}/api/analysis/{analysis_id}")
                    status_res.raise_for_status()
                    data = status_res.json()

                    if data["status"] == "completed":
                        blocks = format_analysis_result(data.get("report", {}), ticker)
                        await respond(blocks=blocks)
                        return
                    elif data["status"] == "failed":
                        await respond(f":x: Analysis failed for {ticker}: {data.get('error', 'Unknown error')}")
                        return

                await respond(f":warning: Analysis for {ticker} timed out. Check the dashboard.")

        except Exception as e:
            logger.exception(f"Error in /analyze for {ticker}")
            await respond(f":x: Error analyzing {ticker}: {str(e)[:200]}")

    @app.command("/portfolio")
    async def handle_portfolio(ack, respond, command):
        """Show portfolio P&L summary."""
        await ack()

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                res = await client.get(f"{_BACKEND_URL}/api/paper-trading/portfolio")
                res.raise_for_status()
                data = res.json()
                blocks = format_portfolio_summary(data)
                await respond(blocks=blocks)
        except Exception as e:
            logger.exception("Error in /portfolio")
            await respond(f":x: Error fetching portfolio: {str(e)[:200]}")

    @app.command("/report")
    async def handle_report(ack, respond, command):
        """Show latest report for a ticker."""
        await ack()

        ticker = (command.get("text") or "").strip().upper()
        if not ticker or not ticker.isalpha() or len(ticker) > 5:
            await respond("Usage: `/report AAPL` — provide a valid ticker symbol")
            return

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                res = await client.get(f"{_BACKEND_URL}/api/reports/{ticker}")
                res.raise_for_status()
                data = res.json()
                blocks = format_report_card(data, ticker)
                await respond(blocks=blocks)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                await respond(f":mag: No report found for {ticker}. Run `/analyze {ticker}` first.")
            else:
                await respond(f":x: Error fetching report for {ticker}")
        except Exception as e:
            logger.exception(f"Error in /report for {ticker}")
            await respond(f":x: Error: {str(e)[:200]}")

    # ── Channel message handlers ─────────────────────────────────

    @app.event("message_deleted")
    async def handle_message_deleted(event, logger):
        """Handle deleted messages — DO NOT delete associated tickets."""
        # When a user deletes their original message, we keep the ticket in the system
        # This preserves queue position and processing state
        message_ts = event.get("deleted_ts")
        logger.info(f"Message {message_ts} was deleted, but ticket preserved in system")
        # Tickets are NOT deleted — they continue processing and remain queryable

    @app.message("")  # Catch all messages
    async def handle_any_message(message, say, logger):
        """Respond to any message in #ford-approvals. Persists as ticket."""
        channel = message.get("channel", "")
        if channel != _APPROVAL_CHANNEL:
            return
        # Don't respond to bot messages
        if message.get("bot_id"):
            return
        
        text = message.get("text", "").strip()
        
        # Validate: reject empty messages
        if not text:
            logger.debug("Empty message received, skipping")
            return
        
        logger.info(f"Message received in #ford-approvals: {text[:100]}")
        
        # ── Ticket ingestion ────────────────────────────────────
        ingestion = get_ingestion_service()
        ticket_id = None
        try:
            ticket_id = ingestion.ingest_slack_message(
                event=message,
                sender_id=message.get("user", "unknown"),
                channel_id=channel,
            )
        except Exception as e:
            logger.exception(f"Failed to ingest message as ticket: {e}")
        
        # Send acknowledgment with ticket info
        ack_msg = None
        if ticket_id is not None:
            try:
                ack_info = ingestion.acknowledge_ticket_immediately(ticket_id)
                ack_msg = ack_info["message"]
                logger.info(f"Ticket #{ticket_id} created and acknowledged")
            except Exception as e:
                logger.exception(f"Failed to acknowledge ticket: {e}")
                ack_msg = f"✅ Message received (ticket #{ticket_id} created). Timestamp: {message.get('ts')}"
        else:
            # Ingestion failed — always send acknowledgment
            ack_msg = f"⚠️ Message received but failed to create ticket. Timestamp: {message.get('ts')}"
            logger.warning(f"Failed to create ticket from message: {text[:100]}")
        
        # ── Route based on content (existing behavior) ──────────
        text_lower = text.lower()
        
        # CLEAR QUEUE COMMAND - purge the ticket queue ONLY (phase-69.1)
        if "clear queue" in text_lower:
            logger.warning("CLEAR QUEUE COMMAND RECEIVED - clearing the ticket queue")
            try:
                import sqlite3
                from contextlib import closing  # phase-23.1.19: ensure FD release

                # phase-69.1 (audit item 3): REMOVED `subprocess.run(["pkill","-9","-f","python"])`.
                # A Slack message in #ford-approvals containing "clear queue" MUST NOT be able
                # to SIGKILL the trading backend / autonomous loop / harness / bot (OWASP command
                # injection / CWE-78: never let external input reach a process-kill sink; use
                # library calls). "Clear queue" now means ONLY: purge the ticket queue.
                db = get_ingestion_service().db
                with closing(sqlite3.connect(db.db_path)) as conn, conn:
                    conn.execute("DELETE FROM tickets")
                    conn.execute("DELETE FROM ticket_counter")
                    conn.execute("INSERT INTO ticket_counter (id, current_number) VALUES (1, 0)")

                await say("Queue cleared: ticket queue purged, counter reset to #1. (No processes were touched.)")
                logger.info("[OK] CLEAR QUEUE: ticket queue purged, counter at 0")
            except Exception as e:
                logger.error(f"Error executing clear queue: {e}")
                await say(f"Clear queue failed: {str(e)[:100]}")
            return
        
        if "status" in text_lower:
            try:
                status_text = _read_status()
                await say(f"📊 *PyFinAgent Status*\n\n{status_text}")
            except Exception as e:
                logger.exception("Error generating status")
                await say(f":x: Error generating status: {str(e)[:200]}")
        
        # Always send acknowledgment (either ticket confirmation or error)
        if ack_msg:
            thread_ts = message.get("thread_ts") or message.get("ts")
            try:
                await say(text=ack_msg, thread_ts=thread_ts)
                logger.debug(f"Acknowledgment sent for message: {text[:50]}")
            except Exception as e:
                logger.exception(f"Failed to send acknowledgment: {e}")

    @app.event("reaction_added")
    async def handle_reaction(event, say):
        """Handle approval reactions on bot-posted push-approval requests.

        phase-75.2 (gap1-01). Authorization is hand-rolled by necessity: Bolt's
        `authorize` is installation-level and performs no per-user check. Order
        matters -- identity first, then the ts binding, so an unauthorized user
        can never reach the push.
        """
        operator = get_settings().slack_operator_user_id
        if not operator:
            # Fail-closed when unconfigured, mirroring is_operator_token_message.
            logger.warning("reaction ignored: slack_operator_user_id unset (fail-closed)")
            return

        # `user` is the reactor; `item_user` is the author of the reacted-to
        # message. Gating on item_user would authorize the wrong party.
        if event.get("user") != operator:
            logger.warning("reaction ignored: non-operator user=%s", event.get("user"))
            return

        item = event.get("item") or {}
        if item.get("channel") != _APPROVAL_CHANNEL:
            return

        ts = item.get("ts")
        if not ts or ts not in _pending_push_ts:
            logger.warning("reaction ignored: ts=%s is not a pending push approval", ts)
            return

        approved_sha, expires_at = _pending_push_ts[ts]
        if time.monotonic() > expires_at:
            _pending_push_ts.pop(ts, None)
            logger.warning("reaction ignored: approval for ts=%s expired", ts)
            await say(text="That push approval expired. Send `PUSH` again to re-request.",
                      thread_ts=ts)
            return

        reaction = event.get("reaction", "")
        if reaction == "white_check_mark":
            _pending_push_ts.pop(ts, None)  # single-use: no replay into repeat pushes

            # phase-75.2.1: re-validate at execution. The operator approved a
            # specific commit list; if HEAD moved since the request, commits
            # they never saw would ride this approval (OWASP 2.6/2.8).
            if approved_sha:
                try:
                    current_sha = await asyncio.to_thread(_resolve_head_sha)
                except Exception as e:
                    logger.error(f"push refused: could not re-resolve HEAD: {e}")
                    await say(text=f"Push refused: could not verify HEAD ({str(e)[:120]}).",
                              thread_ts=ts)
                    return
                if current_sha != approved_sha:
                    logger.warning(
                        "push refused: HEAD moved %s -> %s since approval",
                        approved_sha[:12], current_sha[:12],
                    )
                    await say(
                        text=(
                            f"Push refused: HEAD moved since you approved "
                            f"(`{approved_sha[:12]}` -> `{current_sha[:12]}`). "
                            "Send `PUSH` again to review the current commits."
                        ),
                        thread_ts=ts,
                    )
                    return

            logger.info("Push approved by operator reaction on ts=%s", ts)
            try:
                # to_thread keeps the 30s subprocess off the Socket-Mode loop.
                result = await asyncio.to_thread(
                    subprocess.check_output,
                    ["git", "push", "origin", "main"],
                    cwd=str(_PROJECT_ROOT), text=True, timeout=30,
                    stderr=subprocess.STDOUT,
                )
                await say(text=f"*Pushed to GitHub*\n```{result.strip()}```", thread_ts=ts)
            except subprocess.CalledProcessError as e:
                await say(text=f"*Push failed:*\n```{e.output[:500]}```", thread_ts=ts)
            except Exception as e:
                await say(text=f"*Push error:* {str(e)[:200]}", thread_ts=ts)

        elif reaction == "x":
            _pending_push_ts.pop(ts, None)
            logger.info("Push rejected by operator reaction on ts=%s", ts)
            await say(text="Push rejected. Commits stay local.", thread_ts=ts)
