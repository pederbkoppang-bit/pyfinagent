"""phase-4.6 step 4.6.7: Slack digest smoketest.

Posts a one-line message to the channel in SLACK_TEST_CHANNEL_ID using the
SLACK_BOT_TOKEN, then calls conversations.history to verify the message
landed. Emits JSON verdict and exits 0/1.

Usage:
    python -m backend.slack_bot.digest_test \\
        --channel-env SLACK_TEST_CHANNEL_ID \\
        --text smoketest-4.6.7 --verify-delivery

If SLACK_BOT_TOKEN or the env-var named by --channel-env is missing, the
module exits with verdict=SKIP_ENV_MISSING (exit 2) so CI can distinguish
"env not configured" from "real failure" (exit 1).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def _run(channel: str, text: str, token: str, verify: bool) -> dict:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError

    client = WebClient(token=token)
    result: dict = {"step": "4.6.7", "channel_id_prefix": channel[:4] + "***"}

    t0 = time.monotonic()
    try:
        post = client.chat_postMessage(channel=channel, text=text)
    except SlackApiError as e:
        result.update({"verdict": "FAIL", "reason": "chat_postMessage",
                       "error": e.response.get("error", str(e))})
        return result
    if not post.get("ok"):
        result.update({"verdict": "FAIL", "reason": "post_not_ok",
                       "raw": post.data})
        return result

    ts = post["ts"]
    result["post_ts"] = ts
    result["post_ok"] = True

    if verify:
        try:
            hist = client.conversations_history(channel=channel, latest=ts,
                                                 inclusive=True, limit=5)
        except SlackApiError as e:
            result.update({"verdict": "FAIL", "reason": "conversations_history",
                           "error": e.response.get("error", str(e))})
            return result
        if not hist.get("ok"):
            result.update({"verdict": "FAIL", "reason": "history_not_ok",
                           "raw": hist.data})
            return result
        found = any((m.get("ts") == ts and text in m.get("text", ""))
                    for m in hist.get("messages", []))
        result["history_verified"] = found
        if not found:
            result["verdict"] = "FAIL"
            result["reason"] = "posted_message_not_found_in_history"
            return result

    result["elapsed_s"] = round(time.monotonic() - t0, 3)
    if result["elapsed_s"] > 10:
        result["verdict"] = "FAIL"
        result["reason"] = "round_trip_over_10s"
        return result
    result["verdict"] = "PASS"
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel-env", default="SLACK_TEST_CHANNEL_ID",
                    help="env-var name that holds the target Slack channel id")
    ap.add_argument("--text", default="smoketest-4.6.7")
    ap.add_argument("--verify-delivery", action="store_true")
    args = ap.parse_args()

    token = os.getenv("SLACK_BOT_TOKEN")
    channel = os.getenv(args.channel_env)
    if not token or not channel:
        missing = [k for k, v in (("SLACK_BOT_TOKEN", token),
                                   (args.channel_env, channel)) if not v]
        print(json.dumps({"step": "4.6.7", "verdict": "SKIP_ENV_MISSING",
                          "missing": missing,
                          "message": "Set these env-vars in backend/.env"}))
        return 2

    result = _run(channel, args.text, token, args.verify_delivery)
    print(json.dumps(result))
    if result.get("verdict") == "PASS":
        print("SLACK_DIGEST_OK")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
