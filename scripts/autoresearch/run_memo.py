#!/usr/bin/env python3
"""
scripts/autoresearch/run_memo.py

Nightly gpt-researcher runner. Picks a topic from topics.txt (rotating
on day-of-year modulo N_topics), runs one research pass with Claude +
arxiv/semantic_scholar/duckduckgo retrievers (no Tavily, no OpenAI),
and writes a timestamped markdown memo to handoff/autoresearch/.

The memo is a valid research-gate source for the MAS harness:
.claude/context/research-gate.md includes autoresearch memos in its
accepted-source list. The MAS harness cycle cites them verbatim when
it needs a research-gate check.

Invoked by launchd via scripts/autoresearch/run_nightly.sh.
Can also be run manually:

    source .venv/bin/activate
    python scripts/autoresearch/run_memo.py            # auto-pick by date
    python scripts/autoresearch/run_memo.py --topic-index 3
    python scripts/autoresearch/run_memo.py --topic "custom question..."
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import os
import re
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOPICS = REPO / "scripts" / "autoresearch" / "topics.txt"
MEMO_DIR = REPO / "handoff" / "autoresearch"


def load_topics() -> list[str]:
    if not TOPICS.exists():
        sys.exit(f"topics file missing: {TOPICS}")
    topics: list[str] = []
    for line in TOPICS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        topics.append(line)
    if not topics:
        sys.exit("topics file has no usable topics")
    return topics


def pick_topic(topics: list[str], forced_index: int | None) -> tuple[int, str]:
    if forced_index is not None:
        idx = forced_index % len(topics)
    else:
        # day-of-year modulo N gives a 14-day rotation for 14 topics
        idx = dt.date.today().timetuple().tm_yday % len(topics)
    return idx, topics[idx]


def slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return s[:60] or "memo"


async def run_research(topic: str) -> str:
    # Late import so --help works without the package installed.
    from gpt_researcher import GPTResearcher

    # report_type="detailed_report" balances depth with cost/runtime
    # (5-10 min, ~$1-3 on Claude Sonnet). Switch to "deep" once the
    # first week of memos validates budget impact.
    researcher = GPTResearcher(
        query=topic,
        report_type="detailed_report",
        report_format="markdown",
        verbose=False,
    )
    await researcher.conduct_research()
    report = await researcher.write_report()
    return report


def write_memo(topic: str, idx: int, body: str) -> Path:
    MEMO_DIR.mkdir(parents=True, exist_ok=True)
    date = dt.date.today().isoformat()
    slug = slugify(topic)
    path = MEMO_DIR / f"{date}-topic{idx:02d}-{slug}.md"

    header = (
        f"# Autoresearch memo -- {date}\n\n"
        f"**Topic (index {idx}):** {topic}\n\n"
        f"**Source:** gpt-researcher `detailed_report`, Claude-driven, "
        f"arxiv + semantic_scholar + duckduckgo retrievers.\n\n"
        f"---\n\n"
    )
    path.write_text(header + body + "\n", encoding="utf-8")
    return path


async def _main_async(args: argparse.Namespace) -> int:
    topics = load_topics()
    if args.topic:
        idx, topic = -1, args.topic
    else:
        idx, topic = pick_topic(topics, args.topic_index)

    print(f"[autoresearch] topic idx={idx} -- {topic[:80]}...", flush=True)

    try:
        body = await run_research(topic)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        MEMO_DIR.mkdir(parents=True, exist_ok=True)
        err_path = MEMO_DIR / f"{dt.date.today().isoformat()}-ERROR-topic{idx:02d}.md"
        err_path.write_text(
            f"# Autoresearch FAILED -- {dt.date.today().isoformat()}\n\n"
            f"Topic: {topic}\n\nError: {type(e).__name__}: {e}\n",
            encoding="utf-8",
        )
        print(f"[autoresearch] FAILED -- wrote {err_path}", flush=True)
        return 1

    path = write_memo(topic, idx, body)
    print(f"[autoresearch] wrote {path} ({len(body)} chars)", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Nightly autoresearch memo runner")
    parser.add_argument("--topic", help="Override topic (freeform question)")
    parser.add_argument("--topic-index", type=int, help="Force a specific topics.txt index")
    args = parser.parse_args()

    # Baseline env config for Claude + no-Tavily retrieval. Launchd plist
    # sets ANTHROPIC_API_KEY; everything else is derived here so the
    # script is self-contained and doesn't rely on a shell rc file.
    env_defaults = {
        "FAST_LLM": "anthropic:claude-haiku-4-5",
        "SMART_LLM": "anthropic:claude-sonnet-4-6",
        "STRATEGIC_LLM": "anthropic:claude-opus-4-6",
        "EMBEDDING": "huggingface:BAAI/bge-small-en-v1.5",
        "RETRIEVER": "arxiv,semantic_scholar,duckduckgo",
    }
    for k, v in env_defaults.items():
        os.environ.setdefault(k, v)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not set in environment")

    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
