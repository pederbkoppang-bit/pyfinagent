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


_NETWORK_TOKENS = ("429", "503", "rate limit")


def _is_network_weather(e: BaseException) -> bool:
    """phase-76.9: classify a caught exception as external-retriever
    network weather (arxiv 429/5xx, generic connection/timeout errors)
    vs a real fault. True iff:
      - the exception is an arxiv-package HTTPError (type name "HTTPError"
        with a module starting "arxiv"), OR
      - the exception (or something in its __cause__/__context__ chain) is
        a requests/urllib3/socket ConnectionError/Timeout-class error, OR
      - the exception's str() mentions "429" / "503" / "rate limit"
        (case-insensitive).
    No new dependency: only stdlib exception introspection + best-effort
    optional imports of requests/urllib3 (already transitive deps of
    gpt_researcher; guarded so their absence never breaks classification).
    Kept narrow on purpose (Pitfall P3, research_brief_76.9.md): must not
    widen into a catch-all that swallows real faults.
    """
    def _matches_one(exc: BaseException) -> bool:
        cls = type(exc)
        if cls.__name__ == "HTTPError" and cls.__module__.startswith("arxiv"):
            return True
        if isinstance(exc, (ConnectionError, TimeoutError)):
            return True
        try:
            import requests.exceptions as _rexc
            if isinstance(exc, (_rexc.ConnectionError, _rexc.Timeout)):
                return True
        except ImportError:
            pass
        try:
            import urllib3.exceptions as _uexc
            if isinstance(exc, (_uexc.ConnectTimeoutError, _uexc.ReadTimeoutError,
                                 _uexc.NewConnectionError, _uexc.MaxRetryError)):
                return True
        except ImportError:
            pass
        msg = str(exc).lower()
        return any(tok in msg for tok in _NETWORK_TOKENS)

    seen: set[int] = set()
    exc: BaseException | None = e
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if _matches_one(exc):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


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
        f"semantic_scholar + arxiv + duckduckgo retrievers.\n\n"
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
        if _is_network_weather(e):
            # phase-76.9: external retriever weather (arxiv 429/5xx etc.) --
            # tolerate it. WARN memo, NOT an ERROR memo (downstream counters
            # grep for "-ERROR-"), and rc=0 so run_nightly.sh's 75.11 paging
            # seam does NOT fire. Real faults keep the ERROR/rc=1 path below.
            warn_path = MEMO_DIR / f"{dt.date.today().isoformat()}-WARN-topic{idx:02d}.md"
            warn_path.write_text(
                f"# Autoresearch WARN (network) -- {dt.date.today().isoformat()}\n\n"
                f"Topic: {topic}\n\nError: {type(e).__name__}: {e}\n\n"
                f"External retriever weather; run tolerated per phase-76.9, "
                f"see handoff/autoresearch/root_cause.md\n",
                encoding="utf-8",
            )
            print(f"[autoresearch] WARN (network) -- wrote {warn_path}", flush=True)
            return 0
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


def _gpt_researcher_guard() -> str | None:
    """phase-75.13 (deps-02): return an ASCII FAIL message if the
    gpt_researcher PACKAGE itself is not importable, else None. Without
    this guard, a missing gpt_researcher is only caught by the late
    `from gpt_researcher import GPTResearcher` inside run_research() (:70),
    which IS already loud (the broad except in _main_async -> ERROR file +
    return 1 -> the 75.11 run_nightly.sh paging seam) -- but ONLY if
    _embedding_preflight() below is reached and passes. _embedding_preflight
    is an INTENTIONAL (phase-51.4) away-ops soft-skip that returns 0 the
    moment the configured EMBEDDING backend is missing, which would mask a
    simultaneously-missing gpt_researcher behind a silent exit-0. This guard
    runs BEFORE _embedding_preflight so gpt_researcher's absence is never
    masked and always reaches the loud path."""
    import importlib.util
    if importlib.util.find_spec("gpt_researcher") is None:
        return (
            "[autoresearch] FAIL: gpt_researcher not importable. Install via: "
            "pip install -r scripts/autoresearch/requirements-autoresearch.txt"
        )
    return None


def _embedding_preflight() -> str | None:
    """phase-51.4: return a skip-message if the configured EMBEDDING provider's
    backing module is NOT importable, else None. gpt-researcher builds
    Memory(embedding) UNCONDITIONALLY at GPTResearcher init, so a missing backend
    crashes every run (ModuleNotFoundError -> exit 1 + an ERROR file). Read EMBEDDING
    AFTER the os.environ.setdefault loop so it matches what the library reads."""
    import importlib.util
    provider = os.environ.get("EMBEDDING", "").split(":", 1)[0].strip().lower()
    module = {
        "huggingface": "langchain_huggingface",
        "openai": "langchain_openai",
        "ollama": "langchain_ollama",
    }.get(provider)
    if not module or importlib.util.find_spec(module) is not None:
        return None  # no known backend to check, or it IS installed -> proceed
    pip_hint = {
        "huggingface": "langchain-huggingface sentence-transformers",
        "openai": "langchain-openai",
        "ollama": "langchain-ollama",
    }.get(provider, module)
    return (
        f"autoresearch skipped: embedding provider '{provider}' needs '{module}', "
        f"which is not installed. Enable with: pip install {pip_hint}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Nightly autoresearch memo runner")
    parser.add_argument("--topic", help="Override topic (freeform question)")
    parser.add_argument("--topic-index", type=int, help="Force a specific topics.txt index")
    parser.add_argument(
        "--preflight-only", action="store_true",
        help="phase-62.6 (goal-away-ops): verify deps + embedding preflight and exit 0 "
             "WITHOUT running GPTResearcher (zero LLM spend). The away-window nightly "
             "uses this; full runs resume on the operator token "
             "'AUTORESEARCH SPEND: RESUME' (see handoff/away_ops/pending_tokens.json).",
    )
    args = parser.parse_args()

    # Baseline env config for Claude + no-Tavily retrieval. Launchd plist
    # sets ANTHROPIC_API_KEY; everything else is derived here so the
    # script is self-contained and doesn't rely on a shell rc file.
    # Model IDs now come from backend/config/model_tiers.py so a
    # COST_TIER=live flip at May launch picks up the cheap mapping.
    # Added via repo path so the script still runs standalone.
    sys.path.insert(0, str(REPO))
    from backend.config.model_tiers import resolve_model  # noqa: E402
    # phase-39.1 (OPEN-29): gpt-researcher Config.parse_llm expects the
    # `<llm_provider>:<llm_model>` format (see
    # .venv/lib/.../gpt_researcher/config/config.py:204-221). resolve_model
    # returns just the model id (e.g. "claude-haiku-4-5") -- prefix with
    # the Anthropic provider tag here at the caller boundary so model_tiers
    # stays single-source-of-truth for model ids.
    env_defaults = {
        "FAST_LLM": f"anthropic:{resolve_model('autoresearch_fast')}",
        "SMART_LLM": f"anthropic:{resolve_model('autoresearch_smart')}",
        "STRATEGIC_LLM": f"anthropic:{resolve_model('autoresearch_strategic')}",
        "EMBEDDING": "huggingface:BAAI/bge-small-en-v1.5",
        # phase-76.9: arxiv moved OFF retrievers[0] (the PLANNING slot).
        # gpt-researcher's plan_research() uses retrievers[0] ONLY, and
        # that call is UNGUARDED (upstream issue #1282) while the
        # sub-query fan-out IS wrapped and tolerant. arXiv has been
        # 429-ing polite (3s-delay) clients server-side since ~2026-02,
        # so a fragile retriever in the fatal planning slot is a landmine.
        "RETRIEVER": "semantic_scholar,arxiv,duckduckgo",
    }
    for k, v in env_defaults.items():
        os.environ.setdefault(k, v)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not set in environment")

    # phase-75.13 (deps-02): gpt_researcher package-missing guard -- MUST run
    # BEFORE _embedding_preflight() below so a missing gpt_researcher is
    # never masked by the embedding soft-skip's silent exit 0 (see
    # _gpt_researcher_guard docstring). Loud on purpose: prints to stderr
    # and returns 1, which run_nightly.sh already logs as FAIL + pages
    # after PAGE_AFTER_N consecutive failures (75.11 seam, unmodified here).
    _gpt_fail_msg = _gpt_researcher_guard()
    if _gpt_fail_msg is not None:
        print(_gpt_fail_msg, file=sys.stderr)
        return 1

    # phase-51.4: graceful preflight (see _embedding_preflight) -- skip cleanly
    # (exit 0, no ERROR file, $0) if the configured EMBEDDING backend is absent,
    # instead of crashing every night. NO pip here (owner-gated); self-enables once
    # the dep is installed.
    _skip_msg = _embedding_preflight()
    if _skip_msg is not None:
        print(_skip_msg, file=sys.stderr)
        return 0

    # phase-62.6: $0 away-window mode -- deps verified importable, embedding
    # preflight passed, stop BEFORE any LLM call.
    if args.preflight_only:
        print("preflight-only: deps importable, embedding preflight OK, "
              "skipping GPTResearcher (zero spend)", file=sys.stderr)
        return 0

    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
