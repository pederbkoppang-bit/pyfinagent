"""phase-8.5.3 LLM proposer with narrow-surface diff discipline.

The proposer reads recent autoresearch results + recent git log context and
emits a single-cycle diff bounded to a WHITELIST of files. Diffs are
dict-of-contents (not true `git diff` format) for simplicity: each key is a
target file path, each value is the full new file content proposed by the
LLM. A committing layer (phase-8.5.6 or later) can translate into a real
patch or a `git apply` equivalent.

Scaffold shape; an offline-safe default `llm_call_fn` returns a deterministic
stub diff so tests run without network / API keys.

Fail-open. ASCII-only.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, TypedDict

logger = logging.getLogger(__name__)

WHITELIST: set[str] = {
    "backend/backtest/experiments/optimizer_best.json",
    "backend/autoresearch/candidate_space.yaml",
}


class Diff(TypedDict, total=False):
    files: dict[str, str]  # path -> new content
    rationale: str
    trial_id: str
    read_results_tsv: bool
    read_git_log: bool


def _default_llm_call(*, results_tsv: str, git_log: list[str]) -> Diff:
    """Deterministic stub. Must NOT require a real API key to run."""
    logger.debug("proposer: default stub llm_call invoked")
    return {
        "files": {
            "backend/backtest/experiments/optimizer_best.json": json.dumps(
                {
                    "proposed_by": "stub_llm_call",
                    "params": {"learning_rate": 0.02, "max_depth": 5},
                },
                indent=2,
            )
        },
        "rationale": (
            f"stub proposer; read {len(git_log)} git log lines and "
            f"{len(results_tsv)}-char results.tsv"
        ),
        "trial_id": "stub_0001",
        "read_results_tsv": bool(results_tsv),
        "read_git_log": bool(git_log),
    }


def validate_diff(diff: Diff, whitelist: Iterable[str] = WHITELIST) -> tuple[bool, list[str]]:
    """Return (ok, violations). `violations` lists any file paths outside whitelist."""
    if not isinstance(diff, dict):
        return False, ["diff is not a dict"]
    files = diff.get("files") or {}
    if not isinstance(files, dict) or not files:
        return False, ["diff.files missing or empty"]
    wl = set(whitelist)
    bad: list[str] = [p for p in files.keys() if p not in wl]
    return (len(bad) == 0, bad)


@dataclass
class Proposer:
    """Autoresearch cycle proposer.

    propose(...) returns a Diff dict whose `files` keys are all within
    WHITELIST. A mocked `llm_call_fn` may be injected (typed as a callable
    that takes `results_tsv=` + `git_log=`).
    """

    whitelist: set[str] = field(default_factory=lambda: set(WHITELIST))

    def propose(
        self,
        results_tsv: str,
        git_log: list[str],
        *,
        llm_call_fn: Callable[..., Diff] | None = None,
    ) -> Diff:
        fn = llm_call_fn or _default_llm_call
        try:
            raw = fn(results_tsv=results_tsv, git_log=git_log)
        except Exception as exc:
            logger.warning("proposer: llm_call raised; returning empty diff (%r)", exc)
            return {"files": {}, "rationale": "llm_call_failed", "trial_id": "error", "read_results_tsv": False, "read_git_log": False}
        # Enforce presence of the two "inputs-were-read" flags.
        raw.setdefault("read_results_tsv", bool(results_tsv))
        raw.setdefault("read_git_log", bool(git_log))
        ok, bad = validate_diff(raw, self.whitelist)
        if not ok:
            logger.warning("proposer: diff violates whitelist: %s", bad)
            # Strip bad paths; keep good ones.
            good = {p: c for p, c in (raw.get("files") or {}).items() if p in self.whitelist}
            raw["files"] = good
            raw["rationale"] = (raw.get("rationale") or "") + f" | stripped_paths={bad}"
        return raw


__all__ = ["Proposer", "Diff", "WHITELIST", "validate_diff"]
