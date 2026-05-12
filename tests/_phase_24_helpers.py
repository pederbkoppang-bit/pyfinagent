"""Shared helpers for phase-24.x audit-bucket verifiers.

Each verifier `tests/verify_phase_24_<n>.py` is a thin wrapper around
the `Phase24Verifier` class defined here.

Stdlib only (no third-party deps). Idempotent. Safe to re-run.

Common claims encoded here:
- findings_md_exists_at_<path>
- research_gate_envelope_present_with_gate_passed_true
- external_sources_count_at_least_5
- canonical_url_cited_verbatim_<slug>
- recency_scan_2024_2026_section_present
- at_least_three_phase_25_candidate_steps_proposed
- each_candidate_step_has_files_list_with_absolute_paths
- each_candidate_step_has_draft_verification_command
- harness_log_has_phase_24_<n>_cycle_entry
- executive_summary_section_present

Bucket-specific extra claims are added by each verifier via
`v.check(name, ok, detail)` after the common pack.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AUDIT_DIR = REPO / "docs" / "audits" / "phase-24-2026-05-12"
HARNESS_LOG = REPO / "handoff" / "harness_log.md"

# Map known canonical URL substrings to the literal strings that must
# appear in the findings doc. Verifier asserts presence by substring.
CANONICAL_URLS = {
    # Pseudo-keys used in success_criteria strings -> URL substring to grep
    "paper_trader_py": "paper_trader.py",
    "autonomous_loop_py": "autonomous_loop.py",
    "anthropic_harness_design": "anthropic.com/engineering/harness-design-long-running-apps",
    "multi_agent_orchestrator_py": "multi_agent_orchestrator.py",
    "slack_bot": "backend/slack_bot",
    "backtest_engine_py": "backtest_engine.py",
    "bigquery_client_py": "bigquery_client.py",
    "observability_api_py": "observability_api.py",
    "anthropic_com_engineering": "anthropic.com/engineering",
    "modelcontextprotocol_io": "modelcontextprotocol.io",
    "frontend_src_lib_api_ts": "frontend/src/lib/api.ts",
    "claude_rules_frontend_md": ".claude/rules/frontend.md",
    "project_system_goal_md": "project_system_goal",
    "phase_25": "phase-25",
}


def findings_path(bucket_id: str, slug: str) -> Path:
    """Compose the expected findings doc path for a bucket."""
    return AUDIT_DIR / f"{bucket_id}-{slug}-findings.md"


class Phase24Verifier:
    """Collects PASS/FAIL results and prints a summary block."""

    def __init__(self, bucket_id: str, slug: str):
        self.bucket_id = bucket_id
        self.slug = slug
        self.results: list[tuple[str, str, str]] = []
        self._findings_text: str | None = None
        self._findings_path: Path = findings_path(bucket_id, slug)

    # --- result collection ---

    def check(self, name: str, ok: bool, detail: str = ""):
        flag = "PASS" if ok else "FAIL"
        self.results.append((flag, name, detail))
        return ok

    def _load_findings(self) -> str:
        if self._findings_text is None:
            if self._findings_path.exists():
                self._findings_text = self._findings_path.read_text(encoding="utf-8")
            else:
                self._findings_text = ""
        return self._findings_text

    # --- common assertions (used by every bucket) ---

    def check_findings_exists(self) -> bool:
        key = f"findings_md_exists_at_docs_audits_phase_24_2026_05_12_{self.bucket_id.replace('.', '_')}_{self.slug.replace('-', '_')}_findings_md"
        ok = self._findings_path.exists()
        self.check(key, ok, f"expected file at {self._findings_path}")
        return ok

    def check_research_gate_envelope(self) -> bool:
        text = self._load_findings()
        # Envelope is JSON-ish; look for "gate_passed": true (substring tolerant)
        ok = bool(re.search(r'"gate_passed"\s*:\s*true', text))
        self.check(
            "research_gate_envelope_present_with_gate_passed_true",
            ok,
            'findings doc must include `"gate_passed": true` JSON envelope (researcher subagent output)',
        )
        return ok

    def check_min_external_sources(self, minimum: int = 5) -> bool:
        text = self._load_findings()
        # Count distinct http(s) URLs as a proxy for source count
        urls = set(re.findall(r"https?://[^\s\)\]\"'`<>]+", text))
        ok = len(urls) >= minimum
        self.check(
            f"external_sources_count_at_least_{minimum}",
            ok,
            f"found {len(urls)} distinct URLs; expected >= {minimum}",
        )
        return ok

    def check_canonical_url(self, canonical_key: str) -> bool:
        substring = CANONICAL_URLS.get(canonical_key, canonical_key)
        text = self._load_findings()
        ok = substring in text
        self.check(
            f"canonical_url_cited_verbatim_{canonical_key}",
            ok,
            f"findings doc must cite '{substring}' verbatim somewhere",
        )
        return ok

    def check_recency_scan(self) -> bool:
        text = self._load_findings()
        # Section header "Recency scan" with year range
        ok = bool(re.search(r"recency\s+scan", text, re.IGNORECASE)) and bool(
            re.search(r"202[4-6]", text)
        )
        self.check(
            "recency_scan_2024_2026_section_present",
            ok,
            'findings doc must contain a "Recency scan" section with 2024-2026 references',
        )
        return ok

    def check_phase_25_candidates(self, minimum: int = 3) -> bool:
        text = self._load_findings()
        # Count distinct phase-25.<N> mentions OR "Candidate <N>:" markers
        candidates = set(re.findall(r"phase-25\.\d+(?:\.\d+)?", text))
        candidate_markers = re.findall(r"(?im)^\s*#{0,4}\s*candidate\s*\d+\s*[:.]?", text)
        count = max(len(candidates), len(candidate_markers))
        ok = count >= minimum
        self.check(
            "at_least_three_phase_25_candidate_steps_proposed",
            ok,
            f"found {count} candidates (distinct phase-25.N refs: {len(candidates)}; "
            f"Candidate N: markers: {len(candidate_markers)}); expected >= {minimum}",
        )
        return ok

    def check_candidates_have_files(self) -> bool:
        text = self._load_findings()
        # Look for "Files:" or "files:" lines followed by file paths;
        # at minimum require at least 3 occurrences of "Files:" block + at
        # least one absolute path under it.
        files_blocks = re.findall(
            r"(?im)^\s*(?:files?:|files? to (?:edit|create|modify):)",
            text,
        )
        absolute_paths = re.findall(r"backend/\S+|frontend/\S+|\.claude/\S+", text)
        ok = len(files_blocks) >= 3 and len(absolute_paths) >= 3
        self.check(
            "each_candidate_step_has_files_list_with_absolute_paths",
            ok,
            f"found {len(files_blocks)} 'Files:' blocks and {len(absolute_paths)} path-like refs; "
            "each candidate must list files with absolute-ish paths",
        )
        return ok

    def check_candidates_have_verif_commands(self) -> bool:
        text = self._load_findings()
        # Look for "verification" or "verification command" near each candidate.
        verif_mentions = re.findall(
            r"(?im)verification\s*command|verification:\s*`|tests/verify_",
            text,
        )
        ok = len(verif_mentions) >= 3
        self.check(
            "each_candidate_step_has_draft_verification_command",
            ok,
            f"found {len(verif_mentions)} verification-command mentions; expected >= 3 (one per candidate)",
        )
        return ok

    def check_harness_log_cycle_entry(self) -> bool:
        if not HARNESS_LOG.exists():
            self.check(
                f"harness_log_has_phase_24_{self.bucket_id.replace('.', '_')}_cycle_entry",
                False,
                "handoff/harness_log.md does not exist",
            )
            return False
        log_text = HARNESS_LOG.read_text(encoding="utf-8")
        # Cycle entry format from CLAUDE.md: "## Cycle N -- YYYY-MM-DD -- phase=X.Y result=PASS"
        # Bucket-id appears in the header line.
        pat = re.compile(
            rf"^\s*##\s*cycle\s*\d+\s*--.*phase={re.escape(self.bucket_id)}\s+result=",
            re.IGNORECASE | re.MULTILINE,
        )
        ok = bool(pat.search(log_text))
        self.check(
            f"harness_log_has_phase_24_{self.bucket_id.replace('.', '_')}_cycle_entry",
            ok,
            f"harness_log.md must contain `## Cycle N -- ... phase={self.bucket_id} result=...` header",
        )
        return ok

    def check_executive_summary(self) -> bool:
        text = self._load_findings()
        ok = bool(re.search(r"(?im)^\s*##?\s*executive\s+summary", text))
        self.check(
            "executive_summary_section_present",
            ok,
            'findings doc must include an "Executive summary" section header',
        )
        return ok

    # --- bucket-specific helper for direct grep ---

    def check_grep(self, claim_name: str, pattern: str, *, flags=re.IGNORECASE | re.MULTILINE, min_matches: int = 1, detail: str = ""):
        """Bucket-specific anchor check: grep findings doc for `pattern`."""
        text = self._load_findings()
        matches = re.findall(pattern, text, flags=flags)
        ok = len(matches) >= min_matches
        self.check(
            claim_name,
            ok,
            detail or f"pattern {pattern!r} matched {len(matches)} times; expected >= {min_matches}",
        )
        return ok

    # --- one-line common-pack runner ---

    def run_common_pack(self, canonical_url_key: str):
        """Run all the common checks in canonical order.

        Bucket-specific extra checks should be added BEFORE calling
        `finish()`.
        """
        self.check_findings_exists()
        self.check_research_gate_envelope()
        self.check_min_external_sources(minimum=5)
        self.check_canonical_url(canonical_url_key)
        self.check_recency_scan()
        self.check_phase_25_candidates(minimum=3)
        self.check_candidates_have_files()
        self.check_candidates_have_verif_commands()
        self.check_harness_log_cycle_entry()
        self.check_executive_summary()

    # --- output ---

    def finish(self) -> int:
        print(f"=== phase-24.{self.bucket_id.split('.')[-1]} ({self.slug}) verifier ===")
        fail = 0
        for flag, name, detail in self.results:
            prefix = "[PASS]" if flag == "PASS" else "[FAIL]"
            print(f"  {prefix} {name}")
            if flag == "FAIL" and detail:
                print(f"         -> {detail}")
                fail += 1
        total = len(self.results)
        passed = total - fail
        verdict = "PASS" if fail == 0 else "FAIL"
        print(f"{verdict} ({passed}/{total}) EXIT={0 if fail == 0 else 1}")
        return 0 if fail == 0 else 1
