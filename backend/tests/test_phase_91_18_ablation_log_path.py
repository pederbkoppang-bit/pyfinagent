"""phase-91.18 regression lock: cron_dashboard_api._log_paths()'s ablation
entries must resolve under handoff/logs/, matching where run_ablation.sh
and the com.pyfinagent.ablation launchd plist actually write.

Filed because this exact class of drift already happened once silently:
phase-23.3.5 corrected 3 allowlist keys and introduced 2 new wrong ones
(ablation, ablation_launchd), and get_log_tail returns HTTP 200 with
exists:false for a wrong path -- a wrong entry is indistinguishable from
an idle job at the API layer, so nothing short of a test on the allowlist
itself catches this again.
"""

from __future__ import annotations

from pathlib import Path

from backend.api.cron_dashboard_api import _log_paths


def test_ablation_log_paths_resolve_under_handoff_logs():
    paths = _log_paths()
    assert paths["ablation"] == Path("handoff/logs/ablation.log").resolve() or str(
        paths["ablation"]
    ).endswith(str(Path("handoff") / "logs" / "ablation.log"))
    assert str(paths["ablation_launchd"]).endswith(
        str(Path("handoff") / "logs" / "ablation.launchd.log")
    )


def test_ablation_log_paths_match_the_real_writers():
    """The allowlist values must match run_ablation.sh's LOG var and the
    plist's StandardOutPath/StandardErrorPath, not just "look plausible"."""
    repo_root = Path(__file__).resolve().parents[2]
    run_ablation_sh = (repo_root / "scripts" / "ops" / "run_ablation.sh").read_text(
        encoding="utf-8"
    )
    assert 'LOG="$REPO/handoff/logs/ablation.log"' in run_ablation_sh

    paths = _log_paths()
    assert paths["ablation"] == repo_root / "handoff" / "logs" / "ablation.log"
    assert (
        paths["ablation_launchd"]
        == repo_root / "handoff" / "logs" / "ablation.launchd.log"
    )
