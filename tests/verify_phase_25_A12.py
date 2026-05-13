"""phase-25.A12 verifier -- Playwright visual regression CI baseline.

Closes phase-24.12 F-6 (docs/audits/phase-24-2026-05-12/screenshots/
was empty; no baseline images for visual-regression comparison).

Run: source .venv/bin/activate && python3 tests/verify_phase_25_A12.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PW_CONFIG = REPO / "frontend" / "playwright.config.ts"
GHA = REPO / ".github" / "workflows" / "visual-regression.yml"
HELPERS = REPO / "frontend" / "tests" / "visual-regression" / "helpers" / "visual.ts"
SPECS_DIR = REPO / "frontend" / "tests" / "visual-regression"
SNAPS = REPO / "frontend" / "tests" / "visual-regression" / "snapshots" / "chromium"
PKG_JSON = REPO / "frontend" / "package.json"
README = REPO / "frontend" / "tests" / "visual-regression" / "README.md"

EXPECTED_PAGES = (
    "home",
    "paper-trading",
    "performance",
    "backtest",
    "agents",
    "sovereign",
    "reports",
    "agent-map",
)


def main() -> int:
    results: list[tuple[str, str, str]] = []

    # ---- Claim 1: playwright.config.ts exists.
    results.append((
        "PASS" if PW_CONFIG.exists() else "FAIL",
        "playwright_config_ts_exists",
        f"{PW_CONFIG} must exist",
    ))
    if not PW_CONFIG.exists():
        print("FAIL: playwright.config.ts missing -- cannot continue")
        return 1
    cfg_text = PW_CONFIG.read_text(encoding="utf-8")

    # ---- Claim 2: config declares expected keys.
    required_in_cfg = (
        "testDir",
        "maxDiffPixelRatio",
        "threshold",
        "animations",
        "webServer",
        "NEXT_PUBLIC_E2E_TESTING",
        "chromium",
    )
    missing_cfg = [k for k in required_in_cfg if k not in cfg_text]
    results.append((
        "PASS" if not missing_cfg else "FAIL",
        "playwright_config_declares_required_keys",
        f"playwright.config.ts missing keys: {missing_cfg}",
    ))

    # ---- Claim 3: GHA workflow YAML exists.
    results.append((
        "PASS" if GHA.exists() else "FAIL",
        "github_actions_visual_regression_yml_passes",
        f"{GHA} must exist",
    ))
    gha_text = GHA.read_text(encoding="utf-8") if GHA.exists() else ""

    # ---- Claim 4: workflow uses setup-node, playwright install, conditional update step, artifact uploads.
    has_setup_node = "actions/setup-node" in gha_text
    has_playwright_install = "npx playwright install --with-deps chromium" in gha_text
    has_update_conditional = "update_snapshots == 'true'" in gha_text
    has_artifact_upload = "actions/upload-artifact" in gha_text
    results.append((
        "PASS" if all([has_setup_node, has_playwright_install, has_update_conditional, has_artifact_upload]) else "FAIL",
        "github_actions_workflow_canonical_shape",
        f"workflow must use setup-node + playwright install --with-deps + update_snapshots conditional + artifact upload (missing: {[k for k, v in {'setup-node': has_setup_node, 'playwright_install': has_playwright_install, 'update_conditional': has_update_conditional, 'artifact_upload': has_artifact_upload}.items() if not v]})",
    ))

    # ---- Claim 5: workflow runs on ubuntu-latest.
    runs_on = "ubuntu-latest" in gha_text
    results.append((
        "PASS" if runs_on else "FAIL",
        "workflow_runs_on_ubuntu_latest",
        "workflow must run on ubuntu-latest (Linux baselines required)",
    ))

    # ---- Claim 6: helpers file exports disableAnimations + dynamicMasks.
    helpers_exists = HELPERS.exists()
    helpers_text = HELPERS.read_text(encoding="utf-8") if helpers_exists else ""
    has_disable = "export async function disableAnimations" in helpers_text
    has_masks = "export function dynamicMasks" in helpers_text
    results.append((
        "PASS" if helpers_exists and has_disable and has_masks else "FAIL",
        "visual_helpers_export_disable_animations_and_dynamic_masks",
        "helpers/visual.ts must export disableAnimations and dynamicMasks",
    ))

    # ---- Claim 7: at least 7 page spec files exist and reference toHaveScreenshot.
    spec_files = list(SPECS_DIR.glob("*.spec.ts"))
    page_specs_ok = 0
    for sp in spec_files:
        txt = sp.read_text(encoding="utf-8")
        if "toHaveScreenshot" in txt and "page.goto" in txt:
            page_specs_ok += 1
    results.append((
        "PASS" if page_specs_ok >= 7 else "FAIL",
        "at_least_seven_page_spec_files_with_toHaveScreenshot",
        f"need >=7 *.spec.ts with toHaveScreenshot + goto (found {page_specs_ok})",
    ))

    # ---- Claim 8: snapshots dir populated with per-page subdirs containing .gitkeep.
    snap_dirs_with_keep = 0
    missing_pages = []
    for page in EXPECTED_PAGES:
        d = SNAPS / f"{page}.spec.ts"
        keep = d / ".gitkeep"
        if d.exists() and d.is_dir() and keep.exists():
            snap_dirs_with_keep += 1
        else:
            missing_pages.append(page)
    results.append((
        "PASS" if snap_dirs_with_keep == len(EXPECTED_PAGES) else "FAIL",
        "screenshots_dir_populated_with_per_page_baselines",
        f"each of {len(EXPECTED_PAGES)} pages must have a snapshots/chromium/<page>.spec.ts/.gitkeep ({missing_pages} missing)",
    ))

    # ---- Claim 9: @playwright/test in devDependencies.
    pkg = json.loads(PKG_JSON.read_text(encoding="utf-8"))
    pw_dep = (pkg.get("devDependencies") or {}).get("@playwright/test")
    results.append((
        "PASS" if pw_dep else "FAIL",
        "playwright_test_in_devdependencies",
        f"frontend/package.json devDependencies must include @playwright/test (found {pw_dep!r})",
    ))

    # ---- Claim 10: README documents first-run flow.
    readme_exists = README.exists()
    readme_text = README.read_text(encoding="utf-8") if readme_exists else ""
    has_flow = "first-run" in readme_text.lower() and "update_snapshots" in readme_text.lower()
    results.append((
        "PASS" if readme_exists and has_flow else "FAIL",
        "readme_documents_first_run_flow",
        "frontend/tests/visual-regression/README.md must document the first-run flow with update_snapshots",
    ))

    # ---- Claim 11: dynamicMasks covers timestamps + animations + recharts ticks.
    masks_lc = helpers_text.lower()
    has_time = '"time"' in helpers_text
    has_animate = "animate" in masks_lc
    has_recharts = "recharts" in masks_lc
    results.append((
        "PASS" if has_time and has_animate and has_recharts else "FAIL",
        "dynamic_masks_cover_time_animate_recharts",
        f"dynamicMasks must reference time + animate + recharts (time={has_time}, animate={has_animate}, recharts={has_recharts})",
    ))

    # ---- Claim 12: workflow uses if-condition gate on update-snapshots step.
    if_condition = re.search(
        r"if:\s*\$\{\{\s*github\.event\.inputs\.update_snapshots\s*==\s*'true'\s*\}\}",
        gha_text,
    )
    results.append((
        "PASS" if if_condition else "FAIL",
        "workflow_update_snapshots_step_gated_by_if",
        "workflow must gate the update step with if: ${{ github.event.inputs.update_snapshots == 'true' }}",
    ))

    # ---- Print results.
    n_pass = sum(1 for r in results if r[0] == "PASS")
    n_fail = len(results) - n_pass
    for verdict, claim, detail in results:
        print(f"{verdict}: {claim}")
        if verdict == "FAIL":
            print(f"      {detail}")

    print(f"\n{n_pass}/{len(results)} claims PASS, {n_fail} FAIL")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
