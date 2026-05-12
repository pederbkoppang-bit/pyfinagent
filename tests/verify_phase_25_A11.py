"""phase-25.A11 verifier -- wire /paper-trading/learnings backend endpoint.

Closes phase-24.11 audit F-1 (orphan UI page wired to live backend).

Run: source .venv/bin/activate && python3 tests/verify_phase_25_A11.py
"""
from __future__ import annotations

import importlib.util
import inspect
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAPER_TRADING_API = REPO / "backend" / "api" / "paper_trading.py"
BQ_CLIENT = REPO / "backend" / "db" / "bigquery_client.py"
API_CACHE = REPO / "backend" / "services" / "api_cache.py"
TYPES_TS = REPO / "frontend" / "src" / "lib" / "types.ts"
API_TS = REPO / "frontend" / "src" / "lib" / "api.ts"
VFL = REPO / "frontend" / "src" / "components" / "VirtualFundLearnings.tsx"
LEARNINGS_PAGE = REPO / "frontend" / "src" / "app" / "paper-trading" / "learnings" / "page.tsx"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    results: list[tuple[str, str, str]] = []

    for p in (PAPER_TRADING_API, BQ_CLIENT, API_CACHE, TYPES_TS, API_TS, VFL, LEARNINGS_PAGE):
        if not p.exists():
            print(f"FAIL: required file missing: {p}")
            return 1

    pt_text = PAPER_TRADING_API.read_text(encoding="utf-8")
    bq_text = BQ_CLIENT.read_text(encoding="utf-8")
    cache_text = API_CACHE.read_text(encoding="utf-8")
    types_text = TYPES_TS.read_text(encoding="utf-8")
    api_text = API_TS.read_text(encoding="utf-8")
    vfl_text = VFL.read_text(encoding="utf-8")
    page_text = LEARNINGS_PAGE.read_text(encoding="utf-8")

    # ---- Claim 1: GET /learnings route registered.
    route_decl = re.search(r'@router\.get\(["\']/learnings["\']\)', pt_text)
    has_get_learnings_def = re.search(r'async\s+def\s+get_learnings\s*\(', pt_text)
    results.append((
        "PASS" if route_decl and has_get_learnings_def else "FAIL",
        "new_get_api_paper_trading_learnings_endpoint_registered",
        "paper_trading.py must register @router.get(\"/learnings\") with async def get_learnings",
    ))

    # ---- Claim 2: route signature window_days: int = Query(30, ge=1, le=365).
    sig_match = re.search(
        r"def\s+get_learnings\s*\(\s*window_days:\s*int\s*=\s*Query\(\s*30\s*,\s*ge=1\s*,\s*le=365\s*\)\s*\)",
        pt_text,
    )
    results.append((
        "PASS" if sig_match else "FAIL",
        "get_learnings_signature_window_days_query_30_ge1_le365",
        "get_learnings must accept window_days: int = Query(30, ge=1, le=365)",
    ))

    # ---- Claim 3: response shape -- three required arrays + window_days + collected_at.
    has_compute = re.search(r'def\s+_compute_learnings\s*\(', pt_text)
    required_keys = [
        '"reconciliation_divergences"',
        '"kill_switch_triggers"',
        '"regime_buckets"',
        '"window_days"',
        '"collected_at"',
    ]
    all_keys_present = all(k in pt_text for k in required_keys)
    results.append((
        "PASS" if has_compute and all_keys_present else "FAIL",
        "compute_learnings_returns_required_keys",
        "_compute_learnings must return reconciliation_divergences/kill_switch_triggers/regime_buckets/window_days/collected_at",
    ))

    # ---- Claim 4: BQ helper get_paper_trades_in_window exists with timeout=30.
    bq_helper_match = re.search(
        r'def\s+get_paper_trades_in_window\s*\(\s*self\s*,\s*window_days:\s*int\s*\)\s*->\s*list\[dict\]:',
        bq_text,
    )
    timeout_match = re.search(
        r'get_paper_trades_in_window.*?result\(timeout=30\)',
        bq_text,
        re.DOTALL,
    )
    results.append((
        "PASS" if bq_helper_match and timeout_match else "FAIL",
        "bq_helper_get_paper_trades_in_window_with_timeout_30",
        "bigquery_client.py must define get_paper_trades_in_window(window_days) calling result(timeout=30)",
    ))

    # ---- Claim 5: TypeScript VirtualFundLearningsData declared in types.ts.
    has_vfld = re.search(
        r'export\s+interface\s+VirtualFundLearningsData\b',
        types_text,
    )
    sub_interfaces = [
        r'export\s+interface\s+ReconciliationDivergence\b',
        r'export\s+interface\s+KillSwitchTrigger\b',
        r'export\s+interface\s+RegimeBucket\b',
    ]
    sub_all = all(re.search(p, types_text) for p in sub_interfaces)
    results.append((
        "PASS" if has_vfld and sub_all else "FAIL",
        "virtualfundlearningsdata_type_in_types_ts",
        "frontend/src/lib/types.ts must export VirtualFundLearningsData + 3 sub-interfaces",
    ))

    # ---- Claim 6: component imports types from @/lib/types (no local define).
    vfl_imports_types = re.search(
        r'import\s+type\s+\{\s*VirtualFundLearningsData\s*\}\s+from\s+["\']@/lib/types["\']',
        vfl_text,
    )
    vfl_local_define = re.search(
        r'export\s+interface\s+VirtualFundLearningsData\b',
        vfl_text,
    )
    results.append((
        "PASS" if vfl_imports_types and not vfl_local_define else "FAIL",
        "vfl_component_imports_type_from_lib_types_no_local_define",
        "VirtualFundLearnings.tsx must import VirtualFundLearningsData from @/lib/types and not redeclare it",
    ))

    # ---- Claim 7: learnings/page.tsx calls getPaperLearnings + passes data/loading/error.
    page_calls_getter = "getPaperLearnings" in page_text
    page_passes_props = re.search(
        r'<VirtualFundLearnings[^>]*\bdata=\{[^}]+\}[^>]*\bloading=\{[^}]+\}[^>]*\berror=\{[^}]+\}',
        page_text,
        re.DOTALL,
    )
    results.append((
        "PASS" if page_calls_getter and page_passes_props else "FAIL",
        "frontend_learnings_page_renders_non_empty_states",
        "paper-trading/learnings/page.tsx must call getPaperLearnings and pass data/loading/error to <VirtualFundLearnings />",
    ))

    # ---- Claim 8: api.ts exports getPaperLearnings(windowDays = 30).
    api_export = re.search(
        r'export\s+function\s+getPaperLearnings\s*\(\s*windowDays\s*=\s*30\s*,?\s*\)\s*:\s*Promise<',
        api_text,
        re.DOTALL,
    )
    results.append((
        "PASS" if api_export else "FAIL",
        "api_ts_exports_getPaperLearnings_default_30",
        "frontend/src/lib/api.ts must export getPaperLearnings(windowDays = 30)",
    ))

    # ---- Claim 9: ENDPOINT_TTLS has paper:learnings.
    ttl_present = re.search(r'["\']paper:learnings["\']\s*:\s*[0-9.]+', cache_text)
    results.append((
        "PASS" if ttl_present else "FAIL",
        "api_cache_endpoint_ttls_has_paper_learnings",
        "api_cache.py ENDPOINT_TTLS must include 'paper:learnings'",
    ))

    # ---- Claim 10: behavioral round-trip -- _compute_learnings returns
    # kill_switch_triggers=[] when audit JSONL is absent. Import the module
    # and call with a mock BigQueryClient where get_paper_trades_in_window
    # returns []. Temporarily patch the module-level _KILL_SWITCH_AUDIT_PATH.
    behavior_ok = False
    behavior_err = ""
    try:
        sys.path.insert(0, str(REPO))
        from backend.api import paper_trading as pt_mod  # type: ignore

        class _MockBQ:
            def get_paper_trades_in_window(self, window_days: int):
                return []

        original_path = pt_mod._KILL_SWITCH_AUDIT_PATH
        try:
            pt_mod._KILL_SWITCH_AUDIT_PATH = REPO / "tmp" / "definitely_missing_audit.jsonl"
            result = pt_mod._compute_learnings(_MockBQ(), 30)
        finally:
            pt_mod._KILL_SWITCH_AUDIT_PATH = original_path

        if not isinstance(result, dict):
            behavior_err = f"result not dict: {type(result)}"
        elif result.get("kill_switch_triggers") != []:
            behavior_err = f"kill_switch_triggers not empty: {result.get('kill_switch_triggers')}"
        elif result.get("reconciliation_divergences") != []:
            behavior_err = f"reconciliation_divergences not empty: {result.get('reconciliation_divergences')}"
        elif result.get("regime_buckets") != []:
            behavior_err = f"regime_buckets not empty: {result.get('regime_buckets')}"
        elif result.get("window_days") != 30:
            behavior_err = f"window_days not 30: {result.get('window_days')}"
        elif not result.get("collected_at"):
            behavior_err = "collected_at missing"
        else:
            behavior_ok = True
    except Exception as e:
        behavior_err = f"exception: {type(e).__name__}: {e}"

    results.append((
        "PASS" if behavior_ok else "FAIL",
        "compute_learnings_handles_missing_audit_jsonl_gracefully",
        f"_compute_learnings must return empty arrays + window_days/collected_at when JSONL missing ({behavior_err})",
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
