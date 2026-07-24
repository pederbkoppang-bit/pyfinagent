"""phase-75.16: Cloud Functions + Docker deploy-surface retirement/hardening
+ script bootstrap repair.

Deliberately goes BEYOND the step's immutable verification command (a
text-`assert` chain, `.claude/masterplan.json` step 75.16), which has two
documented escape hatches (research_brief_75.16.md "Mutation matrix"):

  1. The traceback-leak check only greps LINES containing 'yield' or
     'error_message' for the literal token 'format_exc'. Renaming the
     variable (`error_message` -> `err_msg`) still streams the full
     traceback to unauthenticated HTTP callers while the substring check
     passes. test_quant_no_traceback_variable_reaches_yield() below does
     real AST data-flow tracking instead of a line-text grep, so a rename
     cannot hide the leak.
  2. The requirements `==` check is `all('==' in ln for ... if ln.strip()
     and not ln.strip().startswith('#'))` -- a line like
     `pandas  # bump to == later` contains '==' inside a trailing comment
     and passes while the dependency stays unpinned; an emptied-out file
     also passes vacuously (empty iterable -> `all()` is True).
     test_functions_requirements_all_real_pins() strips comments FIRST
     and asserts a non-empty parsed set, closing both holes.

`functions/{quant,earnings}/main.py` cannot be imported in this test
environment on purpose (functions_framework and flask are Cloud-Function
-only deps, not installed here -- verified: both raise ModuleNotFoundError
in this venv) -- every check on those two files is source-text/AST based.
`functions/ingestion/response.py` (new, phase-75.16 leg c) is a genuinely
pure helper (stdlib-only) and IS imported directly.
"""
from __future__ import annotations

import ast
import glob
import importlib.util
import re
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

DEPLOY_AGENTS_SH = REPO_ROOT / "scripts" / "deploy" / "deploy_agents.sh"
INGESTION_DIR = REPO_ROOT / "functions" / "ingestion"
INGESTION_CLOUDBUILD = INGESTION_DIR / "cloudbuild.yaml"
INGESTION_RETIRED_NOTE = INGESTION_DIR / "RETIRED.md"
INGESTION_MAIN = INGESTION_DIR / "main.py"
INGESTION_RESPONSE = INGESTION_DIR / "response.py"
INGESTION_DATA_FETCHERS = INGESTION_DIR / "utils" / "data_fetchers.py"
QUANT_MAIN = REPO_ROOT / "functions" / "quant" / "main.py"
EARNINGS_MAIN = REPO_ROOT / "functions" / "earnings" / "main.py"
BACKEND_DOCKERFILE = REPO_ROOT / "backend" / "Dockerfile"
FRONTEND_DOCKERFILE = REPO_ROOT / "frontend" / "Dockerfile"
PIP_AUDIT_YML = REPO_ROOT / ".github" / "workflows" / "pip-audit.yml"

MIGRATIONS_DIR = REPO_ROOT / "scripts" / "migrations"
FIVE_MIGRATIONS = [
    "migrate_bq_schema.py",
    "migrate_agent_memories.py",
    "migrate_backtest_data.py",
    "migrate_paper_trading.py",
    "migrate_signals_log.py",
]
EXTEND_HISTORICAL = MIGRATIONS_DIR / "extend_historical_data.py"
DEBUG_DIR = REPO_ROOT / "scripts" / "debug"
DELETED_DEBUG_SCRIPTS = [
    "debug_ingestion.py",
    "debug_db_update.py",
    "debug_processor.py",
    "debug_sequential_updates.py",
]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ─────────────────────────────────────────────────────────────────────
# Leg (a) -- scripts/deploy/deploy_agents.sh deleted
# ─────────────────────────────────────────────────────────────────────

def test_deploy_agents_sh_deleted():
    assert not DEPLOY_AGENTS_SH.exists(), (
        "scripts/deploy/deploy_agents.sh must be deleted -- an unguarded "
        "`cd` failure would gcloud-deploy the repo root (incl. backend/.env) "
        "as public --allow-unauthenticated functions"
    )


# ─────────────────────────────────────────────────────────────────────
# Leg (b) -- functions/ingestion/cloudbuild.yaml deleted with a retirement note
# ─────────────────────────────────────────────────────────────────────

def test_ingestion_cloudbuild_deleted_with_retirement_note():
    assert not INGESTION_CLOUDBUILD.exists(), (
        "functions/ingestion/cloudbuild.yaml must be deleted (orphaned, "
        "never-deployed entry-point mismatch -- see research_brief_75.16.md)"
    )
    assert INGESTION_RETIRED_NOTE.exists(), (
        "a retirement note must exist so a future reader knows the deletion "
        "was deliberate, not accidental"
    )
    note = _read(INGESTION_RETIRED_NOTE)
    # Mutation guard: an empty/near-empty file would satisfy a bare
    # exists() check vacuously.
    assert len(note.strip()) > 200, "retirement note is too thin to carry real evidence"
    for keyword in ("entry-point", "ingest_market_data_el", "fe5acdea"):
        assert keyword in note, f"retirement note missing expected evidence keyword {keyword!r}"


# ─────────────────────────────────────────────────────────────────────
# Leg (c) -- ingestion pure status-decision helper (all 4 outcomes) +
# data_fetchers.py re-raises genuine errors
# ─────────────────────────────────────────────────────────────────────

def test_ingestion_response_module_has_no_thirdparty_imports():
    """The whole point of leg (c) is a helper testable without installing
    functions_framework/google-cloud. Guard the guarantee itself."""
    tree = ast.parse(_read(INGESTION_RESPONSE))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name == "typing", f"unexpected import in response.py: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            assert node.module == "typing", f"unexpected import in response.py: {node.module}"


# Fixture table: (fetch_ok, rows_fetched, load_ok) -> (must_contain, expected_status).
# This is a real assertion against the imported helper's ACTUAL return value,
# not a self-referential `helper(x) == helper(x)` check -- mutating any one
# of the "expected" tuples below to a wrong value makes this test fail
# against the correct (unmutated) helper, which is the guard against a
# vacuous/stub fixture (phase-75 mutation-test doctrine: "a guard that
# can't fail doesn't count").
_INGESTION_OUTCOMES = [
    pytest.param(False, 0, None, "Failure", 500, id="fetch-exception"),
    pytest.param(False, 5, True, "Failure", 500, id="fetch-exception-ignores-rows-and-load-ok"),
    pytest.param(True, 0, None, "No Data", 200, id="genuine-no-data"),
    pytest.param(True, 12, True, "Success", 200, id="load-succeeded"),
    pytest.param(True, 12, False, "Failure", 500, id="load-failed"),
]


@pytest.mark.parametrize("fetch_ok,rows_fetched,load_ok,must_contain,expected_status", _INGESTION_OUTCOMES)
def test_ingestion_response_helper_outcomes(fetch_ok, rows_fetched, load_ok, must_contain, expected_status):
    response_mod = _load_module_from_path("phase75_16_ingestion_response", INGESTION_RESPONSE)
    body, status = response_mod.decide_response(fetch_ok, rows_fetched, load_ok)
    assert status == expected_status, f"decide_response({fetch_ok}, {rows_fetched}, {load_ok}) -> status {status}, expected {expected_status}"
    assert must_contain in body, f"decide_response({fetch_ok}, {rows_fetched}, {load_ok}) body {body!r} missing {must_contain!r}"


def test_ingestion_main_routes_through_pure_helper():
    src = _read(INGESTION_MAIN)
    assert "from response import decide_response" in src, "main.py must delegate to the pure helper, not inline the status logic"
    assert "decide_response(" in src


def test_ingestion_data_fetchers_reraises_genuine_errors():
    """The old code's blanket `except Exception: return pd.DataFrame()` made
    a genuine fetch failure indistinguishable from a real no-data range.
    Assert the exception handler for fetch_raw_market_data's top-level try
    re-raises instead of swallowing into an empty DataFrame."""
    tree = ast.parse(_read(INGESTION_DATA_FETCHERS))
    func = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "fetch_raw_market_data"),
        None,
    )
    assert func is not None, "fetch_raw_market_data not found"

    try_nodes = [n for n in ast.walk(func) if isinstance(n, ast.Try)]
    assert try_nodes, "expected a try/except wrapping the fetch body"

    found_reraise = False
    for try_node in try_nodes:
        for handler in try_node.handlers:
            body_stmts = handler.body
            has_bare_or_named_reraise = any(
                isinstance(s, ast.Raise) for s in body_stmts
            )
            swallows_to_empty_df = any(
                isinstance(s, ast.Return)
                and isinstance(s.value, ast.Call)
                and isinstance(s.value.func, ast.Attribute)
                and s.value.func.attr == "DataFrame"
                for s in body_stmts
            )
            if has_bare_or_named_reraise:
                found_reraise = True
            assert not swallows_to_empty_df, (
                "an except handler still returns pd.DataFrame() -- this "
                "swallows the exception and main.py can never observe a "
                "genuine fetch failure"
            )
    assert found_reraise, "no except handler re-raises -- genuine fetch errors are still swallowed somewhere"


# ─────────────────────────────────────────────────────────────────────
# Leg (d) -- quant/main.py: real timeouts + traceback confined to logging
# (AST-based; closes the variable-rename escape hatch the immutable
# assert's line-text grep cannot see)
# ─────────────────────────────────────────────────────────────────────

def _parent_map(tree: ast.AST) -> dict:
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _enclosing_stmt(node: ast.AST, parents: dict):
    n = node
    while n is not None and not isinstance(n, ast.stmt):
        n = parents.get(n)
    return n


def _is_logging_call_stmt(stmt) -> bool:
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        f = stmt.value.func
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
            return f.value.id in ("logging", "logger")
    return False


def test_quant_sec_requests_have_real_timeout_kwarg():
    """AST-based (not `q.count('timeout=')>=2`, which a comment can inflate):
    every requests.get(...) call must carry a real `timeout=` keyword arg."""
    tree = ast.parse(_read(QUANT_MAIN))
    get_calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "get"
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "requests"
    ]
    assert len(get_calls) == 2, f"expected exactly 2 requests.get(...) calls in quant main.py, found {len(get_calls)}"
    for call in get_calls:
        has_timeout = any(kw.arg == "timeout" for kw in call.keywords)
        assert has_timeout, f"requests.get(...) at line {call.lineno} is missing a timeout= kwarg"


def test_quant_no_traceback_variable_reaches_yield():
    """Data-flow guard, not a line-text grep. Kills the documented escape
    hatch: renaming `error_message` to anything else (e.g. `err_msg`) while
    still assigning it `f"...{traceback.format_exc()}"` and yielding it."""
    tree = ast.parse(_read(QUANT_MAIN))
    parents = _parent_map(tree)

    format_exc_calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and (
            (isinstance(n.func, ast.Attribute) and n.func.attr == "format_exc")
            or (isinstance(n.func, ast.Name) and n.func.id == "format_exc")
        )
    ]
    assert format_exc_calls, "expected traceback.format_exc() to still be called (for Cloud Logging)"

    tainted_names: set[str] = set()
    for call in format_exc_calls:
        stmt = _enclosing_stmt(call, parents)
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    tainted_names.add(target.id)
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            tainted_names.add(stmt.target.id)
        elif _is_logging_call_stmt(stmt):
            continue  # format_exc() feeding directly into logging.critical(...) is fine
        else:
            raise AssertionError(
                f"traceback.format_exc() used outside an assignment or a "
                f"logging call at line {getattr(stmt, 'lineno', '?')}"
            )

    for node in ast.walk(tree):
        if isinstance(node, ast.Yield) and node.value is not None:
            names_in_yield = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
            leaked = names_in_yield & tainted_names
            assert not leaked, (
                f"traceback-derived variable(s) {leaked} are reachable from a "
                f"yield statement at line {node.lineno} -- this leaks the "
                f"full traceback to unauthenticated HTTP callers regardless "
                f"of what the variable is named"
            )


def test_quant_error_yield_is_single_line_with_prefix():
    """The orchestrator (orchestrator.py aiter_lines()) only recognizes a
    single `ERROR:`-prefixed line; an embedded newline would split the
    sanitized message across multiple `Quant:`-logged noise lines."""
    tree = ast.parse(_read(QUANT_MAIN))
    error_yields = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Yield) and isinstance(node.value, ast.JoinedStr):
            const_parts = [v.value for v in node.value.values if isinstance(v, ast.Constant)]
            joined = "".join(str(p) for p in const_parts)
            if joined.startswith("ERROR:"):
                error_yields.append((node, joined))
    assert error_yields, "no yield of an f-string starting with 'ERROR:' found"
    for node, joined in error_yields:
        assert "\n" not in joined, f"ERROR: yield at line {node.lineno} contains an embedded newline in its literal parts"


# ─────────────────────────────────────────────────────────────────────
# Leg (e) -- earnings/main.py: env model id, distinguishable NLP failure,
# 4-key validation, no wildcard CORS
# ─────────────────────────────────────────────────────────────────────

def _extract_string_assign(tree: ast.AST, var_name: str) -> str | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == var_name:
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        return node.value.value
    return None


def test_earnings_model_id_is_env_configurable():
    src = _read(EARNINGS_MAIN)
    assert 'os.getenv("EARNINGS_NLP_MODEL"' in src or "os.getenv('EARNINGS_NLP_MODEL'" in src, (
        "model id must come from an environment variable, not a hardcoded literal"
    )
    assert 'GenerativeModel("gemini-1.5-flash-001")' not in src
    assert "GenerativeModel(model_id)" in src or "GenerativeModel(model)" in src


def test_earnings_model_default_matches_backend_workhorse_and_is_not_retired():
    """Cross-file consistency: this Cloud Function can't import the backend
    package (isolated deploy source root), so its default is a duplicated
    literal -- pin it against backend/config/model_tiers.GEMINI_WORKHORSE
    (itself stdlib-only, safe to import) so drift is caught."""
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from backend.config.model_tiers import GEMINI_WORKHORSE
    finally:
        sys.path.pop(0)

    tree = ast.parse(_read(EARNINGS_MAIN))
    default = _extract_string_assign(tree, "EARNINGS_NLP_MODEL_DEFAULT")
    assert default is not None, "EARNINGS_NLP_MODEL_DEFAULT literal not found"
    assert not default.startswith("gemini-1."), f"default model {default!r} looks like a retired gemini-1.x pin"
    assert default == GEMINI_WORKHORSE, (
        f"functions/earnings/main.py default {default!r} has drifted from "
        f"backend/config/model_tiers.GEMINI_WORKHORSE {GEMINI_WORKHORSE!r}"
    )


def test_earnings_nlp_failure_is_distinguishable_not_error_as_data():
    src = _read(EARNINGS_MAIN)
    # The old pattern stored the failure ITSELF as the analysis payload.
    assert re.search(r"nlp_analysis'?\]\s*=\s*\{[\"']error", src) is None, (
        "nlp_analysis must never be set to an {'error': ...} dict -- that "
        "makes failure indistinguishable from real analysis data"
    )
    assert "nlp_status" in src, "expected an explicit nlp_status field to distinguish success/failure"
    assert re.search(r"nlp_status'?\]\s*=\s*['\"]ok['\"]", src), "no explicit ok status assignment found"
    assert re.search(r"nlp_status'?\]\s*=\s*['\"]failed['\"]", src), "no explicit failed status assignment found"
    assert re.search(r"nlp_analysis'?\]\s*=\s*None", src), "expected nlp_analysis to be None (not error-as-data) on failure"


def test_earnings_validates_all_4_required_nlp_keys():
    tree = ast.parse(_read(EARNINGS_MAIN))
    required_keys_set = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "REQUIRED_NLP_KEYS":
                    if isinstance(node.value, ast.Set):
                        required_keys_set = {
                            elt.value for elt in node.value.elts
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                        }
    assert required_keys_set == {
        "forward_sentiment_score",
        "qa_confidence_summary",
        "cyclical_catalysts_detected",
        "key_quotes",
    }, f"REQUIRED_NLP_KEYS mismatch: {required_keys_set}"

    src = _read(EARNINGS_MAIN)
    assert "missing_keys" in src and "REQUIRED_NLP_KEYS" in src, "expected a missing-keys check derived from REQUIRED_NLP_KEYS"
    assert re.search(r"if\s+missing_keys\s*:\s*\n\s*raise", src), "missing_keys must actually gate a raise (validation must be load-bearing, not dead code)"


def test_earnings_cors_never_wildcard_and_matches_tailscale_localhost_idiom():
    src = _read(EARNINGS_MAIN)
    assert "Access-Control-Allow-Origin': '*'" not in src
    assert 'Access-Control-Allow-Origin": "*"' not in src
    assert re.search(r"Access-Control-Allow-Origin[\"']?\s*:\s*[\"']\*[\"']", src) is None, "a wildcard CORS origin literal remains"

    tree = ast.parse(src)
    pattern = _extract_string_assign(tree, "_ALLOWED_ORIGIN_RE")
    # _ALLOWED_ORIGIN_RE is assigned via re.compile(r"..."), which AST sees
    # as a Call, not a plain Constant -- pull the raw string from the Call arg.
    if pattern is None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "_ALLOWED_ORIGIN_RE":
                        call = node.value
                        if isinstance(call, ast.Call) and call.args and isinstance(call.args[0], ast.Constant):
                            pattern = call.args[0].value
    assert pattern is not None, "_ALLOWED_ORIGIN_RE pattern not found"

    compiled = re.compile(pattern)
    # Behavioral parity with backend/main.py::_TAILSCALE_ORIGIN_RE.
    assert compiled.match("http://localhost:3000")
    assert compiled.match("http://100.64.1.2:5173")
    assert compiled.match("http://100.127.255.255:8080")
    assert not compiled.match("http://100.1.2.3:80"), "matched a non-Tailscale 100.x range (publicly routable)"
    assert not compiled.match("http://100.128.0.1:80"), "matched a non-Tailscale 100.x range (publicly routable)"
    assert not compiled.match("https://evil.example.com")
    assert not compiled.match("http://localhost"), "must require an explicit port"


# ─────────────────────────────────────────────────────────────────────
# Leg (f) -- functions/*/requirements.txt fully ==-pinned + pip-audit.yml
# covers them (parsed-line, comments stripped first, non-empty guard --
# closes BOTH documented escape hatches: ==-in-a-comment and an empty file)
# ─────────────────────────────────────────────────────────────────────

_REQ_LINE_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*([0-9A-Za-z.\-+_]+)\s*$")


def _parse_pinned_requirements(text: str) -> dict[str, str]:
    """Strip comments FIRST, then require a real `name==version` line.
    A `pkg  # == 1.0` line has its comment stripped to `pkg`, which does
    NOT match _REQ_LINE_RE -- so the ==-in-comment escape hatch fails this
    parser (unlike the immutable command's raw substring check)."""
    pins: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = _REQ_LINE_RE.match(line)
        assert m, f"non-empty, non-comment line does not parse as a pinned requirement: {raw!r}"
        pins[m.group(1).lower()] = m.group(2)
    return pins


FUNCTIONS_REQUIREMENTS = {
    "ingestion": REPO_ROOT / "functions" / "ingestion" / "requirements.txt",
    "quant": REPO_ROOT / "functions" / "quant" / "requirements.txt",
    "earnings": REPO_ROOT / "functions" / "earnings" / "requirements.txt",
}


@pytest.mark.parametrize("fn_name,path", list(FUNCTIONS_REQUIREMENTS.items()))
def test_functions_requirements_all_real_pins(fn_name, path):
    pins = _parse_pinned_requirements(_read(path))
    # Non-empty guard: closes the vacuous-pass-on-empty-file hole (all()
    # over an empty iterable is True).
    assert pins, f"functions/{fn_name}/requirements.txt parsed to zero real pins -- an emptied file must not pass"


def test_functions_requirements_glob_matches_the_three_known_files():
    found = {Path(p).parent.name for p in glob.glob(str(REPO_ROOT / "functions" / "*" / "requirements.txt"))}
    assert found == set(FUNCTIONS_REQUIREMENTS.keys()), f"unexpected functions/*/requirements.txt set: {found}"


def test_earnings_requirements_still_missing_vertexai_deps_by_design():
    """Documents the deliberate boundary: 75.16.1 owns adding vertexai /
    google-cloud-storage to this file. If a future edit adds them here
    without updating/closing 75.16.1, this test should be revisited (not
    silently left describing a stale state)."""
    pins = _parse_pinned_requirements(_read(FUNCTIONS_REQUIREMENTS["earnings"]))
    assert "vertexai" not in pins
    assert "google-cloud-storage" not in pins


def test_pip_audit_yml_is_valid_yaml_and_covers_functions_requirements():
    data = yaml.safe_load(_read(PIP_AUDIT_YML))
    assert data, "pip-audit.yml failed to parse"

    # PyYAML's default (YAML 1.1) resolver treats a bare `on:` key as the
    # boolean True, not the string "on" -- GitHub Actions workflows always
    # trigger this. Accept either form.
    on_section = data.get("on", data.get(True))
    assert on_section is not None, "workflow has no 'on' trigger section"
    push_paths = on_section["push"]["paths"]
    pr_paths = on_section["pull_request"]["paths"]
    for fn_name, path in FUNCTIONS_REQUIREMENTS.items():
        rel = str(path.relative_to(REPO_ROOT))
        assert rel in push_paths, f"{rel} missing from push paths"
        assert rel in pr_paths, f"{rel} missing from pull_request paths"

    steps = data["jobs"]["audit"]["steps"]
    run_blobs = [s.get("run", "") for s in steps if "run" in s]
    for fn_name, path in FUNCTIONS_REQUIREMENTS.items():
        rel = str(path.relative_to(REPO_ROOT))
        assert any(rel in run for run in run_blobs), f"no pip-audit step invokes --requirement {rel}"


# ─────────────────────────────────────────────────────────────────────
# Leg (g) -- Dockerfiles
# ─────────────────────────────────────────────────────────────────────

def test_backend_dockerfile_copies_real_requirements_and_pins_3_14():
    b = _read(BACKEND_DOCKERFILE)
    assert re.search(r"^FROM python:3\.14-slim", b, re.MULTILINE), "backend/Dockerfile must pin python:3.14-slim"
    assert re.search(r"^COPY backend/requirements\.txt", b, re.MULTILINE), (
        "backend/Dockerfile must COPY the real backend/requirements.txt path "
        "(build context is the repo root; the old bare 'requirements.txt' "
        "pointer does not exist at the repo root)"
    )
    # Real-instruction check (not a substring match, which would false-positive
    # on this file's own explanatory comment prose mentioning the old form).
    copy_instr_lines = [ln for ln in b.splitlines() if ln.strip().startswith("COPY ")]
    assert not any(ln.strip() == "COPY requirements.txt ." for ln in copy_instr_lines), (
        "a stray bare 'COPY requirements.txt .' instruction (pointing at a "
        "nonexistent repo-root file) must not remain"
    )


def test_frontend_dockerfile_uses_npm_ci_with_lockfile():
    fr = _read(FRONTEND_DOCKERFILE)
    assert re.search(r"^COPY package\.json package-lock\.json", fr, re.MULTILINE), (
        "deps stage must COPY the committed package-lock.json alongside package.json"
    )
    assert re.search(r"^RUN npm ci\b", fr, re.MULTILINE), "deps stage must RUN npm ci (not npm install)"
    deps_stage = fr.split("FROM node:20-alpine AS builder")[0]
    # Real-instruction check (not a bare substring match, which would
    # false-positive on this file's own explanatory comment prose that
    # mentions "npm install" while describing what was wrong before).
    deps_instr_lines = [ln for ln in deps_stage.splitlines() if ln.strip().startswith("RUN ")]
    assert not any("npm install" in ln for ln in deps_instr_lines), "deps stage must not fall back to a RUN npm install instruction"


# ─────────────────────────────────────────────────────────────────────
# Leg (h) -- migrations + debug scripts
# ─────────────────────────────────────────────────────────────────────

_BROKEN_FORM_RE = re.compile(r"""\.parent\s*/\s*(['"])backend\1""")
_FIXED_FORM_RE = re.compile(r"""resolve\(\)\.parents\[2\]""")


@pytest.mark.parametrize("filename", FIVE_MIGRATIONS)
def test_migration_uses_parents2_anchor_not_broken_form(filename):
    src = _read(MIGRATIONS_DIR / filename)
    assert not _BROKEN_FORM_RE.search(src), f"{filename} still has the CWD-broken '.parent / \"backend\"' bootstrap"
    assert _FIXED_FORM_RE.search(src), f"{filename} missing the parents[2] anchor"
    assert 'load_dotenv(Path(__file__).resolve().parents[2] / "backend" / ".env")' in src


def test_extend_historical_data_uses_parents2_anchor():
    src = _read(EXTEND_HISTORICAL)
    assert not _BROKEN_FORM_RE.search(src), "extend_historical_data.py still has the broken '.parent / \"backend\"' form"
    assert 'load_dotenv("backend/.env")' not in src, "CWD-relative load_dotenv(\"backend/.env\") must be replaced"
    assert "sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))" not in src, (
        "the wrong sys.path insert (this file's own dir, not repo root) must be replaced"
    )
    assert _FIXED_FORM_RE.search(src), "missing a parents[2]-anchored path"
    assert "from pathlib import Path" in src


@pytest.mark.parametrize("filename", FIVE_MIGRATIONS + ["extend_historical_data.py"])
def test_touched_migration_compiles(filename):
    import py_compile
    py_compile.compile(str(MIGRATIONS_DIR / filename), doraise=True)


@pytest.mark.parametrize("filename", DELETED_DEBUG_SCRIPTS)
def test_stale_debug_script_deleted(filename):
    assert not (DEBUG_DIR / filename).exists(), f"scripts/debug/{filename} must be deleted (unreferenced, broken bootstrap)"


def test_deleted_debug_scripts_have_zero_live_references():
    """Guard against deleting something that's actually still wired in --
    grep the WHOLE repo (excluding this test file's own listing and the
    handoff/archive snapshots, which are historical by design) for any
    reference to the deleted filenames."""
    live_dirs = ["backend", "frontend", "scripts", ".claude", "docs"]
    hits = []
    for filename in DELETED_DEBUG_SCRIPTS:
        stem = Path(filename).stem
        for d in live_dirs:
            for py in (REPO_ROOT / d).rglob("*.py"):
                if py == Path(__file__):
                    continue
                if "__pycache__" in py.parts:
                    continue
                text = py.read_text(encoding="utf-8", errors="ignore")
                if stem in text:
                    hits.append((filename, str(py)))
    assert not hits, f"deleted debug script(s) still referenced by live code: {hits}"


def test_combined_migrations_and_debug_glob_matches_immutable_command_scope():
    """Mirror the immutable command's own glob scope so a file added to
    either directory in the future is covered by both the command and this
    suite without silent drift."""
    mig = "".join(
        _read(Path(p)) for p in glob.glob(str(MIGRATIONS_DIR / "*.py")) + glob.glob(str(DEBUG_DIR / "*.py"))
    )
    assert '.parent / "backend"' not in mig
    assert ".parent / 'backend'" not in mig
