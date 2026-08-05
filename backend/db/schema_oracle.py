"""phase-82.12: the BigQuery schema oracle the codebase never had.

WHY THIS EXISTS, and why it is NOT what the masterplan step description asked for.

Step 82.12 asked for a sweep: "enumerate every BQ column whose declared type is STRING
but which is consumed as a date/datetime/number, and every isinstance/comparison guard
applied to a value read from BQ". The research gate ran that sweep and measured the
answer: **ZERO remaining defects of that shape.** 534 non-test `isinstance` calls, 8
date-typed, 5 live, all 5 correct. The 82.0 fix closed the only vacuous instance, and
`cycle_health._STRING_DATE_TIMESTAMP_COLS` is 6/6 correct today.

A step of the form "go find more of the same" when there is no more of the same has two
failure modes -- an honest empty report that reads as a wasted cycle, or padded false
positives to look productive. So the deliverable is re-framed: build the ORACLE, prove
the current surface clean against it BY CONSTRUCTION, and leave a standing check so the
next instance cannot land silently.

THE SWEEP DID FIND A LIVE DEFECT, of the same root cause but a different symptom, and it
is the reason the oracle covers NAMES as well as TYPES: `_production_fns.py`'s
`nightly_outcome_rebuild` source selects `paper_trades.timestamp` and
`paper_trades.realized_pnl`. Neither column exists (measured: the table has 18 columns;
the real ones are `created_at` STRING and `realized_pnl_pct` FLOAT). BigQuery 400s, an
`except Exception` swallows it, and the job has been running on zero rows. A type-only
oracle would have missed it entirely.

DESIGN -- two-sided derivation, because a name regex is itself a hand-written list.
Criterion 1 requires the scope be "derived from live table schemas rather than from a
hand-written list". A regex over column NAMES satisfies that in letter and violates it
in spirit: it is a hand-written list of name tokens, and it demonstrably under-covers
(`calendar_events.window`, `analysis_results.overall_reliability`,
`strategy_decisions.decay_attribution` all carry date/number semantics under
non-obvious names). So:

  schema side   -- every column of every table, from the live API or a snapshot;
  consumer side -- every read site, with NO name filter;
  join          -- flag only where a consumer applies date/number semantics to a column
                   the oracle declares STRING, or names a column that does not exist.

EVERY STAGE ASSERTS NON-EMPTY. This is not ceremony. The researcher's first scanner
reported "0 unknown identifiers", which reads exactly like a clean bill of health, but
was a relative-path bug plus a non-greedy regex capturing `sunny` out of a
fully-qualified table name. A checker that scans nothing and a codebase with no defects
produce the identical output; only an emptiness assertion tells them apart.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional

PROJECT = "sunny-might-477607-p8"
DATASETS = ("financial_reports", "pyfinagent_data", "pyfinagent_pms", "pyfinagent_hdw")

_REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_PATH = Path(__file__).resolve().parent / "_schema_snapshot.json"

# A fully-qualified table reference inside a backtick-quoted SQL identifier.
# GREEDY on the dataset/table parts -- the researcher's non-greedy version captured
# "sunny" out of `sunny-might-477607-p8.financial_reports.paper_trades` and silently
# resolved nothing.
_FQ_TABLE_RE = re.compile(r"`([A-Za-z0-9_\-]+)\.([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)`")

# SQL constructs that impose DATE/TIME semantics on their argument.
_DATE_FN_RE = re.compile(
    r"\b(CURRENT_DATE|CURRENT_TIMESTAMP|CURRENT_DATETIME|DATE_SUB|DATE_ADD|DATE_DIFF|"
    r"DATE_TRUNC|TIMESTAMP_SUB|TIMESTAMP_ADD|TIMESTAMP_DIFF|TIMESTAMP_TRUNC|"
    r"DATETIME_DIFF|PARSE_DATE|PARSE_TIMESTAMP|EXTRACT|SAFE\.TIMESTAMP|SAFE\.DATE)\b",
    re.I,
)
# SQL constructs that impose NUMERIC semantics on their argument.
_NUM_FN_RE = re.compile(
    r"\b(SUM|AVG|STDDEV|VARIANCE|ROUND|ABS|CEIL|FLOOR|SAFE_DIVIDE|"
    r"SAFE_CAST\s*\(\s*\w+\s+AS\s+(?:FLOAT64|INT64|NUMERIC))\b",
    re.I,
)


# ---------------------------------------------------------------------------
# Schema side
# ---------------------------------------------------------------------------

def fetch_live_schema(client: Any = None) -> dict[str, dict[str, dict[str, str]]]:
    """Read every table's schema from the live BigQuery API.

    Returns {"<dataset>.<table>": {"<column>": {"type": ..., "mode": ...}}}.
    Requires ADC. `refresh_snapshot` persists the result so the guards can run
    offline.
    """
    if client is None:  # pragma: no cover - exercised only with live creds
        from google.cloud import bigquery

        client = bigquery.Client(project=PROJECT)

    out: dict[str, dict[str, dict[str, str]]] = {}
    for dataset in DATASETS:
        try:
            tables = list(client.list_tables(f"{PROJECT}.{dataset}"))
        except Exception:
            # A dataset we cannot see must NOT silently shrink the oracle to
            # nothing; the emptiness assertion below is what catches that.
            continue
        for tbl in tables:
            ref = tbl.reference
            try:
                schema = client.get_table(ref).schema
            except Exception:
                continue
            out[f"{dataset}.{ref.table_id}"] = {
                f.name: {"type": f.field_type, "mode": f.mode or ""} for f in schema
            }
    return out


def refresh_snapshot(client: Any = None, path: Path = SNAPSHOT_PATH) -> dict:
    """Re-read the live schema and persist it. Refuses to write an empty oracle."""
    tables = fetch_live_schema(client)
    n_cols = sum(len(c) for c in tables.values())
    if not tables or n_cols == 0:
        raise RuntimeError(
            "refresh_snapshot: live schema came back EMPTY -- refusing to overwrite "
            "the snapshot. An empty oracle makes every consumer look valid."
        )
    payload = {"project": PROJECT, "datasets": list(DATASETS), "tables": tables}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def load_snapshot(path: Path = SNAPSHOT_PATH) -> dict[str, dict[str, dict[str, str]]]:
    """Load the checked-in oracle so guards run without ADC."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["tables"]


def snapshot_drift(client: Any = None, path: Path = SNAPSHOT_PATH) -> dict:
    """Diff snapshot vs live. THIS DIFF IS THE SCHEMA-DRIFT DETECTOR.

    Returns {"missing_tables", "new_tables", "changed_columns"}; empty everywhere
    means the snapshot is current.
    """
    snap = load_snapshot(path)
    live = fetch_live_schema(client)
    changed = []
    for table in sorted(set(snap) & set(live)):
        for col in sorted(set(snap[table]) | set(live[table])):
            a = snap[table].get(col, {}).get("type")
            b = live[table].get(col, {}).get("type")
            if a != b:
                changed.append({"table": table, "column": col, "snapshot": a, "live": b})
    return {
        "missing_tables": sorted(set(snap) - set(live)),
        "new_tables": sorted(set(live) - set(snap)),
        "changed_columns": changed,
    }


def columns_of_type(
    oracle: dict[str, dict[str, dict[str, str]]], field_type: str
) -> list[tuple[str, str]]:
    """Every (table, column) whose DECLARED type is `field_type`. No name filter."""
    return sorted(
        (t, c)
        for t, cols in oracle.items()
        for c, meta in cols.items()
        if meta.get("type") == field_type
    )


# ---------------------------------------------------------------------------
# Consumer side
# ---------------------------------------------------------------------------

def iter_python_files(root: Path | None = None) -> list[Path]:
    root = root or (_REPO_ROOT / "backend")
    return sorted(
        p
        for p in root.rglob("*.py")
        if "__pycache__" not in p.parts and "/tests/" not in p.as_posix()
    )


def extract_sql_literals(path: Path) -> list[tuple[int, str]]:
    """Every string constant in a file that looks like SQL over a known table.

    AST-based, so an f-string fragment or a comment mentioning SELECT is not
    mistaken for a query.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return []
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value
            if _FQ_TABLE_RE.search(text) and re.search(r"\bFROM\b", text, re.I):
                found.append((node.lineno, text))
        elif isinstance(node, ast.JoinedStr):
            # f-string: reassemble the literal parts so a table reference split
            # across an interpolation is still seen.
            text = "".join(
                v.value for v in node.values
                if isinstance(v, ast.Constant) and isinstance(v.value, str)
            )
            if _FQ_TABLE_RE.search(text) and re.search(r"\bFROM\b", text, re.I):
                found.append((node.lineno, text))
    return found


def tables_in_sql(sql: str) -> list[str]:
    """Resolve `project.dataset.table` references to oracle keys."""
    return sorted({f"{m.group(2)}.{m.group(3)}" for m in _FQ_TABLE_RE.finditer(sql)})


def _semantics_for(sql: str, column: str) -> Optional[str]:
    """Does `sql` apply date or number semantics to `column`? None if neither.

    Looks at a window around each occurrence rather than the whole query, so an
    unrelated SUM() elsewhere does not tar every column in the SELECT list.
    """
    verdict = None
    for m in re.finditer(rf"\b{re.escape(column)}\b", sql):
        window = sql[max(0, m.start() - 60) : m.end() + 60]
        if _DATE_FN_RE.search(window):
            return "date"
        if _NUM_FN_RE.search(window):
            verdict = "number"
        # a bare comparison against a date-shaped literal is date semantics
        if re.search(
            rf"\b{re.escape(column)}\b\s*[<>=!]+\s*'?\d{{4}}-\d{{2}}-\d{{2}}", window
        ):
            return "date"
    return verdict


def _row_key_reads(tree: ast.AST) -> list[tuple[int, str, str]]:
    """Every `x["key"]` / `x.get("key")` read, with the context it flows into.

    Returns (lineno, key, context) where context is one of
    "isinstance_date" | "compare" | "arith" | "plain".

    THIS is the side that matters. The 82.0 defect was not in SQL at all -- it was
    `isinstance(rd, datetime.date)` applied to a value read out of a BQ row whose
    column is declared STRING, so the guard could never fire. A SQL-only scanner
    would have reported that codebase clean.
    """
    reads: list[tuple[int, str, str]] = []

    # Bindings are resolved PER FUNCTION, not per module. A module-wide binder
    # leaks a name across unrelated functions: measured, `raw = trade.get("signals")`
    # in one handler made an unrelated `len(raw) > 50` in a different handler look
    # like a numeric comparison on a STRING column. Two false positives from one
    # shared name -- and a sweep that cries wolf is worse than one with a stated
    # recall limit, because every future reader has to re-litigate it.
    scopes: list[ast.AST] = [tree] + [
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    for scope in scopes:
        reads.extend(_reads_in_scope(scope))
    return reads


def _reads_in_scope(scope: ast.AST) -> list[tuple[int, str, str]]:
    """Row-value reads within ONE function (or module) body."""
    reads: list[tuple[int, str, str]] = []
    nested = {
        id(n) for n in ast.walk(scope)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n is not scope
    }

    def walk(node: ast.AST):
        """Walk `scope` without descending into nested function bodies."""
        for child in ast.iter_child_nodes(node):
            if id(child) in nested:
                continue
            yield child
            yield from walk(child)

    def key_of(node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) \
                and isinstance(node.slice.value, str):
            return node.slice.value
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "get" and node.args \
                and isinstance(node.args[0], ast.Constant) \
                and isinstance(node.args[0].value, str):
            return node.args[0].value
        # `row.date` -- a google.cloud.bigquery Row supports attribute access as
        # well as subscripting, so a subscript-only reader misses half the shapes
        # a BQ consumer can legitimately be written in.
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                and node.value.id in _ROW_NAMES:
            return node.attr
        return None

    # Names bound from a row read. The 82.12 cycle-1 Q/A measured this binder's
    # RECALL ENVELOPE by running 8 shapes of the same defect class through it: only
    # `rd = row["date"]` was seen, and 7 were missed. Those misses are now closed
    # here, because a recall envelope that narrow makes "the sweep found nothing"
    # unfalsifiable -- and, worse, made the anti-cry-wolf test pass for the wrong
    # reason (it could not resolve through ANY call, so it could not tell a correct
    # coercing helper from a genuinely vacuous one; it was blind to both).
    bound: dict[str, str] = {}
    call_bound: set[str] = set()
    for node in walk(scope):
        target_names: list[str] = []
        value: Optional[ast.AST] = None
        if isinstance(node, ast.Assign):
            value = node.value
            for t in node.targets:
                if isinstance(t, ast.Name):
                    target_names.append(t.id)
                elif isinstance(t, ast.Tuple):  # a, b = row["date"], row["value"]
                    for i, el in enumerate(t.elts):
                        if isinstance(el, ast.Name) and isinstance(value, ast.Tuple) \
                                and i < len(value.elts):
                            k = key_of(value.elts[i])
                            if k:
                                bound[el.id] = k
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_names.append(node.target.id)  # rd: date = row["date"]
            value = node.value
        elif isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
            target_names.append(node.target.id)  # (rd := row["date"])
            value = node.value
        if value is not None:
            # Resolve the bound VALUE through call arguments too. The cycle-2 Q/A
            # measured that the TWO-STATEMENT bind-through-call form
            # (`rd = parse(row["date"])` then `isinstance(rd, date)`) was invisible --
            # and that is the exact shape all four real production coercers use
            # (cache.py:137/382, reconciliation.py:41, paper_round_trips.py:39,
            # wash_sale_filter.py:30-32). An envelope that misses the shape the
            # codebase actually writes is not an envelope.
            k = key_of(value)
            via_call = False
            if k is None and isinstance(value, ast.Call):
                for arg in value.args:
                    k = key_of(arg)
                    if k:
                        via_call = True
                        break
            if k:
                for name in target_names:
                    bound[name] = k
                    # A call between the row read and the use is AMBIGUOUS, and the
                    # direction matters. For `isinstance` it is the DEFECT shape --
                    # a non-coercing helper leaves a str and the guard is vacuous, so
                    # the site must stay visible. For arithmetic / ordering it is the
                    # FIX shape -- a coercer's whole job is to change the type, so
                    # `d = datetime.fromisoformat(row["date"])` then `d_exit - d` is
                    # correct and flagging it is cry-wolf. MEASURED: treating both the
                    # same took the scope from 5 sites to 18, and the 13 new ones were
                    # all correctly-coerced values (paper_trader `_l2u` FX rate,
                    # analytics `d_entry`, cycle_health `hb_updated`).
                    if via_call:
                        call_bound.add(name)

    def resolve(node: ast.AST) -> Optional[str]:
        k = key_of(node)
        if k:
            return k
        if isinstance(node, ast.Name):
            return bound.get(node.id)
        # A row value reached THROUGH a call -- isinstance(parse(row["date"]), date).
        # Recursing here is what lets the anti-cry-wolf test actually discriminate:
        # a COERCING helper produces a value the guard can legitimately test, a
        # non-coercing one does not, and previously both were invisible.
        if isinstance(node, ast.Call):
            for arg in node.args:
                k = resolve(arg)
                if k:
                    return k
        return None

    for node in walk(scope):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "isinstance" and len(node.args) == 2:
            k = resolve(node.args[0])
            if k:
                tgt = ast.unparse(node.args[1])
                ctx = "isinstance_date" if re.search(r"\b(date|datetime|_dt)\b", tgt) \
                    else "isinstance_other"
                reads.append((node.lineno, k, ctx))
        elif isinstance(node, ast.Compare):
            # ONLY ordering comparisons. `status == "done"` on a STRING column is
            # correct string equality, not date/number semantics -- counting it
            # would bury the real signal under ~150 false positives (measured).
            if not any(isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE))
                       for op in node.ops):
                continue
            for side in [node.left, *node.comparators]:
                k = resolve(side)
                if k and not (isinstance(side, ast.Name) and side.id in call_bound):
                    reads.append((node.lineno, k, "order_compare"))
        elif isinstance(node, ast.BinOp):
            # `+` on a STRING is concatenation and is legitimate; -, *, /, //, %
            # are numeric semantics and would TypeError on a str.
            if isinstance(node.op, ast.Add):
                continue
            for side in (node.left, node.right):
                k = resolve(side)
                if k and not (isinstance(side, ast.Name) and side.id in call_bound):
                    reads.append((node.lineno, k, "arith"))
    return reads


def derive_python_scope(
    oracle: dict[str, dict[str, dict[str, str]]],
    files: Iterable[Path] | None = None,
) -> dict:
    """Python-side half of criterion 1's two-sided join.

    Any row key that matches a column the oracle declares STRING *somewhere*, read in a
    date/number context. Deliberately name-agnostic: the key set comes from the oracle,
    not from a hand-written token list, so `window` / `overall_reliability` /
    `decay_attribution` are covered exactly like `date` is.
    """
    files = list(files) if files is not None else iter_python_files()
    string_cols: dict[str, set[str]] = {}
    for table, cols in oracle.items():
        for col, meta in cols.items():
            if meta.get("type") == "STRING":
                string_cols.setdefault(col, set()).add(table)

    hits: list[dict] = []
    scanned = 0
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        scanned += 1
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for lineno, key, ctx in _row_key_reads(tree):
            if key in string_cols and ctx != "isinstance_other":
                hits.append({
                    "column": key,
                    "tables": sorted(string_cols[key]),
                    "type": "STRING",
                    "context": ctx,
                    "file": rel,
                    "line": lineno,
                })
    return {
        "hits": hits,
        "files_scanned": scanned,
        "string_columns_in_oracle": len(string_cols),
    }


def derive_scope(
    oracle: dict[str, dict[str, dict[str, str]]],
    files: Iterable[Path] | None = None,
) -> dict:
    """CRITERION 1. The two-sided join, with the counters an emptiness check needs.

    Returns {"scope": [...], "unknown_columns": [...], "files_scanned": int,
             "sql_literals": int, "tables_resolved": int, "columns_in_oracle": int}

    `scope`          -- (table, column, type, semantics, file, line) where the oracle
                        declares STRING and a consumer applies date/number semantics.
    `unknown_columns`-- identifiers a query selects that the table does not have. This
                        is the NAME variant of the same root cause, and it is where the
                        live P1 lives.
    """
    files = list(files) if files is not None else iter_python_files()
    scope: list[dict] = []
    unknown: list[dict] = []
    n_sql = 0
    tables_resolved: set[str] = set()

    for path in files:
        for lineno, sql in extract_sql_literals(path):
            n_sql += 1
            for table in tables_in_sql(sql):
                cols = oracle.get(table)
                if cols is None:
                    continue
                tables_resolved.add(table)
                rel = path.relative_to(_REPO_ROOT).as_posix()

                for col, meta in cols.items():
                    if meta.get("type") != "STRING":
                        continue
                    sem = _semantics_for(sql, col)
                    if sem:
                        scope.append({
                            "table": table, "column": col, "type": "STRING",
                            "semantics": sem, "file": rel, "line": lineno,
                        })

                # NAME check: identifiers selected that the table does not have.
                # Strip `AS <alias>` first -- an alias is a name the query INVENTS, so
                # counting it as a missing column is a false positive (measured: `pnl`
                # in the _production_fns query). And do NOT drop an identifier merely
                # because it collides with a BQ type keyword: `timestamp` is exactly
                # such a name, and it is the real defect here, so a blanket keyword
                # filter would hide the very column this check exists to find.
                select_part = re.split(r"\bFROM\b", sql, flags=re.I)[0]
                aliases = {
                    m.group(1).lower()
                    for m in re.finditer(r"\bAS\s+([A-Za-z_][A-Za-z0-9_]*)", select_part, re.I)
                }
                select_part = re.sub(
                    r"\bAS\s+[A-Za-z_][A-Za-z0-9_]*", " ", select_part, flags=re.I
                )
                for ident in set(re.findall(r"\b([a-z_][a-z0-9_]{2,})\b", select_part)):
                    if ident in cols or ident in aliases or ident in _SQL_NOISE:
                        continue
                    if ident.upper() in _SQL_KEYWORDS and ident.upper() not in _TYPE_NAMES:
                        continue
                    unknown.append({
                        "table": table, "identifier": ident,
                        "file": rel, "line": lineno,
                    })

    return {
        "scope": scope,
        "unknown_columns": unknown,
        "files_scanned": len(files),
        "sql_literals": n_sql,
        "tables_resolved": len(tables_resolved),
        "columns_in_oracle": sum(len(c) for c in oracle.values()),
    }


_SQL_KEYWORDS = {
    "SELECT", "FROM", "WHERE", "AS", "AND", "OR", "NOT", "NULL", "IS", "IN", "ON",
    "GROUP", "ORDER", "BY", "LIMIT", "DESC", "ASC", "DISTINCT", "COUNT", "SUM", "AVG",
    "MAX", "MIN", "CAST", "SAFE_CAST", "FLOAT64", "INT64", "STRING", "TIMESTAMP",
    "DATE", "INTERVAL", "DAY", "CURRENT_TIMESTAMP", "CURRENT_DATE", "COALESCE",
    "SAFE_DIVIDE", "ROUND", "ABS", "CASE", "WHEN", "THEN", "ELSE", "END", "OVER",
    "PARTITION", "ROW_NUMBER", "ARRAY_AGG", "STRUCT", "UNNEST", "WITH", "UNION", "ALL",
}
_SQL_NOISE = {"sunny", "and", "the", "for", "not", "null"}
# Keywords that are ALSO plausible column names. These must stay eligible for the
# unknown-identifier check: `paper_trades.timestamp` is the live defect, and a blanket
# keyword filter would silently exempt it.
_TYPE_NAMES = {"TIMESTAMP", "DATE", "STRING", "DATETIME", "TIME", "INTERVAL", "NUMERIC"}

# Identifier names that plausibly hold a BigQuery Row, for the attribute-access
# form (`row.date`). Deliberately narrow: widening it to every attribute access in
# the repo would flood the scope with unrelated `obj.date` reads, and a sweep that
# cries wolf is worse than one with a stated recall limit.
_ROW_NAMES = {"row", "r", "rec", "record", "trade", "snap", "snapshot"}


def dry_run(sql: str, client: Any = None) -> Optional[str]:
    """Ask BigQuery itself to parse + type-check the SQL. $0 -- dry runs are not billed.

    Returns None if the query is valid, else the error string. This is the instrument
    that would have caught the `_production_fns` defect on the day it shipped.
    """
    from google.api_core.exceptions import BadRequest
    from google.cloud import bigquery

    if client is None:  # pragma: no cover - live path
        client = bigquery.Client(project=PROJECT)
    cfg = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    try:
        client.query(sql, job_config=cfg)
        return None
    except BadRequest as exc:
        return str(exc).splitlines()[0]


__all__ = [
    "PROJECT", "DATASETS", "SNAPSHOT_PATH",
    "fetch_live_schema", "refresh_snapshot", "load_snapshot", "snapshot_drift",
    "columns_of_type", "iter_python_files", "extract_sql_literals", "tables_in_sql",
    "derive_scope", "dry_run",
]
