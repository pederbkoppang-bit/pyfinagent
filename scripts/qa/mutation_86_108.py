#!/usr/bin/env python3
"""phase-86.108 criterion 6: mutation matrix for the parse-failure ledger and
the gated-flag read path.

## Scoring rule, and why it is this strict

A cell is a **KILL** only when ALL of the following hold:

  1. the CONTROL was observed green FIRST (otherwise every cell is UNSCORABLE
     and the matrix is worthless);
  2. pytest exits **1** -- i.e. tests ran and failed. Exit 2 (usage), 3
     (internal), 4 (usage) and especially **exit 5 (no tests collected)** are
     NOT kills. An empty selection exits 5, and scoring `rc != 0` as a kill is
     how a typo'd `-k` gets recorded as a passing matrix;
  3. the mutant still **collects the same number of tests as the control**. A
     mutation that breaks the import collects 0 and "fails" for a reason that
     has nothing to do with the guard -- a mutant that cannot build is not a
     killed mutant;
  4. the **specifically named** test fails. A cell that goes red only via
     collateral damage does not demonstrate that *this* guard catches *this*
     defect. This is the discrimination requirement: the control answer and the
     mutant's answer must differ for the stated reason.

After every cell the file is restored and the restore is verified by SHA-256
against the pre-mutation bytes. A cell whose restore does not verify aborts the
whole run rather than continuing against a dirty tree.
"""
from __future__ import annotations

import hashlib
import pathlib
import re
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[2]
SUITE = "backend/tests/test_phase_86_108_parse_failure_ledger.py"
PY = str(REPO / ".venv" / "bin" / "python")

LEDGER = "backend/agents/parse_failure_ledger.py"
DEBATE = "backend/agents/debate.py"
LLM_PARSE = "backend/agents/llm_parse.py"
RISK = "backend/agents/risk_debate.py"
ORCH = "backend/agents/orchestrator.py"
FLAGS = "backend/config/gated_flags.py"

# id -> (file, old_substring, new_substring, test that MUST fail, what it proves)
CELLS = [
    ("M1", DEBATE,
     '        record_parse_failure(label, KIND_PARSE_FAILED, site=_LEDGER_SITE, model_name=model_name)\n',
     '',
     "test_debate_parse_json_return_values_are_unchanged_by_recording",
     "removing the syntactic-failure record at a real emit site is caught"),

    ("M2", DEBATE,
     '    if not isinstance(data, dict):\n        record_parse_failure(\n'
     '            label, KIND_SCHEMA_VALID_BUT_REJECTED, site=_LEDGER_SITE,\n'
     '            detail=f"parsed to {type(data).__name__}, not dict", model_name=model_name,\n'
     '        )\n        return None\n',
     '    if not isinstance(data, dict):\n        return None\n',
     "test_debate_parse_json_return_values_are_unchanged_by_recording",
     "the valid-JSON-wrong-shape case going unrecorded is caught"),

    ("M3", LLM_PARSE,
     "            KIND_TRUNCATED if truncated else KIND_PARSE_FAILED,",
     "            KIND_PARSE_FAILED,",
     "test_llm_parse_separates_truncated_from_malformed",
     "collapsing truncated into parse_failed (wrong-cause attribution) is caught"),

    ("M4", LEDGER,
     "    except Exception as exc:  # never propagate into the parse path",
     "    except ZeroDivisionError as exc:  # MUTANT: no longer catches",
     "test_recorder_failure_is_counted_and_does_not_propagate",
     "a recorder that propagates into the money path is caught"),

    ("M5", LEDGER,
     "            _LEDGER.note_recorder_error()\n",
     "            pass\n",
     "test_recorder_failure_is_counted_and_does_not_propagate",
     "a fail-open guard that hides its own breakage is caught"),

    ("M6", FLAGS,
     "_SCALAR_TYPES = (bool, int, float)",
     "_SCALAR_TYPES = (bool, int, float, str)",
     "test_no_string_typed_field_can_enter_the_payload",
     "widening the population to str -- the secret-leak path -- is caught"),

    ("M7", FLAGS,
     "    exposed = set(FullSettings.model_fields)\n    return sorted(\n"
     "        name\n        for name, field in Settings.model_fields.items()\n"
     "        if name not in exposed and field.annotation in _SCALAR_TYPES\n    )",
     '    _ = FullSettings, Settings  # MUTANT: hardcoded list, rule ignored\n'
     '    return sorted(["paper_synthesis_integrity_enabled",\n'
     '                   "paper_position_recommendation_fix_enabled",\n'
     '                   "paper_risk_judge_shape_fix_enabled",\n'
     '                   "claude_code_timeout_s", "claude_code_empty_retry_max",\n'
     '                   "paper_soft_sector_diversity_enabled",\n'
     '                   "paper_soft_sector_diversity_w"])',
     "test_population_is_derived_and_excludes_what_fullsettings_exposes",
     "reverting to the step's hardcoded list of 7 is caught"),

    ("M8", FLAGS,
     "        unknown = sorted(wanted - set(names))",
     "        unknown = []  # MUTANT: silently drop unknown names",
     "test_flags_route_reports_an_unknown_name_instead_of_dropping_it",
     "an endpoint that silently drops a typo'd name is caught"),

    ("M9", FLAGS,
     "            if key in keys:\n                out[key] = val.strip().strip('\"').strip(\"'\")",
     "            out[key] = val.strip().strip('\"').strip(\"'\")  # MUTANT: dump every key",
     "test_env_reader_returns_only_requested_keys_and_ignores_comments",
     "turning the .env reader into a file-dump primitive is caught"),

    ("M10", LEDGER,
     '                "records_seen": self._seen,',
     '                "records_seen": len(self._recent),  # MUTANT: gauge as counter',
     "test_records_seen_is_a_counter_and_reconciles_after_eviction",
     "reporting the ring gauge as the event counter is caught"),

    ("M11", LEDGER,
     '            "agent": str(agent or "(no agent phrase)"),',
     '            "agent": str(agent or "(no agent phrase)").split()[-1],  # MUTANT',
     "test_agent_phrase_is_preserved_verbatim_not_collapsed",
     "the original filing's three-analysts-into-one fold is caught"),

    ("M12", LEDGER,
     '            "rail": rail,',
     '            "rail": "",  # MUTANT: drop the rail stamp',
     "test_record_carries_agent_kind_site_and_rail",
     "dropping the rail stamp -- the field the log corpus never had -- is caught"),
# M13 and M14 exist because a Q/A executed a mutant this matrix did not
    # contain -- an INVERTED rail attribution -- and it SURVIVED. The rail's
    # only guard was `assert rec["rail"] in {...}`, a set-membership check that
    # passes for every wrong value in the vocabulary. M12 mutates the field to
    # "", the one value OUTSIDE the vocabulary, i.e. the only mutation such an
    # assertion can catch. These two mutate to values that are wrong but VALID.
    ("M13", LEDGER,
     '    if not str(model_name).startswith("claude-"):',
     '    if str(model_name).startswith("claude-"):  # MUTANT: inverted mapping',
     "test_resolve_rail_disagrees_with_the_flag_only_rule",
     "an INVERTED rail attribution (wrong but in-vocabulary) is caught"),

    ("M14", LEDGER,
     '    if not model_name:\n        return RAIL_UNKNOWN, "no_model_in_scope_at_emit_site"',
     '    if True:  # MUTANT: revert to the flag-only rule, ignoring the model\n'
     '        from backend.config.settings import get_settings as _gs\n'
     '        return ((RAIL_CLAUDE_CODE if bool(getattr(_gs(), "paper_use_claude_code_route", False))\n'
     '                 else RAIL_DIRECT), "flag_only")',
     "test_resolve_rail_disagrees_with_the_flag_only_rule",
     "reverting to the flag-only rule (the defect the Q/A found) is caught"),
# M15-M17 exist because a Q/A executed three mutants at the production CALL
    # SITE that this matrix did not contain, and two SURVIVED 29/29 -- the
    # cycle-2 fix had relocated the defect one seam upstream of its own guards.
    # M15 is the exact mutant that survived.
    ("M15", DEBATE,
     "    return declared or getattr(model, \"model_name\", None) or None",
     "    return \"claude-opus-4-8\"  # MUTANT: hardcoded model at the call-site helper",
     "test_run_debate_records_the_real_client_model",
     "the hardcoded-model mutant that SURVIVED the cycle-2 matrix is now caught"),

    ("M16", RISK,
     "judge_result = _parse_json(judge_text, \"Risk Judge\",\n"
     "                               model_name=_effective_model_name(\n"
     "                                   deep_think_model_name or general_model_name, _judge_model))",
     "judge_result = _parse_json(judge_text, \"Risk Judge\")  # MUTANT: call site stops forwarding",
     "test_every_parse_call_site_forwards_a_model_name",
     "a production call site that stops forwarding the model is caught"),

    ("M17", ORCH,
     "            model_name=_client_model_name(self.synthesis_client))",
     "            model_name=\"claude-opus-4-8\")  # MUTANT: literal at an orchestrator call site",
     "test_every_parse_call_site_forwards_a_model_name",
     "a LITERAL model at the orchestrator sites -- which no driver reaches -- is caught"),
# M18-M19 are the two AST-LEGAL mutants a Q/A executed against the
    # blacklist version of the completeness guard. Both SURVIVED 37/37: the
    # kwarg value was an ast.Call / ast.BoolOp, not an ast.Constant, so a guard
    # that rejected only literals could not see either. They are the reason the
    # guard is now a WHITELIST of accepted shapes.
    ("M18", ORCH,
     "            model_name=_client_model_name(self.synthesis_client))",
     "            model_name=_client_model_name(None))  # MUTANT: fakes 'unknown'",
     "test_every_parse_call_site_forwards_a_model_name",
     "a resolver called with None -- a FALSE 'unknown' while a model is in scope"),

    ("M19", ORCH,
     '                critic_text, "Critic",\n'
     "                model_name=_client_model_name(self.deep_think_client))",
     '                critic_text, "Critic",\n'
     '                model_name=_client_model_name(self.deep_think_client) or "claude-opus-4-8")  # MUTANT',
     "test_every_parse_call_site_forwards_a_model_name",
     "a BoolOp fallback to a hardcoded model -- actively misattributes"),
]

_COLLECTED = re.compile(r"(\d+) passed|(\d+) failed")


def run_suite() -> tuple[int, str]:
    p = subprocess.run(
        [PY, "-m", "pytest", SUITE, "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=REPO, capture_output=True, text=True,
    )
    return p.returncode, p.stdout + p.stderr


def collected(out: str) -> int:
    """Total tests that actually RAN. Used to reject a mutant that broke the
    import and therefore collected nothing."""
    n = 0
    for m in re.finditer(r"(\d+) (passed|failed|error)", out):
        n += int(m.group(1))
    return n


def main() -> int:
    print("=" * 70)
    print("CONTROL (must be GREEN before any cell is scored)")
    print("=" * 70)
    rc, out = run_suite()
    control_n = collected(out)
    print(out.strip().splitlines()[-1] if out.strip() else "(no output)")
    print(f"CONTROL rc={rc}  collected={control_n}")
    if rc != 0 or control_n == 0:
        print("\nCONTROL IS NOT GREEN -- every cell below would be UNSCORABLE. Aborting.")
        return 2

    results = []
    for cid, relpath, old, new, must_fail, proves in CELLS:
        path = REPO / relpath
        original = path.read_bytes()
        digest = hashlib.sha256(original).hexdigest()
        text = original.decode("utf-8")

        if text.count(old) != 1:
            results.append((cid, "UNSCORABLE", f"anchor matched {text.count(old)}x (need exactly 1)", proves))
            print(f"{cid}: UNSCORABLE -- anchor matched {text.count(old)} times")
            continue

        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        try:
            m_rc, m_out = run_suite()
            m_n = collected(m_out)
            named_failed = f"{must_fail}" in m_out
            if m_rc == 5:
                verdict, why = "UNSCORABLE", "pytest exit 5 -- no tests collected"
            elif m_rc == 0:
                verdict, why = "SURVIVED", "suite stayed green under the mutation"
            elif m_rc != 1:
                verdict, why = "UNSCORABLE", f"pytest exit {m_rc} (not a test failure)"
            elif m_n != control_n:
                verdict, why = "UNSCORABLE", f"collected {m_n} != control {control_n} (mutant did not build)"
            elif not named_failed:
                verdict, why = "UNSCORABLE", f"red, but {must_fail} was not the failing test"
            else:
                verdict, why = "KILLED", f"{must_fail} failed; collected {m_n} == control"
        finally:
            # No `return` in this block: a return inside `finally` discards any
            # in-flight exception, which would hide the very failure that made
            # the restore urgent. Record the outcome and act on it after.
            path.write_bytes(original)
            restore_ok = hashlib.sha256(path.read_bytes()).hexdigest() == digest

        if not restore_ok:
            print(f"\nFATAL: restore of {relpath} did NOT verify. Tree is dirty. Aborting.")
            return 3

        results.append((cid, verdict, why, proves))
        print(f"{cid}: {verdict:11s} {why}")

    print("\n" + "=" * 70)
    print("MATRIX")
    print("=" * 70)
    for cid, verdict, why, proves in results:
        print(f"  {cid:4s} {verdict:11s} {proves}")
        if verdict != "KILLED":
            print(f"        -> {why}")
    survivors = [c for c, v, *_ in results if v == "SURVIVED"]
    unscorable = [c for c, v, *_ in results if v == "UNSCORABLE"]
    print(f"\nKILLED={len([1 for _, v, *_ in results if v == 'KILLED'])}/{len(results)}"
          f"  SURVIVORS={survivors or 'none'}  UNSCORABLE={unscorable or 'none'}")

    # Final restore proof: the whole tree must be byte-identical to HEAD for
    # the files this matrix touched, apart from changes made before it ran.
    print("\nRESTORE VERIFIED: every cell re-hashed to its pre-mutation SHA-256.")
    return 0 if not survivors and not unscorable else 1


if __name__ == "__main__":
    raise SystemExit(main())
