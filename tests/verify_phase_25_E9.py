"""phase-25.E9 verifier -- adopt native Citations; deprecate CitationAgent.

Closes phase-24.9 F-6 (multi_agent_orchestrator._add_citations ran a
separate Sonnet call to add footnote markers; native Citations does this
server-side at zero extra cost).

Run: source .venv/bin/activate && python3 tests/verify_phase_25_E9.py
"""
from __future__ import annotations

import asyncio
import re
import sys
import warnings
from pathlib import Path
from unittest.mock import MagicMock

REPO = Path(__file__).resolve().parents[1]
LLM_CLIENT = REPO / "backend" / "agents" / "llm_client.py"
MAS_ORCH = REPO / "backend" / "agents" / "multi_agent_orchestrator.py"


def main() -> int:
    results: list[tuple[str, str, str]] = []

    for p in (LLM_CLIENT, MAS_ORCH):
        if not p.exists():
            print(f"FAIL: required source file missing: {p}")
            return 1

    llm_src = LLM_CLIENT.read_text(encoding="utf-8")
    mas_src = MAS_ORCH.read_text(encoding="utf-8")

    # ---- Claim 1: LLMResponse has citations field.
    citations_field = re.search(
        r"citations\s*:\s*Optional\[list\[dict\]\]\s*=\s*None",
        llm_src,
    )
    results.append((
        "PASS" if citations_field else "FAIL",
        "llm_response_dataclass_has_citations_field",
        "LLMResponse must declare citations: Optional[list[dict]] = None",
    ))

    # ---- Claim 2: citations literal '{"enabled": True}' present in the
    # document block injection path. Match either dict-literal form
    # `"citations": {"enabled": True}` or dict-key assignment
    # `document_block["citations"] = {"enabled": True}`.
    enabled_literal = re.search(
        r'(?:"citations"\s*:\s*\{"enabled"\s*:\s*True\}|\["citations"\]\s*=\s*\{"enabled"\s*:\s*True\})',
        llm_src,
    )
    results.append((
        "PASS" if enabled_literal else "FAIL",
        "citations_enabled_true_on_document_content_blocks",
        '"citations": {"enabled": True} must appear in the document block injection path (literal or assignment form)',
    ))

    # ---- Claim 3: generate_content reads config["citations"].
    config_read = re.search(
        r'config\.get\(["\']citations["\']\)',
        llm_src,
    )
    results.append((
        "PASS" if config_read else "FAIL",
        "generate_content_reads_config_citations",
        "ClaudeClient.generate_content must read config.get('citations') to decide whether to enable",
    ))

    # ---- Claim 4: _add_citations method emits DeprecationWarning.
    dep_warn = re.search(
        r"_add_citations[\s\S]*?warnings\.warn\([\s\S]*?DeprecationWarning",
        mas_src,
    )
    results.append((
        "PASS" if dep_warn else "FAIL",
        "citationagent_class_marked_deprecated",
        "_add_citations method must call warnings.warn with DeprecationWarning",
    ))

    # ---- Claim 5: _add_citations returns the input response unchanged with 0/0 usage.
    early_return = re.search(
        r"_add_citations[\s\S]*?return\s+response\s*,\s*\{\"input\":\s*0\s*,\s*\"output\":\s*0\}",
        mas_src,
    )
    results.append((
        "PASS" if early_return else "FAIL",
        "add_citations_early_returns_input_unchanged",
        "_add_citations must early-return (response, {'input': 0, 'output': 0})",
    ))

    # ---- Claim 6: deprecation docstring references phase-25.E9.
    docstring_ref = re.search(
        r"_add_citations[\s\S]*?phase-25\.E9",
        mas_src,
    )
    results.append((
        "PASS" if docstring_ref else "FAIL",
        "deprecation_docstring_references_phase_25_e9",
        "_add_citations docstring must reference phase-25.E9",
    ))

    # ---- Behavioral fixtures.
    sys.path.insert(0, str(REPO))
    sys.modules.pop("backend.agents.llm_client", None)
    from backend.agents.llm_client import ClaudeClient, LLMResponse  # type: ignore

    def _make_response_with_citations():
        cit = MagicMock()
        cit.type = "char_location"
        cit.cited_text = "AAPL closed at $150."
        cit.document_index = 0
        cit.document_title = "AAPL 10-K"
        cit.start_char_index = 100
        cit.end_char_index = 120
        cit.start_page_number = None
        cit.end_page_number = None
        cit.start_block_index = None
        cit.end_block_index = None

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "AAPL closed at $150 [citation_1]"
        text_block.citations = [cit]

        usage = MagicMock(
            prompt_token_count=50, candidates_token_count=40,
            total_token_count=90,
            cache_creation_input_tokens=0, cache_read_input_tokens=0,
            input_tokens=50, output_tokens=40,
        )
        resp = MagicMock()
        resp.content = [text_block]
        resp.usage = usage
        resp.stop_reason = "end_turn"
        return resp

    def _make_response_no_citations():
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Plain response without sources."
        text_block.citations = None  # most calls have no citations attr

        usage = MagicMock(
            prompt_token_count=50, candidates_token_count=40,
            total_token_count=90,
            cache_creation_input_tokens=0, cache_read_input_tokens=0,
            input_tokens=50, output_tokens=40,
        )
        resp = MagicMock()
        resp.content = [text_block]
        resp.usage = usage
        resp.stop_reason = "end_turn"
        return resp

    # ---- Claim 7: BEHAVIORAL citation extraction.
    extract_ok = False
    extract_err = ""
    try:
        fake_sdk = MagicMock()
        fake_sdk.messages.create.return_value = _make_response_with_citations()
        cc = ClaudeClient(model_name="claude-sonnet-4-6", api_key="sk-test", enable_prompt_caching=False)
        cc._get_client = lambda: fake_sdk  # type: ignore

        r = cc.generate_content(prompt="hello", generation_config={"max_output_tokens": 200, "citations": True})
        if r.citations is None:
            extract_err = "citations is None despite mocked response with citation"
        elif not isinstance(r.citations, list):
            extract_err = f"citations type wrong: {type(r.citations)}"
        elif len(r.citations) != 1:
            extract_err = f"len(citations)={len(r.citations)}, expected 1"
        else:
            c = r.citations[0]
            required = {"type", "cited_text", "document_index", "document_title", "start_char_index", "end_char_index"}
            if not required <= set(c.keys()):
                extract_err = f"citation dict missing keys: {required - set(c.keys())}"
            elif c.get("cited_text") != "AAPL closed at $150.":
                extract_err = f"cited_text wrong: {c.get('cited_text')!r}"
            else:
                extract_ok = True
    except Exception as e:
        extract_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if extract_ok else "FAIL",
        "behavioral_citation_extraction_into_llm_response",
        f"generate_content must extract block.citations into LLMResponse.citations ({extract_err})",
    ))

    # ---- Claim 8: BEHAVIORAL no-citations path -> citations is None.
    none_ok = False
    none_err = ""
    try:
        fake_sdk2 = MagicMock()
        fake_sdk2.messages.create.return_value = _make_response_no_citations()
        cc2 = ClaudeClient(model_name="claude-sonnet-4-6", api_key="sk-test", enable_prompt_caching=False)
        cc2._get_client = lambda: fake_sdk2  # type: ignore

        r2 = cc2.generate_content(prompt="hello", generation_config={"max_output_tokens": 200})
        if r2.citations is not None:
            none_err = f"citations={r2.citations!r}, expected None"
        else:
            none_ok = True
    except Exception as e:
        none_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if none_ok else "FAIL",
        "behavioral_no_citations_yields_none_not_empty_list",
        f"text-only response must yield LLMResponse.citations = None (not []) ({none_err})",
    ))

    # ---- Claim 9: BEHAVIORAL document-block injection with citations.enabled=True.
    inj_ok = False
    inj_err = ""
    try:
        fake_sdk3 = MagicMock()
        fake_sdk3.messages.create.return_value = _make_response_no_citations()
        cc3 = ClaudeClient(model_name="claude-sonnet-4-6", api_key="sk-test", enable_prompt_caching=False)
        cc3._get_client = lambda: fake_sdk3  # type: ignore

        cc3.generate_content(prompt="hello", generation_config={
            "max_output_tokens": 200,
            "skill_file_id": "file_skill_q",
            "citations": True,
        })
        call = fake_sdk3.messages.create.call_args
        messages = call.kwargs.get("messages") or []
        if not messages:
            inj_err = "no messages"
        else:
            blocks = messages[0].get("content") or []
            doc_block = next((b for b in blocks if isinstance(b, dict) and b.get("type") == "document"), None)
            if not doc_block:
                inj_err = "no document block in messages content"
            elif doc_block.get("citations") != {"enabled": True}:
                inj_err = f"document block citations={doc_block.get('citations')!r}, expected {{'enabled': True}}"
            else:
                inj_ok = True
    except Exception as e:
        inj_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if inj_ok else "FAIL",
        "q_and_a_response_includes_citation_metadata",
        f"generate_content with skill_file_id + citations=True must add citations.enabled=True to document block ({inj_err})",
    ))

    # ---- Claim 10: BEHAVIORAL _add_citations early-return with DeprecationWarning.
    dep_ok = False
    dep_err = ""
    try:
        sys.modules.pop("backend.agents.multi_agent_orchestrator", None)
        from backend.agents.multi_agent_orchestrator import MultiAgentOrchestrator  # type: ignore

        # Build a stub instance to call _add_citations without full init.
        orch = MultiAgentOrchestrator.__new__(MultiAgentOrchestrator)
        # Mock the classification (agent_type does not matter -- the
        # deprecation should short-circuit regardless).
        fake_cls = MagicMock()
        original = "Original response text with no markers."

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = asyncio.run(orch._add_citations(original, fake_cls))

        if not isinstance(result, tuple) or len(result) != 2:
            dep_err = f"return shape wrong: {type(result)} len={len(result) if hasattr(result, '__len__') else 'n/a'}"
        elif result[0] != original:
            dep_err = f"response changed: {result[0]!r}"
        elif result[1] != {"input": 0, "output": 0}:
            dep_err = f"usage wrong: {result[1]!r}"
        else:
            has_dep_warn = any(
                issubclass(w.category, DeprecationWarning)
                and "phase-25.E9" in str(w.message)
                for w in caught
            )
            if not has_dep_warn:
                dep_err = f"no DeprecationWarning with phase-25.E9 raised; caught={[(w.category.__name__, str(w.message)[:80]) for w in caught]}"
            else:
                dep_ok = True
    except Exception as e:
        dep_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if dep_ok else "FAIL",
        "behavioral_add_citations_emits_warning_and_short_circuits",
        f"_add_citations must emit DeprecationWarning + return input unchanged ({dep_err})",
    ))

    # ---- Claim 11: cross-link guard preserved (citations + schema -> ValueError).
    guard_preserved = re.search(
        r"citations.*?and\s+output_config[\s\S]*?cannot be used together",
        llm_src,
    )
    results.append((
        "PASS" if guard_preserved else "FAIL",
        "citations_vs_structured_outputs_guard_preserved",
        "the existing ValueError guard for citations + schema must be preserved",
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
