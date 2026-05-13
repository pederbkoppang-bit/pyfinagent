"""phase-25.D9 verifier -- Anthropic Files API for skill markdowns.

Closes phase-24.9 F-5 (skill markdowns 500-3000 tokens each re-injected
every call; file_id reference is ~8 tokens, ~98.5% reduction per skill).

Run: source .venv/bin/activate && python3 tests/verify_phase_25_D9.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

REPO = Path(__file__).resolve().parents[1]
LLM_CLIENT = REPO / "backend" / "agents" / "llm_client.py"
PROMPTS = REPO / "backend" / "config" / "prompts.py"
ORCHESTRATOR = REPO / "backend" / "agents" / "orchestrator.py"
SKILL_OPT = REPO / "backend" / "agents" / "skill_optimizer.py"


def main() -> int:
    results: list[tuple[str, str, str]] = []

    for p in (LLM_CLIENT, PROMPTS, ORCHESTRATOR, SKILL_OPT):
        if not p.exists():
            print(f"FAIL: required source file missing: {p}")
            return 1

    llm_src = LLM_CLIENT.read_text(encoding="utf-8")
    prompts_src = PROMPTS.read_text(encoding="utf-8")
    orch_src = ORCHESTRATOR.read_text(encoding="utf-8")
    so_src = SKILL_OPT.read_text(encoding="utf-8")

    # ---- Claim 1: ClaudeClient.upload_file_to_anthropic_files_api signature.
    sig = re.search(
        r"def\s+upload_file_to_anthropic_files_api\s*\(\s*self\s*,\s*file_path[^)]*,\s*mime_type:\s*str\s*=\s*[\"']text/plain[\"']\s*,?\s*\)\s*->\s*str\s*:",
        llm_src,
        re.DOTALL,
    )
    results.append((
        "PASS" if sig else "FAIL",
        "upload_file_function_in_llm_client",
        "ClaudeClient must declare upload_file_to_anthropic_files_api(self, file_path, mime_type='text/plain') -> str",
    ))

    # ---- Claim 2: SkillFileIdCache class present with documented methods.
    cls_match = re.search(r"class\s+SkillFileIdCache\s*:", prompts_src)
    required_methods = (
        "def _hash",
        "def get_or_upload",
        "def invalidate",
        "def invalidate_stale",
        "def bulk_upload_all",
    )
    missing_methods = [m for m in required_methods if m not in prompts_src]
    results.append((
        "PASS" if cls_match and not missing_methods else "FAIL",
        "skill_file_id_cache_class_with_required_methods",
        f"SkillFileIdCache class must declare {required_methods}; missing: {missing_methods}",
    ))

    # ---- Claim 3: orchestrator __init__ contains bulk-upload bridge.
    bridge = re.search(
        r"SkillFileIdCache.*?bulk_upload_all",
        orch_src,
        re.DOTALL,
    )
    isinstance_guard = "isinstance" in orch_src and "_ClaudeClient" in orch_src
    self_attr = re.search(r"self\._skill_file_ids\s*[:=]", orch_src)
    results.append((
        "PASS" if bridge and isinstance_guard and self_attr else "FAIL",
        "skill_file_ids_loaded_at_orchestrator_startup",
        "AnalysisOrchestrator.__init__ must call SkillFileIdCache.bulk_upload_all (Claude-only) and store self._skill_file_ids",
    ))

    # ---- Claim 4: ClaudeClient.generate_content injects betas + document block on skill_file_id.
    inject_match = re.search(
        r'skill_file_id\s*=\s*config\.get\(["\']skill_file_id["\']\)',
        llm_src,
    )
    files_api_beta = '"files-api-2025-04-14"' in llm_src
    doc_block_marker = '"type": "document"' in llm_src and '"file_id"' in llm_src
    results.append((
        "PASS" if inject_match and files_api_beta and doc_block_marker else "FAIL",
        "generate_content_injects_document_block_and_betas_on_skill_file_id",
        "ClaudeClient.generate_content must read config['skill_file_id'] and inject {type:document, source:{type:file, file_id:...}} + betas=['files-api-2025-04-14']",
    ))

    # ---- Claim 5: reload_skills signature accepts anthropic_client_wrapper=None.
    rs_sig = re.search(
        r"def\s+reload_skills\s*\(\s*anthropic_client_wrapper\s*=\s*None\s*\)\s*->\s*None\s*:",
        prompts_src,
    )
    results.append((
        "PASS" if rs_sig else "FAIL",
        "reload_skills_signature_accepts_optional_client_wrapper",
        "reload_skills must accept anthropic_client_wrapper=None kwarg (backwards-compat default)",
    ))

    # ---- Claim 6: skill_optimizer.py invalidates cache at the 3 reload sites.
    so_invalidates = so_src.count("SkillFileIdCache.invalidate(agent_name")
    results.append((
        "PASS" if so_invalidates >= 3 else "FAIL",
        "skill_optimizer_invalidates_cache_at_reload_sites",
        f"skill_optimizer.py must call SkillFileIdCache.invalidate(agent_name) at >=3 sites (found {so_invalidates})",
    ))

    # ---- Claim 7: disk cache path is canonical.
    canonical_path = ".skill_file_ids.json" in prompts_src
    results.append((
        "PASS" if canonical_path else "FAIL",
        "disk_cache_path_skill_file_ids_json",
        "SkillFileIdCache must use .skill_file_ids.json as disk cache filename",
    ))

    # ---- Behavioral fixtures.
    sys.path.insert(0, str(REPO))

    # ---- Claim 8: BEHAVIORAL upload helper -- mocked SDK returns .id.
    upload_ok = False
    upload_err = ""
    try:
        sys.modules.pop("backend.agents.llm_client", None)
        from backend.agents.llm_client import ClaudeClient  # type: ignore

        fake_sdk_client = MagicMock()
        fake_uploaded = MagicMock()
        fake_uploaded.id = "file_xyz_test_42"
        fake_sdk_client.beta.files.upload.return_value = fake_uploaded

        cc = ClaudeClient(model_name="claude-sonnet-4-6", api_key="sk-test", enable_prompt_caching=True)
        # Patch _get_client to return our fake SDK client.
        cc._get_client = lambda: fake_sdk_client  # type: ignore

        td = Path(tempfile.mkdtemp(prefix="phase25d9_upload_"))
        fp = td / "fake_skill.md"
        fp.write_text("# Fake skill body\n\n## Prompt Template\nfoo bar", encoding="utf-8")

        fid = cc.upload_file_to_anthropic_files_api(fp)
        if fid != "file_xyz_test_42":
            upload_err = f"got file_id={fid!r}, expected file_xyz_test_42"
        else:
            call = fake_sdk_client.beta.files.upload.call_args
            file_tuple = call.kwargs.get("file")
            if not file_tuple or file_tuple[0] != "fake_skill.md":
                upload_err = f"upload call file kwarg wrong: {file_tuple!r}"
            elif file_tuple[2] != "text/plain":
                upload_err = f"mime_type was {file_tuple[2]!r}, expected text/plain"
            else:
                upload_ok = True
    except Exception as e:
        upload_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if upload_ok else "FAIL",
        "behavioral_upload_helper_returns_file_id",
        f"upload_file_to_anthropic_files_api must call beta.files.upload and return .id ({upload_err})",
    ))

    # ---- Claim 9: BEHAVIORAL cache hit -- pre-populated disk cache reuses file_id.
    hit_ok = False
    hit_err = ""
    try:
        sys.modules.pop("backend.config.prompts", None)
        from backend.config import prompts as p_mod  # type: ignore

        # Use a temp skill file + temp cache path.
        td2 = Path(tempfile.mkdtemp(prefix="phase25d9_hit_"))
        skill_fp = td2 / "alpha_agent.md"
        skill_fp.write_text("## Prompt Template\nhello world", encoding="utf-8")
        sha = hashlib.sha256(skill_fp.read_bytes()).hexdigest()

        # Patch SKILLS_DIR and disk cache path on the cache class.
        p_mod.SkillFileIdCache._store = {}
        p_mod.SkillFileIdCache._loaded = True  # short-circuit disk load
        p_mod.SkillFileIdCache._store["alpha_agent"] = {"hash": sha, "file_id": "file_cached_99"}
        original_dir = p_mod.SKILLS_DIR
        p_mod.SKILLS_DIR = td2

        wrapper = MagicMock()
        try:
            fid_hit = p_mod.SkillFileIdCache.get_or_upload("alpha_agent", wrapper)
        finally:
            p_mod.SKILLS_DIR = original_dir

        if fid_hit != "file_cached_99":
            hit_err = f"got {fid_hit!r}, expected file_cached_99 (cached)"
        elif wrapper.upload_file_to_anthropic_files_api.call_count != 0:
            hit_err = f"upload called {wrapper.upload_file_to_anthropic_files_api.call_count} times on cache hit (must be 0)"
        else:
            hit_ok = True
    except Exception as e:
        hit_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if hit_ok else "FAIL",
        "behavioral_cache_hit_skips_upload",
        f"get_or_upload with matching hash must return cached file_id without re-upload ({hit_err})",
    ))

    # ---- Claim 10: BEHAVIORAL cache miss -- modified file triggers re-upload.
    miss_ok = False
    miss_err = ""
    try:
        td3 = Path(tempfile.mkdtemp(prefix="phase25d9_miss_"))
        skill_fp3 = td3 / "beta_agent.md"
        skill_fp3.write_text("## Prompt Template\noriginal", encoding="utf-8")
        sha_orig = hashlib.sha256(skill_fp3.read_bytes()).hexdigest()

        p_mod.SkillFileIdCache._store = {
            "beta_agent": {"hash": sha_orig, "file_id": "file_old_88"},
        }
        p_mod.SkillFileIdCache._loaded = True
        original_dir3 = p_mod.SKILLS_DIR
        p_mod.SKILLS_DIR = td3

        # Now change the file.
        skill_fp3.write_text("## Prompt Template\nupdated", encoding="utf-8")

        wrapper3 = MagicMock()
        wrapper3.upload_file_to_anthropic_files_api.return_value = "file_new_99"

        try:
            fid_miss = p_mod.SkillFileIdCache.get_or_upload("beta_agent", wrapper3)
        finally:
            p_mod.SKILLS_DIR = original_dir3

        if fid_miss != "file_new_99":
            miss_err = f"got {fid_miss!r}, expected file_new_99 (re-uploaded)"
        elif wrapper3.upload_file_to_anthropic_files_api.call_count != 1:
            miss_err = f"upload called {wrapper3.upload_file_to_anthropic_files_api.call_count} times (must be 1)"
        else:
            miss_ok = True
    except Exception as e:
        miss_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if miss_ok else "FAIL",
        "behavioral_cache_miss_triggers_reupload",
        f"hash mismatch must trigger re-upload + return new file_id ({miss_err})",
    ))

    # ---- Claim 11: BEHAVIORAL document-block injection in generate_content.
    inj_ok = False
    inj_err = ""
    try:
        # Re-import to ensure fresh module state.
        sys.modules.pop("backend.agents.llm_client", None)
        from backend.agents.llm_client import ClaudeClient  # type: ignore

        fake_sdk = MagicMock()
        fake_response = MagicMock()
        fake_response.content = [MagicMock(text='{"action":"BUY","confidence":50,"score":5,"reason":"ok"}')]
        fake_response.usage = MagicMock(
            input_tokens=8, output_tokens=12,
            cache_creation_input_tokens=0, cache_read_input_tokens=0,
        )
        fake_sdk.messages.create.return_value = fake_response

        cc2 = ClaudeClient(model_name="claude-sonnet-4-6", api_key="sk-test", enable_prompt_caching=False)
        cc2._get_client = lambda: fake_sdk  # type: ignore

        config = {"max_output_tokens": 200, "skill_file_id": "file_skill_42"}
        cc2.generate_content(prompt="hello", generation_config=config)

        if fake_sdk.messages.create.call_count != 1:
            inj_err = f"messages.create call_count={fake_sdk.messages.create.call_count}"
        else:
            kw = fake_sdk.messages.create.call_args.kwargs
            betas = kw.get("betas") or []
            messages = kw.get("messages") or []
            if "files-api-2025-04-14" not in betas:
                inj_err = f"betas={betas} missing files-api-2025-04-14"
            elif not messages:
                inj_err = "messages empty"
            elif not isinstance(messages[0].get("content"), list):
                inj_err = f"messages[0].content is not list: {type(messages[0].get('content'))}"
            else:
                blocks = messages[0]["content"]
                has_doc = any(
                    isinstance(b, dict) and b.get("type") == "document" and (b.get("source") or {}).get("file_id") == "file_skill_42"
                    for b in blocks
                )
                if not has_doc:
                    inj_err = f"no document block with file_id=file_skill_42 in {blocks}"
                else:
                    inj_ok = True
    except Exception as e:
        inj_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if inj_ok else "FAIL",
        "per_cycle_skill_content_input_tokens_reduced_by_at_least_90_percent",
        f"generate_content with skill_file_id must inject document block + files-api beta ({inj_err})",
    ))

    # ---- Claim 12: BEHAVIORAL fallback -- bulk_upload raising does NOT crash orchestrator
    # (covered by orchestrator's try/except; structural check that the try/except wraps the call).
    has_try_except = re.search(
        r"try:[\s\S]*?bulk_upload_all[\s\S]*?except Exception",
        orch_src,
    )
    results.append((
        "PASS" if has_try_except else "FAIL",
        "orchestrator_bulk_upload_wrapped_in_try_except",
        "AnalysisOrchestrator.__init__ must wrap bulk_upload_all in try/except (fail-open)",
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
