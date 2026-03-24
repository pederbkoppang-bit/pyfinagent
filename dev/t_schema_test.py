"""Test schema conversion exactly as it happens in the live code path."""
import json
import sys
sys.path.insert(0, ".")

from backend.agents.llm_client import GeminiClient
from backend.agents.schemas import (
    ModeratorConsensus, DevilsAdvocateResult,
    SynthesisReport, CriticVerdict,
    RiskAnalystArgument, RiskJudgeVerdict,
)
from vertexai.generative_models import GenerationConfig

configs = [
    ("ModeratorConsensus", ModeratorConsensus, 2048),
    ("DevilsAdvocateResult", DevilsAdvocateResult, 1536),
    ("SynthesisReport", SynthesisReport, 4096),
    ("CriticVerdict", CriticVerdict, 2048),
    ("RiskAnalystArgument", RiskAnalystArgument, 1024),
    ("RiskJudgeVerdict", RiskJudgeVerdict, 1536),
]

for name, cls, max_tokens in configs:
    # Step 1: Raw Pydantic JSON Schema
    raw = cls.model_json_schema()
    has_defs_raw = "$defs" in raw
    
    # Step 2: Flatten
    flat = GeminiClient._flatten_schema(raw)
    flat_dump = json.dumps(flat)
    has_defs_flat = "$defs" in flat_dump
    has_ref_flat = "$ref" in flat_dump
    has_anyof = "anyOf" in flat_dump
    has_title = '"title"' in flat_dump
    has_default = '"default"' in flat_dump
    
    print(f"\n--- {name} ---")
    print(f"  Raw has $defs: {has_defs_raw}")
    print(f"  Flat has $defs: {has_defs_flat}, $ref: {has_ref_flat}, anyOf: {has_anyof}, title: {has_title}, default: {has_default}")
    
    # Step 3: Create GenerationConfig (same as SDK would)
    try:
        gc = GenerationConfig(
            temperature=0.0,
            top_k=1,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
            response_schema=flat,
        )
        print(f"  GenerationConfig: OK")
    except Exception as e:
        print(f"  GenerationConfig FAILED: {type(e).__name__}: {e}")
    
    # Step 4: Also try passing as dict to GenerationConfig (how generate_content does it)
    try:
        config_dict = {
            "temperature": 0.0,
            "top_k": 1,
            "max_output_tokens": max_tokens,
            "response_mime_type": "application/json",
            "response_schema": flat,
        }
        gc2 = GenerationConfig(**config_dict)
        print(f"  Dict-based GenerationConfig: OK")
    except Exception as e:
        print(f"  Dict-based GenerationConfig FAILED: {type(e).__name__}: {e}")

print("\nAll tests complete.")
