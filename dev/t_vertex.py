"""Test flattened schemas against actual Vertex AI SDK validation."""
import json, sys
sys.path.insert(0, ".")

from backend.agents.llm_client import GeminiClient
from backend.agents.schemas import (
    SynthesisReport, CriticVerdict, ModeratorConsensus,
    DevilsAdvocateResult, RiskAnalystArgument, RiskJudgeVerdict,
)

# Try to validate through the actual Vertex AI SDK schema conversion
try:
    from google.cloud.aiplatform_v1beta1.types import openapi as openapi_types
    print("Using aiplatform_v1beta1 openapi types")
except ImportError:
    print("v1beta1 not available")

try:
    from vertexai.generative_models import GenerativeModel, GenerationConfig
    print("vertexai available")
except ImportError:
    print("vertexai not available")
    sys.exit(1)

# Check what the SDK actually does with a schema dict
# Look at how GenerationConfig processes response_schema
for cls in [SynthesisReport, CriticVerdict, ModeratorConsensus,
            DevilsAdvocateResult, RiskAnalystArgument, RiskJudgeVerdict]:
    flat = GeminiClient._flatten_schema(cls.model_json_schema())
    print(f"\n--- {cls.__name__} ---")
    print(json.dumps(flat, indent=2)[:800])
    
    # Try creating a GenerationConfig with it
    try:
        gc = GenerationConfig(
            response_mime_type="application/json",
            response_schema=flat,
            temperature=0.0,
            top_k=1,
            max_output_tokens=4096,
        )
        print(f"  GenerationConfig: OK")
    except Exception as e:
        print(f"  GenerationConfig ERROR: {type(e).__name__}: {e}")

print("\nDone")
