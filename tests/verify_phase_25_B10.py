"""verify_phase_25_B10 -- SecretStr migration for API keys/tokens.

Verifies:
  1. settings.py imports SecretStr from pydantic.
  2. anthropic_api_key field is annotated SecretStr.
  3. openai + alpaca (id + secret) + auth_secret + slack (bot + app) tokens are all SecretStr.
  4. At least 10 .get_secret_value() consumer sites exist across the backend.
  5. repr() on the live Settings instance masks all sensitive fields as '**********'.

Exit 0 on PASS, 1 on FAIL.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

results: list[tuple[str, bool, str]] = []


def claim(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, bool(condition), detail))


# ── Claim 1: settings imports SecretStr ────────────────────────────────
settings_src = (REPO / "backend/config/settings.py").read_text(encoding="utf-8")
has_import = "SecretStr" in settings_src and "from pydantic import" in settings_src
claim(
    "1. settings_imports_secretstr",
    has_import,
    f"import line present: {bool(re.search(r'from pydantic import [^#\\n]*SecretStr', settings_src))}",
)


# ── Claim 2: anthropic_api_key is SecretStr ────────────────────────────
anthropic_typed = bool(
    re.search(r"anthropic_api_key\s*:\s*SecretStr", settings_src)
)
claim(
    "2. anthropic_api_key_is_secretstr_type",
    anthropic_typed,
    "anthropic_api_key: SecretStr present" if anthropic_typed else "still typed as str",
)


# ── Claim 3: openai + alpaca + auth + slack tokens all SecretStr ───────
fields = {
    "openai_api_key": False,
    "alpaca_api_key_id": False,
    "alpaca_api_secret_key": False,
    "auth_secret": False,
    "slack_bot_token": False,
    "slack_app_token": False,
}
for f in fields:
    fields[f] = bool(re.search(rf"\b{f}\s*:\s*SecretStr", settings_src))

all_secret = all(fields.values())
claim(
    "3. openai_alpaca_auth_slack_keys_all_secretstr",
    all_secret,
    " ".join(f"{k}={v}" for k, v in fields.items()),
)


# ── Claim 4: at least 10 .get_secret_value() consumer sites ─────────────
# grep across backend/
proc = subprocess.run(
    ["grep", "-rn", "--include=*.py", ".get_secret_value()", "backend"],
    cwd=str(REPO),
    capture_output=True,
    text=True,
)
lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
# Exclude lines inside settings.py itself (the default values)
consumer_lines = [ln for ln in lines if "/settings.py" not in ln]
n_consumers = len(consumer_lines)
claim(
    "4. downstream_consumers_use_get_secret_value",
    n_consumers >= 10,
    f"consumer call sites={n_consumers} (expected >=10); samples: {consumer_lines[:3]}",
)


# ── Claim 5: repr(settings) masks sensitive fields ──────────────────────
try:
    os.environ.setdefault("GCP_PROJECT_ID", "test-project")
    os.environ.setdefault("RAG_DATA_STORE_ID", "test-store")
    # Set fake sensitive values so we can verify they're masked
    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-12345"
    os.environ["AUTH_SECRET"] = "test-auth-secret-67890"
    os.environ["SLACK_BOT_TOKEN"] = "xoxb-test-token"

    from backend.config.settings import get_settings  # noqa: E402
    get_settings.cache_clear()
    s = get_settings()
    repr_text = repr(s)
    # SecretStr's repr shows '**********' instead of the raw value
    masked_ok = (
        "sk-ant-test-12345" not in repr_text
        and "test-auth-secret-67890" not in repr_text
        and "xoxb-test-token" not in repr_text
        and "**********" in repr_text
    )
    rt_detail = f"masked={'**********' in repr_text} leak={any(v in repr_text for v in ('sk-ant-test-12345', 'test-auth-secret-67890', 'xoxb-test-token'))}"
except Exception as e:
    masked_ok = False
    rt_detail = f"Exception: {type(e).__name__}: {e}"

claim(
    "5. repr_settings_masks_sensitive_fields",
    masked_ok,
    rt_detail,
)


# ── Summary ─────────────────────────────────────────────────────────────
print("\n=== phase-25.B10 verification ===\n")
all_pass = True
for name, ok, detail in results:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}")
    if detail:
        print(f"        -> {detail}")
    if not ok:
        all_pass = False

print()
if all_pass:
    print(f"ALL {len(results)} CLAIMS PASS")
    sys.exit(0)
else:
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"{failed} of {len(results)} claims FAILED")
    sys.exit(1)
