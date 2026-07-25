"""JSON-safety helpers for the API layer (phase-80.1).

WHY THIS EXISTS
---------------
`GET /api/signals/{ticker}` returned HTTP 500 for EVERY ticker. The traceback
(verbatim from backend.log):

    ValueError: Out of range float values are not JSON compliant: np.float64(nan)
      when serializing dict item "1mo"
      when serializing dict item "stock_returns"
      when serializing dict item "sector"

Starlette hardcodes `json.dumps(..., allow_nan=False)` in
`JSONResponse.render` (starlette 1.0.0 responses.py:194-201), so ANY
non-finite float that reaches a response dict is a guaranteed 500. It
cannot be caught by the route's own error handling: the per-signal
`_safe()` wrapper in signals.py guards only the awaited coroutine, while
this failure happens later, during ASGI rendering, after the route has
already returned its dict.

WHY A SANITISER AND NOT ONE OF THE OBVIOUS ALTERNATIVES
-------------------------------------------------------
- `allow_nan=True` emits a bare `NaN` token. RFC 8259 section 6 is
  explicit that Infinity and NaN "are not permitted", so the browser's
  `JSON.parse` throws. That moves the breakage, it does not fix it.
- A Pydantic `response_model` does NOT work: Pydantic's own default for
  `ser_json_inf_nan` is already `'null'`, but FastAPI never consults model
  config on this path (`serialize_response` -> `jsonable_encoder` ->
  `JSONResponse.render`). See fastapi#11821.
- `ORJSONResponse` does render NaN as null, but adds a compiled dependency.

CRITICALLY: THIS IS A DISPLAY-LAYER FIX ONLY
--------------------------------------------
`sector_analysis.get_sector_analysis` and `quant_model.get_quant_model_signal`
are shared between this API and the Layer-1 trading pipeline
(`backend/agents/orchestrator.py:1261` and `:1271-1273`). A sanitiser HERE
cannot change a trading input, because the orchestrator never traverses
Starlette. Fixing the non-finite values at their source in
`backend/tools/` WOULD change trading inputs, and is deliberately NOT done
here -- it belongs to step 80.27.

Do not mistake a green Signals page for a fixed pipeline. The same NaN that
500s this endpoint is silently laundered into a confident NEUTRAL trading
verdict on the pipeline path; that is 80.27, and it is still open.
"""
from __future__ import annotations

import math
from typing import Any

from starlette.responses import JSONResponse

__all__ = ["sanitize_non_finite", "NaNSafeJSONResponse"]


def sanitize_non_finite(obj: Any) -> Any:
    """Recursively replace non-finite floats with None.

    The value is NULLED, never dropped and never coerced to a number:

    - Dropping the key would hide a data outage behind a missing field, and
      a caller doing `.get(k, 0)` would silently read a zero.
    - Coercing to 0.0 is worse -- it turns "no data" into a real number that
      flows into displays and comparisons. `backend/slack_bot/formatters.py:663`
      does exactly that; it is NOT the precedent to copy here.

    So `{"1mo": nan}` becomes `{"1mo": None}` -> `{"1mo": null}` on the wire,
    which is what `JSON.stringify(NaN)` produces in every browser and what
    Pydantic's own `ser_json_inf_nan` default emits.

    Note on ordering: `isinstance(v, float)` MUST be tested before calling
    `math.isfinite`, which raises TypeError on None and on non-numeric types.
    `np.float64` is a subclass of `float`, so the plain isinstance check
    catches numpy scalars too -- `np.isfinite`/`pd.isna` are deliberately NOT
    used (the former raises on non-numeric input, the latter returns an array
    for list input, and neither is safe on an arbitrary payload).

    `bool` is a subclass of `int`, not `float`, so booleans pass through
    untouched.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: sanitize_non_finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_non_finite(v) for v in obj]
    if isinstance(obj, tuple):
        # JSON has no tuple; Starlette would render it as an array anyway.
        return [sanitize_non_finite(v) for v in obj]
    return obj


class NaNSafeJSONResponse(JSONResponse):
    """`JSONResponse` that nulls non-finite floats instead of 500ing.

    Applied as `default_response_class` on the signals router ONLY, so every
    signal sub-endpoint is covered rather than just the one that was reported.
    It is deliberately NOT applied app-wide: that would silently re-encode
    ~70 unrelated routes for no benefit.
    """

    def render(self, content: Any) -> bytes:
        return super().render(sanitize_non_finite(content))
