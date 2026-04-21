"""phase-8.1 TimesFM shadow-logged forecast client.

Thin wrapper over Google's TimesFM foundation model. Shadow-only in this phase:
forecasts are logged to `pyfinagent_data.ts_forecast_shadow_log` but NOT fed to
the live trading pipeline. Promotion/rejection decided at phase-8.4.

Model: `google/timesfm-2.5-200m-pytorch` (Sept 2025 release, per research brief).

Python-version note: the `timesfm` PyPI package requires Python >=3.10,<3.12.
This repo's `.venv` is Python 3.14. The client therefore imports `timesfm`
lazily inside method bodies and fails open (returns `[]` / `{}`) when the
package is absent. Phase-8.3 will revisit once a 3.11 sub-environment is
provisioned or an alternate runtime (Docker / Cloud Run) is chosen.

Fail-open everywhere. ASCII-only.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Iterable

logger = logging.getLogger(__name__)

_MODEL_NAME = "google/timesfm-2.5-200m-pytorch"
_SHADOW_TABLE = "ts_forecast_shadow_log"
_DEFAULT_CONTEXT_LENGTH = 512
_DEFAULT_HORIZON_LENGTH = 20


class TimesFMClient:
    """Lazy-loaded TimesFM forecast client.

    Parameters
    ----------
    context_length : int
        Number of historical points fed to the model. Default 512 (half of
        the 2.5 model's 1024 max).
    horizon_length : int
        Default horizon; overridable per call. Default 20.
    model_name : str | None
        HF hub checkpoint. Default `google/timesfm-2.5-200m-pytorch`.
    """

    def __init__(
        self,
        context_length: int = _DEFAULT_CONTEXT_LENGTH,
        horizon_length: int = _DEFAULT_HORIZON_LENGTH,
        model_name: str | None = None,
    ) -> None:
        self.context_length = context_length
        self.horizon_length = horizon_length
        self.model_name = model_name or _MODEL_NAME
        self._model: Any = None  # lazy

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            import timesfm  # type: ignore[import-not-found]
        except Exception as exc:
            logger.warning("timesfm_client: package absent (%r)", exc)
            return None
        try:
            model = timesfm.TimesFm_2p5_200M_torch.from_pretrained(self.model_name)
            model.compile(
                timesfm.ForecastConfig(
                    max_context=self.context_length,
                    max_horizon=self.horizon_length,
                    normalize_inputs=True,
                    use_continuous_quantile_head=True,
                )
            )
            self._model = model
            return model
        except Exception as exc:
            logger.warning("timesfm_client: model load fail-open (%r)", exc)
            return None

    def forecast(
        self,
        ts: Iterable[float],
        *,
        horizon: int | None = None,
    ) -> list[float]:
        """Forecast `horizon` points for a single series. Fail-open to []."""
        series = [float(x) for x in (ts or [])]
        if len(series) < 2:
            return []
        h = int(horizon if horizon is not None else self.horizon_length)
        if h <= 0:
            return []
        model = self._get_model()
        if model is None:
            return []
        try:
            import numpy as np  # type: ignore[import-not-found]
        except Exception as exc:
            logger.warning("timesfm_client: numpy absent (%r)", exc)
            return []
        try:
            point, _ = model.forecast(horizon=h, inputs=[np.asarray(series, dtype=float)])
            return [float(x) for x in (point[0] if len(point) else [])]
        except Exception as exc:
            logger.warning("timesfm_client: forecast fail-open (%r)", exc)
            return []

    def forecast_batch(
        self,
        tickers: dict[str, Iterable[float]],
        *,
        horizon: int = _DEFAULT_HORIZON_LENGTH,
    ) -> dict[str, list[float]]:
        """Forecast many tickers in one model call. Fail-open per ticker."""
        if not tickers:
            return {}
        model = self._get_model()
        if model is None:
            return {t: [] for t in tickers}
        try:
            import numpy as np  # type: ignore[import-not-found]
        except Exception as exc:
            logger.warning("timesfm_client: numpy absent (%r)", exc)
            return {t: [] for t in tickers}
        clean: list[tuple[str, list[float]]] = []
        for t, series in tickers.items():
            s = [float(x) for x in (series or [])]
            if len(s) >= 2:
                clean.append((t, s))
        if not clean:
            return {t: [] for t in tickers}
        try:
            inputs = [np.asarray(s, dtype=float) for _, s in clean]
            point, _ = model.forecast(horizon=int(horizon), inputs=inputs)
            out: dict[str, list[float]] = {}
            for idx, (t, _s) in enumerate(clean):
                try:
                    out[t] = [float(x) for x in point[idx]]
                except Exception:
                    out[t] = []
            # Tickers that were filtered out (too short) get [].
            for t in tickers:
                out.setdefault(t, [])
            return out
        except Exception as exc:
            logger.warning("timesfm_client: batch forecast fail-open (%r)", exc)
            return {t: [] for t in tickers}

    def shadow_log(
        self,
        ticker: str,
        as_of_date: str,
        horizon: int,
        forecast_values: list[float],
        observed_values: list[float] | None = None,
        *,
        project: str | None = None,
        dataset: str | None = None,
    ) -> bool:
        """Append a single shadow-log row to `ts_forecast_shadow_log`. Fail-open.

        Table creation is NOT attempted here; a separate migration or phase-8.3
        smoketest creates the table. If the table is absent the insert errors
        silently and this returns False.
        """
        try:
            from google.cloud import bigquery  # type: ignore[import-not-found]
        except Exception as exc:
            logger.warning("timesfm_client: google-cloud-bigquery absent (%r)", exc)
            return False
        try:
            from backend.config.settings import get_settings

            s = get_settings()
            proj = project or s.gcp_project_id or ""
            ds = dataset or getattr(s, "bq_dataset_observability", None) or "pyfinagent_data"
        except Exception as exc:
            logger.warning("timesfm_client: settings load fail-open (%r)", exc)
            return False
        try:
            client = bigquery.Client(project=proj) if proj else bigquery.Client()
        except Exception as exc:
            logger.warning("timesfm_client: bigquery.Client init fail-open (%r)", exc)
            return False
        table_ref = f"{proj}.{ds}.{_SHADOW_TABLE}" if proj else f"{ds}.{_SHADOW_TABLE}"
        row = {
            "model_name": self.model_name,
            "ticker": ticker,
            "as_of_date": as_of_date,
            "horizon": int(horizon),
            "forecast_values": list(forecast_values or []),
            "observed_values": list(observed_values) if observed_values else None,
            "logged_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            errors = client.insert_rows_json(table_ref, [row])
            if errors:
                logger.warning("timesfm_client: shadow_log insert errors: %s", errors[:1])
                return False
            return True
        except Exception as exc:
            logger.warning("timesfm_client: shadow_log fail-open (%r)", exc)
            return False


__all__ = ["TimesFMClient"]
