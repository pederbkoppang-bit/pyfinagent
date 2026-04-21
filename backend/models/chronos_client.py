"""phase-8.2 Chronos-Bolt shadow-logged forecast client.

Thin wrapper over Amazon's Chronos-Bolt foundation model
(`amazon/chronos-bolt-small`, 48M params). Shadow-only in this phase;
forecasts are logged to `pyfinagent_data.ts_forecast_shadow_log` via the
same DDL as `timesfm_client.py` (the `model_name` column distinguishes
producers).

API shape (per research brief):
    pipeline = BaseChronosPipeline.from_pretrained(model_name, device_map="cpu")
    result = pipeline.predict(context=torch.Tensor, prediction_length=N)
    # result.shape == [1, num_quantiles, N]
    median_point = result[0, result.shape[1] // 2, :]

Python-version note: `chronos-forecasting` + `torch` not installed in the
repo's 3.14 venv. Both imports are lazy; `_get_pipeline()` returns None on
ImportError and all public methods fail-open.

Fail-open everywhere. ASCII-only.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Iterable

logger = logging.getLogger(__name__)

_MODEL_NAME = "amazon/chronos-bolt-small"
_SHADOW_TABLE = "ts_forecast_shadow_log"
_DEFAULT_CONTEXT_LENGTH = 512
_DEFAULT_HORIZON_LENGTH = 20


class ChronosBoltClient:
    """Lazy-loaded Chronos-Bolt forecast client.

    Parameters mirror TimesFMClient: ``context_length``, ``horizon_length``,
    ``model_name`` (defaults to the Small checkpoint).
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
        self._pipeline: Any = None  # lazy

    def _get_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        try:
            from chronos import BaseChronosPipeline  # type: ignore[import-not-found]
        except Exception as exc:
            logger.warning("chronos_client: chronos-forecasting absent (%r)", exc)
            return None
        try:
            import torch  # type: ignore[import-not-found]  # noqa: F401
        except Exception as exc:
            logger.warning("chronos_client: torch absent (%r)", exc)
            return None
        try:
            pipeline = BaseChronosPipeline.from_pretrained(self.model_name, device_map="cpu")
            self._pipeline = pipeline
            return pipeline
        except Exception as exc:
            logger.warning("chronos_client: pipeline load fail-open (%r)", exc)
            return None

    def _median_from_result(self, result: Any) -> list[float]:
        """Extract the median-quantile point forecast from a predict() result.

        The real pipeline returns a torch tensor; the test stub returns a
        numpy array. Both support `.shape` + numeric indexing.
        """
        try:
            num_quantiles = int(result.shape[1])
            mid = num_quantiles // 2
            values = result[0, mid, :]
            return [float(x) for x in values]
        except Exception as exc:
            logger.warning("chronos_client: result slice fail-open (%r)", exc)
            return []

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
        pipeline = self._get_pipeline()
        if pipeline is None:
            return []
        try:
            import torch  # type: ignore[import-not-found]
        except Exception as exc:
            logger.warning("chronos_client: torch absent in forecast (%r)", exc)
            return []
        try:
            ctx = torch.tensor(series, dtype=torch.float32)
            result = pipeline.predict(context=ctx, prediction_length=h)
            return self._median_from_result(result)
        except Exception as exc:
            logger.warning("chronos_client: forecast fail-open (%r)", exc)
            return []

    def forecast_batch(
        self,
        tickers: dict[str, Iterable[float]],
        *,
        horizon: int = _DEFAULT_HORIZON_LENGTH,
    ) -> dict[str, list[float]]:
        """Per-ticker single forecasts.

        Chronos-Bolt supports true batched input via a 2D tensor, but the
        shadow pilot runs daily over hundreds of tickers on CPU -- sequential
        single-series calls keep the scaffold simple and avoid padding.
        """
        if not tickers:
            return {}
        pipeline = self._get_pipeline()
        if pipeline is None:
            return {t: [] for t in tickers}
        out: dict[str, list[float]] = {}
        for t, series in tickers.items():
            s = [float(x) for x in (series or [])]
            if len(s) < 2:
                out[t] = []
                continue
            out[t] = self.forecast(s, horizon=horizon)
        return out

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
        """Append one shadow-log row. Fail-open. Table DDL owned by phase-8.3."""
        try:
            from google.cloud import bigquery  # type: ignore[import-not-found]
        except Exception as exc:
            logger.warning("chronos_client: google-cloud-bigquery absent (%r)", exc)
            return False
        try:
            from backend.config.settings import get_settings

            s = get_settings()
            proj = project or s.gcp_project_id or ""
            ds = dataset or getattr(s, "bq_dataset_observability", None) or "pyfinagent_data"
        except Exception as exc:
            logger.warning("chronos_client: settings load fail-open (%r)", exc)
            return False
        try:
            client = bigquery.Client(project=proj) if proj else bigquery.Client()
        except Exception as exc:
            logger.warning("chronos_client: bigquery.Client init fail-open (%r)", exc)
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
                logger.warning("chronos_client: shadow_log insert errors: %s", errors[:1])
                return False
            return True
        except Exception as exc:
            logger.warning("chronos_client: shadow_log fail-open (%r)", exc)
            return False


__all__ = ["ChronosBoltClient"]
