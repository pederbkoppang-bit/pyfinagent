"""phase-80.31 -- the anomaly detector's four arrays must share one index.

`anomaly_detector` used to build close/volume/high/low with four INDEPENDENT
`.dropna()` calls. Column-wise dropna destroys positional correspondence the
moment two columns have different NaN patterns -- and `Volume` is `int64`,
which CANNOT hold NaN, so its dropna is a structural no-op. The misalignment
was therefore GUARANTEED, not probabilistic:

    AAPL 1y: 251 raw rows -> close/high/low/open 250, volume 251

`close[-1]` was day T-1 while `volume[-5:]` spanned T-4..T, so the
`volume_5d_vs_60d` anomaly reported a precise-looking mean/std/z triple over
a one-session-misaligned series.

The malformed bar is a COMPLETED session (Fri 2026-07-24), not a
still-forming one -- upstream yfinance issue #2622.

Mutation-tested; matrix in handoff/current/experiment_results_80.31.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.agents.info_gap import _has_non_finite
from backend.tools import anomaly_detector


def _yfinance_shaped_frame(rows: int = 120, with_bad_tail: bool = True) -> pd.DataFrame:
    """A frame in the EXACT shape yfinance returns.

    Trailing row: NaN Open/High/Low/Close, but a REAL int64 Volume. That
    combination is the whole defect -- it is why yfinance's own `keepna`
    mask (`.all(axis=1)`) keeps the row, and why a per-column dropna
    silently shortens three arrays but not the fourth.
    """
    idx = pd.date_range("2026-01-01", periods=rows, freq="B")
    close = np.linspace(100.0, 130.0, rows)
    frame = pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            # int64 on purpose: an int64 column cannot hold NaN.
            "Volume": np.arange(1_000_000, 1_000_000 + rows, dtype="int64"),
        },
        index=idx,
    )
    if with_bad_tail:
        frame.loc[frame.index[-1], ["Open", "High", "Low", "Close"]] = np.nan
    return frame


class _FakeTicker:
    def __init__(self, *a, **k):
        self.frame = _FakeTicker.frame

    def history(self, *a, **k):
        return self.frame.copy()

    @property
    def info(self):
        return {"longName": "Fixture Corp", "sector": "Technology"}


def _run_with(frame, monkeypatch):
    _FakeTicker.frame = frame
    monkeypatch.setattr(anomaly_detector.yf, "Ticker", _FakeTicker)
    return anomaly_detector.get_anomaly_scan("AAPL")


# ── fixture pins: bind to the FIXTURE BUILDER, never to a library fact ──
# (80.1 cycle 1 shipped `assert not math.isfinite(float("nan"))` as a
#  "fixture pin"; it passed under the very mutation it claimed to guard.)


def test_fixture_reproduces_the_exact_yfinance_shape():
    f = _yfinance_shaped_frame()
    assert f["Close"].isna().sum() == 1, "fixture has no malformed bar"
    assert f["Open"].isna().sum() == 1
    assert f["Volume"].notna().all(), "Volume must stay complete"
    assert f["Volume"].dtype == "int64", (
        "Volume must be int64 -- a float column CAN hold NaN, which would "
        "make the per-column dropna symmetric and the defect unreproducible"
    )


def test_fixture_without_bad_tail_is_clean():
    f = _yfinance_shaped_frame(with_bad_tail=False)
    assert f["Close"].isna().sum() == 0


def test_the_defect_shape_is_real_percolumn_dropna_is_asymmetric():
    """Control: this is what the OLD code did, and why it broke."""
    f = _yfinance_shaped_frame()
    assert len(f["Close"].dropna()) == len(f) - 1
    assert len(f["Volume"].dropna()) == len(f)  # int64 -> dropna is a no-op
    assert len(f["Close"].dropna()) != len(f["Volume"].dropna())


# ── criterion 1: equal length AND index-aligned ────────────────────────


def test_module_arrays_are_equal_length_and_index_aligned():
    """CRITERION 1, literally: assert the four lengths are equal.

    Drives the PRODUCTION extractor `_aligned_ohlcv_arrays` directly, so a
    shift injected into any one of the four reads is caught. Asserting on
    arrays the TEST re-derives would be vacuous -- it would check the test's
    own arithmetic, not the module's.
    """
    frame = _yfinance_shaped_frame()
    cleaned = frame.dropna(subset=["Open", "High", "Low", "Close"])

    close, volume, high, low = anomaly_detector._aligned_ohlcv_arrays(cleaned)

    assert len(close) == len(volume) == len(high) == len(low), (
        f"lengths differ: close={len(close)} volume={len(volume)} "
        f"high={len(high)} low={len(low)}"
    )
    assert len(close) == len(cleaned)

    # Equal length is necessary, NOT sufficient: an off-by-one shift can keep
    # lengths equal. Assert real positional correspondence against the frame.
    for i in (0, 1, len(cleaned) // 2, len(cleaned) - 1):
        row = cleaned.iloc[i]
        assert close[i] == row["Close"], f"close desynced at {i}"
        assert volume[i] == row["Volume"], f"volume desynced at {i}"
        assert high[i] == row["High"], f"high desynced at {i}"
        assert low[i] == row["Low"], f"low desynced at {i}"


def test_malformed_session_is_absent_from_every_array():
    """The malformed bar must be gone from ALL four arrays, not just three.

    This is the original defect stated positively: `volume` used to retain
    the bar that `close`/`high`/`low` had lost.
    """
    frame = _yfinance_shaped_frame()
    bad_volume = frame["Volume"].iloc[-1]
    cleaned = frame.dropna(subset=["Open", "High", "Low", "Close"])

    close, volume, high, low = anomaly_detector._aligned_ohlcv_arrays(cleaned)

    assert len(close) == len(frame) - 1
    assert bad_volume not in set(volume.tolist()), (
        "the malformed session's volume survived into the volume array"
    )


def test_volume_anomaly_uses_the_metric_key_not_type(monkeypatch):
    """Pins the payload key the anomaly assertions depend on.

    An earlier version of this suite asserted on `a.get("type")`. The module
    writes `"metric"` (`_append_if_anomalous`), so that set was `{None}` and
    the assertion was true for EVERY implementation -- a tautology. This test
    exists so the key can never silently drift again.
    """
    frame = _yfinance_shaped_frame()
    out = _run_with(frame, monkeypatch)
    anomalies = out.get("anomalies", [])
    # WITHOUT this the loop below runs ZERO times and the test is vacuous --
    # exactly the failure mode it was written to close. Q/A cycle 2 proved it:
    # a module that never appends an anomaly passed the whole suite.
    assert anomalies, (
        "payload carried no anomalies -- the key pin never executed. "
        f"signal={out.get('signal')!r} summary={out.get('summary')!r}"
    )
    for a in anomalies:
        assert "metric" in a, f"anomaly dict has no 'metric' key: {sorted(a)}"
        assert "type" not in a


def test_a_nan_volume_suppresses_the_volume_anomaly_explicitly(monkeypatch, caplog):
    """Regression the Q/A found: my own fix introduced this path.

    `Volume` is out of the row-drop subset, so a float64 NaN volume now
    reaches the 60-day window. Before phase-80.31 the per-column dropna
    removed it. Left alone it makes std NaN, `_z` return None, and the
    anomaly vanish SILENTLY -- the phase-80.27 defect family. It must be
    skipped LOUDLY instead.
    """
    import logging

    frame = _yfinance_shaped_frame(with_bad_tail=False)
    frame["Volume"] = frame["Volume"].astype("float64")
    frame.loc[frame.index[-1], "Volume"] = np.nan

    with caplog.at_level(logging.WARNING, logger="backend.tools.anomaly_detector"):
        out = _run_with(frame, monkeypatch)

    metrics = {a.get("metric") for a in out.get("anomalies", [])}
    assert "volume_5d_vs_60d" not in metrics, (
        "a volume anomaly was scored against a NaN baseline"
    )
    assert any("volume window" in r.getMessage() for r in caplog.records), (
        "the suppression was silent -- no warning was logged"
    )
    assert _has_non_finite(out) is False


def test_a_high_only_nan_row_is_also_excluded(monkeypatch):
    """Closes the 'coincidental alignment' finding STRUCTURALLY.

    Today Yahoo happens to null all four OHLC columns together, so dropping
    on `Close` alone would remove the same row and look correct. Nothing
    enforces that. A future NaN-High/real-Close row would leave `high`
    desynced from `close` while every length still matched.

    This fixture nulls ONLY High. The row must still be excluded, which is
    true iff all four price columns are in the dropna subset. It fails if
    the subset is narrowed to `["Close"]`.

    The observable is the sufficiency guard: a 20-row frame whose last row
    has a NaN High leaves only 19 usable rows, so the module must reject it.
    If the subset were `["Close"]` the row would survive, 20 >= 20 would
    pass, and the scan would proceed on a desynced `high` array.
    """
    frame = _yfinance_shaped_frame(rows=20, with_bad_tail=False)
    frame.loc[frame.index[-1], "High"] = np.nan

    out = _run_with(frame, monkeypatch)
    assert out.get("signal") == "ERROR", (
        "a NaN-High row was not excluded -- `high` is desynced from `close` "
        f"and the scan ran anyway: {out.get('summary')}"
    )
    assert "insufficient" in out.get("summary", "").lower()


def test_payload_has_no_non_finite_values(monkeypatch):
    """Sharper than 'returns 200'.

    `anomaly` is HIGH criticality in `_SOURCE_CRITICALITY`, so under what
    phase-80.27 shipped, a single non-finite anywhere in this payload now
    makes it a critical_gap. Asserting the payload is finite is the real
    regression guard.
    """
    out = _run_with(_yfinance_shaped_frame(), monkeypatch)
    assert _has_non_finite(out) is False, out


def test_payload_is_json_serialisable(monkeypatch):
    import json

    out = _run_with(_yfinance_shaped_frame(), monkeypatch)
    blob = json.dumps(out, default=str)
    assert "NaN" not in blob


def test_clean_frame_still_works(monkeypatch):
    """No malformed bar at all -- the fix must not change the happy path."""
    out = _run_with(_yfinance_shaped_frame(with_bad_tail=False), monkeypatch)
    assert out.get("signal") != "ERROR"
    assert _has_non_finite(out) is False


# ── the dropna must run BEFORE the sufficiency guard ───────────────────


def test_sufficiency_guard_counts_usable_rows_not_raw_rows(monkeypatch):
    """A frame with 20 raw rows but only 19 usable ones must be rejected.

    If the row-wise dropna ran AFTER `len(hist) < 20`, this frame would pass
    the guard and then compute on 19 rows.
    """
    frame = _yfinance_shaped_frame(rows=20)
    out = _run_with(frame, monkeypatch)
    assert out.get("signal") == "ERROR"
    assert "insufficient" in out.get("summary", "").lower()


def test_end_to_end_alignment_invariant_is_enforced_at_runtime(monkeypatch, caplog):
    """CRITERION 3's literal target: an END-TO-END alignment test.

    The helper test above drives `_aligned_ohlcv_arrays` directly, so it
    catches a shift injected INTO the helper -- but not a change that
    bypasses the helper entirely (e.g. restoring the four per-column
    `.dropna()` calls in `get_anomaly_scan`). Q/A cycle 1 found exactly that
    hole: A1 was killed, but only by the dropna-ORDER and SUBSET-WIDTH tests,
    never by an alignment assertion.

    `get_anomaly_scan` now enforces the invariant at runtime, so this drives
    the real entry point and fails whenever the four arrays desync -- however
    that desync is introduced.
    """
    import logging

    frame = _yfinance_shaped_frame()
    with caplog.at_level(logging.ERROR, logger="backend.tools.anomaly_detector"):
        out = _run_with(frame, monkeypatch)

    assert out.get("signal") != "ERROR", (
        "the runtime alignment invariant tripped on a well-formed frame: "
        f"{out.get('summary')}"
    )
    assert not any("misaligned" in r.getMessage() for r in caplog.records)


def test_runtime_invariant_actually_catches_a_desync(monkeypatch, caplog):
    """The invariant must be able to FIRE, or it is decoration.

    Force the misalignment the old code produced -- a volume array one
    element longer than close -- and assert the module refuses to score
    rather than emitting confident numbers across mismatched sessions.
    """
    import logging

    real = anomaly_detector._aligned_ohlcv_arrays

    def desynced(hist):
        close, volume, high, low = real(hist)
        return close[:-1], volume, high[:-1], low[:-1]  # the original defect's shape

    monkeypatch.setattr(anomaly_detector, "_aligned_ohlcv_arrays", desynced)

    with caplog.at_level(logging.ERROR, logger="backend.tools.anomaly_detector"):
        out = _run_with(_yfinance_shaped_frame(), monkeypatch)

    assert out.get("signal") == "ERROR"
    assert "misaligned" in out.get("summary", "").lower()
    assert any("misaligned" in r.getMessage() for r in caplog.records)


def test_nan_volume_mid_window_is_also_caught(monkeypatch, caplog):
    """WARN (Q/A cycle 2): the guard's WINDOW was unpinned.

    The only NaN-volume fixture put the NaN on the last row, inside BOTH the
    5-day and 60-day windows -- so narrowing the guard from `volume[-60:]` to
    `volume[-5:]` survived, silently reintroducing the suppression the guard
    exists to prevent. This puts the NaN at ~-30: inside the 60-day baseline,
    outside the 5-day recent window.
    """
    import logging

    frame = _yfinance_shaped_frame(with_bad_tail=False)
    frame["Volume"] = frame["Volume"].astype("float64")
    frame.iloc[-30, frame.columns.get_loc("Volume")] = np.nan

    with caplog.at_level(logging.WARNING, logger="backend.tools.anomaly_detector"):
        out = _run_with(frame, monkeypatch)

    metrics = {a.get("metric") for a in out.get("anomalies", [])}
    assert "volume_5d_vs_60d" not in metrics
    assert any("volume window" in r.getMessage() for r in caplog.records), (
        "a NaN inside the 60-day baseline was suppressed SILENTLY"
    )
    assert _has_non_finite(out) is False


def test_a_same_length_roll_is_caught_by_the_invariant(monkeypatch, caplog):
    """WARN (Q/A cycle 2): the invariant enforced LENGTH, not position.

    `np.roll` keeps every length identical while desyncing every session, and
    it survived 14/14 with three metrics silently vanishing. The invariant now
    also spot-checks both endpoints against the source frame.
    """
    import logging

    real = anomaly_detector._aligned_ohlcv_arrays

    def rolled(hist):
        close, volume, high, low = real(hist)
        return np.roll(close, 1), volume, high, low

    monkeypatch.setattr(anomaly_detector, "_aligned_ohlcv_arrays", rolled)

    with caplog.at_level(logging.ERROR, logger="backend.tools.anomaly_detector"):
        out = _run_with(_yfinance_shaped_frame(), monkeypatch)

    assert out.get("signal") == "ERROR", (
        "a same-length roll desynced every session and was not caught"
    )
    assert any("misaligned" in r.getMessage() for r in caplog.records)
