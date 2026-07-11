# phase-69.2 tracked follow-ons

## FO-69.2-A — True fracdiff-at-predict (per-ticker time-series FFD)  [condition of C4 acceptance]

**Raised by**: 69.2 Q/A (workflow structured-output verdict, 2026-07-11) as the mandatory
tracked follow-on that conditions the ACCEPT-ON-INTENT ruling for criterion 69.2-C4
("fracdiff transform and NaN-fill applied identically at predict as train").

**What 69.2 delivered** (harm-reduction, offline, do-no-harm):
- The NaN-fill half is now literally identical: `_build_training_data` persists the
  post-fracdiff per-feature TRAIN medians (`self._train_feature_medians`) and
  `_build_predict_features` applies them at predict (was `fillna(0)`) — the sklearn
  fit-on-train / apply-at-predict discipline.
- Non-stationary features at predict are placed on the train (fracdiff'd) scale via the
  train median instead of raw price/market-cap levels (fixes the register's concrete
  "model fed raw levels it never trained on" defect).

**Honest limitation (documented in `_build_predict_features` docstring)**: the windowed FFD
convolution is NOT applied identically — a single cross-sectional predict row has no series
window, so non-stationary features are median-NEUTRALIZED (a constant train median) at predict
and carry no cross-sectional signal. This is disclosed neutralization, not equivalence.

**Why not fixed in 69.2**: the true fix (per-ticker time-series FFD with the persisted
data-independent weight vector) belongs in the feature builder `build_feature_vector`
(`historical_data.py`), which is LIVE-ADJACENT (imported by `backend/agents/mcp_servers/data_server.py`).
Touching it breaches 69.2's zero-live-surface constraint. The in-scope literal alternative
(neutralize the interleaved fracdiff at TRAIN too) would materially change the incumbent
model's features and cannot be re-validated under the historical_macro freeze.

**True fix (future live-adjacent step, gated on historical_macro un-freeze for re-validation)**:
1. Move the fractional-differentiation of the 5 `_NON_STATIONARY` features INTO
   `build_feature_vector` so both train (`:583`) and predict (`:789`) call sites obtain
   already-fracdiff'd features from the identical code path (fetch the ticker's own trailing
   price/fundamental series, apply the fixed-width FFD weights, take the last value).
2. Remove the interleaved-matrix fracdiff at `backtest_engine.py:628-637` (statistically
   dubious — mixes tickers) once (1) supplies per-ticker fracdiff'd features.
3. Re-validate the incumbent under the corrected features (needs the operator un-freeze token).

**Owner / routing**: file as a phase-69.4 hand-off seed (63.3-style) AND a masterplan note;
do NOT execute here.
