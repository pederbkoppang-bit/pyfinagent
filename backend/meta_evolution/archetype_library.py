"""phase-10.7.3 Algorithm Discovery archetype seed library.

Canonical seed corpus for the meta-evolution loop. Six archetypes that
mirror `STRATEGY_REGISTRY` from `backend/backtest/backtest_engine.py:32`
plus one forward-declaration (`sentiment_event_driven`).

Design references (per phase-10.7.3 research brief):
- QuantEvolve (arXiv 2510.18569v1, 2025): C+1 island seeding -- one simple
  representative seed per category.
- AlphaEvolve (DeepMind blog 2025) + DeepEvolve (arXiv 2510.06056, 2025):
  single best-known seed; diversity emerges from evolution.

This module is pure data: dataclass + tuple constants + a small lookup
helper. No I/O, no logging, no external deps. The directive-rewriter
(phase-10.7.2) and future cron-budget allocator (phase-10.7.4) read
`ARCHETYPES` to plan mutations and slot allocations.

Structure mirrors `backend/meta_evolution/alpha_velocity.py` (10.7.1)
and `backend/meta_evolution/directive_rewriter.py` (10.7.2):
@dataclass + module-level tuple + factory + zero I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

ALLOWED_REGIMES: frozenset[str] = frozenset(
    {"ALL", "BULL", "BEAR", "RANGING", "NEUTRAL", "VOLATILE", "EASING", "HIKING"}
)

IMPLEMENTED_STRATEGY_IDS: frozenset[str] = frozenset(
    {"triple_barrier", "quality_momentum", "mean_reversion", "factor_model", "meta_label", "blend"}
)


@dataclass(frozen=True)
class Archetype:
    """One seed archetype for the algorithm-discovery evolutionary loop.

    Fields:
      strategy_id: snake_case key. If `is_implemented=True`, MUST appear
        in `IMPLEMENTED_STRATEGY_IDS` (mirror of STRATEGY_REGISTRY +
        the `blend` optimizer-only strategy). If False, this is a
        forward-declaration; the backtest engine will fall back to
        `triple_barrier` until a label method is added.
      name: human-readable, title-case.
      description: 1-3 sentence academic basis + signal logic.
      default_params: subset of keys from `quant_optimizer._PARAM_BOUNDS`.
        Used as the optimizer warm-start point. MUST be non-empty
        (>= 2 keys) per the QuantEvolve "functional but improvable"
        seed-quality requirement.
      expected_regime: one of `ALLOWED_REGIMES`. Filters which macro
        regime the alpha-velocity comparison applies to (10.7.1 uses
        this to avoid cross-regime score pollution).
      directive_template: short jinja-ish template the 10.7.2 rewriter
        substitutes. MUST contain at least one of `{name}` or
        `{strategy_id}` so the LLM rewriter can specialize the prompt.
      is_implemented: True if STRATEGY_REGISTRY has a real label method;
        False = forward-declaration archetype (silent fallback risk).
    """

    strategy_id: str
    name: str
    description: str
    default_params: dict[str, object] = field(default_factory=dict)
    expected_regime: str = "ALL"
    directive_template: str = ""
    is_implemented: bool = True

    def __post_init__(self) -> None:
        if not self.strategy_id or not self.strategy_id.strip():
            raise ValueError("Archetype.strategy_id must be non-empty")
        if not self.name or not self.name.strip():
            raise ValueError(f"Archetype.name must be non-empty (id={self.strategy_id})")
        if not self.description or not self.description.strip():
            raise ValueError(
                f"Archetype.description must be non-empty (id={self.strategy_id})"
            )
        if self.expected_regime not in ALLOWED_REGIMES:
            raise ValueError(
                f"Archetype.expected_regime={self.expected_regime!r} not in "
                f"ALLOWED_REGIMES={sorted(ALLOWED_REGIMES)} (id={self.strategy_id})"
            )
        if "{name}" not in self.directive_template and "{strategy_id}" not in self.directive_template:
            raise ValueError(
                f"Archetype.directive_template must contain {{name}} or "
                f"{{strategy_id}} placeholder (id={self.strategy_id})"
            )
        if self.is_implemented and self.strategy_id not in IMPLEMENTED_STRATEGY_IDS:
            raise ValueError(
                f"Archetype.is_implemented=True but strategy_id={self.strategy_id!r} "
                f"not in IMPLEMENTED_STRATEGY_IDS={sorted(IMPLEMENTED_STRATEGY_IDS)}. "
                f"This would cause silent fallback to triple_barrier in the engine."
            )


ARCHETYPES: tuple[Archetype, ...] = (
    Archetype(
        strategy_id="triple_barrier",
        name="Triple Barrier",
        description=(
            "Lopez de Prado AFML Ch. 3 path-dependent labelling. Three "
            "barriers (take-profit, stop-loss, max holding period) define "
            "the label; first-touch wins. Regime-agnostic; the backbone "
            "label method for all other strategies."
        ),
        default_params={
            "tp_pct": 10.0,
            "sl_pct": 7.0,
            "holding_days": 90,
            "vol_barrier_multiplier": 0.0,
        },
        expected_regime="ALL",
        directive_template=(
            "Propose a variant of the {name} archetype (strategy_id={strategy_id}) "
            "that improves Sharpe while preserving the path-dependent labelling "
            "semantics. Focus on volatility-adjusted barriers, holding-horizon "
            "tuning, or label-quality filters."
        ),
        is_implemented=True,
    ),
    Archetype(
        strategy_id="quality_momentum",
        name="Quality Momentum",
        description=(
            "Asness, Frazzini, Pedersen (2019) -- combine 12-1 month price "
            "momentum with quality factors (profitability, leverage, earnings "
            "stability). Goes long high-quality high-momentum stocks. Works "
            "best in trending markets where quality compounds."
        ),
        default_params={
            "tp_pct": 15.0,
            "sl_pct": 8.0,
            "holding_days": 120,
            "momentum_weight": 0.6,
        },
        expected_regime="BULL",
        directive_template=(
            "Propose a variant of the {name} archetype (strategy_id={strategy_id}) "
            "that strengthens the quality-screen + momentum-rank composite. "
            "Consider Piotroski F-score components, alternative momentum windows, "
            "or sector-neutral ranking."
        ),
        is_implemented=True,
    ),
    Archetype(
        strategy_id="mean_reversion",
        name="Mean Reversion",
        description=(
            "Lo & MacKinlay (1990) -- short-horizon (1-4 week) reversal of "
            "price moves. Goes long oversold (low RSI, below SMA) and exits "
            "on mean reversion. Holding period MUST be short; use "
            "mr_holding_days, not the shared holding_days."
        ),
        default_params={
            "tp_pct": 5.0,
            "sl_pct": 4.0,
            "mr_holding_days": 10,
            "holding_days": 30,
        },
        expected_regime="RANGING",
        directive_template=(
            "Propose a variant of the {name} archetype (strategy_id={strategy_id}) "
            "that exploits short-horizon reversal. Tune mr_holding_days, RSI "
            "threshold, or distance-from-SMA gates -- but never extend the "
            "holding period above 30 days (the strategy decays past that)."
        ),
        is_implemented=True,
    ),
    Archetype(
        strategy_id="factor_model",
        name="Factor Model",
        description=(
            "Fama-French five-factor model (market, size, value, profitability, "
            "investment). Ranks the universe by factor-loaded composite score "
            "and goes long top decile. Long-horizon, regime-neutral, low-turnover."
        ),
        default_params={
            "tp_pct": 20.0,
            "sl_pct": 12.0,
            "holding_days": 180,
            "max_positions": 25,
        },
        expected_regime="NEUTRAL",
        directive_template=(
            "Propose a variant of the {name} archetype (strategy_id={strategy_id}) "
            "that adjusts factor weights or adds a quality / low-volatility "
            "tilt. Preserve the long-horizon, low-turnover spirit."
        ),
        is_implemented=True,
    ),
    Archetype(
        strategy_id="meta_label",
        name="Meta-Label Filtering",
        description=(
            "Lopez de Prado AFML Ch. 3 secondary classifier. Primary signal "
            "(from triple_barrier) generates direction; meta-model filters "
            "low-confidence trades. Current state: stub -- engine reuses "
            "_compute_triple_barrier_label; full two-stage training not yet "
            "wired. Optimizer should treat this as forward-improvable."
        ),
        default_params={
            "tp_pct": 10.0,
            "sl_pct": 7.0,
            "holding_days": 90,
            "min_samples_leaf": 20,
        },
        expected_regime="ALL",
        directive_template=(
            "Propose a variant of the {name} archetype (strategy_id={strategy_id}) "
            "that moves toward a true two-stage classifier. Either propose a "
            "secondary-model parameter, a confidence-threshold gate, or a "
            "sample-weighting scheme."
        ),
        is_implemented=True,
    ),
    Archetype(
        strategy_id="sentiment_event_driven",
        name="Sentiment / Event-Driven",
        description=(
            "Post-Earnings Announcement Drift (PEAD) + sentiment overlay. "
            "FinBERT or LLM-scored news classifies events; positions held "
            "through the drift window (3-10 days). Forward-declaration: NOT "
            "yet in STRATEGY_REGISTRY -- the engine will silently fall back "
            "to triple_barrier until a label method ships. Tracked here so "
            "the optimizer can plan toward it."
        ),
        default_params={
            "tp_pct": 4.0,
            "sl_pct": 3.0,
            "holding_days": 30,
            "mr_holding_days": 5,
        },
        expected_regime="VOLATILE",
        directive_template=(
            "Propose a forward-design for the {name} archetype "
            "(strategy_id={strategy_id}). Specify the event-trigger source "
            "(earnings, 8-K, news), the sentiment-scoring pipeline, and the "
            "label horizon. Output should be a research directive, not a "
            "live param set, until is_implemented flips to True."
        ),
        is_implemented=False,
    ),
)


def get_archetype(strategy_id: str) -> Optional[Archetype]:
    """Lookup helper. Returns None if `strategy_id` is unknown."""
    for arch in ARCHETYPES:
        if arch.strategy_id == strategy_id:
            return arch
    return None
