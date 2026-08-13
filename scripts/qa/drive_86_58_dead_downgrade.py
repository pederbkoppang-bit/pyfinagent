"""86.58: drive decide_trades to measure the signal_downgrade rule, flags OFF *and ON*.

CYCLE-2 REWRITE (Q/A wf_b127735e-55b returned FAIL). The previous version
asserted both flags were False and ABORTED otherwise, then hand-set
`pos.recommendation='BUY'` as a stand-in for what the fix would write. The
production flag-read therefore executed in ZERO cells, and the blast-radius
number it produced INVERTED under real measurement: published 1-of-1, actual
0-of-2. An assertion pinning the subject to its default is the signature of a
proxy -- see auto-memory `feedback_assert_the_property_not_a_proxy`.

This version ENTERS the condition: it builds a flags-ON Settings via
model_copy and drives the same call. It never writes, never promotes a flag on
disk, and never touches the live book -- the override is in-process only.

Run:
  source .venv/bin/activate && PYTHONPATH=$(pwd) python scripts/qa/drive_86_58_dead_downgrade.py
"""
from backend.config.settings import Settings
from backend.services.portfolio_manager import (
    decide_trades, _BUY_RECS, _DOWNGRADE_RECS, _SELL_RECS,
)

TICKER = "NTAP"
FLAGS = ("paper_position_recommendation_fix_enabled",
         "paper_recommendation_vocab_fix_enabled")


def make_position(recommendation: str) -> dict:
    # stop_loss_price far BELOW current_price so the stop-loss branch (which
    # precedes the rule and `continue`s) cannot pre-empt it and make a cell
    # green for the wrong reason.
    return {"ticker": TICKER, "quantity": 5.35, "avg_entry_price": 177.85,
            "current_price": 201.91, "market_value": 1079.54,
            "cost_basis": 950.90, "stop_loss_price": 100.00,
            "recommendation": recommendation, "sector": "Technology"}


def make_analysis(rec: str) -> dict:
    return {"ticker": TICKER, "recommendation": rec,
            "analysis_date": "2026-08-13T00:00:00Z", "final_score": 5.0}


def run(pos_rec: str, fresh_rec: str, settings):
    orders = decide_trades(
        current_positions=[make_position(pos_rec)],
        candidate_analyses=[], holding_analyses=[make_analysis(fresh_rec)],
        portfolio_state={"nav": 23900.18, "cash": 22820.64,
                         "positions_value": 1079.54, "position_count": 1},
        settings=settings)
    return [(o.ticker, o.action, o.reason) for o in orders]


def fired(orders):
    return any(r == "signal_downgrade" for _, _, r in orders)


def main():
    off = Settings()
    for f in FLAGS:
        if not hasattr(off, f):
            print(f"ABORT: {f} missing from Settings"); return 2
    print("=== FLAG STATE ===")
    print("  OFF cells use the real defaults:")
    for f in FLAGS:
        print(f"    {f} = {getattr(off, f)}")
    # ENTER THE CONDITION. In-process only -- writes nothing, promotes nothing.
    on = off.model_copy(update={f: True for f in FLAGS})
    print("  ON cells use an in-process override (no .env write, no promotion):")
    for f in FLAGS:
        print(f"    {f} = {getattr(on, f)}")
    assert all(getattr(on, f) is True for f in FLAGS), "override did not take"
    assert all(getattr(off, f) is False for f in FLAGS), "defaults are not OFF"

    print("\n=== VOCABULARY (from source) ===")
    print(f"  _BUY_RECS={sorted(_BUY_RECS)} _DOWNGRADE_RECS={sorted(_DOWNGRADE_RECS)} "
          f"_SELL_RECS={sorted(_SELL_RECS)}")
    print(f"  DERIVED: sell_signal fires first and continues, so the ONLY fresh rec that can")
    print(f"           reach signal_downgrade is _DOWNGRADE_RECS - _SELL_RECS = "
          f"{sorted(_DOWNGRADE_RECS - _SELL_RECS)}")

    cells = {}
    print("\n=== FLAGS OFF ===")
    for name, pos in (("A", "new_buy_signal"), ("B", "BUY"), ("C", "swap_buy")):
        cells[name] = run(pos, "HOLD", off)
        print(f"  {name}  pos={pos!r:18} fresh='HOLD' -> {cells[name]}  fired={fired(cells[name])}")
    cells["D"] = run("BUY", "SELL", off)
    print(f"  D  pos='BUY' fresh='SELL' -> {cells['D']}  (sell_signal PRE-EMPTS; "
          f"testing with SELL would prove nothing)")

    print("\n=== FLAGS ON (the condition criterion 3 names) ===")
    for name, pos in (("E", "new_buy_signal"), ("F", "BUY"), ("G", "swap_buy")):
        cells[name] = run(pos, "HOLD", on)
        print(f"  {name}  pos={pos!r:18} fresh='HOLD' -> {cells[name]}  fired={fired(cells[name])}")

    # DISCRIMINATION CONTROL: a value that is dead OFF and live ON proves the
    # probe actually reads flag state rather than ignoring it.
    d_off, d_on = run("Strong Buy", "HOLD", off), run("Strong Buy", "HOLD", on)
    print(f"\n=== DISCRIMINATION CONTROL: pos='Strong Buy', fresh='HOLD' ===")
    print(f"  flags OFF -> {d_off}  fired={fired(d_off)}")
    print(f"  flags ON  -> {d_on}  fired={fired(d_on)}")

    print("\n=== VERDICT ===")
    ok = True
    if not fired(cells["B"]):
        print("  CONTROL B RED -> every OFF cell is UNSCORABLE."); ok = False
    if not fired(cells["F"]):
        print("  CONTROL F RED -> every ON cell is UNSCORABLE."); ok = False
    if not (not fired(d_off) and fired(d_on)):
        print("  DISCRIMINATION CONTROL FAILED -> the probe does not read flag state;"
              " no ON/OFF comparison below is trustworthy."); ok = False
    if not ok:
        return 1
    print("  Controls GREEN: B (OFF) and F (ON) both fire; the discrimination control")
    print("  is dead OFF and live ON, so the probe demonstrably reads flag state.")
    print(f"  A new_buy_signal OFF dead: {not fired(cells['A'])}   "
          f"E new_buy_signal ON dead: {not fired(cells['E'])}")
    print(f"  C swap_buy       OFF dead: {not fired(cells['C'])}   "
          f"G swap_buy       ON dead: {not fired(cells['G'])}")
    print("\n  MEASURED: a held row carrying an order REASON is dead under BOTH flag")
    print("  states. Flipping the flag does NOT revive the rule for rows already on")
    print("  disk, because flag-ON _resolve_rec maps the value to _UNRECOGNISED_REC")
    print("  (in none of the three sets) and the field is written only by execute_buy.")
    print("  Blast radius at PROMOTION TIME is therefore 0 for existing rows;")
    print("  exposure begins at the next execute_buy that rewrites the field.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
