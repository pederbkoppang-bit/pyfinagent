"""86.58 criterion 1: PROVE the dead rule by DRIVING decide_trades, not by reading source.

Every cell is paired with a positive control. A test that can only ever report
"no signal_downgrade" is vacuous -- cell B exists to show this harness CAN
observe the rule firing, so cell A's absence is evidence rather than silence.

Read-only: constructs in-memory dicts and calls decide_trades. Touches no BQ,
no live book, no flags.
"""
from backend.config.settings import Settings
from backend.services.portfolio_manager import (
    decide_trades, _BUY_RECS, _DOWNGRADE_RECS, _SELL_RECS,
)

TICKER = "NTAP"


def make_position(recommendation: str) -> dict:
    # Shaped like a real paper_positions row. stop_loss_price is set far BELOW
    # current_price so the stop-loss branch (which precedes and `continue`s)
    # cannot pre-empt the rule under test -- otherwise cell B would go green
    # for the wrong reason.
    return {
        "ticker": TICKER, "quantity": 5.35, "avg_entry_price": 177.85,
        "current_price": 201.91, "market_value": 1079.54,
        "cost_basis": 950.90, "stop_loss_price": 100.00,
        "recommendation": recommendation, "sector": "Technology",
    }


def make_analysis(rec: str) -> dict:
    return {"ticker": TICKER, "recommendation": rec,
            "analysis_date": "2026-08-13T00:00:00Z", "final_score": 5.0}


def run(pos_rec: str, fresh_rec: str, settings: Settings):
    orders = decide_trades(
        current_positions=[make_position(pos_rec)],
        candidate_analyses=[],
        holding_analyses=[make_analysis(fresh_rec)],
        portfolio_state={"nav": 23900.18, "cash": 22820.64,
                         "positions_value": 1079.54, "position_count": 1},
        settings=settings,
    )
    return [(o.ticker, o.action, o.reason) for o in orders]


def main():
    s = Settings()

    # Flags must be OFF for the claim under test. Assert, never assume -- and
    # if the attribute is missing entirely, say so rather than silently passing.
    print("=== FLAG STATE (asserted, not assumed) ===")
    for flag in ("paper_position_recommendation_fix_enabled",
                 "paper_recommendation_vocab_fix_enabled"):
        if not hasattr(s, flag):
            print(f"  {flag}: ATTRIBUTE ABSENT -- cannot assert OFF"); return 2
        v = getattr(s, flag)
        print(f"  {flag} = {v}")
        assert v is False, f"{flag} is {v}, expected False; aborting"

    print("\n=== VOCABULARY, read from source ===")
    print(f"  _BUY_RECS       = {sorted(_BUY_RECS)}")
    print(f"  _DOWNGRADE_RECS = {sorted(_DOWNGRADE_RECS)}")
    print(f"  _SELL_RECS      = {sorted(_SELL_RECS)}")
    reachable = _DOWNGRADE_RECS - _SELL_RECS
    print(f"  DERIVED: sell_signal fires first and `continue`s, so the ONLY fresh")
    print(f"           recommendation that can ever REACH signal_downgrade is:")
    print(f"           _DOWNGRADE_RECS - _SELL_RECS = {sorted(reachable)}")

    def has_downgrade(orders):
        return any(r == "signal_downgrade" for _, _, r in orders)

    print("\n=== CELLS ===")
    results = {}

    # A -- THE CLAIM. Held row carries an order REASON; fresh analysis HOLD.
    a = run("new_buy_signal", "HOLD", s)
    results["A"] = a
    print(f"  A  pos.recommendation='new_buy_signal', fresh='HOLD'  -> {a}")
    print(f"     signal_downgrade present? {has_downgrade(a)}   (claim: False)")

    # B -- POSITIVE CONTROL. Identical except the held row carries a real
    # recommendation. If B is also empty the harness proves nothing.
    b = run("BUY", "HOLD", s)
    results["B"] = b
    print(f"  B  pos.recommendation='BUY',            fresh='HOLD'  -> {b}")
    print(f"     signal_downgrade present? {has_downgrade(b)}   (control: True)")

    # C -- the other off-vocabulary reason actually observed in paper_trades.
    c = run("swap_buy", "HOLD", s)
    results["C"] = c
    print(f"  C  pos.recommendation='swap_buy',       fresh='HOLD'  -> {c}")
    print(f"     signal_downgrade present? {has_downgrade(c)}   (claim: False)")

    # D -- shows WHY fresh='SELL' cannot be used to test this rule: the
    # sell_signal branch pre-empts it, so a SELL would go green regardless of
    # the held row's value and would prove nothing about signal_downgrade.
    d = run("BUY", "SELL", s)
    results["D"] = d
    print(f"  D  pos.recommendation='BUY',            fresh='SELL'  -> {d}")
    print(f"     reason is 'sell_signal', NOT 'signal_downgrade' -> "
          f"{[r for _,_,r in d]}")

    print("\n=== VERDICT ===")
    ok = True
    if has_downgrade(results["B"]):
        print("  CONTROL GREEN: the harness CAN observe signal_downgrade firing.")
    else:
        print("  CONTROL RED: cell B did not fire -> every other cell is "
              "UNSCORABLE. Fix the probe before believing any absence.")
        ok = False
    if ok:
        dead_a = not has_downgrade(results["A"])
        dead_c = not has_downgrade(results["C"])
        print(f"  A (new_buy_signal) dead: {dead_a}")
        print(f"  C (swap_buy)       dead: {dead_c}")
        if dead_a and dead_c:
            print("\n  PROVEN BY DRIVING: a held row whose recommendation field "
                  "carries an order\n  reason cannot be sold by signal_downgrade, "
                  "while an otherwise identical\n  row carrying 'BUY' is sold. "
                  "The rule is structurally dead on the real data.")
        else:
            print("\n  CLAIM NOT PROVEN -- an off-vocabulary row DID trigger it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
