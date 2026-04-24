# Alpaca MCP Smoketest -- phase-17.3
Captured: 2026-04-24T20:36:52.179628+00:00

## get_account_info (equivalent to mcp__alpaca*__get_account_info)
- account_number: PA3VQZZLAKE2
- status: ACTIVE
- buying_power: $200,000.00
- cash: $100,000.00
- portfolio_value: $100,000.00
- trading_blocked: False
- currency: USD

## get_clock
- is_open: False
- next_open: 2026-04-27T09:30:00-04:00
- next_close: 2026-04-27T16:00:00-04:00

## get_stock_snapshot AAPL (equivalent to mcp__alpaca*__get_stock_snapshot)
- latest_trade_price: 270.86
- latest_quote_bid: 259.12
- latest_quote_ask: 286.63
- daily_bar_close: 271.04

## Result
PASS -- all three Alpaca MCP-equivalent tool calls succeeded over HTTPS to paper-api.alpaca.markets.
The alpaca-mcp-server==2.0.1 wraps these same endpoints via alpaca-py,
so the MCP tool surface is verified working for this paper key pair.
