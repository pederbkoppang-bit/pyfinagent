---
paths:
  - "backend/tools/**"
---

# Data Tools — 16 Enrichment Tool Conventions

## Tool Contract
All tools in `backend/tools/` return a consistent structure:
```json
{ "ticker": "AAPL", "signal": "BULLISH", "summary": "...", "data": { ... } }
```

## Tool Registry (16 tools)
| Tool | Source | Rate Limits |
|------|--------|-------------|
| `alphavantage.py` | AV News API (50 articles) | 5 req/min |
| `yfinance_tool.py` | Yahoo Finance | None |
| `sec_insider.py` | SEC EDGAR Form 4 | Custom User-Agent required |
| `options_flow.py` | yfinance options chain | None |
| `social_sentiment.py` | AV Social Sentiment API | 5 req/min |
| `patent_tracker.py` | USPTO PatentsView API | None |
| `earnings_tone.py` | API Ninjas | API key required |
| `fred_data.py` | Federal Reserve FRED (7 series) | API key required |
| `alt_data.py` | Google Trends (pytrends) | Rate limited, US-only |
| `sector_analysis.py` | yfinance + 11 SPDR ETFs | None |
| `nlp_sentiment.py` | Vertex AI text-embedding-005 | Vertex AI quota |
| `anomaly_detector.py` | Multi-dimensional Z-score | None |
| `monte_carlo.py` | 1,000 GBM simulations | None |
| `slack.py` | Slack Webhook | None |
| `screener.py` | yfinance + Wikipedia S&P 500 | None |
| `quant_model.py` | MDA cache + yfinance live features | None |

## Conventions
- All async tools use `aiohttp` or async wrappers
- External API calls respect rate limits with automatic retry and exponential backoff
- SEC EDGAR requires User-Agent: `FirstName LastName email@domain.com`
- Sector routing: orchestrator skips irrelevant tools per sector (e.g., patents for Financial Services)
- Error returns: `{ "ticker": "...", "signal": "ERROR", "summary": "...", "data": {} }`
