// Multi-market currency + market metadata. Pure, dependency-free, unit-testable.
//
// Mirrors the backend single source of truth:
//   - MARKET_CONFIG (backend/backtest/markets.py:26-62) -> currency + benchmark
//   - market_for_symbol() (backend/backtest/markets.py:106-120) -> suffix is the
//     source of truth for which market a yfinance symbol belongs to.
//
// DO-NO-HARM: every currency defaults to USD/en-US, and `marketForSymbol` returns
// "US" for a bare (un-suffixed) ticker. So US rows render byte-identically to the
// pre-multi-market UI; only EU (.DE/.PA/.AS/.F) and KR (.KS/.KQ) tickers diverge.

// Type-only import (no runtime dependency): NumberFlow's `Format` narrows
// Intl.NumberFormatOptions (notation: 'standard' | 'compact' only), so the animated
// cells need the narrower type, not the raw Intl options.
import type { Format } from "@number-flow/react";

export type MarketCode = "US" | "EU" | "KR" | "NO" | "CA";

const DEFAULT_MARKET = "US";

// Mirror of MARKET_CONFIG[*].currency. Nordics: NO=NOK, SE=SEK, DK=DKK, FI=EUR
// (Finland is euro-zone), IS=ISK.
export const MARKET_CURRENCY: Record<string, string> = {
  US: "USD",
  EU: "EUR",
  KR: "KRW",
  NO: "NOK",
  SE: "SEK",
  DK: "DKK",
  FI: "EUR",
  IS: "ISK",
  CA: "CAD",
};

// Short, human label for the per-market benchmark KPI ("vs <label>").
// Backend benchmark tickers: SPY, ^GDAXI, ^KS11, ^OSEAX, ^OMX, ^OMXC25, ^OMXH25,
// ^OMXIPI, ^GSPTSE.
export const MARKET_BENCHMARK_LABEL: Record<string, string> = {
  US: "SPY",
  EU: "DAX",
  KR: "KOSPI",
  NO: "OSEAX",
  SE: "OMXS30",
  DK: "OMXC25",
  FI: "OMXH25",
  IS: "OMXI",
  CA: "TSX",
};

// Friendly exchange name per market (for the Market column + chip tooltip).
export const MARKET_EXCHANGE: Record<string, string> = {
  US: "NYSE/Nasdaq",
  EU: "XETRA",
  KR: "KRX",
  NO: "Oslo Børs",
  SE: "Nasdaq Stockholm",
  DK: "Nasdaq Copenhagen",
  FI: "Nasdaq Helsinki",
  IS: "Nasdaq Iceland",
  CA: "TSX",
};

// Compact exchange tag for inline display in the dense Market column.
export const MARKET_EXCHANGE_SHORT: Record<string, string> = {
  US: "NYSE",
  EU: "XETRA",
  KR: "KRX",
  NO: "OSE",
  SE: "STO",
  DK: "CPH",
  FI: "HEL",
  IS: "ICE",
  CA: "TSX",
};

// Canonical display order for the market filter + session strip.
export const MARKET_ORDER = ["US", "EU", "NO", "SE", "DK", "FI", "IS", "CA", "KR"];

export function exchangeFor(market: string): string {
  return MARKET_EXCHANGE[(market || "").toUpperCase()] ?? "";
}

// Locale per currency. EUR uses en-IE to keep '.'/',' grouping consistent with the
// USD-dominant columns (de-DE would flip to "1.234,56 €" and churn separators across
// columns); KRW ko-KR; default en-US. Reversible product choice (research brief).
const CURRENCY_LOCALE: Record<string, string> = {
  USD: "en-US",
  EUR: "en-IE",
  KRW: "ko-KR",
  NOK: "nb-NO",
  SEK: "sv-SE",
  DKK: "da-DK",
  ISK: "is-IS",
  CAD: "en-CA",
};

// Tailwind dot color per market. STATIC literal map (Tailwind JIT cannot compile
// template-built class strings -- see .claude/rules/frontend.md rule 3). NO flag
// emoji: a colored dot + the market code convey the market (WCAG: not color alone).
export const MARKET_DOT_CLASS: Record<string, string> = {
  US: "bg-sky-400",
  EU: "bg-amber-400",
  KR: "bg-violet-400",
  NO: "bg-rose-400",
  SE: "bg-cyan-400",
  DK: "bg-pink-400",
  FI: "bg-teal-400",
  IS: "bg-lime-400",
  CA: "bg-emerald-400",
};

// TS port of backend market_for_symbol(). Suffix is the source of truth; a bare
// ticker -> US. .KS/.KQ -> KR; .DE/.PA/.AS/.F -> EU; .OL -> NO; .TO -> CA.
export function marketForSymbol(symbol: string | null | undefined): string {
  const s = (symbol ?? "").toUpperCase();
  if (s.endsWith(".KS") || s.endsWith(".KQ")) return "KR";
  if (s.endsWith(".DE") || s.endsWith(".PA") || s.endsWith(".AS") || s.endsWith(".F")) return "EU";
  if (s.endsWith(".OL")) return "NO"; // Oslo Børs
  if (s.endsWith(".ST")) return "SE"; // Nasdaq Stockholm
  if (s.endsWith(".CO")) return "DK"; // Nasdaq Copenhagen
  if (s.endsWith(".HE")) return "FI"; // Nasdaq Helsinki
  if (s.endsWith(".IC")) return "IS"; // Nasdaq Iceland
  if (s.endsWith(".TO")) return "CA";
  return DEFAULT_MARKET;
}

// Resolve a market code: explicit `market` wins, else derive from the ticker suffix.
export function resolveMarket(opts: { market?: string | null; ticker?: string | null }): string {
  const explicit = (opts.market ?? "").trim();
  if (explicit) return explicit.toUpperCase();
  return marketForSymbol(opts.ticker);
}

// phase-56.1 (55.1 F-1): per-position USD market value for live overlays.
// US keeps the exact legacy live formula (livePrice ?? current_price ?? entry) x qty;
// non-US uses the backend's USD `market_value` -- a client-side livePrice x qty
// would be LOCAL notional (the away-week NAV=345,968-on-$23.8K bug). Mirrors the
// goal-multimarket-ux `mvUsd` in positions/page.tsx; extracted here so every
// NAV/concentration consumer shares one FX-safe formula.
export function positionMarketValueUsd(
  pos: {
    market?: string | null;
    ticker?: string | null;
    quantity: number;
    current_price?: number | null;
    avg_entry_price: number;
    market_value?: number | null;
  },
  livePrice?: number | null,
): number {
  const isUs = resolveMarket({ market: pos.market, ticker: pos.ticker }) === "US";
  if (isUs) {
    const px = livePrice ?? pos.current_price ?? pos.avg_entry_price;
    return px * pos.quantity;
  }
  return pos.market_value ?? 0;
}

// Resolve a currency code: explicit base_currency/currency wins, else the resolved
// market's currency, else USD (keeps the US/legacy path byte-identical).
export function resolveCurrency(opts: {
  baseCurrency?: string | null;
  currency?: string | null;
  market?: string | null;
  ticker?: string | null;
}): string {
  const explicit = (opts.baseCurrency ?? opts.currency ?? "").trim();
  if (explicit) return explicit.toUpperCase();
  const mkt = resolveMarket({ market: opts.market, ticker: opts.ticker });
  return MARKET_CURRENCY[mkt] ?? "USD";
}

function localeForCurrency(currency: string): string {
  return CURRENCY_LOCALE[currency.toUpperCase()] ?? "en-US";
}

// Locale-correct currency string. `narrowSymbol` -> USD "$", EUR "€", KRW "₩".
// No forced fraction digits: Intl uses the currency default (USD/EUR=2, KRW=0) --
// forcing minimumFractionDigits:2 would wrongly render "₩1,234,567.00" (research).
export function formatCurrency(
  value: number | null | undefined,
  currency: string = "USD",
): string {
  if (value == null || !Number.isFinite(value)) return "—";
  const cur = (currency || "USD").toUpperCase();
  try {
    return new Intl.NumberFormat(localeForCurrency(cur), {
      style: "currency",
      currency: cur,
      currencyDisplay: "narrowSymbol",
    }).format(value);
  } catch {
    // Unknown currency code -> never throw inside a table cell; fall back to USD.
    return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }).format(value);
  }
}

// USD convenience -- VALUE / NAV / COST-BASIS / FEE columns are always USD base.
export function formatUsd(value: number | null | undefined): string {
  return formatCurrency(value, "USD");
}

// NumberFlow `format` prop (animated cells) for a currency. Same no-forced-fraction
// rule as formatCurrency so KRW animates at 0 decimals.
export function numberFlowFormat(currency: string = "USD"): Format {
  return {
    style: "currency",
    currency: (currency || "USD").toUpperCase(),
    currencyDisplay: "narrowSymbol",
  };
}

// NumberFlow `locales` prop for a currency (Latin-digit locales only -- safe; never
// pass ar/fa/bn which NumberFlow cannot render).
export function numberFlowLocale(currency: string = "USD"): string {
  return localeForCurrency((currency || "USD").toUpperCase());
}

// ── Market sessions (goal-multimarket-ux #6) ──────────────────────────────────
// Regular cash-session hours per exchange, in the market's local timezone. This is
// a SESSION HEURISTIC for the open/closed indicator: it honours the weekday + local
// trading window but does NOT account for exchange holidays or half-days (the backend
// `is_trading_day` + exchange_calendars owns the authoritative gate; this is a UI hint).
const MARKET_HOURS: Record<string, { tz: string; openMin: number; closeMin: number }> = {
  US: { tz: "America/New_York", openMin: 9 * 60 + 30, closeMin: 16 * 60 }, // 09:30-16:00 ET
  EU: { tz: "Europe/Berlin", openMin: 9 * 60, closeMin: 17 * 60 + 30 }, // XETRA 09:00-17:30 CET
  KR: { tz: "Asia/Seoul", openMin: 9 * 60, closeMin: 15 * 60 + 30 }, // KRX 09:00-15:30 KST
  NO: { tz: "Europe/Oslo", openMin: 9 * 60, closeMin: 16 * 60 + 20 }, // Oslo Børs 09:00-16:20 CET
  SE: { tz: "Europe/Stockholm", openMin: 9 * 60, closeMin: 17 * 60 + 30 }, // Stockholm 09:00-17:30
  DK: { tz: "Europe/Copenhagen", openMin: 9 * 60, closeMin: 17 * 60 }, // Copenhagen 09:00-17:00
  FI: { tz: "Europe/Helsinki", openMin: 10 * 60, closeMin: 18 * 60 + 30 }, // Helsinki 10:00-18:30 EET
  IS: { tz: "Atlantic/Reykjavik", openMin: 9 * 60 + 30, closeMin: 15 * 60 + 30 }, // Iceland 09:30-15:30
};

// True if `market` is within its regular weekday cash session at `now`. Holiday-blind
// (see note above). Unknown markets -> false.
export function isMarketOpen(market: string, now: Date = new Date()): boolean {
  const cfg = MARKET_HOURS[(market || "").toUpperCase()];
  if (!cfg) return false;
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: cfg.tz,
    weekday: "short",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).formatToParts(now);
  const wd = parts.find((p) => p.type === "weekday")?.value ?? "";
  if (wd === "Sat" || wd === "Sun") return false;
  let hh = Number(parts.find((p) => p.type === "hour")?.value ?? "0");
  if (hh === 24) hh = 0; // some ICU builds emit "24" at local midnight
  const mm = Number(parts.find((p) => p.type === "minute")?.value ?? "0");
  const cur = hh * 60 + mm;
  return cur >= cfg.openMin && cur < cfg.closeMin;
}
