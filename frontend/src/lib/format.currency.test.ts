import { describe, it, expect, afterEach, vi } from "vitest";
import {
  formatCurrency,
  formatUsd,
  numberFlowLocale,
  resolveCurrency,
  resolveLocalCurrency,
} from "./format";

/**
 * phase-61.3 — currency resolution + the single en-US USD locale policy.
 *
 * Two things are proven here, and they are different:
 *
 *  1. `resolveLocalCurrency` is MARKET-first, so a LOCAL price column cannot inherit
 *     `base_currency` (which the backend hardcodes to "USD" on every row and which,
 *     per the PaperPosition contract, describes the USD columns). The contrast tests
 *     against `resolveCurrency` are the point: if both helpers behaved the same way,
 *     the new one would be dead weight.
 *
 *  2. Formatting is locale-DETERMINISTIC. Note what this can and cannot show: the
 *     helpers in format.ts always pass an explicit locale, so a forced nb-NO default
 *     cannot break them — that is exactly the regression guard. The real
 *     nondeterminism lived in the NumberFlow `locales` PROP, which is asserted at the
 *     prop level in positions-columns.currency.test.tsx, because jsdom's ICU happily
 *     resolves `undefined` to en-US and would hide the bug here.
 *
 * Assertions are regex/absence-based rather than exact-string wherever the ICU build
 * could legitimately differ: vercel/next.js#79397 (nodejs/node#48120) documents the
 * SAME explicit locale producing different grouping across Node builds. The one exact
 * string asserted below ("$1,234.56") is stable across the builds in that evidence.
 */

const USD_SYMBOL_ON_A_NUMBER = /\$\s?\d/;

afterEach(() => {
  vi.restoreAllMocks();
});

describe("resolveLocalCurrency — market-first by contract (criterion 2)", () => {
  it("resolves KR to KRW from the market even though the row says base_currency USD", () => {
    // This is the exact row shape the backend ships: market "KR", base_currency "USD".
    expect(resolveLocalCurrency({ market: "KR", ticker: "005930.KS" })).toBe("KRW");
  });

  it("resolves EU to EUR and US to USD", () => {
    expect(resolveLocalCurrency({ market: "EU", ticker: "SAP.DE" })).toBe("EUR");
    expect(resolveLocalCurrency({ market: "US", ticker: "NTAP" })).toBe("USD");
  });

  it("falls back to the ticker suffix when the row carries no market", () => {
    expect(resolveLocalCurrency({ ticker: "005930.KS" })).toBe("KRW");
    expect(resolveLocalCurrency({ ticker: "SAP.DE" })).toBe("EUR");
    expect(resolveLocalCurrency({ ticker: "NTAP" })).toBe("USD");
  });

  it("DIFFERS from resolveCurrency on exactly the case that caused the defect", () => {
    const row = { market: "KR", ticker: "005930.KS", baseCurrency: "USD" };
    // The old path: explicit base_currency wins -> a won price rendered with "$".
    expect(resolveCurrency(row)).toBe("USD");
    // The new path ignores it by contract.
    expect(resolveLocalCurrency(row)).toBe("KRW");
  });

  it("leaves resolveCurrency untouched for genuinely-explicit surfaces", () => {
    expect(resolveCurrency({ currency: "eur", ticker: "NTAP" })).toBe("EUR");
    expect(resolveCurrency({ ticker: "005930.KS" })).toBe("KRW");
  });
});

describe("no USD symbol on a non-USD magnitude (criterion 2)", () => {
  it("formats a KRW price with a won symbol and no dollar sign", () => {
    const cur = resolveLocalCurrency({ market: "KR", ticker: "005930.KS" });
    const rendered = formatCurrency(70_000, cur);
    expect(rendered).toMatch(/₩|KRW/);
    expect(rendered).not.toMatch(USD_SYMBOL_ON_A_NUMBER);
  });

  it("formats a EUR price with a euro symbol and no dollar sign", () => {
    const cur = resolveLocalCurrency({ market: "EU", ticker: "SAP.DE" });
    const rendered = formatCurrency(150.25, cur);
    expect(rendered).toMatch(/€|EUR/);
    expect(rendered).not.toMatch(USD_SYMBOL_ON_A_NUMBER);
  });

  it("renders KRW with no decimal places (ISO 4217 minor units)", () => {
    // Forcing minimumFractionDigits:2 would render "₩1,234,567.00" — the pitfall the
    // shared formatter exists to avoid.
    expect(formatCurrency(1_234_567, "KRW")).not.toMatch(/[.,]00\b/);
  });

  it("still renders USD rows with a dollar sign (do-no-harm)", () => {
    const cur = resolveLocalCurrency({ market: "US", ticker: "NTAP" });
    expect(formatCurrency(177.85, cur)).toMatch(USD_SYMBOL_ON_A_NUMBER);
  });
});

describe("one locale policy for USD cells (criterion 3)", () => {
  it("pins USD to en-US", () => {
    expect(numberFlowLocale("USD")).toBe("en-US");
    expect(numberFlowLocale()).toBe("en-US");
  });

  it("formats USD in en-US shape under a forced nb-NO runtime default", () => {
    // Simulate the operator's browser: make an OMITTED locale resolve to nb-NO. A
    // formatter that leaked the runtime default would now render "1 234,56" with a
    // non-breaking space; one that pins its locale is unaffected.
    const Real = Intl.NumberFormat;
    // Constructed via Reflect.construct rather than a subclass: Intl's legacy
    // constructor semantics put the internal slots on a symbol property when the
    // receiver is already a NumberFormat instance, so `class X extends
    // Intl.NumberFormat` yields an object whose `.format` getter does not work.
    const spy = vi.spyOn(Intl, "NumberFormat").mockImplementation(function (
      this: unknown,
      locales?: Intl.LocalesArgument,
      options?: Intl.NumberFormatOptions,
    ) {
      return Reflect.construct(Real, [locales ?? "nb-NO", options]);
    } as unknown as typeof Intl.NumberFormat);

    // Prove the harness actually bites: an omitted locale now formats Norwegian.
    const leaked = new Intl.NumberFormat(undefined, {
      style: "currency",
      currency: "USD",
    }).format(1234.56);
    expect(leaked).not.toBe("$1,234.56");

    // The shared formatters pass an explicit locale, so they are unmoved.
    expect(formatCurrency(1234.56, "USD")).toBe("$1,234.56");
    expect(formatUsd(1234.56)).toBe("$1,234.56");

    spy.mockRestore();
  });

  it("returns an em dash rather than throwing on null/NaN", () => {
    expect(formatCurrency(null, "USD")).toBe("—");
    expect(formatCurrency(Number.NaN, "USD")).toBe("—");
    expect(formatUsd(undefined)).toBe("—");
  });

  it("falls back to USD instead of throwing on an unknown currency code", () => {
    expect(formatCurrency(10, "NOTACURRENCY")).toMatch(USD_SYMBOL_ON_A_NUMBER);
  });
});
