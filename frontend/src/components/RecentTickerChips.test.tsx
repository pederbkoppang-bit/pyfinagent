import { describe, it, expect, afterEach, beforeEach, vi } from "vitest";
import { render, cleanup, fireEvent } from "@testing-library/react";
import { RecentTickerChips, _internals } from "./RecentTickerChips";

// JSDOM in this vitest setup does not provide a writable Storage by
// default. Stub a minimal in-memory localStorage so the component's
// hydration + write-back logic exercises against real I/O behavior.
const memStore: Map<string, string> = new Map();
const localStorageMock = {
  getItem: (k: string) => memStore.get(k) ?? null,
  setItem: (k: string, v: string) => { memStore.set(k, v); },
  removeItem: (k: string) => { memStore.delete(k); },
  clear: () => { memStore.clear(); },
  key: (i: number) => Array.from(memStore.keys())[i] ?? null,
  get length() { return memStore.size; },
};

const TEST_KEY = "pyfinagent.test.recentTickers";

beforeEach(() => {
  vi.stubGlobal("localStorage", localStorageMock);
  Object.defineProperty(window, "localStorage", {
    value: localStorageMock,
    writable: true,
  });
  memStore.clear();
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  memStore.clear();
});

describe("RecentTickerChips (phase-44.6)", () => {
  it("renders nothing when localStorage is empty", () => {
    const { container } = render(
      <RecentTickerChips onSelect={() => {}} storageKey={TEST_KEY} />,
    );
    expect(container.querySelector('[role="group"]')).toBeNull();
  });

  it("hydrates from localStorage", () => {
    window.localStorage.setItem(TEST_KEY, JSON.stringify(["AAPL", "MSFT"]));
    const { container } = render(
      <RecentTickerChips onSelect={() => {}} storageKey={TEST_KEY} />,
    );
    const buttons = container.querySelectorAll("button");
    expect(buttons.length).toBe(2);
    expect(buttons[0].textContent).toBe("AAPL");
    expect(buttons[1].textContent).toBe("MSFT");
  });

  it("emits onSelect when a chip is clicked", () => {
    window.localStorage.setItem(TEST_KEY, JSON.stringify(["NVDA"]));
    const onSelect = vi.fn();
    const { container } = render(
      <RecentTickerChips onSelect={onSelect} storageKey={TEST_KEY} />,
    );
    fireEvent.click(container.querySelector("button")!);
    expect(onSelect).toHaveBeenCalledWith("NVDA");
  });

  it("prepends a newly submitted ticker and persists it", () => {
    const { rerender, container } = render(
      <RecentTickerChips onSelect={() => {}} storageKey={TEST_KEY} />,
    );
    rerender(
      <RecentTickerChips onSelect={() => {}} recentlySubmitted="GOOG" storageKey={TEST_KEY} />,
    );
    const buttons = container.querySelectorAll("button");
    expect(buttons.length).toBe(1);
    expect(buttons[0].textContent).toBe("GOOG");
    expect(JSON.parse(window.localStorage.getItem(TEST_KEY) ?? "[]")).toEqual(["GOOG"]);
  });

  it("dedupes when submitting an existing ticker (moves it to front)", () => {
    window.localStorage.setItem(TEST_KEY, JSON.stringify(["AAPL", "MSFT", "GOOG"]));
    const { rerender, container } = render(
      <RecentTickerChips onSelect={() => {}} storageKey={TEST_KEY} />,
    );
    rerender(
      <RecentTickerChips onSelect={() => {}} recentlySubmitted="MSFT" storageKey={TEST_KEY} />,
    );
    const buttons = container.querySelectorAll("button");
    expect(Array.from(buttons).map((b) => b.textContent)).toEqual(["MSFT", "AAPL", "GOOG"]);
  });

  it("caps the list to MAX_CHIPS (5)", () => {
    window.localStorage.setItem(
      TEST_KEY,
      JSON.stringify(["AAPL", "MSFT", "GOOG", "AMZN", "META"]),
    );
    const { rerender, container } = render(
      <RecentTickerChips onSelect={() => {}} storageKey={TEST_KEY} />,
    );
    rerender(
      <RecentTickerChips onSelect={() => {}} recentlySubmitted="TSLA" storageKey={TEST_KEY} />,
    );
    const buttons = container.querySelectorAll("button");
    expect(buttons.length).toBe(5);
    expect(buttons[0].textContent).toBe("TSLA");
    expect(Array.from(buttons).map((b) => b.textContent)).not.toContain("META");
  });

  it("uppercases and trims submitted tickers", () => {
    const { rerender, container } = render(
      <RecentTickerChips onSelect={() => {}} storageKey={TEST_KEY} />,
    );
    rerender(
      <RecentTickerChips onSelect={() => {}} recentlySubmitted="  nvda  " storageKey={TEST_KEY} />,
    );
    expect(container.querySelector("button")?.textContent).toBe("NVDA");
  });

  it("ignores empty/whitespace submissions", () => {
    const { rerender, container } = render(
      <RecentTickerChips onSelect={() => {}} storageKey={TEST_KEY} />,
    );
    rerender(
      <RecentTickerChips onSelect={() => {}} recentlySubmitted="   " storageKey={TEST_KEY} />,
    );
    expect(container.querySelector('[role="group"]')).toBeNull();
  });

  it("has role=group + aria-label", () => {
    window.localStorage.setItem(TEST_KEY, JSON.stringify(["AAPL"]));
    const { container } = render(
      <RecentTickerChips onSelect={() => {}} storageKey={TEST_KEY} />,
    );
    const group = container.querySelector('[role="group"]');
    expect(group).not.toBeNull();
    expect(group?.getAttribute("aria-label")).toBe("Recent tickers");
  });

  it("each chip has aria-label and target-size compliant min-h", () => {
    window.localStorage.setItem(TEST_KEY, JSON.stringify(["AAPL"]));
    const { container } = render(
      <RecentTickerChips onSelect={() => {}} storageKey={TEST_KEY} />,
    );
    const button = container.querySelector("button")!;
    expect(button.getAttribute("aria-label")).toBe("Analyze AAPL");
    expect(button.className).toContain("min-h-[24px]");
  });

  it("internals export sensible constants", () => {
    expect(_internals.MAX_CHIPS).toBe(5);
    expect(_internals.STORAGE_KEY).toBe("pyfinagent.signals.recentTickers");
  });
});
