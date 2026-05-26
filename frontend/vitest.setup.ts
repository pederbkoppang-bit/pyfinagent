import "@testing-library/jest-dom/vitest";

// phase-44.2 cycle-68: jsdom doesn't ship ResizeObserver, which
// Recharts (transitively via Tremor DonutChart) requires. Minimal
// no-op shim so tests that render charts can mount without crashing.
if (typeof window !== "undefined" && !("ResizeObserver" in window)) {
  class ResizeObserverShim {
    observe(): void {}
    unobserve(): void {}
    disconnect(): void {}
  }
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (window as any).ResizeObserver = ResizeObserverShim;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (globalThis as any).ResizeObserver = ResizeObserverShim;
}
