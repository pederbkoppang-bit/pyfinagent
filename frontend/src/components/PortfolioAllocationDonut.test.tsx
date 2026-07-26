import { describe, it, expect, afterEach, beforeEach, vi } from "vitest";
import { render, cleanup, fireEvent, act } from "@testing-library/react";
import { PortfolioAllocationDonut } from "./PortfolioAllocationDonut";

// phase-80.5: no grace timer ships. The tooltip unmounts synchronously, as it
// did at HEAD. Fake timers are retained only so any future timer-based work
// fails loudly here rather than leaking across tests.

beforeEach(() => vi.useFakeTimers());
afterEach(() => {
  vi.runOnlyPendingTimers();
  vi.useRealTimers();
  cleanup();
});

describe("PortfolioAllocationDonut (phase-44.2 cycle 70 Option B)", () => {
  it("renders empty state when no slices", () => {
    const { container } = render(
      <PortfolioAllocationDonut slices={[]} totalNav={0} />,
    );
    expect(container.textContent).toContain("No allocation data yet.");
  });

  it("renders empty state when all slices are zero", () => {
    const { container } = render(
      <PortfolioAllocationDonut
        slices={[{ name: "Cash", value: 0 }]}
        totalNav={0}
      />,
    );
    expect(container.textContent).toContain("No allocation data yet.");
  });

  it("renders title + subtitle when data is present", () => {
    const { container } = render(
      <PortfolioAllocationDonut
        slices={[
          { name: "Technology", value: 6000 },
          { name: "Cash", value: 4000 },
        ]}
        totalNav={10000}
      />,
    );
    expect(container.textContent).toContain("Allocation");
    expect(container.textContent).toContain("NAV split by sector + cash");
  });

  it("renders each sector + cash in the legend list", () => {
    const { container } = render(
      <PortfolioAllocationDonut
        slices={[
          { name: "Technology", value: 6000 },
          { name: "Industrials", value: 2000 },
          { name: "Cash", value: 2000 },
        ]}
        totalNav={10000}
      />,
    );
    const items = container.querySelectorAll("li");
    expect(items.length).toBe(3);
  });

  it("computes percentages from the slice total", () => {
    const { container } = render(
      <PortfolioAllocationDonut
        slices={[
          { name: "Technology", value: 7500 },
          { name: "Cash", value: 2500 },
        ]}
        totalNav={10000}
      />,
    );
    expect(container.textContent).toContain("75.0%");
    expect(container.textContent).toContain("25.0%");
  });

  it("has region role + aria-label", () => {
    const { container } = render(
      <PortfolioAllocationDonut
        slices={[{ name: "Cash", value: 10000 }]}
        totalNav={10000}
        title="Portfolio mix"
      />,
    );
    const region = container.querySelector('[role="region"]');
    expect(region).not.toBeNull();
    expect(region?.getAttribute("aria-label")).toBe("Portfolio mix");
  });

  it("sorts slices descending by value", () => {
    const { container } = render(
      <PortfolioAllocationDonut
        slices={[
          { name: "Cash", value: 2000 },
          { name: "Technology", value: 6000 },
          { name: "Industrials", value: 2000 },
        ]}
        totalNav={10000}
      />,
    );
    const items = Array.from(container.querySelectorAll("li"));
    expect(items[0].textContent?.startsWith("Technology")).toBe(true);
  });

  it("falls back to slice total when totalNav is null", () => {
    const { container } = render(
      <PortfolioAllocationDonut
        slices={[
          { name: "Technology", value: 6000 },
          { name: "Cash", value: 4000 },
        ]}
        totalNav={null}
      />,
    );
    expect(container.textContent).toContain("60.0%");
    expect(container.textContent).toContain("40.0%");
  });

  // phase-44.2 cycle 70 additions: inline-SVG implementation guarantees

  it("renders one SVG donut + one <circle> per slice plus a track ring", () => {
    const { container } = render(
      <PortfolioAllocationDonut
        slices={[
          { name: "Technology", value: 6000 },
          { name: "Industrials", value: 2000 },
          { name: "Cash", value: 2000 },
        ]}
        totalNav={10000}
      />,
    );
    const svg = container.querySelector("svg");
    expect(svg).not.toBeNull();
    expect(svg?.getAttribute("role")).toBe("img");
    // 1 track ring + 3 slice circles = 4 total
    expect(container.querySelectorAll("circle").length).toBe(4);
  });

  it("uses JIT-safe stroke-* classes (not template-string concatenation)", () => {
    const { container } = render(
      <PortfolioAllocationDonut
        slices={[
          { name: "Technology", value: 6000 },
          { name: "Industrials", value: 4000 },
        ]}
        totalNav={10000}
      />,
    );
    // Technology -> blue; Industrials -> amber per SECTOR_COLOR_MAP
    const html = container.innerHTML;
    expect(html).toContain("stroke-blue-500");
    expect(html).toContain("stroke-amber-500");
  });

  it("shows no tooltip at rest", () => {
    const { container } = render(
      <PortfolioAllocationDonut
        slices={[
          { name: "Technology", value: 6000 },
          { name: "Cash", value: 4000 },
        ]}
        totalNav={10000}
      />,
    );
    expect(container.querySelector('[role="tooltip"]')).toBeNull();
  });

  it("shows tooltip on slice hover with the slice's dollar value + percent", () => {
    const { container } = render(
      <PortfolioAllocationDonut
        slices={[
          { name: "Technology", value: 7500 },
          { name: "Cash", value: 2500 },
        ]}
        totalNav={10000}
      />,
    );
    const sliceCircles = Array.from(container.querySelectorAll("circle")).slice(
      1,
    ); // skip track ring
    fireEvent.mouseEnter(sliceCircles[0]);
    const tooltip = container.querySelector('[role="tooltip"]');
    expect(tooltip).not.toBeNull();
    expect(tooltip?.textContent).toContain("Technology");
    // Locale-tolerant: en-US uses "$7,500", others use "$7 500" / "$7.500" etc.
    expect(tooltip?.textContent?.replace(/[\s,. ]/g, "")).toContain("$7500");
    expect(tooltip?.textContent).toContain("75.0%");
  });

  it("tooltip dismisses on mouseleave", () => {
    const { container } = render(
      <PortfolioAllocationDonut
        slices={[
          { name: "Technology", value: 6000 },
          { name: "Cash", value: 4000 },
        ]}
        totalNav={10000}
      />,
    );
    const sliceCircles = Array.from(container.querySelectorAll("circle")).slice(
      1,
    );
    fireEvent.mouseEnter(sliceCircles[0]);
    expect(container.querySelector('[role="tooltip"]')).not.toBeNull();
    fireEvent.mouseLeave(sliceCircles[0]);
    expect(container.querySelector('[role="tooltip"]')).toBeNull();
  });

  it("has NO svg <title> -- it duplicated the styled tooltip (phase-80.5 defect 2)", () => {
    const { container } = render(
      <PortfolioAllocationDonut
        slices={[
          { name: "Technology", value: 5000 },
          { name: "Cash", value: 5000 },
        ]}
        totalNav={10000}
      />,
    );
    // The browser renders <title> as its OWN unstyled native OS tooltip, at the
    // same time as the styled role="tooltip" div -- so one hover produced TWO
    // tooltips (operator screenshot, 2026-07-25). Inverted from the phase-44.2
    // assertion that required them.
    const titles = Array.from(container.querySelectorAll("svg title"));
    expect(titles.length, "a native <title> would duplicate the styled tooltip").toBe(0);
    expect(container.querySelectorAll('[role="tooltip"]').length).toBe(0);
  });

  // ── phase-80.5: the no-layout-shift invariant ──────────────────────────
  //
  // jsdom CANNOT measure geometry -- verified, not assumed: getBoundingClientRect
  // returns all zeros even with inline width/height, offsetHeight is 0, and
  // elementFromPoint/isPointInFill are undefined. So "hovering causes zero layout
  // shift" cannot be asserted dimensionally here. It is pinned STRUCTURALLY
  // instead, and the real geometry is measured in the Playwright live_check.

  it("hovering adds NO element to the card root -- the layout-shift invariant", () => {
    const { container } = render(
      <PortfolioAllocationDonut
        slices={[
          { name: "Technology", value: 6000 },
          { name: "Cash", value: 4000 },
        ]}
        totalNav={10000}
      />,
    );
    const card = container.querySelector('[role="region"]')!;
    const atRest = card.children.length;

    const slice = Array.from(container.querySelectorAll("circle")).slice(1)[0];
    fireEvent.mouseEnter(slice);
    expect(container.querySelector('[role="tooltip"]'), "precondition: tooltip is open").not.toBeNull();

    // THE ASSERTION. An in-flow tooltip is a 4th child of the card root and
    // grows the card by 66px (54px box + mt-3), which propagates through the
    // items-stretch grid row at positions/page.tsx:151 and pushes everything
    // below down -- the operator's reported page jump. Out-of-flow content
    // adds no child here and no height there.
    //
    // Class-independent on purpose: no strings, no CSS, just DOM structure.
    expect(
      card.children.length,
      "the tooltip must not be an in-flow child of the card root",
    ).toBe(atRest);
  });

  it("renders the tooltip absolutely positioned", () => {
    // Complements the structural test above by pinning the MECHANISM.
    // getComputedStyle does resolve classes from an injected <style> in jsdom
    // (measured), even though geometry does not work.
    const style = document.createElement("style");
    style.textContent = ".absolute{position:absolute}";
    document.head.appendChild(style);
    try {
      const { container } = render(
        <PortfolioAllocationDonut
          slices={[{ name: "Technology", value: 6000 }, { name: "Cash", value: 4000 }]}
          totalNav={10000}
        />,
      );
      const slice = Array.from(container.querySelectorAll("circle")).slice(1)[0];
      fireEvent.mouseEnter(slice);
      const tip = container.querySelector('[role="tooltip"]')! as HTMLElement;
      expect(getComputedStyle(tip).position).toBe("absolute");
    } finally {
      document.head.removeChild(style);
    }
  });

  it("keeps the tooltip INSIDE the card -- no portal, no top layer", () => {
    // Guards the operator-rejected regression directly: cycle-69's Tremor
    // tooltip was portaled and escaped the card with white-on-dark styling.
    // The Popover API would reintroduce it via the top layer.
    const { container } = render(
      <PortfolioAllocationDonut
        slices={[{ name: "Technology", value: 6000 }, { name: "Cash", value: 4000 }]}
        totalNav={10000}
      />,
    );
    const slice = Array.from(container.querySelectorAll("circle")).slice(1)[0];
    fireEvent.mouseEnter(slice);

    const card = container.querySelector('[role="region"]')!;
    const tip = document.querySelector('[role="tooltip"]');
    expect(tip, "tooltip should exist while hovered").not.toBeNull();
    expect(card.contains(tip!), "tooltip escaped the card -- portal regression").toBe(true);
    expect(tip!.closest("[popover]"), "Popover top layer is the rejected pattern").toBeNull();
  });

  it("slice circles use fill=none so the donut HOLE is not hover-active", () => {
    // fill="transparent" is a COLOUR with alpha 0, and under
    // pointer-events: visiblePainted an interior is hit-testable whenever fill
    // is anything other than `none`. So every slice's full disc -- including
    // the hole inside the ring -- was hover-active, all slices overlapped, and
    // the LAST in document order won. Since data is sorted DESCENDING, that was
    // the SMALLEST slice: hovering dead centre asserted the wrong sector AND
    // fired the page reflow. Both values paint no interior, so this is visually
    // identical.
    const { container } = render(
      <PortfolioAllocationDonut
        slices={[{ name: "Technology", value: 6000 }, { name: "Cash", value: 4000 }]}
        totalNav={10000}
      />,
    );
    const circles = Array.from(container.querySelectorAll("circle"));
    expect(circles.length).toBeGreaterThan(0);
    for (const c of circles) {
      expect(c.getAttribute("fill"), "transparent keeps the interior hit-testable").toBe("none");
    }
    // the centre labels must not intercept either -- they sit last in the SVG
    const texts = Array.from(container.querySelectorAll("svg text"));
    expect(texts.length).toBeGreaterThan(0);
    for (const t of texts) {
      expect(t.getAttribute("pointer-events")).toBe("none");
    }
  });

  it("centre label and legend agree on the rounded percentage", () => {
    // :258 used toFixed(0) while the legend 8px away used toFixed(1), so the
    // same quantity rendered as "6%" and "5.6%" side by side.
    //
    // Scoped to the centre <text> deliberately: asserting on
    // container.textContent would pass vacuously because the legend already
    // contains the one-decimal string.
    const { container } = render(
      <PortfolioAllocationDonut
        slices={[{ name: "Technology", value: 5600 }, { name: "Cash", value: 4400 }]}
        totalNav={10000}
      />,
    );
    const slice = Array.from(container.querySelectorAll("circle")).slice(1)[0];
    fireEvent.mouseEnter(slice);

    const centre = Array.from(container.querySelectorAll("svg text"))
      .map((t) => t.textContent ?? "")
      .join(" ");
    expect(centre).toContain("56.0%");
    expect(centre, "centre must not round to a different figure than the legend").not.toContain("56%");
  });

  it("Escape closes the tooltip (WCAG 1.4.13 dismissible)", () => {
    // The overlay tooltip now OBSCURES content, which revokes SC 1.4.13's
    // "does not obscure or replace other content" exception -- so Dismissible
    // is genuinely required rather than vacuously satisfied. The listener is
    // document-level on purpose: the old onKeyDown lived on the container and
    // never fired on the mouse path, because focus is elsewhere entirely.
    const { container } = render(
      <PortfolioAllocationDonut
        slices={[{ name: "Technology", value: 6000 }, { name: "Cash", value: 4000 }]}
        totalNav={10000}
      />,
    );
    const slice = Array.from(container.querySelectorAll("circle")).slice(1)[0];
    fireEvent.mouseEnter(slice);
    expect(container.querySelector('[role="tooltip"]')).not.toBeNull();

    act(() => {
      fireEvent.keyDown(document, { key: "Escape" });
    });
    expect(
      container.querySelector('[role="tooltip"]'),
      "Escape must dismiss an obscuring tooltip",
    ).toBeNull();
  });

  it("the tooltip's offset parent is the positioned chart row, not the viewport", () => {
    // Without `relative` on the chart row, `absolute` resolves against the
    // nearest positioned ancestor -- and there is none, so the tooltip would
    // pin to the VIEWPORT: exactly the escaping behaviour the operator
    // rejected in cycle-69, and every other test stays green.
    // jsdom has no layout, so this pins the containing block structurally.
    const { container } = render(
      <PortfolioAllocationDonut
        slices={[{ name: "Technology", value: 6000 }, { name: "Cash", value: 4000 }]}
        totalNav={10000}
      />,
    );
    const slice = Array.from(container.querySelectorAll("circle")).slice(1)[0];
    fireEvent.mouseEnter(slice);
    const tip = container.querySelector('[role="tooltip"]')!;
    // classList.contains, NOT className.includes: the cycle-2 Q/A measured that
    // a substring check accepts `lg:relative`, which establishes a containing
    // block only at >=1024px -- below that the tooltip escapes to the viewport,
    // which is the operator-rejected behaviour, with a green suite.
    expect(
      tip.parentElement?.classList.contains("relative"),
      "parent must carry an UNCONDITIONAL `relative` (a `lg:` variant leaves small screens broken)",
    ).toBe(true);
  });

  const twoSlices = { slices: [{ name: "Technology", value: 6000 }, { name: "Cash", value: 4000 }], totalNav: 10000 };

  it("leaving a LEGEND row closes the tooltip", () => {
    // Found by line-indexed mutation: deleting the legend's onMouseLeave left
    // the whole suite green while leaving the tooltip stuck open after the
    // pointer left the row. The arc path was guarded; this one was not.
    const { container } = render(
      <PortfolioAllocationDonut
        slices={[{ name: "Technology", value: 6000 }, { name: "Cash", value: 4000 }]}
        totalNav={10000}
      />,
    );
    const row = container.querySelectorAll("li")[0];
    fireEvent.mouseEnter(row);
    expect(container.querySelector('[role="tooltip"]')).not.toBeNull();
    fireEvent.mouseLeave(row);
    expect(
      container.querySelector('[role="tooltip"]'),
      "tooltip stayed open after the pointer left the legend row",
    ).toBeNull();
  });

  it("leaving the TOOLTIP closes it -- no stuck overlay", () => {
    // The tooltip overlays the legend, so once the pointer enters it, the
    // tooltip's OWN onMouseLeave is the only thing that can close it. Deleting
    // that line left the suite green while producing a permanently stuck
    // overlay sitting on top of the legend rows.
    const { container } = render(
      <PortfolioAllocationDonut
        slices={[{ name: "Technology", value: 6000 }, { name: "Cash", value: 4000 }]}
        totalNav={10000}
      />,
    );
    const slice = Array.from(container.querySelectorAll("circle")).slice(1)[0];
    fireEvent.mouseEnter(slice);
    const tip = container.querySelector('[role="tooltip"]')!;
    expect(tip).not.toBeNull();

    fireEvent.mouseEnter(tip);
    fireEvent.mouseLeave(tip);
    expect(
      container.querySelector('[role="tooltip"]'),
      "tooltip stayed open after the pointer left it -- stuck overlay",
    ).toBeNull();
  });
});
