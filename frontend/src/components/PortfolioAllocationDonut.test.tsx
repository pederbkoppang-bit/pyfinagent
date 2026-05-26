import { describe, it, expect, afterEach } from "vitest";
import { render, cleanup, fireEvent } from "@testing-library/react";
import { PortfolioAllocationDonut } from "./PortfolioAllocationDonut";

afterEach(() => cleanup());

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

  it("each slice has a <title> child for native SVG hover tooltips", () => {
    const { container } = render(
      <PortfolioAllocationDonut
        slices={[
          { name: "Technology", value: 5000 },
          { name: "Cash", value: 5000 },
        ]}
        totalNav={10000}
      />,
    );
    const titles = Array.from(container.querySelectorAll("svg title"));
    expect(titles.length).toBe(2);
    expect(titles[0].textContent).toContain("Technology");
    expect(titles[0].textContent).toContain("50.0%");
  });
});
