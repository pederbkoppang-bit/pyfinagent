import { describe, it, expect, afterEach } from "vitest";
import { render, cleanup } from "@testing-library/react";
import { PortfolioAllocationDonut } from "./PortfolioAllocationDonut";

afterEach(() => cleanup());

describe("PortfolioAllocationDonut (phase-44.2 cycle 68)", () => {
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
});
