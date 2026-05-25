import { describe, it, expect, afterEach } from "vitest";
import { render, cleanup } from "@testing-library/react";
import { SectorBarList } from "./SectorBarList";

afterEach(() => cleanup());

describe("SectorBarList (phase-44.0 foundation + phase-44.2 Option B)", () => {
  it("renders empty state when no items", () => {
    const { container } = render(<SectorBarList items={[]} capPct={30} />);
    expect(container.textContent).toContain("No positions yet.");
  });

  it("renders custom empty state", () => {
    const { container } = render(
      <SectorBarList items={[]} capPct={30} emptyState="custom-empty" />,
    );
    expect(container.textContent).toContain("custom-empty");
  });

  it("renders title with cap description", () => {
    const { container } = render(
      <SectorBarList
        items={[{ name: "Technology", value: 25 }]}
        capPct={30}
        amberZonePct={5}
      />,
    );
    expect(container.textContent).toContain("Sector concentration");
    expect(container.textContent).toContain("30%");
    expect(container.textContent).toContain("5pp");
  });

  it("renders custom title", () => {
    const { container } = render(
      <SectorBarList
        items={[{ name: "Technology", value: 10 }]}
        capPct={30}
        title="Per-sector exposure"
      />,
    );
    expect(container.textContent).toContain("Per-sector exposure");
  });

  it("renders each sector in the data", () => {
    const items = [
      { name: "Technology", value: 28 },
      { name: "Healthcare", value: 15 },
      { name: "Financials", value: 10 },
    ];
    const { container } = render(<SectorBarList items={items} capPct={30} />);
    for (const item of items) {
      expect(container.textContent).toContain(item.name);
    }
  });

  // phase-44.2 Option B: per-item color tokens are load-bearing.
  it("emerald band when value is far below the cap", () => {
    const { container } = render(
      <SectorBarList items={[{ name: "Technology", value: 10 }]} capPct={30} amberZonePct={5} />,
    );
    const bar = container.querySelector('[data-band="emerald"]');
    expect(bar).not.toBeNull();
  });

  it("amber band when value is within amberZonePct of the cap", () => {
    const { container } = render(
      <SectorBarList items={[{ name: "Technology", value: 27 }]} capPct={30} amberZonePct={5} />,
    );
    const bar = container.querySelector('[data-band="amber"]');
    expect(bar).not.toBeNull();
  });

  it("rose band when value is at or over the cap", () => {
    const { container } = render(
      <SectorBarList items={[{ name: "Technology", value: 35 }]} capPct={30} />,
    );
    const bar = container.querySelector('[data-band="rose"]');
    expect(bar).not.toBeNull();
  });

  it("sorts items by value descending", () => {
    const items = [
      { name: "C", value: 5 },
      { name: "A", value: 25 },
      { name: "B", value: 15 },
    ];
    const { container } = render(<SectorBarList items={items} capPct={30} />);
    const names = Array.from(container.querySelectorAll("li")).map(
      (li) => li.textContent?.match(/^([A-Z])/)?.[1],
    );
    expect(names).toEqual(["A", "B", "C"]);
  });

  it("renders progressbar role with aria-valuenow per sector", () => {
    const { container } = render(
      <SectorBarList items={[{ name: "Technology", value: 28 }]} capPct={30} />,
    );
    const progressbar = container.querySelector('[role="progressbar"]');
    expect(progressbar).not.toBeNull();
    expect(progressbar?.getAttribute("aria-valuenow")).toBe("28");
  });

  it("wraps sector in an anchor when href is supplied", () => {
    const { container } = render(
      <SectorBarList
        items={[{ name: "Technology", value: 10, href: "/sector/tech" }]}
        capPct={30}
      />,
    );
    const link = container.querySelector('a[href="/sector/tech"]');
    expect(link).not.toBeNull();
  });
});
