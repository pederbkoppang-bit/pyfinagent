import { describe, it, expect, afterEach } from "vitest";
import { render, cleanup } from "@testing-library/react";
import { SectorBarList } from "./SectorBarList";

afterEach(() => cleanup());

describe("SectorBarList (phase-44.0 foundation)", () => {
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
});
