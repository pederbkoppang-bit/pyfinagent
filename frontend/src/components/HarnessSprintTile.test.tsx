import { describe, it, expect, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import { HarnessSprintTile } from "./HarnessSprintTile";
import type { HarnessSprintWeekState } from "@/lib/types";

const FULL_DATA: HarnessSprintWeekState = {
  weekIso: "2026-W17",
  thu: { batchId: "b9686bc5-aaaa-bbbb-cccc-dddd1111ffff", candidatesKicked: 128 },
  fri: { promotedIds: ["g1", "g2"], rejectedIds: ["r1", "r2", "r3"] },
  monthly: { sortinoDelta: 0.42, approvalPending: false, approved: true },
};

const PENDING_DATA: HarnessSprintWeekState = {
  weekIso: "2026-W17",
  thu: { batchId: "b9686bc5-aaaa-bbbb-cccc-dddd1111ffff", candidatesKicked: 128 },
  fri: { promotedIds: ["g1"], rejectedIds: [] },
  monthly: { sortinoDelta: 0.35, approvalPending: true, approved: false },
};

describe("HarnessSprintTile", () => {
  afterEach(() => cleanup());

  it("tile_renders_weekly_state", () => {
    const { container } = render(<HarnessSprintTile data={FULL_DATA} />);

    const weekly = container.querySelector('[data-section="weekly-state"]');
    expect(weekly).not.toBeNull();

    // Thu candidates count rendered
    const thuCell = container.querySelector('[data-cell="thu-candidates"]');
    expect(thuCell).not.toBeNull();
    expect(thuCell?.textContent).toContain("128");

    // Fri promoted count rendered
    const friCell = container.querySelector('[data-cell="fri-promoted-count"]');
    expect(friCell).not.toBeNull();
    expect(friCell?.textContent).toContain("2");

    // Week ISO surfaced
    const section = container.querySelector('[aria-label="Sprint state"]');
    expect(section?.getAttribute("data-week-iso")).toBe("2026-W17");
  });

  it("tile_renders_monthly_sortino_delta", () => {
    const { container } = render(<HarnessSprintTile data={FULL_DATA} />);

    const deltaCell = container.querySelector('[data-cell="sortino-delta"]');
    expect(deltaCell).not.toBeNull();
    // Positive delta with sign prefix
    expect(deltaCell?.textContent).toBe("+0.420");

    // Pending state also renders a sortino-delta cell (amber)
    cleanup();
    const { container: pendingContainer } = render(
      <HarnessSprintTile data={PENDING_DATA} />
    );
    const pendingDelta = pendingContainer.querySelector(
      '[data-cell="sortino-delta"]'
    );
    expect(pendingDelta?.textContent).toBe("+0.350");
  });

  it("read_only_no_mutation_controls", () => {
    const { container } = render(<HarnessSprintTile data={FULL_DATA} />);

    // Canonical RTL read-only guard: no buttons.
    expect(screen.queryAllByRole("button")).toHaveLength(0);
    // No form inputs of any kind.
    expect(container.querySelectorAll("input")).toHaveLength(0);
    expect(container.querySelectorAll("select")).toHaveLength(0);
    expect(container.querySelectorAll("textarea")).toHaveLength(0);
    // No form elements.
    expect(container.querySelectorAll("form")).toHaveLength(0);
  });

  it("renders empty state when data is null", () => {
    const { container } = render(<HarnessSprintTile data={null} />);
    expect(container.textContent).toContain("No sprint activity yet");
    // Empty state still has no mutation controls.
    expect(screen.queryAllByRole("button")).toHaveLength(0);
  });

  it("handles partial data (fri null, monthly null)", () => {
    const partial: HarnessSprintWeekState = {
      weekIso: "2026-W17",
      thu: { batchId: "b9686bc5", candidatesKicked: 128 },
      fri: null,
      monthly: null,
    };
    const { container } = render(<HarnessSprintTile data={partial} />);
    expect(container.textContent).toContain("Not fired yet");
    expect(container.textContent).toContain("Awaiting last trading Friday");
    // Monthly section is still rendered, just without the delta cell.
    expect(
      container.querySelector('[data-cell="sortino-delta"]')
    ).toBeNull();
  });
});
