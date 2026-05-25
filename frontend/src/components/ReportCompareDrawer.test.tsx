import { describe, it, expect, afterEach, vi } from "vitest";
import { render, cleanup, fireEvent } from "@testing-library/react";
import { ReportCompareDrawer } from "./ReportCompareDrawer";
import type { ReportSummary } from "@/lib/types";

afterEach(() => cleanup());

function rs(ticker: string, analysis_date: string): ReportSummary {
  return {
    ticker,
    company_name: `${ticker} Inc.`,
    analysis_date,
    final_score: 7.5,
    recommendation: "BUY",
    summary: "test",
  };
}

describe("ReportCompareDrawer (phase-44.4)", () => {
  it("renders nothing when closed", () => {
    const { container } = render(
      <ReportCompareDrawer
        open={false}
        onClose={() => {}}
        reports={[]}
        selected={new Set()}
        onToggle={() => {}}
        onStartCompare={() => {}}
        comparing={false}
      />,
    );
    expect(container.querySelector('[role="dialog"]')).toBeNull();
  });

  it("renders role=dialog + aria-modal=true when open", () => {
    const { container } = render(
      <ReportCompareDrawer
        open
        onClose={() => {}}
        reports={[]}
        selected={new Set()}
        onToggle={() => {}}
        onStartCompare={() => {}}
        comparing={false}
      />,
    );
    const dialog = container.querySelector('[role="dialog"]');
    expect(dialog).not.toBeNull();
    expect(dialog?.getAttribute("aria-modal")).toBe("true");
  });

  it("shows the list of reports with aria-pressed", () => {
    const reports = [rs("AAPL", "2026-05-20"), rs("MSFT", "2026-05-19")];
    const { container } = render(
      <ReportCompareDrawer
        open
        onClose={() => {}}
        reports={reports}
        selected={new Set(["AAPL|2026-05-20"])}
        onToggle={() => {}}
        onStartCompare={() => {}}
        comparing={false}
      />,
    );
    const buttons = container.querySelectorAll('button[aria-pressed]');
    expect(buttons.length).toBe(2);
    expect(buttons[0].getAttribute("aria-pressed")).toBe("true");
    expect(buttons[1].getAttribute("aria-pressed")).toBe("false");
  });

  it("calls onToggle when a report button is clicked", () => {
    const onToggle = vi.fn();
    const reports = [rs("AAPL", "2026-05-20")];
    const { container } = render(
      <ReportCompareDrawer
        open
        onClose={() => {}}
        reports={reports}
        selected={new Set()}
        onToggle={onToggle}
        onStartCompare={() => {}}
        comparing={false}
      />,
    );
    fireEvent.click(container.querySelector('button[aria-pressed]')!);
    expect(onToggle).toHaveBeenCalledWith("AAPL|2026-05-20");
  });

  it("Compare button disabled when fewer than 2 selected", () => {
    const reports = [rs("AAPL", "2026-05-20")];
    const { container } = render(
      <ReportCompareDrawer
        open
        onClose={() => {}}
        reports={reports}
        selected={new Set(["AAPL|2026-05-20"])}
        onToggle={() => {}}
        onStartCompare={() => {}}
        comparing={false}
      />,
    );
    const compareBtn = Array.from(container.querySelectorAll("button")).find(
      (b) => b.textContent === "Compare",
    );
    expect(compareBtn?.hasAttribute("disabled")).toBe(true);
  });

  it("Compare button enabled when 2+ selected; clicking it calls onStartCompare + onClose", () => {
    const onStartCompare = vi.fn();
    const onClose = vi.fn();
    const reports = [rs("AAPL", "2026-05-20"), rs("MSFT", "2026-05-19")];
    const { container } = render(
      <ReportCompareDrawer
        open
        onClose={onClose}
        reports={reports}
        selected={new Set(["AAPL|2026-05-20", "MSFT|2026-05-19"])}
        onToggle={() => {}}
        onStartCompare={onStartCompare}
        comparing={false}
      />,
    );
    const compareBtn = Array.from(container.querySelectorAll("button")).find(
      (b) => b.textContent === "Compare",
    );
    expect(compareBtn?.hasAttribute("disabled")).toBe(false);
    fireEvent.click(compareBtn!);
    expect(onStartCompare).toHaveBeenCalled();
    expect(onClose).toHaveBeenCalled();
  });

  it("Escape key calls onClose", () => {
    const onClose = vi.fn();
    render(
      <ReportCompareDrawer
        open
        onClose={onClose}
        reports={[]}
        selected={new Set()}
        onToggle={() => {}}
        onStartCompare={() => {}}
        comparing={false}
      />,
    );
    fireEvent.keyDown(window, { key: "Escape" });
    expect(onClose).toHaveBeenCalled();
  });

  it("Cancel button calls onClose", () => {
    const onClose = vi.fn();
    const { container } = render(
      <ReportCompareDrawer
        open
        onClose={onClose}
        reports={[]}
        selected={new Set()}
        onToggle={() => {}}
        onStartCompare={() => {}}
        comparing={false}
      />,
    );
    const cancelBtn = Array.from(container.querySelectorAll("button")).find(
      (b) => b.textContent === "Cancel",
    );
    fireEvent.click(cancelBtn!);
    expect(onClose).toHaveBeenCalled();
  });

  it("backdrop click calls onClose", () => {
    const onClose = vi.fn();
    const { container } = render(
      <ReportCompareDrawer
        open
        onClose={onClose}
        reports={[]}
        selected={new Set()}
        onToggle={() => {}}
        onStartCompare={() => {}}
        comparing={false}
      />,
    );
    const backdrop = container.querySelector('[aria-hidden="true"]') as HTMLElement;
    fireEvent.click(backdrop);
    expect(onClose).toHaveBeenCalled();
  });

  it("close button has aria-label", () => {
    const { container } = render(
      <ReportCompareDrawer
        open
        onClose={() => {}}
        reports={[]}
        selected={new Set()}
        onToggle={() => {}}
        onStartCompare={() => {}}
        comparing={false}
      />,
    );
    const closeBtn = container.querySelector('button[aria-label="Close compare drawer"]');
    expect(closeBtn).not.toBeNull();
  });
});
