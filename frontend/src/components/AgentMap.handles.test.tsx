/**
 * phase-80.3 -- AgentNode must render its own handles.
 *
 * React Flow only auto-creates handles for its BUILT-IN node types. This
 * graph uses a custom `agent` nodeType, and without handles
 * `getEdgePosition()` returns null and EVERY edge is silently dropped --
 * measured as 24 edges x 5 render passes = 120
 * "[React Flow]: Couldn't create edge for source handle id: null" warnings,
 * and a topology view that drew no topology.
 *
 * SCOPE, stated honestly: this file guards the STATIC contract only -- that
 * two handles exist, of the right types, in the right positions. It does NOT
 * assert that edges are drawn, and it cannot: React Flow measures real DOM
 * boxes, and jsdom has no layout. `vitest.setup.ts` ships a no-op
 * ResizeObserver whose callback never fires, and there is no
 * DOMMatrixReadOnly / offsetWidth / getBBox. A jsdom "edges are drawn"
 * assertion would be asserting on mocked geometry.
 *
 * Criteria 2, 3, 4 and 5 are therefore verified by PLAYWRIGHT against the
 * running app -- see handoff/current/live_check_80.3.md. This test exists so
 * that deleting the handles fails fast in CI rather than silently returning
 * the page to a zero-edge state.
 */
import type * as React from "react";

import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";

import { AgentNode } from "./AgentMap";

// `AgentNode` takes React Flow's full `NodeProps`, which carries ~15 required
// fields this guard has no use for. Render it through a narrow local alias so
// the test supplies only what it asserts on -- and so `tsc` stays clean, since
// the step's immutable verification command runs `tsc --noEmit` and a type
// error here would fail the whole gate.
const Node = AgentNode as unknown as (props: { data: unknown }) => React.ReactElement;

const NODE_DATA = {
  name: "main",
  kind: "harness",
  provider: "anthropic",
  model: "claude-opus-5",
  inWorkflow: false,
};

describe("AgentNode handles (phase-80.3)", () => {
  it("renders BOTH a target and a source handle", () => {
    const { container } = render(
      <ReactFlowProvider>
        <Node data={NODE_DATA} />
      </ReactFlowProvider>
    );

    const handles = container.querySelectorAll(".react-flow__handle");
    expect(
      handles.length,
      "AgentNode rendered no handles -- every edge will be dropped"
    ).toBe(2);

    const positions = Array.from(handles).map((h) =>
      h.getAttribute("data-handlepos")
    );
    expect(positions).toContain("top");
    expect(positions).toContain("bottom");

    // Pin the PAIRING, not just presence: for a dagre "TB" layout the target
    // belongs at the top and the source at the bottom. Swapping them still
    // binds edges, so this is the only thing that catches it.
    const target = container.querySelector(".react-flow__handle.target");
    const source = container.querySelector(".react-flow__handle.source");
    expect(target).not.toBeNull();
    expect(source).not.toBeNull();
    expect(target!.getAttribute("data-handlepos")).toBe("top");
    expect(source!.getAttribute("data-handlepos")).toBe("bottom");

    // Keep the handles UNNAMED. NOTE, measured rather than assumed: naming
    // them does NOT drop the edges -- getHandle is
    // `(!handleId ? bounds[0] : bounds.find(...)) || null`, and every edge
    // here omits sourceHandle, so a falsy handleId takes bounds[0] whatever
    // the id is. This is hygiene: unnamed is the simplest configuration that
    // provably binds. (Q/A cycle 2 executed getEdgePosition both ways and got
    // identical output.)
    for (const h of Array.from(handles)) {
      const id = h.getAttribute("data-handleid");
      expect(
        id === null || id === "",
        `handle has id=${String(id)} -- keep handles UNNAMED. Naming does NOT break binding (getHandle falls back to bounds[0] for a falsy handleId); unnamed is simply the configuration every edge here assumes, since they omit sourceHandle/targetHandle`
      ).toBe(true);
    }
  });

  it("does not hide the handles with display:none", () => {
    // Measured, not assumed: `display:none` does NOT zero out the edges --
    // getHandleBounds returns null only on ZERO querySelectorAll matches. It
    // yields a zero-size rect, so edges anchor at the node origin: MIS-DRAWN,
    // not absent. Still worth pinning; `visibility:hidden`/`opacity:0` are the
    // correct ways to hide a handle.
    const { container } = render(
      <ReactFlowProvider>
        <Node data={NODE_DATA} />
      </ReactFlowProvider>
    );
    const handles = container.querySelectorAll(".react-flow__handle");
    // Without this the loop can no-op over an empty NodeList and assert
    // nothing -- the same empty-iteration vacuity found in phase-80.31.
    expect(handles.length).toBe(2);
    for (const h of Array.from(handles)) {
      const el = h as HTMLElement;
      expect(el.style.display).not.toBe("none");
      // Word-boundary match with an optional Tailwind `!` important modifier:
      // `overflow-hidden` is benign, while a bare `hidden` OR `!hidden` is not.
      expect(/(^|\s)!?hidden(\s|$)/.test(el.className)).toBe(false);
    }
  });

  it("still renders the node's own content", () => {
    render(
      <ReactFlowProvider>
        <Node data={NODE_DATA} />
      </ReactFlowProvider>
    );
    // guards against a future "fix" that returns only handles
    expect(screen.getByTestId("agent-node")).toBeTruthy();
  });
});
