// phase-73 fix (2026-05-26): backward-compat redirect to the new root
// location at /learnings (moved out of /paper-trading/ to break the
// double-chrome rendering bug -- cockpit layout wrapped this sub-route
// in cockpit chrome that didn't apply to a peer destination).

import { redirect } from "next/navigation";

export default function Page() {
  redirect("/learnings");
}
