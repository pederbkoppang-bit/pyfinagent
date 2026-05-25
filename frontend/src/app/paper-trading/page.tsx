// phase-44.2 -- /paper-trading index redirects to /paper-trading/positions.
//
// The cockpit is split into 6 sub-routes (positions/trades/nav/reality-gap/
// exit-quality/manage). The Sidebar entry still points at /paper-trading;
// this redirect preserves that link.

import { redirect } from "next/navigation";

export default function Page() {
  redirect("/paper-trading/positions");
}
