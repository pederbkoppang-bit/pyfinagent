"use client";

// phase-44.2 -- exit-quality sub-route. Hosts the MFE/MAE scatter.

import { MfeMaeScatter } from "@/components/MfeMaeScatter";

export default function ExitQualityPage() {
  return (
    <div
      role="tabpanel"
      id="panel-exit-quality"
      aria-labelledby="tab-exit-quality"
      tabIndex={0}
    >
      <MfeMaeScatter />
    </div>
  );
}
