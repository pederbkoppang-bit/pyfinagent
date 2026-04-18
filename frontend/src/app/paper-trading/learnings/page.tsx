"use client";

import { Sidebar } from "@/components/Sidebar";
import { VirtualFundLearnings } from "@/components/VirtualFundLearnings";

// phase-4.7.7: thin wrapper page hosting the VirtualFundLearnings
// component. Live data hookup (/api/paper-trading/learnings or
// similar) lands in a follow-up backend step; today the component
// renders empty states with honest "not yet collected" messaging.

export default function PaperTradingLearningsPage() {
  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <main className="flex flex-1 flex-col overflow-hidden">
        <div className="flex-1 overflow-y-auto scrollbar-thin px-6 py-6 md:px-8">
          <VirtualFundLearnings />
        </div>
      </main>
    </div>
  );
}
