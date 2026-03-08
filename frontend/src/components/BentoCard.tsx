"use client";

import { clsx } from "clsx";
import { ReactNode } from "react";

export function BentoCard({
  children,
  className,
  glow,
}: {
  children: ReactNode;
  className?: string;
  glow?: boolean;
}) {
  return (
    <div
      className={clsx(
        "rounded-2xl border border-navy-700 bg-navy-800/70 p-6 backdrop-blur-lg",
        glow && "alpha-score-glow",
        className
      )}
    >
      {children}
    </div>
  );
}
