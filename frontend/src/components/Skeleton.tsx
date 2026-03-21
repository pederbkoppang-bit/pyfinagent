import { clsx } from "clsx";

/* ── Atomic skeletons ── */

export function SkeletonPulse({ className }: { className?: string }) {
  return (
    <div
      className={clsx(
        "animate-pulse rounded-[var(--radius-card,12px)] bg-slate-700/40",
        className,
      )}
    />
  );
}

/** Card-sized skeleton with optional height */
export function SkeletonCard({ className, h = "h-28" }: { className?: string; h?: string }) {
  return (
    <SkeletonPulse
      className={clsx(
        "w-full",
        h,
        className,
      )}
    />
  );
}

/* ── Composite skeletons for page layouts ── */

/** Grid of N skeleton cards (e.g. summary metric cards) */
export function SkeletonGrid({ count = 4, h = "h-28" }: { count?: number; h?: string }) {
  return (
    <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
      {Array.from({ length: count }).map((_, i) => (
        <SkeletonCard key={i} h={h} />
      ))}
    </div>
  );
}

/** Full page skeleton: metric grid + tall content area */
export function PageSkeleton() {
  return (
    <div className="space-y-6">
      <SkeletonGrid count={4} />
      <SkeletonCard h="h-64" />
    </div>
  );
}
