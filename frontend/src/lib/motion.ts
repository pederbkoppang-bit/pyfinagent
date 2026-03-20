/**
 * Shared motion variants and spring presets for consistent animations.
 * Usage: import { fadeIn, staggerContainer } from "@/lib/motion";
 */
import type { Variants, Transition } from "motion/react";

// ── Spring presets ──
export const springSnappy: Transition = {
  type: "spring",
  stiffness: 300,
  damping: 24,
};

export const springGentle: Transition = {
  type: "spring",
  stiffness: 100,
  damping: 20,
};

// ── Fade in ──
export const fadeIn: Variants = {
  hidden: { opacity: 0 },
  visible: { opacity: 1, transition: { duration: 0.3 } },
};

// ── Slide up + fade ──
export const slideUp: Variants = {
  hidden: { opacity: 0, y: 12 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { type: "spring", stiffness: 300, damping: 24 },
  },
};

// ── Staggered container ──
export const staggerContainer: Variants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.04,
      delayChildren: 0.1,
    },
  },
};

// ── Staggered item (child) ──
export const staggerItem: Variants = {
  hidden: { opacity: 0, y: 12 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { type: "spring", stiffness: 300, damping: 24 },
  },
};

// ── Scale on hover/tap ──
export const hoverTap = {
  whileHover: { scale: 1.02 },
  whileTap: { scale: 0.98 },
  transition: { type: "spring", stiffness: 400, damping: 17 } as Transition,
};
