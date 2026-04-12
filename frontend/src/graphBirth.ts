/** Total time the birth animation runs (ms) */
export const BIRTH_TOTAL_MS = 2000;

const GROW_MS = 880;

/** Overshoot ease for “pop in” */
export function easeOutBack(t: number): number {
  const c1 = 1.60158;
  const c3 = c1 + 1;
  const x = Math.min(1, Math.max(0, t));
  return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2;
}

/**
 * Scale (visual size) and neon tint alpha for a node birth / “learned” pulse.
 * Smooth, no discrete jumps — driven by elapsed ms from animation start.
 */
export function birthVisual(elapsed: number): { scale: number; tintAlpha: number } {
  if (elapsed <= 0) {
    return { scale: 0.18, tintAlpha: 0.92 };
  }
  if (elapsed >= BIRTH_TOTAL_MS) {
    return { scale: 1, tintAlpha: 0 };
  }

  if (elapsed < GROW_MS) {
    const t = elapsed / GROW_MS;
    const scale = 0.18 + 0.82 * easeOutBack(t);
    const tintAlpha = 0.88 * (1 - t * 0.55);
    return { scale, tintAlpha };
  }

  const settle = elapsed - GROW_MS;
  const span = BIRTH_TOTAL_MS - GROW_MS;
  const u = Math.min(1, settle / span);
  const pulse = 1 + 0.055 * Math.sin(settle / 42) * (1 - u);
  const scale = pulse * (0.98 + 0.02 * (1 - u));
  const tintAlpha = 0.42 * (1 - u) ** 1.35;
  return { scale, tintAlpha };
}
