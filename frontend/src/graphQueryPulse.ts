/** Query “neural activity” pulse length (ms), ~1–2s */
export const QUERY_PULSE_TOTAL_MS = 1650;

/**
 * Scale multiplier for impacted nodes: brief swell, decaying ripples, settles at 1.
 * Smooth in time for use with requestAnimationFrame-driven redraws.
 */
export function queryPulseScale(elapsed: number): number {
  if (elapsed <= 0) return 1;
  if (elapsed >= QUERY_PULSE_TOTAL_MS) return 1;

  const e = elapsed;
  const t = e / QUERY_PULSE_TOTAL_MS;
  const decay = Math.exp(-e / 420);

  const swell = 1 + 0.11 * Math.sin(Math.min(1, e / 135) * Math.PI) * (1 - 0.35 * t);
  const waveA = 1 + 0.048 * Math.sin(e / 52) * decay;
  const waveB = 1 + 0.028 * Math.sin(e / 36 + 0.7) * Math.exp(-e / 620);
  const settle = 1 + (1 - t) ** 2 * 0.012 * Math.sin(e / 28);

  return swell * waveA * waveB * settle;
}
