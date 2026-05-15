export const VIEW_W = 1000;
export const VIEW_H = 80;
export const PAD_X = 10;
export const PAD_TOP = 8;
export const PAD_BOTTOM = 14;
export const USABLE_W = VIEW_W - PAD_X * 2;
export const CURVE_H = VIEW_H - PAD_TOP - PAD_BOTTOM;
export const BASE_PADDING = 5;

export const SEGMENTS = [
  { from: 0, to: 100, weight: 21 },
  { from: 100, to: 1000, weight: 36 },
  { from: 1000, to: 10000, weight: 90 },
  { from: 10000, to: 50000, weight: 80 },
] as const;

export const SEG_LAYOUT = (() => {
  const totalWeight = SEGMENTS.reduce((s, seg) => s + seg.weight + BASE_PADDING, 0);
  let acc = PAD_X;
  return SEGMENTS.map((seg) => {
    const w = ((seg.weight + BASE_PADDING) / totalWeight) * USABLE_W;
    const start = acc;
    acc += w;
    return { ...seg, xStart: start, xEnd: acc };
  });
})();

export function stepToX(step: number): number {
  if (step <= SEG_LAYOUT[0].from) return SEG_LAYOUT[0].xStart;
  const last = SEG_LAYOUT[SEG_LAYOUT.length - 1];
  if (step >= last.to) return last.xEnd;
  for (const seg of SEG_LAYOUT) {
    if (step >= seg.from && step <= seg.to) {
      const frac = (step - seg.from) / (seg.to - seg.from);
      return seg.xStart + frac * (seg.xEnd - seg.xStart);
    }
  }
  return last.xEnd;
}

export function lossToY(loss: number, logMin: number, logMax: number): number {
  const clamped = Math.max(loss, Math.exp(logMin));
  const t = (Math.log(clamped) - logMin) / (logMax - logMin || 1);
  return PAD_TOP + (1 - t) * CURVE_H;
}
