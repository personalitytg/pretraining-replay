import { useMemo } from 'react';

import { PAD_X, VIEW_W, stepToX } from '../lib/timeline';
import { useDataStore } from '../store/data';
import type { Probe } from '../types';

const SPARK_H = 56;
const SPARK_PAD_TOP = 6;
const SPARK_PAD_BOTTOM = 6;
const SPARK_CURVE_H = SPARK_H - SPARK_PAD_TOP - SPARK_PAD_BOTTOM;

function valueToY(
  val: number,
  kind: 'fraction' | 'number' | 'boolean',
  minV: number,
  maxV: number,
): number {
  if (kind === 'fraction') {
    const clamped = Math.max(0, Math.min(1, val));
    return SPARK_PAD_TOP + (1 - clamped) * SPARK_CURVE_H;
  }
  if (kind === 'boolean') {
    return SPARK_PAD_TOP + (val ? 0 : SPARK_CURVE_H);
  }
  const span = maxV - minV || 1;
  const t = (val - minV) / span;
  return SPARK_PAD_TOP + (1 - t) * SPARK_CURVE_H;
}

function formatValue(val: number | boolean, kind: 'fraction' | 'number' | 'boolean'): string {
  if (kind === 'boolean') return (val as boolean) ? 'pass' : 'fail';
  if (kind === 'fraction') return (val as number).toFixed(2);
  return Math.round(val as number).toLocaleString('en-US');
}

function valueColor(
  val: number | boolean,
  kind: 'fraction' | 'number' | 'boolean',
  threshold: number | null,
): string {
  if (kind === 'fraction' && threshold !== null) {
    return (val as number) >= threshold
      ? 'text-emerald-600 dark:text-emerald-400'
      : 'text-amber-600 dark:text-amber-400';
  }
  if (kind === 'boolean') {
    return (val as boolean)
      ? 'text-emerald-600 dark:text-emerald-400'
      : 'text-neutral-500 dark:text-neutral-400';
  }
  return 'text-neutral-700 dark:text-neutral-200';
}

function ProbeCard({ probe }: { probe: Probe }) {
  const manifest = useDataStore((s) => s.manifest)!;
  const currentStep = useDataStore((s) => s.currentStep);
  const checkpoints = manifest.checkpoints;
  const series = manifest.probe_sparklines[probe.id] ?? [];

  const idxRaw = checkpoints.findIndex((c) => c.step === currentStep);
  const idx = idxRaw < 0 ? 0 : idxRaw;
  const curVal = series[idx];

  const { minV, maxV } = useMemo(() => {
    if (probe.kind !== 'number') return { minV: 0, maxV: 1 };
    let mn = Number.POSITIVE_INFINITY;
    let mx = Number.NEGATIVE_INFINITY;
    for (const v of series) {
      if (typeof v === 'number') {
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
    }
    if (!Number.isFinite(mn)) mn = 0;
    if (!Number.isFinite(mx)) mx = 1;
    return { minV: mn, maxV: mx };
  }, [series, probe.kind]);

  const curX = stepToX(currentStep);
  const curIsActive =
    probe.kind === 'fraction' && probe.threshold !== null
      ? (curVal as number) >= probe.threshold
      : probe.kind === 'boolean'
        ? (curVal as boolean)
        : true;

  const pathColorClass =
    probe.kind === 'fraction'
      ? curIsActive
        ? 'stroke-emerald-500 dark:stroke-emerald-400'
        : 'stroke-amber-500 dark:stroke-amber-400'
      : probe.kind === 'boolean'
        ? 'stroke-neutral-400 dark:stroke-neutral-600'
        : 'stroke-neutral-500 dark:stroke-neutral-400';

  let pathD = '';
  if (probe.kind !== 'boolean') {
    pathD = checkpoints
      .map((c, i) => {
        const v = series[i];
        if (typeof v !== 'number') return '';
        const x = stepToX(c.step);
        const y = valueToY(v, probe.kind, minV, maxV);
        return `${i === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`;
      })
      .filter((s) => s.length > 0)
      .join(' ');
  }

  const thresholdY =
    probe.kind === 'fraction' && probe.threshold !== null
      ? SPARK_PAD_TOP + (1 - probe.threshold) * SPARK_CURVE_H
      : null;

  const curY =
    typeof curVal === 'number'
      ? valueToY(curVal, probe.kind, minV, maxV)
      : valueToY(curVal === true ? 1 : 0, 'boolean', 0, 1);

  return (
    <article className="rounded-lg border border-neutral-200 p-4 dark:border-neutral-800">
      <header className="flex items-baseline justify-between">
        <h3 className="text-sm text-neutral-900 dark:text-neutral-100">{probe.title}</h3>
        <span className={'font-mono text-sm ' + valueColor(curVal, probe.kind, probe.threshold)}>
          {curVal === undefined ? '—' : formatValue(curVal, probe.kind)}
        </span>
      </header>
      <svg
        viewBox={`0 0 ${VIEW_W} ${SPARK_H}`}
        className="mt-2 block w-full"
        preserveAspectRatio="none"
      >
        <line
          x1={PAD_X}
          x2={VIEW_W - PAD_X}
          y1={SPARK_PAD_TOP + SPARK_CURVE_H}
          y2={SPARK_PAD_TOP + SPARK_CURVE_H}
          className="stroke-neutral-200 dark:stroke-neutral-800"
          strokeWidth={1}
        />
        {thresholdY !== null && (
          <line
            x1={PAD_X}
            x2={VIEW_W - PAD_X}
            y1={thresholdY}
            y2={thresholdY}
            className="stroke-neutral-300 dark:stroke-neutral-700"
            strokeWidth={1}
            strokeDasharray="3,3"
          />
        )}
        {probe.kind === 'boolean' ? (
          checkpoints.map((c, i) => {
            if (i === checkpoints.length - 1) return null;
            const next = checkpoints[i + 1];
            const v = series[i];
            const isPass = v === true;
            return (
              <line
                key={c.step}
                x1={stepToX(c.step)}
                x2={stepToX(next.step)}
                y1={SPARK_PAD_TOP + SPARK_CURVE_H / 2}
                y2={SPARK_PAD_TOP + SPARK_CURVE_H / 2}
                className={
                  isPass
                    ? 'stroke-emerald-500 dark:stroke-emerald-400'
                    : 'stroke-neutral-300 dark:stroke-neutral-700'
                }
                strokeWidth={4}
              />
            );
          })
        ) : (
          <path
            d={pathD}
            className={pathColorClass}
            fill="none"
            strokeWidth={1.5}
            strokeLinejoin="round"
            strokeLinecap="round"
            vectorEffect="non-scaling-stroke"
          />
        )}
        <line
          x1={curX}
          x2={curX}
          y1={SPARK_PAD_TOP}
          y2={SPARK_PAD_TOP + SPARK_CURVE_H}
          className="stroke-blue-500/50 dark:stroke-blue-400/50"
          strokeWidth={1}
        />
        <circle
          cx={curX}
          cy={curY}
          r={3}
          className="fill-blue-500 stroke-white dark:fill-blue-400 dark:stroke-neutral-900"
          strokeWidth={1.5}
          vectorEffect="non-scaling-stroke"
        />
      </svg>
    </article>
  );
}

export function ProbeView() {
  const manifest = useDataStore((s) => s.manifest);
  if (!manifest) {
    return <div className="text-sm text-neutral-500 dark:text-neutral-500">loading manifest...</div>;
  }
  return (
    <section className="space-y-4">
      {manifest.probes.map((probe) => (
        <ProbeCard key={probe.id} probe={probe} />
      ))}
    </section>
  );
}
