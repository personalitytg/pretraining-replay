import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import {
  CURVE_H,
  PAD_TOP,
  PAD_X,
  SEG_LAYOUT,
  VIEW_H,
  VIEW_W,
  lossToY,
  stepToX,
} from '../lib/timeline';
import { useDataStore } from '../store/data';
import type { CheckpointMeta } from '../types';

function nearestCheckpoint(checkpoints: CheckpointMeta[], targetX: number): CheckpointMeta {
  let best = checkpoints[0];
  let bestDist = Math.abs(stepToX(best.step) - targetX);
  for (let i = 1; i < checkpoints.length; i++) {
    const d = Math.abs(stepToX(checkpoints[i].step) - targetX);
    if (d < bestDist) {
      bestDist = d;
      best = checkpoints[i];
    }
  }
  return best;
}

function PlayIcon() {
  return (
    <svg viewBox="0 0 20 20" width="16" height="16" aria-hidden="true">
      <polygon points="5,3 5,17 17,10" fill="currentColor" />
    </svg>
  );
}

function PauseIcon() {
  return (
    <svg viewBox="0 0 20 20" width="16" height="16" aria-hidden="true">
      <rect x="5" y="3" width="3.5" height="14" fill="currentColor" />
      <rect x="11.5" y="3" width="3.5" height="14" fill="currentColor" />
    </svg>
  );
}

function TourIcon() {
  return (
    <svg viewBox="0 0 20 20" width="14" height="14" aria-hidden="true">
      <polygon points="3,3 3,17 11,10" fill="currentColor" />
      <polygon points="11,3 11,17 19,10" fill="currentColor" />
    </svg>
  );
}

const TOUR_INTERVAL_MS = 8000;

const PLAY_INTERVAL_MS = 220;

export function Scrubber() {
  const manifest = useDataStore((s) => s.manifest);
  const discoveriesRaw = useDataStore((s) => s.discoveries);
  const discoveries = useMemo(() => discoveriesRaw ?? [], [discoveriesRaw]);
  const currentStep = useDataStore((s) => s.currentStep);
  const setStep = useDataStore((s) => s.setStep);
  const isPlaying = useDataStore((s) => s.isPlaying);
  const togglePlay = useDataStore((s) => s.togglePlay);

  const [tourIndex, setTourIndex] = useState<number | null>(null);

  const [discoveryHover, setDiscoveryHover] = useState<{ id: string; x: number; title: string } | null>(
    null,
  );

  const svgRef = useRef<SVGSVGElement>(null);
  const draggingRef = useRef(false);
  const rafRef = useRef<number | null>(null);
  const pendingStepRef = useRef<number | null>(null);
  const [hover, setHover] = useState<{ step: number; loss: number; x: number; y: number } | null>(
    null,
  );

  const checkpoints = useMemo<CheckpointMeta[]>(
    () => manifest?.checkpoints ?? [],
    [manifest],
  );

  const { logMin, logMax, curvePath } = useMemo(() => {
    if (checkpoints.length === 0) return { logMin: 0, logMax: 1, curvePath: '' };
    const losses = checkpoints.map((c) => c.loss_train).filter((v) => v > 0);
    const minL = Math.min(...losses);
    const maxL = Math.max(...losses);
    const lMin = Math.log(minL);
    const lMax = Math.log(maxL);
    const path = checkpoints
      .map((c, i) => {
        const x = stepToX(c.step);
        const y = lossToY(c.loss_train, lMin, lMax);
        return `${i === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`;
      })
      .join(' ');
    return { logMin: lMin, logMax: lMax, curvePath: path };
  }, [checkpoints]);

  const clientToSvgX = useCallback((clientX: number): number => {
    const svg = svgRef.current;
    if (!svg) return 0;
    const rect = svg.getBoundingClientRect();
    const ratio = (clientX - rect.left) / rect.width;
    return Math.max(PAD_X, Math.min(VIEW_W - PAD_X, ratio * VIEW_W));
  }, []);

  const flushPendingStep = useCallback(() => {
    rafRef.current = null;
    if (pendingStepRef.current !== null) {
      setStep(pendingStepRef.current);
      pendingStepRef.current = null;
    }
  }, [setStep]);

  const scheduleStep = useCallback(
    (step: number) => {
      pendingStepRef.current = step;
      if (rafRef.current === null) {
        rafRef.current = requestAnimationFrame(flushPendingStep);
      }
    },
    [flushPendingStep],
  );

  useEffect(() => {
    if (tourIndex === null) return;
    if (discoveries.length === 0) {
      setTourIndex(null);
      return;
    }
    const d = discoveries[tourIndex];
    if (!d) {
      setTourIndex(null);
      return;
    }
    useDataStore.getState().setStep(d.step);
    const t = setTimeout(() => {
      if (tourIndex + 1 < discoveries.length) {
        setTourIndex(tourIndex + 1);
      } else {
        setTourIndex(null);
      }
    }, TOUR_INTERVAL_MS);
    return () => clearTimeout(t);
  }, [tourIndex, discoveries]);

  useEffect(() => {
    if (!isPlaying || checkpoints.length === 0) return;
    const interval = setInterval(() => {
      const state = useDataStore.getState();
      const cur = state.currentStep;
      const i = checkpoints.findIndex((c) => c.step === cur);
      if (i < 0 || i >= checkpoints.length - 1) {
        state.togglePlay();
        return;
      }
      state.setStep(checkpoints[i + 1].step);
      const view = state.currentView;
      for (let k = 2; k <= 6 && i + k < checkpoints.length; k++) {
        const s = checkpoints[i + k].step;
        void state.ensureCheckpoint(s);
        if (view === 'attention') void state.ensureAttention(s);
        if (view === 'embedding') void state.ensureEmbedding(s);
      }
    }, PLAY_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [isPlaying, checkpoints]);

  useEffect(() => {
    if (checkpoints.length === 0) return;

    const onMove = (e: PointerEvent) => {
      if (!draggingRef.current) return;
      const xSvg = clientToSvgX(e.clientX);
      const c = nearestCheckpoint(checkpoints, xSvg);
      scheduleStep(c.step);
    };
    const onUp = () => {
      draggingRef.current = false;
    };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    window.addEventListener('pointercancel', onUp);
    return () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      window.removeEventListener('pointercancel', onUp);
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [checkpoints, clientToSvgX, scheduleStep]);

  if (!manifest || checkpoints.length === 0) return null;

  const totalSteps = manifest.training.total_steps;
  const currentMeta = checkpoints.find((c) => c.step === currentStep) ?? checkpoints[0];
  const thumbX = stepToX(currentMeta.step);
  const thumbY = lossToY(currentMeta.loss_train, logMin, logMax);

  const onPointerDown = (e: React.PointerEvent<SVGSVGElement>) => {
    e.preventDefault();
    if (tourIndex !== null) setTourIndex(null);
    if (isPlaying) togglePlay();
    draggingRef.current = true;
    const xSvg = clientToSvgX(e.clientX);
    const c = nearestCheckpoint(checkpoints, xSvg);
    scheduleStep(c.step);
  };

  const onPointerMoveHover = (e: React.PointerEvent<SVGSVGElement>) => {
    if (draggingRef.current) return;
    const xSvg = clientToSvgX(e.clientX);
    const c = nearestCheckpoint(checkpoints, xSvg);
    setHover({
      step: c.step,
      loss: c.loss_train,
      x: stepToX(c.step),
      y: lossToY(c.loss_train, logMin, logMax),
    });
  };

  const onPointerLeave = () => {
    if (!draggingRef.current) setHover(null);
  };

  const dividers = SEG_LAYOUT.slice(0, -1).map((seg) => seg.xEnd);
  const dividerLabels: { x: number; label: string }[] = [
    { x: SEG_LAYOUT[0].xEnd, label: '100' },
    { x: SEG_LAYOUT[1].xEnd, label: '1k' },
    { x: SEG_LAYOUT[2].xEnd, label: '10k' },
  ];

  return (
    <div className="relative w-full select-none px-4 py-6">
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={() => {
            if (tourIndex !== null) setTourIndex(null);
            togglePlay();
          }}
          aria-label={isPlaying ? 'Pause' : 'Play'}
          className="focus-ring flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-neutral-300 text-neutral-700 transition-colors hover:border-blue-500 hover:text-blue-500 dark:border-neutral-700 dark:text-neutral-300 dark:hover:border-blue-400 dark:hover:text-blue-400"
        >
          {isPlaying ? <PauseIcon /> : <PlayIcon />}
        </button>
        <button
          type="button"
          onClick={() => {
            if (tourIndex !== null) {
              setTourIndex(null);
              return;
            }
            if (discoveries.length === 0) return;
            if (isPlaying) togglePlay();
            setTourIndex(0);
          }}
          aria-label={tourIndex !== null ? 'Exit tour' : 'Tour discoveries'}
          disabled={discoveries.length === 0}
          className={
            'focus-ring flex h-8 w-8 shrink-0 items-center justify-center rounded-full border transition-colors disabled:opacity-30 ' +
            (tourIndex !== null
              ? 'border-emerald-500 bg-emerald-500 text-white dark:border-emerald-400 dark:bg-emerald-400 dark:text-neutral-900'
              : 'border-neutral-300 text-neutral-700 hover:border-emerald-500 hover:text-emerald-500 dark:border-neutral-700 dark:text-neutral-300 dark:hover:border-emerald-400 dark:hover:text-emerald-400')
          }
        >
          <TourIcon />
        </button>
        <div className="relative flex-1">
      <svg
        ref={svgRef}
        viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
        className="block w-full touch-none cursor-pointer"
        preserveAspectRatio="none"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMoveHover}
        onPointerLeave={onPointerLeave}
      >
        <line
          x1={PAD_X}
          x2={VIEW_W - PAD_X}
          y1={PAD_TOP + CURVE_H}
          y2={PAD_TOP + CURVE_H}
          className="stroke-neutral-200 dark:stroke-neutral-800"
          strokeWidth={1}
        />
        {dividers.map((x) => (
          <line
            key={x}
            x1={x}
            x2={x}
            y1={PAD_TOP}
            y2={PAD_TOP + CURVE_H}
            className="stroke-neutral-200 dark:stroke-neutral-800"
            strokeWidth={1}
            strokeDasharray="2,2"
          />
        ))}
        <path
          d={curvePath}
          className="stroke-neutral-300 dark:stroke-neutral-700"
          fill="none"
          strokeWidth={1.5}
          strokeLinejoin="round"
          strokeLinecap="round"
          vectorEffect="non-scaling-stroke"
        />
        {discoveries && discoveries.length > 0 && (
          <g>
            {discoveries.map((d) => {
              const cx = stepToX(d.step);
              const cy = PAD_TOP - 2;
              const isActive = d.step === currentStep;
              return (
                <g key={d.id}>
                  {isActive && (
                    <circle
                      cx={cx}
                      cy={cy}
                      r={8}
                      fill="none"
                      className="stroke-emerald-500/40 dark:stroke-emerald-400/40"
                      strokeWidth={2}
                      vectorEffect="non-scaling-stroke"
                    />
                  )}
                  <circle
                    cx={cx}
                    cy={cy}
                    r={4}
                    className="cursor-pointer fill-emerald-500 stroke-white dark:fill-emerald-400 dark:stroke-neutral-900"
                    strokeWidth={1.5}
                    vectorEffect="non-scaling-stroke"
                    onPointerDown={(e) => {
                      e.stopPropagation();
                      useDataStore.getState().setStep(d.step);
                    }}
                    onPointerEnter={() => setDiscoveryHover({ id: d.id, x: cx, title: d.title })}
                    onPointerLeave={() =>
                      setDiscoveryHover((h) => (h && h.id === d.id ? null : h))
                    }
                  />
                </g>
              );
            })}
          </g>
        )}
        <line
          x1={thumbX}
          x2={thumbX}
          y1={PAD_TOP}
          y2={PAD_TOP + CURVE_H}
          className="stroke-blue-500/40 dark:stroke-blue-400/40"
          strokeWidth={1}
        />
        <circle
          cx={thumbX}
          cy={thumbY}
          r={6}
          className="fill-blue-500 stroke-white drop-shadow-sm dark:fill-blue-400 dark:stroke-neutral-900"
          strokeWidth={2}
          vectorEffect="non-scaling-stroke"
        />
        {dividerLabels.map((d) => (
          <text
            key={d.label}
            x={d.x}
            y={VIEW_H - 3}
            textAnchor="middle"
            className="fill-neutral-400 dark:fill-neutral-600"
            fontSize={9}
          >
            {d.label}
          </text>
        ))}
      </svg>

      {hover && svgRef.current && (
        <div
          className="pointer-events-none absolute -translate-x-1/2 -translate-y-full rounded bg-neutral-900 px-2 py-1 font-mono text-xs text-white shadow dark:bg-neutral-100 dark:text-neutral-900"
          style={{
            left: `${(hover.x / VIEW_W) * 100}%`,
            top: `${(hover.y / VIEW_H) * 100}%`,
          }}
        >
          step {hover.step.toLocaleString('en-US')} · loss {hover.loss.toFixed(2)}
        </div>
      )}
      {discoveryHover && svgRef.current && (
        <div
          className="pointer-events-none absolute -translate-x-1/2 -translate-y-full rounded bg-neutral-900 px-2 py-1 font-mono text-xs text-white shadow dark:bg-neutral-100 dark:text-neutral-900"
          style={{
            left: `${(discoveryHover.x / VIEW_W) * 100}%`,
            top: `${((PAD_TOP - 6) / VIEW_H) * 100}%`,
          }}
        >
          {discoveryHover.title}
        </div>
      )}
        </div>
      </div>

      <div className="mt-1 flex justify-between font-mono text-xs text-neutral-500 dark:text-neutral-500">
        <span className="hidden sm:inline">step 0</span>
        <span>
          step {currentMeta.step.toLocaleString('en-US')} · loss_train {currentMeta.loss_train.toFixed(2)}{' '}
          · loss_val {currentMeta.loss_val.toFixed(2)}
        </span>
        <span className="hidden sm:inline">step {totalSteps.toLocaleString('en-US')}</span>
      </div>
    </div>
  );
}
