import { useEffect, useRef, useState } from 'react';
import { useDataStore } from '../store/data';

const CATEGORY_COLOR: Record<string, string> = {
  article:       '#94a3b8',
  common_noun:   '#3b82f6',
  verb_present:  '#10b981',
  verb_past:     '#059669',
  adjective:     '#f59e0b',
  pronoun:       '#a855f7',
  name:          '#ec4899',
  function_word: '#6b7280',
  number:        '#ef4444',
  punctuation:   '#0ea5e9',
  rare:          '#d1d5db',
};
const DEFAULT_COLOR = '#94a3b8';

const ANIM_DURATION_MS = 200;
const POINT_RADIUS = 3;
const HOVER_RADIUS = 14;

const CANVAS_W = 720;
const CANVAS_H = 520;
const PAD = 16;
const DPR = Math.min(typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1, 2);

export function EmbeddingView() {
  const currentStep = useDataStore((s) => s.currentStep);
  const ensureEmbedding = useDataStore((s) => s.ensureEmbedding);
  const embeddingCache = useDataStore((s) => s.embeddingCache);
  const toi = useDataStore((s) => s.tokensOfInterest);
  const manifest = useDataStore((s) => s.manifest);
  const embeddingCacheVersion = useDataStore((s) => s.embeddingCacheVersion);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<{
    from: Float32Array | null;
    to: Float32Array | null;
    startTime: number;
    rafId: number | null;
    currentDisplay: Float32Array | null;
  }>({ from: null, to: null, startTime: 0, rafId: null, currentDisplay: null });

  const redrawRef = useRef<((hoveredIdx: number) => void) | null>(null);

  const [hover, setHover] = useState<{ index: number; x: number; y: number } | null>(null);

  useEffect(() => {
    void ensureEmbedding(currentStep);
  }, [currentStep, ensureEmbedding]);

  useEffect(() => {
    const data = embeddingCache.get(currentStep);
    if (!data) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    if (!manifest) return;

    const state = animRef.current;
    if (state.currentDisplay && state.currentDisplay.length === data.coords.length) {
      state.from = new Float32Array(state.currentDisplay);
    } else {
      state.from = new Float32Array(data.coords);
      state.currentDisplay = new Float32Array(data.coords.length);
    }
    state.to = data.coords;
    state.startTime = performance.now();

    const xrange = manifest.viz_settings.embedding_xrange;
    const yrange = manifest.viz_settings.embedding_yrange;
    const dx = xrange[1] - xrange[0];
    const dy = yrange[1] - yrange[0];

    const projX = (vx: number) => PAD + ((vx - xrange[0]) / dx) * (CANVAS_W - 2 * PAD);
    const projY = (vy: number) => (CANVAS_H - PAD) - ((vy - yrange[0]) / dy) * (CANVAS_H - 2 * PAD);

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);

    const categories = toi?.tokens.map((t) => t.category) ?? [];

    const drawFromDisplay = (hoveredIdx: number) => {
      const display = state.currentDisplay;
      if (!display) return;
      ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);
      ctx.strokeStyle = 'rgba(120,120,120,0.15)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(PAD, projY(0)); ctx.lineTo(CANVAS_W - PAD, projY(0));
      ctx.moveTo(projX(0), PAD); ctx.lineTo(projX(0), CANVAS_H - PAD);
      ctx.stroke();

      const n = display.length / 2;
      for (let i = 0; i < n; i++) {
        if (i === hoveredIdx) continue;
        const px = projX(display[i * 2]);
        const py = projY(display[i * 2 + 1]);
        const cat = categories[i] ?? 'rare';
        ctx.fillStyle = CATEGORY_COLOR[cat] ?? DEFAULT_COLOR;
        ctx.beginPath();
        ctx.arc(px, py, POINT_RADIUS, 0, Math.PI * 2);
        ctx.fill();
      }

      if (hoveredIdx >= 0 && hoveredIdx < n) {
        const px = projX(display[hoveredIdx * 2]);
        const py = projY(display[hoveredIdx * 2 + 1]);
        const cat = categories[hoveredIdx] ?? 'rare';
        ctx.fillStyle = CATEGORY_COLOR[cat] ?? DEFAULT_COLOR;
        ctx.beginPath();
        ctx.arc(px, py, POINT_RADIUS + 1, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(px, py, POINT_RADIUS + 3, 0, Math.PI * 2);
        ctx.stroke();
      }
    };

    redrawRef.current = drawFromDisplay;

    const draw = (t: number) => {
      const from = state.from!;
      const to = state.to!;
      const n = from.length / 2;

      if (!state.currentDisplay || state.currentDisplay.length !== from.length) {
        state.currentDisplay = new Float32Array(from.length);
      }
      for (let i = 0; i < n; i++) {
        const fx = from[i * 2], fy = from[i * 2 + 1];
        const tx = to[i * 2],   ty = to[i * 2 + 1];
        state.currentDisplay[i * 2]     = fx + (tx - fx) * t;
        state.currentDisplay[i * 2 + 1] = fy + (ty - fy) * t;
      }
      drawFromDisplay(-1);
    };

    const tick = (now: number) => {
      const elapsed = now - state.startTime;
      const t = Math.min(1, elapsed / ANIM_DURATION_MS);
      draw(t);
      if (t < 1) {
        state.rafId = requestAnimationFrame(tick);
      } else {
        state.rafId = null;
        draw(1);
      }
    };

    if (state.rafId !== null) cancelAnimationFrame(state.rafId);
    state.rafId = requestAnimationFrame(tick);

    return () => {
      if (state.rafId !== null) {
        cancelAnimationFrame(state.rafId);
        state.rafId = null;
      }
    };
  }, [currentStep, embeddingCache, embeddingCacheVersion, manifest, toi]);

  useEffect(() => {
    redrawRef.current?.(hover?.index ?? -1);
  }, [hover, manifest, toi]);

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !manifest) return;
    const rect = canvas.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * CANVAS_W;
    const py = ((e.clientY - rect.top) / rect.height) * CANVAS_H;
    const display = animRef.current.currentDisplay;
    if (!display) { setHover(null); return; }
    const xrange = manifest.viz_settings.embedding_xrange;
    const yrange = manifest.viz_settings.embedding_yrange;
    const dx = xrange[1] - xrange[0];
    const dy = yrange[1] - yrange[0];
    const projX = (vx: number) => PAD + ((vx - xrange[0]) / dx) * (CANVAS_W - 2 * PAD);
    const projY = (vy: number) => (CANVAS_H - PAD) - ((vy - yrange[0]) / dy) * (CANVAS_H - 2 * PAD);

    let bestIdx = -1, bestDist = HOVER_RADIUS * HOVER_RADIUS;
    const n = display.length / 2;
    for (let i = 0; i < n; i++) {
      const dxp = projX(display[i * 2]) - px;
      const dyp = projY(display[i * 2 + 1]) - py;
      const d2 = dxp * dxp + dyp * dyp;
      if (d2 < bestDist) { bestDist = d2; bestIdx = i; }
    }
    if (bestIdx >= 0) {
      setHover({ index: bestIdx, x: e.clientX - rect.left, y: e.clientY - rect.top });
    } else {
      setHover(null);
    }
  };

  const onPointerLeave = () => setHover(null);

  const hoveredToken = hover && toi ? toi.tokens[hover.index] : null;
  const usedCategories = toi
    ? Array.from(new Set(toi.tokens.map((t) => t.category)))
    : [];

  return (
    <section>
      <p className="mb-3 text-xs text-neutral-500 dark:text-neutral-500">
        211 GPT-2 tokens projected to 2D via PCA. Axes fixed to the final-step
        range; early in training, tokens cluster at the origin and spread as
        the model learns categorical structure.
      </p>
      <div className="relative inline-block w-full">
        <canvas
          ref={canvasRef}
          width={CANVAS_W * DPR}
          height={CANVAS_H * DPR}
          className="block w-full max-w-full rounded border border-neutral-200 bg-white dark:border-neutral-800 dark:bg-neutral-950"
          style={{ aspectRatio: `${CANVAS_W} / ${CANVAS_H}` }}
          onPointerMove={onPointerMove}
          onPointerLeave={onPointerLeave}
        />
        {hoveredToken && hover && (
          <div
            className="pointer-events-none absolute -translate-x-1/2 -translate-y-full rounded bg-neutral-900 px-2 py-1 font-mono text-xs text-white shadow dark:bg-neutral-100 dark:text-neutral-900"
            style={{ left: hover.x, top: hover.y - 8 }}
          >
            &quot;{hoveredToken.token_text}&quot; · {hoveredToken.category}
          </div>
        )}
      </div>
      <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-2 text-xs">
        {usedCategories.map((c) => (
          <span key={c} className="inline-flex items-center gap-1.5 text-neutral-600 dark:text-neutral-400">
            <span
              className="inline-block h-2.5 w-2.5 rounded-full"
              style={{ backgroundColor: CATEGORY_COLOR[c] ?? DEFAULT_COLOR }}
            />
            {c.replace(/_/g, ' ')}
          </span>
        ))}
      </div>
    </section>
  );
}
