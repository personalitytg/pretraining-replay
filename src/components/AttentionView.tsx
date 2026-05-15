import { useEffect, useRef, useState } from 'react';
import { useDataStore } from '../store/data';
import type { AttentionData } from '../types';

const CANONICAL_TOKENS_FALLBACK = ['Once', ' upon', ' a', ' time', ',', ' there', ' was', ' a'];

function tokenLabel(t: string): string {
  return t.startsWith(' ') ? t.slice(1) : t;
}

function getAttn(d: AttentionData, layer: number, head: number, query: number, key: number): number {
  const idx = ((layer * d.n_head + head) * d.seq_len + query) * d.seq_len + key;
  return d.values[idx];
}

interface Hover {
  layer: number;
  head: number;
  query: number;
  key: number;
}

function TokenBar({
  tokens,
  activeQuery,
  activeKey,
}: {
  tokens: readonly string[];
  activeQuery: number | null;
  activeKey: number | null;
}) {
  return (
    <div className="flex flex-wrap items-center gap-x-1 gap-y-1 font-mono text-xs">
      {tokens.map((tok, i) => {
        const isQ = activeQuery === i;
        const isK = activeKey === i;
        const cls = isQ
          ? 'rounded bg-emerald-500 px-1.5 py-0.5 text-white'
          : isK
          ? 'rounded bg-amber-400 px-1.5 py-0.5 text-neutral-900'
          : 'px-1.5 py-0.5 text-neutral-500';
        return (
          <span key={i} className="flex items-center gap-1">
            <span className={cls}>{tokenLabel(tok)}</span>
            {i < tokens.length - 1 && (
              <span className="text-neutral-300 dark:text-neutral-700">|</span>
            )}
          </span>
        );
      })}
    </div>
  );
}

function Heatmap({
  layer,
  head,
  data,
  hover,
  setHover,
}: {
  layer: number;
  head: number;
  data: AttentionData;
  hover: Hover | null;
  setHover: (h: Hover | null) => void;
}) {
  const N = data.seq_len;
  const STEP = 60 / N;
  const cells = [];
  for (let q = 0; q < N; q++) {
    for (let k = 0; k < N; k++) {
      const masked = k > q;
      const val = masked ? 0 : getAttn(data, layer, head, q, k);
      const isHover = hover?.layer === layer && hover?.head === head && hover?.query === q && hover?.key === k;
      cells.push(
        <rect
          key={`${q}-${k}`}
          x={k * STEP}
          y={q * STEP}
          width={STEP}
          height={STEP}
          fill={masked ? 'rgba(120,120,120,0.06)' : `rgba(58,95,205,${val.toFixed(3)})`}
          stroke={isHover ? 'currentColor' : 'none'}
          strokeWidth={isHover ? 1 : 0}
          vectorEffect="non-scaling-stroke"
          onMouseEnter={() => setHover({ layer, head, query: q, key: k })}
        />
      );
    }
  }
  return (
    <svg
      viewBox="0 0 60 60"
      className="block w-full text-neutral-900 dark:text-neutral-100"
      onMouseLeave={() => setHover(null)}
    >
      {cells}
    </svg>
  );
}

export function AttentionView() {
  const currentStep = useDataStore((s) => s.currentStep);
  const ensureAttention = useDataStore((s) => s.ensureAttention);
  const attentionCache = useDataStore((s) => s.attentionCache);
  const ensureCheckpoint = useDataStore((s) => s.ensureCheckpoint);
  const checkpointCache = useDataStore((s) => s.checkpointCache);
  const attentionAnnotations = useDataStore((s) => s.attentionAnnotations);
  const manifest = useDataStore((s) => s.manifest);
  useDataStore((s) => s.attentionCacheVersion);
  useDataStore((s) => s.cacheVersion);

  const canonicalTokens = manifest?.training.canonical_input_tokens ?? CANONICAL_TOKENS_FALLBACK;

  const [hover, setHover] = useState<Hover | null>(null);
  const lastSeenRef = useRef<AttentionData | null>(null);
  const verifiedRef = useRef(false);

  useEffect(() => {
    void ensureAttention(currentStep);
  }, [currentStep, ensureAttention]);

  useEffect(() => {
    if (verifiedRef.current) return;
    void ensureCheckpoint(0).then(() => {
      const cp = checkpointCache.get(0);
      if (!cp) return;
      const actual = cp.top_5_next.map((e) => e.context_token);
      if (actual.length !== canonicalTokens.length) {
        console.warn(`[AttentionView] canonical token count mismatch: expected ${canonicalTokens.length}, got ${actual.length}`);
        return;
      }
      for (let i = 0; i < canonicalTokens.length; i++) {
        if (actual[i] !== canonicalTokens[i]) {
          console.warn(`[AttentionView] canonical token mismatch at ${i}: expected "${canonicalTokens[i]}", got "${actual[i]}"`);
        }
      }
      verifiedRef.current = true;
    }).catch(() => {/* loader handles */});
  }, [ensureCheckpoint, checkpointCache, canonicalTokens]);

  const data = attentionCache.get(currentStep);
  if (data) lastSeenRef.current = data;
  const renderData = data ?? lastSeenRef.current;

  return (
    <section>
      <TokenBar tokens={canonicalTokens} activeQuery={hover?.query ?? null} activeKey={hover?.key ?? null} />
      <div className="mt-3 min-h-[24px]">
        {hover && renderData && (
          <div className="font-mono text-xs text-neutral-500 dark:text-neutral-500">
            L{hover.layer} H{hover.head} · query &quot;{tokenLabel(canonicalTokens[hover.query])}&quot; → key &quot;{tokenLabel(canonicalTokens[hover.key])}&quot; · weight {getAttn(renderData, hover.layer, hover.head, hover.query, hover.key).toFixed(3)}
          </div>
        )}
      </div>
      <div className="mt-4">
        {renderData ? (
          <div className="grid grid-cols-6 gap-2">
            {Array.from({ length: 6 * 6 }, (_, idx) => {
              const layer = Math.floor(idx / 6);
              const head = idx % 6;
              const annotation = attentionAnnotations?.find(
                (a) => a.layer === layer && a.head === head && currentStep >= a.form_step,
              );
              return (
                <div
                  key={`${layer}-${head}`}
                  className="space-y-1"
                  title={annotation?.description}
                >
                  <div className="font-mono text-[9px] uppercase tracking-widest text-neutral-400 dark:text-neutral-600">
                    L{layer} H{head}
                  </div>
                  {annotation && (
                    <div className="mt-0.5 text-[9px] leading-tight text-emerald-600 dark:text-emerald-400">
                      {annotation.title}
                    </div>
                  )}
                  <Heatmap layer={layer} head={head} data={renderData} hover={hover} setHover={setHover} />
                </div>
              );
            })}
          </div>
        ) : (
          <div className="text-sm text-neutral-400 dark:text-neutral-600">loading attention patterns...</div>
        )}
      </div>
    </section>
  );
}
