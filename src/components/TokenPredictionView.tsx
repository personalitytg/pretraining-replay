import { useEffect, useRef, useState } from 'react';
import { useDataStore } from '../store/data';
import type { TopFiveEntry } from '../types';

const POPUP_W = 288;
const POPUP_H_ESTIMATE = 200;
const POPUP_MARGIN = 8;

function displayToken(t: string): string {
  if (t.startsWith(' ')) return '·' + t.slice(1);
  return t;
}

interface PopupState {
  entry: TopFiveEntry;
  left: number;
  top: number;
}

function TokenChip({
  entry,
  onEnter,
  onLeave,
  isActive,
}: {
  entry: TopFiveEntry;
  onEnter: (entry: TopFiveEntry, rect: DOMRect) => void;
  onLeave: () => void;
  isActive: boolean;
}) {
  const ref = useRef<HTMLButtonElement>(null);
  const handleEnter = () => {
    if (!ref.current) return;
    onEnter(entry, ref.current.getBoundingClientRect());
  };
  return (
    <button
      ref={ref}
      type="button"
      onMouseEnter={handleEnter}
      onMouseLeave={onLeave}
      onFocus={handleEnter}
      onBlur={onLeave}
      className={
        'focus-ring w-full flex-1 rounded border px-3 py-2 text-center font-mono text-sm transition-colors sm:flex-initial ' +
        (isActive
          ? 'border-blue-500 bg-white text-blue-700 dark:border-blue-400 dark:bg-neutral-900 dark:text-blue-300'
          : 'border-neutral-300 bg-white text-neutral-900 hover:border-blue-500 hover:text-blue-700 dark:border-neutral-700 dark:bg-neutral-900 dark:text-neutral-100 dark:hover:border-blue-400 dark:hover:text-blue-300')
      }
    >
      {displayToken(entry.context_token)}
    </button>
  );
}

export function TokenPredictionView() {
  const currentStep = useDataStore((s) => s.currentStep);
  const cache = useDataStore((s) => s.checkpointCache);
  const ensureCheckpoint = useDataStore((s) => s.ensureCheckpoint);
  useDataStore((s) => s.cacheVersion);

  const [popup, setPopup] = useState<PopupState | null>(null);

  useEffect(() => {
    void ensureCheckpoint(currentStep);
  }, [currentStep, ensureCheckpoint]);

  useEffect(() => {
    if (!popup) return;
    const hide = () => setPopup(null);
    window.addEventListener('scroll', hide, true);
    window.addEventListener('resize', hide);
    return () => {
      window.removeEventListener('scroll', hide, true);
      window.removeEventListener('resize', hide);
    };
  }, [popup]);

  const handleEnter = (entry: TopFiveEntry, rect: DOMRect) => {
    const roomBelow = window.innerHeight - rect.bottom;
    const roomAbove = rect.top;
    const placeBelow = roomBelow >= POPUP_H_ESTIMATE + POPUP_MARGIN || roomBelow >= roomAbove;
    const top = placeBelow
      ? rect.bottom + POPUP_MARGIN
      : rect.top - POPUP_H_ESTIMATE - POPUP_MARGIN;
    let left = rect.left + rect.width / 2 - POPUP_W / 2;
    if (left + POPUP_W > window.innerWidth - POPUP_MARGIN) {
      left = window.innerWidth - POPUP_W - POPUP_MARGIN;
    }
    if (left < POPUP_MARGIN) left = POPUP_MARGIN;
    setPopup({ entry, left, top });
  };

  const data = cache.get(currentStep);
  const entries: TopFiveEntry[] = data?.top_5_next ?? [];

  return (
    <section>
      <h2 className="text-sm font-medium text-neutral-900 dark:text-neutral-100">
        Model&apos;s next-token predictions
      </h2>
      <p className="mt-1 text-xs text-neutral-500 dark:text-neutral-400">
        Canonical input: <code className="font-mono text-neutral-700 dark:text-neutral-300">Once upon a time, there was a</code>. Hover any token to see the model&apos;s top-5 most likely next tokens at the current step.
      </p>
      <div className="mt-6 flex flex-col gap-2 sm:flex-row sm:items-stretch">
        {entries.length === 0 ? (
          <div className="text-sm text-neutral-400 dark:text-neutral-600">loading predictions...</div>
        ) : (
          entries.map((entry) => (
            <TokenChip
              key={entry.position}
              entry={entry}
              onEnter={handleEnter}
              onLeave={() => setPopup(null)}
              isActive={popup?.entry.position === entry.position}
            />
          ))
        )}
      </div>
      <p className="mt-6 text-xs text-neutral-400 dark:text-neutral-600">
        Tokens with a leading space are shown with a · prefix (GPT-2 BPE distinguishes &quot;cat&quot; from &quot; cat&quot;).
      </p>

      {popup && (
        <div
          className="pointer-events-none fixed z-50 w-72 rounded-lg bg-neutral-900 p-3 text-white shadow-xl dark:bg-neutral-100 dark:text-neutral-900"
          style={{ left: popup.left, top: popup.top }}
        >
          <div className="mb-2 font-mono text-[10px] uppercase tracking-widest text-neutral-400 dark:text-neutral-500">
            Top-5 next predictions
          </div>
          <div className="space-y-1.5">
            {popup.entry.top_5.map((p, i) => (
              <div key={i} className="flex items-center gap-2">
                <span className="w-20 truncate font-mono text-xs">{displayToken(p.token)}</span>
                <div className="h-2 flex-1 overflow-hidden rounded bg-neutral-700 dark:bg-neutral-300">
                  <div
                    className="h-full bg-blue-400 dark:bg-blue-600"
                    style={{ width: `${Math.max(2, p.prob * 100)}%` }}
                  />
                </div>
                <span className="w-12 text-right font-mono text-[10px] tabular-nums">
                  {(p.prob * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}
