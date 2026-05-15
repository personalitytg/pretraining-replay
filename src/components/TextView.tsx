import { useEffect, useMemo } from 'react';

import { useDataStore } from '../store/data';
import type { GenerationSample } from '../types';

const WORD_SPLIT = /(\s+)/;

function diffWords(prev: string, cur: string): Array<{ text: string; isNew: boolean }> {
  const prevTokens = prev.split(WORD_SPLIT).filter((t) => t.length > 0);
  const counts = new Map<string, number>();
  for (const t of prevTokens) {
    if (/^\s+$/.test(t)) continue;
    counts.set(t, (counts.get(t) ?? 0) + 1);
  }
  const curTokens = cur.split(WORD_SPLIT).filter((t) => t.length > 0);
  return curTokens.map((t) => {
    if (/^\s+$/.test(t)) return { text: t, isNew: false };
    const left = counts.get(t) ?? 0;
    if (left > 0) {
      counts.set(t, left - 1);
      return { text: t, isNew: false };
    }
    return { text: t, isNew: true };
  });
}

function renderContinuation(
  continuation: string,
  prevContinuation: string | null,
  highlight: boolean,
) {
  if (!highlight || prevContinuation === null) {
    return <span className="text-neutral-900 dark:text-neutral-100">{continuation}</span>;
  }
  const parts = diffWords(prevContinuation, continuation);
  return (
    <span className="text-neutral-900 dark:text-neutral-100">
      {parts.map((p, i) =>
        p.isNew ? (
          <mark
            key={i}
            className="rounded-sm bg-amber-100 px-0.5 text-amber-900 dark:bg-amber-400/20 dark:text-amber-200"
          >
            {p.text}
          </mark>
        ) : (
          <span key={i}>{p.text}</span>
        ),
      )}
    </span>
  );
}

function normalizeWhitespace(s: string): string {
  return s
    .replace(/�+/g, '')
    .replace(/\s+/g, ' ')
    .replace(/^\s+/, '');
}

function splitGeneration(sample: GenerationSample, promptText: string) {
  if (sample.text.startsWith(promptText)) {
    const raw = sample.text.slice(promptText.length);
    return {
      prefix: promptText,
      continuation: normalizeWhitespace(raw),
    };
  }
  return { prefix: '', continuation: normalizeWhitespace(sample.text) };
}

export function TextView() {
  const manifest = useDataStore((s) => s.manifest);
  const currentStep = useDataStore((s) => s.currentStep);
  const currentPromptId = useDataStore((s) => s.currentPromptId);
  const diffMode = useDataStore((s) => s.diffMode);
  const toggleDiff = useDataStore((s) => s.toggleDiff);
  const cache = useDataStore((s) => s.checkpointCache);
  const ensureCheckpoint = useDataStore((s) => s.ensureCheckpoint);
  useDataStore((s) => s.cacheVersion);

  const prevStep = useMemo(() => {
    if (!manifest) return null;
    const idx = manifest.checkpoints.findIndex((c) => c.step === currentStep);
    if (idx <= 0) return null;
    return manifest.checkpoints[idx - 1].step;
  }, [manifest, currentStep]);

  useEffect(() => {
    if (diffMode && prevStep !== null) {
      void ensureCheckpoint(prevStep);
    }
  }, [diffMode, prevStep, ensureCheckpoint]);

  if (!manifest) {
    return <div className="text-sm text-neutral-500 dark:text-neutral-500">loading manifest...</div>;
  }

  const prompt = manifest.prompts.find((p) => p.id === currentPromptId);
  const promptText = prompt?.text ?? '';

  const checkpoint = cache.get(currentStep);
  const prevCheckpoint = prevStep !== null ? cache.get(prevStep) : null;
  const samples = checkpoint?.generations[currentPromptId];
  const prevSamples = prevCheckpoint?.generations[currentPromptId];
  const diffActive = diffMode && prevStep !== null && !!prevSamples;

  return (
    <section>
      <div className="flex items-end gap-4">
        <button
          type="button"
          onClick={toggleDiff}
          className={
            'focus-ring h-[30px] rounded border px-3 text-xs uppercase tracking-wide transition-colors ' +
            (diffMode
              ? 'border-amber-400 bg-amber-50 text-amber-900 dark:border-amber-400/60 dark:bg-amber-400/10 dark:text-amber-200'
              : 'border-neutral-300 bg-white text-neutral-600 hover:text-neutral-900 dark:border-neutral-700 dark:bg-neutral-900 dark:text-neutral-400 dark:hover:text-neutral-100')
          }
        >
          diff: {diffMode ? 'on' : 'off'}
        </button>
        {!diffMode && (
          <span className="text-xs text-neutral-500 dark:text-neutral-500">
            highlight words new vs previous step
          </span>
        )}
      </div>

      <div className="mt-6 space-y-6">
        {!checkpoint && cache.size === 0 && (
          <div className="text-sm text-neutral-400 dark:text-neutral-600">
            loading...
          </div>
        )}
        {checkpoint && !samples && (
          <div className="text-sm text-neutral-400 dark:text-neutral-600">
            no generations for prompt &quot;{currentPromptId}&quot;
          </div>
        )}
        {checkpoint &&
          samples?.map((s) => {
            const { prefix, continuation } = splitGeneration(s, promptText);
            const prevSample = prevSamples?.find((p) => p.seed === s.seed);
            const prevContinuation = prevSample
              ? splitGeneration(prevSample, promptText).continuation
              : null;
            return (
              <div
                key={s.seed}
                className="border-l-2 border-neutral-200 pl-3 dark:border-neutral-800"
              >
                <div className="font-sans text-xs uppercase tracking-wider text-neutral-400 dark:text-neutral-600">
                  seed {s.seed}
                </div>
                <pre
                  key={`${currentStep}-${s.seed}-${diffActive ? 'd' : 'n'}`}
                  className="mt-1 animate-textview-fade whitespace-pre-wrap font-mono text-sm leading-relaxed"
                >
                  {prefix && (
                    <span className="text-neutral-400 dark:text-neutral-600">{prefix}</span>
                  )}
                  {renderContinuation(continuation, prevContinuation, diffActive)}
                </pre>
              </div>
            );
          })}
      </div>
    </section>
  );
}
