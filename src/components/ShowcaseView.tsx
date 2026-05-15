import { useDataStore } from '../store/data';
import type { Discovery } from '../types';

function truncate(text: string, maxChars = 120): string {
  if (text.length <= maxChars) return text;
  const cut = text.slice(0, maxChars);
  const lastSpace = cut.lastIndexOf(' ');
  return (lastSpace > 60 ? cut.slice(0, lastSpace) : cut) + '…';
}

function ShowcaseTile({ d }: { d: Discovery }) {
  const setMode = useDataStore((s) => s.setMode);
  const setStep = useDataStore((s) => s.setStep);
  const setPromptId = useDataStore((s) => s.setPromptId);
  const setView = useDataStore((s) => s.setView);

  const open = () => {
    setMode('interactive');
    setView('text');
    setPromptId(d.headline_example.prompt_id);
    setStep(d.step);
  };

  return (
    <article className="flex flex-col gap-3 rounded-lg border border-neutral-200 bg-white p-5 transition-colors hover:border-emerald-500 dark:border-neutral-800 dark:bg-neutral-900 dark:hover:border-emerald-400">
      <div className="flex items-baseline justify-between">
        <span className="font-mono text-xs uppercase tracking-widest text-emerald-700 dark:text-emerald-400">
          step {d.step.toLocaleString('en-US')}
        </span>
      </div>
      <h3 className="text-lg font-medium text-neutral-900 dark:text-neutral-100">
        {d.title}
      </h3>
      <div className="space-y-3">
        <div>
          <div className="mb-1 font-mono text-[10px] uppercase tracking-widest text-neutral-400 dark:text-neutral-600">
            Before · step {d.before_example.step?.toLocaleString('en-US') ?? '—'}
          </div>
          <p className="font-mono text-xs leading-relaxed text-neutral-500 dark:text-neutral-500">
            {truncate(d.before_example.generation, 140)}
          </p>
        </div>
        <div>
          <div className="mb-1 font-mono text-[10px] uppercase tracking-widest text-emerald-700 dark:text-emerald-400">
            After
          </div>
          <p className="font-mono text-xs leading-relaxed text-neutral-700 dark:text-neutral-300">
            {truncate(d.headline_example.generation, 140)}
          </p>
        </div>
      </div>
      <button
        type="button"
        onClick={open}
        className="focus-ring mt-2 self-start text-xs font-medium text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300"
      >
        View this moment →
      </button>
    </article>
  );
}

export function ShowcaseView() {
  const discoveries = useDataStore((s) => s.discoveries) ?? [];
  return (
    <main className="flex-1 min-h-0 overflow-y-auto">
      <div className="mx-auto max-w-5xl px-6 py-10">
        <h1 className="text-2xl font-medium text-neutral-900 dark:text-neutral-100">
          Eight moments of discovery
        </h1>
        <p className="mt-2 max-w-2xl text-sm text-neutral-500 dark:text-neutral-400">
          Curated training-step transitions where the model first acquires a capability. Click any moment to open the interactive timeline at that point.
        </p>
        <div className="mt-8 grid grid-cols-1 gap-4 md:grid-cols-2">
          {discoveries.map((d) => (
            <ShowcaseTile key={d.id} d={d} />
          ))}
        </div>
      </div>
    </main>
  );
}
