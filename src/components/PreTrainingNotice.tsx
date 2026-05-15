import { useDataStore } from '../store/data';

export function PreTrainingNotice() {
  const discoveries = useDataStore((s) => s.discoveries);
  const currentStep = useDataStore((s) => s.currentStep);
  const currentMode = useDataStore((s) => s.currentMode);

  if (currentMode !== 'interactive') return null;
  if (!discoveries || discoveries.length === 0) return null;

  const firstDiscoveryStep = discoveries.reduce(
    (m, d) => (d.step < m ? d.step : m),
    discoveries[0].step,
  );
  if (currentStep >= firstDiscoveryStep) return null;
  if (discoveries.some((d) => d.step === currentStep)) return null;

  return (
    <aside className="mb-6 rounded-lg border border-neutral-200 bg-neutral-50 px-4 py-3 dark:border-neutral-800 dark:bg-neutral-900/40">
      <div className="font-mono text-[10px] uppercase tracking-widest text-neutral-500 dark:text-neutral-400">
        before training · step {currentStep.toLocaleString('en-US')}
      </div>
      <p className="mt-1 text-sm text-neutral-600 dark:text-neutral-400">
        Random GPT-2 vocabulary, before any TinyStories training. Drag the
        timeline to step 100 to watch grammar emerge.
      </p>
    </aside>
  );
}
