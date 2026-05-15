import { useDataStore } from '../store/data';

export function DiscoveryAnnotation() {
  const discoveries = useDataStore((s) => s.discoveries);
  const currentStep = useDataStore((s) => s.currentStep);
  if (!discoveries) return null;
  const active = discoveries.find((d) => d.step === currentStep);
  if (!active) return null;

  return (
    <aside className="mb-6 rounded-lg border-l-2 border-emerald-500 bg-emerald-50 px-4 py-3 dark:border-emerald-400 dark:bg-emerald-400/5">
      <div className="font-mono text-[10px] uppercase tracking-widest text-emerald-700 dark:text-emerald-400">
        moment of discovery · step {active.step.toLocaleString('en-US')}
      </div>
      <div className="mt-1 font-medium text-neutral-900 dark:text-neutral-100">
        {active.title}
      </div>
      <p className="mt-1 text-sm text-neutral-600 dark:text-neutral-400">
        {active.explanation}
      </p>
    </aside>
  );
}
