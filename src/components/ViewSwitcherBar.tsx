import { useDataStore } from '../store/data';
import type { ViewType } from '../types';

interface Tab {
  id: ViewType;
  label: string;
}

const TABS: Tab[] = [
  { id: 'text', label: 'Text' },
  { id: 'probes', label: 'Probes' },
  { id: 'attention', label: 'Attention' },
  { id: 'embedding', label: 'Embedding' },
  { id: 'predict', label: 'Predict' },
];

export function ViewSwitcherBar() {
  const view = useDataStore((s) => s.currentView);
  const setView = useDataStore((s) => s.setView);
  return (
    <div className="flex items-center gap-1 border-b border-neutral-200 pb-2 dark:border-neutral-800">
      {TABS.map((t) => {
        const active = view === t.id;
        return (
          <button
            key={t.id}
            type="button"
            onClick={() => setView(t.id)}
            className={
              'focus-ring rounded px-3 py-1 font-mono text-[11px] uppercase tracking-widest transition-colors ' +
              (active
                ? 'bg-neutral-900 text-white dark:bg-neutral-100 dark:text-neutral-900'
                : 'text-neutral-500 hover:text-neutral-900 dark:text-neutral-500 dark:hover:text-neutral-100')
            }
          >
            <span className="sm:hidden">{t.label.charAt(0)}</span>
            <span className="hidden sm:inline">{t.label}</span>
          </button>
        );
      })}
    </div>
  );
}
