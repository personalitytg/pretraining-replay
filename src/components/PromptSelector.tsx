import { useDataStore } from '../store/data';

export function PromptSelector() {
  const manifest = useDataStore((s) => s.manifest);
  const currentPromptId = useDataStore((s) => s.currentPromptId);
  const setPromptId = useDataStore((s) => s.setPromptId);
  const currentView = useDataStore((s) => s.currentView);
  const isTextActive = currentView === 'text';

  if (!manifest) return null;

  const showcase = manifest.prompts.filter((p) => p.role === 'showcase' || !p.role);
  const probes = manifest.prompts.filter((p) => p.role === 'probe');

  return (
    <div className="flex flex-col gap-1">
      <label className="flex items-center gap-2 text-[10px] uppercase tracking-widest text-neutral-500 dark:text-neutral-500">
        <span className="shrink-0">Prompt</span>
        <select
          value={currentPromptId}
          onChange={(e) => setPromptId(e.target.value)}
          className="focus-ring max-w-full flex-1 rounded border border-neutral-300 bg-white px-2 py-1 font-sans text-xs normal-case tracking-normal text-neutral-900 dark:border-neutral-700 dark:bg-neutral-900 dark:text-neutral-100"
        >
          {showcase.length > 0 && (
            <optgroup label="Showcase">
              {showcase.map((p) => (
                <option key={p.id} value={p.id}>{p.text || p.id}</option>
              ))}
            </optgroup>
          )}
          {probes.length > 0 && (
            <optgroup label="Probes">
              {probes.map((p) => (
                <option key={p.id} value={p.id}>{p.text || p.id}</option>
              ))}
            </optgroup>
          )}
        </select>
      </label>
      {!isTextActive && (
        <p className="pl-[3.5rem] text-[10px] text-neutral-400 dark:text-neutral-600 sm:pl-[4rem]">
          applies to Text view only
        </p>
      )}
    </div>
  );
}
