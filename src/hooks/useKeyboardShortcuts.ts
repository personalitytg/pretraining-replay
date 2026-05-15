import { useEffect } from 'react';

import { useDataStore } from '../store/data';

function isFormTarget(t: EventTarget | null): boolean {
  if (!(t instanceof HTMLElement)) return false;
  const tag = t.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return true;
  return t.isContentEditable;
}

export function useKeyboardShortcuts(): void {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (isFormTarget(e.target)) return;
      const state = useDataStore.getState();
      const m = state.manifest;
      if (!m) return;
      const cps = m.checkpoints;
      if (cps.length === 0) return;
      const idxRaw = cps.findIndex((c) => c.step === state.currentStep);
      const idx = idxRaw < 0 ? 0 : idxRaw;

      switch (e.key) {
        case ' ':
        case 'Spacebar': {
          e.preventDefault();
          state.togglePlay();
          return;
        }
        case 'ArrowLeft': {
          e.preventDefault();
          if (e.shiftKey) {
            const ds = state.discoveries ?? [];
            for (let i = ds.length - 1; i >= 0; i--) {
              if (ds[i].step < state.currentStep) {
                state.setStep(ds[i].step);
                return;
              }
            }
            return;
          }
          if (idx > 0) state.setStep(cps[idx - 1].step);
          return;
        }
        case 'ArrowRight': {
          e.preventDefault();
          if (e.shiftKey) {
            const ds = state.discoveries ?? [];
            const next = ds.find((d) => d.step > state.currentStep);
            if (next) state.setStep(next.step);
            return;
          }
          if (idx < cps.length - 1) state.setStep(cps[idx + 1].step);
          return;
        }
        case 'Home': {
          e.preventDefault();
          state.setStep(cps[0].step);
          return;
        }
        case 'End': {
          e.preventDefault();
          state.setStep(cps[cps.length - 1].step);
          return;
        }
        case '1': {
          e.preventDefault();
          state.setView('text');
          return;
        }
        case '2': {
          e.preventDefault();
          state.setView('probes');
          return;
        }
        case '3': {
          e.preventDefault();
          state.setView('attention');
          return;
        }
        case '4': {
          e.preventDefault();
          state.setView('embedding');
          return;
        }
        case '5': {
          e.preventDefault();
          state.setView('predict');
          return;
        }
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
}
