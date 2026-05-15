import { useEffect } from 'react';

import { AttentionView } from './components/AttentionView';
import { DiscoveryAnnotation } from './components/DiscoveryAnnotation';
import { PreTrainingNotice } from './components/PreTrainingNotice';
import { EmbeddingView } from './components/EmbeddingView';
import { Header } from './components/Header';
import { ProbeView } from './components/ProbeView';
import { PromptSelector } from './components/PromptSelector';
import { Scrubber } from './components/Scrubber';
import { ShowcaseView } from './components/ShowcaseView';
import { StepContext } from './components/StepContext';
import { TextView } from './components/TextView';
import { TokenPredictionView } from './components/TokenPredictionView';
import { ViewSwitcherBar } from './components/ViewSwitcherBar';
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts';
import { useDataStore } from './store/data';

export default function App() {
  const loadManifest = useDataStore((s) => s.loadManifest);
  const loadDiscoveries = useDataStore((s) => s.loadDiscoveries);
  const loadTokensOfInterest = useDataStore((s) => s.loadTokensOfInterest);
  const loadAttentionAnnotations = useDataStore((s) => s.loadAttentionAnnotations);
  const manifestError = useDataStore((s) => s.manifestError);
  const currentView = useDataStore((s) => s.currentView);
  const currentMode = useDataStore((s) => s.currentMode);

  useKeyboardShortcuts();

  useEffect(() => {
    void loadManifest().then(() => {
      useDataStore.getState().hydrateFromURL();
    });
    void loadDiscoveries();
    void loadTokensOfInterest();
    void loadAttentionAnnotations();
  }, [loadManifest, loadDiscoveries, loadTokensOfInterest, loadAttentionAnnotations]);

  useEffect(() => {
    const onHash = () => useDataStore.getState().hydrateFromURL();
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

  return (
    <div className="flex h-screen flex-col bg-white font-sans text-neutral-900 dark:bg-neutral-950 dark:text-neutral-100">
      <Header />
      {currentMode === 'showcase' ? (
        <ShowcaseView />
      ) : (
        <>
          <main className="flex-1 min-h-0 overflow-y-auto">
            <div className="mx-auto max-w-3xl px-6 py-8">
              {manifestError ? (
                <div className="text-sm text-red-700 dark:text-red-400">
                  failed to load manifest: {manifestError}
                </div>
              ) : (
                <>
                  <PreTrainingNotice />
                  <DiscoveryAnnotation />
                  {currentView === 'text' && <TextView />}
                  {currentView === 'probes' && <ProbeView />}
                  {currentView === 'attention' && <AttentionView />}
                  {currentView === 'embedding' && <EmbeddingView />}
                  {currentView === 'predict' && <TokenPredictionView />}
                </>
              )}
            </div>
          </main>
          <div className="shrink-0 border-t border-neutral-200 bg-white dark:border-neutral-800 dark:bg-neutral-950">
            <div className="mx-auto max-w-3xl space-y-3 px-6 py-3">
              <PromptSelector />
              <ViewSwitcherBar />
              <StepContext />
              <Scrubber />
            </div>
          </div>
        </>
      )}
    </div>
  );
}
