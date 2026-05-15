import { create } from 'zustand';

import {
  fetchAttention,
  fetchAttentionAnnotations,
  fetchCheckpoint,
  fetchDiscoveries,
  fetchEmbedding,
  fetchManifest,
  fetchTokensOfInterest,
} from '../data/loader';
import type {
  AttentionAnnotation,
  AttentionData,
  CheckpointData,
  Discovery,
  EmbeddingData,
  Manifest,
  Theme,
  TokensOfInterest,
  ViewMode,
  ViewType,
} from '../types';

const CACHE_LIMIT = 50;
const THEME_STORAGE_KEY = 'theme';

function readInitialTheme(): Theme {
  if (typeof document !== 'undefined' && document.documentElement.classList.contains('dark')) {
    return 'dark';
  }
  return 'light';
}

function applyThemeToDom(theme: Theme): void {
  if (typeof document === 'undefined') return;
  const cl = document.documentElement.classList;
  if (theme === 'dark') cl.add('dark');
  else cl.remove('dark');
  try {
    localStorage.setItem(THEME_STORAGE_KEY, theme);
  } catch {
    /* ignore */
  }
}

function buildHash(
  step: number,
  promptId: string,
  diff: boolean,
  view: ViewType,
  mode: ViewMode,
): string {
  const parts: string[] = [];
  if (mode !== 'interactive') parts.push(`mode=${mode}`);
  parts.push(`step=${step}`);
  if (promptId) parts.push(`prompt=${encodeURIComponent(promptId)}`);
  if (diff) parts.push('diff=on');
  if (view !== 'text') parts.push(`view=${view}`);
  return '#' + parts.join('&');
}

function writeURL(
  step: number,
  promptId: string,
  diff: boolean,
  view: ViewType,
  mode: ViewMode,
): void {
  if (typeof window === 'undefined') return;
  const next = buildHash(step, promptId, diff, view, mode);
  if (window.location.hash !== next) {
    window.history.replaceState(null, '', next);
  }
}

interface DataState {
  manifest: Manifest | null;
  manifestError: string | null;
  discoveries: Discovery[] | null;
  attentionAnnotations: AttentionAnnotation[] | null;
  currentStep: number;
  currentPromptId: string;
  diffMode: boolean;
  isPlaying: boolean;
  theme: Theme;
  checkpointCache: Map<number, CheckpointData>;
  inflight: Map<number, Promise<CheckpointData>>;
  cacheVersion: number;
  attentionCache: Map<number, AttentionData>;
  attentionInflight: Map<number, Promise<AttentionData>>;
  attentionCacheVersion: number;
  embeddingCache: Map<number, EmbeddingData>;
  embeddingInflight: Map<number, Promise<EmbeddingData>>;
  embeddingCacheVersion: number;
  tokensOfInterest: TokensOfInterest | null;
  currentView: ViewType;
  currentMode: ViewMode;

  loadManifest: () => Promise<void>;
  loadDiscoveries: () => Promise<void>;
  loadAttentionAnnotations: () => Promise<void>;
  setStep: (step: number) => void;
  setPromptId: (id: string) => void;
  toggleDiff: () => void;
  togglePlay: () => void;
  toggleTheme: () => void;
  hydrateFromURL: () => void;
  ensureCheckpoint: (step: number) => Promise<CheckpointData>;
  ensureAttention: (step: number) => Promise<AttentionData>;
  ensureEmbedding: (step: number) => Promise<EmbeddingData>;
  loadTokensOfInterest: () => Promise<void>;
  setView: (view: ViewType) => void;
  setMode: (mode: ViewMode) => void;
}

export const useDataStore = create<DataState>((set, get) => ({
  manifest: null,
  manifestError: null,
  discoveries: null,
  attentionAnnotations: null,
  currentStep: 0,
  currentPromptId: '',
  diffMode: false,
  isPlaying: false,
  theme: readInitialTheme(),
  checkpointCache: new Map(),
  inflight: new Map(),
  cacheVersion: 0,
  attentionCache: new Map(),
  attentionInflight: new Map(),
  attentionCacheVersion: 0,
  embeddingCache: new Map(),
  embeddingInflight: new Map(),
  embeddingCacheVersion: 0,
  tokensOfInterest: null,
  currentView: 'text',
  currentMode: 'interactive',

  loadManifest: async () => {
    try {
      const manifest = await fetchManifest();
      const firstStep = manifest.checkpoints[0]?.step ?? 0;
      const showcase = manifest.prompts.find((p) => p.role === 'showcase' || !p.role);
      const firstPromptId = showcase?.id ?? manifest.prompts[0]?.id ?? '';
      set({
        manifest,
        manifestError: null,
        currentStep: firstStep,
        currentPromptId: firstPromptId,
      });
      void get().ensureCheckpoint(firstStep);
    } catch (e) {
      set({ manifestError: (e as Error).message });
    }
  },

  loadDiscoveries: async () => {
    try {
      const discoveries = await fetchDiscoveries();
      set({ discoveries });
    } catch {
      set({ discoveries: [] });
    }
  },

  loadAttentionAnnotations: async () => {
    try {
      const attentionAnnotations = await fetchAttentionAnnotations();
      set({ attentionAnnotations });
    } catch {
      set({ attentionAnnotations: [] });
    }
  },

  setStep: (step) => {
    if (step === get().currentStep) return;
    set({ currentStep: step });
    const s = get();
    writeURL(s.currentStep, s.currentPromptId, s.diffMode, s.currentView, s.currentMode);
    void get().ensureCheckpoint(step);
  },

  setPromptId: (id) => {
    if (id === get().currentPromptId) return;
    set({ currentPromptId: id });
    const s = get();
    writeURL(s.currentStep, s.currentPromptId, s.diffMode, s.currentView, s.currentMode);
  },

  toggleDiff: () => {
    set((s) => ({ diffMode: !s.diffMode }));
    const s = get();
    writeURL(s.currentStep, s.currentPromptId, s.diffMode, s.currentView, s.currentMode);
  },

  togglePlay: () => set((s) => ({ isPlaying: !s.isPlaying })),

  toggleTheme: () => {
    const next: Theme = get().theme === 'dark' ? 'light' : 'dark';
    applyThemeToDom(next);
    set({ theme: next });
  },

  hydrateFromURL: () => {
    if (typeof window === 'undefined') return;
    const m = get().manifest;
    if (!m) return;
    const hash = window.location.hash.replace(/^#/, '');
    if (!hash) return;
    const params = new URLSearchParams(hash);

    let step = get().currentStep;
    const stepStr = params.get('step');
    if (stepStr !== null) {
      const target = Number(stepStr);
      if (Number.isFinite(target) && m.checkpoints.length > 0) {
        step = m.checkpoints.reduce((best, c) =>
          Math.abs(c.step - target) < Math.abs(best.step - target) ? c : best,
        ).step;
      }
    }

    let promptId = get().currentPromptId;
    const promptStr = params.get('prompt');
    if (promptStr !== null && m.prompts.some((p) => p.id === promptStr)) {
      promptId = promptStr;
    }

    const diff = params.get('diff') === 'on';

    const viewStr = params.get('view');
    const allowedViews: ViewType[] = ['text', 'probes', 'attention', 'embedding', 'predict'];
    const view: ViewType = allowedViews.includes(viewStr as ViewType)
      ? (viewStr as ViewType)
      : 'text';

    const modeStr = params.get('mode');
    const mode: ViewMode = modeStr === 'showcase' ? 'showcase' : 'interactive';

    set({
      currentStep: step,
      currentPromptId: promptId,
      diffMode: diff,
      currentView: view,
      currentMode: mode,
    });
    writeURL(step, promptId, diff, view, mode);
    void get().ensureCheckpoint(step);
  },

  ensureCheckpoint: async (step) => {
    const { checkpointCache, inflight } = get();
    const cached = checkpointCache.get(step);
    if (cached) return cached;
    const pending = inflight.get(step);
    if (pending) return pending;

    const promise = fetchCheckpoint(step)
      .then((data) => {
        const { checkpointCache: cache, inflight: inflightNow, cacheVersion } = get();
        cache.set(step, data);
        if (cache.size > CACHE_LIMIT) {
          const oldestKey = cache.keys().next().value;
          if (oldestKey !== undefined) cache.delete(oldestKey);
        }
        inflightNow.delete(step);
        set({ cacheVersion: cacheVersion + 1 });
        return data;
      })
      .catch((e) => {
        get().inflight.delete(step);
        throw e;
      });

    inflight.set(step, promise);
    return promise;
  },

  ensureAttention: async (step) => {
    const { attentionCache, attentionInflight } = get();
    const cached = attentionCache.get(step);
    if (cached) return cached;
    const pending = attentionInflight.get(step);
    if (pending) return pending;
    const promise = fetchAttention(step)
      .then((data) => {
        const s = get();
        s.attentionCache.set(step, data);
        if (s.attentionCache.size > CACHE_LIMIT) {
          const oldest = s.attentionCache.keys().next().value;
          if (oldest !== undefined) s.attentionCache.delete(oldest);
        }
        s.attentionInflight.delete(step);
        set({ attentionCacheVersion: s.attentionCacheVersion + 1 });
        return data;
      })
      .catch((e) => {
        get().attentionInflight.delete(step);
        throw e;
      });
    attentionInflight.set(step, promise);
    return promise;
  },

  ensureEmbedding: async (step) => {
    const { embeddingCache, embeddingInflight } = get();
    const cached = embeddingCache.get(step);
    if (cached) return cached;
    const pending = embeddingInflight.get(step);
    if (pending) return pending;
    const promise = fetchEmbedding(step)
      .then((data) => {
        const s = get();
        s.embeddingCache.set(step, data);
        if (s.embeddingCache.size > CACHE_LIMIT) {
          const oldest = s.embeddingCache.keys().next().value;
          if (oldest !== undefined) s.embeddingCache.delete(oldest);
        }
        s.embeddingInflight.delete(step);
        set({ embeddingCacheVersion: s.embeddingCacheVersion + 1 });
        return data;
      })
      .catch((e) => {
        get().embeddingInflight.delete(step);
        throw e;
      });
    embeddingInflight.set(step, promise);
    return promise;
  },

  loadTokensOfInterest: async () => {
    try {
      const toi = await fetchTokensOfInterest();
      set({ tokensOfInterest: toi });
    } catch {
      set({ tokensOfInterest: { tokens: [] } });
    }
  },

  setView: (view) => {
    if (view === get().currentView) return;
    set({ currentView: view });
    const s = get();
    writeURL(s.currentStep, s.currentPromptId, s.diffMode, s.currentView, s.currentMode);
  },

  setMode: (mode) => {
    if (mode === get().currentMode) return;
    set({ currentMode: mode });
    const s = get();
    writeURL(s.currentStep, s.currentPromptId, s.diffMode, s.currentView, s.currentMode);
  },
}));
