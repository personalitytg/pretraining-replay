import { useState } from 'react';

import { useDataStore } from '../store/data';

const REPO_URL = 'https://github.com/personalitytg/pretraining-replay';

function GithubIcon({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 16 16"
      width="20"
      height="20"
      fill="currentColor"
      aria-hidden="true"
      className={className}
    >
      <path
        fillRule="evenodd"
        d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"
      />
    </svg>
  );
}

function SunIcon() {
  return (
    <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
    </svg>
  );
}

function LinkIcon() {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M10 13a5 5 0 0 0 7.07 0l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.72" />
      <path d="M14 11a5 5 0 0 0-7.07 0l-3 3a5 5 0 0 0 7.07 7.07l1.72-1.72" />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
    </svg>
  );
}

export function Header() {
  const manifest = useDataStore((s) => s.manifest);
  const theme = useDataStore((s) => s.theme);
  const toggleTheme = useDataStore((s) => s.toggleTheme);
  const currentMode = useDataStore((s) => s.currentMode);
  const setMode = useDataStore((s) => s.setMode);
  const isShowcase = currentMode === 'showcase';
  const isDark = theme === 'dark';

  const [copied, setCopied] = useState(false);
  const onShare = async () => {
    try {
      await navigator.clipboard.writeText(window.location.href);
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch {
      /* ignore */
    }
  };

  return (
    <header className="border-b border-neutral-200 px-6 py-4 dark:border-neutral-800">
      <div className="mx-auto flex max-w-3xl items-center justify-between gap-2 sm:gap-4">
        <a
          href={REPO_URL}
          target="_blank"
          rel="noopener noreferrer"
          aria-label="View source on GitHub"
          className="focus-ring text-neutral-500 transition-colors hover:text-neutral-900 dark:text-neutral-400 dark:hover:text-white"
        >
          <GithubIcon />
        </a>
        <div className="flex-1 text-center">
          <h1 className="text-lg font-medium text-neutral-900 dark:text-neutral-100">
            Pretraining Replay
          </h1>
          <p className="mt-0.5 text-xs text-neutral-500 dark:text-neutral-500">
            {manifest ? manifest.run_id : 'loading run...'}
          </p>
        </div>
        <button
          type="button"
          onClick={onShare}
          aria-label="Copy link to this moment"
          className="focus-ring relative text-neutral-500 transition-colors hover:text-neutral-900 dark:text-neutral-400 dark:hover:text-white"
        >
          <LinkIcon />
          {copied && (
            <span className="pointer-events-none absolute right-0 top-full mt-1 whitespace-nowrap rounded bg-neutral-900 px-2 py-1 font-mono text-[10px] text-white shadow dark:bg-neutral-100 dark:text-neutral-900">
              Link copied
            </span>
          )}
        </button>
        <button
          type="button"
          onClick={() => setMode(isShowcase ? 'interactive' : 'showcase')}
          className="focus-ring rounded border border-neutral-300 px-2.5 py-1 text-xs uppercase tracking-wide text-neutral-600 hover:border-emerald-500 hover:text-emerald-700 dark:border-neutral-700 dark:text-neutral-400 dark:hover:border-emerald-400 dark:hover:text-emerald-300"
        >
          {isShowcase ? 'Interactive' : 'Showcase'}
        </button>
        <button
          type="button"
          onClick={toggleTheme}
          aria-label={isDark ? 'Switch to light theme' : 'Switch to dark theme'}
          className="focus-ring text-neutral-500 transition-colors hover:text-neutral-900 dark:text-neutral-400 dark:hover:text-white"
        >
          {isDark ? <SunIcon /> : <MoonIcon />}
        </button>
      </div>
    </header>
  );
}
