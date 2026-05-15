import { useDataStore } from '../store/data';

function formatTokens(n: number): string {
  if (n >= 1_000_000_000) return `${(n / 1_000_000_000).toFixed(1)}B`;
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1000) return `${(n / 1000).toFixed(0)}K`;
  return `${n}`;
}

function formatGpuTime(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  if (s >= 3600) {
    const h = Math.floor(s / 3600);
    const m = Math.round((s % 3600) / 60);
    return `${h}h ${m}m`;
  }
  if (s >= 60) {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${m}m ${sec}s`;
  }
  return `${s}s`;
}

export function StepContext() {
  const manifest = useDataStore((s) => s.manifest);
  const discoveries = useDataStore((s) => s.discoveries);
  const currentStep = useDataStore((s) => s.currentStep);

  if (!manifest) return null;

  const ckpt = manifest.checkpoints.find((c) => c.step === currentStep) ?? manifest.checkpoints[0];
  const total = manifest.training.total_steps;
  const pct = total > 0 ? ((currentStep / total) * 100).toFixed(0) : '0';
  const tokens = formatTokens(ckpt.tokens_seen);
  const gpu = formatGpuTime(ckpt.wallclock_seconds);

  let countdown: { kind: 'next' | 'final'; text: string } | null = null;
  if (discoveries && discoveries.length > 0) {
    const here = discoveries.find((d) => d.step === currentStep);
    const next = discoveries.find((d) => d.step > currentStep);
    if (here) {
      countdown = { kind: 'next', text: `→ Current moment: ${here.title}` };
    } else if (next) {
      const delta = (next.step - currentStep).toLocaleString('en-US');
      countdown = { kind: 'next', text: `→ Next moment: ${next.title} in ${delta} steps` };
    } else {
      countdown = { kind: 'final', text: '→ Final moment passed' };
    }
  }

  return (
    <section className="rounded-lg border border-neutral-200 bg-neutral-50 px-5 py-4 dark:border-neutral-800 dark:bg-neutral-900/40">
      <div className="font-mono text-2xl font-medium tabular-nums text-neutral-900 dark:text-neutral-100">
        STEP {currentStep.toLocaleString('en-US')}
      </div>
      <div className="mt-1 text-sm text-neutral-500 dark:text-neutral-500">
        {pct}% trained · {tokens} tokens seen · ≈{gpu} GPU time
      </div>
      {countdown && (
        <div
          className={
            'mt-3 text-sm ' +
            (countdown.kind === 'next'
              ? 'text-blue-600 dark:text-blue-400'
              : 'text-neutral-500 dark:text-neutral-500')
          }
        >
          {countdown.text}
        </div>
      )}
    </section>
  );
}
