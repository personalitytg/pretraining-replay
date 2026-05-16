# Pretraining Replay

A browser-based scrubber through the pretraining of a 30M GPT-2-style language model on TinyStories. Drag the timeline and watch grammar emerge from random tokens.

**Live site:** https://personalitytg.github.io/pretraining-replay/

## What you're looking at

- 6-layer, 6-head, d_model=384 GPT-2-style transformer (~30M params)
- Trained on TinyStories (Eldan & Li 2023) for 50,000 steps on a Kaggle T4
- 227 cadence checkpoints, log-spaced (every 5 steps from 0 to 100, every 25 to 1k, every 100 to 10k, every 500 to 50k)
- GPT-2 BPE tokenizer
- The browser never runs the model — all generations, attention patterns, embedding projections, and probe results are pre-computed offline and served as static JSON

## Curated moments

A few of the discovery markers on the timeline:

- **Step 100** — recognizable English words emerge from random BPE noise
- **Step 325** — quotation marks first reliably close in dialogue
- **Step 400** — "The moral of the story is..." appears, 200 steps before the model can write the body it summarizes
- **Step 7000** — plural agreement after "There were two ___"

More markers in the app.

## Reproducibility

Three independent pieces of evidence:

1. **Public Weights & Biases run** with timestamped per-cadence loss/lr/tokens:
   [https://wandb.ai/personalitytg-personality/pretraining-replay](https://wandb.ai/personalitytg-personality/pretraining-replay/reports/Pretraining-replay--VmlldzoxNjkwMTY1NA?accessToken=lb7ljpbx5w0597olek0c9gwc5wyk689w5a82hfem2p57vm7e3rtlpdnhvuz7k44n)
   The timestamps cannot be backdated.
2. **SHA-256 hash of every cadence checkpoint** is committed in
   `public/data/manifest.json` under each `checkpoints[i].sha256`. Anyone who
   reproduces the training run can compare their hashes against ours.
3. **Full training pipeline** is in this repo. To regenerate everything from
   scratch (TinyStories download, tokenize, train, derive artifacts), follow
   `docs/REPRODUCE.md`.

Model weights are not redistributed. The training pipeline is fully scripted —
reproduce locally or on a free Kaggle T4 (~7 hours).

## Tech

React 18 + TypeScript (strict) + Vite + Tailwind + Zustand + D3 (timeline) +
HTML canvas (embedding scatter). PyTorch 2.x + tiktoken + sklearn for the
offline pipeline.

## License

MIT — see `LICENSE`.
