#!/usr/bin/env python3
"""Refresh generation strings in a curated discoveries.json against fresh per-checkpoint JSONs.

Reads the existing curated discoveries.json (title, step, probe_id, prompt_id,
seed, explanation, tweet_text are KEPT). For each entry, re-fetches the actual
generation text from public/data/checkpoints/step_NNNNNN.json based on
(step, prompt_id, seed) and writes it back into headline_example.generation /
before_example.generation. Whitespace is normalized the same way the frontend does.
"""
import argparse
import json
from pathlib import Path


def normalize(t: str) -> str:
    t = t.replace('\r', ' ').replace('\n', ' ').replace('\t', ' ')
    while '  ' in t:
        t = t.replace('  ', ' ')
    return t.strip()


def fetch_generation(data_dir: Path, step: int, prompt_id: str, seed: int) -> str:
    p = data_dir / "checkpoints" / f"step_{step:06d}.json"
    if not p.exists():
        raise SystemExit(f"missing checkpoint json: {p}")
    d = json.loads(p.read_text(encoding="utf-8"))
    gens = d.get("generations", {}).get(prompt_id, [])
    for g in gens:
        if g["seed"] == seed:
            return normalize(g["text"])
    raise SystemExit(f"no seed {seed} for prompt {prompt_id} at step {step}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="public/data")
    ap.add_argument("--in", dest="inp", default="public/data/discoveries.json")
    ap.add_argument("--out", dest="out", default="public/data/discoveries.json")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    curated = json.loads(Path(args.inp).read_text(encoding="utf-8"))

    for entry in curated:
        h = entry["headline_example"]
        b = entry["before_example"]
        h["generation"] = fetch_generation(data_dir, entry["step"], h["prompt_id"], h["seed"])
        b["generation"] = fetch_generation(data_dir, b["step"], b["prompt_id"], b["seed"])

    Path(args.out).write_text(json.dumps(curated, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"refreshed {len(curated)} discoveries; wrote {args.out}")


if __name__ == "__main__":
    main()
