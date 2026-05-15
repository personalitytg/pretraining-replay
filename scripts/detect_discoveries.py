"""Apply the section-8.1 discovery detection algorithm to probe_sparklines and
emit a draft discoveries.json (schema 8.4). Hand-curation in Phase 3 narrows
this down to 8-12 entries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from generate_artifacts import PROBE_TO_PROMPT


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, required=True)
    return p.parse_args()


def passes(value, kind: str, threshold) -> bool:
    if kind == "fraction":
        return float(value) >= float(threshold)
    if kind == "boolean":
        return bool(value)
    return False


def find_first_passing(values: list, kind: str, threshold) -> int | None:
    for i in range(len(values) - 2):
        if all(passes(values[i + j], kind, threshold) for j in range(3)):
            return i
    return None


def load_generation(out_dir: Path, step: int, prompt_id: str, seed: int) -> str:
    path = out_dir / "checkpoints" / f"step_{step:06d}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for g in data["generations"][prompt_id]:
        if g["seed"] == seed:
            return g["text"]
    raise KeyError(f"no seed={seed} for prompt {prompt_id} at step {step}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    cadence_steps = [c["step"] for c in manifest["checkpoints"]]
    sparklines = manifest["probe_sparklines"]

    discoveries = []
    for probe in manifest["probes"]:
        if probe["kind"] == "number":
            continue
        values = sparklines[probe["id"]]
        idx = find_first_passing(values, probe["kind"], probe["threshold"])
        if idx is None:
            continue
        discovery_step = cadence_steps[idx]
        before_step = cadence_steps[max(idx - 2, 0)]
        prompt_id = PROBE_TO_PROMPT.get(probe["id"], "p_opener")
        try:
            head_gen = load_generation(out_dir, discovery_step, prompt_id, 0)
            before_gen = load_generation(out_dir, before_step, prompt_id, 0)
        except (FileNotFoundError, KeyError) as e:
            print(f"skip {probe['id']}: {e}")
            continue
        discoveries.append({
            "id": f"first_{probe['id']}",
            "title": f"Auto: {probe['title']}",
            "step": discovery_step,
            "probe_id": probe["id"],
            "headline_example": {
                "prompt_id": prompt_id, "seed": 0, "generation": head_gen,
            },
            "before_example": {
                "prompt_id": prompt_id, "seed": 0,
                "step": before_step, "generation": before_gen,
            },
            "explanation": (
                f"Auto-detected: at step {discovery_step:,}, probe "
                f"'{probe['id']}' first crossed threshold {probe['threshold']} "
                f"for 3 consecutive cadence points. Compare to step "
                f"{before_step:,}. (Curate this text in Phase 3.)"
            ),
            "tweet_text": (
                f"Step {discovery_step:,}: '{probe['title']}' emerges. (Auto-draft.)"
            ),
        })

    discoveries.sort(key=lambda d: d["step"])
    (out_dir / "discoveries.json").write_text(
        json.dumps(discoveries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary = ", ".join(f"{d['id']}@{d['step']}" for d in discoveries) or "(none)"
    print(f"detected {len(discoveries)} discoveries: [{summary}]")


if __name__ == "__main__":
    main()
