"""Build manifest.json (schema 9.2) by aggregating training_log.jsonl,
per-checkpoint JSONs, tokens_of_interest.json, and _pca_meta.json."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from generate_artifacts import PROMPTS, PROBES_FOR_MANIFEST
from model import GPT, GPTConfig


MANIFEST_VERSION = "1.0.0"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--run-id", type=str, default=None)
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)

    cfg = yaml.safe_load((runs_dir / "config_snapshot.yaml").read_text(encoding="utf-8"))
    log = {rec["step"]: rec for rec in load_jsonl(runs_dir / "training_log.jsonl")}
    tokens_of_interest = json.loads(
        (out_dir / "tokens_of_interest.json").read_text(encoding="utf-8")
    )
    pca_meta = json.loads((out_dir / "_pca_meta.json").read_text(encoding="utf-8"))

    ckpt_jsons = []
    for p in sorted((out_dir / "checkpoints").glob("step_*.json")):
        ckpt_jsons.append(json.loads(p.read_text(encoding="utf-8")))
    if not ckpt_jsons:
        raise SystemExit("no per-checkpoint JSONs found")
    cadence_steps = [c["step"] for c in ckpt_jsons]

    micro_bs = int(cfg["micro_batch_size"])
    grad_accum = int(cfg["grad_accum_steps"])
    ctx_len = int(cfg["ctx_len"])
    tokens_per_step = micro_bs * ctx_len * grad_accum
    batch_size_effective = micro_bs * grad_accum
    total_steps_actual = max(cadence_steps)
    total_tokens_seen = total_steps_actual * tokens_per_step

    gpt_cfg = GPTConfig(
        n_layer=cfg["n_layer"], n_head=cfg["n_head"], d_model=cfg["d_model"],
        d_ff=cfg["d_ff"], ctx_len=ctx_len, vocab_size=cfg["vocab_size"],
        dropout=cfg.get("dropout", 0.0),
    )
    param_count = GPT(gpt_cfg).num_params()

    ckpt_pt_dir = runs_dir / "checkpoints"
    checkpoints = []
    for step in cadence_steps:
        rec = log.get(step)
        if rec is None:
            raise SystemExit(f"step {step} present in artifacts but not in training_log.jsonl")
        entry = {
            "step": step,
            "tokens_seen": step * tokens_per_step,
            "wallclock_seconds": rec["wallclock_seconds"],
            "loss_train": rec["loss_train"],
            "loss_val": rec["loss_val"],
            "lr": rec["lr"],
        }
        pt_path = ckpt_pt_dir / f"step_{step:06d}.pt"
        if pt_path.exists():
            entry["sha256"] = sha256_file(pt_path)
        else:
            entry["sha256"] = None
            print(f"[warn] checkpoint missing for hash: {pt_path}")
        checkpoints.append(entry)

    probe_sparklines: dict[str, list] = {p["id"]: [] for p in PROBES_FOR_MANIFEST}
    for cj in ckpt_jsons:
        results = cj["probe_results"]
        for probe in PROBES_FOR_MANIFEST:
            probe_sparklines[probe["id"]].append(results[probe["id"]])

    run_id = args.run_id or cfg.get("run_id", "unknown_run")

    manifest = {
        "version": MANIFEST_VERSION,
        "run_id": run_id,
        "model": {
            "n_layer": gpt_cfg.n_layer,
            "n_head": gpt_cfg.n_head,
            "d_model": gpt_cfg.d_model,
            "d_ff": gpt_cfg.d_ff,
            "ctx_len": gpt_cfg.ctx_len,
            "vocab_size": gpt_cfg.vocab_size,
            "param_count": param_count,
        },
        "training": {
            "dataset": "TinyStories",
            "tokenizer": "gpt2",
            "total_steps": total_steps_actual,
            "batch_size_effective": batch_size_effective,
            "tokens_per_step": tokens_per_step,
            "total_tokens_seen": total_tokens_seen,
            "peak_lr": float(cfg["peak_lr"]),
            "seed": int(cfg["seed"]),
        },
        "checkpoints": checkpoints,
        "prompts": PROMPTS,
        "probes": PROBES_FOR_MANIFEST,
        "probe_sparklines": probe_sparklines,
        "tokens_of_interest_count": len(tokens_of_interest["tokens"]),
        "viz_settings": {
            "embedding_xrange": pca_meta["embedding_xrange"],
            "embedding_yrange": pca_meta["embedding_yrange"],
        },
    }

    out_path = out_dir / "manifest.json"
    out_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {out_path}: {len(checkpoints)} checkpoints, "
          f"{len(probe_sparklines)} probe sparklines, param_count={param_count:,}")


if __name__ == "__main__":
    main()
