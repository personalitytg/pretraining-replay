"""Sanity-check generated artifacts against the schema in section 9.

Usage: python scripts/verify_artifacts.py [--data-dir public/data]
Exits non-zero on any mismatch with a human-readable error.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path


REQUIRED_MANIFEST_TOP = {
    "version": str,
    "run_id": str,
    "model": dict,
    "training": dict,
    "checkpoints": list,
    "prompts": list,
    "probes": list,
    "tokens_of_interest_count": int,
    "viz_settings": dict,
}

REQUIRED_MODEL = {
    "n_layer": int, "n_head": int, "d_model": int, "d_ff": int,
    "ctx_len": int, "vocab_size": int, "param_count": int,
}

REQUIRED_TRAINING = {
    "dataset": str, "tokenizer": str, "total_steps": int,
    "batch_size_effective": int, "tokens_per_step": int,
    "total_tokens_seen": int, "peak_lr": (int, float), "seed": int,
}

REQUIRED_CKPT_META = {
    "step": int, "tokens_seen": int, "wallclock_seconds": (int, float),
    "loss_train": (int, float), "loss_val": (int, float), "lr": (int, float),
}

REQUIRED_PROMPT = {"id": str, "text": str}
REQUIRED_PROBE = {"id": str, "title": str, "kind": str}

ATTENTION_HEADER = 8
ATTENTION_BODY = 6 * 6 * 8 * 8
ATTENTION_TOTAL = ATTENTION_HEADER + ATTENTION_BODY  # 2312

EMBEDDING_HEADER = 16


class VerifyError(Exception):
    pass


def check_keys(obj, schema, where):
    for key, expected in schema.items():
        if key not in obj:
            raise VerifyError(f"{where}: missing key '{key}'")
        if not isinstance(obj[key], expected):
            raise VerifyError(
                f"{where}: key '{key}' has type {type(obj[key]).__name__}, expected {expected}"
            )


def verify_manifest(manifest: dict) -> list[int]:
    check_keys(manifest, REQUIRED_MANIFEST_TOP, "manifest")
    check_keys(manifest["model"], REQUIRED_MODEL, "manifest.model")
    check_keys(manifest["training"], REQUIRED_TRAINING, "manifest.training")

    if not manifest["checkpoints"]:
        raise VerifyError("manifest.checkpoints is empty")
    for i, c in enumerate(manifest["checkpoints"]):
        check_keys(c, REQUIRED_CKPT_META, f"manifest.checkpoints[{i}]")

    if not manifest["prompts"]:
        raise VerifyError("manifest.prompts is empty")
    for i, p in enumerate(manifest["prompts"]):
        check_keys(p, REQUIRED_PROMPT, f"manifest.prompts[{i}]")

    if not manifest["probes"]:
        raise VerifyError("manifest.probes is empty")
    for i, p in enumerate(manifest["probes"]):
        check_keys(p, REQUIRED_PROBE, f"manifest.probes[{i}]")
        if "threshold" not in p:
            raise VerifyError(f"manifest.probes[{i}]: missing 'threshold'")

    viz = manifest["viz_settings"]
    for k in ("embedding_xrange", "embedding_yrange"):
        if k not in viz or not isinstance(viz[k], list) or len(viz[k]) != 2:
            raise VerifyError(f"manifest.viz_settings.{k}: must be a 2-element list")

    return [c["step"] for c in manifest["checkpoints"]]


def verify_per_checkpoint(path: Path, expected_step: int) -> None:
    if not path.exists():
        raise VerifyError(f"missing per-checkpoint file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    for k in ("step", "generations", "top_5_next", "weight_stats", "probe_results"):
        if k not in data:
            raise VerifyError(f"{path.name}: missing key '{k}'")
    if data["step"] != expected_step:
        raise VerifyError(f"{path.name}: step mismatch ({data['step']} vs {expected_step})")


def verify_attention(path: Path) -> None:
    if not path.exists():
        raise VerifyError(f"missing attention file: {path}")
    raw = path.read_bytes()
    if len(raw) != ATTENTION_TOTAL:
        raise VerifyError(f"{path.name}: size {len(raw)} != {ATTENTION_TOTAL}")
    magic, n_layers, n_heads, seq_len, _ = struct.unpack("<4sBBBB", raw[:ATTENTION_HEADER])
    if magic != b"ATTN":
        raise VerifyError(f"{path.name}: bad magic {magic!r}")
    if (n_layers, n_heads, seq_len) != (6, 6, 8):
        raise VerifyError(f"{path.name}: dims {(n_layers, n_heads, seq_len)} != (6, 6, 8)")


def verify_embedding(path: Path, expected_n_tokens: int = 200) -> None:
    if not path.exists():
        raise VerifyError(f"missing embedding file: {path}")
    raw = path.read_bytes()
    expected_total = EMBEDDING_HEADER + expected_n_tokens * 2 * 4
    if len(raw) != expected_total:
        raise VerifyError(f"{path.name}: size {len(raw)} != {expected_total}")
    magic, n_tokens, n_dims, _ = struct.unpack("<4sIII", raw[:EMBEDDING_HEADER])
    if magic != b"EMB1":
        raise VerifyError(f"{path.name}: bad magic {magic!r}")
    if (n_tokens, n_dims) != (expected_n_tokens, 2):
        raise VerifyError(
            f"{path.name}: dims {(n_tokens, n_dims)} != ({expected_n_tokens}, 2)"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="public/data")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    try:
        manifest_path = data_dir / "manifest.json"
        if not manifest_path.exists():
            raise VerifyError(f"missing manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        steps = verify_manifest(manifest)

        for extra in ("discoveries.json", "tokens_of_interest.json"):
            if not (data_dir / extra).exists():
                raise VerifyError(f"missing file: {extra}")
            json.loads((data_dir / extra).read_text(encoding="utf-8"))

        n_curated = int(manifest.get("tokens_of_interest_count", 200))
        for step in steps:
            name = "step_" + str(step).zfill(6)
            verify_per_checkpoint(data_dir / "checkpoints" / (name + ".json"), step)
            verify_attention(data_dir / "attention" / (name + ".bin"))
            verify_embedding(
                data_dir / "embeddings" / (name + ".bin"),
                expected_n_tokens=n_curated,
            )
    except VerifyError as e:
        print("FAIL:", e)
        return 1

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
