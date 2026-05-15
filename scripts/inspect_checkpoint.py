"""Print metadata of a cadence checkpoint, assert all weights are bfloat16,
and verify it reloads cleanly into a fresh GPT (with tied lm_head re-tied
in __init__)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import GPT, GPTConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", type=str)
    args = parser.parse_args()

    ckpt = torch.load(Path(args.ckpt), map_location="cpu", weights_only=False)
    print("keys:", list(ckpt.keys()))
    print("step:", ckpt["step"])
    print("loss_train:", ckpt["loss_train"])
    print("loss_val:", ckpt["loss_val"])
    print("lr:", ckpt["lr"])
    print("tokens_seen:", ckpt["tokens_seen"])
    print("wallclock_seconds:", ckpt["wallclock_seconds"])
    print("model_config:", ckpt["model_config"])

    sd = ckpt["model_state_dict"]
    print("model_state_dict entries:", len(sd))
    head = sd.get("lm_head.weight")
    if head is None:
        head = sd.get("tok_emb.weight")
    if head is not None:
        print("lm_head/tok_emb weight shape:", tuple(head.shape), "dtype:", head.dtype)

    dtypes = {v.dtype for v in sd.values()}
    print("dtypes present:", dtypes)
    assert all(v.dtype == torch.bfloat16 for v in sd.values()), "not all bf16"
    print("OK: all bfloat16")

    cfg = GPTConfig(**ckpt["model_config"])
    model = GPT(cfg)
    sd_fp = {k: v.to(torch.float32) for k, v in sd.items()}
    result = model.load_state_dict(sd_fp, strict=False)
    print("missing:", list(result.missing_keys))
    print("unexpected:", list(result.unexpected_keys))
    assert result.unexpected_keys == [], f"unexpected keys: {result.unexpected_keys}"
    assert result.missing_keys == ["lm_head.weight"], (
        f"expected only lm_head.weight to be missing, got {result.missing_keys}"
    )

    model.eval()
    with torch.no_grad():
        idx = torch.zeros(1, 4, dtype=torch.long)
        logits, _ = model(idx)
    print(f"reload OK: 1 tied key skipped, forward shape={tuple(logits.shape)}")


if __name__ == "__main__":
    main()
