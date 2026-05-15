"""Download TinyStories, tokenize with GPT-2 BPE, write train.bin and val.bin (uint16).

Output format matches nanoGPT exactly.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


EOT = 50256  # GPT-2 <|endoftext|>


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="datasets/tinystories")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("loading dataset roneneldan/TinyStories ...")
    ds = load_dataset("roneneldan/TinyStories")

    stories = []
    for split_name in ("train", "validation"):
        if split_name in ds:
            stories.extend(ds[split_name]["text"])

    if args.smoke:
        stories = stories[:1000]

    n_stories = len(stories)
    print("stories:", n_stories)

    enc = tiktoken.get_encoding("gpt2")
    chunks: list[np.ndarray] = []
    total_tokens = 0
    max_id = 0

    for text in tqdm(stories, desc="tokenizing", unit="story"):
        ids = enc.encode_ordinary(text)
        ids.append(EOT)
        if ids:
            local_max = max(ids)
            if local_max > max_id:
                max_id = local_max
        arr = np.asarray(ids, dtype=np.uint32)
        chunks.append(arr)
        total_tokens += len(arr)

    if max_id >= 65536:
        raise RuntimeError(f"token id {max_id} does not fit in uint16")

    all_tokens = np.concatenate(chunks).astype(np.uint16)
    del chunks

    n_val = max(1, int(round(args.val_fraction * total_tokens)))
    n_train = total_tokens - n_val
    train_tokens = all_tokens[:n_train]
    val_tokens = all_tokens[n_train:]

    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"
    train_tokens.tofile(train_path)
    val_tokens.tofile(val_path)

    elapsed = time.time() - t0
    train_mb = train_path.stat().st_size / (1024 * 1024)
    val_mb = val_path.stat().st_size / (1024 * 1024)

    print()
    print("stories:", n_stories)
    print("total tokens:", total_tokens)
    print(f"train: {len(train_tokens)} tokens, {train_mb:.2f} MB -> {train_path}")
    print(f"val:   {len(val_tokens)} tokens, {val_mb:.2f} MB -> {val_path}")
    print(f"max token id: {max_id}")
    print(f"elapsed: {elapsed:.1f} s")


if __name__ == "__main__":
    main()
