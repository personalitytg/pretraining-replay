"""nanoGPT-style training loop with cadence checkpointing and resumption.

Cadence checkpoints (227 of them on the canonical run) hold a bf16 model snapshot
plus minimal metadata; they feed downstream artifact generation. A separate
resume_latest.pt holds fp32 weights, optimizer state, and RNG state for
crash-recovery and is overwritten at every cadence step (and every
resume_interval_steps). Saves are atomic (tmp + os.replace).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from model import GPT, GPTConfig

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# Hidden test-only knob: stop the loop after this step (used by the resume smoke).
# Not part of the public CLI.
_CRASH_AT = os.environ.get("PRETRAIN_CRASH_AT")


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_cadence_steps() -> list[int]:
    steps: set[int] = set()
    for s in range(0, 101, 5):
        steps.add(s)
    for s in range(125, 1001, 25):
        steps.add(s)
    for s in range(1100, 10001, 100):
        steps.add(s)
    for s in range(10500, 50001, 500):
        steps.add(s)
    out = sorted(steps)
    assert len(out) == 227, f"expected 227 cadence steps, got {len(out)}"
    return out


def resolve_cadence(value) -> list[int]:
    if value == "section_6_5":
        return compute_cadence_steps()
    if isinstance(value, list) and all(isinstance(s, int) for s in value):
        return sorted(set(value))
    raise ValueError(f"unrecognized cadence value: {value!r}")


def make_get_batch(data: np.memmap, ctx_len: int, batch_size: int, device: str):
    is_cuda = device == "cuda"

    def get_batch() -> tuple[torch.Tensor, torch.Tensor]:
        ix = torch.randint(len(data) - ctx_len - 1, (batch_size,))
        x = torch.stack(
            [torch.from_numpy(data[i : i + ctx_len].astype(np.int64)) for i in ix]
        )
        y = torch.stack(
            [torch.from_numpy(data[i + 1 : i + 1 + ctx_len].astype(np.int64)) for i in ix]
        )
        if is_cuda:
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)
        return x, y

    return get_batch


def make_lr_fn(peak_lr: float, min_lr: float, warmup: int, max_steps: int):
    def get_lr(step: int) -> float:
        if step < warmup:
            return peak_lr * (step + 1) / warmup
        progress = (step - warmup) / max(1, max_steps - warmup)
        progress = min(max(progress, 0.0), 1.0)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (peak_lr - min_lr) * cos

    return get_lr


def build_optimizer(model: torch.nn.Module, cfg: dict, use_fused: bool):
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg["weight_decay"]},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    kwargs = dict(lr=cfg["peak_lr"], betas=(cfg["beta1"], cfg["beta2"]))
    if use_fused:
        try:
            return torch.optim.AdamW(groups, fused=True, **kwargs)
        except TypeError:
            pass
    return torch.optim.AdamW(groups, **kwargs)


@torch.no_grad()
def evaluate(model, get_train_batch, get_val_batch, eval_iters: int, autocast_ctx) -> dict:
    model.eval()
    out = {}
    for split, get_batch in (("train", get_train_batch), ("val", get_val_batch)):
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch()
            with autocast_ctx():
                _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = float(np.mean(losses))
    model.train()
    return out


def atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def truncate_jsonl_after(path: Path, max_step: int) -> None:
    if not path.exists():
        return
    kept = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("step", -1) <= max_step:
            kept.append(line)
    path.write_text(("\n".join(kept) + "\n") if kept else "", encoding="utf-8")


def fresh_start_cleanup(out_dir: Path) -> None:
    for name in ("training_log.jsonl", "resume_latest.pt", "resume_latest.pt.tmp", "final.pt"):
        p = out_dir / name
        if p.exists():
            p.unlink()
    ckpt_dir = out_dir / "checkpoints"
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    resume_path = out_dir / "resume_latest.pt"
    log_path = out_dir / "training_log.jsonl"

    if args.resume and not resume_path.exists():
        print(f"--resume requested but {resume_path} not found", file=sys.stderr)
        sys.exit(1)

    if not args.resume:
        if any(out_dir.iterdir()):
            print(f"warning: {out_dir} is not empty; overwriting (no --resume)")
        fresh_start_cleanup(out_dir)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg_path, out_dir / "config_snapshot.yaml")

    wandb_run = None
    if WANDB_AVAILABLE and os.environ.get("WANDB_API_KEY"):
        wandb_run = wandb.init(
            project="pretraining-replay",
            name=cfg.get("run_id", "tinystories_30m_v1"),
            config=cfg,
            mode="online",
        )
        print(f"[wandb] online: {wandb_run.url}")
    else:
        print("[wandb] disabled (no WANDB_API_KEY or wandb not installed)")

    cadence_list = resolve_cadence(cfg["cadence"])
    cadence_set = set(cadence_list)
    resume_interval = int(cfg.get("resume_interval_steps", 1000))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        # is_bf16_supported() returns True on Turing/Volta where bf16 is software-emulated
        # and 4-5x slower than native fp16. Require Ampere (sm_80) or newer for bf16.
        major, minor = torch.cuda.get_device_capability(0)
        pt_dtype = torch.bfloat16 if major >= 8 else torch.float16
        print(
            f"device: cuda ({torch.cuda.get_device_name(0)}, "
            f"sm_{major}{minor}), dtype: {pt_dtype}"
        )
    else:
        pt_dtype = torch.float32
        print(f"device: cpu, dtype: {pt_dtype}")

    data_dir = Path(cfg["data_dir"])
    train_data = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")
    print(f"train tokens: {len(train_data):,}, val tokens: {len(val_data):,}")

    ctx_len = cfg["ctx_len"]
    micro_bs = cfg["micro_batch_size"]
    grad_accum = cfg["grad_accum_steps"]
    get_train_batch = make_get_batch(train_data, ctx_len, micro_bs, device)
    get_val_batch = make_get_batch(val_data, ctx_len, micro_bs, device)

    gpt_cfg = GPTConfig(
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        d_model=cfg["d_model"],
        d_ff=cfg["d_ff"],
        ctx_len=ctx_len,
        vocab_size=cfg["vocab_size"],
        dropout=cfg["dropout"],
    )
    model_config_dict = {
        "n_layer": gpt_cfg.n_layer,
        "n_head": gpt_cfg.n_head,
        "d_model": gpt_cfg.d_model,
        "d_ff": gpt_cfg.d_ff,
        "ctx_len": gpt_cfg.ctx_len,
        "vocab_size": gpt_cfg.vocab_size,
    }

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])

    model = GPT(gpt_cfg).to(device)
    print(f"params: {model.num_params():,}")

    optimizer = build_optimizer(model, cfg, use_fused=(device == "cuda"))
    use_scaler = device == "cuda" and pt_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None
    autocast_enabled = pt_dtype != torch.float32

    def autocast_ctx():
        return torch.amp.autocast(
            device_type=device, dtype=pt_dtype, enabled=autocast_enabled
        )

    get_lr = make_lr_fn(
        cfg["peak_lr"], cfg["min_lr"], cfg["warmup_steps"], cfg["max_steps"]
    )

    start_step = 0
    wallclock_accum = 0.0
    if args.resume:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        torch.set_rng_state(ckpt["rng_state_torch"])
        if device == "cuda" and ckpt.get("rng_state_torch_cuda") is not None:
            torch.cuda.set_rng_state_all(ckpt["rng_state_torch_cuda"])
        np.random.set_state(ckpt["rng_state_numpy"])
        random.setstate(ckpt["rng_state_python"])
        start_step = ckpt["step"] + 1
        wallclock_accum = float(ckpt.get("wallclock_accum_seconds", 0.0))
        truncate_jsonl_after(log_path, ckpt["step"])
        print(f"resuming from step={ckpt['step']} (wall accum {wallclock_accum:.1f}s)")

    max_steps = cfg["max_steps"]
    log_interval = cfg["log_interval"]
    eval_iters = cfg["eval_iters"]
    grad_clip = cfg["grad_clip"]
    tokens_per_step = micro_bs * grad_accum * ctx_len

    t_segment_start = time.time()
    last_log_print = t_segment_start
    last_train_loss: float | None = None

    def total_wall() -> float:
        return wallclock_accum + (time.time() - t_segment_start)

    def save_resume(step: int) -> None:
        payload = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "rng_state_torch": torch.get_rng_state(),
            "rng_state_torch_cuda": (
                torch.cuda.get_rng_state_all() if device == "cuda" else None
            ),
            "rng_state_numpy": np.random.get_state(),
            "rng_state_python": random.getstate(),
            "config": cfg,
            "wallclock_accum_seconds": total_wall(),
        }
        atomic_save(payload, resume_path)

    def save_cadence(step: int, loss_train: float, loss_val: float, lr: float) -> Path:
        sd = model.state_dict()
        sd.pop("lm_head.weight", None)  # tied to tok_emb.weight; reloader re-ties via __init__
        bf16_state = {
            k: v.detach().to(torch.bfloat16).contiguous() for k, v in sd.items()
        }
        payload = {
            "step": step,
            "model_state_dict": bf16_state,
            "model_config": model_config_dict,
            "loss_train": loss_train,
            "loss_val": loss_val,
            "lr": lr,
            "wallclock_seconds": total_wall(),
            "tokens_seen": step * tokens_per_step,
        }
        path = ckpt_dir / f"step_{step:06d}.pt"
        atomic_save(payload, path)
        return path

    def do_cadence(step: int, lr: float) -> None:
        losses = evaluate(
            model, get_train_batch, get_val_batch, eval_iters, autocast_ctx
        )
        loss_train = float(losses["train"])
        loss_val = float(losses["val"])
        rec = {
            "step": step,
            "loss_train": round(loss_train, 4),
            "loss_val": round(loss_val, 4),
            "lr": lr,
            "wallclock_seconds": round(total_wall(), 2),
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        if wandb_run is not None:
            wandb_run.log({
                "step": step,
                "loss_train": loss_train,
                "loss_val": loss_val,
                "lr": lr,
                "tokens_seen": step * tokens_per_step,
            }, step=step)
        save_cadence(step, loss_train, loss_val, lr)
        save_resume(step)
        print(
            f"[cadence] step={step} loss_train={loss_train:.4f} "
            f"loss_val={loss_val:.4f} lr={lr:.2e}"
        )

    if start_step == 0 and 0 in cadence_set:
        do_cadence(0, get_lr(0))
        start_step = 1
    elif start_step == 0:
        start_step = 1

    model.train()
    for step in range(start_step, max_steps + 1):
        lr = get_lr(step)
        for g in optimizer.param_groups:
            g["lr"] = lr

        accum_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            x, y = get_train_batch()
            with autocast_ctx():
                _, loss = model(x, y)
                loss = loss / grad_accum
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accum_loss += loss.item()

        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        last_train_loss = accum_loss

        if step % log_interval == 0:
            now = time.time()
            dt = now - last_log_print
            tps = tokens_per_step * log_interval / max(dt, 1e-9)
            last_log_print = now
            print(
                f"step={step:>5} loss={last_train_loss:.4f} "
                f"lr={lr:.2e} tok/s={tps:.0f} elapsed={total_wall():.1f}s"
            )

        if step in cadence_set:
            do_cadence(step, lr)
        elif step % resume_interval == 0:
            save_resume(step)

        if _CRASH_AT and step == int(_CRASH_AT):
            print(f"PRETRAIN_CRASH_AT={_CRASH_AT}: simulating crash")
            sys.exit(0)

    if wandb_run is not None:
        wandb_run.finish()
    print(f"done. total wall: {total_wall():.1f}s")


if __name__ == "__main__":
    main()
