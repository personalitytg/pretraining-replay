"""nanoGPT-style 30M decoder-only transformer."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384
    d_ff: int = 1536
    ctx_len: int = 256
    vocab_size: int = 50257
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.d_head = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.ctx_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tied

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        assert T <= self.cfg.ctx_len, f"sequence length {T} exceeds ctx_len {self.cfg.ctx_len}"
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size), targets.view(-1)
            )
        return logits, loss

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def forward_with_attention(
        self, idx: torch.Tensor
    ) -> tuple[torch.Tensor, list[dict]]:
        """Analytical forward that exposes per-layer attention. Mirrors block logic
        but uses an explicit q@k^T path instead of scaled_dot_product_attention so
        we can return the pre-softmax logits and post-softmax weights."""
        B, T = idx.shape
        assert T <= self.cfg.ctx_len
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)

        attn_per_layer: list[dict] = []
        d_head = self.cfg.d_model // self.cfg.n_head
        scale = 1.0 / math.sqrt(d_head)
        mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=idx.device), diagonal=1
        )

        for block in self.blocks:
            h = block.ln1(x)
            qkv = block.attn.qkv(h)
            q, k, v = qkv.split(self.cfg.d_model, dim=2)
            q = q.view(B, T, self.cfg.n_head, d_head).transpose(1, 2)
            k = k.view(B, T, self.cfg.n_head, d_head).transpose(1, 2)
            v = v.view(B, T, self.cfg.n_head, d_head).transpose(1, 2)
            attn_logits = (q @ k.transpose(-2, -1)) * scale
            attn_logits_masked = attn_logits.masked_fill(mask, float("-inf"))
            attn_weights = F.softmax(attn_logits_masked, dim=-1)
            out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, self.cfg.d_model)
            x = x + block.attn.proj(out)
            x = x + block.mlp(block.ln2(x))
            attn_per_layer.append({
                "weights": attn_weights.detach(),
                "logits": attn_logits_masked.detach(),
            })

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, attn_per_layer

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        finished = torch.zeros(idx.shape[0], dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.ctx_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :].to(torch.float32) / max(temperature, 1e-6)
            if top_k is not None:
                k = min(top_k, logits.shape[-1])
                v, _ = torch.topk(logits, k, dim=-1)
                logits = torch.where(
                    logits < v[:, [-1]], torch.full_like(logits, -float("inf")), logits
                )
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1).squeeze(-1)
            if eos_token_id is not None:
                next_tok = torch.where(
                    finished, torch.full_like(next_tok, eos_token_id), next_tok
                )
                finished = finished | (next_tok == eos_token_id)
            idx = torch.cat([idx, next_tok.unsqueeze(-1)], dim=1)
            if eos_token_id is not None and bool(finished.all()):
                break
        return idx


if __name__ == "__main__":
    import time

    torch.manual_seed(42)
    cfg = GPTConfig()
    model = GPT(cfg)

    n_params = model.num_params()
    print(f"params: {n_params:,}")
    assert 28_000_000 < n_params < 32_000_000, f"param count out of range: {n_params}"

    t0 = time.time()
    idx = torch.randint(0, cfg.vocab_size, (2, 64))
    targets = torch.randint(0, cfg.vocab_size, (2, 64))
    logits, loss = model(idx, targets)
    print("logits.shape:", tuple(logits.shape))
    assert tuple(logits.shape) == (2, 64, cfg.vocab_size)
    assert loss is not None
    print(f"loss: {loss.item():.4f}")
    assert 10.0 < loss.item() < 11.5, f"loss out of expected range: {loss.item()}"

    loss.backward()
    print("ok: backward succeeded")
    print(f"smoke elapsed: {time.time() - t0:.2f}s")
