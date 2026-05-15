"""Phase 2 / block 2a: per-checkpoint text artifacts.

For each cadence checkpoint, produce a partial per-checkpoint JSON with
schema-9.3 fields step, generations, and top_5_next. weight_stats and
probe_results are added in block 2b; attention/embedding bins in block 2c.
"""

from __future__ import annotations

import argparse
import json
import re
import string
import struct
import time
from pathlib import Path

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F
from sklearn.decomposition import IncrementalPCA

from model import GPT, GPTConfig


PROMPTS = [
    {"id": "s_named_girl", "text": "Once upon a time, there was a little girl named", "role": "showcase"},
    {"id": "s_scared", "text": "Lily was scared because", "role": "showcase"},
    {"id": "s_dragon", "text": "The dragon roared and", "role": "showcase"},
    {"id": "s_cat_dog", "text": "The cat saw the dog and", "role": "showcase"},
    {"id": "s_sunny", "text": "It was a sunny day", "role": "showcase"},
    {"id": "p_dialogue", "text": "He said, ", "role": "probe"},
    {"id": "p_past", "text": "Yesterday, the boy ", "role": "probe"},
    {"id": "p_plural", "text": "There were two ", "role": "probe"},
    {"id": "p_descriptor", "text": "The big red ", "role": "probe"},
    {"id": "p_named", "text": "Lily and the cat ", "role": "probe"},
]
CANONICAL_INPUT = "Once upon a time, there was a"
EOT_TOKEN_ID = 50256
GEN_PARAMS = dict(max_new_tokens=60, temperature=0.8, top_k=50)
N_SEEDS = 5

PROMPT_TEXT = {p["id"]: p["text"] for p in PROMPTS}

PROBES_FOR_MANIFEST = [
    {"id": "uses_quotation_marks", "title": "Uses quotation marks correctly", "kind": "fraction", "threshold": 0.6},
    {"id": "past_tense_after_yesterday", "title": "Past tense after 'Yesterday'", "kind": "fraction", "threshold": 0.6},
    {"id": "plural_after_two", "title": "Plural noun after 'two'", "kind": "fraction", "threshold": 0.6},
    {"id": "coherent_three_word_phrase", "title": "Coherent three-word phrase", "kind": "fraction", "threshold": 0.4},
    {"id": "multi_sentence", "title": "Multi-sentence output", "kind": "fraction", "threshold": 0.4},
    {"id": "consistent_named_entity", "title": "Consistent named entity", "kind": "fraction", "threshold": 0.4},
    {"id": "vocabulary_size", "title": "Distinct vocabulary used", "kind": "number", "threshold": None},
    {"id": "loss_below_threshold", "title": "Validation loss below 2.5", "kind": "boolean", "threshold": 2.5},
]

PROBE_TO_PROMPT = {
    "uses_quotation_marks": "p_dialogue",
    "past_tense_after_yesterday": "p_past",
    "plural_after_two": "p_plural",
    "coherent_three_word_phrase": "s_sunny",
    "multi_sentence": "s_sunny",
    "consistent_named_entity": "p_named",
}

# Curated tokens for the embedding scatter view. Each (text, category) is resolved
# to a single GPT-2 BPE token id at runtime; entries that don't tokenize to exactly
# one token are dropped with a warning.
_NAMED_TOKENS: list[tuple[str, str]] = (
    [(t, "article") for t in ["the", " the", "a", " a", "an", " an"]]
    + [(t, "common_noun") for t in [
        " cat", " dog", " boy", " girl", " house", " tree", " day", " night",
        " man", " woman", " child", " mother", " father", " friend", " bird",
        " ball", " car", " sun", " moon", " water", " food", " toy", " book",
        " story", " thing", " hand", " head", " eye", " way", " time",
    ]]
    + [(t, "verb_present") for t in [
        " is", " was", " has", " had", " can", " will", " would", " want",
        " like", " love", " see", " hear", " feel", " think", " know",
        " say", " tell", " ask", " play", " run", " walk", " sit", " stand",
        " sleep", " eat",
    ]]
    + [(t, "verb_past") for t in [
        " went", " saw", " ran", " walked", " played", " said", " told",
        " asked", " thought", " knew", " felt", " heard", " ate", " sat",
        " stood", " jumped", " laughed", " cried", " wanted", " liked",
        " loved", " found", " gave", " took", " made",
    ]]
    + [(t, "adjective") for t in [
        " big", " small", " tall", " short", " happy", " sad", " good", " bad",
        " nice", " kind", " mean", " smart", " fast", " slow", " new", " old",
        " young", " hot", " cold", " warm", " soft", " hard", " bright",
        " dark", " quiet",
    ]]
    + [(t, "pronoun") for t in [
        " he", " she", " it", " they", " we", " you", " I", " his", " her",
        " its", " their", " my",
    ]]
    + [(t, "name") for t in [
        " Lily", " Tom", " Ben", " Sam", " Anna", " Lucy", " Max", " Tim",
        " John", " Mary",
    ]]
    + [(t, "function_word") for t in [
        " and", " but", " or", " so", " then", " when", " if", " because",
        " in", " on", " to", " from", " with", " for", " of",
    ]]
    + [(t, "number") for t in [" one", " two", " three", " four", " five"]]
    + [(t, "punctuation") for t in [".", ",", "!", "?", '"', "'", ":", ";"]]
)


def build_tokens_of_interest(enc) -> list[dict]:
    out: list[dict] = []
    seen_ids: set[int] = set()
    dropped = 0
    for text, cat in _NAMED_TOKENS:
        ids = enc.encode_ordinary(text)
        if len(ids) != 1:
            dropped += 1
            continue
        tid = ids[0]
        if tid in seen_ids:
            continue
        seen_ids.add(tid)
        out.append({"id": tid, "token_text": text, "category": cat})
    rng = np.random.default_rng(seed=42)
    pool = np.arange(5000, 50000)
    pick = rng.choice(pool, size=80, replace=False)
    rare_added = 0
    for tid in pick:
        if rare_added >= 50:
            break
        tid = int(tid)
        if tid in seen_ids:
            continue
        try:
            text = enc.decode([tid])
        except Exception:
            continue
        seen_ids.add(tid)
        out.append({"id": tid, "token_text": text, "category": "rare"})
        rare_added += 1
    if dropped:
        print(f"warning: {dropped} named tokens skipped (not single-token)")
    print(f"tokens_of_interest: {len(out)} entries")
    return out

PAST_TENSE_RE = re.compile(
    r"\b(?:was|went|ran|saw|said|did|had|made|took|gave|got|came|"
    r"knew|told|felt|found|wanted|liked|played|walked|jumped|"
    r"opened|closed|looked|turned|laughed|cried|asked|answered|"
    r"began|started|finished|remembered)\b"
)
QUOTE_RE = re.compile(r'".+?"')
MULTI_SENT_RE = re.compile(r"\. [A-Z]")
NAMED_RE = re.compile(r"\b(Lily|she|her|hers)\b")
COHERENT_RES = [
    re.compile(r"\bthere was a \w+\b"),
    re.compile(r"\bin a \w+ \w+\b"),
    re.compile(r"\b[A-Z][a-z]+ was a \w+\b"),
    re.compile(r"\bonce there was \w+\b"),
    re.compile(r"\b[A-Z][a-z]+ saw a \w+\b"),
]


def quote_pred(c: str) -> bool:
    return bool(QUOTE_RE.search(c))


def past_tense_pred(c: str) -> bool:
    return bool(PAST_TENSE_RE.search(c[:30]))


def plural_pred(c: str) -> bool:
    stripped = c.lstrip()
    if not stripped:
        return False
    first = stripped.split()[0].strip(string.punctuation)
    return first.endswith("s")


def coherent_phrase_pred(c: str) -> bool:
    return any(r.search(c) for r in COHERENT_RES)


def multi_sentence_pred(c: str) -> bool:
    return bool(MULTI_SENT_RE.search(c))


def named_entity_pred(c: str) -> bool:
    return bool(NAMED_RE.search(c))


def fraction_pass(generations: list[dict], prompt_text: str, pred) -> float:
    if not generations:
        return 0.0
    hits = 0
    for g in generations:
        text = g["text"]
        cont = text[len(prompt_text):] if text.startswith(prompt_text) else text
        if pred(cont):
            hits += 1
    return round(hits / len(generations), 3)


def vocab_size(generations_dict: dict) -> int:
    seen: set[int] = set()
    for gens in generations_dict.values():
        for g in gens:
            seen.update(g["tokens"])
    return len(seen)


def compute_probe_results(generations: dict, loss_val: float) -> dict:
    return {
        "uses_quotation_marks": fraction_pass(
            generations["p_dialogue"], PROMPT_TEXT["p_dialogue"], quote_pred
        ),
        "past_tense_after_yesterday": fraction_pass(
            generations["p_past"], PROMPT_TEXT["p_past"], past_tense_pred
        ),
        "plural_after_two": fraction_pass(
            generations["p_plural"], PROMPT_TEXT["p_plural"], plural_pred
        ),
        "coherent_three_word_phrase": fraction_pass(
            generations["s_sunny"], PROMPT_TEXT["s_sunny"], coherent_phrase_pred
        ),
        "multi_sentence": fraction_pass(
            generations["s_sunny"], PROMPT_TEXT["s_sunny"], multi_sentence_pred
        ),
        "consistent_named_entity": fraction_pass(
            generations["p_named"], PROMPT_TEXT["p_named"], named_entity_pred
        ),
        "vocabulary_size": vocab_size(generations),
        "loss_below_threshold": bool(loss_val < 2.5),
    }


@torch.no_grad()
def compute_weight_stats(model: GPT) -> dict:
    emb = model.tok_emb.weight.detach().to(torch.float32).cpu()
    out_emb = model.lm_head.weight.detach().to(torch.float32).cpu()

    sv = torch.linalg.svdvals(emb)[:5].tolist()

    layer_norms = []
    for i, block in enumerate(model.blocks):
        layer_norms.append({
            "layer": i,
            "qkv_norm": round(
                block.attn.qkv.weight.detach().float().norm().item(), 4
            ),
            "attn_out_norm": round(
                block.attn.proj.weight.detach().float().norm().item(), 4
            ),
            "ffn_up_norm": round(
                block.mlp.up.weight.detach().float().norm().item(), 4
            ),
            "ffn_down_norm": round(
                block.mlp.down.weight.detach().float().norm().item(), 4
            ),
        })

    return {
        "embedding_norm": round(emb.norm().item(), 4),
        "output_embedding_norm": round(out_emb.norm().item(), 4),
        "layer_norms": layer_norms,
        "embedding_top5_singular_values": [round(float(s), 4) for s in sv],
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--steps", type=str, default=None,
                   help="optional comma-separated step filter, e.g. 0,25,50")
    return p.parse_args()


def resolve_device(spec: str) -> str:
    if spec == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return spec


def discover_checkpoints(runs_dir: Path, step_filter: set[int] | None):
    ckpt_dir = runs_dir / "checkpoints"
    out = []
    for p in sorted(ckpt_dir.glob("step_*.pt")):
        step = int(p.stem.split("_")[1])
        if step_filter is None or step in step_filter:
            out.append((step, p))
    return out


def trim_at_eot(tokens: list[int]) -> list[int]:
    if EOT_TOKEN_ID in tokens:
        return tokens[: tokens.index(EOT_TOKEN_ID)]
    return tokens


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"device: {device}")

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    step_filter = None
    if args.steps:
        step_filter = {int(s) for s in args.steps.split(",")}

    ckpts = discover_checkpoints(runs_dir, step_filter)
    if not ckpts:
        raise SystemExit(f"no checkpoints found in {runs_dir}/checkpoints/")
    print(f"found {len(ckpts)} checkpoints")

    enc = tiktoken.get_encoding("gpt2")
    prompt_ids = {p["id"]: enc.encode_ordinary(p["text"]) for p in PROMPTS}
    canonical_ids = enc.encode_ordinary(CANONICAL_INPUT)
    print(f"canonical input: {len(canonical_ids)} tokens -> {canonical_ids}")

    tokens_of_interest = build_tokens_of_interest(enc)
    (out_dir / "tokens_of_interest.json").write_text(
        json.dumps({"tokens": tokens_of_interest}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    curated_ids = np.array([t["id"] for t in tokens_of_interest], dtype=np.int64)
    n_curated = len(curated_ids)

    rng = np.random.default_rng(seed=42)
    fit_sample_ids = np.unique(np.concatenate([
        rng.choice(50257, size=1000, replace=False),
        curated_ids,
    ]))
    ipca = IncrementalPCA(n_components=2)
    curated_per_step: dict[int, np.ndarray] = {}

    (out_dir / "attention").mkdir(parents=True, exist_ok=True)
    (out_dir / "embeddings").mkdir(parents=True, exist_ok=True)

    if device == "cuda":
        model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        model_dtype = torch.float32

    for k, (step, path) in enumerate(ckpts, start=1):
        t0 = time.time()
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = GPTConfig(**ckpt["model_config"])
        model = GPT(cfg)
        sd_fp = {kk: v.to(torch.float32) for kk, v in ckpt["model_state_dict"].items()}
        result = model.load_state_dict(sd_fp, strict=False)
        assert result.missing_keys == ["lm_head.weight"], result.missing_keys
        assert result.unexpected_keys == [], result.unexpected_keys
        model = model.to(device=device, dtype=model_dtype).eval()

        generations: dict[str, list[dict]] = {}
        for prompt in PROMPTS:
            pid = prompt["id"]
            ptext = prompt["text"]
            ptoks = prompt_ids[pid]
            seeds_out = []
            for seed in range(N_SEEDS):
                torch.manual_seed(seed)
                idx = torch.tensor([ptoks], dtype=torch.long, device=device)
                out = model.generate(idx, eos_token_id=EOT_TOKEN_ID, **GEN_PARAMS)
                new_tokens = trim_at_eot(out[0, len(ptoks):].tolist())
                text = ptext + enc.decode(new_tokens)
                seeds_out.append({"seed": seed, "tokens": new_tokens, "text": text})
            generations[pid] = seeds_out

        idx_canon = torch.tensor([canonical_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, _ = model(idx_canon)
        probs = F.softmax(logits[0].to(torch.float32), dim=-1)
        topk_vals, topk_ids = torch.topk(probs, 5, dim=-1)
        top_5_next = []
        for pos in range(len(canonical_ids)):
            cands = []
            for j in range(5):
                tid = int(topk_ids[pos, j].item())
                cands.append({
                    "token": enc.decode([tid]),
                    "prob": round(float(topk_vals[pos, j].item()), 6),
                })
            top_5_next.append({
                "position": pos,
                "context_token": enc.decode([canonical_ids[pos]]),
                "top_5": cands,
            })

        weight_stats = compute_weight_stats(model)

        attn_idx = torch.tensor([canonical_ids], dtype=torch.long, device=device)
        _, attn_data = model.forward_with_attention(attn_idx)
        T = len(canonical_ids)
        attn_stack = torch.stack([d["weights"] for d in attn_data], dim=0)
        attn_stack = attn_stack.squeeze(1).to(torch.float32).cpu()  # [n_layer, n_head, T, T]
        n_layer, n_head, _, _ = attn_stack.shape
        q = (attn_stack * 127.0).round().clamp(-127, 127).to(torch.int8).numpy()
        header = struct.pack("<4sBBBB", b"ATTN", n_layer, n_head, T, 0)
        (out_dir / "attention" / f"step_{step:06d}.bin").write_bytes(
            header + q.tobytes()
        )

        upper_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        valid_mask = ~upper_mask
        attn_logit_mean: list[float] = []
        attn_logit_std: list[float] = []
        for d in attn_data:
            lg = d["logits"][0].to(torch.float32).cpu()  # [n_head, T, T]
            valid = lg[:, valid_mask]
            attn_logit_mean.append(round(float(valid.mean().item()), 4))
            attn_logit_std.append(round(float(valid.std().item()), 4))
        weight_stats["attention_logit_mean_per_layer"] = attn_logit_mean
        weight_stats["attention_logit_std_per_layer"] = attn_logit_std

        emb_full = model.tok_emb.weight.detach().to(torch.float32).cpu().numpy()
        ipca.partial_fit(emb_full[fit_sample_ids])
        curated_per_step[step] = emb_full[curated_ids].copy()

        loss_val = float(ckpt["loss_val"])
        probe_results = compute_probe_results(generations, loss_val)

        payload = {
            "step": step,
            "generations": generations,
            "top_5_next": top_5_next,
            "weight_stats": weight_stats,
            "probe_results": probe_results,
        }
        out_path = out_dir / "checkpoints" / f"step_{step:06d}.json"
        out_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        dt = time.time() - t0
        print(f"[{k}/{len(ckpts)}] step={step} gen_time={dt:.1f}s -> {out_path.name}")

    print("transforming curated embeddings via fitted IncrementalPCA")
    all_x: list[float] = []
    all_y: list[float] = []
    for step, curated_emb in curated_per_step.items():
        coords = ipca.transform(curated_emb).astype(np.float32)
        if not np.isfinite(coords).all():
            raise RuntimeError(f"non-finite PCA coords at step {step}")
        n_tok, n_dim = coords.shape
        header = struct.pack("<4sIII", b"EMB1", n_tok, n_dim, 0)
        (out_dir / "embeddings" / f"step_{step:06d}.bin").write_bytes(
            header + coords.tobytes()
        )
        all_x.extend(coords[:, 0].tolist())
        all_y.extend(coords[:, 1].tolist())

    pca_meta = {
        "embedding_xrange": [round(float(min(all_x)), 4), round(float(max(all_x)), 4)],
        "embedding_yrange": [round(float(min(all_y)), 4), round(float(max(all_y)), 4)],
        "n_tokens": n_curated,
        "pca_components_shape": list(ipca.components_.shape),
    }
    (out_dir / "_pca_meta.json").write_text(
        json.dumps(pca_meta, indent=2), encoding="utf-8"
    )
    print("pca_meta:", json.dumps(pca_meta))

    from verify_artifacts import verify_attention, verify_embedding
    for step, _ in ckpts:
        verify_attention(out_dir / "attention" / f"step_{step:06d}.bin")
        verify_embedding(
            out_dir / "embeddings" / f"step_{step:06d}.bin",
            expected_n_tokens=n_curated,
        )
    print(f"verifier OK on {len(ckpts)} attention + {len(ckpts)} embedding bins")


if __name__ == "__main__":
    main()
