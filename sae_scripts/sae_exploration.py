"""Layer-8 MLP-out SAE on a negation-style prompt: top features at the last
token, then top vocab directions by ``W_dec[feat] @ W_U``."""

from __future__ import annotations

import torch
from transformer_lens import HookedTransformer

from sae_lens import SAE


RELEASE = "gpt2-small-mlp-out-v5-32k"
SAE_ID = "blocks.8.hook_mlp_out"
PROMPT = "The man is not happy, he is"


def _feature_acts_dense(x: torch.Tensor) -> torch.Tensor:
    return x.to_dense() if x.is_sparse else x


def top_vocab_tokens_for_feature(
    model: HookedTransformer,
    sae: SAE,
    feature_idx: int,
    k: int = 15,
) -> list[tuple[str, float]]:
    logit_weights = sae.W_dec[feature_idx] @ model.W_U
    vals, ids = torch.topk(logit_weights.detach(), k)
    out: list[tuple[str, float]] = []
    for v, tid in zip(vals.tolist(), ids.tolist()):
        tok_str = model.tokenizer.decode([tid])
        out.append((tok_str, float(v)))
    return out


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    prepend_bos = model.cfg.default_prepend_bos

    sae = SAE.from_pretrained(RELEASE, SAE_ID, device=device)
    hook_name = sae.cfg.metadata.hook_name
    assert hook_name is not None

    toks = model.to_tokens(PROMPT, prepend_bos=prepend_bos)

    with torch.inference_mode():
        _, cache = model.run_with_cache(
            toks,
            names_filter=lambda n: n == hook_name,
        )
        mlp_out = cache[hook_name]
        feature_acts = _feature_acts_dense(sae.encode(mlp_out))

    last = feature_acts.shape[1] - 1
    vec = feature_acts[0, last]
    top = torch.topk(vec, k=10)

    print(f"Device: {device}")
    print(f"Release: {RELEASE!r}, SAE id: {SAE_ID!r}")
    print(f"Hook: {hook_name}")
    print(f"Prompt: {PROMPT!r}")
    print(f"prepend_bos={prepend_bos}, seq_len={toks.shape[1]}, final index={last}")
    print("\nTop 10 SAE features at the final token (index, activation):")
    for rank, (idx, val) in enumerate(
        zip(top.indices.tolist(), top.values.tolist()), start=1
    ):
        print(f"  {rank:2d}. feature {idx:5d}  activation {val:.6f}")

    top_idx = int(top.indices[0].item())
    print(f"\nTop-1 feature: {top_idx}")
    print("Top tokens by W_dec @ W_U:")
    for tok, w in top_vocab_tokens_for_feature(model, sae, top_idx, k=15):
        print(f"  {w:+.4f}  {tok!r}")


if __name__ == "__main__":
    main()
