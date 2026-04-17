"""Contrastive SAE activations (final token): negated minus clean, top gains."""

from __future__ import annotations

import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

PROMPT_CLEAN = "The man is happy, he is"
PROMPT_NEGATED = "The man is not happy, he is"

RELEASE = "gpt2-small-mlp-out-v5-32k"
SAE_ID = "blocks.8.hook_mlp_out"


def _feature_acts_dense(x: torch.Tensor) -> torch.Tensor:
    return x.to_dense() if x.is_sparse else x


def _final_mlp_out(
    model: HookedTransformer,
    prompt: str,
    hook_name: str,
) -> torch.Tensor:
    _, cache = model.run_with_cache(prompt, names_filter=[hook_name])
    mlp_out = cache[hook_name]
    return mlp_out[0, -1].detach()


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)

    sae = SAE.from_pretrained(RELEASE, SAE_ID, device=device)
    hook_name = sae.cfg.metadata.hook_name
    assert hook_name is not None
    assert hook_name == SAE_ID
    model_kw = dict(sae.cfg.metadata.model_from_pretrained_kwargs or {})

    model = HookedTransformer.from_pretrained(
        "gpt2-small",
        device=device,
        **model_kw,
    )

    with torch.inference_mode():
        final_clean = _final_mlp_out(model, PROMPT_CLEAN, hook_name)
        final_neg = _final_mlp_out(model, PROMPT_NEGATED, hook_name)

        acts_clean = _feature_acts_dense(sae.encode(final_clean.unsqueeze(0)))
        acts_negated = _feature_acts_dense(sae.encode(final_neg.unsqueeze(0)))

    clean_1d = acts_clean[0]
    neg_1d = acts_negated[0]
    feature_diff = neg_1d - clean_1d

    pos_mask = feature_diff > 0
    if not bool(pos_mask.any()):
        print("No features with positive negated−clean difference.")
        return

    pos_idx = torch.nonzero(pos_mask, as_tuple=False).squeeze(-1)
    k = min(10, int(pos_idx.numel()))
    sub_diff = feature_diff[pos_idx]
    top_vals, top_rel = torch.topk(sub_diff, k=k)
    top_feat_ids = pos_idx[top_rel]

    print(f"Release: {RELEASE!r}, SAE id: {SAE_ID!r}")
    print(f"Hook: {hook_name!r}")
    print(f"Clean prompt:   {PROMPT_CLEAN!r}")
    print(f"Negated prompt: {PROMPT_NEGATED!r}\n")
    print(
        f"{'Feature':>8}  "
        f"{'Negated':>12}  "
        f"{'Clean':>12}  "
        f"{'Difference':>12}"
    )
    print("-" * 52)
    for fid in top_feat_ids.tolist():
        i = int(fid)
        d = float(feature_diff[i].item())
        a_n = float(neg_1d[i].item())
        a_c = float(clean_1d[i].item())
        print(f"{i:8d}  {a_n:12.6f}  {a_c:12.6f}  {d:12.6f}")


if __name__ == "__main__":
    main()
