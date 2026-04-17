"""
Compare attention vs MLP contribution to the sad-minus-happy logit direction
on the negation-style prompt.

For each layer L, take blocks.L.hook_attn_out and blocks.L.hook_mlp_out at the
final token, fold through ln_final.hook_scale (LayerNormPre), and dot with
W_U[:, sad] - W_U[:, happy]. Plots both curves across layers 0..11.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from transformer_lens import HookedTransformer

PROMPT = "The man is not happy, he is"


def first_token_id(model: HookedTransformer, fragment: str) -> int:
    ids = model.tokenizer.encode(fragment, add_special_tokens=False)
    assert len(ids) >= 1, f"No tokens for fragment: {fragment!r}"
    return int(ids[0])


def logit_diff_direction(
    model: HookedTransformer, id_sad: int, id_happy: int
) -> torch.Tensor:
    return model.W_U[:, id_sad] - model.W_U[:, id_happy]


def attribution_score(
    vec: torch.Tensor,
    d_logit: torch.Tensor,
    scale: torch.Tensor,
) -> float:
    """vec in residual stream pre ln_final; d_logit in ln_final output space."""
    v = vec - vec.mean()
    s = scale.to(dtype=torch.float32).clamp(min=1e-8)
    v_tilde = v / s
    return float(torch.dot(v_tilde, d_logit.to(dtype=torch.float32)).item())


def main() -> None:
    model = HookedTransformer.from_pretrained("gpt2-small")

    prepend_bos = model.cfg.default_prepend_bos
    toks = model.to_tokens(PROMPT, prepend_bos=prepend_bos)
    last_pos = toks.shape[1] - 1
    device = model.W_U.device
    n_layers = model.cfg.n_layers

    id_sad = first_token_id(model, " sad")
    id_happy = first_token_id(model, " happy")

    hook_names = {"ln_final.hook_scale"}
    for L in range(n_layers):
        hook_names.add(f"blocks.{L}.hook_attn_out")
        hook_names.add(f"blocks.{L}.hook_mlp_out")

    with torch.inference_mode():
        _, cache = model.run_with_cache(
            toks, names_filter=lambda n: n in hook_names
        )

    scale_last = cache["ln_final.hook_scale"][0, last_pos, 0]
    d_logit = logit_diff_direction(model, id_sad, id_happy).to(device)

    attn_scores: list[float] = []
    mlp_scores: list[float] = []
    for L in range(n_layers):
        attn_vec = cache[f"blocks.{L}.hook_attn_out"][0, last_pos, :]
        mlp_vec = cache[f"blocks.{L}.hook_mlp_out"][0, last_pos, :]
        attn_scores.append(attribution_score(attn_vec, d_logit, scale_last))
        mlp_scores.append(attribution_score(mlp_vec, d_logit, scale_last))

    layers = list(range(n_layers))

    print(f"Prompt: {PROMPT!r}")
    print(f"Final token index: {last_pos}")
    print(
        f"Token ids: ' sad' -> {id_sad}, ' happy' -> {id_happy}  "
        f"(ln_final.hook_scale at last pos: {float(scale_last):.6f})"
    )
    print()
    print(
        "Score = dot( (v - mean(v)) / scale , W_U[:,sad] - W_U[:,happy] )\n"
        "Layer | attn_out | mlp_out"
    )
    print("-" * 42)
    for L in layers:
        print(f"  {L:2d}  | {attn_scores[L]:+8.4f} | {mlp_scores[L]:+8.4f}")

    out = Path(__file__).resolve().parent / "negation_attn_vs_mlp_attribution.png"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(layers, attn_scores, marker="o", label="hook_attn_out")
    ax.plot(layers, mlp_scores, marker="s", label="hook_mlp_out")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(layers)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Attribution (sad - happy direction)")
    ax.set_title(
        "Negation prompt: attn vs MLP write onto logit-diff direction\n"
        f"{PROMPT!r}"
    )
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print()
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
