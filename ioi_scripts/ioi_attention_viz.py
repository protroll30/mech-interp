"""IOI clean prompt: attention pattern for layer 9 head 9."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from transformer_lens import HookedTransformer

CLEAN = "When John and Mary went to the store, John gave a bottle of milk to"


def main() -> None:
    model = HookedTransformer.from_pretrained("gpt2-small")
    prepend_bos = model.cfg.default_prepend_bos
    toks = model.to_tokens(CLEAN, prepend_bos=prepend_bos)

    with torch.inference_mode():
        _, cache = model.run_with_cache(
            toks,
            names_filter=lambda n: n == "blocks.9.attn.hook_pattern",
        )

    pattern = cache["blocks.9.attn.hook_pattern"]
    attn = pattern[0, 9, :, :].float().cpu().numpy()

    labels = model.to_str_tokens(toks, prepend_bos=prepend_bos)
    if isinstance(labels[0], list):
        labels = labels[0]
    if len(labels) != attn.shape[0]:
        labels = [str(i) for i in range(attn.shape[0])]

    out = Path(__file__).resolve().parent / "ioi_clean_L9H9_attention.png"
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(attn, cmap="viridis", aspect="auto", origin="upper")
    ax.set_title("Clean IOI — Layer 9 Head 9 attention (query rows, key columns)")
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    step = max(1, len(labels) // 32)
    ticks = list(range(0, len(labels), step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([labels[i] for i in ticks], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(ticks)
    ax.set_yticklabels([labels[i] for i in ticks], fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Attention weight")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

    print(f"Saved attention heatmap to {out}")
    print(f"Pattern shape [batch, head, q, k]: {tuple(pattern.shape)}")


if __name__ == "__main__":
    main()
