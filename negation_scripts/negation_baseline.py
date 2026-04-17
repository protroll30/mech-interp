"""
Negation-style prompt on GPT-2 small: logit lens on hook_resid_post.

Clean: "The man is not happy, he is" -> at final position compare logit(" sad")
vs logit(" happy"). Baseline ("The man is very happy") is printed for the same
metric on the full model only.

Logit lens: logits_L = unembed(ln_final(resid_post[L])) at the final sequence
index, for each layer L.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from transformer_lens import HookedTransformer

CLEAN = "The man is not happy, he is"
BASELINE = "The man is very happy"


def first_token_id(model: HookedTransformer, fragment: str) -> int:
    ids = model.tokenizer.encode(fragment, add_special_tokens=False)
    assert len(ids) >= 1, f"No tokens for fragment: {fragment!r}"
    return int(ids[0])


def logit_diff_at(
    logits: torch.Tensor, last_pos: int, id_sad: int, id_happy: int
) -> float:
    return float(
        logits[0, last_pos, id_sad].item() - logits[0, last_pos, id_happy].item()
    )


def main() -> None:
    model = HookedTransformer.from_pretrained("gpt2-small")
    prepend_bos = model.cfg.default_prepend_bos
    n_layers = model.cfg.n_layers

    id_sad = first_token_id(model, " sad")
    id_happy = first_token_id(model, " happy")

    sad_tok = model.tokenizer.decode([id_sad])
    happy_tok = model.tokenizer.decode([id_happy])

    clean_toks = model.to_tokens(CLEAN, prepend_bos=prepend_bos)
    baseline_toks = model.to_tokens(BASELINE, prepend_bos=prepend_bos)
    last_clean = clean_toks.shape[1] - 1
    last_base = baseline_toks.shape[1] - 1

    hook_names = {f"blocks.{L}.hook_resid_post" for L in range(n_layers)}

    with torch.inference_mode():
        clean_logits, cache = model.run_with_cache(
            clean_toks, names_filter=lambda n: n in hook_names
        )
        base_logits = model(baseline_toks)

    ld_full_clean = logit_diff_at(clean_logits, last_clean, id_sad, id_happy)
    ld_full_base = logit_diff_at(base_logits, last_base, id_sad, id_happy)

    ld_lens: list[float] = []
    for L in range(n_layers):
        resid = cache[f"blocks.{L}.hook_resid_post"]
        lens_logits = model.unembed(model.ln_final(resid))
        ld_lens.append(logit_diff_at(lens_logits, last_clean, id_sad, id_happy))

    ld_last = ld_lens[n_layers - 1]
    print(f"Clean prompt: {CLEAN!r}")
    print(f"Baseline prompt: {BASELINE!r}")
    print(
        f"Token ids (first subword): ' sad' -> {id_sad} ({sad_tok!r}), "
        f"' happy' -> {id_happy} ({happy_tok!r})"
    )
    print(
        f"Full model LD at last pos (clean): logit(sad) - logit(happy) = {ld_full_clean:.4f}"
    )
    print(
        f"Full model LD at last pos (baseline): logit(sad) - logit(happy) = {ld_full_base:.4f}"
    )
    print(
        f"Logit lens LD at last pos (layer {n_layers - 1} resid_post): {ld_last:.4f}"
    )
    if abs(ld_last - ld_full_clean) > 0.05:
        print(
            "(Note: small mismatch vs full forward logits can happen; "
            "check TL hook placement.)"
        )

    layers = list(range(n_layers))
    flip_to_sad: int | None = None
    for L in range(n_layers):
        if ld_lens[L] > 0:
            flip_to_sad = L
            break

    if flip_to_sad is None:
        print("Logit lens never crosses LD > 0 (sad never ahead of happy in lens).")
    else:
        print(
            f"First layer where logit lens LD > 0 (sad favored): L{flip_to_sad}"
        )

    out = Path(__file__).resolve().parent / "negation_logit_lens_ld.png"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(layers, ld_lens, marker="o", color="C0", label="Logit lens (clean)")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(layers)
    ax.set_xlabel("Layer (resid_post after block L)")
    ax.set_ylabel("LD = logit(' sad') - logit(' happy') at final token")
    ax.set_title(
        "Negation clean prompt: logit lens LD vs layer\n"
        f"{CLEAN!r}"
    )
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
