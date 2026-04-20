"""Q/K/V path patching for L7H5: which pathway carries negation signal into the head?

Clean = negated prompt, Corrupt = ``very`` control (same tokenization length).
Patches one of hook_q / hook_k / hook_v for head 5 only from the clean run into the
corrupt forward. Metric: logit diff (sad - happy) at the last position.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from transformer_lens import HookedTransformer

MODEL_NAME = "gpt2-small"
CLEAN_PROMPT = "The man is not happy, he is"
CORRUPT_PROMPT = "The man is very happy, he is"
LAYER = 7
HEAD = 5

EPS = 1e-6


def first_token_id(model: HookedTransformer, fragment: str) -> int:
    ids = model.tokenizer.encode(fragment, add_special_tokens=False)
    assert len(ids) >= 1, f"No tokens for fragment: {fragment!r}"
    return int(ids[0])


def logit_diff_last(
    logits: torch.Tensor, id_sad: int, id_happy: int
) -> float:
    final = logits[0, -1]
    return float((final[id_sad] - final[id_happy]).item())


def make_patch_one_head_hook(
    clean_activation: torch.Tensor,
    head: int,
) -> Callable[[torch.Tensor, object], torch.Tensor]:
    """Overwrite [batch, pos, head, :] from the clean cache."""

    def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
        out = activation.clone()
        out[:, :, head, :] = clean_activation[:, :, head, :].to(
            device=out.device, dtype=out.dtype
        )
        return out

    return hook_fn


def recovery_pct(ld_clean: float, ld_corrupt: float, ld_patched: float) -> float | None:
    """(Patched - Corrupt) / (Clean - Corrupt). None if denominator ~ 0."""
    denom = ld_clean - ld_corrupt
    if not math.isfinite(denom) or abs(denom) < EPS:
        return None
    return (ld_patched - ld_corrupt) / denom


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)

    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
    prepend_bos = model.cfg.default_prepend_bos

    clean_toks = model.to_tokens(CLEAN_PROMPT, prepend_bos=prepend_bos)
    corrupt_toks = model.to_tokens(CORRUPT_PROMPT, prepend_bos=prepend_bos)
    if clean_toks.shape != corrupt_toks.shape:
        raise ValueError(
            "Clean and corrupt prompts must tokenize to the same shape for patching."
        )

    id_sad = first_token_id(model, " sad")
    id_happy = first_token_id(model, " happy")

    q_name = f"blocks.{LAYER}.attn.hook_q"
    k_name = f"blocks.{LAYER}.attn.hook_k"
    v_name = f"blocks.{LAYER}.attn.hook_v"
    names_filter = [q_name, k_name, v_name]

    with torch.inference_mode():
        _, clean_cache = model.run_with_cache(
            clean_toks, names_filter=names_filter
        )

        logits_clean = model(clean_toks)
        logits_corrupt = model(corrupt_toks)

    ld_clean = logit_diff_last(logits_clean, id_sad, id_happy)
    ld_corrupt = logit_diff_last(logits_corrupt, id_sad, id_happy)

    print(f"Model: {MODEL_NAME}, layer {LAYER}, head {HEAD}")
    print(f"Clean (negated):  {CLEAN_PROMPT!r}")
    print(f"Corrupt (control): {CORRUPT_PROMPT!r}")
    print(f"seq_len={clean_toks.shape[1]}, logit diff = logit(' sad') - logit(' happy') at last pos")
    print()
    print(f"Baseline LD - Clean:   {ld_clean:+.6f}")
    print(f"Baseline LD - Corrupt: {ld_corrupt:+.6f}")
    print(f"Effect window (Clean - Corrupt): {ld_clean - ld_corrupt:+.6f}")
    print()

    interventions = [
        ("Query (hook_q)", q_name, clean_cache[q_name]),
        ("Key (hook_k)", k_name, clean_cache[k_name]),
        ("Value (hook_v)", v_name, clean_cache[v_name]),
    ]

    results: list[tuple[str, float, float | None]] = []

    with torch.inference_mode():
        for label, hook_name, clean_act in interventions:
            hook_fn = make_patch_one_head_hook(clean_act, HEAD)
            with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
                logits_p = model(corrupt_toks)
            ld_p = logit_diff_last(logits_p, id_sad, id_happy)
            rec = recovery_pct(ld_clean, ld_corrupt, ld_p)
            results.append((label, ld_p, rec))

    print("Path patching (clean L7H5 slice -> corrupt run, one pathway at a time):")
    print(f"{'Pathway':<22}  {'Patched LD':>14}  {'Recovery %':>14}")
    print("-" * 54)

    best_label = None
    best_rec = -float("inf")

    for label, ld_p, rec in results:
        if rec is None:
            rec_str = "n/a (|Clean-Corrupt| ~ 0)"
        else:
            rec_str = f"{100.0 * rec:12.2f}%"
            if rec > best_rec:
                best_rec = rec
                best_label = label

        print(f"{label:<22}  {ld_p:+14.6f}  {rec_str:>14}")

    print()
    print(
        "Recovery % = (Patched_LD - Corrupt_LD) / (Clean_LD - Corrupt_LD). "
        "100% means the patch fully closes the clean-vs-corrupt gap on this metric."
    )
    if best_label is not None and math.isfinite(best_rec):
        print(
            f"Largest recovery on this run: {best_label} "
            f"({100.0 * best_rec:.1f}%)."
        )


if __name__ == "__main__":
    main()
