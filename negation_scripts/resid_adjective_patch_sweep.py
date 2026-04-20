"""Residual-stream activation patching at the adjective token (layers 0-6).

Clean = negated prompt, Corrupt = ``very`` control. For each layer L, patch
``blocks.L.hook_resid_post`` at the `` happy`` position only, replacing the corrupt
vector with the clean residual. Metric: logit(sad) - logit(happy) at the final position.

A spike in recovery % at layer L suggests negation-relevant state has accumulated in the
residual by the end of that block at the adjective position (informal ``router'' readout).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from transformer_lens import HookedTransformer

MODEL_NAME = "gpt2-small"
CLEAN_PROMPT = "The man is not happy, he is"
CORRUPT_PROMPT = "The man is very happy, he is"
LAYERS = range(0, 7)  # 0 .. 6 inclusive
ADJ_TOKEN_FRAGMENT = " happy"
EPS = 1e-6


def first_token_id(model: HookedTransformer, fragment: str) -> int:
    ids = model.tokenizer.encode(fragment, add_special_tokens=False)
    assert len(ids) >= 1, f"No tokens for fragment: {fragment!r}"
    return int(ids[0])


def logit_diff_last(logits: torch.Tensor, id_sad: int, id_happy: int) -> float:
    final = logits[0, -1]
    return float((final[id_sad] - final[id_happy]).item())


def find_adjective_index(str_tokens: tuple[str, ...] | list[str]) -> int:
    for i, t in enumerate(str_tokens):
        if t.strip() == "happy" or t == ADJ_TOKEN_FRAGMENT:
            return i
    raise ValueError(f"No adjective token matching {ADJ_TOKEN_FRAGMENT!r} in {list(str_tokens)}")


def make_resid_post_patch_hook(
    clean_vec: torch.Tensor,
    pos: int,
) -> Callable[[torch.Tensor, object], torch.Tensor]:
    """Replace resid_post[:, pos, :] with clean_vec (batch 0)."""

    def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
        out = activation.clone()
        v = clean_vec.to(device=out.device, dtype=out.dtype)
        out[:, pos, :] = v
        return out

    return hook_fn


def recovery_pct(ld_clean: float, ld_corrupt: float, ld_patched: float) -> float | None:
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
        raise ValueError("Clean and corrupt prompts must tokenize to the same shape.")

    str_clean = model.to_str_tokens(CLEAN_PROMPT, prepend_bos=prepend_bos)
    str_corrupt = model.to_str_tokens(CORRUPT_PROMPT, prepend_bos=prepend_bos)
    adj_idx = find_adjective_index(str_clean)
    adj_corrupt = find_adjective_index(str_corrupt)
    if adj_idx != adj_corrupt:
        raise ValueError(
            f"Adjective index mismatch: clean {adj_idx} vs corrupt {adj_corrupt}"
        )
    if str_clean[adj_idx] != str_corrupt[adj_idx]:
        raise ValueError(
            f"Adjective token mismatch at {adj_idx}: "
            f"{str_clean[adj_idx]!r} vs {str_corrupt[adj_idx]!r}"
        )

    id_sad = first_token_id(model, " sad")
    id_happy = first_token_id(model, " happy")

    hook_names = [f"blocks.{L}.hook_resid_post" for L in LAYERS]

    with torch.inference_mode():
        _, clean_cache = model.run_with_cache(
            clean_toks, names_filter=hook_names
        )
        _, corrupt_cache = model.run_with_cache(
            corrupt_toks, names_filter=hook_names
        )

        # Sanity: negated vs very should differ in residual at the adjective position.
        chk_L = 3
        chk_key = f"blocks.{chk_L}.hook_resid_post"
        if torch.allclose(
            clean_cache[chk_key][0, adj_idx],
            corrupt_cache[chk_key][0, adj_idx],
        ):
            print(
                f"Warning: clean and corrupt {chk_key} at adjective idx {adj_idx} are "
                "nearly identical; patching may have weak effects."
            )

        logits_clean = model(clean_toks)
        logits_corrupt = model(corrupt_toks)

    ld_clean = logit_diff_last(logits_clean, id_sad, id_happy)
    ld_corrupt = logit_diff_last(logits_corrupt, id_sad, id_happy)
    denom = ld_clean - ld_corrupt

    print(f"Model: {MODEL_NAME}")
    print(f"Clean:   {CLEAN_PROMPT!r}")
    print(f"Corrupt: {CORRUPT_PROMPT!r}")
    print(f"Adjective token: {str_clean[adj_idx]!r} at sequence index {adj_idx}")
    print(f"Metric: logit(' sad') - logit(' happy') at last position")
    print()
    print(f"Baseline LD - Clean:   {ld_clean:+.6f}")
    print(f"Baseline LD - Corrupt: {ld_corrupt:+.6f}")
    print(f"Effect window (Clean - Corrupt): {denom:+.6f}")
    print()
    print(f"{'Layer':>5}  |  {'Patched LD':>14}  |  {'Recovery %':>12}")
    print("-" * 42)

    rows: list[tuple[int, float, float | None]] = []

    with torch.inference_mode():
        for L in LAYERS:
            key = f"blocks.{L}.hook_resid_post"
            clean_vec = clean_cache[key][0, adj_idx].clone()
            hook_fn = make_resid_post_patch_hook(clean_vec, adj_idx)
            with model.hooks(fwd_hooks=[(key, hook_fn)]):
                logits_p = model(corrupt_toks)
            ld_p = logit_diff_last(logits_p, id_sad, id_happy)
            rec = recovery_pct(ld_clean, ld_corrupt, ld_p)
            rows.append((L, ld_p, rec))
            rec_str = (
                f"{100.0 * rec:10.2f}%"
                if rec is not None
                else "         n/a"
            )
            print(f"{L:5d}  |  {ld_p:+14.6f}  |  {rec_str:>12}")

    print()
    print(
        "Recovery % = (Patched_LD - Corrupt_LD) / (Clean_LD - Corrupt_LD). "
        "A sharp increase vs neighboring layers suggests negation-relevant residual "
        "content at the adjective position is present after that block."
    )

    valid = [(L, r) for L, _, r in rows if r is not None]
    if len(valid) >= 2:
        best_L = max(valid, key=lambda x: x[1])[0]
        print(f"Largest recovery on this sweep: layer {best_L}.")


if __name__ == "__main__":
    main()
