"""Head-level path patching at the adjective position (layers 1-2) for negation vs very.

Patches a single head's ``hook_result`` slice at seq index 5 (`` happy``) from the clean
run into the corrupt forward. Ranks heads by recovery on logit(sad) - logit(happy) at the
last token.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from transformer_lens import HookedTransformer

MODEL_NAME = "gpt2-small"
CLEAN_PROMPT = "The man is not happy, he is"
CORRUPT_PROMPT = "The man is very happy, he is"
LAYERS = (1, 2)
N_HEADS = 12
EPS = 1e-6


def first_token_id(model: HookedTransformer, fragment: str) -> int:
    ids = model.tokenizer.encode(fragment, add_special_tokens=False)
    assert len(ids) >= 1, f"No tokens for fragment: {fragment!r}"
    return int(ids[0])


def logit_diff_last(logits: torch.Tensor, id_sad: int, id_happy: int) -> float:
    final = logits[0, -1]
    return float((final[id_sad] - final[id_happy]).item())


def find_token_index(str_tokens: list[str], predicate) -> int:
    for i, t in enumerate(str_tokens):
        if predicate(t):
            return i
    raise ValueError("No matching token index")


def recovery_pct(ld_clean: float, ld_corrupt: float, ld_patched: float) -> float | None:
    denom = ld_clean - ld_corrupt
    if not math.isfinite(denom) or abs(denom) < EPS:
        return None
    return (ld_patched - ld_corrupt) / denom


def make_hook_result_head_pos_patch_hook(
    clean_hr: torch.Tensor,
    head: int,
    pos: int,
) -> Callable[[torch.Tensor, object], torch.Tensor]:
    """Overwrite hook_result[:, pos, head, :] from clean cache."""

    def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
        out = activation.clone()
        src = clean_hr[:, pos, head, :].to(device=out.device, dtype=out.dtype)
        out[:, pos, head, :] = src
        return out

    return hook_fn


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)

    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
    model.set_use_attn_result(True)
    prepend_bos = model.cfg.default_prepend_bos

    clean_toks = model.to_tokens(CLEAN_PROMPT, prepend_bos=prepend_bos)
    corrupt_toks = model.to_tokens(CORRUPT_PROMPT, prepend_bos=prepend_bos)
    if clean_toks.shape != corrupt_toks.shape:
        raise ValueError("Clean and corrupt prompts must tokenize to the same shape.")

    str_clean = model.to_str_tokens(CLEAN_PROMPT, prepend_bos=prepend_bos)
    str_corrupt = model.to_str_tokens(CORRUPT_PROMPT, prepend_bos=prepend_bos)

    def is_happy(tok: str) -> bool:
        return "happy" in tok.lower()

    adj_idx = find_token_index(str_clean, is_happy)
    if str_clean[adj_idx] != str_corrupt[adj_idx]:
        raise ValueError("Adjective token mismatch between clean and corrupt.")

    # Clean prompt negation marker (BPE usually gives a leading-space token).
    neg_idx = find_token_index(str_clean, lambda t: t.strip() == "not")

    id_sad = first_token_id(model, " sad")
    id_happy = first_token_id(model, " happy")

    hr_keys = [f"blocks.{L}.attn.hook_result" for L in LAYERS]
    pattern_keys = [f"blocks.{L}.attn.hook_pattern" for L in LAYERS]

    with torch.inference_mode():
        _, clean_cache = model.run_with_cache(
            clean_toks, names_filter=hr_keys + pattern_keys
        )
        _, corrupt_cache = model.run_with_cache(corrupt_toks, names_filter=hr_keys)
        for L in LAYERS:
            k = f"blocks.{L}.attn.hook_result"
            assert clean_cache[k].shape == corrupt_cache[k].shape

        logits_clean = model(clean_toks)
        logits_corrupt = model(corrupt_toks)

    ld_clean = logit_diff_last(logits_clean, id_sad, id_happy)
    ld_corrupt = logit_diff_last(logits_corrupt, id_sad, id_happy)

    print(f"Model: {MODEL_NAME}")
    print(f"Clean:   {CLEAN_PROMPT!r}")
    print(f"Corrupt: {CORRUPT_PROMPT!r}")
    print(f"Adjective: {str_clean[adj_idx]!r} at index {adj_idx}")
    print(f"Negation token (clean, 'not'): {str_clean[neg_idx]!r} at index {neg_idx}")
    print(f"Metric: logit(' sad') - logit(' happy') at last position")
    print()
    print(f"Baseline LD - Clean:   {ld_clean:+.6f}")
    print(f"Baseline LD - Corrupt: {ld_corrupt:+.6f}")
    print(f"Effect window (Clean - Corrupt): {ld_clean - ld_corrupt:+.6f}")
    print()

    results: list[tuple[int, int, float, float | None]] = []

    with torch.inference_mode():
        for L in LAYERS:
            hook_name = f"blocks.{L}.attn.hook_result"
            clean_hr = clean_cache[hook_name]
            for H in range(N_HEADS):
                hook_fn = make_hook_result_head_pos_patch_hook(
                    clean_hr, H, adj_idx
                )
                with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
                    logits_p = model(corrupt_toks)
                ld_p = logit_diff_last(logits_p, id_sad, id_happy)
                rec = recovery_pct(ld_clean, ld_corrupt, ld_p)
                results.append((L, H, ld_p, rec))

    ranked = sorted(
        results,
        key=lambda x: float("-inf") if x[3] is None else x[3],
        reverse=True,
    )

    print("Top 5 heads by Recovery % (patch clean head write at adjective pos only):")
    print(f"{'Rank':>4}  {'Head':>8}  {'Patched LD':>14}  {'Recovery %':>12}")
    print("-" * 48)
    for rank, (L, H, ld_p, rec) in enumerate(ranked[:5], start=1):
        rec_str = f"{100.0 * rec:10.2f}%" if rec is not None else "         n/a"
        print(f"{rank:4d}  L{L}H{H}     {ld_p:+14.6f}  {rec_str:>12}")

    print()
    print(
        "Recovery % = (Patched_LD - Corrupt_LD) / (Clean_LD - Corrupt_LD). "
        "Higher => that head's clean write at the adjective position carries more "
        "negation-correlated signal for this metric."
    )

    best_L, best_H, _, best_rec = ranked[0]
    if best_rec is None:
        return

    # Optional rigor: attention from adjective (query) to negation key on clean run
    pat_key = f"blocks.{best_L}.attn.hook_pattern"
    pat = clean_cache[pat_key][0, best_H]
    attn_q_adj_to_k_not = float(pat[adj_idx, neg_idx].item())
    print()
    print(
        f"Top head L{best_L}H{best_H}: clean attention weight "
        f"query pos {adj_idx} ({str_clean[adj_idx]!r}) -> "
        f"key pos {neg_idx} ({str_clean[neg_idx]!r}): {attn_q_adj_to_k_not:.6f}"
    )
    print(
        "(High weight supports reading this head as attending from the adjective "
        "position back to the negation token on the clean prompt.)"
    )


if __name__ == "__main__":
    main()
