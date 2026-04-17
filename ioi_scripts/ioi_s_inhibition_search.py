"""IOI: project clean sender ``hook_result`` at S2 into name-mover ``hook_q``."""

from __future__ import annotations

from typing import List, Tuple

import torch
from transformer_lens import HookedTransformer

CLEAN = "When John and Mary went to the store, John gave a bottle of milk to"
CORRUPT = "When John and Mary went to the store, Mary gave a bottle of milk to"

NAME_MOVER_TARGETS: List[Tuple[int, int]] = [(9, 9), (8, 10), (7, 9)]

SENDER_LAYERS = range(0, 9)
N_HEADS = 12


def first_token_id(model: HookedTransformer, fragment: str) -> int:
    ids = model.tokenizer.encode(fragment, add_special_tokens=False)
    assert len(ids) >= 1, f"No tokens for fragment: {fragment!r}"
    return int(ids[0])


def logit_diff_ioi(
    logits: torch.Tensor,
    id_mary: int,
    id_john: int,
) -> torch.Tensor:
    last = logits.shape[1] - 1
    return logits[0, last, id_mary] - logits[0, last, id_john]


def second_john_token_index(model: HookedTransformer, text: str, prepend_bos: bool) -> int:
    """Token index (into model.to_tokens row) of the token spanning the *second* 'John'."""
    j0 = text.find("John")
    assert j0 != -1, "No 'John' in prompt"
    j1 = text.find("John", j0 + len("John"))
    assert j1 != -1, "Expected two 'John' substrings for IOI S2"

    enc = model.tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    idx = None
    for i, (s, e) in enumerate(enc["offset_mapping"]):
        if s <= j1 < e or (s == j1 and e == j1):
            idx = i
            break
    assert idx is not None, "Tokenizer offset map did not cover second 'John'"
    if prepend_bos:
        idx += 1
    return int(idx)


def make_set_q_at_s2_from_sender_vec(
    s2: int,
    target_head: int,
    q_override: torch.Tensor,
):
    """q_override shape (d_head,); written to activation[:, s2, target_head, :]."""

    def hook_fn(activation, hook=None):
        x = activation.clone()
        q = q_override.to(device=x.device, dtype=x.dtype)
        x[:, s2, target_head, :] = q
        return x

    return hook_fn


def build_q_patch_hooks_for_sender(
    model: HookedTransformer,
    sender_layer: int,
    sender_head: int,
    s2: int,
    clean_cache,
) -> list:
    key = f"blocks.{sender_layer}.attn.hook_result"
    v = clean_cache[key][0, s2, sender_head, :].clone()
    hooks = []
    for Lt, Ht in NAME_MOVER_TARGETS:
        w_q = model.blocks[Lt].attn.W_Q[Ht]
        q_override = v @ w_q
        hooks.append(
            (
                f"blocks.{Lt}.attn.hook_q",
                make_set_q_at_s2_from_sender_vec(s2, Ht, q_override),
            )
        )
    return hooks


def main() -> None:
    model = HookedTransformer.from_pretrained("gpt2-small")
    model.set_use_attn_result(True)
    prepend_bos = model.cfg.default_prepend_bos

    clean_toks = model.to_tokens(CLEAN, prepend_bos=prepend_bos)
    corrupt_toks = model.to_tokens(CORRUPT, prepend_bos=prepend_bos)
    assert clean_toks.shape == corrupt_toks.shape

    s2 = second_john_token_index(model, CLEAN, prepend_bos)
    id_mary = first_token_id(model, " Mary")
    id_john = first_token_id(model, " John")

    with torch.inference_mode():
        logits_clean, clean_cache = model.run_with_cache(clean_toks)
        logits_corrupt, _ = model.run_with_cache(corrupt_toks)

    ld_clean = logit_diff_ioi(logits_clean, id_mary, id_john)
    ld_corrupt = logit_diff_ioi(logits_corrupt, id_mary, id_john)
    denom = ld_clean - ld_corrupt

    print(f"S2 token index (second 'John'): {s2}")
    print(f"Name mover Q targets: {NAME_MOVER_TARGETS}")
    print(f"Clean LD:    {ld_clean.item():.4f}")
    print(f"Corrupt LD:  {ld_corrupt.item():.4f}")
    print(f"Denominator: {denom.item():.4f}\n")

    if denom.abs() < 1e-5:
        print("Denominator near zero; abort sender sweep.")
        return

    n_send = len(SENDER_LAYERS) * N_HEADS
    deltas = torch.empty(n_send, dtype=torch.float32)
    recoveries = torch.empty(n_send, dtype=torch.float32)

    si = 0
    with torch.inference_mode():
        for Ls in SENDER_LAYERS:
            for Hs in range(N_HEADS):
                fwd_hooks = build_q_patch_hooks_for_sender(
                    model, Ls, Hs, s2, clean_cache
                )
                logits_p = model.run_with_hooks(corrupt_toks, fwd_hooks=fwd_hooks)
                ld_p = logit_diff_ioi(logits_p, id_mary, id_john)
                deltas[si] = (ld_p - ld_corrupt).detach().cpu()
                recoveries[si] = (deltas[si] / denom).detach().cpu()
                si += 1

    topk = torch.topk(recoveries, k=min(12, recoveries.numel()))
    print("Top senders by % recovery toward clean LD:")
    print("  (Patched_LD - Corrupt_LD) / (Clean_LD - Corrupt_LD)\n")
    for rank, (val, idx) in enumerate(zip(topk.values.tolist(), topk.indices.tolist()), start=1):
        Ls = idx // N_HEADS
        Hs = idx % N_HEADS
        print(
            f"  {rank:2}. L{Ls}H{Hs}: recovery={val:.4f} ({100.0 * val:+.1f}%), "
            f"ΔLD={deltas[idx].item():+.4f}"
        )


if __name__ == "__main__":
    main()
