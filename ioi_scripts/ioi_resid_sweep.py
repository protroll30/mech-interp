"""IOI: patch clean ``hook_resid_mid`` at S2 into corrupt, one layer at a time."""

from __future__ import annotations

import torch
from transformer_lens import HookedTransformer

CLEAN = "When John and Mary went to the store, John gave a bottle of milk to"
CORRUPT = "When John and Mary went to the store, Mary gave a bottle of milk to"


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


def make_patch_resid_mid_s2(layer: int, s2: int, clean_vec: torch.Tensor):
    """clean_vec shape (d_model,) taken from clean_cache at [0, s2, :]."""

    def hook_fn(activation, hook=None):
        x = activation.clone()
        v = clean_vec.to(device=x.device, dtype=x.dtype)
        x[:, s2, :] = v
        return x

    return hook_fn


def main() -> None:
    model = HookedTransformer.from_pretrained("gpt2-small")
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

    print(f"S2 token index (second 'John' slot in clean): {s2}")
    print(f"Clean logit diff (Mary - John @ last):   {ld_clean.item():+.4f}")
    print(f"Corrupt logit diff (Mary - John @ last): {ld_corrupt.item():+.4f}")
    print(
        "\nPer-layer: patch ONLY blocks.L.hook_resid_mid[:, S2, :] from clean into corrupt.\n"
        "Positive LD favors Mary at the final position.\n"
    )

    first_positive = None
    first_cross_clean = None

    n_layers = model.cfg.n_layers
    with torch.inference_mode():
        for layer in range(n_layers):
            key = f"blocks.{layer}.hook_resid_mid"
            clean_vec = clean_cache[key][0, s2, :].clone()
            hook = make_patch_resid_mid_s2(layer, s2, clean_vec)
            logits_p = model.run_with_hooks(
                corrupt_toks,
                fwd_hooks=[(key, hook)],
            )
            ld_p = logit_diff_ioi(logits_p, id_mary, id_john)
            d = (ld_p - ld_corrupt).item()
            mark = ""
            if ld_corrupt.item() <= 0 and ld_p.item() > 0 and first_positive is None:
                first_positive = layer
                mark += "  <-- first layer where LD>0 (Mary favored over John)"
            if ld_p.item() >= ld_clean.item() and first_cross_clean is None:
                first_cross_clean = layer
                if not mark:
                    mark += "  <-- first layer where patched LD >= clean LD"
            print(f"L{layer:2d}: patched_LD={ld_p.item():+.4f}  (Δ vs corrupt={d:+.4f}){mark}")

    print()
    if first_positive is not None:
        print(
            f"Earliest layer where corrupt LD is pushed above 0 (Mary > John): L{first_positive}"
        )
    else:
        print("No single-layer S2 resid_mid patch pushed LD above 0 in this sweep.")

    if first_cross_clean is not None:
        print(f"Earliest layer where patched LD reaches clean LD: L{first_cross_clean}")
    else:
        print("No single-layer patch fully reached clean LD in this sweep.")


if __name__ == "__main__":
    main()
