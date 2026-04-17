"""IOI: clean ``hook_result`` per head into corrupt; recovery toward clean LD."""

import torch
from transformer_lens import HookedTransformer

CLEAN = "When John and Mary went to the store, John gave a bottle of milk to"
CORRUPT = "When John and Mary went to the store, Mary gave a bottle of milk to"

LATE_LAYERS = range(7, 12)
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
    """Mary minus John at the final sequence position (batch 0)."""
    last = logits.shape[1] - 1
    return logits[0, last, id_mary] - logits[0, last, id_john]


def make_patch_hook_result_head_from_clean(
    layer: int,
    head: int,
    clean_cache,
):
    key = f"blocks.{layer}.attn.hook_result"
    clean_head = clean_cache[key][:, :, head, :].clone()

    def hook_fn(activation, hook=None):
        x = activation.clone()
        x[:, :, head, :] = clean_head.to(device=x.device, dtype=x.dtype)
        return x

    return hook_fn


def main() -> None:
    model = HookedTransformer.from_pretrained("gpt2-small")
    model.set_use_attn_result(True)
    prepend_bos = model.cfg.default_prepend_bos

    clean_toks = model.to_tokens(CLEAN, prepend_bos=prepend_bos)
    corrupt_toks = model.to_tokens(CORRUPT, prepend_bos=prepend_bos)
    assert clean_toks.shape == corrupt_toks.shape, (
        f"Token length mismatch: {clean_toks.shape} vs {corrupt_toks.shape}"
    )

    id_mary = first_token_id(model, " Mary")
    id_john = first_token_id(model, " John")

    with torch.inference_mode():
        logits_clean, clean_cache = model.run_with_cache(clean_toks)
        logits_corrupt, _ = model.run_with_cache(corrupt_toks)

    ld_clean = logit_diff_ioi(logits_clean, id_mary, id_john)
    ld_corrupt = logit_diff_ioi(logits_corrupt, id_mary, id_john)
    denom = ld_clean - ld_corrupt

    print(f"Clean LD (Mary - John @ last pos):   {ld_clean.item():.4f}")
    print(f"Corrupt LD (Mary - John @ last pos): {ld_corrupt.item():.4f}")
    print(f"Clean − Corrupt (denominator):       {denom.item():.4f}")

    if denom.abs() < 1e-5:
        print("Denominator near zero; skip recovery grid.")
        return

    recovery = torch.empty(len(LATE_LAYERS), N_HEADS, dtype=torch.float32)

    with torch.inference_mode():
        for li, layer in enumerate(LATE_LAYERS):
            for head in range(N_HEADS):
                hook = make_patch_hook_result_head_from_clean(layer, head, clean_cache)
                logits_p = model.run_with_hooks(
                    corrupt_toks,
                    fwd_hooks=[(f"blocks.{layer}.attn.hook_result", hook)],
                )
                ld_p = logit_diff_ioi(logits_p, id_mary, id_john)
                recovery[li, head] = ((ld_p - ld_corrupt) / denom).detach().cpu()

    print("\n% recovery = (Patched_LD − Corrupt_LD) / (Clean_LD − Corrupt_LD)")
    print("(per head: clean hook_result patched into corrupt forward)\n")

    flat = recovery.reshape(-1)
    topk = torch.topk(flat, k=min(8, flat.numel()))
    print("Top heads by recovery:")
    for rank, (val, idx) in enumerate(zip(topk.values.tolist(), topk.indices.tolist()), start=1):
        layer = list(LATE_LAYERS)[idx // N_HEADS]
        head = idx % N_HEADS
        print(f"  {rank}. L{layer}H{head}: {val:.4f} ({100.0 * val:.1f}%)")

    print("\nFull recovery grid [layer 7..11, head 0..11]:")
    for li, layer in enumerate(LATE_LAYERS):
        row = "  ".join(f"{recovery[li, h]:+.3f}" for h in range(N_HEADS))
        print(f"L{layer}: {row}")


if __name__ == "__main__":
    main()
