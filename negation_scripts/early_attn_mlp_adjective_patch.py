"""Path patching at the adjective (layer 1-2 attn vs MLP outs) for negation vs very.

Patches a single hook's activation at seq index 5 only: replace corrupt with clean.
Compares whether early MLPs amplify the signal vs attention sublayers on logit(sad)-logit(happy).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from transformer_lens import HookedTransformer

MODEL_NAME = "gpt2-small"
CLEAN_PROMPT = "The man is not happy, he is"
CORRUPT_PROMPT = "The man is very happy, he is"
ADJ_POS = 5
EPS = 1e-6

PATCHES: list[tuple[str, str, str]] = [
    ("L1 Attention out", "blocks.1.hook_attn_out", "attn"),
    ("L1 MLP out", "blocks.1.hook_mlp_out", "mlp"),
    ("L2 Attention out", "blocks.2.hook_attn_out", "attn"),
    ("L2 MLP out", "blocks.2.hook_mlp_out", "mlp"),
]


def first_token_id(model: HookedTransformer, fragment: str) -> int:
    ids = model.tokenizer.encode(fragment, add_special_tokens=False)
    assert len(ids) >= 1, f"No tokens for fragment: {fragment!r}"
    return int(ids[0])


def logit_diff_last(logits: torch.Tensor, id_sad: int, id_happy: int) -> float:
    final = logits[0, -1]
    return float((final[id_sad] - final[id_happy]).item())


def recovery_pct(ld_clean: float, ld_corrupt: float, ld_patched: float) -> float | None:
    denom = ld_clean - ld_corrupt
    if not math.isfinite(denom) or abs(denom) < EPS:
        return None
    return (ld_patched - ld_corrupt) / denom


def make_pos_patch_hook(
    clean_act: torch.Tensor,
    pos: int,
) -> Callable[[torch.Tensor, object], torch.Tensor]:
    """Overwrite activation[:, pos, :] from clean (batch 0)."""

    def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
        out = activation.clone()
        src = clean_act[:, pos, :].to(device=out.device, dtype=out.dtype)
        out[:, pos, :] = src
        return out

    return hook_fn


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
    if ADJ_POS >= len(str_clean) or "happy" not in str_clean[ADJ_POS].lower():
        raise ValueError(
            f"Expected adjective at index {ADJ_POS}, got {str_clean!r}"
        )

    id_sad = first_token_id(model, " sad")
    id_happy = first_token_id(model, " happy")

    hook_names = [h for _, h, _ in PATCHES]

    with torch.inference_mode():
        _, clean_cache = model.run_with_cache(
            clean_toks, names_filter=hook_names
        )
        logits_clean = model(clean_toks)
        logits_corrupt = model(corrupt_toks)

    ld_clean = logit_diff_last(logits_clean, id_sad, id_happy)
    ld_corrupt = logit_diff_last(logits_corrupt, id_sad, id_happy)

    print(f"Model: {MODEL_NAME}")
    print(f"Clean:   {CLEAN_PROMPT!r}")
    print(f"Corrupt: {CORRUPT_PROMPT!r}")
    print(f"Patch position: {ADJ_POS} ({str_clean[ADJ_POS]!r})")
    print(f"Metric: logit(' sad') - logit(' happy') at last position")
    print()
    print(f"Baseline LD - Clean:   {ld_clean:+.6f}")
    print(f"Baseline LD - Corrupt: {ld_corrupt:+.6f}")
    print(f"Effect window (Clean - Corrupt): {ld_clean - ld_corrupt:+.6f}")
    print()

    rows: list[tuple[str, str, float, float | None]] = []

    with torch.inference_mode():
        for label, hook_name, _kind in PATCHES:
            clean_act = clean_cache[hook_name]
            hook_fn = make_pos_patch_hook(clean_act, ADJ_POS)
            with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
                logits_p = model(corrupt_toks)
            ld_p = logit_diff_last(logits_p, id_sad, id_happy)
            rec = recovery_pct(ld_clean, ld_corrupt, ld_p)
            rows.append((label, hook_name, ld_p, rec))

    print(f"{'Intervention':<22}  {'Patched LD':>12}  {'Recovery %':>12}")
    print("-" * 52)
    for label, _hn, ld_p, rec in rows:
        rs = f"{100.0 * rec:10.2f}%" if rec is not None else "         n/a"
        print(f"{label:<22}  {ld_p:+12.6f}  {rs:>12}")

    print()
    print("Layer 1: Attention out vs MLP out (recovery %)")
    l1_attn = next(r for r in rows if r[0].startswith("L1 Attention"))[3]
    l1_mlp = next(r for r in rows if r[0].startswith("L1 MLP"))[3]
    print(
        f"  Attention: {100.0 * l1_attn:.2f}%"
        if l1_attn is not None
        else "  Attention: n/a"
    )
    print(
        f"  MLP:       {100.0 * l1_mlp:.2f}%"
        if l1_mlp is not None
        else "  MLP:       n/a"
    )
    if l1_attn is not None and l1_mlp is not None:
        if l1_mlp > l1_attn:
            print("  => MLP recovery larger at layer 1 (suggestive of amplification).")
        elif l1_attn > l1_mlp:
            print("  => Attention recovery larger at layer 1.")
        else:
            print("  => Similar recovery.")

    print()
    print("Layer 2: Attention out vs MLP out (recovery %)")
    l2_attn = next(r for r in rows if r[0].startswith("L2 Attention"))[3]
    l2_mlp = next(r for r in rows if r[0].startswith("L2 MLP"))[3]
    print(
        f"  Attention: {100.0 * l2_attn:.2f}%"
        if l2_attn is not None
        else "  Attention: n/a"
    )
    print(
        f"  MLP:       {100.0 * l2_mlp:.2f}%"
        if l2_mlp is not None
        else "  MLP:       n/a"
    )
    if l2_attn is not None and l2_mlp is not None:
        if l2_mlp > l2_attn:
            print("  => MLP recovery larger at layer 2 (suggestive of amplification).")
        elif l2_attn > l2_mlp:
            print("  => Attention recovery larger at layer 2.")
        else:
            print("  => Similar recovery.")

    print()
    print(
        "Recovery % = (Patched_LD - Corrupt_LD) / (Clean_LD - Corrupt_LD). "
        "Compare attn vs mlp within a layer; larger recovery means that sublayer's "
        "clean state at the adjective carries more of the clean-corrupt gap on this metric."
    )


if __name__ == "__main__":
    main()
