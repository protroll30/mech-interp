"""Nonsense-name copy task: clean vs corrupt induction; ``INDUCTION_NEEDLE``
must appear twice verbatim in the clean string (BPE)."""


from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import torch
from transformer_lens import HookedTransformer


RUN_EDGE_SEARCH = True

EARLY_LAYER_RANGE = range(0, 5)


INDUCTION_NEEDLE = " Argl"
TAIL_CLEAN = " Flargh"
FIRST_CORRUPT = " Zump Qrinx"
CONT = TAIL_CLEAN
WRONG_FIRST = " Vosjq"

CEO_PREFIX = "The CEO of the company is"
CEO_MID = ". Later that day,"


def encode_no_bos(model: HookedTransformer, text: str) -> torch.Tensor:
    ids = model.tokenizer.encode(text, add_special_tokens=False)
    return torch.tensor(ids, dtype=torch.long, device=model.W_E.device)


def find_subsequence_starts(haystack: torch.Tensor, needle: torch.Tensor) -> List[int]:
    """All start indices i where haystack[i : i + len(needle)] == needle."""
    n, m = haystack.numel(), needle.numel()
    if m == 0 or n < m:
        return []
    hits: List[int] = []
    for i in range(n - m + 1):
        if torch.equal(haystack[i : i + m], needle):
            hits.append(i)
    return hits


def build_clean_string() -> str:
    """Second occurrence is only INDUCTION_NEEDLE; model predicts CONT (= TAIL_CLEAN)."""
    return f"{CEO_PREFIX}{INDUCTION_NEEDLE}{TAIL_CLEAN}{CEO_MID}{INDUCTION_NEEDLE} "


def build_corrupt_string() -> str:
    """First name differs; second half still ends with INDUCTION_NEEDLE + same trailing layout."""
    return f"{CEO_PREFIX}{FIRST_CORRUPT}{CEO_MID}{INDUCTION_NEEDLE} "


def pad_corrupt_to_clean_token_len(
    model: HookedTransformer, clean_s: str, corrupt_s: str, prepend_bos: bool
) -> Tuple[str, str]:
    """Append spaces to corrupt until token length matches clean (for resid_post alignment)."""
    while model.to_tokens(corrupt_s, prepend_bos=prepend_bos).shape[1] < model.to_tokens(
        clean_s, prepend_bos=prepend_bos
    ).shape[1]:
        corrupt_s += " "
    lc = model.to_tokens(clean_s, prepend_bos=prepend_bos).shape[1]
    lr = model.to_tokens(corrupt_s, prepend_bos=prepend_bos).shape[1]
    assert lc == lr, (
        "Corrupt token length exceeds clean after padding; shorten FIRST_CORRUPT / TAIL_CLEAN "
        f"or adjust strings (clean_len={lc}, corrupt_len={lr})."
    )
    return clean_s, corrupt_s


def make_overwrite_k_from_resid4_hook_factory(
    model: HookedTransformer,
    layer: int,
    head_indices: list[int],
    clean_cache,
    resid4_corrupt: torch.Tensor,
):
    ln1_scale_clean = clean_cache[f"blocks.{layer}.ln1.hook_scale"]
    normalized_corrupt = resid4_corrupt / ln1_scale_clean

    def hook_fn(activation, hook=None):
        x = activation.clone()
        z = normalized_corrupt.to(device=x.device, dtype=x.dtype)
        for H in head_indices:
            w = model.blocks[layer].attn.W_K[H].to(device=x.device, dtype=x.dtype)
            x[:, :, H, :] = z @ w
        return x

    return hook_fn


def build_k_resid4_hooks(
    model: HookedTransformer,
    target_pairs: torch.Tensor,
    clean_cache,
    resid4_corrupt: torch.Tensor,
    n_layers: int,
) -> list:
    fwd_hooks = []
    for layer in range(n_layers):
        in_layer = target_pairs[:, 0] == layer
        head_indices = target_pairs[in_layer, 1].tolist()
        if not head_indices:
            continue
        fwd_hooks.append(
            (
                f"blocks.{layer}.attn.hook_k",
                make_overwrite_k_from_resid4_hook_factory(
                    model, layer, head_indices, clean_cache, resid4_corrupt
                ),
            )
        )
    return fwd_hooks


def logit_diff_at_position(
    logits: torch.Tensor,
    pos: int,
    correct_id: int,
    wrong_id: int,
) -> torch.Tensor:
    """logit(correct) - logit(wrong) at batch 0, position pos (shape scalar tensor)."""
    row = logits[0, pos]
    return row[correct_id] - row[wrong_id]


def make_get_logit_diff_last_tok(correct_id: int, wrong_id: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """Returns get_logit_diff(logits) using the last sequence position (batch 0)."""

    def get_logit_diff(logits: torch.Tensor) -> torch.Tensor:
        last_pos = logits.shape[1] - 1
        return logit_diff_at_position(logits, last_pos, correct_id, wrong_id)

    return get_logit_diff


def make_path_patch_early_to_late_k_hook(
    model: HookedTransformer,
    early_L: int,
    early_H: int,
    late_L: int,
    late_H: int,
    clean_cache,
    corrupt_cache,
):
    """
    Path patch: LN-normalize early per-head hook_result (clean LN of late block),
    project through W_K of (late_L, late_H), subtract clean projection and add
    corrupt projection on blocks.{late_L}.attn.hook_k head late_H.
    """
    ln1 = clean_cache[f"blocks.{late_L}.ln1.hook_scale"]
    h_c = clean_cache[f"blocks.{early_L}.attn.hook_result"][:, :, early_H, :] / ln1
    h_r = corrupt_cache[f"blocks.{early_L}.attn.hook_result"][:, :, early_H, :] / ln1
    w_k = model.blocks[late_L].attn.W_K[late_H]
    clean_proj = h_c @ w_k
    corrupt_proj = h_r @ w_k

    def hook_fn(activation, hook=None):
        x = activation.clone()
        cp = clean_proj.to(device=x.device, dtype=x.dtype)
        crp = corrupt_proj.to(device=x.device, dtype=x.dtype)
        x[:, :, late_H, :] = x[:, :, late_H, :] - cp + crp
        return x

    return hook_fn


def run_edge_search(
    model: HookedTransformer,
    clean_toks: torch.Tensor,
    clean_cache,
    corrupt_cache,
    logits_clean: torch.Tensor,
    target_pairs: torch.Tensor,
    correct_id: int,
    wrong_id: int,
    n_heads: int,
    out_path: Path,
) -> torch.Tensor:
    """
    For each sender (L in 0..4, H in 0..11) and each receiver in ``target_pairs``
    (thresholded induction heads), path-patch early hook_result through that receiver's
    W_K into its hook_k; record clean minus patched logit diff at the last sequence index.
    """
    get_logit_diff = make_get_logit_diff_last_tok(correct_id, wrong_id)
    ld_clean = get_logit_diff(logits_clean)

    n_early = len(EARLY_LAYER_RANGE) * n_heads
    n_recv = int(target_pairs.shape[0])
    drops = torch.empty(n_early, n_recv, device="cpu", dtype=torch.float32)

    recv_labels = [f"L{int(p[0].item())}H{int(p[1].item())}" for p in target_pairs]

    ei = 0
    for early_L in EARLY_LAYER_RANGE:
        for early_H in range(n_heads):
            for lj in range(n_recv):
                late_L = int(target_pairs[lj, 0].item())
                late_H = int(target_pairs[lj, 1].item())
                hook = make_path_patch_early_to_late_k_hook(
                    model,
                    early_L,
                    early_H,
                    late_L,
                    late_H,
                    clean_cache,
                    corrupt_cache,
                )
                logits_p = model.run_with_hooks(
                    clean_toks,
                    fwd_hooks=[(f"blocks.{late_L}.attn.hook_k", hook)],
                )
                ld_p = get_logit_diff(logits_p)
                drops[ei, lj] = (ld_clean - ld_p).detach().cpu()
            ei += 1

    _plot_edge_heatmap(drops, n_heads, out_path, recv_labels)
    return drops


def _plot_edge_heatmap(
    drops: torch.Tensor,
    n_heads: int,
    out_path: Path,
    receiver_labels: list[str],
) -> None:
    n_early, n_recv = drops.shape
    fig_w = max(10.0, 0.35 * n_recv + 4.0)
    fig, ax = plt.subplots(figsize=(fig_w, 14))
    im = ax.imshow(drops.numpy(), aspect="auto", origin="lower", cmap="viridis")
    ax.set_xlabel("Receiver (induction heads from threshold)")
    ax.set_ylabel("Sender (L0-4, rows block by layer then H)")
    ax.set_title("Path-patch edge scores: Δ logit diff (clean − patched)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    yticks = [i * n_heads + n_heads // 2 for i in range(n_early // n_heads)]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"L{i}" for i in EARLY_LAYER_RANGE])

    ax.set_xticks(list(range(n_recv)))
    ax.set_xticklabels(receiver_labels, rotation=45, ha="right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    model = HookedTransformer.from_pretrained("gpt2-small")
    model.set_use_attn_result(True)
    prepend_bos = model.cfg.default_prepend_bos

    clean_str = build_clean_string()
    corrupt_str = build_corrupt_string()
    clean_str, corrupt_str = pad_corrupt_to_clean_token_len(
        model, clean_str, corrupt_str, prepend_bos
    )

    clean_toks = model.to_tokens(clean_str, prepend_bos=prepend_bos)
    corrupt_toks = model.to_tokens(corrupt_str, prepend_bos=prepend_bos)

    needle = encode_no_bos(model, INDUCTION_NEEDLE)
    hay = clean_toks[0]
    starts = find_subsequence_starts(hay, needle)
    assert len(starts) == 2, (
        f"Need exactly two identical tokenized occurrences of INDUCTION_NEEDLE={INDUCTION_NEEDLE!r} "
        f"in the clean prompt; found {len(starts)}. Fix CEO_PREFIX / CEO_MID / spacing."
    )
    s1, s2 = starts[0], starts[1]
    m = needle.numel()
    e1, e2 = s1 + m, s2 + m
    assert torch.equal(hay[s1:e1], hay[s2:e2]), "Needle spans must tokenize identically."

    query_positions = torch.arange(s2, e2, device=model.W_E.device, dtype=torch.long)
    key_positions = torch.arange(s1, s1 + (e2 - s2), device=model.W_E.device, dtype=torch.long)
    assert key_positions.numel() == query_positions.numel()

    pred_pos = e2 - 1
    cont_ids = encode_no_bos(model, CONT)
    assert cont_ids.numel() >= 1
    correct_id = int(cont_ids[0].item())
    wrong_ids = encode_no_bos(model, WRONG_FIRST)
    assert wrong_ids.numel() >= 1
    wrong_id = int(wrong_ids[0].item())
    assert wrong_id != correct_id, "Pick a wrong token different from the correct first token."

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    get_logit_diff = make_get_logit_diff_last_tok(correct_id, wrong_id)
    logits, cache = model.run_with_cache(clean_toks)
    ld_clean = get_logit_diff(logits)
    eval_pos = logits.shape[1] - 1

    induction_scores = torch.empty(n_layers, n_heads, device=model.W_E.device)
    for layer in range(n_layers):
        pattern = cache[f"blocks.{layer}.attn.hook_pattern"]
        paired = pattern[:, :, query_positions, key_positions]
        induction_scores[layer] = paired.mean(dim=(0, 2))

    target_mask = induction_scores > 0.20
    target_pairs = torch.nonzero(target_mask, as_tuple=False)
    n_target = int(target_pairs.shape[0])

    _, corrupted_cache = model.run_with_cache(corrupt_toks)
    resid_post_4_corrupt = corrupted_cache["blocks.4.hook_resid_post"]

    if RUN_EDGE_SEARCH and n_target > 0:
        heatmap_path = Path(__file__).resolve().parent / "edge_search_path_patch_heatmap.png"
        print(
            f"\nEdge search (60 senders x {n_target} induction receivers): writing {heatmap_path} ..."
        )
        run_edge_search(
            model,
            clean_toks,
            cache,
            corrupted_cache,
            logits,
            target_pairs,
            correct_id,
            wrong_id,
            n_heads,
            heatmap_path,
        )
    elif RUN_EDGE_SEARCH:
        print("\nEdge search skipped: no induction heads above threshold (n_target=0).")

    k_hooks = build_k_resid4_hooks(
        model, target_pairs, cache, resid_post_4_corrupt, n_layers
    )
    logits_p = model.run_with_hooks(clean_toks, fwd_hooks=k_hooks)
    ld_patched = get_logit_diff(logits_p)

    print(f"Prompt (clean): {clean_str!r}")
    print(f"Prompt (corrupt): {corrupt_str!r}")
    print(
        f"INDUCTION_NEEDLE token len: {m} | copy-site pred_pos={pred_pos} | "
        f"logit eval_pos (last tok)={eval_pos} | heads>0.2: {n_target}"
    )
    print(f"Logit diff clean (correct - wrong): {ld_clean.item():.4f}")
    print(f"Logit diff after K patch from corrupt resid@L4: {ld_patched.item():.4f}")


if __name__ == "__main__":
    main()
