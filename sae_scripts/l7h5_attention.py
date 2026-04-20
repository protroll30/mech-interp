"""Visualize and analyze L7H5 attention on a negation prompt (gpt2-small).

Writes an interactive CircuitsVis HTML file and prints last-token attention stats.
"""

from __future__ import annotations

import math
from pathlib import Path

import circuitsvis as cv
import torch
from transformer_lens import HookedTransformer

MODEL_NAME = "gpt2-small"
PROMPT = "The man is not happy, he is"
LAYER = 7
HEAD = 5

# Key substrings for grouping (GPT-2 BPE often uses leading spaces).
NEGATION_MARKERS = ("not", "never", "failed")
ADJECTIVE_MARKER = "happy"

OUT_HTML = Path(__file__).resolve().parent / "l7h5_attention.html"


def _token_matches_negation(tok: str) -> bool:
    t = tok.lower().strip()
    return any(m in t for m in NEGATION_MARKERS)


def _token_is_adjective_happy(tok: str) -> bool:
    return "happy" in tok.lower()


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)

    hook_pattern = f"blocks.{LAYER}.attn.hook_pattern"

    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
    prepend_bos = model.cfg.default_prepend_bos
    toks = model.to_tokens(PROMPT, prepend_bos=prepend_bos)
    str_tokens = model.to_str_tokens(PROMPT, prepend_bos=prepend_bos)

    with torch.inference_mode():
        _, cache = model.run_with_cache(
            toks,
            names_filter=lambda name: name == hook_pattern,
        )

    pattern = cache[hook_pattern][0, HEAD].float().cpu()
    # [query_pos, key_pos]
    seq_len = pattern.shape[0]

    # Interactive visualization (single head matrix: dest query x source key)
    rendered = cv.attention.attention_heads(
        pattern,
        list(str_tokens),
        attention_head_names=[f"L{LAYER}H{HEAD}"],
        mask_upper_tri=True,
    )
    OUT_HTML.write_text(str(rendered), encoding="utf-8")
    print(f"Wrote {OUT_HTML}")

    # Attention from the final token (query) to all key positions
    q_last = seq_len - 1
    last_row = pattern[q_last].tolist()
    print(f"\nAttention from final query token {q_last!r} ({str_tokens[q_last]!r}) to all keys:")
    print(f"{'key_idx':>7}  {'token':<18}  {'weight':>10}")
    for j, w in enumerate(last_row):
        print(f"{j:7d}  {str_tokens[j]!r:<18}  {w:10.6f}")
    print(f"Sum: {sum(last_row):.6f}")

    neg_indices = [j for j in range(seq_len) if _token_matches_negation(str_tokens[j])]
    happy_indices = [j for j in range(seq_len) if _token_is_adjective_happy(str_tokens[j])]

    attn_neg = sum(last_row[j] for j in neg_indices)
    attn_happy = sum(last_row[j] for j in happy_indices)

    print("\nNegation vs adjective (key positions for this prompt):")
    print(f"  Negation key indices ({', '.join(NEGATION_MARKERS)}): {neg_indices}")
    print(f"  'Happy' key indices: {happy_indices}")
    print(f"  Sum attention to negation token(s): {attn_neg:.6f}")
    print(f"  Sum attention to 'happy' token(s):  {attn_happy:.6f}")

    if attn_happy > 1e-8:
        ratio = attn_neg / attn_happy
        print(
            f"\nNegation attention ratio (sum attn to negation / sum attn to happy): "
            f"{ratio:.4f}"
        )
    else:
        print("\nNegation attention ratio: undefined (no mass on 'happy' key).")
        ratio = float("nan")

    if not math.isfinite(ratio):
        pass
    elif ratio > 1.0:
        print("  (Ratio > 1: more last-token mass on negation key(s) than on 'happy'.)")
    elif ratio < 1.0:
        print("  (Ratio < 1: more last-token mass on 'happy' than on negation key(s).)")


if __name__ == "__main__":
    main()
