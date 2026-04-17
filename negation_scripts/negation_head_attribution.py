"""Negation prompt: rank attention head writes by projection onto sad-minus-happy
in ln_final space (center, divide by ``ln_final.hook_scale``, dot with ``W_U`` diff)."""

import torch
from transformer_lens import HookedTransformer

PROMPT = "The man is not happy, he is"
LAYERS = (5, 6, 7)


def first_token_id(model: HookedTransformer, fragment: str) -> int:
    ids = model.tokenizer.encode(fragment, add_special_tokens=False)
    assert len(ids) >= 1, f"No tokens for fragment: {fragment!r}"
    return int(ids[0])


def logit_diff_direction(model: HookedTransformer, id_sad: int, id_happy: int) -> torch.Tensor:
    return model.W_U[:, id_sad] - model.W_U[:, id_happy]


def head_write(
    cache: dict, layer: int, head: int, last_pos: int, device: torch.device
) -> torch.Tensor:
    key = f"blocks.{layer}.attn.hook_result"
    return cache[key][0, last_pos, head, :].to(device=device, dtype=torch.float32)


def attribution_score(
    head_vec: torch.Tensor,
    d_logit: torch.Tensor,
    scale: torch.Tensor,
) -> float:
    h = head_vec - head_vec.mean()
    s = scale.to(dtype=torch.float32).clamp(min=1e-8)
    h_tilde = h / s
    return float(torch.dot(h_tilde, d_logit.to(dtype=torch.float32)).item())


def main() -> None:
    model = HookedTransformer.from_pretrained("gpt2-small")
    model.set_use_attn_result(True)

    prepend_bos = model.cfg.default_prepend_bos
    toks = model.to_tokens(PROMPT, prepend_bos=prepend_bos)
    last_pos = toks.shape[1] - 1
    device = model.W_U.device

    id_sad = first_token_id(model, " sad")
    id_happy = first_token_id(model, " happy")

    hook_names = {
        f"blocks.{L}.attn.hook_result" for L in LAYERS
    } | {"ln_final.hook_scale"}

    with torch.inference_mode():
        _, cache = model.run_with_cache(
            toks, names_filter=lambda n: n in hook_names
        )

    scale_last = cache["ln_final.hook_scale"][0, last_pos, 0]
    d_logit = logit_diff_direction(model, id_sad, id_happy).to(device)

    print(f"Prompt: {PROMPT!r}")
    print(f"Final token index: {last_pos}")
    print(
        f"Token ids: ' sad' -> {id_sad}, ' happy' -> {id_happy}  "
        f"(ln_final.hook_scale at last pos: {float(scale_last):.6f})"
    )
    print()

    for layer in LAYERS:
        scores: list[tuple[int, float]] = []
        for head in range(model.cfg.n_heads):
            h = head_write(cache, layer, head, last_pos, device)
            s = attribution_score(h, d_logit, scale_last)
            scores.append((head, s))

        scores.sort(key=lambda x: x[1], reverse=True)

        print(f"=== Layer {layer} (heads ranked by score, high first) ===")
        for rank, (head, sc) in enumerate(scores, start=1):
            print(f"  {rank:2d}. H{head:02d}  {sc:+.6f}")
        print()


if __name__ == "__main__":
    main()
