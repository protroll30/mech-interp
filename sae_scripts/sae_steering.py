"""Dose-response steering of one SAE decoder direction at layer-8 MLP out."""

from __future__ import annotations

import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

PROMPT = "The man is happy, he is"
RELEASE = "gpt2-small-mlp-out-v5-32k"
SAE_ID = "blocks.8.hook_mlp_out"
FEATURE_ID = 20151

COEFFICIENTS = [0.0, 10.0, 30.0, 50.0, 70.0, 100.0, 150.0]


def _ld_and_top(
    model: HookedTransformer,
    logits: torch.Tensor,
    id_sad: int,
    id_happy: int,
) -> tuple[float, str]:
    final = logits[0, -1, :]
    ld = (final[id_sad] - final[id_happy]).item()
    top_id = int(final.argmax().item())
    top_str = model.tokenizer.decode([top_id])
    return float(ld), top_str


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)

    print("Loading SAE and model...")
    sae = SAE.from_pretrained(RELEASE, SAE_ID, device=device)
    hook_name = sae.cfg.metadata.hook_name
    assert hook_name is not None, "SAE metadata missing hook_name"
    assert hook_name == SAE_ID, f"Expected hook {SAE_ID!r}, got {hook_name!r}"
    model_kw = dict(sae.cfg.metadata.model_from_pretrained_kwargs or {})

    model = HookedTransformer.from_pretrained(
        "gpt2-small",
        device=device,
        **model_kw,
    )

    steering_vec = sae.W_dec[FEATURE_ID].to(device=device)

    id_sad = model.tokenizer.encode(" sad", add_special_tokens=False)[0]
    id_happy = model.tokenizer.encode(" happy", add_special_tokens=False)[0]

    def make_steering_hook(delta: torch.Tensor):
        def steering_hook(acts: torch.Tensor, hook) -> None:
            acts[0, -1, :] += delta.to(dtype=acts.dtype, device=acts.device)

        return steering_hook

    print(f"\nPrompt: {PROMPT!r}")
    print(f"Steering feature {FEATURE_ID} at hook {hook_name!r}\n")
    print(f"{'Dose':<8} | {'LD (Sad - Happy)':<18} | {'Top prediction'}")
    print("-" * 55)

    with torch.inference_mode():
        for coeff in COEFFICIENTS:
            c = float(coeff)
            delta = c * steering_vec
            logits = model.run_with_hooks(
                PROMPT,
                fwd_hooks=[(hook_name, make_steering_hook(delta))],
            )
            ld, top = _ld_and_top(model, logits, id_sad, id_happy)
            print(f"{coeff:<8.1f} | {ld:<+18.4f} | {top!r}")


if __name__ == "__main__":
    main()
