"""Necessity / circuit probes for negation-style completions on GPT-2 small.

1) SAE feature ablation at MLP8-out with sign-aware metrics (support vs suppress).
2) Linear Direct Logit Attribution (DLA) on attention heads and MLP outs, layers 4-7.
3) Path patching (clean z into corrupt forward) on layers 4-7 to find causal heads.
4) Control: feature 20151 activation on negated vs ``very`` (repetition / n-gram check).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

MODEL_NAME = "gpt2-small"
RELEASE = "gpt2-small-mlp-out-v5-32k"
SAE_ID = "blocks.8.hook_mlp_out"
FEATURE_ID = 20151

# Layers where early negation-like computation is hypothesized (before MLP8).
CIRCUIT_LAYERS = range(4, 8)
N_HEADS = 12  # gpt2-small

# Primary minimal pair for DLA + path patching (same token length).
PROMPT_NEGATED = "The man is not happy, he is"
PROMPT_CONTROL_VERYSAME = "The man is very happy, he is"


@dataclass(frozen=True)
class NegationExample:
    """Prompt and token pair for logit diff (antonym - original) at last position."""

    name: str
    prompt: str
    original_token: str
    antonym_token: str


NEGATION_DATASET: list[NegationExample] = [
    NegationExample(
        name="happy_man_not",
        prompt="The man is not happy, he is",
        original_token=" happy",
        antonym_token=" sad",
    ),
    NegationExample(
        name="good_food_not",
        prompt="The food was not good, it was",
        original_token=" good",
        antonym_token=" bad",
    ),
    NegationExample(
        name="large_box_not",
        prompt="The box is not large, it is",
        original_token=" large",
        antonym_token=" small",
    ),
]

ROBUSTNESS_DATASET: list[NegationExample] = [
    NegationExample(
        name="happy_man_never",
        prompt="The man is never happy, he is",
        original_token=" happy",
        antonym_token=" sad",
    ),
    NegationExample(
        name="good_food_failed",
        prompt="The food failed to be good, it was",
        original_token=" good",
        antonym_token=" bad",
    ),
]


def _feature_acts_dense(x: torch.Tensor) -> torch.Tensor:
    return x.to_dense() if x.is_sparse else x


def _encode_single_token_id(model: HookedTransformer, spaced_token: str) -> int:
    ids = model.tokenizer.encode(spaced_token, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(
            f"Expected a single BPE token for {spaced_token!r}, got {ids!r}"
        )
    return int(ids[0])


def logit_diff_last(
    logits: torch.Tensor, id_antonym: int, id_original: int
) -> float:
    final = logits[0, -1]
    return float((final[id_antonym] - final[id_original]).item())


def logit_diff_direction(model: HookedTransformer, id_antonym: int, id_original: int) -> torch.Tensor:
    """Direction in residual space for (logit_ant - logit_orig) if logits = resid @ W_U (linear part)."""
    w = model.W_U.to(dtype=torch.float32)
    return w[:, id_antonym] - w[:, id_original]


def describe_delta(delta: float) -> str:
    """How ablation changed LD = logit(antonym) - logit(original), relative to clean."""
    if delta < -1e-6:
        return "supports antonym-vs-original separation (ablation lowered LD)"
    if delta > 1e-6:
        return "suppresses antonym-vs-original separation (ablation raised LD)"
    return "negligible effect on LD"


def aligned_impact_fraction(delta_feat: float, delta_full: float) -> float | None:
    """
    |delta_feat| / |delta_full| only when both have the same sign and full effect is nonzero.
    Same sign means the feature and full MLP8 push LD in the same direction vs clean.
    """
    if not math.isfinite(delta_feat) or not math.isfinite(delta_full):
        return None
    if abs(delta_full) < 1e-6:
        return None
    if delta_feat * delta_full <= 0:
        return None
    return abs(delta_feat) / abs(delta_full)


def make_full_mlp_zero_hook() -> Callable[[torch.Tensor, object], torch.Tensor]:
    def hook_fn(acts: torch.Tensor, hook) -> torch.Tensor:
        return torch.zeros_like(acts)

    return hook_fn


def make_sae_feature_zero_hook(
    sae: SAE, feature_idx: int
) -> Callable[[torch.Tensor, object], torch.Tensor]:
    def hook_fn(acts: torch.Tensor, hook) -> torch.Tensor:
        feats = _feature_acts_dense(sae.encode(acts))
        feats = feats.clone()
        feats[..., feature_idx] = 0.0
        out = sae.decode(feats)
        return out.to(dtype=acts.dtype, device=acts.device)

    return hook_fn


def run_conditions(
    model: HookedTransformer,
    hook_name: str,
    sae: SAE,
    example: NegationExample,
    id_original: int,
    id_antonym: int,
) -> dict[str, float]:
    prepend_bos = model.cfg.default_prepend_bos
    toks = model.to_tokens(example.prompt, prepend_bos=prepend_bos)
    out: dict[str, float] = {}
    with torch.inference_mode():
        with model.hooks(fwd_hooks=[]):
            logits_clean = model(toks)
        out["clean"] = logit_diff_last(logits_clean, id_antonym, id_original)

        full_hook = make_full_mlp_zero_hook()
        with model.hooks(fwd_hooks=[(hook_name, full_hook)]):
            logits_full = model(toks)
        out["full_mlp"] = logit_diff_last(logits_full, id_antonym, id_original)

        feat_hook = make_sae_feature_zero_hook(sae, FEATURE_ID)
        with model.hooks(fwd_hooks=[(hook_name, feat_hook)]):
            logits_feat = model(toks)
        out["feature"] = logit_diff_last(logits_feat, id_antonym, id_original)
    return out


def print_ablation_block(
    title: str,
    examples: list[NegationExample],
    model: HookedTransformer,
    hook_name: str,
    sae: SAE,
) -> None:
    print(f"\n{'=' * 72}")
    print(title)
    print(f"{'=' * 72}")
    print(
        "Readout: LD = logit(antonym) - logit(original) at the last token. "
        "Higher LD = stronger preference for the antonym."
    )
    print(
        "Ablation delta = LD_ablated - LD_clean. Negative => component was helping "
        "the antonym edge; positive => it was pulling LD down (MLP8 can be net suppressive)."
    )

    for ex in examples:
        id_orig = _encode_single_token_id(model, ex.original_token)
        id_ant = _encode_single_token_id(model, ex.antonym_token)
        m = run_conditions(model, hook_name, sae, ex, id_orig, id_ant)
        ld_c = m["clean"]
        ld_full = m["full_mlp"]
        ld_feat = m["feature"]
        d_full = ld_full - ld_c
        d_feat = ld_feat - ld_c
        frac = aligned_impact_fraction(d_feat, d_full)

        print(f"\n[{ex.name}] {ex.prompt!r}")
        print(
            f"  Tokens: antonym={ex.antonym_token!r} (id={id_ant}), "
            f"original={ex.original_token!r} (id={id_orig})"
        )
        print("  LD (antonym - original):")
        print(f"    Clean:              {ld_c:+.4f}")
        print(f"    Full MLP8 zero:     {ld_full:+.4f}")
        print(f"    Feature {FEATURE_ID} zero: {ld_feat:+.4f}")
        print("  Ablation delta (= LD_ablated - LD_clean):")
        print(f"    Full MLP8:     {d_full:+.4f}  ({describe_delta(d_full)})")
        print(f"    Feature only:  {d_feat:+.4f}  ({describe_delta(d_feat)})")
        if frac is not None:
            print(
                f"  |Feature delta| / |Full MLP8 delta| (same-sign only): {frac:.3f}"
            )
        else:
            print(
                "  |Feature delta| / |Full MLP8 delta|: n/a (opposite signs or tiny full delta)"
            )


def last_token_feature_activation(
    model: HookedTransformer,
    sae: SAE,
    hook_name: str,
    prompt: str,
    feature_idx: int,
) -> float:
    prepend_bos = model.cfg.default_prepend_bos
    toks = model.to_tokens(prompt, prepend_bos=prepend_bos)
    with torch.inference_mode():
        _, cache = model.run_with_cache(toks, names_filter=lambda n: n == hook_name)
        mlp_out = cache[hook_name]
        acts = _feature_acts_dense(sae.encode(mlp_out))
    return float(acts[0, -1, feature_idx].item())


def print_feature_repetition_probe(
    model: HookedTransformer, sae: SAE, hook_name: str
) -> None:
    print(f"\n{'=' * 72}")
    print('Feature activation probe: "not" vs "very" (shallow repetition / n-gram check)')
    print(f"{'=' * 72}")
    a_not = last_token_feature_activation(
        model, sae, hook_name, PROMPT_NEGATED, FEATURE_ID
    )
    a_very = last_token_feature_activation(
        model, sae, hook_name, PROMPT_CONTROL_VERYSAME, FEATURE_ID
    )
    print(f"Prompt A (negated): {PROMPT_NEGATED!r}")
    print(f"Prompt B (control): {PROMPT_CONTROL_VERYSAME!r}")
    print(f"Feature {FEATURE_ID} activation at last token: A={a_not:.6f}, B={a_very:.6f}")
    ratio = a_very / a_not if abs(a_not) > 1e-8 else float("nan")
    if math.isfinite(ratio):
        print(f"Ratio B/A: {ratio:.3f} (near 1 => similar drive; supports shallow / repetition hypothesis)")
    print(
        "Interpretation: strong activation on both minimal pairs is evidence the feature "
        "tracks local 'X happy, he is' structure, not negation specifically."
    )


def dla_layers_4_to_7(
    model: HookedTransformer,
    cache: dict,
    w_diff: torch.Tensor,
) -> tuple[
    list[tuple[int, int, float]],
    list[tuple[int, float]],
    list[tuple[int, float]],
]:
    """
    Linear DLA: dot(head_output, w_diff) with head_output = z_h @ W_O[h].
    Returns sorted per-head scores, per-layer attn totals, per-layer MLP totals.
    """
    w_diff = w_diff.to(dtype=torch.float32)
    per_head: list[tuple[int, int, float]] = []
    per_layer_attn: list[tuple[int, float]] = []
    per_layer_mlp: list[tuple[int, float]] = []

    for L in CIRCUIT_LAYERS:
        z = cache[f"blocks.{L}.attn.hook_z"][0, -1].float()
        w_o = model.blocks[L].attn.W_O.float()
        for h in range(N_HEADS):
            head_vec = z[h] @ w_o[h]
            per_head.append((L, h, float(head_vec @ w_diff)))

        attn_out = cache[f"blocks.{L}.hook_attn_out"][0, -1].float()
        per_layer_attn.append((L, float(attn_out @ w_diff)))

        mlp_out = cache[f"blocks.{L}.hook_mlp_out"][0, -1].float()
        per_layer_mlp.append((L, float(mlp_out @ w_diff)))

    per_head.sort(key=lambda t: abs(t[2]), reverse=True)
    return per_head, per_layer_attn, per_layer_mlp


def print_dla_block(model: HookedTransformer, id_orig: int, id_ant: int) -> None:
    print(f"\n{'=' * 72}")
    print("Direct Logit Attribution (linear): layers 4-7, last token")
    print(f"{'=' * 72}")
    print(
        "Uses w_diff = W_U[:, antonym] - W_U[:, original] (ignores ln_final). "
        "Head term: dot((z_h @ W_O[h]), w_diff). Positive => pushes toward antonym vs original."
    )
    prepend_bos = model.cfg.default_prepend_bos
    toks = model.to_tokens(PROMPT_NEGATED, prepend_bos=prepend_bos)
    w_diff = logit_diff_direction(model, id_ant, id_orig)

    names_filter = [f"blocks.{L}.attn.hook_z" for L in CIRCUIT_LAYERS]
    names_filter += [f"blocks.{L}.hook_attn_out" for L in CIRCUIT_LAYERS]
    names_filter += [f"blocks.{L}.hook_mlp_out" for L in CIRCUIT_LAYERS]

    with torch.inference_mode():
        _, cache = model.run_with_cache(toks, names_filter=names_filter)

    per_head, layer_attn_pairs, layer_mlp_list = dla_layers_4_to_7(model, cache, w_diff)

    print("\nTop attention heads by |DLA| (layers 4-7):")
    for L, h, s in per_head[:15]:
        print(f"  L{L}H{h}:  {s:+.4f}")

    print("\nLayer totals (attention hook_attn_out @ w_diff):")
    for L, s in layer_attn_pairs:
        print(f"  Layer {L}: {s:+.4f}")

    print("\nLayer totals (MLP hook_mlp_out @ w_diff):")
    for L, s in layer_mlp_list:
        print(f"  Layer {L}: {s:+.4f}")


def make_z_patch_hook(
    clean_z: torch.Tensor, head: int
) -> Callable[[torch.Tensor, object], torch.Tensor]:
    """Replace one head's z with clean run values (activation patching / path patch on z)."""

    def hook_fn(z: torch.Tensor, hook) -> torch.Tensor:
        out = z.clone()
        out[:, :, head, :] = clean_z[:, :, head, :].to(
            device=out.device, dtype=out.dtype
        )
        return out

    return hook_fn


def path_patching_grid(
    model: HookedTransformer,
    clean_prompt: str,
    corrupt_prompt: str,
    id_orig: int,
    id_ant: int,
) -> list[tuple[int, int, float, float]]:
    """
    Clean = negated, corrupt = 'very' (same length). For each (L,h), run corrupt forward
    with hook_z for that head taken from clean. Returns list of (L, h, LD_patched, effect)
    where effect = LD_patched - LD_corrupt (movement toward clean/negated behavior if positive).
    """
    prepend_bos = model.cfg.default_prepend_bos
    clean_toks = model.to_tokens(clean_prompt, prepend_bos=prepend_bos)
    corrupt_toks = model.to_tokens(corrupt_prompt, prepend_bos=prepend_bos)
    if clean_toks.shape != corrupt_toks.shape:
        raise ValueError("Clean and corrupt prompts must tokenize to the same shape.")

    z_keys = [f"blocks.{L}.attn.hook_z" for L in CIRCUIT_LAYERS]
    with torch.inference_mode():
        _, clean_cache = model.run_with_cache(clean_toks, names_filter=z_keys)
        with model.hooks(fwd_hooks=[]):
            logits_corrupt = model(corrupt_toks)
        ld_corrupt = logit_diff_last(logits_corrupt, id_ant, id_orig)

    results: list[tuple[int, int, float, float]] = []
    with torch.inference_mode():
        for L in CIRCUIT_LAYERS:
            clean_z = clean_cache[f"blocks.{L}.attn.hook_z"]
            hook_z_name = f"blocks.{L}.attn.hook_z"
            for h in range(N_HEADS):
                hook_fn = make_z_patch_hook(clean_z, h)
                with model.hooks(fwd_hooks=[(hook_z_name, hook_fn)]):
                    logits_p = model(corrupt_toks)
                ld_p = logit_diff_last(logits_p, id_ant, id_orig)
                effect = ld_p - ld_corrupt
                results.append((L, h, ld_p, effect))

    return results


def print_path_patching_block(model: HookedTransformer, id_orig: int, id_ant: int) -> None:
    print(f"\n{'=' * 72}")
    print("Path patching (layers 4-7): clean z -> corrupt forward")
    print(f"{'=' * 72}")
    print(f"Clean (negated):  {PROMPT_NEGATED!r}")
    print(f"Corrupt (control): {PROMPT_CONTROL_VERYSAME!r}")
    print(
        "Effect = LD_patched - LD_corrupt. Positive => patching that head's z from the "
        "negated run increases antonym-vs-original separation on the corrupt prompt "
        "(moves logit diff in the same direction as introducing negation signal)."
    )

    with torch.inference_mode():
        prepend_bos = model.cfg.default_prepend_bos
        toks_c = model.to_tokens(PROMPT_NEGATED, prepend_bos=prepend_bos)
        toks_v = model.to_tokens(PROMPT_CONTROL_VERYSAME, prepend_bos=prepend_bos)
        ld_clean = logit_diff_last(model(toks_c), id_ant, id_orig)
        ld_corrupt = logit_diff_last(model(toks_v), id_ant, id_orig)

    print(f"LD clean (negated):   {ld_clean:+.4f}")
    print(f"LD corrupt ('very'): {ld_corrupt:+.4f}")

    grid = path_patching_grid(
        model, PROMPT_NEGATED, PROMPT_CONTROL_VERYSAME, id_orig, id_ant
    )
    grid_sorted = sorted(grid, key=lambda t: abs(t[3]), reverse=True)

    print("\nTop heads by |effect| (patch clean z into corrupt run):")
    for L, h, ld_p, effect in grid_sorted[:18]:
        print(f"  L{L}H{h}:  LD_patched={ld_p:+.4f}  effect={effect:+.4f}")


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
        MODEL_NAME,
        device=device,
        **model_kw,
    )

    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME}, SAE: {RELEASE!r}, hook: {hook_name!r}")
    print(f"Feature ID: {FEATURE_ID}")

    print_ablation_block(
        'SAE ablation metrics ("not" prompts)', NEGATION_DATASET, model, hook_name, sae
    )
    print_ablation_block(
        'Robustness (no "not")', ROBUSTNESS_DATASET, model, hook_name, sae
    )

    print_feature_repetition_probe(model, sae, hook_name)

    id_orig = _encode_single_token_id(model, " happy")
    id_ant = _encode_single_token_id(model, " sad")
    print_dla_block(model, id_orig, id_ant)
    print_path_patching_block(model, id_orig, id_ant)

    print("\nNotes:")
    print(
        "  - MLP8 full ablation can increase LD; interpret with signed deltas, not "
        "'percent drop'."
    )
    print(
        "  - DLA omits ln_final; use patching for causal claims about heads 4-7."
    )


if __name__ == "__main__":
    main()
