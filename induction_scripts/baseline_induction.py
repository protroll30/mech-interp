import functools

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2-small")
model.set_use_attn_result(True)

N = 50
batch_size = 8

vocab_size = model.cfg.d_vocab
seq = torch.randint(0, vocab_size, (batch_size, N), dtype=torch.long, device=model.W_E.device)

bos_token_id = model.tokenizer.bos_token_id
if bos_token_id is None:
    bos_token_id = model.tokenizer.eos_token_id

bos = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=seq.device)
prompts = torch.cat([bos, seq, seq], dim=-1)

n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads

logits, cache = model.run_with_cache(prompts)

logits_first = logits[:, 0:50, :].reshape(-1, vocab_size)
targets_first = prompts[:, 1:51].reshape(-1)
loss_first = F.cross_entropy(logits_first, targets_first)

logits_second = logits[:, 50:100, :].reshape(-1, vocab_size)
targets_second = prompts[:, 51:101].reshape(-1)
loss_second = F.cross_entropy(logits_second, targets_second)


def second_seq_ce(logits_tensor: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits_tensor[:, 50:100, :].reshape(-1, vocab_size),
        targets_second,
    )


random_baseline = torch.log(torch.tensor(vocab_size, dtype=torch.float32, device=logits.device))
print(f"Loss first seq (random-ish context):  {loss_first.item():.4f}")
print(f"Loss second seq (after repetition):   {loss_second.item():.4f}")
print(f"Random CE baseline ln(vocab_size):    {random_baseline.item():.4f}")

query_t = torch.arange(51, 101, device=prompts.device)
key_t = query_t - 49
induction_scores = torch.empty(n_layers, n_heads, device=prompts.device)
for layer in range(n_layers):
    pattern = cache[f"blocks.{layer}.attn.hook_pattern"]
    paired = pattern[:, :, query_t, key_t]
    induction_scores[layer] = paired.mean(dim=(0, 2))

target_mask = induction_scores > 0.20
target_pairs = torch.nonzero(target_mask, as_tuple=False)
n_target = target_pairs.shape[0]
print(f"\nInduction heads with score > 0.20: {n_target}")

seq_b = torch.randint(0, vocab_size, (batch_size, N), dtype=torch.long, device=model.W_E.device)
seq_c = torch.randint(0, vocab_size, (batch_size, N), dtype=torch.long, device=model.W_E.device)
prompts_corrupted = torch.cat([bos, seq_b, seq_c], dim=-1)
_, corrupted_cache = model.run_with_cache(prompts_corrupted)

resid_post_4_corrupt = corrupted_cache["blocks.4.hook_resid_post"]


def make_overwrite_k_from_resid4_hook(
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
            W_K = model.blocks[layer].attn.W_K[H]
            w = W_K.to(device=x.device, dtype=x.dtype)
            x[:, :, H, :] = z @ w
        return x

    return hook_fn


k_from_resid4_fwd_hooks = []
for layer in range(n_layers):
    in_layer = target_pairs[:, 0] == layer
    head_indices = target_pairs[in_layer, 1].tolist()
    if not head_indices:
        continue
    k_from_resid4_fwd_hooks.append(
        (
            f"blocks.{layer}.attn.hook_k",
            make_overwrite_k_from_resid4_hook(layer, head_indices, cache, resid_post_4_corrupt),
        )
    )
logits_k_resid4 = model.run_with_hooks(prompts, fwd_hooks=k_from_resid4_fwd_hooks)
print(
    f"\nSecond-seq CE (hook_k from LN-scaled corrupt resid_post@L4, {n_target} heads): "
    f"{second_seq_ce(logits_k_resid4).item():.4f}"
)

h_early_clean = cache["blocks.0.attn.hook_result"][:, :, 5, :]
h_early_corrupt = corrupted_cache["blocks.0.attn.hook_result"][:, :, 5, :]
ln1_scale_clean = cache["blocks.5.ln1.hook_scale"]
h_early_clean = h_early_clean / ln1_scale_clean
h_early_corrupt = h_early_corrupt / ln1_scale_clean
W_K_late_h5 = model.blocks[5].attn.W_K[5]
clean_projected = h_early_clean @ W_K_late_h5
corrupt_projected = h_early_corrupt @ W_K_late_h5


def path_patch_l5h5_k_from_early(activation, clean_proj, corrupt_proj, hook=None):
    x = activation.clone()
    cp = clean_proj.to(device=x.device, dtype=x.dtype)
    crp = corrupt_proj.to(device=x.device, dtype=x.dtype)
    x[:, :, 5, :] = x[:, :, 5, :] - cp + crp
    return x


path_patch_k_hook = functools.partial(
    path_patch_l5h5_k_from_early,
    clean_proj=clean_projected,
    corrupt_proj=corrupt_projected,
)
logits_path_k = model.run_with_hooks(
    prompts,
    fwd_hooks=[("blocks.5.attn.hook_k", path_patch_k_hook)],
)
print(f"\nSecond-seq CE (path patch L0H5 -> L5H5 K): {second_seq_ce(logits_path_k).item():.4f}")


def patch_heads_from_corrupted(activation, head_indices: list[int], corrupted_cache, hook=None):
    if not head_indices:
        return activation
    x = activation.clone()
    corr = corrupted_cache[hook.name]
    corr_slice = corr[:, :, head_indices, :].to(device=x.device, dtype=x.dtype)
    x[:, :, head_indices, :] = corr_slice
    return x


def build_patching_fwd_hooks(corrupted_cache, qkv_hook: str) -> list:
    fwd_hooks = []
    for layer in range(n_layers):
        in_layer = target_pairs[:, 0] == layer
        head_indices = target_pairs[in_layer, 1].tolist()
        if not head_indices:
            continue
        hook_fn = functools.partial(
            patch_heads_from_corrupted,
            head_indices=head_indices,
            corrupted_cache=corrupted_cache,
        )
        fwd_hooks.append((f"blocks.{layer}.attn.{qkv_hook}", hook_fn))
    return fwd_hooks


print(f"\nSecond-seq CE (clean): {loss_second.item():.4f}")

for pathway_name, hook_suffix in (
    ("Value (hook_v)", "hook_v"),
    ("Key (hook_k)", "hook_k"),
    ("Query (hook_q)", "hook_q"),
):
    hooks = build_patching_fwd_hooks(corrupted_cache, hook_suffix)
    logits_p = model.run_with_hooks(prompts, fwd_hooks=hooks)
    ce = second_seq_ce(logits_p).item()
    print(f"Second-seq CE patch {pathway_name} only on {n_target} heads: {ce:.4f}")
