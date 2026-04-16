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

logits, cache = model.run_with_cache(
    prompts,
    names_filter=lambda name: name.endswith("hook_pattern"),
)
# Causal LM: logits[:, t] predicts prompts[:, t + 1].
logits_first = logits[:, 0:50, :].reshape(-1, vocab_size)
targets_first = prompts[:, 1:51].reshape(-1)
loss_first = F.cross_entropy(logits_first, targets_first)

logits_second = logits[:, 50:100, :].reshape(-1, vocab_size)
targets_second = prompts[:, 51:101].reshape(-1)
loss_second = F.cross_entropy(logits_second, targets_second)

random_baseline = torch.log(torch.tensor(vocab_size, dtype=torch.float32, device=logits.device))
print(f"Loss first seq (random-ish context):  {loss_first.item():.4f}")
print(f"Loss second seq (after repetition):   {loss_second.item():.4f}")
print(f"Random CE baseline ln(vocab_size):    {random_baseline.item():.4f}")

n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
query_t = torch.arange(51, 101, device=prompts.device)
# Induction attends one past the duplicate match in the first copy (not same-index duplicate).
key_t = query_t - 49
induction_scores = torch.empty(n_layers, n_heads, device=prompts.device)
for layer in range(n_layers):
    pattern = cache[f"blocks.{layer}.attn.hook_pattern"]
    paired = pattern[:, :, query_t, key_t]
    induction_scores[layer] = paired.mean(dim=(0, 2))

flat = induction_scores.reshape(-1)
top_vals, top_idx = torch.topk(flat, k=5)
print("\nTop 5 heads by induction score (query 51-100, key t-49 = successor of duplicate match):")
top_induction_heads: list[tuple[int, int]] = []
for rank, (val, lin) in enumerate(zip(top_vals.tolist(), top_idx.tolist()), start=1):
    L = lin // n_heads
    H = lin % n_heads
    top_induction_heads.append((L, H))
    print(f"  {rank}. Layer {L}, Head {H}: {val:.6f}")
heads_by_layer: dict[int, list[int]] = {}
for layer, head in top_induction_heads:
    heads_by_layer.setdefault(layer, []).append(head)


def ablate_attn_heads(head_indices: list[int]):
    def forward_hook(activation, hook=None):
        x = activation.clone()
        x[:, :, head_indices, :] = 0.0
        return x

    return forward_hook


fwd_hooks = [
    (f"blocks.{layer}.attn.hook_result", ablate_attn_heads(heads))
    for layer, heads in sorted(heads_by_layer.items())
]
logits_ablated = model.run_with_hooks(prompts, fwd_hooks=fwd_hooks)
logits_second_abl = logits_ablated[:, 50:100, :].reshape(-1, vocab_size)
loss_second_ablated = F.cross_entropy(logits_second_abl, targets_second)
print(f"\nSecond-seq CE loss (zero-ablated top induction heads): {loss_second_ablated.item():.4f}")