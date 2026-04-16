# IOI Circuit Discovery on GPT-2 Small (Script Suite)

> **TL;DR (this run):** Short John/Mary template: clean favors Mary (LD +3.48), corrupt favors John (LD −4.37). Strongest name movers by recovery are L9H9, L8H10, L7H9 (about 14–16% each). S2 is token 10. Patching clean `hook_resid_mid` at S2 from layers 0–6 into corrupt brings back Mary; layers 7–10 make things worse vs that shallow patch; L11 is a no-op. S-inhibition-style Q edits at S2 peak near 2% recovery, so under this setup most of the signal is not going through that narrow Q path.

---

## Prompt pair (shared)

| Role | Text |
|------|------|
| **Clean** | `When John and Mary went to the store, John gave a bottle of milk to` |
| **Corrupt** | `When John and Mary went to the store, Mary gave a bottle of milk to` |

**Tokenization (this run):** `seq_len = 17`, `prepend_bos = True`, evaluation index `last = 16`.

**Metric:** at the last position,

**LD = logit(Mary first token) − logit(John first token)**

| Token | ID (this run) |
|-------|----------------:|
| First subword of `" Mary"` | 5335 |
| First subword of `" John"` | 1757 |

- Higher LD → favors Mary as the indirect object.  
- Negative LD → favors John (matches corrupt story).

---

## Scripts at a glance

| File | What it does |
|------|----------------|
| `ioi_baseline.py` | Token-length check; clean LD only. |
| `ioi_name_movers.py` | Layers 7–11, each head: patch clean `hook_result` into corrupt; % recovery. |
| `ioi_s_inhibition_search.py` | S2 = second `"John"` slot; inject sender clean `hook_result` @ S2 into Q of (9,9), (8,10), (7,9) via v @ W_Q. |
| `ioi_resid_sweep.py` | Layers 0–11: replace `hook_resid_mid[:, S2, :]` alone with clean on corrupt. |
| `ioi_attention_viz.py` | Clean run; `blocks.9.attn.hook_pattern`, head 9 → `ioi_clean_L9H9_attention.png`. |

---

## Results summary (this machine, one run)

| Quantity | Value |
|----------|------:|
| Clean LD | +3.4811 |
| Corrupt LD | −4.3722 |
| Denominator (clean − corrupt) | +7.8534 |
| S2 token index | 10 |
| Top name movers (recovery) | L9H9 15.6%, L8H10 15.5%, L7H9 13.9% |
| Best S-inhibition sender | L0H11 2.0% (ΔLD +0.1571) |
| Earliest `resid_mid` @ S2 with LD > 0 | L0 (patched LD +3.47) |
| Any single layer reaches clean LD? | No (best shallow layers +3.47, short of +3.48) |
| Attention artifact | `ioi_scripts/ioi_clean_L9H9_attention.png`, pattern shape (1, 12, 17, 17) |

---

## 1. Baseline (`ioi_baseline.py`)

Clean LD +3.48 means the model prefers Mary over John at the completion after the clean template.

---

## 2. Name movers (`ioi_name_movers.py`)

Recovery: `(Patched_LD − Corrupt_LD) / (Clean_LD − Corrupt_LD)`.

**Top 8 (this run):**

| Rank | Head | Recovery | % |
|:----:|------|-----------:|:-:|
| 1 | L9H9 | 0.1564 | 15.6% |
| 2 | L8H10 | 0.1550 | 15.5% |
| 3 | L7H9 | 0.1387 | 13.9% |
| 4 | L8H6 | 0.1289 | 12.9% |
| 5 | L7H3 | 0.1197 | 12.0% |
| 6 | L10H6 | 0.0755 | 7.5% |
| 7 | L9H7 | 0.0526 | 5.3% |
| 8 | L10H0 | 0.0417 | 4.2% |

The top three match the fixed receiver triple in `ioi_s_inhibition_search.py` (9,9), (8,10), (7,9).

The printed 5×12 grid has a few negative cells (e.g. L10H7 −0.35, L11H10 −0.18): patching that head’s clean write hurts recovery on corrupt (anti-name-mover or context clash under this metric).

---

## 3. S-inhibition search (`ioi_s_inhibition_search.py`)

S2 = 10. Same clean / corrupt / denom as above.

Best sender is L0H11 at about 2.0% recovery (ΔLD +0.16). The rest of the top 12 sit near 1.1–1.6%. With this construction (project sender `hook_result` @ S2 into name-mover Q via v @ W_Q), the effect is small and spread out, unlike the about 15% you get from transplanting a single head’s `hook_result`. Useful negative result: most of the IOI signal is not flowing through that narrow Q rewrite, at least not with this linear map.

---

## 4. Residual mid sweep (`ioi_resid_sweep.py`)

Patch only `blocks.{L}.hook_resid_mid[:, 10, :]` from clean into corrupt.

| Layer | Patched LD | Δ vs corrupt |
|:-----:|-----------:|-------------:|
| 0 | +3.4734 | +7.8456 |
| 1 | +3.4681 | +7.8403 |
| 2 | +3.4593 | +7.8316 |
| 3 | +3.3063 | +7.6786 |
| 4 | +2.7403 | +7.1126 |
| 5 | +2.7794 | +7.1516 |
| 6 | +2.4114 | +6.7836 |
| 7 | −0.2581 | +4.1141 |
| 8 | −3.5762 | +0.7961 |
| 9 | −3.9952 | +0.3770 |
| 10 | −3.8920 | +0.4802 |
| 11 | −4.3722 | +0.0000 |

**Readout:**

- L0 is the first layer where patched LD > 0 (Mary over John). Shallow clean S2 mid-residual carries almost the full clean margin (3.47 vs 3.48).
- L7–10 overshoot back toward John vs the best shallow patches (LD goes negative again).
- L11 matches unpatched corrupt (Δ 0): replacing only S2 mid-residual at the last block does nothing. Either that slice no longer steers the logit, or the hook already matches corrupt at that site.

---

## 5. Attention visualization (`ioi_attention_viz.py`)

Saved `ioi_scripts/ioi_clean_L9H9_attention.png`. Tensor shape (batch=1, heads=12, pos=17, pos=17).

Check high-attention keys for query row 16 (final position) and query row 10 (S2) against token labels on the axes; see whether L9H9 attends to Mary, S1, or other IOI-relevant positions.

---

## 6. Reproducing these numbers

```bash
python ioi_scripts/ioi_baseline.py
python ioi_scripts/ioi_name_movers.py
python ioi_scripts/ioi_s_inhibition_search.py
python ioi_scripts/ioi_resid_sweep.py
python ioi_scripts/ioi_attention_viz.py
```

---

## 7. Interpretation guardrails

- Short template on GPT-2 small; numbers are diagnostic, not a full IOI benchmark.  
- LD uses first subwords only (ids 5335 / 1757 here).  
- `ioi_s_inhibition_search.py` receivers are fixed to the top name movers from this grid; change prompts, re-check alignment.

---

## 8. Future work

**Scaling to longer contexts.** Here the names are a few tokens apart and the story fits in 17 positions. The `hook_resid_mid` @ S2 sweep suggests very early layers already encode something close to the clean IOI margin at the second subject. Stress test: does L0–L6 dominance hold when there is enough filler that local bigram / positional tricks cannot fake coreference (e.g. on the order of 50 tokens between the two names, or a controlled filler sweep)? If shallow patches stop recovering clean LD while late name movers stay necessary, that splits “IOI-like local cues” from “genuine long-range IOI” on small models.

**Path patching the MLPs.** So far we only patch attention outputs and one residual slice. Late heads (L7–L9) plausibly add inhibition / redistribution on top of whatever identity signal early blocks write. Path patching (or neuron-level cuts) on MLP post, especially which units at which layers turn an L0-ish representation at S2 into the L9-era signal that supports Mary over John, would make that story more mechanistic than architectural. Even a short “top-k MLP features by indirect effect on LD” pass would shrink the search space.

---

## 9. Closing

Strong clean IOI margin, strong corrupt flip, three late heads each recovering about 14–16% via `hook_result`, shallow S2 mid-residual almost fully restoring the clean margin, deep S2 mid-residual fighting that fix, and a narrow Q-injection path that barely moves the needle. That is a tight loop for circuit discovery, plus one concrete figure (`ioi_clean_L9H9_attention.png`) for the headline name-mover head.
