# Negation-style completion on GPT-2 small

Small experiment on one prompt: after *The man is not happy, he is*, does the model lean toward continuing with *sad* or *happy*? Then a few cheap linear probes (logit lens, head dots, attn vs MLP writes) to see where the preference shows up. All of this is one run on one machine; treat it as a lab note, not a paper.

---

## Prompts and metric

| Label | Text |
|-------|------|
| Clean | `The man is not happy, he is` |
| Baseline (comparison only) | `The man is very happy` |

**LD** at the **last input token** (index **8** with default BOS on this prompt):

`LD = logit(first token of " sad") - logit(first token of " happy")`

| Fragment | Token id (this run) |
|----------|--------------------:|
| `" sad"` | 6507 |
| `" happy"` | 3772 |

Positive LD means *sad* is scored above *happy* for that readout.

---

## Scripts

| File | What it does |
|------|----------------|
| `negation_baseline.py` | Full-model LD on clean and baseline; **logit lens** on `blocks.L.hook_resid_post` for every L, plot vs layer. |
| `negation_head_attribution.py` | For layers 5, 6, 7: each head’s `hook_result` write at the last token, dotted onto the sad-minus-happy direction (with `ln_final.hook_scale` fold). Ranks heads. |
| `negation_mlp_attribution.py` | Per layer: same dot product for **`hook_attn_out`** and **`hook_mlp_out`** at the last token; line plot attn vs MLP. |

The dot-product recipes are spelled in the script docstrings. They are **linear** approximations around LayerNormPre at the final block, so they rank directions and layers; they do not sum to the final logit difference across layers.

---

## Numbers from this run

**Full model**

| Prompt | LD |
|--------|---:|
| Clean | +0.9682 |
| Baseline | −1.9236 |

So the negation framing flips the sign of the same contrast vs the upbeat baseline.

**Logit lens** (`unembed(ln_final(resid_post[L]))` at the last position)

- First layer where lens LD goes positive (sad ahead in that readout): **L2**.
- Lens at **L11** matched the full model LD (**+0.9682**), which is a nice sanity check.

Figure: `negation_logit_lens_ld.png`.

**Head attribution** (layers 5–7, best head first)

Rough story: **L6** is carried by **H07**, **H08**, **H11** (all positive scores on this axis). **L5** peaks at **H09**. **L7** peaks at **H05**. Full tables are in the terminal log when you rerun `negation_head_attribution.py`.

**Attention vs MLP** (whole-block writes at the last token)

| Layer | attn `hook_attn_out` | mlp `hook_mlp_out` |
|:-----:|---------------------:|-------------------:|
| 0 | +0.0300 | −0.0227 |
| 1 | +0.0687 | −0.0318 |
| 2 | +0.0769 | −0.0084 |
| 3 | +0.0562 | −0.0176 |
| 4 | +0.0271 | −0.0669 |
| 5 | +0.1041 | −0.1381 |
| 6 | +0.3152 | +0.0795 |
| 7 | +0.1928 | −0.1575 |
| 8 | +0.1826 | −0.2344 |
| 9 | +0.6067 | −0.0463 |
| 10 | +0.1661 | +0.2523 |
| 11 | +0.0500 | +0.0861 |

Figure: `negation_attn_vs_mlp_attribution.png`.

On this prompt, **attention** scores stay **positive** at every layer in that table, while **MLP** is **negative** through much of the middle and only turns clearly **positive** again at **L10–L11**. The biggest attention spike is **L9**, with **L6** also large. So if you are asking “heads or FFN for the readout we plotted?”, the blunt answer from this slice is **mostly attention-shaped**, with MLP acting more like it is pulling the other way in several mid layers (under this one linear map).

---

## How to read the “flip”

The logit lens crosses zero at **L2**: that only means “if you stopped after block 2 and ran `ln_final` + unembed, sad would already be ahead.” It is not the same as “the model finished negation reasoning there.” The attn vs MLP plot is layerwise **writes**, not a cumulative residual curve, so do not try to line it up one-to-one with the logit lens x-axis as if they were the same object.

---

## Reproduce

```bash
python negation_scripts/negation_baseline.py
python negation_scripts/negation_head_attribution.py
python negation_scripts/negation_mlp_attribution.py
```

---

## Limits

One template, `gpt2-small`, first subword only for *sad* / *happy*. Change the string or tokenizer settings and the ids and curves will move.
