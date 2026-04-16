# Evidence of a Bipartite Induction Circuit in GPT-2 Small  
### *Automated path patching on a nonsense-name copy task*

> **TL;DR:** Start from a clear induction signal, then cut the graph down: early heads in layers 0–4 whose path-patched edges hurt a logit-difference copy metric the most, pointing into a small set of thresholded late induction heads.

This is a walkthrough of one pipeline on GPT-2 small for an Argl / Flargh style nonsense prompt (repetition in context, not dictionary luck). Goal is not to map every head, but to get a subgraph you can actually argue about.

---

## Pipeline at a glance

| Stage | Question | What we did |
|------:|----------|-------------|
| **1** | Does induction show up? | String-built prompts, identical BPE spans, correlational attention score |
| **2** | Which heads? | Threshold late heads (often on the order of 18 receivers) |
| **3** | Are they causal? | Activation patching (Q/K/V), K from corrupted `resid_post` |
| **4** | How does attention split? | Pathway-specific patches |
| **5** | Which *edges*? | Path patch early `hook_result` → late `hook_k` via late **W_K** |
| **6** | Who feeds whom? | **60×*n* receivers** automated heatmap (*n* = thresholded heads) |

---

## 1. Hypothesis

**Induction** here: a rare sequence shows up once, then the prefix repeats later; the model should continue the same tail even when the continuation is nonsense (no real English prior). Some heads should link the second occurrence back to the first so copying the right tail wins.

---

## 2. Localization (correlational)

- **Tokenization:** Prompts from `model.to_tokens`, same literal substring twice so BPE does not quietly align different bytes.
- **Counterfactual:** Corrupt prompt with no shared first-half text vs clean, so K/V are not accidentally the same run.
- **Score:** Attention from queries on the second span to keys on the aligned first span (successor-style offset, not naive duplicate-token matching).
- **Head set:** Heads above a cutoff (e.g. score > 0.2), usually on the order of 18 “induction receivers.”

That stage answers: which heads line up with the pattern?

---

## 3. Causal proof (interventions)

- **Q / K / V patching:** With a real corrupt cache, all three paths can move the metric. If corrupt matched clean on the prefix, K/V patches can look dead while Q still jumps.
- **Global K stress test:** Overwrite late `hook_k` using LN-scaled corrupted `resid_post` after layer 4 (clean LN1 scale per receiver layer).

That stage answers: do those heads and pathways actually matter for the metric?

---

## 4. Mechanistic detail (Q vs K vs V)

| Pathway | Role |
|---------|------|
| **Q / K** | Whether the head matches the right positions |
| **V** | What gets written once attention fires |

So attention is not one wire; different paths do different parts of the job.

---

## 5. Circuit mapping (path patching)

**Path patching** swaps one channel:

1. Read the early head’s `hook_result` slice (per-head write).
2. Divide by the receiver block’s clean `ln1.hook_scale` (freeze LN as a linearized scale).
3. Multiply by that receiver head’s `W_K`.
4. On the receiver’s `hook_k`, replace clean vs corrupt projections along that channel:  
   **patched K** = **K** − **(clean projection)** + **(corrupt projection)**.

That is a direct test of an early sender → late receiver key edge.

---

## 6. Automation (edge heatmap)

We map which early heads send into which late induction heads with automated path patching.

### Search grid (pruned on purpose)

| Role | Scope | Count |
|------|-------|------:|
| **Senders** | Every head in layers 0–4 | **60** |
| **Receivers** | Only thresholded induction heads | ***n*** (often about 18) |
| **Full late grid** *(not used)* | All heads in layers 5–7 | 12×12 = **144** would pair with 60 for **60×144** |

We do **not** patch 60 × 144. We patch **60 × *n*** instead of 60 × 36 (three full late layers):

| Layout | Edges |
|--------|------:|
| Naive “all late layers 5–7” | 60 × 36 = **2160** |
| **This writeup / code** | 60 × *n* ≈ **60 × 18 = 1080** |

Here *n* is **`n_target`**: the number of heads that pass the induction threshold (your “about 18”).

### Cell meaning

After patching one sender–receiver edge, we record the drop in logit difference (correct token vs a plausible wrong token at the eval index):

```text
Δ logit diff  =  (clean)  −  (patched)
```

- Larger positive Δ: that edge helps the model favor the correct continuation under this metric.  
- Negative Δ: patching helped the correct answer (a brake, anti-copy, or similar under this counterfactual).

### How to read the heatmap

- **Bright horizontal bands:** A few early senders carry most of the effect on specific receiver columns. Layer 0 often has previous-token-style heads: they move information from position *t*−1 into *t*, which late induction can reuse.
- **Mid-layer stripes (layers 2–3):** Backup / cleanup routing after the first residual updates.
- **One hot column:** On a sparse nonsense prompt, one receiver (e.g. L5H5) can dominate: many heads *can* do induction, but this string may put most of the mass through one late pathway.
- **Dark (negative) spots:** Patching raises the logit diff, so that sender usually suppresses the good answer here (negative induction, anti-repetition, or similar).

Use the figure’s row index for sender layer/head (L0–L4 grid) and column labels `L*H*` for receivers.

---

## 7. Scope

**In scope** for this controlled nonsense-copy setup:

1. Localize induction-like heads.  
2. Stress them with patching / K overwrite.  
3. Sketch a bipartite graph from automated path patching.

**Out of scope:** Claiming that “three stripes” or “one dominant column” holds on all text. This is one prompt family and one metric; natural repeats will look more spread out.

---

## 8. Closing

Hypothesis, localization, causal tests, Q/K/V split, path edges, then the automated heatmap. Not the full 12×12 = 144 head graph squared, but a pruned 60×*n* edge panel that carries most of the measured effect on the Argl-Flargh style task.
