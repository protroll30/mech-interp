# Evidence of a Bipartite Induction Circuit in GPT-2 Small  
### *Automated path patching on a nonsense-name copy task*

> **TL;DR** — We go from “induction exists” to a **pruned sender→receiver graph**: early heads (layers 0–4) whose path-patched edges most hurt a **logit-difference** copy metric, feeding a small set of **thresholded late induction heads**.

This note summarizes a mechanistic interpretability pipeline on **GPT-2 small** for an **Argl / Flargh**-style **nonsense-name** prompt: the model must rely on **repetition in context**, not lexical priors. The aim is to shrink a **124M-parameter** model, for this task, into a **small causal subgraph** you can actually test.

---

## Pipeline at a glance

| Stage | Question | What we did |
|------:|----------|-------------|
| **1** | Does induction show up? | String-built prompts, identical BPE spans, correlational attention score |
| **2** | Which heads? | Threshold late heads (~18 receivers) |
| **3** | Are they causal? | Activation patching (Q/K/V), K from corrupted `resid_post` |
| **4** | How does attention split? | Pathway-specific patches |
| **5** | Which *edges*? | Path patch early `hook_result` → late `hook_k` via late **W_K** |
| **6** | Who feeds whom? | **60×*n* receivers** automated heatmap (*n* = thresholded heads) |

---

## 1. Hypothesis

**Induction** means: after a rare sequence appears once, the model can continue it when a **prefix repeats** later—even when **there is no plausible English continuation** (nonsense names). We expect **some attention heads** to link the **second occurrence** back to the **first** and support copying the right continuation.

---

## 2. Localization (correlational)

- **Tokenization** — Prompts from **`model.to_tokens`**, with the **same literal substring** twice so BPE does not fake you out.
- **Counterfactual** — A corrupt prompt with **no shared first-half text** vs clean, so K/V caches are a real counterfactual (not accidentally identical).
- **Score** — Attention from queries on the **second** span to keys on the **aligned first** span (successor-style offset, not naive duplicate-token matching).
- **Head set** — Keep heads above a cutoff (e.g. score **> 0.2**), typically **~18** “induction receivers.”

*Answers:* **which heads correlate with the induction pattern?**

---

## 3. Causal proof (interventions)

- **Q / K / V patching** — With a proper corrupt cache, **all three** pathways can show an effect; if the corrupt prefix matched clean, K/V patches can look dead while Q still spikes.
- **Global K stress test** — Overwrite late **`hook_k`** using **LN-scaled** corrupted **`resid_post`** after layer 4 (clean **LN1 scale** per receiver layer).

*Answers:* **do those heads and pathways actually matter for the metric?**

---

## 4. Mechanistic detail (Q vs K vs V)

| Pathway | Role |
|---------|------|
| **Q / K** | Whether the head **matches** the right positions |
| **V** | **What** gets written once attention fires |

So attention is not a single wire: different paths carry different parts of the job.

---

## 5. Circuit mapping (path patching)

**Path patching** swaps a *specific* channel:

1. Read the early head’s **`hook_result`** slice (per-head write).
2. Divide by the receiver block’s **clean** **`ln1.hook_scale`** (freeze LN as linearized scale).
3. Multiply by that receiver head’s **`W_K`**.
4. On the receiver’s **`hook_k`**, replace **clean vs corrupt** projections along that channel:  
   **patched K** = **K** − **(clean projection)** + **(corrupt projection)**.

That is a direct test of an **early sender → late receiver key** edge.

---

## 6. Automation (edge heatmap)

We map **which early heads send into which late induction heads** with **automated path patching**.

### Search grid (pruned on purpose)

| Role | Scope | Count |
|------|-------|------:|
| **Senders** | Every head in **layers 0–4** | **60** |
| **Receivers** | Only **thresholded** induction heads | ***n*** (often **~18**) |
| **Full late grid** *(not used)* | All heads in layers 5–7 | 12×12 = **144** would pair with 60 for **60×144** |

We **do not** patch **60 × 144**. We patch **60 × *n*** instead of **60 × 36** (three full late layers):

| Layout | Edges |
|--------|------:|
| Naive “all late layers 5–7” | 60 × 36 = **2160** |
| **This writeup / code** | 60 × *n* ≈ **60 × 18 = 1080** |

Here *n* is **`n_target`**: the number of heads that pass the induction threshold (your “~18”).

### Cell meaning

After patching **one** sender–receiver edge, we record the **drop in logit difference** (correct token vs a plausible wrong token at the evaluation index):

```text
Δ logit diff  =  (clean)  −  (patched)
```

- **Larger positive Δ** → that edge **helps** the model favor the correct continuation under this metric.  
- **Negative Δ** → patching **helped** the correct answer (a “brake” or anti-copy head under this counterfactual).

### How to read the heatmap

- **Bright horizontal bands** — A few **early senders** drive most of the effect on **specific receiver columns**. Layer **0** often shows **previous-token–style** heads: they move information from **position *t*−1** into position ***t***, which late induction can reuse.
- **Mid-layer stripes (layers 2–3)** — **Backup / cleanup** routing after the first residual updates.
- **One hot column** — On a **sparse** nonsense prompt, **one receiver** (e.g. **L5H5**) can dominate: many heads *can* do induction, but **this string** may commit most of the work to **one late pathway**.
- **Dark (negative) spots** — Patching **raises** the logit diff → that sender usually **suppresses** the good answer here (negative induction, anti-repetition, or similar).

Use the figure’s **row** index for **sender layer/head** (L0–L4 grid) and **column labels** `L*H*` for receivers.

---

## 7. What we claim — and what we do not

**We claim** for this **controlled** nonsense-copy setup we can:

1. **Localize** induction-like heads.  
2. **Stress them causally** (patching / K overwrite).  
3. **Sketch a bipartite graph** from automated path patching.

**We do not claim** that “three stripes” or “one dominant column” holds on **all** text—only on **this prompt family** and **this metric**. Natural repeats may look **more distributed**.

---

## 8. Closing

**Hypothesis → localization → causal tests → Q/K/V split → path edges → automated heatmap.**  
That is the story of **“Evidence of a Bipartite Induction Circuit in GPT-2 Small via Automated Path Patching”**: not the full **12×12 = 144** head graph squared, but a **pruned** **60×*n*** edge panel that explains **most of the measured effect** on the **Argl–Flargh**-style task.
