# Evidence of a Bipartite Induction Circuit in GPT-2 Small (Automated Path Patching)

This note summarizes a full mechanistic interpretability pipeline on **GPT-2 small** for a **nonsense-name copy** task (“Argl / Flargh” style prompts). The goal is to move from a 124M-parameter black box to a **small, testable subgraph** that explains most of the behavior **for this specific task**.

---

## 1. Hypothesis

**Induction:** After seeing a rare token sequence once, the model can predict the same continuation when a prefix repeats later—even when **lexical priors are useless** (nonsense names, no real-world association). We hypothesize that a subset of attention heads implements this by attending from the **second occurrence** back to the **first**, then copying the appropriate continuation.

---

## 2. Localization (Correlational)

We:

- Built **string prompts** with `model.to_tokens`, forcing **identical BPE substrings** for the repeated prefix so induction is not an artifact of mismatched tokenization.
- Used a **fully non-repeating counterfactual** corrupt prompt for activation patching (so early-layer K/V caches are not accidentally identical to clean on the first half of the sequence).
- Scored heads with an **induction-style attention metric**: queries over the **second** span of the repeated prefix, keys at the **aligned** positions in the **first** span (offset chosen so the circuit targets “successor-of-match” behavior, not naive duplicate-token identity).
- **Thresholded** heads (e.g. score > 0.2) to obtain a small set of **candidate induction receivers** (~18 heads in typical runs).

This step answers: *which heads correlate with the induction pattern?*

---

## 3. Causal Proof (Interventions)

We then asked *are those heads necessary?*

- **Activation patching** on Q/K/V pathways with the true counterfactual cache showed that **all three pathways** can matter when the corrupt cache actually breaks the repeated structure (earlier pitfall: sharing the first 50 tokens between clean and corrupt makes K/V patches invisible).
- **Key overwrites from a global corrupted residual** (`resid_post` after layer 4, LN scale frozen from the clean run per receiver layer) stress-tests whether induction can be killed by scrambling what late heads read as “context.”

This step answers: *does ablating or corrupting those components hurt performance in the way induction predicts?*

---

## 4. Mechanistic Detail (Q / K / V)

Splitting the intervention by pathway separates:

- **Whether** the head forms the right match (Q/K),
- **What** it writes into the stream once it attends (V),

and shows that the story is not “attention is one wire”—different pathways carry different parts of the computation.

---

## 5. Circuit Mapping (Path Patching)

**Path patching** replaces a *path-specific* contribution: take an early per-head **write** (from `hook_result`), linearize it through the **receiver’s** \(W_K\) under a **frozen LayerNorm scale** (clean run’s `ln1.hook_scale` at the receiver block), and swap clean vs corrupt projections into the **receiver’s** `hook_k`. That tests a **hypothesized edge** from an early sender head to a late induction head’s key computation.

---

## 6. Automation (Edge Heatmap)

To map **which early heads send signal into which late induction heads**, we ran **automated path patching**:

- **Senders:** every head in **layers 0–4** (60 heads).
- **Receivers:** only the **thresholded induction heads** (~18), not all \(12 \times 12\) late heads—reducing the grid from \(60 \times 36\) to **\(60 \times n_{\text{target}})\)** (e.g. **~1,080** edges instead of **~2,160**).

Each cell records the **drop in logit difference** (correct vs plausible wrong token at the evaluation position) after patching that single edge: **Δ = (clean logit diff) − (patched logit diff)**. Large positive values mean the edge **supports** the correct continuation under this metric.

### How to read the heatmap

- **Bright horizontal stripes (senders):** a few early heads account for most of the causal effect on specific receiver columns when patched. In GPT-2 small, **layer-0 “previous token” style heads** often appear as strong senders: they move **position \(t-1\)** information into the residual at \(t\), which downstream induction heads can reuse.
- **Mid-layer senders (layers 2–3):** sometimes show up as **secondary senders**—heads that refine or route information after the first residual updates, making late induction easier to read.
- **Column intensity:** for a **sparse** nonsense task, one sometimes sees **one receiver column** (e.g. **L5H5**) dominate the panel: many heads *can* do induction in principle, but **this prompt** may route most of the task through a **narrow late pathway**.
- **Dark / negative patches:** a patch that **increases** logit difference implies the patched sender normally **hurts** the correct answer under this counterfactual—consistent with **negative induction**, **anti-copy**, or **regularization-like** behavior (e.g. reducing pathological repetition).

Exact head indices for each stripe should be read off the saved axis labels (row index → layer and head within the L0–L4 sender grid; column label → `L*H*` receiver).

---

## 7. Interpretation (What we claim—and what we do not)

**We claim:** For this **controlled** nonsense-copy setup, we can:

1. **Localize** induction-like heads correlatively.
2. **Causally stress** them with patching and K-overwrites.
3. **Map a bipartite sender–receiver graph** between early layers and a small receiver set via automated path patching.

**We do not claim:** That three stripes or a single dominant receiver are **universal** for all text; they are **evidence about this distribution of prompts** and this metric. The same machinery may look more distributed on natural Wikipedia-style repeats.

---

## 8. Closing

Starting from “induction exists,” we progressed through **localization → causal interventions → Q/K/V decomposition → path-level edges → automated edge search**. In writeup language, that is exactly the arc of **“Evidence of a Bipartite Induction Circuit in GPT-2 Small via Automated Path Patching”**: not the full 144-node attention graph, but a **deliberately pruned** subgraph that explains **most of the measured effect** for the **Argl–Flargh**-style task.
