# Competing Subsystems and Distributed Routing in GPT-2 Small: A Systems View of Semantic Negation

## Abstract

We report a mechanistic interpretability study of **GPT-2 Small** on a controlled **semantic negation** completion task (for example, *The man is not happy, he is [sad]*). The project began with a familiar but fragile hypothesis: that **late MLP computation** implements logical negation and that a single **sparse autoencoder (SAE) feature** at MLP output might act as a discrete "negation unit." **Causal interventions** overturned that picture. The evidence supports a **systems-level** account: **competing subsystems** whose outputs **superimpose in residual space**. A **fast associative prior** anchored in late MLP channels (including a high-variance SAE direction) tends to reinforce **local lexical continuity** (for example *happy* after *happy*). A **slower, distributed attention-driven pathway** routes polarity information from the negation site to the adjective, then **late attention heads** (notably **L7H5**) **actuate** the antonym direction primarily through the **value pathway**, partially **overriding** the prior. We separate **correlational** probes (contrastive SAE ranking, linear direct logit attribution) from **interventional** probes (residual patching, head patching, sublayer patching, QKV patching, SAE zero-ablation). Every substantive claim below is tied to a **specific script** in this repository.

---

## 1. Systems overview: heuristic prior versus logical override

Complex engineered systems, and learned dynamical systems, often exhibit **default dynamics**: shallow feedback that returns state toward familiar attractors. They also admit **override pathways** that inject task-specific corrections. Large language models are not literal programs with explicit *if-then* negation gates at single neurons. They are high-dimensional **vector computers** where **attention** and **MLPs** contribute additive updates to a shared residual stream.

We use two informal labels, grounded in our measurements rather than in any literal module boundary:

1. **Fast associative subsystem (suppressive prior).** Late MLP blocks, here centered on **layer 8 output**, implement **shallow contextual prediction** aligned with **n-gram and repetition structure**. In SAE space, **feature 20151** behaves as a **structural skip-gram correlate**: it activates under local template continuity, not only under logical negation.

2. **Slow computational subsystem (logical override).** Implemented as **distributed attention routing** in early layers, then **semantically directed attention** in mid-to-late layers. It does not read as a single gate. It reads as **constructive superposition**: multiple heads contribute partial updates that, together, move the final position’s logits toward the **antonym** relative to a **very**-style control.

The empirical work tests **causal necessity and recovery**, not only interpretability stories suggested by heatmaps.

---

## 2. Methodology

**Model and libraries.** All experiments use **GPT-2 Small** through **TransformerLens** (`HookedTransformer`, activation caches, forward hooks). SAE experiments use **SAELens** with release **`gpt2-small-mlp-out-v5-32k`** at hook **`blocks.8.hook_mlp_out`**.

**Canonical minimal pair.** Unless noted:

- **Clean (negated):** `The man is not happy, he is`
- **Corrupt (control):** `The man is very happy, he is`

Tokenizations match in length; the adjective token **` happy`** aligns at **sequence index 5** under default settings in our scripts.

**Primary metric (final token).** Let `id_sad` and `id_happy` be tokenizer ids for leading-space **` sad`** and **` happy`**. Define

**LD = logit(` sad`) - logit(` happy`)**

at the **last** input position. Higher LD favors the antonym relative to the repeated adjective reading.

**Recovery under patching** of corrupt runs with clean activations:

**Recovery = (LD_patched - LD_corrupt) / (LD_clean - LD_corrupt)**

when the denominator is nonzero. Recovery estimates how much of the **clean-versus-corrupt separation** a single intervention restores.

**Contrast versus causation.**

- **Correlational:** contrastive activation differences in SAE latents; linear **direct logit attribution (DLA)** using `W_U` directions without `ln_final` (a linearization, reported as such in code comments).
- **Causal:** activation **patching** (residual, head write, sublayer output, QKV slices), and **SAE latent zero-ablation** with decode-back to MLP output.

Scripts in **`negation_scripts/`** and **`sae_scripts/`** implement the metrics exactly; numbers in prose refer to **representative runs** produced by those scripts on CPU unless you re-execute on other hardware.

---

## 3. Phase 1: The suppressive prior and the falsification of feature 20151

### 3.1 Correlational origin: contrastive SAE search

[`sae_scripts/sae_contrastive_search.py`](sae_scripts/sae_contrastive_search.py) compares **final-token** MLP-out activations on a **clean** versus **negated** prompt, encodes both with the SAE, and ranks latents by **positive** differences **negated minus clean**. That procedure is appropriate for **hypothesis generation**: it surfaces features that **co-vary** with the presence of *not*.

**Systems reading:** contrastive ranking answers "what differs," not "what causes the logit change."

### 3.2 Causal tests: necessity, signed effects, and controls

[`sae_scripts/sae_necessity_test.py`](sae_scripts/sae_necessity_test.py) performs **zero-ablation** in SAE space: encode MLP output, set latent **20151** to zero, decode, replace **`blocks.8.hook_mlp_out`**. On several negation-style prompts, **ablating** the feature **increases** LD relative to the unablated forward pass. Equivalently, the feature’s presence **suppresses** the antonym edge on those measurements. The script therefore reports **signed ablation deltas** **LD_ablated - LD_clean**, not naive "percent drop" formulas that flip sign when late MLPs behave as **net suppressors** on LD.

The same script compares feature activation on **not** versus **very** minimal pairs. Feature **20151** is **not** a clean detector of *not* alone; it tracks **shared surface structure** around *happy*.

**Full MLP8 zero-ablation** in that file provides a **ceiling-style** reference for how destructive removing all of layer 8 MLP output is for the metric, and **aligned ratio** summaries only when feature and full-layer effects share sign.

### 3.3 Systems synthesis for Phase 1

Late MLP channels at this depth participate in a **default continuation prior** that favors **repeating the visible predicate**. SAE feature **20151** is better understood as a **compressive axis** aligned with that prior than as a **logic node**. This is the central **falsification** of a single-feature negation story and the strongest **causal** highlight of the SAE portion of the repo.

Related exploratory and steering scripts (phenomenology, not necessity): [`sae_scripts/sae_exploration.py`](sae_scripts/sae_exploration.py), [`sae_scripts/sae_steering.py`](sae_scripts/sae_steering.py).

---

## 4. Phase 2: Distributed routers (layers 1 and 2)

### 4.1 When the adjective residual becomes informative

[`negation_scripts/resid_adjective_patch_sweep.py`](negation_scripts/resid_adjective_patch_sweep.py) patches **`blocks.L.hook_resid_post`** **only** at the **adjective** position, swapping the **clean** vector into the **corrupt** forward, for **L = 0 ... 6**. Recovery **jumps** by mid depth: in reported runs, recovery reaches on the order of **full** restoration of the clean-corrupt gap by **layer 2-3**, which localizes **when** the adjective site carries enough negation-relevant state for this metric.

**Systems reading:** the residual at the adjective is not "empty" until late depth; a **distributed** sequence of blocks builds a state there that differs systematically between *not* and *very*.

### 4.2 Head-level writes at the adjective

[`negation_scripts/router_heads_l1_l2_sweep.py`](negation_scripts/router_heads_l1_l2_sweep.py) requires **`model.set_use_attn_result(True)`** so **`hook_result`** is materialized. For each head in layers **1** and **2**, it patches **only** the adjective position’s head slice from **clean** into **corrupt**. Heads such as **L1H0** rank highly on recovery among this narrow intervention class. The script optionally prints **clean** attention from the **adjective** query position to the **` not`** key position for the top head, supporting the interpretation **read negation, write at adjective**.

**Systems reading:** routing is **distributed** and **sparse in head space**; no single head is the whole story.

### 4.3 Sublayer decomposition: attention versus MLP at the same site

[`negation_scripts/early_attn_mlp_adjective_patch.py`](negation_scripts/early_attn_mlp_adjective_patch.py) patches **`hook_attn_out`** versus **`hook_mlp_out`** at layers **1** and **2**, adjective only. In reported runs, **attention outputs** recover more of the gap than **MLP outputs**; MLP patches can **hurt** recovery.

**Systems reading:** early **MLP** substeps at this token are not a simple **gain amplifier** for the routed negation signal. The constructive state is carried disproportionately by **attention-driven** updates, consistent with **distributed routing** rather than a localized MLP "amplifier."

---

## 5. Phase 3: Semantic remappers (layers 6-7, emphasis on L7H5)

### 5.1 Correlational direction: linear DLA screen

[`sae_scripts/sae_necessity_test.py`](sae_scripts/sae_necessity_test.py) includes a **linear DLA** block for layers **4-7**: per-head **(z_h @ W_O[h])** dotted with **W_U[sad] - W_U[happy]**, plus layer aggregates for **`hook_attn_out`** and **`hook_mlp_out`**. **L7H5** registers among the largest head-level terms in representative runs; layer **6** shows a **cluster** of heads in the same screen.

**Caveat:** DLA omits `ln_final`. It is an **orientation** tool, not a substitute for patching.

### 5.2 Causal pathway decomposition: Q versus K versus V

[`negation_scripts/l7h5_qkv_patching.py`](negation_scripts/l7h5_qkv_patching.py) patches **only head 5** on **`hook_q`**, **`hook_k`**, or **`hook_v`** from **clean** into **corrupt**. In representative runs, **value** patching recovers a large fraction of the clean-corrupt gap (**on the order of tens of percent**, near **45%** in one CPU log), while **query** and **key** patches are comparatively weak.

**Systems reading:** for this head, **what is mixed into the output** along the value pathway carries the intervention information relevant to LD, consistent with **actuators** that **read** prepared residual content and **write** toward logits, rather than only re-shaping compatibility with **Q/K** alone.

### 5.3 Attention visualization as auxiliary evidence

[`sae_scripts/l7h5_attention.py`](sae_scripts/l7h5_attention.py) renders an interactive **CircuitsVis** figure ([`sae_scripts/l7h5_attention.html`](sae_scripts/l7h5_attention.html)) and prints the **last-query** attention row. This supports qualitative inspection; **causal** claims rest on patching and ablation, not on the figure alone.

---

## 6. Conclusion: systems synthesis

GPT-2 Small does not resolve this negation template through a single interpretable gate. It resolves it through **superposed updates**:

- A **late associative prior**, expressed in MLP output space and partially captured by high-variance SAE directions such as **20151**, **fights** the antonym logit by reinforcing **local continuity**.
- An **attention-mediated override** **routes** polarity-related information **early** into the adjective residual **without** relying on early MLP amplification in our sublayer tests.
- **Late attention heads**, including **L7H5**, **actuate** the antonym direction largely through the **value pathway**, consistent with **reading** a prepared residual and **writing** into the logit readout.

This is **systems thinking** in the narrow sense we intend: **competing subsystems**, **distributed routing**, and **measured overrides**, grounded in **interventions** first.

---

## 7. Code reference map

| Claim family | Primary scripts |
|--------------|------------------|
| Contrastive SAE ranking (correlation) | [`sae_scripts/sae_contrastive_search.py`](sae_scripts/sae_contrastive_search.py) |
| SAE zero-ablation, signed metrics, DLA, controls | [`sae_scripts/sae_necessity_test.py`](sae_scripts/sae_necessity_test.py) |
| Residual depth sweep at adjective | [`negation_scripts/resid_adjective_patch_sweep.py`](negation_scripts/resid_adjective_patch_sweep.py) |
| Head-level router sweep L1-L2 | [`negation_scripts/router_heads_l1_l2_sweep.py`](negation_scripts/router_heads_l1_l2_sweep.py) |
| Early attn versus MLP sublayer test | [`negation_scripts/early_attn_mlp_adjective_patch.py`](negation_scripts/early_attn_mlp_adjective_patch.py) |
| L7H5 QKV patching | [`negation_scripts/l7h5_qkv_patching.py`](negation_scripts/l7h5_qkv_patching.py) |
| L7H5 attention visualization | [`sae_scripts/l7h5_attention.py`](sae_scripts/l7h5_attention.py) |
| SAE exploration and steering (context) | [`sae_scripts/sae_exploration.py`](sae_scripts/sae_exploration.py), [`sae_scripts/sae_steering.py`](sae_scripts/sae_steering.py) |
| Earlier linear attributions (context) | [`negation_scripts/negation_head_attribution.py`](negation_scripts/negation_head_attribution.py), [`negation_scripts/negation_mlp_attribution.py`](negation_scripts/negation_mlp_attribution.py) |

---

## 8. Limitations

Linear DLA ignores layer norm at unembedding. Single-site patches underestimate **distributed** computation. Recovery can exceed **100%** when nonlinear coupling makes patched activations interact with later blocks non-monotonically. All quantitative statements should be **recomputed** from scripts when precision matters for publication.
