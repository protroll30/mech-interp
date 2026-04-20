# End-to-End Causal Structure of Semantic Negation in GPT-2 Small: Circuit Discovery and SAE Falsification

## Abstract

This repository documents a mechanistic interpretability study of **GPT-2 Small** focused on **semantic negation** in template completions (for example, *The man is not happy, he is [sad]*). We combine **TransformerLens** interventions (path patching, residual patching, QKV patching) with **SAELens** analyses on a public MLP-output sparse autoencoder. The original working hypothesis was that **late MLPs** at layer 8 implement reversal, and that **SAE feature 20151** is a dedicated negation feature. Controlled **causal** tests falsified that story. The evidence instead supports a **two-stage attention circuit** that establishes polarity early, plus a **late suppressive prior** implemented partly as a **structural skip-gram style** feature that fights the logically correct antonym logit. This note states the **updated causal graph**, separates **correlation from causation** explicitly, and maps each claim to the **exact scripts** that justify it.

---

## Directory structure (experiment code)

High level layout:

```text
mech-interp/
├── induction_scripts/          # Induction-style circuits (baseline + path patching)
├── ioi_scripts/                # IOI-style name routing and patching
├── negation_scripts/           # Negation completion: attribution + causal sweeps (this writeup’s core)
├── sae_scripts/                # SAE loading, contrastive search, necessity tests, L7H5 attention viz
├── formal_writeup.md           # This document
├── requirements.txt
└── README.md
```

**Negation scripts (causal graph):**

| Script | Role |
|--------|------|
| [`negation_scripts/resid_adjective_patch_sweep.py`](negation_scripts/resid_adjective_patch_sweep.py) | Residual `hook_resid_post` patch at the **adjective** token, layers 0-6, corrupt run. Locates where clean negation state appears in depth. |
| [`negation_scripts/router_heads_l1_l2_sweep.py`](negation_scripts/router_heads_l1_l2_sweep.py) | **Head-level** `hook_result` patch at adjective position only, layers 1-2. Ranks router heads; optional clean attention query(adjective)->key(not). |
| [`negation_scripts/early_attn_mlp_adjective_patch.py`](negation_scripts/early_attn_mlp_adjective_patch.py) | Sublayer patches: `hook_attn_out` vs `hook_mlp_out` at layer 1 and 2, adjective only. Tests whether early MLPs **amplify** the routed signal. |
| [`negation_scripts/l7h5_qkv_patching.py`](negation_scripts/l7h5_qkv_patching.py) | Q, K, and V path patching for **L7H5** on corrupt, from clean cache. |
| [`negation_scripts/negation_baseline.py`](negation_scripts/negation_baseline.py), [`negation_head_attribution.py`](negation_scripts/negation_head_attribution.py), [`negation_mlp_attribution.py`](negation_scripts/negation_mlp_attribution.py) | Earlier linear readouts and baselines on the fixed prompt family. |

**SAE scripts (feature-level and attention):**

| Script | Role |
|--------|------|
| [`sae_scripts/sae_contrastive_search.py`](sae_scripts/sae_contrastive_search.py) | **Correlation:** `Act_negated − Act_clean` in SAE space (false-positive generator if read causally). |
| [`sae_scripts/sae_necessity_test.py`](sae_scripts/sae_necessity_test.py) | **Causality:** SAE latent zero-ablation, signed metrics, DLA layers 4-7, z-patch screen, "very" control for feature 20151. |
| [`sae_scripts/sae_exploration.py`](sae_scripts/sae_exploration.py), [`sae_scripts/sae_steering.py`](sae_scripts/sae_steering.py) | Exploration and steering doses on MLP8 (historical baselines). |
| [`sae_scripts/l7h5_attention.py`](sae_scripts/l7h5_attention.py) | CircuitsVis HTML + last-token attention row + negation vs happy mass. |

Supporting assets: [`sae_scripts/l7h5_attention.html`](sae_scripts/l7h5_attention.html) (generated visualization).

---

## Methodology: circuits first, then negation as the main pipeline

**Induction** (`induction_scripts/`) and **IOI** (`ioi_scripts/`) are the methodological backbone of the repo: they establish how we think about **clean versus corrupt pairs**, **activation caching**, **head patching**, and **recovery-style metrics**. Induction studies repeated structure and successor-style routing; IOI studies role and name identity under minimal edits. Those folders are foundational context for **how** we run interventions.

The **negation pipeline** is where we invested the causal graph. Unless stated otherwise, the canonical pair is:

- **Clean (negated):** `The man is not happy, he is`
- **Corrupt (control):** `The man is very happy, he is` (same length, adjective aligned)

The primary **metric** is **logit difference at the final token**:

**LD = logit(` sad`) − logit(` happy`)** (leading-space BPE fragments as in the code).

**Recovery** when patching corrupt with clean components:

**Recovery = (LD_patched − LD_corrupt) / (LD_clean − LD_corrupt)**

when the denominator is nonzero. This is a standard way to express how much of the clean-to-corrupt gap a single intervention closes.

We distinguish:

- **Observational / correlational:** contrastive activation differences, DLA-style dot products, attention maps.
- **Interventional / causal:** ablations, path patching, residual patching, QKV patching.

---

## Results: the negation circuit (three-part graph)

### Part A. Suppressive prior: late MLPs and SAE feature 20151

**Early hypothesis:** Layer 8 MLP output “computes” negation, and feature **20151** is a negation feature.

**Falsification (causal):** In [`sae_scripts/sae_necessity_test.py`](sae_scripts/sae_necessity_test.py), we **zero** latent 20151 in the SAE reconstruction of `blocks.8.hook_mlp_out`, then decode back. On the negation prompts, **ablation often increases LD**, so the feature’s presence **suppresses** the antonym edge relative to the ablated state. Full MLP8 zero-ablation can move LD in the same direction, which motivated **signed** reporting: **Δ = LD_ablated − LD_clean** (negative means the component was helping the antonym readout). The script also compares activation on **not** versus **very** at the last token; feature 20151 is **not** specific to “not”.

**Conclusion:** Late MLP activity at this hook includes a **copy-style prior** aligned with repeating local structure (for example favoring *happy* after *happy*). Feature **20151** behaves like a **structural skip-gram correlate**, not a faithful negation atom. Steering in [`sae_scripts/sae_steering.py`](sae_scripts/sae_steering.py) remains useful as a **phenomenological** dose curve, but it must not be read as proving a “negation feature” in the logical sense after the necessity tests.

### Part B. Step 1: distributed routers (layers 1 and 2)

**Residual evidence:** [`negation_scripts/resid_adjective_patch_sweep.py`](negation_scripts/resid_adjective_patch_sweep.py) patches **`hook_resid_post`** at the **adjective** token only (index 5), swapping clean into corrupt, for layers 0-6. Recovery **jumps** by early depth (in our runs, near **full recovery** by layer 2-3), which pins **when** the adjective residual carries negation-relevant state.

**Head evidence:** [`negation_scripts/router_heads_l1_l2_sweep.py`](negation_scripts/router_heads_l1_l2_sweep.py) patches a **single head's** `hook_result` at the adjective only, for layers 1-2. Heads such as **L1H0** rank highly; the script prints **clean** attention from query position **adjective** to key position **` not`**, which supports the reading that early heads **read** the negation token and **write** into the adjective site.

**Sublayer check (amplification):** [`negation_scripts/early_attn_mlp_adjective_patch.py`](negation_scripts/early_attn_mlp_adjective_patch.py) patches **`hook_attn_out`** versus **`hook_mlp_out`** at layers 1-2, adjective only. In our runs, **attention outs** recover more than **MLP outs**; MLP patches can even **hurt** recovery. That is evidence against a simple story where early MLPs **amplify** the routed negation vector. Routing reads as **attention-driven** and **distributed**, not as an MLP gain stage at this site.

### Part C. Step 2: semantic remappers (layers 6-7, L7H5)

**Correlational direction:** [`sae_scripts/sae_necessity_test.py`](sae_scripts/sae_necessity_test.py) includes **linear DLA** (dot into **W_U[sad] - W_U[happy]**) for attention and MLP hooks in layers 4-7. **L7H5** shows a large positive head term in that screen, with a layer 6 cluster also contributing in the aggregate table.

**Causal pathway:** [`negation_scripts/l7h5_qkv_patching.py`](negation_scripts/l7h5_qkv_patching.py) patches **Q**, **K**, or **V** for **L7H5** from clean into corrupt. In our runs, **Value** patching recovers most of the gap (**on the order of tens of percent**), while **Q** and **K** are comparatively weak. That is consistent with late heads **reading** a prepared residual mostly through the **value pathway** and writing toward the antonym direction at the final position, while competing with the late MLP suppressive prior from Part A.

**Attention visualization:** [`sae_scripts/l7h5_attention.py`](sae_scripts/l7h5_attention.py) exports [`sae_scripts/l7h5_attention.html`](sae_scripts/l7h5_attention.html) and prints the last-row mass; the **negation attention ratio** there compares mass on negation keys versus **happy** for interpretation.

---

## The falsification of feature 20151: why contrastive search was misleading

[`sae_scripts/sae_contrastive_search.py`](sae_scripts/sae_contrastive_search.py) ranks features by **positive** differences **negated minus clean** in SAE space at a chosen token. That design is excellent for **hypothesis generation**: it surfaces latents that **co-occur** with the surface cue *not*.

It is **not** a causal identification strategy. Correlation **does not** equal causation: a feature can track the negated template for **structural** reasons (shared bigrams, clause shape, repetition) while the **causal** role of its MLP channel is to **support** a **happy**-like prior.

[`sae_scripts/sae_necessity_test.py`](sae_scripts/sae_necessity_test.py) closes the loop:

1. **Zero-ablation** of latent 20151 shifts logits in the direction consistent with **removing a suppressor** of the antonym edge on several prompts.
2. The **not** versus **very** activation probe shows the feature is **not** a clean “not detector.”
3. **Full MLP8** ablation and **feature-only** ablation are reported with **aligned** recovery ratios only when effects share sign, to avoid bogus “percent explained” when the layer acts as a **net suppressor** on LD.

Together, this is the repo’s clearest **rigor** story: we **named** a feature from correlation, then **broke** the naive interpretation with necessity-style tests.

---

## Code map (quick reference)

| Finding | Primary scripts |
|--------|-------------------|
| Residual “jump” at adjective by depth | [`resid_adjective_patch_sweep.py`](negation_scripts/resid_adjective_patch_sweep.py) |
| Early router heads L1-L2 | [`router_heads_l1_l2_sweep.py`](negation_scripts/router_heads_l1_l2_sweep.py) |
| Attn vs MLP at L1-2 (amplification test) | [`early_attn_mlp_adjective_patch.py`](negation_scripts/early_attn_mlp_adjective_patch.py) |
| L7H5 QKV decomposition | [`l7h5_qkv_patching.py`](negation_scripts/l7h5_qkv_patching.py) |
| DLA screen + SAE necessity + controls | [`sae_necessity_test.py`](sae_scripts/sae_necessity_test.py) |
| Contrastive false-positive generator | [`sae_contrastive_search.py`](sae_scripts/sae_contrastive_search.py) |
| L7H5 attention figure | [`l7h5_attention.py`](sae_scripts/l7h5_attention.py) |

---

## Limitations (short)

Linear DLA ignores `ln_final` and is best paired with patching. Single-head patches at a single position underestimate distributed routing. Recovery percentages can exceed 100% when nonlinear interactions couple patches to later blocks. We report those caveats in the scripts where they matter.

---

## Citation style for this repo

If you reuse this narrative, cite the **scripts** and the **metric definitions** inside them as the ground truth for numbers, because exact logits vary slightly with library versions and hardware.
