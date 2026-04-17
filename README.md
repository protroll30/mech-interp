# Transformer Forensics: From Circuits to SAE Features

## Overview

This repository is a structured, end-to-end mechanistic interpretability study of **GPT-2 Small**. The goal is not to fine-tune behavior from the outside, but to **reverse engineer internal computation**: we trace how information is routed, where symbolic and semantic updates occur, and finally which **sparse autoencoder (SAE) features** implement specific concepts. The tooling is built on **TransformerLens** for full access to activations and hooks, and **SAELens** for loading public SAEs and running feature-level analyses.

The arc is deliberately layered. We start with **circuits** (multi-head pathways that move logits), move to **logic** (where negation is actually implemented in depth), and finish with **atomic features** (interpretable directions in activation space and causal steering).

---

## Directory structure

```text
mech-interp/
├── induction_scripts/
│   ├── baseline_induction.py
│   ├── universal_induction.py
│   └── induction_circuit_writeup.md
├── ioi_scripts/
│   ├── ioi_baseline.py
│   ├── ioi_name_movers.py
│   ├── ioi_resid_sweep.py
│   ├── ioi_s_inhibition_search.py
│   ├── ioi_attention_viz.py
│   └── ioi_circuit_writeup.md
├── negation_scripts/
│   ├── negation_baseline.py
│   ├── negation_head_attribution.py
│   ├── negation_mlp_attribution.py
│   └── negation_circuit_writeup.md
├── sae_scripts/
│   ├── sae_exploration.py
│   ├── sae_contrastive_search.py
│   └── sae_steering.py
├── requirements.txt
├── LICENSE
└── README.md
```

**`induction_scripts/`**  
Scripts and notes for **induction-style behavior**: repeated structure in context, and how the model completes copies or patterns. The writeup summarizes what the runs show on controlled prompts.

**`ioi_scripts/`**  
A compact **Indirect Object Identification (IOI)** style suite on the classic John and Mary template: clean versus corrupt stories, name-mover patching, residual mid sweeps, inhibition-style probes, and attention visualizations. `ioi_circuit_writeup.md` records quantitative takeaways from this repo’s runs.

**`negation_scripts/`**  
Negation completion and attribution: baselines, logit lens style traces, per-head writes, and **attention versus MLP** comparisons on a fixed prompt family. `negation_circuit_writeup.md` is the lab-style record of metrics and figures.

**`sae_scripts/`**  
**SAELens** workflows on **`gpt2-small-mlp-out-v5-32k`** at **`blocks.8.hook_mlp_out`**: exploratory feature ranking, **contrastive** clean versus negated prompts, and **activation steering** with a dose-response curve for a targeted feature.

---

## Key experiments and findings

### Phase 1: Circuit level (where information moves)

At this scale, the model looks like a network of **attention heads that route information** between positions and depths, plus MLP blocks that mix and transform representations.

**Induction (`induction_scripts/`)**  
Induction heads implement a recognizable pattern: attend back to earlier context, retrieve a consistent earlier token or relation, and use it to drive the next prediction. The scripts here probe that behavior on copy-style prompts; the writeup ties the numbers back to a concrete circuit story.

**IOI (`ioi_scripts/`)**  
The IOI template isolates **syntax and role structure** (who gave what to whom) from surface tokens. Attention heads act as **routers**: they move token identities and relational cues so that the final position can favor the correct indirect object (Mary versus John) under clean versus corrupt patching. The suite measures **recovery** under head patching, sweeps **where in depth** a clean residual write is sufficient at the critical site, and probes narrower inhibition channels. Together, this is the macroscopic picture: **which heads carry the decision, and through which residual pathways**.

### Phase 2: Logic level (how concepts transform)

**Negation (`negation_scripts/`)**  
Here the prompt fixes a simple semantic tension: after *The man is not happy, he is*, does the model lean toward a negative continuation (for example *sad*) or slip back toward *happy*? The interesting finding, supported by linear readouts in these scripts, is a **division of labor**:

- **Attention** largely **preserves and propagates the baseline syntactic frame** (subject, copula, parallel clause), much like the IOI setting where heads route structure.
- **MLPs** carry a disproportionate share of the **semantic reversal**: the step that actually pushes the representation toward the negated reading rather than repeating the positive predicate.

So the story sharpens: attention is often the **plumbing**, while MLPs are where **polarity and lexical choice** begin to separate. That motivates Phase 3, because SAEs on MLP outputs are a natural place to look for **feature-sized** units of meaning.

### Phase 3: Feature level (what the model is actually thinking)

**SAE analysis (`sae_scripts/`)**  
We load a public **MLP-output SAE** at layer 8 and treat each latent as a candidate **atomic concept direction** in the space the MLP writes into.

1. **`sae_exploration.py`**  
   Baseline loading, forward pass, and **top activating features** at a chosen token. This grounds the vocabulary of latents before any contrastive or causal work.

2. **`sae_contrastive_search.py`**  
   **Contrastive search** compares two prompts that differ mainly in negation, for example *The man is happy, he is* versus *The man is not happy, he is*. We take the **final-token** MLP-out vector for each run, encode both with the SAE, and rank features by **positive activation differences**. Features that spike only under negation are candidates for a **pure negation axis**. In this project line, **Feature 20151** is highlighted as that contrastive negation feature.

3. **`sae_steering.py`**  
   **Causal evidence** comes from **activation steering**: we add `coeff * W_dec[20151]` into **`blocks.8.hook_mlp_out`** at the **last input token** and sweep `coeff` over a dose grid. The printed table tracks **logit(` sad`) minus logit(` happy`)** and the **greedy next token** after the intervention. The point of the curve is to show a **dose-response**: small injections barely move the readout, while larger injections **invert** the local preference away from a naively *happy* continuation. In the project line summarized here, strong enough doses **flip the greedy prediction toward negation-style wording**, including **`not`**, while the sad-minus-happy gap moves in the direction expected if the model is being pulled into a negative or negated state. That combination is the **causal** capstone on Feature **20151**, not only a contrastive correlate.

Together, these phases mirror a common MI workflow: **locate the circuit, localize the transforming submodule, then name and test the features inside it**.

---

## Installation and setup

**Prerequisites**

- A recent **Python** (3.10+ recommended).
- **PyTorch** installed for your platform (CPU or CUDA). SAELens and TransformerLens follow your existing Torch install.

**Clone and environment**

```bash
git clone <your-repo-url> mech-interp
cd mech-interp
python -m venv .venv
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

**Dependencies**

Core libraries are pinned in `requirements.txt`:

- **`transformer-lens`** for `HookedTransformer`, hooks, patching, and caches.
- **`sae-lens`** for `SAE.from_pretrained`, encoders, and public release weights.
- **`matplotlib`** for plots produced by some attribution scripts.

Install with:

```bash
pip install -U pip
pip install -r requirements.txt
```

If you use CUDA, install the matching **torch** wheel from the official PyTorch instructions first, then run `pip install -r requirements.txt` so dependency resolution sees your Torch build.

**Optional notes**

- First runs will download **GPT-2 Small** and SAE weights from Hugging Face; ensure you have disk space and (if needed) network access.
- Scripts are written as **command-line experiments**: open a file, read the constants at the top, then run with `python path/to/script.py` once your environment is active.

---

## How to read this repo

Start with the **markdown writeups** in each folder for the narrative and tables, then open the corresponding **`.py`** files for the exact definitions of prompts, hooks, and metrics. The SAE phase assumes familiarity with the negation scripts, because the same prompt family motivates which contrast you run in **`sae_contrastive_search.py`** and which intervention you stress-test in **`sae_steering.py`**.


