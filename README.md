# Transformer Forensics: Negation Circuits and SAE Falsification in GPT-2 Small

## What this repo is

This is a **mechanistic interpretability** codebase for **GPT-2 Small**. We use **TransformerLens** for full forward access, hooks, and caches, and **SAELens** where we study a public sparse autoencoder on MLP outputs. The centerpiece is an **end-to-end causal story for semantic negation** in a fixed completion template, backed by path patching, residual patching, QKV patching, and necessity-style SAE ablations.

The headline result is **not** “layer 8 MLP implements negation” or “SAE feature 20151 is the negation feature.” Controlled runs **falsified** those hypotheses. See **[`formal_writeup.md`](formal_writeup.md)** for the full graph: **early attention routers**, **late attention value pathways (including L7H5)**, and a **late suppressive MLP prior** (including feature **20151** as a **structural** correlate, not a logical negation atom).

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
│   ├── resid_adjective_patch_sweep.py      # residual patch at " happy", layers 0-6
│   ├── router_heads_l1_l2_sweep.py         # head-level hook_result patch, L1-L2
│   ├── early_attn_mlp_adjective_patch.py   # attn vs MLP out at L1-2
│   ├── l7h5_qkv_patching.py                # Q/K/V path patching for L7H5
│   ├── negation_baseline.py
│   ├── negation_head_attribution.py
│   ├── negation_mlp_attribution.py
│   └── negation_circuit_writeup.md         # legacy lab notes (superseded by formal writeup)
├── sae_scripts/
│   ├── sae_exploration.py
│   ├── sae_contrastive_search.py         # correlational: negated minus clean
│   ├── sae_steering.py                   # dose response on decoder direction
│   ├── sae_necessity_test.py             # causal: ablation, DLA, controls
│   ├── l7h5_attention.py                 # CircuitsVis HTML + attention stats
│   └── l7h5_attention.html               # generated viz (open in browser)
├── formal_writeup.md                     # main scientific narrative + code map
├── requirements.txt
├── LICENSE
└── README.md
```

**`induction_scripts/`**  
Induction-style behavior: repeated structure, path patching edges, and a short writeup.

**`ioi_scripts/`**  
Indirect Object Identification style tasks: clean versus corrupt stories, name movers, residual sweeps, inhibition probes. Good **methodological** background for patching and recovery metrics.

**`negation_scripts/`**  
Causal interventions on the negation minimal pair (see `formal_writeup.md`). This is where the **router** and **L7H5 QKV** evidence lives.

**`sae_scripts/`**  
SAE loading (**`gpt2-small-mlp-out-v5-32k`**, **`blocks.8.hook_mlp_out`**), contrastive ranking, steering, necessity tests, and L7H5 attention visualization.

---

## How to read the science

1. Start with **[`formal_writeup.md`](formal_writeup.md)** (abstract, methodology, three-part causal graph, falsification of feature 20151, script table).
2. Open the cited **`.py`** files for exact prompts, hooks, and metric formulas.
3. Optional: older folder writeups (`negation_circuit_writeup.md`, `ioi_circuit_writeup.md`, `induction_circuit_writeup.md`) for historical context; the **authoritative** negation story is the formal writeup plus the scripts it names.

---

## Installation and setup

**Prerequisites**

- Python **3.10+** recommended.
- **PyTorch** for your platform (CPU or CUDA).

**Environment**

```bash
git clone <your-repo-url> mech-interp
cd mech-interp
python -m venv .venv
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
source .venv/bin/activate
```

**Dependencies**

```bash
pip install -U pip
pip install -r requirements.txt
```

Core packages include **transformer-lens**, **sae-lens**, **matplotlib**, and **circuitsvis** (for [`sae_scripts/l7h5_attention.py`](sae_scripts/l7h5_attention.py)). Install a CUDA **torch** build first if you use GPU, then run `pip install -r requirements.txt`.

First runs download **GPT-2 Small** and SAE weights; allow disk space and network as needed.

**Running experiments**

```bash
python negation_scripts/resid_adjective_patch_sweep.py
python sae_scripts/sae_necessity_test.py
```

Constants (prompts, layer IDs, feature ID) live at the top of each script.

---

## License

See `LICENSE` in the repository root.
