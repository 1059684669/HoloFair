# HoloFair: Unified T2I Fairness Evaluation and Fair-GRPO Debiasing

[![Paper](https://img.shields.io/badge/Paper-ICML%202026-blue.svg)](https://icml.cc/virtual/2026/poster/62706)
[![arXiv](https://img.shields.io/badge/arXiv-2026.xxxxx-b31b1b.svg)](https://arxiv.org/)
[![Website](https://img.shields.io/badge/Website-Project%20Page-green.svg)]()
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-ffd21e.svg)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey.svg)](LICENSE)

**A comprehensive benchmark and RL-based debiasing framework for demographic fairness in Text-to-Image models.**

This repository contains the official implementation for **HoloFair** (ICML 2026). We introduce an end-to-end framework that evaluates deep-semantic fairness in T2I models and mitigates biases via reinforcement learning, without degrading image quality.

---

## 💡 Core Idea: Beyond Surface-Level Audits

Existing fairness evaluations only test default distributions (e.g., *"a photo of a person"*), missing the deeper biases that emerge under semantically loaded contexts. For instance, SDXL achieves the highest default diversity but collapses severely when prompted with *"a professional person"*. **HoloFair** exposes these hidden biases.

<div align="center">
<img src="assets/overview.png" width="88%">
</div>

**Figure 1.** Overview of the HoloFair framework. Our end-to-end pipeline consists of three stages: Dataset Construction, Classifier Training, and Fairness Evaluation.

---

## 🔎 The Three Components

### 📊 MGBI Metric

The **Multi-attribute, Group-wise Bias Index** jointly measures two complementary aspects of fairness:

- **Intrinsic Diversity (ID)**: Geometric mean of normalized entropies across gender, age, and race on neutral prompts. Penalizes models that are diverse on one attribute but collapsed on another.
- **Context-Robust Diversity (CA<sub>q</sub>)**: 10th percentile of per-trigger diversity scores under bias-inducing semantic contexts (grounded in the Stereotype Content Model). Captures near-worst-case behavior.
- **MGBI = √(ID × CA<sub>q</sub>)**: A single unified score. The geometric mean ensures a model cannot compensate low ID with high CA<sub>q</sub> or vice versa.

### 🔍 HoloFair Benchmark

An end-to-end evaluation framework built on:
- **Prompt Sets**: 300 neutral + 450 semantic trigger prompts (Eval), 10,000 prompts (Train), 300 biased prompts (Gen)
- **RBD Dataset**: 119K images from FairFace, UTKFace, in-the-wild portraits, and AI-generated images
- **SpaFreq Classifier**: Dual-stream DINOv2 architecture combining spatial semantics and wavelet frequency features (89.67% overall accuracy)

### ⚡ Fair-GRPO Debiasing

A reinforcement-learning method using a **multi-attribute per-prompt reward function**:
- Log-ratio base reward penalizes over-represented categories and rewards under-represented ones
- Zero-centered, clipped, and aggregated across gender, age, and race
- KL-regularized policy optimization prevents quality degradation
- Architecture-agnostic: validated on both UNet (SD1.5) and MMDiT (SD3.5M)


<div align="center">
<img src="assets/Fair-GRPO.png" width="88%">
</div>

**Figure 2.** Overview of the Fair-GRPO Debiasing method.



---

## 📈 Key Results

| | SD3.5M | SD1.5 |
|:--|:--:|:--:|
| **MGBI improvement** | +29.9% (0.52→0.68) | +20.2% (0.66→0.79) |
| **FID (lower is better)** | 135.09 (best) | 134.51 (best) |
| **CLIP-Score** | 0.2317 (improved) | 0.2237 (improved) |

> Fair-GRPO achieves state-of-the-art fairness on both architectures while maintaining or improving image quality. Full comparison tables are available in the paper.

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/RuyiChen11/HoloFair.git
cd HoloFair
conda create -n holofair python=3.10 -y
conda activate holofair
pip install -r requirements.txt
```

### 2. Download Weights

SpaFreq classifier weights and Fair-GRPO LoRA weights are hosted on HuggingFace:

| Model | Description | Link |
|:------|:------------|:----:|
| SpaFreq-Gender | Gender classifier (2 classes) | [🤗 Download]() |
| SpaFreq-Age | Age classifier (3 classes) | [🤗 Download]() |
| SpaFreq-Race | Race classifier (5 classes) | [🤗 Download]() |
| Fair-GRPO SD3.5M | Debiased LoRA weights | [🤗 Download]() |
| Fair-GRPO SD1.5 | Debiased LoRA weights | [🤗 Download]() |

### 3. Evaluate a T2I Model

```bash
python benchmark/evaluate.py \
    --model_path <path_to_model> \
    --prompt_set benchmark/prompts/eval_set.json \
    --classifier_weights classifiers/weights/ \
    --output_dir results/
```

### 4. Train Fair-GRPO

```bash
# SD3.5M (MMDiT architecture)
accelerate launch fair_grpo/train_sd3.py \
    --config fair_grpo/configs/sd3_fairness.py

# SD1.5 (UNet architecture)
accelerate launch fair_grpo/train_sd15.py \
    --config fair_grpo/configs/sd15_fairness.py
```

---

## 📁 Repository Structure

```
HoloFair/
├── benchmark/                     # HoloFair Evaluation
│   ├── evaluate.py                # MGBI evaluation script
│   ├── metrics/mgbi.py            # ID, CAq, MGBI implementation
│   └── prompts/                   # Eval & Train prompt sets
│
├── classifiers/                   # SpaFreq Classifiers
│   ├── train_classifier.py        # Training script
│   └── models/spafreq.py         # DINOv2 dual-stream architecture
│
├── fair_grpo/                     # Fair-GRPO Debiasing
│   ├── train_sd3.py               # SD3.5M training
│   ├── train_sd15.py              # SD1.5 training
│   ├── reward/                    # Multi-attribute reward function
│   ├── diffusers_patch/           # Pipeline & SDE with log-prob
│   └── configs/                   # Training configurations
│
└── data/                          # Dataset utilities
```

---

## 🧩 Extensibility

HoloFair is designed as an open framework:

- **New attributes**: The MGBI metric operates on any categorical distribution. Adding new demographic dimensions requires only training a new classifier head.
- **New models**: The evaluation pipeline accepts any T2I model. No architectural assumptions.
- **Non-uniform targets**: The reward function generalizes to arbitrary target distributions by specifying desired proportions. No code changes needed.
- **New semantic triggers**: Additional bias-inducing contexts can be added without modifying the evaluation framework.

---

## 📖 Citing Our Work

If you find HoloFair useful in your research, please consider citing:

```bibtex
@inproceedings{chen2026holofair,
  title     = {HoloFair: Unified T2I Fairness Evaluation and Fair-GRPO Debiasing},
  author    = {Chen, Ruyi and Xu, Xiaogang and Zhang, Chiyu and Wu, Jiafei and Fang, Liming and Zhou, Lu},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

---

## ⚠️ Ethics Statement

All classifiers and metrics operate at the distributional level for group-level assessment, not individual profiling. We acknowledge that discrete demographic taxonomies are inherently reductive and that demographic classifiers carry misuse risks. All artifacts are released under licensing terms that restrict usage to research purposes, and annotated datasets comply with relevant privacy and data-protection regulations.

---

## 🙏 Acknowledgements

This work was supported by the National Natural Science Foundation of China (62132008, U22B2030, 62472218) and the Natural Science Foundation of Jiangsu Province (BK20220075).

---

**License**: [Apache 2.0](LICENSE). All classifier models and annotated datasets are restricted to **research purposes only**.
