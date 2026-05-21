<div align="center">

# HoloFair: Unified T2I Fairness Evaluation and Fair-GRPO Debiasing

[![Conference](https://img.shields.io/badge/ICML-2026-blue)](https://icml.cc/virtual/2026/poster/62706)
[![Paper](https://img.shields.io/badge/arXiv-2026.xxxxx-b31b1b)](https://arxiv.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/🤗-Models-orange)](https://huggingface.co/)

*A comprehensive benchmark and debiasing framework for demographic fairness in Text-to-Image models.*

[Paper]() | [Project Page]() | [Dataset]() | [Models]()

<img src="assets/overview.png" width="85%">

</div>

---

## 🔥 News

- **[2026.5.1]** 🎉 Paper accepted at **ICML 2026**!
- **[2026.6]** Code, models, and datasets will be released. Coming soon.

## 💡 Overview

Text-to-Image (T2I) models often perpetuate societal biases — even a neutral prompt like *"a photo of a person"* produces heavily skewed demographics. Worse, these biases intensify under semantically loaded contexts (e.g., *"a professional person"*), yet existing auditing methods only test default distributions.

**HoloFair** addresses this dual challenge with three contributions:

- **MGBI Metric** — Multi-attribute, Group-wise Bias Index that jointly measures intrinsic diversity (ID) and context-robust conditional diversity (CA<sub>q</sub>) via normalized Shannon entropy and geometric mean aggregation.
- **HoloFair Benchmark** — An end-to-end evaluation framework comprising prompt sets (Gen/Eval/Train), the RBD dataset (119K images from FairFace, UTKFace, in-the-wild, and AI-generated sources), and SpaFreq dual-stream classifiers (DINOv2 + wavelet frequency features). We evaluate 8 mainstream T2I models across both generation-only (SDXL, SD3.5-Large, Flux1-dev, SANA-1.5) and unified multimodal architectures (Show-o, Harmon, Bagel, Blip3-o).
- **Fair-GRPO** — A reinforcement-learning-based debiasing method using a multi-attribute per-prompt reward function with KL-regularized policy optimization. 

## 📋 Main Results

### Fairness Benchmark (8 T2I Models)

| Type | Model | ID ↑ | CA₀.₁ ↑ | CA_mean ↑ | MGBI ↑ |
|:-----|:------|:----:|:-------:|:---------:|:------:|
| Gen-only | SDXL | 0.8186 | 0.2865 | 0.4313 | 0.4843 |
| Gen-only | SD3.5-Large | 0.7480 | 0.3693 | 0.5456 | 0.5255 |
| Gen-only | Flux1-dev | 0.6858 | 0.6702 | 0.6945** | 0.6780 |
| Gen-only | SANA-1.5 | 0.7820 | 0.3821 | 0.5794 | 0.5466 |
| Unified | Show-o | 0.7005 | 0.6013 | 0.6646 | 0.6490 |
| Unified | Harmon | 0.5320 | 0.4661 | 0.5042 | 0.4979 |
| Unified | Bagel | 0.6152 | 0.5004 | 0.5830 | 0.5549 |
| Unified | Blip3-o | 0.4030 | 0.1856 | 0.3370 | 0.2735 |

> **Key finding:** SDXL achieves the highest ID (0.82) but near-lowest CA₀.₁ (0.29). High default diversity ≠ conditional robustness. Evaluations limited to default distributions would erroneously rank SDXL as the fairest model.

### Debiasing Results (Cross-Architecture)

| Method | ID ↑ | CA₀.₁ ↑ | MGBI ↑ | CLIP ↑ | FID ↓ |
|:-------|:----:|:-------:|:------:|:------:|:-----:|
| SD1.5 baseline | 0.6708 | 0.6404 | 0.6554 | 0.2197 | 165.37 |
| EFA | 0.7217 | 0.6953 | 0.7084 | 0.2211 | 139.97 |
| **Fair-GRPO (SD1.5)** | **0.8591** | **0.7230** | **0.7881** | **0.2237** | **134.51** |
| SD3.5M baseline | 0.7316 | 0.3711 | 0.5211 | 0.2288 | 143.26 |
| Balancing Act | 0.7460 | 0.4486 | 0.5785 | 0.2311 | 155.60 |
| **Fair-GRPO (SD3.5M)** | **0.8221** | **0.5579** | **0.6772** | **0.2317** | **135.09** |

> Fair-GRPO achieves SOTA fairness on both UNet (SD1.5) and MMDiT (SD3.5M) architectures, with the best FID scores and stable CLIP-Score / Pickscore.

## 🛠️ Installation

```bash
git clone https://github.com/RuyiChen11/HoloFair.git
cd HoloFair
conda create -n holofair python=3.10 -y
conda activate holofair
pip install -r requirements.txt
```

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

## 🚀 Usage

### Evaluate a T2I Model with MGBI

```bash
python benchmark/evaluate.py \
    --model_path <path_to_model> \
    --prompt_set benchmark/prompts/eval_set.json \
    --classifier_weights classifiers/weights/ \
    --output_dir results/
```

### Train Fair-GRPO

```bash
# SD3.5M (MMDiT architecture)
accelerate launch fair_grpo/train_sd3.py \
    --config fair_grpo/configs/sd3_fairness.py

# SD1.5 (UNet architecture)
accelerate launch fair_grpo/train_sd15.py \
    --config fair_grpo/configs/sd15_fairness.py
```

## 📦 Model Zoo

| Model | Description | Link |
|:------|:------------|:----:|
| SpaFreq-Gender | Gender classifier (2 classes) | [🤗 Download]() |
| SpaFreq-Age | Age classifier (3 classes) | [🤗 Download]() |
| SpaFreq-Race | Race classifier (5 classes) | [🤗 Download]() |
| Fair-GRPO SD3.5M | Debiased LoRA weights | [🤗 Download]() |
| Fair-GRPO SD1.5 | Debiased LoRA weights | [🤗 Download]() |

## 📖 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{chen2026holofair,
  title     = {HoloFair: Unified T2I Fairness Evaluation and Fair-GRPO Debiasing},
  author    = {Chen, Ruyi and Xu, Xiaogang and Zhang, Chiyu and Wu, Jiafei and Fang, Liming and Zhou, Lu},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

## 📜 License

This project is released under the [Apache 2.0 License](LICENSE). All classifier models and annotated datasets are restricted to **research purposes only**.

## 🙏 Acknowledgements

This work was supported by the National Natural Science Foundation of China (62132008, U22B2030, 62472218) and the Natural Science Foundation of Jiangsu Province (BK20220075).
