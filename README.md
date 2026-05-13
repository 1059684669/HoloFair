# HoloFair
<div align="center">

# HoloFair

### Unified T2I Fairness Evaluation and Fair-GRPO Debiasing

[![Conference](https://img.shields.io/badge/ICML-2026-blue)](https://icml.cc/Conferences/2026)
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

- **[2026.xx]** 🎉 Paper accepted at **ICML 2026**!
- **[2026.xx]** Code, models, and datasets released.

## 💡 Highlights

<table>
<tr>
<td width="33%" align="center">
<b>📊 MGBI Metric</b><br>
Jointly measures intrinsic diversity and context-robust conditional diversity across gender, age, and race.
</td>
<td width="33%" align="center">
<b>🔍 HoloFair Benchmark</b><br>
End-to-end evaluation of 8 mainstream T2I models with prompt sets, RBD dataset, and SpaFreq classifiers.
</td>
<td width="33%" align="center">
<b>⚡ Fair-GRPO</b><br>
RL-based debiasing that improves MGBI by 29.9% on SD3.5M and 20.2% on SD1.5 without quality loss.
</td>
</tr>
</table>

## 📋 Main Results

<details open>
<summary><b>Fairness Benchmark (8 T2I Models)</b></summary>

| Type | Model | ID ↑ | CA₀.₁ ↑ | CA_mean ↑ | MGBI ↑ |
|:-----|:------|:----:|:-------:|:---------:|:------:|
| Gen-only | SDXL | 0.8186 | 0.2865 | 0.4313 | 0.4843 |
| Gen-only | SD3.5-Large | 0.7480 | 0.3693 | 0.5456 | 0.5255 |
| Gen-only | Flux1-dev | 0.6858 | **0.6702** | **0.6945** | **0.6780** |
| Gen-only | SANA-1.5 | 0.7820 | 0.3821 | 0.5794 | 0.5466 |
| Unified | Show-o | 0.7005 | 0.6013 | 0.6646 | 0.6490 |
| Unified | Harmon | 0.5320 | 0.4661 | 0.5042 | 0.4979 |
| Unified | Bagel | 0.6152 | 0.5004 | 0.5830 | 0.5549 |
| Unified | Blip3-o | 0.4030 | 0.1856 | 0.3370 | 0.2735 |

</details>

<details open>
<summary><b>Debiasing Results (Cross-Architecture)</b></summary>

| Method | MGBI ↑ | CLIP-Score ↑ | Pickscore ↑ | FID ↓ |
|:-------|:------:|:-----------:|:----------:|:-----:|
| SD3.5M baseline | 0.5211 | 0.2288 | 0.2122 | 143.26 |
| **SD3.5M + Fair-GRPO** | **0.6772** | **0.2317** | 0.2118 | **135.09** |
| SD1.5 baseline | 0.6554 | 0.2197 | 0.2079 | 165.37 |
| **SD1.5 + Fair-GRPO** | **0.7881** | **0.2237** | 0.2071 | **134.51** |

</details>

## 🛠️ Installation

```bash
git clone https://github.com/your-username/HoloFair.git
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
│   └── models/spafreq.py          # Dual-stream architecture
│
├── fair_grpo/                     # Fair-GRPO Debiasing
│   ├── train_sd3.py               # SD3.5M training
│   ├── train_sd15.py              # SD1.5 training
│   ├── reward/                    # Multi-attribute reward
│   ├── diffusers_patch/           # Pipeline & SDE patches
│   └── configs/                   # Training configurations
│
└── data/                          # Dataset utilities
```

## 🚀 Usage

### Evaluate a T2I Model

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
    --config fair_grpo/configs/sd3_fairness.py:my_fairness_sd3

# SD1.5 (UNet architecture)
accelerate launch fair_grpo/train_sd15.py \
    --config fair_grpo/configs/sd15_fairness.py:my_fairness_sd15
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
@inproceedings{author2026holofair,
  title     = {HoloFair: Unified T2I Fairness Evaluation and Fair-GRPO Debiasing},
  author    = {},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

## 📜 License

This project is released under the [Apache 2.0 License](LICENSE). All classifier models and annotated datasets are restricted to **research purposes only**.

## 🙏 Acknowledgements

This work was supported by the National Natural Science Foundation of China (62132008, U22B2030, 62472218) and the Natural Science Foundation of Jiangsu Province (BK20220075).
