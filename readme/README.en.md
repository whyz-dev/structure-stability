# Structure Stability Challenge

[한국어](README.ko.md) | [English](README.en.md)

Solution code for Dacon's Structure Stability Prediction competition (66th/484 Teams) <br>

## Competition Overview
Participants must build an AI model that uses two-view images of a structure to predict the probability that the structure becomes unstable within the first 10 seconds of simulation, and the probability that it remains stable.

<details>
<summary><strong>[1] Data Labels</strong></summary>

Each sample label is defined from the physics simulation result.

| Label | Definition |
|------|------|
| Stable | The structure shows no meaningful movement or deformation during the first 10 seconds of simulation |
| Unstable | The cumulative movement distance reaches at least 1.5 cm within 10 seconds, or structural collapse occurs |

Some samples are designed as boundary cases where stability is difficult to determine from appearance alone, requiring **precise visual physics reasoning**.

To support reasoning about physical changes over time, **the training data includes 10-second simulation videos**.

</details>

<details>
<summary><strong>[2] Dataset Composition and Training Strategy</strong></summary>

This competition evaluates <strong>how robustly a model trained in a controlled environment works under highly variable real-world conditions</strong>.

| Data | Samples | Environment | Purpose |
|------|---------|-------------|---------|
| Train | 1,000 | Laboratory environment with fixed lighting and camera coordinates | Learn basic physical laws and structural features |
| Dev | 100 | Same randomized lighting and camera-coordinate setting as the evaluation environment | Validate model adaptability to the evaluation environment |
| Test | 1,000 | Same randomized environment setting as the dev data | Determine the final ranking |

Participants may use both train and dev data for model training. However, they should avoid **overfitting to the fixed environment of the training data**.

The core of this competition is to build robust training strategies against evaluation-environment variability through data augmentation, external data collection, and generalizable model design that can infer universal physical causality.

</details>

## Key Contributions

```
1. Designed a MultiView Bidirectional Cross Attention Model, improving logloss by 48.2% and accuracy by 3.13% over the existing FeatureFusion model
2. Extracted video feature vectors with VideoMAE and trained with Distillation Regularization, improving logloss by 86.3% and accuracy by 8% over the baseline
3. Performed HyperParameter Search over BackBone, Epochs, Batch Size, and related settings, improving logloss by 91.2% and accuracy by 6% over the worst trial
```
<br>

<img src="../pdf/preview/structure-stability-competition-1.png" alt="Structure Stability Physics Reasoning AI Competition page 1" width="100%">

<img src="../pdf/preview/structure-stability-competition-2.png" alt="Structure Stability Physics Reasoning AI Competition page 2" width="100%">

[View Original PDF](<../pdf/구조물 안정성 물리 추론 AI 경진대회.pdf>)

## Getting Started

### 1. Set Up a Virtual Environment
```bash
git clone https://github.com/whyz-dev/structure-stability.git
cd structure-stability
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Register the Virtual Environment Kernel
```bash
python3 -m ipykernel install --user --name .venv --display-name stability
```

Use this virtual environment as the kernel when running the notebooks.

## Repository Map

| Path | Role |
|------|------|
| `data/` | Train/dev/test metadata and contest data |
| `src/` | Shared preprocessing, augmentation, model, reproducibility, and output-path utilities |
| `src/models/` | MultiView Feature Fusion and Cross Attention model implementations |
| `notebooks/eda/` | EDA, preprocessing analysis, and feature-selection notebooks |
| `notebooks/train/` | Baseline, regularization, distillation, and ablation training notebooks |
| `notebooks/test/` | Backbone tests, seed sweeps, ensemble, and submission-analysis notebooks |
| `code/` | Model-comparison, regularization, distillation, and backbone-selection experiments |
| `code/huggingface/` | Hugging Face Hub upload/load utilities |
| `tools/` | Experiment analysis, ensembling, and preprocessing-ablation scripts |
| `tools/physics_solution/` | Physics-aware pipeline and Colab workflow |
| `tools/simulator/` | Structure generation and rendering experiments |
| `outputs/submissions/` | Generated submission CSV artifacts |
| `outputs/model_comparison/` | Model-comparison results, histories, and submission artifacts |
| `outputs/eda_preprocessing/` | EDA/preprocessing analysis artifacts |
| `pdf/` | Contest PDF and README preview images |
| `readme/` | Korean and English README documents |
