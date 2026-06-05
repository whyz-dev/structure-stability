# Structure Stability

[![Documentation](https://img.shields.io/badge/Documentation-GitHub%20Pages-0A66C2)](https://jungseong.github.io/contests/structure-stability/)
[![Dual View Encoder](https://img.shields.io/badge/Dual%20View-Encoder-EE4C2C)](https://jungseong.github.io/contests/structure-stability/#core-workflows)
[![Checkerboard Norm](https://img.shields.io/badge/Checkerboard-Top%20View%20Norm-5C3EE8)](https://jungseong.github.io/contests/structure-stability/#core-workflows)
[![Grouped CV](https://img.shields.io/badge/Grouped%20CV-Fold%20Ensemble-F9AB00)](https://jungseong.github.io/contests/structure-stability/#core-workflows)

[한국어](README.ko.md) | [English](README.en.md)

Physics-aware dual-view computer-vision workspace for structure-stability prediction. The mainline combines front/top-view image modeling, checkerboard-based top-view rotation normalization, motion pseudo targets, grouped cross validation, and fold ensemble submission generation.

## Documentation

- [Live contest documentation](https://jungseong.github.io/contests/structure-stability/)
- [Getting Started](https://jungseong.github.io/contests/structure-stability/#getting-started)
- [Core Workflows](https://jungseong.github.io/contests/structure-stability/#core-workflows)

## Getting Started

```bash
git clone https://github.com/whyz-dev/structure-stability.git
cd structure-stability
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the packaged physics-aware pipeline:

```bash
cd tools/physics_solution
python full_physics_solution.py full-run \
  --out-dir runs/final \
  --backbone efficientnet_v2_s \
  --pretrained \
  --image-size 320 \
  --batch-size 8 \
  --epochs 12 \
  --num-folds 5
```

## Repository Map

| Path | Role |
|------|------|
| `src/` | Shared preprocessing, augmentation, model, reproducibility utilities |
| `notebooks/eda/` | EDA and feature-selection notebooks |
| `notebooks/train/` | Training experiments and ablation notebooks |
| `code/` | Regularization, distillation, and backbone-selection experiments |
| `tools/physics_solution/` | Packaged physics-aware pipeline and Colab workflow |
| `outputs/submissions/` | Generated submission artifacts |
