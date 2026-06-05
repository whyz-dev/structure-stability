# Structure Stability

[![Documentation](https://img.shields.io/badge/Documentation-GitHub%20Pages-0A66C2)](https://jungseong.github.io/contests/structure-stability/)
[![Dual View Encoder](https://img.shields.io/badge/Dual%20View-Encoder-EE4C2C)](https://jungseong.github.io/contests/structure-stability/#core-workflows)
[![Checkerboard Norm](https://img.shields.io/badge/Checkerboard-Top%20View%20Norm-5C3EE8)](https://jungseong.github.io/contests/structure-stability/#core-workflows)
[![Grouped CV](https://img.shields.io/badge/Grouped%20CV-Fold%20Ensemble-F9AB00)](https://jungseong.github.io/contests/structure-stability/#core-workflows)

[한국어](readme/README.ko.md) | [English](readme/README.en.md)

구조물 안정성 예측을 위한 physics-aware dual-view 컴퓨터 비전 워크스페이스입니다. 주요 파이프라인은 front/top view 이미지 모델링, top view 체커보드 기반 회전 정규화, motion pseudo target, grouped cross validation, fold ensemble submission 생성을 결합합니다.

## 문서

- [대회 문서](https://jungseong.github.io/contests/structure-stability/)
- [시작하기](https://jungseong.github.io/contests/structure-stability/#getting-started)
- [주요 워크플로우](https://jungseong.github.io/contests/structure-stability/#core-workflows)
## 시작하기

```bash
git clone https://github.com/whyz-dev/structure-stability.git
cd structure-stability
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

패키지화된 physics-aware 파이프라인 실행:

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

## 저장소 구조

| 경로 | 역할 |
|------|------|
| `src/` | 공통 전처리, 증강, 모델, 재현성 유틸리티 |
| `notebooks/eda/` | EDA 및 feature selection 노트북 |
| `notebooks/train/` | 학습 실험 및 ablation 노트북 |
| `code/` | regularization, distillation, backbone selection 실험 |
| `tools/physics_solution/` | physics-aware 파이프라인과 Colab 워크플로우 |
| `outputs/submissions/` | 생성된 submission 산출물 |
