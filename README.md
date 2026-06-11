# Structure Stability Challenge

[한국어](readme/README.ko.md) | [English](readme/README.en.md)

데이콘이 주관한 구조물 안정성 예측 경진대회의 솔루션 코드입니다 (66th/484 Teams) <br>

## 대회 설명
참가자는 구조물에 대해 제공되는 2가지 시점의 이미지 데이터를 입력으로 활용하여, 시뮬레이션 시작 10초 동안 구조물이 불안정(unstable) 상태로 전환될 확률과 안정(stable) 상태를 유지할 확률을 예측하는 AI 모델을 개발해야 합니다.

<details>
<summary><strong>[1] 데이터 라벨</strong></summary>

샘플의 라벨(Label)은 물리 시뮬레이션 결과를 기반으로 다음과 같이 정의됩니다.

| 라벨 | 정의 |
|------|------|
| 안정(stable) | 시뮬레이션 시작 후 10초 동안 구조물에 의미 있는 이동이나 변형이 발생하지 않은 경우 |
| 불안정(unstable) | 시뮬레이션 시작 후 10초 이내에 누적 이동 거리가 1.5cm 이상 발생하거나 구조적 붕괴가 나타난 경우 |

일부 샘플들은 외형만으로 안정 여부를 구분하기 어려운 경계(Boundary) 특성을 가지도록 구성되어 있어, **시각 정보 기반의 정밀한 물리 추론**이 요구됩니다. 

이에 따라 구조물의 물리 변화 과정을 참고할 수 있도록, **학습 데이터(train)에는 10초 분량의 시뮬레이션 영상**이 함께 제공됩니다.

</details>

<details>
<summary><strong>[2] 데이터셋 구성 및 학습 전략</strong></summary>

본 대회는 정제된 환경에서 학습한 모델이 <strong>변동성이 큰 실제 환경에서 얼마나 강건하게 작동하는지</strong> 평가합니다.

| 데이터 | 샘플 수 | 환경 | 활용 목적 |
|--------|---------|------|-----------|
| 학습 데이터(train) | 1,000개 | 광원 및 카메라 좌표가 고정된 실험실 환경 | 기본적인 물리 법칙과 구조적 특징 학습 |
| 개발 데이터(dev) | 100개 | 광원 및 카메라 좌표가 무작위로 변동하는 실제 평가 환경과 동일한 설정 | 평가 환경에 대한 모델의 적응력 검증 |
| 평가 데이터(test) | 1,000개 | 개발 데이터와 동일한 무작위 환경 설정 | 최종 순위 결정을 위한 평가 |

참가자는 학습 데이터(train)와 개발 데이터(dev)를 모델 학습에 모두 활용할 수 있습니다. 다만, **고정된 환경의 학습 데이터에만 오버피팅(Overfitting)되지 않도록** 주의해야 합니다. 

실제 평가 환경의 변동성에 대비하여 데이터 증강(Augmentation), 외부 데이터 수집, 그리고 강건한 학습 전략을 수립하는 것이 이번 대회의 핵심이며, 특히 보편적인 물리적 인과관계를 추론할 수 있는 모델 설계가 요구됩니다.

</details>

## Key Contributions
```
1. front/top view 이미지 전처리
2. 
3. 
4.
5.
```

## 문서
- [시작하기](https://jungseong.github.io/contests/structure-stability/#getting-started)
- [주요 워크플로우](https://jungseong.github.io/contests/structure-stability/#core-workflows)

## 시작하기

### 1. 가상환경 설정
```bash
git clone https://github.com/whyz-dev/structure-stability.git
cd structure-stability
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 가상환경 커널 등록
```bash
python -m ipykernel install --user --name .venv --display-name stability
```
위 가상환경을 커널로 사용하여 각 노트북을 실행할 수 있습니다.

### 3. 최종 파이프라인 실행:
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
