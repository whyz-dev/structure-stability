# Physics-aware dual-view solution package

## Files
- `eda_report.md` : 샘플에서 실제로 확인한 EDA 결과와 최종 전략
- `full_physics_solution.py` : motion target 추출 + design holdout + pooled grouped CV + submission 생성 코드
- `checkerboard_rectification.py` : 체커보드 바닥을 이용한 top-view 회전 정규화
- `geometry_reasoning.py` : support polygon / collapse margin 근사 기하 피처 추출
- `run_colab_oneclick.py` : Colab에서 Drive zip을 로컬로 복사/압축해제 후 전체 파이프라인 실행
- `run_colab_oneclick.sh` : Colab 원클릭 셸 진입점
- `README_COLAB.md` : Colab 실행 가이드
- `train.command` : macOS에서 motion 추출 + CV 학습 실행
- `infer.command` : macOS에서 학습 결과로 submission 생성
- `full_pipeline.command` : macOS에서 motion 추출 + 학습 + submission 생성까지 한 번에 실행

## 현재 기본 데이터셋 경로
- 기본 자동 감지 대상: `/Users/mgo/Downloads/open (7) 2`
- 또는 환경변수로 지정:
```bash
export PHYSICS_DATA_ROOT="/원하는/데이터셋/경로"
```

## macOS 원클릭 실행

Finder에서 더블클릭해도 되고, 터미널에서는 아래처럼 실행하면 됩니다.
최초 1회는 `.venv` 생성과 패키지 설치 때문에 시간이 조금 걸릴 수 있습니다.

```bash
cd /Users/mgo/Downloads/physics_solution
./train.command
./infer.command
./full_pipeline.command
```

## Colab 원클릭 실행
- 프로젝트 코드는 데이터와 분리해서 올리세요.
- `make_colab_bundle.sh` 로 코드만 담긴 가벼운 zip을 만들 수 있습니다.
- 번들 zip 안에는 [PhysicsSolution_Colab_OneClick.ipynb](/Users/mgo/Downloads/physics_solution/PhysicsSolution_Colab_OneClick.ipynb) 가 포함됩니다.
- 데이터 zip 기본 경로: `/content/drive/MyDrive/open (7).zip`
- Colab에서는 zip을 먼저 로컬 `/content/physics_solution_runtime` 아래로 복사한 뒤 압축 해제하고 학습합니다.
- 진행 상태는 Colab 출력창에서 `extract-motion`, `train 1/12`, `valid 1/12` 같은 실시간 진행바로 보입니다.
- 실행법은 [README_COLAB.md](/Users/mgo/Downloads/physics_solution/README_COLAB.md) 참고

## 체커보드 정규화 판단
- `top view` 체커보드 기반 회전 정규화는 효과가 있어서 기본 `ON` 입니다.
- full homography / perspective rectification은 자동화 신뢰도가 낮아 기본 파이프라인에는 넣지 않았습니다.
- 끄고 싶으면 CLI에 `--no-checkerboard-top-normalize` 를 붙이면 됩니다.

## 학습 로그 기준
- 반드시 `DEV OOF LOGLOSS` 를 확인하세요.
- 목표 범위: `0.015 ~ 0.03`
- `0.05` 이상이면 설계 문제로 봐야 합니다.
- `0.0001` 이하이면 dev 과적합 의심입니다.
- `train-design` 과 `cv-train` 실행 후 자동으로 `dev_logloss_report.json` 또는 `dev_oof_logloss_report.json` 이 저장됩니다.

## 현재 모델 구조
- 기본 메인라인은 검증된 `two-view encoder + checkerboard top rotation + motion pseudo target` 입니다.
- `geometry reasoning` 브랜치는 코드에 들어있지만 기본값은 `OFF` 입니다.
- 추가되는 기하 신호:
  - top support polygon 근사 폭/면적/중심
  - front slenderness / base width / top width / tilt / top-heaviness
  - collapse margin proxy
- 실험하고 싶으면 `--enable-geometry-reasoning` 를 붙여 켤 수 있습니다.

## CLI workflow

### 1) train video에서 motion target 추출
```bash
python full_physics_solution.py extract-motion
```

### 2) 설계 고정 전: train -> dev holdout
```bash
python3 full_physics_solution.py train-design \
  --out-dir runs/design \
  --backbone efficientnet_v2_s \
  --pretrained \
  --image-size 320 \
  --batch-size 128 \
  --epochs 12 \
  --num-workers 0 \
  --data-root ../../data
```

### 3) 설계 고정 후: train+dev pooled grouped CV
```bash
python full_physics_solution.py cv-train \
  --out-dir runs/final \
  --backbone efficientnet_v2_s \
  --pretrained \
  --image-size 320 \
  --batch-size 8 \
  --epochs 12 \
  --num-folds 5
```

### 4) fold ensemble submission
```bash
python full_physics_solution.py make-submission \
  --out-dir runs/final \
  --run-dir runs/final \
  --backbone efficientnet_v2_s \
  --image-size 320 \
  --batch-size 8 \
  --tta-passes 4
```

### 5) 전체 파이프라인 한 번에 실행
```bash
python full_physics_solution.py full-run \
  --out-dir runs/final \
  --backbone efficientnet_v2_s \
  --pretrained \
  --image-size 320 \
  --batch-size 8 \
  --epochs 12 \
  --num-folds 5
```

## Notes
- `train-design` 단계에서는 **dev를 validation으로만** 쓰고, 구조/손실/증강이 고정되기 전에는 dev를 train에 섞지 않는 쪽을 권장합니다.
- `cv-train` 단계는 **설계가 확정된 뒤**에만 돌리세요.
- logloss가 주목적이므로, fold별 temperature scaling과 soft target이 포함되어 있습니다.
- checkerboard 기반 homography rectification은 유망하지만, 이 패키지의 메인라인에는 넣지 않았습니다. 그건 별도 실험 브랜치로 다루는 편이 안전합니다.
- geometry reasoning 브랜치는 현재 기본 `OFF` 입니다. 검증 없이 메인라인에 섞지 마세요.
