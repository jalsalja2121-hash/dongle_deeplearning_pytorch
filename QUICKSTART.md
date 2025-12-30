# Quick Start Guide

## 빠른 시작

### 1. 기본 실행 (Vision Transformer)

```bash
# 가상환경 활성화
.\venv\Scripts\activate

# 학습 실행
python train.py
```

기본적으로 `configs/experiments/exp_vit_rps.yaml` 설정을 사용합니다.

### 2. ResNet50 사용

```bash
python train.py --config configs/experiments/exp_resnet_rps.yaml
```

### 3. 하이퍼파라미터 변경

원하는 설정만 변경하려면 새 실험 파일을 만드세요:

**configs/experiments/my_exp.yaml**:
```yaml
experiment:
  name: my_quick_test
  description: "Quick test with 10 epochs"

model_config: configs/models/vit_b16.yaml
dataset_config: configs/datasets/rock_paper_scissor.yaml

overrides:
  training:
    epochs: 10  # 빠른 테스트를 위해 10 에포크만
    batch_size: 32

output:
  save_dir: outputs/my_quick_test
  model_name: model.pth
  plot_name: history.png
```

실행:
```bash
python train.py --config configs/experiments/my_exp.yaml
```

## 자주 사용하는 설정 변경

### Learning Rate 변경

`configs/experiments/` 아래 실험 파일에서:

```yaml
overrides:
  training:
    learning_rate: 1e-4  # 원하는 값으로 변경
```

### Optimizer 변경

```yaml
overrides:
  training:
    optimizer:
      type: adamw  # adam, adamw, sgd 중 선택
```

### Batch Size 변경

```yaml
overrides:
  training:
    batch_size: 64  # 원하는 크기로
```

### Epochs 변경

```yaml
overrides:
  training:
    epochs: 50  # 원하는 에포크 수로
```

## 결과 확인

학습이 완료되면 `outputs/{experiment_name}/` 폴더에:
- 모델 가중치 (`.pth` 파일)
- 학습 그래프 (`.png` 파일)

가 저장됩니다.

## 사용 가능한 모델

- `MyVit_b_16`: Vision Transformer B/16
- `MyResNet50`: ResNet50

설정 파일: `configs/models/`

## 사용 가능한 데이터셋

- `RockPaperScissorsDataset`: 가위바위보 데이터셋

설정 파일: `configs/datasets/`

## 새 데이터셋 추가 (간단 버전)

1. `data/` 폴더에 데이터셋 복사 (train/valid/test 폴더 구조)
2. `configs/datasets/my_dataset.yaml` 생성
3. `datasets/my_dataset.py` 작성 (rock_paper_scissor.py 복사 후 수정)
4. `datasets/__init__.py`에 import 추가

자세한 내용은 README_new.md 참조.
