# 딥러닝 이미지 분류 프레임워크

PyTorch 기반의 모듈화된 이미지 분류 학습 프레임워크입니다. YAML 설정 파일을 통해 다양한 모델과 데이터셋 조합을 쉽게 실험할 수 있습니다.

## 주요 기능

- **모듈화된 구조**: 모델, 데이터셋, 학습 설정을 독립적으로 관리
- **YAML 기반 설정**: 코드 수정 없이 실험 설정 변경 가능
- **모델 레지스트리**: 새로운 모델을 쉽게 추가하고 관리
- **데이터셋 레지스트리**: 다양한 데이터셋을 플러그인 방식으로 추가
- **실험 조합**: 모델과 데이터셋을 자유롭게 조합하여 실험

## 프로젝트 구조

```
dongle_deeplearning_pytorch/
├── configs/                       # YAML 설정 파일
│   ├── models/                   # 모델 설정
│   │   ├── vit_b16.yaml         # Vision Transformer
│   │   └── resnet50.yaml        # ResNet50
│   ├── datasets/                # 데이터셋 설정
│   │   └── rock_paper_scissor.yaml
│   └── experiments/             # 실험 조합
│       └── exp_vit_rps.yaml
├── data/                         # 데이터셋 저장소
│   └── rock_paper_scissor/
├── models/                       # 모델 정의
│   ├── __init__.py
│   ├── registry.py              # 모델 레지스트리
│   ├── vit.py                   # ViT 구현
│   └── resnet.py                # ResNet 구현
├── datasets/                     # 데이터셋 로더
│   ├── __init__.py
│   ├── registry.py              # 데이터셋 레지스트리
│   ├── base.py                  # 베이스 클래스
│   └── rock_paper_scissor.py   # 가위바위보 데이터셋
├── utils/                        # 유틸리티
│   ├── config.py                # YAML 로더
│   └── visualize.py             # 시각화
├── trainer.py                    # 학습 로직
├── train.py                      # 메인 실행 스크립트
└── README.md
```

## 설치

### 1. 가상환경 생성 및 활성화

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. 필수 패키지 설치

```bash
pip install torch torchvision matplotlib pillow numpy pyyaml
```

## 사용 방법

### 1. 기본 실행 (기본 설정 사용)

```bash
python train.py
```

### 2. 특정 실험 설정으로 실행

```bash
python train.py --config configs/experiments/exp_vit_rps.yaml
```

### 3. 새로운 실험 설정 생성

`configs/experiments/my_experiment.yaml`:

```yaml
experiment:
  name: my_custom_experiment
  description: "Custom experiment with specific settings"

# 기존 모델/데이터셋 설정 재사용
model_config: configs/models/resnet50.yaml
dataset_config: configs/datasets/rock_paper_scissor.yaml

# 원하는 설정만 오버라이드
overrides:
  training:
    epochs: 50
    batch_size: 64
    learning_rate: 1e-4

output:
  save_dir: outputs/my_experiment
  model_name: my_model.pth
  plot_name: history.png
```

실행:

```bash
python train.py --config configs/experiments/my_experiment.yaml
```

## 새로운 모델 추가하기

### 1. 모델 클래스 작성

`models/my_model.py`:

```python
import torch.nn as nn
from .registry import register_model

@register_model('MyCustomModel')
class MyCustomModel(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        # 모델 구현
        pass

    def forward(self, x):
        # Forward pass 구현
        return x
```

### 2. 모델 등록

`models/__init__.py`에 추가:

```python
from .my_model import MyCustomModel
```

### 3. 모델 설정 파일 생성

`configs/models/my_model.yaml`:

```yaml
model:
  name: my_custom_model
  type: MyCustomModel
  params:
    num_classes: 3
    pretrained: true

training:
  epochs: 100
  batch_size: 32
  learning_rate: 1e-3
  # ... 기타 설정
```

## 새로운 데이터셋 추가하기

### 1. 데이터셋 클래스 작성

`datasets/my_dataset.py`:

```python
from torchvision import transforms
from .base import BaseDataModule
from .registry import register_dataset

@register_dataset('MyCustomDataset')
class MyCustomDataset(BaseDataModule):
    def __init__(self, data_root='.', batch_size=32, image_size=(224, 224), num_workers=0, num_classes=10):
        super().__init__(data_root, batch_size, image_size, num_workers)
        self.num_classes = num_classes

    def _get_train_transforms(self):
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

    def _get_validation_transforms(self):
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])
```

### 2. 데이터셋 등록

`datasets/__init__.py`에 추가:

```python
from .my_dataset import MyCustomDataset
```

### 3. 데이터셋 설정 파일 생성

`configs/datasets/my_dataset.yaml`:

```yaml
dataset:
  name: my_custom_dataset
  type: MyCustomDataset
  data_root: data/my_custom_dataset
  num_classes: 10
  class_names:
    - class1
    - class2
    # ... 기타 클래스
```

## 실험 조합 예시

### ViT + Rock Paper Scissor

```bash
python train.py --config configs/experiments/exp_vit_rps.yaml
```

### ResNet50 + Rock Paper Scissor

`configs/experiments/exp_resnet_rps.yaml` 생성 후:

```bash
python train.py --config configs/experiments/exp_resnet_rps.yaml
```

## 설정 파일 구조

### 모델 설정 (configs/models/)

```yaml
model:
  name: model_name
  type: ModelClassName
  params:
    num_classes: 3
    feature_extractor: false
    pretrained: true

training:
  epochs: 100
  batch_size: 32
  learning_rate: 1e-3

  optimizer:
    type: adam  # adam, adamw, sgd
    weight_decay: 0.1
    momentum: 0.9

  scheduler:
    use: true
    type: cosine  # step, cosine, reduce
    step_size: 100
    gamma: 0.1

  loss:
    type: crossentropy  # crossentropy, focal, label_smoothing
    label_smoothing: 0.0

data:
  image_size: [224, 224]
  num_workers: 0
```

### 데이터셋 설정 (configs/datasets/)

```yaml
dataset:
  name: dataset_name
  type: DatasetClassName
  data_root: data/dataset_folder
  num_classes: 3
  class_names:
    - class1
    - class2
    - class3
```

### 실험 설정 (configs/experiments/)

```yaml
experiment:
  name: experiment_name
  description: "Experiment description"

model_config: configs/models/model.yaml
dataset_config: configs/datasets/dataset.yaml

overrides:
  training:
    epochs: 50  # 모델 설정 오버라이드

output:
  save_dir: outputs/experiment_name
  model_name: model.pth
  plot_name: training_history.png
```

## 출력 결과

학습이 완료되면 `outputs/` 디렉토리에 다음이 저장됩니다:

- `{model_name}.pth`: 학습된 모델 가중치
- `{plot_name}.png`: 학습/검증 Loss 및 Accuracy 그래프

## 기존 코드 (main.py) 사용

기존 방식으로도 계속 사용 가능합니다:

```bash
python main.py
```

## 라이선스

Educational Purpose
