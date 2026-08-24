# Rock-Paper-Scissors Image Classifier

PyTorch와 Vision Transformer를 사용한 가위바위보 이미지 분류 프로젝트

**실무형 모듈화 구조 | YAML 기반 실험 관리**

---

## 🎯 프로젝트 특징

- **모듈화 설계**: Registry Pattern으로 데이터셋/모델 쉽게 추가
- **YAML 설정**: 코드 수정 없이 실험 관리
- **재현성 보장**: Config/Results 자동 저장
- **확장 가능**: 새로운 모델, 데이터셋 쉽게 통합

---

## 📁 프로젝트 구조

```
AI/
├── configs/                    # 실험 설정 파일
│   ├── datasets/              # 데이터셋 설정
│   │   ├── rps_dataset1.yaml
│   │   └── rps_dataset2.yaml
│   ├── models/                # 모델 설정
│   │   ├── vit_feature_extractor.yaml
│   │   ├── vit_finetune.yaml
│   │   └── resnet50_feature_extractor.yaml
│   ├── experiments/           # 실험 조합
│   │   ├── vit_rps_quick.yaml    # 빠른 테스트 (10 epochs)
│   │   └── vit_rps_full.yaml     # 전체 학습 (300 epochs)
│   └── download/              # 다운로드 설정
│       ├── roboflow.yaml      # Roboflow 설정 (gitignore됨)
│       └── roboflow.example.yaml
│
├── data/                      # 데이터셋 저장 위치 (gitignore됨)
│   └── Rock-Paper-Scissor--2/ # Roboflow 데이터셋
│       ├── train/
│       ├── valid/
│       └── test/
│
├── src/                       # 핵심 모듈
│   ├── datasets/             # Dataset Registry
│   ├── models/               # Model Registry
│   └── trainers/             # Training Logic
│
├── scripts/                   # 유틸리티 스크립트
│   ├── download_roboflow.py  # 데이터셋 다운로드
│   └── extract_dataset.py    # 데이터셋 압축 해제
│
├── main.py                    # 메인 학습 스크립트
├── model.py                   # Legacy 모델 정의
└── trainer.py                 # Legacy Trainer
```

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성 및 활성화
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 패키지 설치
pip install torch torchvision matplotlib pillow numpy pyyaml
```

### 2. 데이터셋 다운로드

**방법 1: YAML Config 사용 (권장)**
```bash
# 1. Config 파일 복사 및 수정
cp configs/download/roboflow.example.yaml configs/download/roboflow.yaml
# roboflow.yaml 파일을 열어서 workspace, project, version 설정

# 2. 환경변수로 API key 설정 (보안 권장)
export ROBOFLOW_API_KEY=your_api_key_here  # Linux/Mac
set ROBOFLOW_API_KEY=your_api_key_here     # Windows

# 3. 다운로드 실행
pip install roboflow
python scripts/download_roboflow.py --config configs/download/roboflow.yaml
```

**방법 2: 커맨드라인 사용**
```bash
python scripts/download_roboflow.py \
  --workspace cpecgm3 \
  --project rock-paper-scissor-p13xv \
  --version 2 \
  --location .
```

**방법 3: 수동 다운로드**
- Roboflow에서 직접 다운로드 후 `data/` 폴더에 배치
- 예: `data/Rock-Paper-Scissor--2/train/`, `data/Rock-Paper-Scissor--2/valid/`, `data/Rock-Paper-Scissor--2/test/`

### 3. 학습 실행

```bash
# 빠른 테스트 (10 epochs)
python main.py --config configs/experiments/vit_rps_quick.yaml

# 전체 학습 (300 epochs, ViT 논문 기반)
python main.py --config configs/experiments/vit_rps_full.yaml
```

---

## 💡 핵심 개념

### 1. Registry Pattern

새로운 데이터셋/모델을 쉽게 추가할 수 있는 구조

```python
# src/datasets/__init__.py
DATASET_REGISTRY = {
    'rps': RockPaperScissorsDataModule,
    # 새 데이터셋 여기에 추가
}

# src/models/__init__.py
MODEL_REGISTRY = {
    'vit_b16': ViTB16,
    'resnet50': ResNet50,
    # 새 모델 여기에 추가
}
```

### 2. YAML 기반 실험 관리

**실험 설정 = 데이터셋 + 모델 + 학습 파라미터**

```yaml
# configs/experiments/my_experiment.yaml
experiment:
  name: 'my_exp'
  description: 'Custom experiment'
  seed: 42

dataset_config: 'configs/datasets/rps_dataset1.yaml'
model_config: 'configs/models/vit_feature_extractor.yaml'

training:
  epochs: 50
  learning_rate: 0.001
  optimizer:
    type: 'adam'
    weight_decay: 0.1
  scheduler:
    use: true
    type: 'cosine'
  loss:
    type: 'crossentropy'
```

### 3. 모듈화된 구성 요소

- **Dataset**: `src/datasets/` - 데이터 로딩 및 전처리
- **Model**: `src/models/` - ViT, ResNet50 등 모델 정의
- **Trainer**: `src/trainers/` - 학습/검증/테스트 로직

---

## 📝 사용 가이드

### 새로운 데이터셋 추가

**방법 1: YAML만 수정 (기존 구조 사용)**
```yaml
# configs/datasets/my_dataset.yaml
type: 'rps'
data_root: 'data/my_dataset'  # data/ 폴더에 저장
batch_size: 64
image_size: [224, 224]
num_workers: 4
```

**방법 2: 새로운 Dataset 클래스 추가**
```python
# src/datasets/__init__.py

class MyCustomDataset(BaseDataModule):
    def setup(self):
        # 커스텀 로딩 로직
        pass

DATASET_REGISTRY['my_custom'] = MyCustomDataset
```

### 새로운 모델 추가

```python
# src/models/__init__.py

class MyModel(nn.Module):
    def __init__(self, num_classes=3, **kwargs):
        super().__init__()
        # 모델 정의

    def forward(self, x):
        return x

MODEL_REGISTRY['my_model'] = MyModel
```

```yaml
# configs/models/my_model.yaml
type: 'my_model'
num_classes: 3
pretrained: true
```

### 하이퍼파라미터 조정

기존 실험 config 복사 후 수정:

```bash
cp configs/experiments/vit_rps_quick.yaml configs/experiments/my_exp.yaml
# my_exp.yaml 편집
python main.py --config configs/experiments/my_exp.yaml
```

---

## 📊 학습 결과

학습 완료 후 `outputs/{experiment_name}/` 폴더에 자동 저장:

```
outputs/vit_rps_quick/
├── model.pth              # 학습된 모델 가중치
├── training_history.png   # Loss/Accuracy 그래프
├── config.yaml            # 재현을 위한 설정 파일
└── results.yaml           # 최종 성능 지표
```

**results.yaml 예시:**
```yaml
test_accuracy: 99.77
test_loss: 0.0234
best_val_accuracy: 99.85
final_train_accuracy: 99.92
```

---

## 🎨 데이터셋 예시

<div align="center">

| 바위 (Rock) | 보 (Paper) | 가위 (Scissors) |
|:---:|:---:|:---:|
| ![Rock](assets/0bioBZYFCXqJIulm_png.rf.1b4aac018e47af8accf0928ec9bc6fc1.jpg) | ![Paper](assets/0cb6cVL8pkfi4wF6_png.rf.3aa17e9337fe612b1142e5b26ea48d12.jpg) | ![Scissors](assets/0CSaM2vL2cWX6Cay_png.rf.eda063b787e4ab94f65b3b6e3d2efcb3.jpg) |

</div>

- **녹색 크로마키 배경**: 손 영역 분리 용이
- **다양한 각도**: 여러 손 모양 학습
- **224x224 해상도**: ViT/ResNet 입력 크기
- **Data Augmentation**: 랜덤 플립, 회전 등

---

## ⚙️ 설정 옵션

### Optimizer

```yaml
optimizer:
  type: 'adam'      # adam, adamw, sgd
  weight_decay: 0.1
  momentum: 0.9     # SGD only
```

### Learning Rate Scheduler

```yaml
scheduler:
  use: true
  type: 'cosine'    # step, cosine, reduce
  step_size: 300    # CosineAnnealingLR: T_max
  gamma: 0.1        # StepLR: decay factor
```

### Loss Function

```yaml
loss:
  type: 'crossentropy'      # crossentropy, focal, label_smoothing
  label_smoothing: 0.1      # label_smoothing type에서만 사용
```

---

## 🏗️ 모델 아키텍처

### Vision Transformer (ViT-B/16)
- **Patch Size**: 16x16
- **Hidden Dim**: 768
- **Attention Heads**: 12
- **Transformer Layers**: 12
- **Parameters**: ~86M

### ResNet50
- **Layers**: 50 (Conv + Residual Blocks)
- **Parameters**: ~25M
- **Pretrained**: ImageNet

---

## 🔬 실험 관리 Best Practices

1. **실험명 규칙**: `{model}_{dataset}_{특징}`
   - 예: `vit_rps_quick`, `resnet50_aug_heavy`

2. **Config 버전 관리**: Git으로 YAML 파일 추적

3. **결과 비교**: `outputs/*/results.yaml` 비교

4. **재현성**: 동일한 config로 재실행 → 동일한 결과

---

## 📦 의존성

```txt
torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.5.0
pillow>=9.0.0
numpy>=1.21.0
pyyaml>=6.0
```

---

## 🔧 문제 해결

**Q: CUDA out of memory**
```yaml
dataset:
  batch_size: 16  # 32에서 16으로 줄이기
```

**Q: 데이터셋 경로 오류**
```yaml
dataset:
  data_root: 'data/Rock-Paper-Scissor--2'  # data/ 폴더 확인
```

**Q: 모델이 수렴하지 않음**
```yaml
training:
  learning_rate: 0.0001  # Learning rate 낮추기
  optimizer:
    type: 'adamw'        # Optimizer 변경
```

---

## 💻 고급 사용법

### 커스텀 Transform 추가

```python
# src/datasets/__init__.py

class RockPaperScissorsDataModule(BaseDataModule):
    def _get_train_transforms(self):
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2),  # 추가
            transforms.ToTensor()
        ])
```

### 다중 GPU 학습

```python
# src/trainers/base.py
model = nn.DataParallel(model)  # 추가
```

---

## 📚 참고 자료

- [ViT Paper (ICLR 2021)](https://arxiv.org/abs/2010.11929)
- [PyTorch Vision Transformer](https://pytorch.org/vision/stable/models/vision_transformer.html)
- [Roboflow Dataset](https://universe.roboflow.com/)

---

## 🤝 기여

새로운 데이터셋, 모델, 실험을 자유롭게 추가하세요!

1. Fork the repository
2. Create your feature branch
3. Add your dataset/model to registry
4. Create config YAML
5. Submit a pull request

---

## 📄 라이선스

Educational Purpose

---

**Made for AI Practitioners** 🚀
