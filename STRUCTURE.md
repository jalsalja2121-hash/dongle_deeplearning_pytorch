# 프로젝트 구조 설명

## 📋 전체 구조

```
AI/
├── configs/                         # 실험 설정 YAML 파일
│   ├── datasets/                   # 데이터셋 설정
│   ├── models/                     # 모델 설정
│   └── experiments/                # 실험 조합 설정
│
├── src/                            # 핵심 모듈 (모듈화된 코드)
│   ├── datasets/__init__.py       # Dataset Registry + Factory
│   ├── models/__init__.py         # Model Registry + Factory
│   └── trainers/                  # Training 로직
│       ├── __init__.py
│       └── base.py                # RockPaperScissorsTrainer
│
├── scripts/                        # 유틸리티 스크립트
│   ├── download_roboflow.py       # Roboflow 데이터셋 다운로드
│   └── extract_dataset.py         # 압축 해제
│
├── assets/                         # README용 이미지
│
├── main.py                         # 메인 학습 스크립트 (YAML 필수)
├── model.py                        # Legacy 모델 (호환성)
├── trainer.py                      # Legacy Trainer (호환성)
│
└── README.md                       # 프로젝트 문서
```

---

## 🎯 설계 철학

### 1. **모듈화 (Modularity)**
- 각 컴포넌트(데이터셋, 모델, 학습)를 독립적인 모듈로 분리
- 변경 사항이 다른 부분에 영향을 주지 않음

### 2. **Registry Pattern**
- 새로운 데이터셋/모델을 Dictionary에 등록만 하면 사용 가능
- Factory 함수로 YAML 설정만으로 인스턴스 생성

### 3. **YAML 기반 설정**
- 코드와 설정 완전 분리
- 실험 재현성 보장
- 버전 관리 용이

---

## 📁 디렉토리 상세 설명

### `configs/`
실험 설정 파일들을 계층적으로 관리

```
configs/
├── datasets/              # 데이터셋 설정만 분리
│   ├── rps_dataset1.yaml # Rock-Paper-Scissor--2
│   └── rps_dataset2.yaml # 루트 폴더 데이터
│
├── models/                # 모델 설정만 분리
│   ├── vit_feature_extractor.yaml
│   ├── vit_finetune.yaml
│   └── resnet50_feature_extractor.yaml
│
└── experiments/           # 데이터셋 + 모델 + 하이퍼파라미터 조합
    ├── vit_rps_quick.yaml   # 빠른 테스트
    └── vit_rps_full.yaml    # 전체 학습
```

**장점:**
- 데이터셋/모델 재사용 가능
- 같은 모델로 여러 데이터셋 실험 가능
- 실험 조합을 YAML로 명확히 기록

### `src/`
핵심 비즈니스 로직

```
src/
├── datasets/
│   └── __init__.py        # DATASET_REGISTRY + create_dataset()
│
├── models/
│   └── __init__.py        # MODEL_REGISTRY + create_model()
│
└── trainers/
    ├── __init__.py
    └── base.py            # RockPaperScissorsTrainer
```

**Registry Pattern 예시:**

```python
# src/datasets/__init__.py
DATASET_REGISTRY = {
    'rps': RockPaperScissorsDataModule,
}

def create_dataset(config):
    dataset_type = config['type']
    return DATASET_REGISTRY[dataset_type](**config)
```

**사용:**
```python
config = {'type': 'rps', 'data_root': 'data/', ...}
dataset = create_dataset(config)  # 자동으로 적절한 클래스 인스턴스 생성
```

### `scripts/`
일회성 또는 유틸리티 스크립트

- `download_roboflow.py`: Roboflow API로 데이터셋 다운로드
- `extract_dataset.py`: 압축 파일 해제

---

## 🔄 실행 흐름

### 1. 설정 로드
```
main.py --config experiments/vit_rps_quick.yaml
    ↓
load_config()
    ↓
├── experiments/vit_rps_quick.yaml 로드
├── datasets/rps_dataset1.yaml 로드 (참조)
└── models/vit_feature_extractor.yaml 로드 (참조)
```

### 2. 컴포넌트 생성
```
create_dataset(config['dataset'])
    ↓
DATASET_REGISTRY['rps'] → RockPaperScissorsDataModule 인스턴스

create_model(config['model'])
    ↓
MODEL_REGISTRY['vit_b16'] → ViTB16 인스턴스

RockPaperScissorsTrainer(model, ...)
    ↓
Trainer 인스턴스
```

### 3. 학습 실행
```
trainer.fit(train_loader, val_loader, epochs)
    ↓
각 에포크마다:
    - train_epoch()
    - evaluate()
    - scheduler.step() (옵션)
    ↓
history 반환
```

### 4. 결과 저장
```
outputs/{experiment_name}/
├── model.pth           # 모델 가중치
├── training_history.png # 그래프
├── config.yaml         # 사용된 설정
└── results.yaml        # 성능 지표
```

---

## 🔧 확장 방법

### 새 데이터셋 추가

1. **YAML만 수정 (기존 구조 재사용)**
```yaml
# configs/datasets/my_dataset.yaml
type: 'rps'  # 기존 타입 재사용
data_root: 'new_data_path'
```

2. **새 클래스 추가 (완전 커스텀)**
```python
# src/datasets/__init__.py

class MyCustomDataset(BaseDataModule):
    def setup(self):
        # 커스텀 로직
        pass

DATASET_REGISTRY['my_custom'] = MyCustomDataset
```

```yaml
# configs/datasets/my_dataset.yaml
type: 'my_custom'  # 새 타입
```

### 새 모델 추가

```python
# src/models/__init__.py

class EfficientNet(nn.Module):
    def __init__(self, num_classes=3, **kwargs):
        super().__init__()
        # 모델 정의

MODEL_REGISTRY['efficientnet'] = EfficientNet
```

```yaml
# configs/models/efficientnet.yaml
type: 'efficientnet'
num_classes: 3
```

---

## 💡 설계 패턴

### 1. Factory Pattern
```python
def create_dataset(config):
    """설정에 따라 적절한 데이터셋 객체 생성"""
    return DATASET_REGISTRY[config['type']](**config)
```

### 2. Registry Pattern
```python
DATASET_REGISTRY = {
    'rps': RockPaperScissorsDataModule,
    'cifar10': CIFAR10DataModule,  # 추가 가능
}
```

### 3. Configuration as Code
```yaml
# YAML로 모든 설정 관리
experiment:
  name: 'my_exp'

dataset_config: 'configs/datasets/...'
model_config: 'configs/models/...'

training:
  epochs: 50
  learning_rate: 0.001
```

---

## 🎓 Best Practices

### 1. 실험 관리
```
experiments/
├── vit_rps_baseline.yaml      # 베이스라인
├── vit_rps_aug_heavy.yaml     # Heavy augmentation
├── vit_rps_lr_0001.yaml       # LR 실험
└── resnet50_rps_baseline.yaml # 모델 비교
```

### 2. 버전 관리
- `configs/`는 Git으로 관리
- `outputs/`는 .gitignore
- 실험마다 `config.yaml` 자동 저장

### 3. 재현성
- Seed 고정: `experiment.seed`
- Config 저장: `outputs/{exp}/config.yaml`
- 결과 기록: `outputs/{exp}/results.yaml`

---

## 🚀 실무 워크플로우

### 1. 프로토타이핑
```bash
# 빠른 테스트 (10 epochs)
python main.py --config configs/experiments/vit_rps_quick.yaml
```

### 2. 하이퍼파라미터 튜닝
```bash
# vit_rps_quick.yaml 복사 후 수정
cp configs/experiments/vit_rps_quick.yaml configs/experiments/vit_lr_exp.yaml
# learning_rate 변경
python main.py --config configs/experiments/vit_lr_exp.yaml
```

### 3. 최종 학습
```bash
# 최적 설정으로 전체 학습
python main.py --config configs/experiments/vit_rps_full.yaml
```

### 4. 결과 분석
```bash
# 모든 실험 결과 비교
ls outputs/*/results.yaml
```

---

## 📈 확장 로드맵

### 단기
- [ ] WandB 통합 (실험 추적)
- [ ] Early Stopping
- [ ] Gradient Accumulation

### 중기
- [ ] Multi-GPU 지원
- [ ] Mixed Precision Training
- [ ] TensorBoard 통합

### 장기
- [ ] AutoML (하이퍼파라미터 자동 튜닝)
- [ ] Model Serving (FastAPI)
- [ ] CI/CD 파이프라인

---

**이 구조는 AI 실무자들이 사용하는 표준 패턴을 따릅니다.**
