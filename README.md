# PyTorch Image Classification Lab

PyTorch로 이미지 분류 모델을 비교하고, 데이터·모델·학습 설정을 모듈화한 프로젝트입니다.
가위바위보 분류와 흉부 X-ray 폐렴 분류 실험을 하나의 학습 파이프라인에서 실행할 수 있습니다.

## 프로젝트 요약

| 실험 | 모델 | 분류 클래스 | 핵심 결과 |
|---|---|---|---|
| 가위바위보 이미지 분류 | AlexNet, ViT | Rock, Paper, Scissors | 모델별 YAML 실험 구성 |
| 흉부 X-ray 폐렴 분류 | ResNet50 | NORMAL, PNEUMONIA | Test Accuracy **93.13%** |

### 구현 특징

- Dataset·Model·Trainer를 Registry Pattern으로 분리
- YAML 파일로 모델과 하이퍼파라미터 조합 관리
- ImageNet 사전학습 모델을 활용한 Feature Extraction 및 Fine-tuning 지원
- Cross Entropy, Label Smoothing, Focal Loss 지원
- StepLR, CosineAnnealingLR, ReduceLROnPlateau 지원
- Early Stopping, 성능 그래프, 설정 및 결과 자동 저장

## 흉부 X-ray 폐렴 분류

흉부 X-ray 한 장을 입력받아 `NORMAL` 또는 `PNEUMONIA`로 분류하는 이진 분류 실험입니다.
객체 위치를 찾는 Detection 모델이 아니라 영상 전체를 판별하는 Classification 모델입니다.

### 모델 및 학습 구성

- Backbone: ImageNet 사전학습 ResNet50
- 입력 크기: 224 × 224
- 출력 클래스: 2개 (`NORMAL`, `PNEUMONIA`)
- Optimizer: Adam
- Learning rate: 0.0001
- Scheduler: CosineAnnealingLR
- Loss: Cross Entropy
- Early stopping patience: 5

### 기록된 실험 결과

| 지표 | 결과 |
|---|---:|
| Best validation accuracy | 96.22% |
| Final training accuracy | 95.54% |
| Test accuracy | **93.13%** |
| Test loss | 0.1751 |

> 위 수치는 `outputs/resnet50_xray_pneumonia/results.yaml`에 저장된 기존 실험 결과입니다.
> 의료 진단용 성능을 의미하지 않으며, 학습용 이미지 분류 프로젝트입니다.

## 프로젝트 구조

```text
.
├── configs/
│   ├── datasets/          # 데이터 경로와 전처리 설정
│   ├── models/            # 모델별 설정
│   └── experiments/       # 데이터셋·모델·학습 설정 조합
├── outputs/               # 저장된 실험 설정과 결과 지표
├── scripts/               # 데이터 다운로드·압축 해제 도구
├── src/
│   ├── datasets/          # ImageFolder 데이터 모듈과 Registry
│   ├── models/            # AlexNet, ResNet50, ViT
│   └── trainers/          # 학습·검증·테스트 로직
├── main.py                # 공통 학습 진입점
└── requirements.txt
```

## 설치

```bash
git clone https://github.com/jalsalja2121-hash/dongle_deeplearning_pytorch.git
cd dongle_deeplearning_pytorch
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## 데이터 준비

데이터 파일은 용량과 라이선스 문제로 저장소에 포함하지 않습니다. 각 데이터셋은 아래처럼 클래스별 폴더로 배치합니다.

흉부 X-ray:

```text
data/Chest_Xray/
├── train/{NORMAL,PNEUMONIA}/
├── val/{NORMAL,PNEUMONIA}/       # valid 폴더명도 지원
└── test/{NORMAL,PNEUMONIA}/
```

가위바위보:

```text
data/Rock-Paper-Scissor--2/
├── train/{paper,rock,scissors}/
├── valid/{paper,rock,scissors}/
└── test/{paper,rock,scissors}/
```

다른 위치에 데이터를 저장했다면 해당 `configs/datasets/*.yaml`의 `data_root`만 변경하면 됩니다.

## 실행

흉부 X-ray 폐렴 분류:

```bash
python main.py --config configs/experiments/resnet50_xray.yaml
```

가위바위보 빠른 실험:

```bash
python main.py --config configs/experiments/vit_rps_quick.yaml
python main.py --config configs/experiments/alexnet_rps_quick.yaml
```

학습이 끝나면 `outputs/{experiment_name}/`에 모델 가중치, 학습 그래프, 설정, 결과가 저장됩니다.
모델 가중치(`*.pth`)와 원본 데이터는 `.gitignore`에 의해 GitHub에 업로드되지 않습니다.

## 설정 확장

새로운 실험은 기존 YAML을 복사해 데이터셋, 모델, 학습률, Epoch 등을 변경하여 추가할 수 있습니다.
새 모델이나 데이터 형식은 `src/models` 또는 `src/datasets`의 Registry에 등록해 공통 파이프라인에서 사용할 수 있습니다.
