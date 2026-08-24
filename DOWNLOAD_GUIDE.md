# Roboflow 데이터셋 다운로드 가이드

## 📥 개요

이 가이드는 Roboflow에서 데이터셋을 다운로드하는 방법을 설명합니다.

---

## 🔑 API Key 받기

1. [Roboflow](https://roboflow.com/)에 로그인
2. Settings → API → Private API Key 복사

---

## 🚀 다운로드 방법

### 방법 1: YAML Config 사용 (권장)

**장점:**
- ✅ API key를 환경변수로 관리 (보안)
- ✅ 프로젝트 정보를 파일로 관리
- ✅ 재사용 가능

**단계:**

```bash
# 1. 예제 config 파일 복사
cp configs/download/roboflow.example.yaml configs/download/roboflow.yaml

# 2. roboflow.yaml 파일 수정
# workspace, project, version 정보 입력

# 3. API key를 환경변수로 설정
export ROBOFLOW_API_KEY=your_api_key_here  # Linux/Mac
set ROBOFLOW_API_KEY=your_api_key_here     # Windows

# 4. roboflow 패키지 설치
pip install roboflow

# 5. 다운로드 실행
python scripts/download_roboflow.py --config configs/download/roboflow.yaml
```

**roboflow.yaml 예시:**
```yaml
api_key: null  # 환경변수 사용
workspace: 'cpecgm3'
project: 'rock-paper-scissor-p13xv'
version: 2
format: 'folder'
location: '.'
```

---

### 방법 2: 커맨드라인 직접 사용

**장점:**
- ✅ 빠른 일회성 다운로드
- ✅ Config 파일 불필요

**단계:**

```bash
# 환경변수로 API key 설정 (선택)
export ROBOFLOW_API_KEY=your_api_key_here

# 다운로드 실행
python scripts/download_roboflow.py \
  --workspace cpecgm3 \
  --project rock-paper-scissor-p13xv \
  --version 2 \
  --format folder \
  --location .
```

**또는 API key를 직접 입력:**
```bash
python scripts/download_roboflow.py \
  --api-key YOUR_API_KEY \
  --workspace cpecgm3 \
  --project rock-paper-scissor-p13xv \
  --version 2
```

---

### 방법 3: 수동 다운로드

1. [Roboflow Universe](https://universe.roboflow.com/)에서 프로젝트 찾기
2. "Download Dataset" 클릭
3. Format 선택: "Folder Structure"
4. 다운로드 후 프로젝트 루트에 압축 해제
5. `configs/datasets/rps_dataset1.yaml`의 `data_root` 경로 업데이트

---

## 📂 다운로드 후 폴더 구조

```
data/
└── Rock-Paper-Scissor--2/
    ├── train/
    │   ├── rock/
    │   ├── paper/
    │   └── scissors/
    ├── valid/
    │   ├── rock/
    │   ├── paper/
    │   └── scissors/
    └── test/
        ├── rock/
        ├── paper/
        └── scissors/
```

---

## 🔧 고급 옵션

### 다른 포맷으로 다운로드

```bash
# YOLOv5 포맷
python scripts/download_roboflow.py \
  --config configs/download/roboflow.yaml \
  --format yolov5

# COCO 포맷
python scripts/download_roboflow.py \
  --config configs/download/roboflow.yaml \
  --format coco
```

### 다운로드 위치 지정

```bash
python scripts/download_roboflow.py \
  --config configs/download/roboflow.yaml \
  --location ./datasets/my_dataset
```

---

## 🛡️ 보안 Best Practices

### ❌ 하지 말아야 할 것

```yaml
# roboflow.yaml에 직접 API key 저장 (절대 금지!)
api_key: 'E8WIPjtVKD19G5AAvr5A'  # ❌ Git에 커밋됨!
```

### ✅ 권장 방법

**Option 1: 환경변수 (가장 안전)**
```bash
export ROBOFLOW_API_KEY=your_key_here
python scripts/download_roboflow.py --config configs/download/roboflow.yaml
```

**Option 2: .env 파일 사용**
```bash
# .env 파일 생성 (이미 .gitignore에 포함됨)
echo "ROBOFLOW_API_KEY=your_key_here" > .env

# python-dotenv 설치
pip install python-dotenv

# 스크립트에서 자동 로드
```

**Option 3: 임시로 커맨드라인 사용**
```bash
python scripts/download_roboflow.py --api-key YOUR_KEY --workspace ... --project ...
```

---

## 🐛 문제 해결

### "roboflow module not found"
```bash
pip install roboflow
```

### "API key is required"
```bash
# 환경변수 설정 확인
echo $ROBOFLOW_API_KEY  # Linux/Mac
echo %ROBOFLOW_API_KEY%  # Windows

# 또는 직접 입력
python scripts/download_roboflow.py --api-key YOUR_KEY ...
```

### "Project not found"
- workspace와 project 이름이 정확한지 확인
- Roboflow URL 참고: `https://universe.roboflow.com/{workspace}/{project}`

### 다운로드 속도가 느림
- 네트워크 연결 확인
- Roboflow 서버 상태 확인
- 작은 버전부터 테스트

---

## 📚 추가 자료

- [Roboflow 공식 문서](https://docs.roboflow.com/)
- [Roboflow Python SDK](https://github.com/roboflow/roboflow-python)
- [Roboflow Universe](https://universe.roboflow.com/)

---

## 💡 다음 단계

다운로드 완료 후:

1. **Dataset Config 확인**
   ```bash
   # configs/datasets/rps_dataset1.yaml
   data_root: 'data/Rock-Paper-Scissor--2'  # data/ 폴더 경로
   ```

2. **학습 시작**
   ```bash
   python main.py --config configs/experiments/vit_rps_quick.yaml
   ```

3. **결과 확인**
   ```bash
   ls outputs/vit_rps_quick/
   ```

---

**Happy Training! 🚀**
