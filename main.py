import torch
import matplotlib.pyplot as plt
from dataset import RockPaperScissorsDataModule
from model import MyVit_b_16  # MyResNet50도 사용 가능
from trainer import RockPaperScissorsTrainer


def plot_training_history(history, save_path='training_history.png'):
    """
    학습 히스토리 시각화

    Args:
        history (dict): trainer.fit()의 반환값
        save_path (str): 저장할 이미지 경로
    """
    train_loss = history['train']['loss']
    train_acc = history['train']['accuracy']
    val_loss = history['val']['loss']
    val_acc = history['val']['accuracy']

    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.title('Loss Trend')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.plot(epochs, train_loss, 'b-', label='Train Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.title('Accuracy Trend')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.plot(epochs, train_acc, 'b-', label='Train Accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")
    plt.show()


def main():
    """가위바위보 이미지 분류 메인 실행 함수"""

    # ========================================
    # 하이퍼파라미터 설정 (ViT-B/16 논문 기반)
    # ========================================
    # 데이터 설정
    BATCH_SIZE = 32  # 기존 설정 유지 (ViT 논문: 512)
    IMAGE_SIZE = (224, 224)  # ViT 논문: 224x224
    NUM_WORKERS = 0

    # 학습 설정
    EPOCHS = 300  # ViT 논문: 300 epochs (ImageNet-21k fine-tuning)
    LEARNING_RATE = 1e-3  # 기존 설정 유지 (ViT 논문: 3e-3)
    FEATURE_EXTRACTOR = True  # Feature Extractor 모드 (ViT 논문: False)

    # 옵티마이저 설정 (ViT 논문: Adam with β1=0.9, β2=0.999)
    OPTIMIZER = 'adam'
    WEIGHT_DECAY = 0.1  # ViT 논문: 0.1 (weight decay)
    MOMENTUM = 0.9  # Adam에서는 사용 안 함

    # Learning Rate Scheduler 설정 (ViT 논문: Linear warmup + decay)
    USE_SCHEDULER = True
    SCHEDULER_NAME = 'cosine'  # ViT 논문: Cosine annealing
    SCHEDULER_STEP_SIZE = 300  # Total epochs
    SCHEDULER_GAMMA = 0.1  # 사용 안 함 (cosine에서)

    # Loss Function 설정
    LOSS_FUNCTION = 'crossentropy'  # ViT 논문: Cross-entropy
    LABEL_SMOOTHING = 0.0  # 필요시 0.1 사용 가능

    # ========================================
    # 1. 데이터 준비
    # ========================================
    print("=" * 60)
    print("1. 데이터 로딩 중...")
    print("=" * 60)

    # 데이터 추출 (처음 한 번만 실행)
    # import extract
    # extract.extract_data()

    # 데이터 모듈 인스턴스 생성
    data_module = RockPaperScissorsDataModule(
        data_root='rock_paper_scissor',
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_workers=NUM_WORKERS
    )

    # 데이터셋 로드 및 초기화
    data_module.setup()

    # 클래스 정보 출력
    print(f"\n클래스 개수: {data_module.get_num_classes()}")
    print(f"클래스 이름: {data_module.get_class_names()}")

    # DataLoader 가져오기
    train_loader = data_module.get_train_loader()
    val_loader = data_module.get_validation_loader()
    test_loader = data_module.get_test_loader()

    # 샘플 이미지 시각화 (선택사항)
    # data_module.visualize_batch(num_images=16, dataset='train')

    # ========================================
    # 2. 모델 초기화
    # ========================================
    print("\n" + "=" * 60)
    print("2. 모델 초기화 중...")
    print("=" * 60)

    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")

    # 모델 선택
    # Option 1: Vision Transformer (ViT)
    model = MyVit_b_16(
        num_classes=3,
        feature_extractor=FEATURE_EXTRACTOR,
        pretrained=True
    )

    # Option 2: ResNet50 (주석 해제하여 사용)
    # model = MyResNet50(
    #     num_classes=3,
    #     feature_extractor=FEATURE_EXTRACTOR,
    #     pretrained=True
    # )

    print(f"Model: {model.__class__.__name__}")
    print(f"Mode: {'Feature Extractor' if FEATURE_EXTRACTOR else 'Fine-tuning'}")

    # ========================================
    # 3. 학습 설정
    # ========================================
    print("\n" + "=" * 60)
    print("3. 학습 시작...")
    print("=" * 60)

    # Trainer 초기화
    trainer = RockPaperScissorsTrainer(
        model=model,
        device=device,
        learning_rate=LEARNING_RATE,
        optimizer_name=OPTIMIZER,
        weight_decay=WEIGHT_DECAY,
        momentum=MOMENTUM,
        loss_function_name=LOSS_FUNCTION,
        label_smoothing=LABEL_SMOOTHING,
        use_scheduler=USE_SCHEDULER,
        scheduler_name=SCHEDULER_NAME,
        scheduler_step_size=SCHEDULER_STEP_SIZE,
        scheduler_gamma=SCHEDULER_GAMMA
    )

    print(f"Loss Function: {LOSS_FUNCTION.upper()}")
    if LOSS_FUNCTION == 'label_smoothing':
        print(f"  Label Smoothing: {LABEL_SMOOTHING}")
    print(f"Optimizer: {OPTIMIZER.upper()}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Weight Decay: {WEIGHT_DECAY}")
    if OPTIMIZER.lower() == 'sgd':
        print(f"Momentum: {MOMENTUM}")
    if USE_SCHEDULER:
        print(f"Scheduler: {SCHEDULER_NAME.upper()} (step_size={SCHEDULER_STEP_SIZE}, gamma={SCHEDULER_GAMMA})")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}\n")

    # 학습 실행
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        verbose=True
    )

    # ========================================
    # 4. 학습 히스토리 시각화
    # ========================================
    print("\n" + "=" * 60)
    print("4. 학습 히스토리 시각화 중...")
    print("=" * 60)

    plot_training_history(history, save_path='training_history.png')

    # ========================================
    # 5. 테스트
    # ========================================
    print("\n" + "=" * 60)
    print("5. 테스트 평가 중...")
    print("=" * 60)

    trainer.test(test_loader, verbose=True)

    # ========================================
    # 6. 모델 저장
    # ========================================
    print("\n" + "=" * 60)
    print("6. 모델 저장 중...")
    print("=" * 60)

    trainer.save_model('rock_paper_scissors_model.pth')

    print("\n" + "=" * 60)
    print("학습 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()