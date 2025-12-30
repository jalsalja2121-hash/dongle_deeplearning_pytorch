"""
통합 학습 스크립트
다양한 모델과 데이터셋 조합으로 실험 가능
"""
import os
import argparse
import torch

# 모델과 데이터셋 자동 등록을 위한 import
import models
import datasets
from utils import load_experiment_config, plot_training_history
from trainer import RockPaperScissorsTrainer


def create_output_dir(save_dir):
    """출력 디렉토리 생성"""
    os.makedirs(save_dir, exist_ok=True)
    print(f"Output directory: {save_dir}")


def main(config_path):
    """
    메인 실행 함수

    Args:
        config_path: 실험 설정 파일 경로
    """
    print("=" * 70)
    print("Starting Training with Configuration-based System")
    print("=" * 70)

    # ========================================
    # 1. 설정 로드
    # ========================================
    print("\n[1/6] Loading configuration...")
    config = load_experiment_config(config_path)

    exp_name = config['experiment']['name']
    print(f"Experiment: {exp_name}")
    print(f"Description: {config['experiment'].get('description', 'N/A')}")

    # 출력 디렉토리 생성
    save_dir = config['output']['save_dir']
    create_output_dir(save_dir)

    # ========================================
    # 2. 데이터셋 준비
    # ========================================
    print("\n[2/6] Preparing dataset...")

    dataset_config = config['dataset']
    data_config = config.get('data', {})

    # 데이터셋 인스턴스 생성
    dataset_module = datasets.create_dataset(dataset_config)

    # 배치 크기와 이미지 크기 설정
    dataset_module.batch_size = config['training']['batch_size']
    dataset_module.image_size = tuple(data_config.get('image_size', [224, 224]))
    dataset_module.num_workers = data_config.get('num_workers', 0)

    # 데이터셋 로드
    dataset_module.setup()

    # DataLoader 가져오기
    train_loader = dataset_module.get_train_loader()
    val_loader = dataset_module.get_validation_loader()
    test_loader = dataset_module.get_test_loader()

    # ========================================
    # 3. 모델 초기화
    # ========================================
    print("\n[3/6] Initializing model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")

    # 모델 설정에 num_classes 추가
    model_config = config['model'].copy()
    model_config['params']['num_classes'] = dataset_module.get_num_classes()

    # 모델 생성
    model = models.create_model(model_config)
    print(f"Model: {model_config['name']} ({model_config['type']})")
    print(f"Number of classes: {dataset_module.get_num_classes()}")

    # ========================================
    # 4. Trainer 초기화
    # ========================================
    print("\n[4/6] Setting up trainer...")

    training_config = config['training']
    optimizer_config = training_config['optimizer']
    scheduler_config = training_config['scheduler']
    loss_config = training_config['loss']

    trainer = RockPaperScissorsTrainer(
        model=model,
        device=device,
        learning_rate=training_config['learning_rate'],
        optimizer_name=optimizer_config['type'],
        weight_decay=optimizer_config.get('weight_decay', 0),
        momentum=optimizer_config.get('momentum', 0.9),
        loss_function_name=loss_config['type'],
        label_smoothing=loss_config.get('label_smoothing', 0.0),
        use_scheduler=scheduler_config.get('use', False),
        scheduler_name=scheduler_config.get('type', 'step'),
        scheduler_step_size=scheduler_config.get('step_size', 5),
        scheduler_gamma=scheduler_config.get('gamma', 0.1)
    )

    print(f"Optimizer: {optimizer_config['type'].upper()}")
    print(f"Learning Rate: {training_config['learning_rate']}")
    print(f"Loss Function: {loss_config['type'].upper()}")
    print(f"Epochs: {training_config['epochs']}")
    print(f"Batch Size: {training_config['batch_size']}")

    # ========================================
    # 5. 학습 실행
    # ========================================
    print("\n[5/6] Training...")
    print("=" * 70)

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=training_config['epochs'],
        verbose=True
    )

    # ========================================
    # 6. 평가 및 저장
    # ========================================
    print("\n[6/6] Evaluation and saving...")

    # 학습 히스토리 시각화
    plot_path = os.path.join(save_dir, config['output']['plot_name'])
    plot_training_history(history, save_path=plot_path)

    # 테스트
    print("\nTesting on test set...")
    trainer.test(test_loader, verbose=True)

    # 모델 저장
    model_path = os.path.join(save_dir, config['output']['model_name'])
    trainer.save_model(model_path)

    print("\n" + "=" * 70)
    print(f"Training completed! Results saved to: {save_dir}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train models with YAML configuration')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/experiments/exp_vit_rps.yaml',
        help='Path to experiment config file (default: configs/experiments/exp_vit_rps.yaml)'
    )

    args = parser.parse_args()

    main(args.config)
