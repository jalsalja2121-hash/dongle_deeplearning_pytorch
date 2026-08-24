"""
Rock-Paper-Scissors Classifier Training Script
YAML 기반 실험 관리 시스템
"""
import argparse
from pathlib import Path
import yaml
import torch
import matplotlib.pyplot as plt

from src.datasets import create_dataset
from src.models import create_model
from src.trainers import create_trainer


def print_section(title):
    """섹션 헤더 출력"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_config(config_path):
    """YAML 설정 파일 로드 및 병합"""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 참조된 설정 파일들 로드
    if 'dataset_config' in config:
        with open(config['dataset_config'], 'r', encoding='utf-8') as f:
            config['dataset'] = yaml.safe_load(f)

    if 'model_config' in config:
        with open(config['model_config'], 'r', encoding='utf-8') as f:
            config['model'] = yaml.safe_load(f)

    return config


def plot_training_history(history, save_path):
    """학습 히스토리 시각화"""
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    epochs = range(1, len(history['train']['loss']) + 1)

    # Loss plot
    ax1.plot(epochs, history['train']['loss'], 'b-', label='Train')
    ax1.plot(epochs, history['val']['loss'], 'r-', label='Validation')
    ax1.set_title('Loss Trend')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy plot
    ax2.plot(epochs, history['train']['accuracy'], 'b-', label='Train')
    ax2.plot(epochs, history['val']['accuracy'], 'r-', label='Validation')
    ax2.set_title('Accuracy Trend')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"✅ Plot saved: {save_path}")


def main(args):
    """메인 학습 파이프라인"""

    # 1. 설정 로드
    print("=" * 70)
    print("EXPERIMENT SETUP")
    print("=" * 70)

    config = load_config(args.config)
    exp_name = config['experiment']['name']
    print(f"Experiment: {exp_name}")
    print(f"Config: {args.config}")
    print(f"Description: {config['experiment'].get('description', 'N/A')}\n")

    # 출력 디렉토리 생성
    output_dir = Path(config['output']['save_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. 데이터셋 준비
    print_section("DATASET")
    dataset = create_dataset(config['dataset'])
    dataset.setup()

    train_loader = dataset.get_train_loader()
    val_loader = dataset.get_validation_loader()
    test_loader = dataset.get_test_loader()

    # 3. 모델 초기화
    print_section("MODEL")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_config = config['model'].copy()
    model_config['num_classes'] = dataset.get_num_classes()
    model = create_model(model_config)

    print(f"Architecture: {model_config['type']}")
    print(f"Mode: {'Feature Extractor' if model_config['feature_extractor'] else 'Fine-tuning'}")

    # 4. Trainer 설정
    print_section("TRAINING CONFIG")
    training_cfg = config['training']
    trainer = create_trainer(model, device, training_cfg)

    print(f"Epochs: {training_cfg['epochs']}")
    print(f"Learning Rate: {training_cfg['learning_rate']}")
    print(f"Optimizer: {training_cfg['optimizer']['type'].upper()}")
    print(f"Loss: {training_cfg['loss']['type'].upper()}\n")

    # 5. 학습 실행
    print_section("TRAINING")

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=training_cfg['epochs'],
        verbose=True
    )

    # 6. 평가 및 저장
    print_section("EVALUATION & SAVING")

    # 시각화
    plot_path = output_dir / config['output']['plot_name']
    plot_training_history(history, plot_path)

    # 테스트
    test_loss, test_acc = trainer.test(test_loader, verbose=True)

    # 모델 저장
    model_path = output_dir / config['output']['model_name']
    trainer.save_model(str(model_path))

    # 설정 및 결과 저장
    with open(output_dir / 'config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)

    results = {
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'best_val_accuracy': float(max(history['val']['accuracy'])),
        'final_train_accuracy': float(history['train']['accuracy'][-1])
    }
    with open(output_dir / 'results.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(results, f, default_flow_style=False)

    # 최종 요약
    print_section("TRAINING COMPLETED")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Best Val Accuracy: {max(history['val']['accuracy']):.2f}%")
    print(f"Results: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train Rock-Paper-Scissors Classifier',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --config configs/experiments/vit_rps_quick.yaml
  python main.py --config configs/experiments/vit_rps_full.yaml
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to experiment config YAML file'
    )

    args = parser.parse_args()
    main(args)
