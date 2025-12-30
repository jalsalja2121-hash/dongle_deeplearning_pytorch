"""
Visualization utilities
"""
import matplotlib.pyplot as plt


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
