import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss - 불균형 데이터셋에 유용한 손실 함수

    Args:
        alpha (float): 클래스 가중치 (기본값: 0.25)
        gamma (float): focusing parameter (기본값: 2.0)
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class RockPaperScissorsTrainer:
    """
    가위바위보 모델 학습을 위한 Trainer 클래스

    Args:
        model: 학습할 PyTorch 모델
        device: 학습에 사용할 디바이스 ('cuda' 또는 'cpu')
        learning_rate (float): 학습률 (기본값: 1e-6)
        optimizer_name (str): 옵티마이저 종류 ('adam', 'sgd', 'adamw') (기본값: 'adam')
        weight_decay (float): 가중치 감쇠 (L2 regularization) (기본값: 0)
        momentum (float): SGD 모멘텀 (기본값: 0.9)
        loss_function_name (str): 손실 함수 종류 ('crossentropy', 'label_smoothing', 'focal') (기본값: 'crossentropy')
        label_smoothing (float): Label Smoothing 값 (기본값: 0.0)
        use_scheduler (bool): Learning Rate Scheduler 사용 여부 (기본값: False)
        scheduler_name (str): 스케줄러 종류 ('step', 'cosine', 'reduce') (기본값: 'step')
        scheduler_step_size (int): 스케줄러 스텝 크기 (기본값: 5)
        scheduler_gamma (float): 학습률 감소 비율 (기본값: 0.1)
    """

    def __init__(
        self,
        model,
        device='cpu',
        learning_rate=1e-6,
        optimizer_name='adam',
        weight_decay=0,
        momentum=0.9,
        loss_function_name='crossentropy',
        label_smoothing=0.0,
        use_scheduler=False,
        scheduler_name='step',
        scheduler_step_size=5,
        scheduler_gamma=0.1
    ):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name.lower()
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.use_scheduler = use_scheduler
        self.scheduler_name = scheduler_name.lower()
        self.loss_function_name = loss_function_name.lower()
        self.label_smoothing = label_smoothing

        # 손실함수 정의
        self.loss_function = self._create_loss_function()

        # 옵티마이저 정의
        self.optimizer = self._create_optimizer()

        # 스케줄러 정의
        self.scheduler = None
        if self.use_scheduler:
            self.scheduler = self._create_scheduler(scheduler_step_size, scheduler_gamma)

        # 학습 히스토리
        self.train_history = {'loss': [], 'accuracy': []}
        self.val_history = {'loss': [], 'accuracy': []}

    def _create_loss_function(self):
        """손실 함수 생성"""
        if self.loss_function_name == 'crossentropy':
            return nn.CrossEntropyLoss()
        elif self.loss_function_name == 'label_smoothing':
            return nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        elif self.loss_function_name == 'focal':
            # Focal Loss는 직접 구현 (불균형 데이터셋에 유용)
            return FocalLoss(alpha=0.25, gamma=2.0)
        else:
            raise ValueError(f"Unknown loss function: {self.loss_function_name}. Choose from 'crossentropy', 'label_smoothing', 'focal'.")

    def _create_optimizer(self):
        """옵티마이저 생성"""
        if self.optimizer_name == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name == 'sgd':
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}. Choose from 'adam', 'adamw', 'sgd'.")

    def _create_scheduler(self, step_size, gamma):
        """학습률 스케줄러 생성"""
        if self.scheduler_name == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
        elif self.scheduler_name == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=step_size
            )
        elif self.scheduler_name == 'reduce':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=gamma,
                patience=step_size
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler_name}. Choose from 'step', 'cosine', 'reduce'.")

    def train_epoch(self, dataloader):
        """
        1 에포크 학습 수행

        Args:
            dataloader: 학습 데이터로더

        Returns:
            tuple: (평균 손실, 평균 정확도)
        """
        self.model.train()  # 신경망을 학습모드로 전환

        train_loss_sum = train_correct = train_total = 0
        total_train_batch = len(dataloader)

        for images, labels in dataloader:
            x_train = images.to(self.device)
            y_train = labels.to(self.device)

            # Forward pass
            outputs = self.model(x_train)
            loss = self.loss_function(outputs, y_train)

            # Backward pass 및 최적화
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 통계 계산
            train_loss_sum += loss.item()
            train_total += y_train.size(0)
            train_correct += (torch.argmax(outputs, 1) == y_train).sum().item()

        # 평균 계산
        train_avg_loss = train_loss_sum / total_train_batch
        train_avg_accuracy = 100 * train_correct / train_total

        # 히스토리 저장
        self.train_history['loss'].append(train_avg_loss)
        self.train_history['accuracy'].append(train_avg_accuracy)

        return train_avg_loss, train_avg_accuracy

    def evaluate(self, dataloader):
        """
        검증/테스트 데이터셋 평가

        Args:
            dataloader: 검증/테스트 데이터로더

        Returns:
            tuple: (평균 손실, 평균 정확도)
        """
        self.model.eval()  # 신경망을 평가모드로 전환

        with torch.no_grad():
            val_loss_sum = val_correct = val_total = 0
            total_val_batch = len(dataloader)

            for images, labels in dataloader:
                x_val = images.to(self.device)
                y_val = labels.to(self.device)

                # Forward pass
                outputs = self.model(x_val)
                loss = self.loss_function(outputs, y_val)

                # 통계 계산
                val_loss_sum += loss.item()
                val_total += y_val.size(0)
                val_correct += (torch.argmax(outputs, 1) == y_val).sum().item()

            # 평균 계산
            val_avg_loss = val_loss_sum / total_val_batch
            val_avg_accuracy = 100 * val_correct / val_total

        return val_avg_loss, val_avg_accuracy

    def fit(self, train_loader, val_loader, epochs=10, verbose=True):
        """
        모델 학습 수행

        Args:
            train_loader: 학습 데이터로더
            val_loader: 검증 데이터로더
            epochs (int): 학습 에포크 수 (기본값: 10)
            verbose (bool): 학습 진행 상황 출력 여부 (기본값: True)

        Returns:
            dict: 학습 및 검증 히스토리
        """
        for epoch in range(1, epochs + 1):
            # 학습
            train_loss, train_acc = self.train_epoch(train_loader)

            # 검증
            val_loss, val_acc = self.evaluate(val_loader)

            # 검증 히스토리 저장
            self.val_history['loss'].append(val_loss)
            self.val_history['accuracy'].append(val_acc)

            # Learning Rate Scheduler 스텝
            if self.scheduler is not None:
                if self.scheduler_name == 'reduce':
                    # ReduceLROnPlateau는 val_loss를 기준으로 조정
                    self.scheduler.step(val_loss)
                else:
                    # StepLR, CosineAnnealingLR은 에포크 기준
                    self.scheduler.step()

                current_lr = self.optimizer.param_groups[0]['lr']
            else:
                current_lr = self.learning_rate

            # 진행 상황 출력
            if verbose:
                print(f'Epoch [{epoch}/{epochs}]')
                print(f'  Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%')
                print(f'  Val   - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%')
                if self.scheduler is not None:
                    print(f'  LR    - {current_lr:.2e}')
                print()

        return {
            'train': self.train_history,
            'val': self.val_history
        }

    def test(self, test_loader, verbose=True):
        """
        테스트 데이터셋 평가

        Args:
            test_loader: 테스트 데이터로더
            verbose (bool): 결과 출력 여부 (기본값: True)

        Returns:
            tuple: (테스트 손실, 테스트 정확도)
        """
        test_loss, test_acc = self.evaluate(test_loader)

        if verbose:
            print('=== Test Results ===')
            print(f'Accuracy: {test_acc:.2f}%')
            print(f'Loss: {test_loss:.4f}')

        return test_loss, test_acc

    def save_model(self, filepath):
        """
        모델 가중치 저장

        Args:
            filepath (str): 저장할 파일 경로
        """
        torch.save(self.model.state_dict(), filepath)
        print(f'Model saved to {filepath}')

    def load_model(self, filepath):
        """
        모델 가중치 로드

        Args:
            filepath (str): 로드할 파일 경로
        """
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f'Model loaded from {filepath}')

    def get_history(self):
        """
        학습 히스토리 반환

        Returns:
            dict: 학습 및 검증 히스토리
        """
        return {
            'train': self.train_history,
            'val': self.val_history
        }
