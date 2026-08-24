"""
ResNet50 Model for Rock-Paper-Scissors Classification
"""
import torch.nn as nn
from torchvision import models


class ResNet50(nn.Module):
    """
    ResNet50 기반 가위바위보 분류 모델

    Args:
        num_classes (int): 출력 클래스 개수 (기본값: 3)
        feature_extractor (bool): True면 사전학습 파라미터 동결 (기본값: False)
        pretrained (bool): 사전학습된 가중치 사용 여부 (기본값: True)
    """

    def __init__(self, num_classes=3, feature_extractor=False, pretrained=True):
        super().__init__()

        # 사전 학습된 ResNet50 모델 로드
        if pretrained:
            weights = models.ResNet50_Weights.DEFAULT
            pretrained_model = models.resnet50(weights=weights)
        else:
            pretrained_model = models.resnet50(weights=None)

        # Feature Extractor 모드: 사전학습된 파라미터 동결
        if feature_extractor:
            for param in pretrained_model.parameters():
                param.requires_grad = False

        # Classifier 부분 교체 (fc)
        in_features = pretrained_model.fc.in_features
        pretrained_model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

        self.model = pretrained_model

    def forward(self, x):
        """
        Forward pass

        Args:
            x: 입력 텐서 (batch_size, 3, 224, 224)

        Returns:
            logits: 출력 로짓 (batch_size, num_classes)
        """
        return self.model(x)
