"""
Model registry and factory
"""
from .vit import ViTB16
from .resnet import ResNet50
from .alexnet import AlexNet


# Model Registry
MODEL_REGISTRY = {
    'vit_b16': ViTB16,
    'vit': ViTB16,  # alias
    'resnet50': ResNet50,
    'alexnet': AlexNet,
}


def create_model(config):
    """
    Factory function to create model

    Args:
        config (dict): Model configuration
            - type (str): Model type (vit_b16, resnet50, alexnet)
            - num_classes (int): Number of output classes
            - feature_extractor (bool): Freeze pretrained weights
            - pretrained (bool): Use pretrained weights

    Returns:
        nn.Module: Model instance

    Raises:
        ValueError: If model type is not in registry
    """
    model_type = config.get('type', 'vit')
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )

    return MODEL_REGISTRY[model_type](
        num_classes=config.get('num_classes', 3),
        feature_extractor=config.get('feature_extractor', False),
        pretrained=config.get('pretrained', True)
    )


__all__ = ['ViTB16', 'ResNet50', 'AlexNet', 'MODEL_REGISTRY', 'create_model']
