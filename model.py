"""
Legacy model module for backward compatibility

DEPRECATED: Use src.models instead
This module is kept for backward compatibility only.
"""
from src.models import ViTB16 as MyVit_b_16
from src.models import ResNet50 as MyResNet50

__all__ = ['MyVit_b_16', 'MyResNet50']
