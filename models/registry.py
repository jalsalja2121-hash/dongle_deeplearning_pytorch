"""
Model Registry System
"""
from typing import Dict, Type, Any
import torch.nn as nn


class ModelRegistry:
    """Central registry for all available models"""

    _models: Dict[str, Type[nn.Module]] = {}

    @classmethod
    def register(cls, name: str, model_class: Type[nn.Module]):
        """
        Register a model class

        Args:
            name: Model name identifier
            model_class: Model class
        """
        cls._models[name] = model_class
        print(f"Registered model: {name}")

    @classmethod
    def get(cls, name: str) -> Type[nn.Module]:
        """
        Get model class by name

        Args:
            name: Model name

        Returns:
            Model class

        Raises:
            ValueError: If model not found
        """
        if name not in cls._models:
            available = ', '.join(cls._models.keys())
            raise ValueError(f"Model '{name}' not found. Available models: {available}")

        return cls._models[name]

    @classmethod
    def list_models(cls):
        """List all registered models"""
        return list(cls._models.keys())

    @classmethod
    def create(cls, name: str, **kwargs) -> nn.Module:
        """
        Create model instance

        Args:
            name: Model name
            **kwargs: Model parameters

        Returns:
            Model instance
        """
        model_class = cls.get(name)
        return model_class(**kwargs)


def register_model(name: str):
    """
    Decorator to register a model

    Usage:
        @register_model('my_model')
        class MyModel(nn.Module):
            ...
    """
    def decorator(model_class: Type[nn.Module]):
        ModelRegistry.register(name, model_class)
        return model_class

    return decorator


def create_model(config: Dict[str, Any]) -> nn.Module:
    """
    Create model from configuration dictionary

    Args:
        config: Model configuration

    Returns:
        Model instance
    """
    model_type = config.get('type')
    params = config.get('params', {})

    if not model_type:
        raise ValueError("Model config must specify 'type'")

    return ModelRegistry.create(model_type, **params)
