"""
Configuration loading and merging utilities
"""
import os
import yaml
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file

    Args:
        config_path: Path to YAML config file

    Returns:
        Dictionary containing configuration
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge multiple configuration dictionaries
    Later configs override earlier ones

    Args:
        *configs: Variable number of config dictionaries

    Returns:
        Merged configuration dictionary
    """
    def deep_merge(base: Dict, override: Dict) -> Dict:
        """Recursively merge two dictionaries"""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    if not configs:
        return {}

    result = configs[0].copy()
    for config in configs[1:]:
        result = deep_merge(result, config)

    return result


def load_experiment_config(exp_config_path: str) -> Dict[str, Any]:
    """
    Load experiment configuration and merge with model/dataset configs

    Args:
        exp_config_path: Path to experiment config file

    Returns:
        Complete merged configuration
    """
    # Load experiment config
    exp_config = load_config(exp_config_path)

    # Load model config
    model_config_path = exp_config.get('model_config')
    if model_config_path:
        model_config = load_config(model_config_path)
    else:
        model_config = {}

    # Load dataset config
    dataset_config_path = exp_config.get('dataset_config')
    if dataset_config_path:
        dataset_config = load_config(dataset_config_path)
    else:
        dataset_config = {}

    # Merge configs: model -> dataset -> experiment -> overrides
    final_config = merge_configs(
        model_config,
        dataset_config,
        exp_config,
        exp_config.get('overrides', {})
    )

    return final_config
