"""
Trainer registry and factory
"""
from .rps_trainer import RockPaperScissorsTrainer
from .losses import FocalLoss


# Trainer Registry
TRAINER_REGISTRY = {
    'rps': RockPaperScissorsTrainer,
    'rock_paper_scissors': RockPaperScissorsTrainer,
}


def create_trainer(model, device, config):
    """
    Factory function to create trainer

    Args:
        model: PyTorch model to train
        device: Training device ('cuda' or 'cpu')
        config (dict): Training configuration
            - type (str): Trainer type (optional, defaults to 'rps')
            - learning_rate (float): Learning rate
            - epochs (int): Number of training epochs
            - optimizer (dict): Optimizer configuration
                - type (str): Optimizer type (adam, sgd, adamw)
                - weight_decay (float): Weight decay
                - momentum (float): Momentum for SGD
            - loss (dict): Loss configuration
                - type (str): Loss type (crossentropy, label_smoothing, focal)
                - label_smoothing (float): Label smoothing value
            - scheduler (dict): Scheduler configuration
                - use (bool): Whether to use scheduler
                - type (str): Scheduler type (step, cosine, reduce)
                - step_size (int): Step size
                - gamma (float): Learning rate decay factor

    Returns:
        Trainer instance

    Raises:
        ValueError: If trainer type is not in registry
    """
    trainer_type = config.get('type', 'rps')

    if trainer_type not in TRAINER_REGISTRY:
        raise ValueError(
            f"Unknown trainer type: {trainer_type}. "
            f"Available trainers: {list(TRAINER_REGISTRY.keys())}"
        )

    trainer_class = TRAINER_REGISTRY[trainer_type]

    # Extract optimizer config
    optimizer_cfg = config['optimizer']

    # Extract loss config
    loss_cfg = config['loss']

    # Extract scheduler config
    scheduler_cfg = config['scheduler']

    # Extract early stopping config (optional)
    early_stopping_cfg = config.get('early_stopping', {})

    return trainer_class(
        model=model,
        device=device,
        learning_rate=config['learning_rate'],
        optimizer_name=optimizer_cfg['type'],
        weight_decay=optimizer_cfg.get('weight_decay', 0),
        momentum=optimizer_cfg.get('momentum', 0.9),
        loss_function_name=loss_cfg['type'],
        label_smoothing=loss_cfg.get('label_smoothing', 0.0),
        use_scheduler=scheduler_cfg['use'],
        scheduler_name=scheduler_cfg.get('type', 'step'),
        scheduler_step_size=scheduler_cfg.get('step_size', 5),
        scheduler_gamma=scheduler_cfg.get('gamma', 0.1),
        early_stopping_patience=early_stopping_cfg.get('patience', None),
        early_stopping_min_delta=early_stopping_cfg.get('min_delta', 0.0001),
        early_stopping_monitor=early_stopping_cfg.get('monitor', 'val_loss')
    )


__all__ = ['RockPaperScissorsTrainer', 'FocalLoss', 'TRAINER_REGISTRY', 'create_trainer']
