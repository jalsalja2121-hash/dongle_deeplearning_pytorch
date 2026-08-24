"""
Legacy trainer module for backward compatibility

DEPRECATED: Use src.trainers instead
This module is kept for backward compatibility only.
"""
from src.trainers import RockPaperScissorsTrainer, FocalLoss

__all__ = ['RockPaperScissorsTrainer', 'FocalLoss']
