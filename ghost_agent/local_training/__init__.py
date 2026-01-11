"""
Module: ghost_agent/local_training/__init__.py
DPDP §: 9(4) - Purpose limitation through encrypted gradients
Description: Local training with privacy preservation
"""

from .local_trainer import LocalTrainer

__all__ = ["LocalTrainer"]