"""
Module: ghost_agent/__init__.py
DPDP §: 8(2)(a) - Data residency compliance
Description: Hospital-side Ghost Agent for federated learning
"""

from .main import GhostAgent
from .emr_wrapper import EMRWrapper
from .local_training import LocalTrainer
from .privacy_engine import PrivacyEngine
from .ghost_pack import GhostPack

__all__ = ["GhostAgent", "EMRWrapper", "LocalTrainer", "PrivacyEngine", "GhostPack"]