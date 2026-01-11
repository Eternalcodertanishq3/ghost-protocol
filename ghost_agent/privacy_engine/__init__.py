"""
Module: ghost_agent/privacy_engine/__init__.py
DPDP §: 9(4) - Privacy preservation through differential privacy
Description: Privacy engine with Gaussian DP and sparsity
"""

from .privacy_engine import PrivacyEngine

__all__ = ["PrivacyEngine"]