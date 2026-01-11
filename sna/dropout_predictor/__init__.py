"""
Module: sna/dropout_predictor/__init__.py
DPDP §: 9(4) - Predictive analytics for hospital participation
Description: Predicts hospital dropout risk to maintain federated learning stability
Byzantine: Dropout predictions aggregated with geometric median (tolerates f < n/2 malicious)
Test: pytest tests/test_dropout.py::test_dropout_prediction
"""

from .dropout_predictor import DropoutPredictor

__all__ = ["DropoutPredictor"]