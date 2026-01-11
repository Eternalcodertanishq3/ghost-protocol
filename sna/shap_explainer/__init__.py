"""
Module: sna/shap_explainer/__init__.py
DPDP §: 9(4) - Model interpretability for privacy-preserving AI
Description: SHAP (SHapley Additive exPlanations) for federated model interpretability
Byzantine: SHAP values computed locally, aggregated with geometric median
Test: pytest tests/test_shap.py::test_shap_explanation
"""

from .shap_explainer import SHAPExplainer

__all__ = ["SHAPExplainer"]