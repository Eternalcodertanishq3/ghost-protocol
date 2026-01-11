"""
Module: sna/drift_incentivizer/__init__.py
DPDP §: 9(4) - Data drift detection with privacy preservation
Description: Incentivizes hospitals to report data drift for model adaptation
Byzantine: Drift consensus with Byzantine agreement (tolerates f < n/3 malicious reports)
Test: pytest tests/test_drift.py::test_drift_incentivization
"""

from .drift_incentivizer import DriftIncentivizer

__all__ = ["DriftIncentivizer"]