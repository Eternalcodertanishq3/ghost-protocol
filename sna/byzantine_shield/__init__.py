"""
Module: sna/byzantine_shield/__init__.py
DPDP §: 9(4) - Byzantine-robust aggregation
Description: Byzantine Shield for malicious node detection
"""

from .byzantine_shield import ByzantineShield, HospitalUpdate, ModelUpdate, AggregationStrategy, ByzantineAnalysisResult

__all__ = ["ByzantineShield", "HospitalUpdate", "ModelUpdate", "AggregationStrategy", "ByzantineAnalysisResult"]