"""
Module: sna/adaptive_clustering/__init__.py
DPDP §: 9(4) - Non-IID data clustering for improved FL performance
Description: Adaptive clustering of hospitals based on data similarity and performance
Byzantine: Cluster assignments robust to malicious hospitals (tolerates f < n/3 per cluster)
Test: pytest tests/test_clustering.py::test_adaptive_clustering
"""

from .adaptive_clustering import AdaptiveClustering

__all__ = ["AdaptiveClustering"]