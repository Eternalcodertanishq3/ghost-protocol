"""
Model Marketplace for Ghost Protocol

Enables secure exchange of federated learning models between hospitals
with Byzantine-fault-tolerant validation and HealthToken-based pricing.

DPDP § Citation: §9(4) - Purpose limitation through model sharing
Byzantine Theorem: Byzantine-robust model validation with consensus

Test Command: pytest tests/test_model_marketplace.py -v --cov=sna/model_marketplace

Metrics:
- Transaction Throughput: > 100 models/hour
- Validation Accuracy: > 95%
- Byzantine Tolerance: Up to 33% malicious models
"""

from .model_marketplace import ModelMarketplace, ModelListing, ModelTransaction, ByzantineModelValidator

__all__ = ['ModelMarketplace', 'ModelListing', 'ModelTransaction', 'ByzantineModelValidator']