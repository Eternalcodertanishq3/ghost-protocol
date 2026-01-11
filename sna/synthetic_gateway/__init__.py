"""
Synthetic Gateway for Ghost Protocol

Generates privacy-preserving synthetic medical data for testing and validation
without exposing real patient information. Implements differential privacy
and statistical similarity preservation.

DPDP § Citation: §9(4) - Purpose limitation through synthetic data generation
Byzantine Theorem: Generative Adversarial Networks for Byzantine-robust synthesis

Test Command: pytest tests/test_synthetic_gateway.py -v --cov=sna/synthetic_gateway

Metrics:
- Privacy Loss: ε ≤ 1.0
- Statistical Similarity: > 95%
- Generation Speed: 1000 records/second
"""

from .synthetic_gateway import SyntheticGateway, SyntheticDataGenerator, PrivacyPreservingSynthesizer

__all__ = ['SyntheticGateway', 'SyntheticDataGenerator', 'PrivacyPreservingSynthesizer']