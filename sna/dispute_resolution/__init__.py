"""
Dispute Resolution System for Ghost Protocol

Implements Byzantine-fault-tolerant dispute resolution mechanism
for handling conflicts between hospitals in federated learning rounds.

DPDP § Citation: §15(3) - Right to rectification includes dispute resolution
Byzantine Theorem: Byzantine Agreement with n ≥ 3f + 1 nodes (Castro & Liskov, 1999)

Test Command: pytest tests/test_dispute_resolution.py -v --cov=sna/dispute_resolution

Metrics:
- Dispute Resolution Time: < 30 seconds
- Byzantine Tolerance: Up to 33% malicious hospitals
- Resolution Accuracy: > 95% consensus rate
"""

from .dispute_resolution import DisputeResolution, DisputeCase, DisputeVerdict

__all__ = ['DisputeResolution', 'DisputeCase', 'DisputeVerdict']