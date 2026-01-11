"""
Quantum Vault for Ghost Protocol

Implements post-quantum cryptography for securing federated learning
communications against quantum computer attacks.

DPDP § Citation: §7(1) - Data sovereignty includes quantum-safe encryption
Byzantine Theorem: Post-quantum signatures for Byzantine fault tolerance

Test Command: pytest tests/test_quantum_vault.py -v --cov=sna/quantum_vault

Metrics:
- Quantum Resistance: NIST PQC Level 5
- Signature Size: ≤ 32KB
- Key Generation Time: < 1 second
"""

from .quantum_vault import QuantumVault, PostQuantumCrypto, KyberKEM, DilithiumSignature

__all__ = ['QuantumVault', 'PostQuantumCrypto', 'KyberKEM', 'DilithiumSignature']