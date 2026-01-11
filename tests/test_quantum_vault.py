"""
Test suite for Quantum Vault

Tests post-quantum cryptography implementation for quantum-safe communications.

DPDP § Citation: §7(1) - Data sovereignty includes quantum-safe encryption standards
Byzantine Theorem: Post-quantum digital signatures for Byzantine consensus (Boneh et al., 2021)

Test Command: pytest tests/test_quantum_vault.py -v --cov=sna/quantum_vault

Metrics:
- Quantum Resistance: NIST PQC Level 5 (equivalent to AES-256)
- Signature Size: ≤ 32KB (Dilithium-5)
- Key Generation Time: < 1 second
- Decryption Failure Rate: < 10^-6
"""

import pytest
import asyncio
import json
import hashlib
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import redis.asyncio as redis

from sna.quantum_vault import (
    QuantumVault,
    QuantumKeyPair,
    QuantumCertificate,
    KyberKEM,
    DilithiumSignature,
    QuantumSecurityLevel,
    PostQuantumAlgorithm
)


@pytest.fixture
async def redis_client():
    """Create a mock Redis client."""
    client = Mock(spec=redis.Redis)
    client.hset = AsyncMock(return_value=1)
    client.hgetall = AsyncMock(return_value={})
    client.scan_iter = AsyncMock(return_value=[])
    client.lrange = AsyncMock(return_value=[])
    client.lpush = AsyncMock(return_value=1)
    client.delete = AsyncMock(return_value=1)
    return client


@pytest.fixture
async def quantum_vault(redis_client):
    """Create a quantum vault instance."""
    vault = QuantumVault(redis_client, key_rotation_days=90)
    return vault


@pytest.fixture
def sample_keypair():
    """Create a sample quantum key pair."""
    return QuantumKeyPair(
        public_key=b"public_key_data_" + b"x" * 100,
        private_key=b"private_key_data_" + b"y" * 100,
        algorithm=PostQuantumAlgorithm.KYBER_1024,
        security_level=QuantumSecurityLevel.LEVEL_5
    )


@pytest.fixture
def sample_certificate():
    """Create a sample quantum certificate."""
    now = datetime.utcnow()
    return QuantumCertificate(
        hospital_id="hospital_001",
        public_key=b"public_key_certificate_" + b"z" * 100,
        signature_algorithm=PostQuantumAlgorithm.DILITHIUM_5,
        security_level=QuantumSecurityLevel.LEVEL_5,
        issued_at=now,
        expires_at=now + timedelta(days=365),
        signature=b"signature_data_" + b"s" * 64,
        issuer_id="Ghost_Protocol_CA",
        serial_number="cert_serial_123"
    )


class TestQuantumSecurityLevel:
    """Test quantum security levels."""
    
    def test_security_levels(self):
        """Test all security levels are defined."""
        levels = [
            QuantumSecurityLevel.LEVEL_1,
            QuantumSecurityLevel.LEVEL_2,
            QuantumSecurityLevel.LEVEL_3,
            QuantumSecurityLevel.LEVEL_4,
            QuantumSecurityLevel.LEVEL_5
        ]
        
        for level in levels:
            assert isinstance(level, QuantumSecurityLevel)
            assert 1 <= level.value <= 5
            
    def test_security_level_comparison(self):
        """Test security level ordering."""
        assert QuantumSecurityLevel.LEVEL_1 < QuantumSecurityLevel.LEVEL_2
        assert QuantumSecurityLevel.LEVEL_2 < QuantumSecurityLevel.LEVEL_3
        assert QuantumSecurityLevel.LEVEL_3 < QuantumSecurityLevel.LEVEL_4
        assert QuantumSecurityLevel.LEVEL_4 < QuantumSecurityLevel.LEVEL_5


class TestPostQuantumAlgorithm:
    """Test post-quantum algorithms."""
    
    def test_algorithm_values(self):
        """Test algorithm value definitions."""
        algorithms = [
            PostQuantumAlgorithm.KYBER_512,
            PostQuantumAlgorithm.KYBER_768,
            PostQuantumAlgorithm.KYBER_1024,
            PostQuantumAlgorithm.DILITHIUM_2,
            PostQuantumAlgorithm.DILITHIUM_3,
            PostQuantumAlgorithm.DILITHIUM_5,
            PostQuantumAlgorithm.FALCON_512,
            PostQuantumAlgorithm.FALCON_1024,
            PostQuantumAlgorithm.SPHINCS_PLUS
        ]
        
        for algorithm in algorithms:
            assert isinstance(algorithm, PostQuantumAlgorithm)
            assert len(algorithm.value) > 0


class TestQuantumKeyPair:
    """Test quantum key pair management."""
    
    def test_keypair_creation(self, sample_keypair):
        """Test creating a quantum key pair."""
        assert sample_keypair.public_key.startswith(b"public_key_data_")
        assert sample_keypair.private_key.startswith(b"private_key_data_")
        assert sample_keypair.algorithm == PostQuantumAlgorithm.KYBER_1024
        assert sample_keypair.security_level == QuantumSecurityLevel.LEVEL_5
        
    def test_keypair_expiration(self, sample_keypair):
        """Test key pair expiration checking."""
        # Fresh keypair should not be expired
        assert sample_keypair.is_expired() is False
        
        # Create expired keypair
        expired_keypair = QuantumKeyPair(
            public_key=b"expired_public",
            private_key=b"expired_private",
            algorithm=PostQuantumAlgorithm.KYBER_512,
            security_level=QuantumSecurityLevel.LEVEL_1,
            expires_at=datetime.utcnow() - timedelta(days=1)
        )
        
        assert expired_keypair.is_expired() is True
        
    def test_key_id_generation(self, sample_keypair):
        """Test key ID generation from public key."""
        key_id = sample_keypair.get_key_id()
        
        assert len(key_id) == 16
        assert key_id == hashlib.sha256(sample_keypair.public_key).hexdigest()[:16]
        
        # Key ID should be deterministic
        key_id_2 = sample_keypair.get_key_id()
        assert key_id == key_id_2


class TestQuantumCertificate:
    """Test quantum certificate management."""
    
    def test_certificate_creation(self, sample_certificate):
        """Test creating a quantum certificate."""
        assert sample_certificate.hospital_id == "hospital_001"
        assert sample_certificate.signature_algorithm == PostQuantumAlgorithm.DILITHIUM_5
        assert sample_certificate.security_level == QuantumSecurityLevel.LEVEL_5
        assert sample_certificate.issuer_id == "Ghost_Protocol_CA"
        
    def test_certificate_validity(self, sample_certificate):
        """Test certificate validity checking."""
        # Fresh certificate should be valid
        assert sample_certificate.is_valid() is True
        
        # Create expired certificate
        expired_cert = QuantumCertificate(
            hospital_id="hospital_expired",
            public_key=b"expired_public_key",
            signature_algorithm=PostQuantumAlgorithm.DILITHIUM_2,
            security_level=QuantumSecurityLevel.LEVEL_2,
            issued_at=datetime.utcnow() - timedelta(days=400),
            expires_at=datetime.utcnow() - timedelta(days=35),
            signature=b"expired_signature"
        )
        
        assert expired_cert.is_valid() is False
        
    def test_certificate_serialization(self, sample_certificate):
        """Test certificate serialization to dictionary."""
        cert_dict = sample_certificate.to_dict()
        
        assert cert_dict['hospital_id'] == "hospital_001"
        assert cert_dict['signature_algorithm'] == "dilithium_5"
        assert cert_dict['security_level'] == 5
        assert cert_dict['issuer_id'] == "Ghost_Protocol_CA"
        assert len(cert_dict['serial_number']) > 0
        
        # Public key and signature should be hex encoded
        assert isinstance(cert_dict['public_key'], str)
        assert isinstance(cert_dict['signature'], str)


class TestKyberKEM:
    """Test Kyber Key Encapsulation Mechanism."""
    
    @pytest.mark.asyncio
    async def test_kyber_key_generation(self):
        """Test Kyber key pair generation."""
        kyber = KyberKEM()
        
        # Test different security levels
        for level in [QuantumSecurityLevel.LEVEL_1, QuantumSecurityLevel.LEVEL_3, QuantumSecurityLevel.LEVEL_5]:
            keypair = await kyber.generate_keypair(level)
            
            assert isinstance(keypair, QuantumKeyPair)
            assert keypair.algorithm in [PostQuantumAlgorithm.KYBER_512, PostQuantumAlgorithm.KYBER_1024]
            assert keypair.security_level == level
            assert len(keypair.public_key) > 0
            assert len(keypair.private_key) > 0
            
    @pytest.mark.asyncio
    async def test_kyber_encapsulation_decapsulation(self):
        """Test Kyber encapsulation and decapsulation."""
        kyber = KyberKEM()
        
        # Generate key pair
        keypair = await kyber.generate_keypair(QuantumSecurityLevel.LEVEL_5)
        
        # Test encapsulation
        ciphertext, shared_secret = await kyber.encapsulate(keypair.public_key)
        
        assert len(ciphertext) > 0
        assert len(shared_secret) == 32  # 256-bit shared secret
        
        # Test decapsulation
        recovered_secret = await kyber.decapsulate(keypair.private_key, ciphertext)
        
        assert recovered_secret == shared_secret
        
    @pytest.mark.asyncio
    async def test_kyber_encryption_decryption(self):
        """Test Kyber-based encryption and decryption."""
        kyber = KyberKEM()
        
        # Generate key pair
        keypair = await kyber.generate_keypair(QuantumSecurityLevel.LEVEL_3)
        
        # Test data
        plaintext = b"This is secret medical data for federated learning"
        
        # Encrypt
        ciphertext = await kyber.encrypt(keypair.public_key, plaintext)
        
        assert len(ciphertext) > len(plaintext)
        assert ciphertext != plaintext
        
        # Decrypt
        decrypted = await kyber.decrypt(keypair.private_key, ciphertext)
        
        assert decrypted == plaintext


class TestDilithiumSignature:
    """Test Dilithium digital signature implementation."""
    
    @pytest.mark.asyncio
    async def test_dilithium_key_generation(self):
        """Test Dilithium key pair generation."""
        dilithium = DilithiumSignature()
        
        # Test different security levels
        for level in [QuantumSecurityLevel.LEVEL_2, QuantumSecurityLevel.LEVEL_3, QuantumSecurityLevel.LEVEL_5]:
            keypair = await dilithium.generate_keypair(level)
            
            assert isinstance(keypair, QuantumKeyPair)
            assert keypair.algorithm in [
                PostQuantumAlgorithm.DILITHIUM_2,
                PostQuantumAlgorithm.DILITHIUM_3,
                PostQuantumAlgorithm.DILITHIUM_5
            ]
            assert keypair.security_level == level
            
    @pytest.mark.asyncio
    async def test_dilithium_signing_verification(self):
        """Test Dilithium signing and verification."""
        dilithium = DilithiumSignature()
        
        # Generate key pair
        keypair = await dilithium.generate_keypair(QuantumSecurityLevel.LEVEL_5)
        
        # Test data
        message = b"Federated learning model update from hospital"
        
        # Sign
        signature = await dilithium.sign(keypair.private_key, message)
        
        assert len(signature) > 0
        assert isinstance(signature, bytes)
        
        # Verify
        is_valid = await dilithium.verify(keypair.public_key, message, signature)
        
        assert is_valid is True
        
    @pytest.mark.asyncio
    async def test_dilithium_signature_tampering(self):
        """Test signature verification with tampered data."""
        dilithium = DilithiumSignature()
        
        keypair = await dilithium.generate_keypair(QuantumSecurityLevel.LEVEL_3)
        
        # Original message and signature
        message = b"Original message"
        signature = await dilithium.sign(keypair.private_key, message)
        
        # Tampered message
        tampered_message = b"Tampered message"
        
        # Verification should fail
        is_valid = await dilithium.verify(keypair.public_key, tampered_message, signature)
        assert is_valid is False
        
    @pytest.mark.asyncio
    async def test_dilithium_certificate_operations(self):
        """Test Dilithium certificate signing and verification."""
        dilithium = DilithiumSignature()
        
        # Generate CA key pair
        ca_keypair = await dilithium.generate_keypair(QuantumSecurityLevel.LEVEL_5)
        
        # Generate hospital key pair
        hospital_keypair = await dilithium.generate_keypair(QuantumSecurityLevel.LEVEL_5)
        
        # Sign certificate
        certificate = await dilithium.sign_certificate(
            ca_keypair.private_key,
            "hospital_001",
            hospital_keypair.public_key,
            QuantumSecurityLevel.LEVEL_5,
            365
        )
        
        assert isinstance(certificate, QuantumCertificate)
        assert certificate.hospital_id == "hospital_001"
        assert certificate.issuer_id == "Ghost_Protocol_CA"
        
        # Verify certificate
        is_valid = await dilithium.verify_certificate(ca_keypair.public_key, certificate)
        assert is_valid is True


class TestQuantumVault:
    """Test main quantum vault functionality."""
    
    @pytest.mark.asyncio
    async def test_quantum_vault_initialization(self, quantum_vault):
        """Test quantum vault initialization."""
        assert quantum_vault.default_security_level == QuantumSecurityLevel.LEVEL_5
        assert quantum_vault.key_rotation_days == 90
        assert quantum_vault.ca_keypair is None
        
    @pytest.mark.asyncio
    async def test_ca_initialization(self, quantum_vault):
        """Test Certificate Authority initialization."""
        key_id = await quantum_vault.initialize_ca()
        
        assert quantum_vault.ca_keypair is not None
        assert len(key_id) == 16
        assert quantum_vault.ca_keypair.algorithm == PostQuantumAlgorithm.DILITHIUM_5
        
        # Verify CA data was stored
        quantum_vault.redis.hset.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_hospital_key_generation(self, quantum_vault):
        """Test hospital quantum key generation."""
        hospital_id = "hospital_001"
        
        keypair = await quantum_vault.generate_hospital_keypair(hospital_id)
        
        assert isinstance(keypair, QuantumKeyPair)
        assert keypair.security_level == QuantumSecurityLevel.LEVEL_5
        
        # Verify keys were stored
        quantum_vault.redis.hset.assert_called_with(
            f"quantum_keys:{hospital_id}",
            mapping=AsyncMock.ANY
        )
        
    @pytest.mark.asyncio
    async def test_hospital_key_retrieval(self, quantum_vault, sample_keypair):
        """Test retrieving hospital quantum keys."""
        hospital_id = "hospital_001"
        
        # Mock Redis response
        quantum_vault.redis.hgetall = AsyncMock(return_value={
            'encryption_public_key': sample_keypair.public_key.hex(),
            'encryption_private_key': sample_keypair.private_key.hex(),
            'signing_public_key': sample_keypair.public_key.hex(),
            'signing_private_key': sample_keypair.private_key.hex(),
            'security_level': '5',
            'created_at': datetime.utcnow().isoformat(),
            'expires_at': (datetime.utcnow() + timedelta(days=365)).isoformat()
        })
        
        retrieved_keypair = await quantum_vault.get_hospital_keypair(hospital_id)
        
        assert retrieved_keypair is not None
        assert retrieved_keypair.public_key == sample_keypair.public_key
        assert retrieved_keypair.private_key == sample_keypair.private_key
        
    @pytest.mark.asyncio
    async def test_certificate_issuance(self, quantum_vault, sample_keypair):
        """Test quantum certificate issuance."""
        # Initialize CA first
        await quantum_vault.initialize_ca()
        
        hospital_id = "hospital_001"
        
        certificate = await quantum_vault.issue_certificate(
            hospital_id,
            sample_keypair.public_key,
            365
        )
        
        assert isinstance(certificate, QuantumCertificate)
        assert certificate.hospital_id == hospital_id
        assert certificate.security_level == QuantumSecurityLevel.LEVEL_5
        assert certificate.issuer_id == "Ghost_Protocol_CA"
        
        # Verify certificate was stored
        quantum_vault.redis.hset.assert_called_with(
            f"quantum_cert:{hospital_id}",
            mapping=AsyncMock.ANY
        )
        
    @pytest.mark.asyncio
    async def test_certificate_verification(self, quantum_vault, sample_certificate):
        """Test quantum certificate verification."""
        # Initialize CA
        await quantum_vault.initialize_ca()
        
        hospital_id = "hospital_001"
        
        # Mock certificate retrieval
        quantum_vault.get_certificate = AsyncMock(return_value=sample_certificate)
        
        # Mock Dilithium verification to return True
        with patch.object(quantum_vault.dilithium, 'verify_certificate', return_value=True):
            is_valid = await quantum_vault.verify_certificate(hospital_id)
            assert is_valid is True
            
    @pytest.mark.asyncio
    async def test_quantum_secure_encryption(self, quantum_vault, sample_keypair, sample_certificate):
        """Test quantum-safe encryption and decryption."""
        # Initialize CA and generate keys
        await quantum_vault.initialize_ca()
        
        hospital_id = "hospital_001"
        plaintext = b"Secret federated learning gradient updates"
        
        # Mock key and certificate retrieval
        quantum_vault.get_hospital_keypair = AsyncMock(return_value=sample_keypair)
        quantum_vault.get_certificate = AsyncMock(return_value=sample_certificate)
        
        # Encrypt
        ciphertext, session_id = await quantum_vault.quantum_secure_encrypt(
            hospital_id, plaintext
        )
        
        assert len(ciphertext) > 0
        assert len(session_id) == 16
        
        # Decrypt
        quantum_vault.get_hospital_keypair = AsyncMock(return_value=sample_keypair)
        
        decrypted, decrypted_session_id = await quantum_vault.quantum_secure_decrypt(
            hospital_id, ciphertext
        )
        
        assert decrypted == plaintext
        assert decrypted_session_id == session_id
        
    @pytest.mark.asyncio
    async def test_quantum_signature_operations(self, quantum_vault, sample_keypair, sample_certificate):
        """Test quantum-safe signature operations."""
        # Initialize CA and set up keys
        await quantum_vault.initialize_ca()
        
        hospital_id = "hospital_001"
        data = b"Model update from hospital with quantum signature"
        
        # Mock key and certificate retrieval
        quantum_vault.get_hospital_keypair = AsyncMock(return_value=sample_keypair)
        quantum_vault.get_certificate = AsyncMock(return_value=sample_certificate)
        
        # Sign
        signature, signature_id = await quantum_vault.quantum_sign_data(hospital_id, data)
        
        assert len(signature) > 0
        assert len(signature_id) == 16
        
        # Mock certificate verification
        quantum_vault.verify_certificate = AsyncMock(return_value=True)
        
        # Verify signature
        is_valid = await quantum_vault.quantum_verify_signature(hospital_id, data, signature)
        
        assert is_valid is True
        
    @pytest.mark.asyncio
    async def test_key_rotation(self, quantum_vault, sample_keypair):
        """Test quantum key rotation."""
        hospital_id = "hospital_001"
        
        # Mock key generation
        new_keypair = QuantumKeyPair(
            public_key=b"new_public_key",
            private_key=b"new_private_key",
            algorithm=PostQuantumAlgorithm.KYBER_1024,
            security_level=QuantumSecurityLevel.LEVEL_5
        )
        
        quantum_vault.generate_hospital_keypair = AsyncMock(return_value=new_keypair)
        quantum_vault.issue_certificate = AsyncMock(return_value=Mock())
        
        # Mock old key retrieval
        quantum_vault.get_hospital_keypair = AsyncMock(return_value=sample_keypair)
        
        result = await quantum_vault.rotate_keys(hospital_id)
        
        assert result is True
        
        # Verify rotation was logged
        quantum_vault.redis.lpush.assert_called_with("key_rotations", AsyncMock.ANY)
        
    @pytest.mark.asyncio
    async def test_emergency_key_revocation(self, quantum_vault):
        """Test emergency key revocation."""
        hospital_id = "hospital_001"
        reason = "Suspected key compromise detected"
        
        result = await quantum_vault.emergency_key_revocation(hospital_id, reason)
        
        assert result is True
        
        # Verify keys were deleted
        quantum_vault.redis.delete.assert_any_call(f"quantum_keys:{hospital_id}")
        quantum_vault.redis.delete.assert_any_call(f"quantum_cert:{hospital_id}")
        
        # Verify revocation was logged
        quantum_vault.redis.lpush.assert_called_with("emergency_revocations", AsyncMock.ANY)
        
    @pytest.mark.asyncio
    async def test_quantum_security_status(self, quantum_vault):
        """Test quantum security status reporting."""
        # Mock CA info
        quantum_vault.redis.hgetall = AsyncMock(return_value={
            'public_key': 'ca_public_key_hex',
            'algorithm': 'dilithium_5',
            'security_level': '5',
            'created_at': datetime.utcnow().isoformat()
        })
        
        # Mock certificate count
        quantum_vault.redis.scan_iter = AsyncMock(return_value=[
            'quantum_cert:hospital_001',
            'quantum_cert:hospital_002',
            'quantum_cert:hospital_003'
        ])
        
        status = await quantum_vault.get_quantum_security_status()
        
        assert status['active_certificates'] == 3
        assert status['ca_initialized'] is True
        assert status['ca_algorithm'] == 'dilithium_5'
        assert status['default_security_level'] == 5
        assert status['quantum_resistance_level'] == 'NIST_PQC_Level_5'
        assert status['hybrid_mode'] is True


class TestIntegrationScenarios:
    """Test integration scenarios for quantum-safe communications."""
    
    @pytest.mark.asyncio
    async def test_complete_quantum_secure_communication(self, redis_client):
        """Test complete quantum-secure communication flow."""
        # Initialize vaults for two hospitals
        vault_a = QuantumVault(redis_client)
        vault_b = QuantumVault(redis_client)
        
        # Initialize CAs
        await vault_a.initialize_ca()
        await vault_b.initialize_ca()
        
        # Generate key pairs
        keypair_a = await vault_a.generate_hospital_keypair("hospital_A")
        keypair_b = await vault_b.generate_hospital_keypair("hospital_B")
        
        # Issue certificates
        cert_a = await vault_a.issue_certificate("hospital_A", keypair_a.public_key)
        cert_b = await vault_b.issue_certificate("hospital_B", keypair_b.public_key)
        
        # Hospital A encrypts message for Hospital B
        message = b"Federated learning model update with quantum security"
        ciphertext, session_id = await vault_a.quantum_secure_encrypt("hospital_B", message)
        
        # Hospital B decrypts message
        decrypted_message, received_session_id = await vault_b.quantum_secure_decrypt("hospital_B", ciphertext)
        
        assert decrypted_message == message
        assert received_session_id == session_id
        
        # Hospital B signs response
        response = b"Model update received and validated"
        signature, sig_id = await vault_b.quantum_sign_data("hospital_B", response)
        
        # Hospital A verifies signature
        is_valid = await vault_a.quantum_verify_signature("hospital_B", response, signature)
        
        assert is_valid is True
        
    @pytest.mark.asyncio
    async def test_byzantine_resistance_with_quantum_signatures(self, redis_client):
        """Test Byzantine resistance using quantum signatures."""
        vault = QuantumVault(redis_client)
        await vault.initialize_ca()
        
        # Generate keys for multiple hospitals
        hospitals = ["hospital_1", "hospital_2", "hospital_3", "hospital_4"]
        keypairs = {}
        
        for hospital in hospitals:
            keypair = await vault.generate_hospital_keypair(hospital)
            await vault.issue_certificate(hospital, keypair.public_key)
            keypairs[hospital] = keypair
            
        # Simulate Byzantine consensus with quantum signatures
        proposal = b"Proposed model update for round 10"
        signatures = {}
        
        # All hospitals sign the proposal
        for hospital in hospitals:
            signature, sig_id = await vault.quantum_sign_data(hospital, proposal)
            signatures[hospital] = signature
            
        # Verify all signatures
        valid_signatures = 0
        for hospital, signature in signatures.items():
            is_valid = await vault.quantum_verify_signature(hospital, proposal, signature)
            if is_valid:
                valid_signatures += 1
                
        # All signatures should be valid
        assert valid_signatures == len(hospitals)
        
        # Now simulate a Byzantine node with invalid signature
        byzantine_signature = b"invalid_signature_data"
        is_byzantine_valid = await vault.quantum_verify_signature("hospital_1", proposal, byzantine_signature)
        
        assert is_byzantine_valid is False
        
    @pytest.mark.asyncio
    async def test_quantum_key_rotation_workflow(self, redis_client):
        """Test quantum key rotation workflow."""
        vault = QuantumVault(redis_client, key_rotation_days=1)  # Short rotation period for testing
        await vault.initialize_ca()
        
        hospital_id = "hospital_rotation_test"
        
        # Initial key generation
        initial_keypair = await vault.generate_hospital_keypair(hospital_id)
        initial_cert = await vault.issue_certificate(hospital_id, initial_keypair.public_key)
        
        # Simulate key rotation
        with patch('datetime.utcnow', return_value=datetime.utcnow() + timedelta(days=2)):
            rotation_result = await vault.rotate_keys(hospital_id)
            
        assert rotation_result is True
        
        # Verify new keys were generated
        new_keypair = await vault.get_hospital_keypair(hospital_id)
        assert new_keypair is not None
        assert new_keypair.get_key_id() != initial_keypair.get_key_id()
        
        # Verify rotation was logged
        redis_client.lpush.assert_called_with("key_rotations", AsyncMock.ANY)


class TestPerformanceMetrics:
    """Test performance and security metrics."""
    
    @pytest.mark.asyncio
    async def test_key_generation_performance(self, quantum_vault):
        """Test that key generation meets performance requirements."""
        import time
        
        start_time = time.time()
        keypair = await quantum_vault.kyber.generate_keypair(QuantumSecurityLevel.LEVEL_5)
        end_time = time.time()
        
        generation_time = end_time - start_time
        
        # Should generate keys in less than 1 second
        assert generation_time < 1.0, f"Key generation took {generation_time}s, expected < 1s"
        
    @pytest.mark.asyncio
    async def test_signature_size_compliance(self, quantum_vault):
        """Test that signatures meet size requirements."""
        dilithium = DilithiumSignature()
        keypair = await dilithium.generate_keypair(QuantumSecurityLevel.LEVEL_5)
        
        message = b"Test message for signature size verification"
        signature = await dilithium.sign(keypair.private_key, message)
        
        # Dilithium-5 signatures should be ≤ 32KB (4595 bytes actual)
        assert len(signature) <= 32768, f"Signature size {len(signature)} exceeds 32KB limit"
        assert len(signature) <= 4595, f"Signature size {len(signature)} exceeds Dilithium-5 size"
        
    @pytest.mark.asyncio
    async def test_encryption_decryption_consistency(self, quantum_vault):
        """Test encryption/decryption consistency."""
        kyber = KyberKEM()
        
        # Test with multiple messages
        test_messages = [
            b"Small message",
            b"Medium sized message for testing quantum encryption",
            b"Large message " * 100 + b"for testing quantum encryption with longer data"
        ]
        
        for message in test_messages:
            keypair = await kyber.generate_keypair(QuantumSecurityLevel.LEVEL_5)
            
            # Encrypt and decrypt
            ciphertext = await kyber.encrypt(keypair.public_key, message)
            decrypted = await kyber.decrypt(keypair.private_key, ciphertext)
            
            assert decrypted == message, f"Decryption failed for message of length {len(message)}"
            
    @pytest.mark.asyncio
    async def test_security_level_hierarchy(self, quantum_vault):
        """Test that higher security levels provide stronger protection."""
        # This is a conceptual test - in practice, higher security levels
        # would use larger parameter sets
        
        levels = [
            QuantumSecurityLevel.LEVEL_1,
            QuantumSecurityLevel.LEVEL_3,
            QuantumSecurityLevel.LEVEL_5
        ]
        
        key_sizes = []
        
        for level in levels:
            keypair = await quantum_vault.kyber.generate_keypair(level)
            key_sizes.append(len(keypair.public_key))
            
        # Higher security levels should generally have larger key sizes
        assert key_sizes[0] <= key_sizes[1] <= key_sizes[2]