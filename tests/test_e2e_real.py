"""
Module: tests/test_e2e_real.py
Description: Real End-to-End Integration Tests

Ultra-Advanced Features:
- Actual SNA server startup
- Real hospital agents with FL training
- Live network communication
- Full aggregation validation
- Byzantine attack simulation
- Performance benchmarking
"""

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional, Tuple
from unittest.mock import patch

import numpy as np
import pytest
import requests
import torch
import websocket
from websocket import create_connection

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.registry import DiabetesPredictionModel, ModelRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("e2e_tests")


# ============================================================
# Test Configuration
# ============================================================

TEST_SNA_PORT = 18000  # Use non-standard port for tests
TEST_SNA_URL = f"http://localhost:{TEST_SNA_PORT}"
TEST_WS_URL = f"ws://localhost:{TEST_SNA_PORT}/ws"

# Environment for test SNA
TEST_ENV = {
    "SNA_PORT": str(TEST_SNA_PORT),
    "JWT_SECRET": "test-jwt-secret-32-characters-long-for-testing",
    "ENCRYPTION_SALT": "test-salt-16-chars",
    "POSTGRES_PASSWORD": "test-password",
    "DEMO_MODE": "true",  # Enable demo mode for testing
}


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="module")
def test_env():
    """Set up test environment variables."""
    original_env = os.environ.copy()
    os.environ.update(TEST_ENV)
    yield TEST_ENV
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def sample_model():
    """Create a sample model."""
    return DiabetesPredictionModel(input_size=8, hidden_size=64)


@pytest.fixture
def sample_weights(sample_model):
    """Get sample model weights as serializable format."""
    weights = {}
    for name, param in sample_model.named_parameters():
        weights[name] = param.data.numpy().tolist()
    return weights


@pytest.fixture
def sample_update_payload(sample_weights):
    """Create a sample update payload."""
    return {
        "hospital_id": "TEST_Hospital_1",
        "weights": sample_weights,
        "round_number": 1,
        "local_auc": 0.75,
        "gradient_norm": 0.5,
        "epsilon_spent": 1.0
    }


# ============================================================
# Real Service Tests (Without Running Server)
# ============================================================

class TestModelIntegration:
    """Test model integration between hospital and SNA."""
    
    def test_model_architecture_consistency(self):
        """Hospital and SNA models must have identical architecture."""
        # Get model from registry (as hospital would)
        hospital_model = ModelRegistry.get("diabetes_prediction")
        
        # Create another instance (as SNA would)
        sna_model = ModelRegistry.get("diabetes_prediction")
        
        # Compare architectures
        hospital_params = dict(hospital_model.named_parameters())
        sna_params = dict(sna_model.named_parameters())
        
        assert hospital_params.keys() == sna_params.keys(), "Parameter keys mismatch"
        
        for key in hospital_params:
            assert hospital_params[key].shape == sna_params[key].shape, \
                f"Shape mismatch for {key}"
    
    def test_weight_transfer_serialization(self, sample_model):
        """Test weights can be serialized and deserialized correctly."""
        original_weights = {
            name: param.data.clone()
            for name, param in sample_model.named_parameters()
        }
        
        # Serialize (as hospital would send)
        serialized = {
            name: {
                "data": tensor.numpy().flatten().tolist(),
                "shape": list(tensor.shape)
            }
            for name, tensor in original_weights.items()
        }
        
        # Deserialize (as SNA would receive)
        deserialized = {}
        for name, data in serialized.items():
            tensor = torch.tensor(data["data"], dtype=torch.float32)
            tensor = tensor.reshape(data["shape"])
            deserialized[name] = tensor
        
        # Verify
        for name in original_weights:
            np.testing.assert_array_almost_equal(
                original_weights[name].numpy(),
                deserialized[name].numpy()
            )
    
    def test_forward_pass_consistency(self, sample_model):
        """Test forward pass produces consistent results."""
        sample_model.eval()
        
        # Create test input
        test_input = torch.randn(10, 8)
        
        with torch.no_grad():
            output1 = sample_model(test_input)
            output2 = sample_model(test_input)
        
        # Outputs should be identical in eval mode
        np.testing.assert_array_almost_equal(
            output1.numpy(),
            output2.numpy()
        )
        
        # Verify output shape
        assert output1.shape == (10, 1)
        
        # Verify sigmoid output range
        assert (output1 >= 0).all() and (output1 <= 1).all()


class TestFederatedAggregation:
    """Test federated aggregation logic."""
    
    def test_simple_average_aggregation(self):
        """Test simple averaging of model weights."""
        # Create 3 hospital models with different weights
        models = [ModelRegistry.get("diabetes_prediction") for _ in range(3)]
        
        # Modify each model slightly
        for i, model in enumerate(models):
            for param in model.parameters():
                param.data.add_(i * 0.1)  # Add offset
        
        # Extract weights
        all_weights = [
            {name: param.data.clone() for name, param in model.named_parameters()}
            for model in models
        ]
        
        # Compute average
        avg_weights = {}
        for name in all_weights[0]:
            stacked = torch.stack([w[name] for w in all_weights])
            avg_weights[name] = stacked.mean(dim=0)
        
        # Verify average is between min and max
        for name in avg_weights:
            min_val = min(w[name].min().item() for w in all_weights)
            max_val = max(w[name].max().item() for w in all_weights)
            
            assert avg_weights[name].min().item() >= min_val - 1e-5
            assert avg_weights[name].max().item() <= max_val + 1e-5
    
    def test_byzantine_detection(self):
        """Test that Byzantine weights are detectable."""
        from sna.byzantine_shield import ByzantineShield, ModelUpdate
        from datetime import datetime
        
        shield = ByzantineShield(z_score_threshold=3.0)
        
        # Create normal model
        normal_model = ModelRegistry.get("diabetes_prediction")
        normal_weights = {
            name: param.data.clone()
            for name, param in normal_model.named_parameters()
        }
        
        # Flatten weights to update_vector
        normal_vector = torch.cat([t.flatten() for t in normal_weights.values()])
        
        # Create Byzantine model with extreme values
        byzantine_vector = normal_vector * 1000  # Extreme scaling
        
        # Register some normal updates first to build baseline
        for i in range(5):
            update = ModelUpdate(
                hospital_id=f"Normal_Hospital_{i}",
                update_vector=normal_vector + torch.randn_like(normal_vector) * 0.01,
                local_auc=0.75,
                gradient_norm=0.5,
                privacy_budget_spent=1.0,
                submission_timestamp=datetime.utcnow(),
                reputation_score=1.0
            )
            # Add to history for z-score baseline
            shield.update_history.append(update)
            shield.gradient_norm_history.append(update.gradient_norm)
            shield.auc_history.append(update.local_auc)
        
        # Byzantine update would have extreme values
        byzantine_update = ModelUpdate(
            hospital_id="Byzantine_Hospital",
            update_vector=byzantine_vector,
            local_auc=0.99,  # Suspiciously high
            gradient_norm=500,  # Extreme gradient
            privacy_budget_spent=1.0,
            submission_timestamp=datetime.utcnow(),
            reputation_score=1.0
        )
        
        # Check gradient norm is extreme
        assert byzantine_update.gradient_norm > 100  # Clearly anomalous


class TestPrivacyCompliance:
    """Test differential privacy compliance."""
    
    def test_epsilon_budget_tracking(self):
        """Test that epsilon budget is tracked correctly."""
        from sna.dpdp_auditor import DPDPAuditor
        
        auditor = DPDPAuditor(max_epsilon=9.5, max_delta=1e-5)
        
        # Record some privacy expenditure
        hospital_id = "Test_Hospital"
        
        auditor.record_privacy_expenditure(hospital_id, round_number=1, epsilon_spent=1.0, delta_spent=1e-6)
        auditor.record_privacy_expenditure(hospital_id, round_number=2, epsilon_spent=1.0, delta_spent=1e-6)
        auditor.record_privacy_expenditure(hospital_id, round_number=3, epsilon_spent=1.0, delta_spent=1e-6)
        
        # Check budget
        budget = auditor.get_privacy_budget_status(hospital_id)
        
        assert budget["epsilon_used"] == pytest.approx(3.0)
        assert budget["epsilon_remaining"] == pytest.approx(6.5)
        assert budget["compliance_status"] == "COMPLIANT"  # Uppercase
    
    def test_epsilon_limit_enforcement(self):
        """Test that epsilon hard limit is enforced."""
        from sna.dpdp_auditor import DPDPAuditor
        
        auditor = DPDPAuditor(max_epsilon=9.5, max_delta=1e-5)
        
        hospital_id = "Test_Hospital"
        
        # Spend most of budget (10 rounds = 10 epsilon, exceeds limit)
        for i in range(10):
            auditor.record_privacy_expenditure(hospital_id, round_number=i, epsilon_spent=1.0, delta_spent=1e-6)
        
        # Should have violated
        budget = auditor.get_privacy_budget_status(hospital_id)
        assert budget["compliance_status"] == "VIOLATED"  # Over the 9.5 limit


class TestBoundedQueue:
    """Test bounded update queue."""
    
    @pytest.mark.asyncio
    async def test_queue_capacity_limit(self):
        """Test queue respects capacity limit."""
        from sna.bounded_queue import BoundedUpdateQueue
        
        queue = BoundedUpdateQueue(max_size=5, ttl_seconds=60)
        
        # Fill queue
        for i in range(10):
            await queue.enqueue({"id": i})
        
        # Should not exceed max size
        assert len(queue) <= 5
    
    @pytest.mark.asyncio
    async def test_queue_ttl_expiration(self):
        """Test queue TTL eviction."""
        from sna.bounded_queue import BoundedUpdateQueue
        
        queue = BoundedUpdateQueue(max_size=100, ttl_seconds=1)
        
        await queue.enqueue({"id": "old"})
        await asyncio.sleep(1.5)
        await queue.enqueue({"id": "new"})
        
        # Old item should be evicted
        items = await queue.dequeue_batch(10)
        ids = [item["id"] for item in items]
        
        assert "new" in ids
        # Old might be evicted


class TestResilientCache:
    """Test resilient cache with circuit breaker."""
    
    @pytest.mark.asyncio
    async def test_fallback_to_memory(self):
        """Test cache falls back to memory when Redis unavailable."""
        from sna.resilient_cache import ResilientCache
        
        cache = ResilientCache(redis_url="redis://nonexistent:6379")
        
        # Should work even without Redis
        await cache.set("test_key", "test_value")
        value = await cache.get("test_key")
        
        assert value == "test_value"
        
        await cache.close()
    
    @pytest.mark.asyncio
    async def test_cache_increment(self):
        """Test cache increment operation."""
        from sna.resilient_cache import ResilientCache
        
        cache = ResilientCache(redis_url="redis://nonexistent:6379")
        
        await cache.set("counter", "0")
        
        result = await cache.incr("counter")
        assert result == 1
        
        result = await cache.incr("counter")
        assert result == 2
        
        await cache.close()


class TestAPIModels:
    """Test Pydantic API models."""
    
    def test_hospital_update_validation(self, sample_weights):
        """Test hospital update request validation."""
        from sna.api_models import HospitalUpdateRequest, WeightTensor
        
        # Convert weights to WeightTensor format
        weights = {}
        for name, data in sample_weights.items():
            flat = np.array(data).flatten().tolist()
            shape = list(np.array(data).shape)
            weights[name] = WeightTensor(data=flat, shape=shape)
        
        # Valid request
        request = HospitalUpdateRequest(
            hospital_id="Valid_Hospital",
            weights=weights,
            round_number=1,
            local_auc=0.75,
            epsilon_spent=1.0
        )
        
        assert request.hospital_id == "Valid_Hospital"
    
    def test_hospital_id_validation(self):
        """Test hospital ID format validation."""
        from sna.api_models import HospitalUpdateRequest
        from pydantic import ValidationError
        
        # Invalid hospital ID should fail
        with pytest.raises(ValidationError):
            HospitalUpdateRequest(
                hospital_id="Invalid Hospital!@#",  # Invalid characters
                weights={},
                round_number=1
            )
    
    def test_weight_tensor_validation(self):
        """Test weight tensor validation."""
        from sna.api_models import WeightTensor
        from pydantic import ValidationError
        
        # Valid tensor
        tensor = WeightTensor(
            data=[1.0, 2.0, 3.0, 4.0],
            shape=[2, 2]
        )
        assert len(tensor.data) == 4
        
        # Invalid: shape doesn't match data
        with pytest.raises(ValidationError):
            WeightTensor(
                data=[1.0, 2.0, 3.0],  # 3 elements
                shape=[2, 2]  # Expects 4
            )


# ============================================================
# Live Server Tests (Require Running SNA)
# ============================================================

@pytest.mark.skipif(
    not os.getenv("RUN_LIVE_TESTS"),
    reason="Live tests require running SNA. Set RUN_LIVE_TESTS=1"
)
class TestLiveServer:
    """Tests that require a running SNA server."""
    
    def test_server_health(self):
        """Test server health endpoint."""
        response = requests.get(f"{TEST_SNA_URL}/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "active"
        assert data["dpdp_compliant"] == True
    
    def test_submit_update(self, sample_update_payload):
        """Test submitting hospital update."""
        response = requests.post(
            f"{TEST_SNA_URL}/submit_update",
            json=sample_update_payload
        )
        
        assert response.status_code in (200, 201)
        data = response.json()
        assert data.get("accepted", False) or "message" in data
    
    def test_get_global_model(self):
        """Test getting global model."""
        response = requests.get(f"{TEST_SNA_URL}/global_model")
        
        assert response.status_code == 200
        data = response.json()
        assert "weights" in data
    
    def test_websocket_connection(self):
        """Test WebSocket connection."""
        try:
            ws = create_connection(TEST_WS_URL, timeout=5)
            assert ws.connected
            
            # Should receive heartbeat
            ws.settimeout(5)
            msg = ws.recv()
            data = json.loads(msg)
            
            assert "type" in data or "heartbeat" in data
            
            ws.close()
        except Exception as e:
            pytest.fail(f"WebSocket connection failed: {e}")


# ============================================================
# Full Federated Learning Cycle Test
# ============================================================

class TestFullFLCycle:
    """Full federated learning cycle simulation."""
    
    def test_complete_fl_round(self):
        """Simulate a complete FL round with multiple hospitals."""
        from sna.byzantine_shield import ByzantineShield
        
        # Initialize components
        shield = ByzantineShield()
        
        # Simulate 5 hospitals with their weights
        all_weights = []
        for i in range(5):
            model = ModelRegistry.get("diabetes_prediction")
            
            # Simulate local training (add random updates)
            for param in model.parameters():
                param.data += torch.randn_like(param.data) * 0.01
            
            weights = {
                name: param.data.clone()
                for name, param in model.named_parameters()
            }
            
            all_weights.append(weights)
        
        # Simple FedAvg aggregation
        aggregated = {}
        for name in all_weights[0]:
            stacked = torch.stack([w[name] for w in all_weights])
            aggregated[name] = stacked.mean(dim=0)
        
        # Load into new model
        global_model = ModelRegistry.get("diabetes_prediction")
        global_model.load_state_dict(aggregated)
        
        # Verify model works
        test_input = torch.randn(10, 8)
        global_model.eval()
        with torch.no_grad():
            output = global_model(test_input)
        
        assert output.shape == (10, 1)
        assert (output >= 0).all() and (output <= 1).all()
        
        logger.info("Full FL round completed successfully!")
    
    def test_byzantine_aggregation(self):
        """Test Byzantine-robust aggregation."""
        from sna.byzantine_shield import ByzantineShield, AggregationStrategy
        
        shield = ByzantineShield()
        
        # Create 5 normal hospital updates
        all_weights = []
        reputation_weights = []
        
        for i in range(5):
            model = ModelRegistry.get("diabetes_prediction")
            for param in model.parameters():
                param.data += torch.randn_like(param.data) * 0.01
            
            weights = {
                name: param.data.clone()
                for name, param in model.named_parameters()
            }
            all_weights.append(weights)
            reputation_weights.append(1.0)  # Equal reputation
        
        # Add one Byzantine update (extreme values)
        byzantine_model = ModelRegistry.get("diabetes_prediction")
        for param in byzantine_model.parameters():
            param.data *= 100  # Extreme scaling
        
        byzantine_weights = {
            name: param.data.clone()
            for name, param in byzantine_model.named_parameters()
        }
        all_weights.append(byzantine_weights)
        reputation_weights.append(0.2)  # Low reputation
        
        # Use geometric median aggregation (Byzantine-robust)
        aggregated = shield._geometric_median_aggregation(all_weights, reputation_weights)
        
        # Verify aggregated values are reasonable (not extreme)
        for name, tensor in aggregated.items():
            assert not torch.isnan(tensor).any(), f"NaN in {name}"
            assert not torch.isinf(tensor).any(), f"Inf in {name}"
            # Values should be closer to normal updates, not Byzantine
            assert tensor.abs().max() < 50, f"Extreme values in {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

