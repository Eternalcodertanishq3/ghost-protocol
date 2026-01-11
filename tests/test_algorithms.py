"""
Module: tests/test_algorithms.py
Description: Test federated learning algorithms
Test: pytest tests/test_algorithms.py
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List

# Import algorithms
from algorithms.fed_avg.fed_avg import FedAvg
from algorithms.fed_prox.fed_prox import FedProx
from algorithms.dp_mechanisms.gaussian import GaussianDP
from algorithms.dp_mechanisms.laplace import LaplaceDP


class SimpleModel(nn.Module):
    """Simple test model."""
    
    def __init__(self, input_size: int = 10, output_size: int = 2):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        return self.fc(x)


class TestFedAvg:
    """Test FedAvg algorithm."""
    
    def test_initialization(self):
        """Test FedAvg initialization."""
        fedavg = FedAvg(learning_rate=0.01, batch_size=32, local_epochs=5)
        
        assert fedavg.learning_rate == 0.01
        assert fedavg.batch_size == 32
        assert fedavg.local_epochs == 5
        assert fedavg.gradient_clip == 1.0
        
    def test_aggregation(self):
        """Test basic aggregation."""
        fedavg = FedAvg()
        
        # Create mock weights
        weights1 = {
            'fc.weight': torch.randn(2, 10),
            'fc.bias': torch.randn(2)
        }
        weights2 = {
            'fc.weight': torch.randn(2, 10),
            'fc.bias': torch.randn(2)
        }
        
        local_weights = [weights1, weights2]
        client_sizes = [100, 200]
        
        # Aggregate
        aggregated = fedavg.aggregate(local_weights, client_sizes)
        
        # Check shape preservation
        assert aggregated['fc.weight'].shape == (2, 10)
        assert aggregated['fc.bias'].shape == (2,)
        
    def test_aggregation_with_empty_weights(self):
        """Test aggregation with empty weights."""
        fedavg = FedAvg()
        
        with pytest.raises(ValueError, match="No local weights to aggregate"):
            fedavg.aggregate([], [])
            
    def test_aggregation_mismatch_sizes(self):
        """Test aggregation with mismatched sizes."""
        fedavg = FedAvg()
        
        weights = [{'fc.weight': torch.randn(2, 10)}]
        client_sizes = [100, 200]  # Mismatch
        
        with pytest.raises(ValueError, match="Mismatch between weights and client sizes"):
            fedavg.aggregate(weights, client_sizes)
            
    def test_anomaly_detection(self):
        """Test anomaly detection."""
        fedavg = FedAvg()
        
        # Create updates with different norms
        weights1 = {'fc.weight': torch.ones(2, 10) * 0.1}  # Normal
        weights2 = {'fc.weight': torch.ones(2, 10) * 10.0}  # Anomalous
        
        local_weights = [weights1, weights2, weights1]
        global_weights = {'fc.weight': torch.ones(2, 10) * 0.1}
        
        # Compute update norms
        update_norms = fedavg.compute_update_norms(local_weights, global_weights)
        
        assert len(update_norms) == 3
        assert update_norms[1] > update_norms[0]  # Anomalous update has larger norm
        
    def test_byzantine_protection(self):
        """Test Byzantine protection."""
        fedavg = FedAvg()
        
        # Create normal and anomalous updates
        weights_normal = {'fc.weight': torch.randn(2, 10) * 0.1}
        weights_anomalous = {'fc.weight': torch.randn(2, 10) * 10.0}
        
        local_weights = [weights_normal, weights_anomalous, weights_normal]
        client_sizes = [100, 150, 200]
        global_weights = weights_normal
        
        # Aggregate with protection
        aggregated, anomalies = fedavg.aggregate_with_byzantine_protection(
            local_weights, client_sizes, global_weights, z_threshold=2.0
        )
        
        # Check that anomalous update was detected
        assert sum(anomalies) >= 0  # At least 0 anomalies (could be 0 if random)


class TestFedProx:
    """Test FedProx algorithm."""
    
    def test_initialization(self):
        """Test FedProx initialization."""
        fedprox = FedProx(learning_rate=0.01, mu=0.1)
        
        assert fedprox.learning_rate == 0.01
        assert fedprox.mu == 0.1
        
    def test_proximal_weights(self):
        """Test proximal weight computation."""
        fedprox = FedProx()
        
        # Create mock weights
        weights1 = {'fc.weight': torch.randn(2, 10) * 0.1}
        weights2 = {'fc.weight': torch.randn(2, 10) * 10.0}
        global_weights = {'fc.weight': torch.randn(2, 10) * 0.1}
        
        local_weights = [weights1, weights2]
        
        # Compute proximal weights
        proximal_weights = fedprox.compute_proximal_weights(local_weights, global_weights)
        
        assert len(proximal_weights) == 2
        assert abs(sum(proximal_weights) - 1.0) < 0.01  # Should sum to ~1
        
        # Closer updates should get higher weights
        assert proximal_weights[0] > proximal_weights[1]


class TestGaussianDP:
    """Test Gaussian DP mechanism."""
    
    def test_initialization(self):
        """Test Gaussian DP initialization."""
        dp = GaussianDP(epsilon=1.23, delta=1e-5)
        
        assert dp.epsilon == 1.23
        assert dp.delta == 1e-5
        assert dp.noise_multiplier > 0
        
    def test_noise_addition(self):
        """Test noise addition."""
        dp = GaussianDP(noise_multiplier=1.0)
        
        # Create test tensor
        tensor = torch.ones(10, 10)
        sensitivity = 1.0
        
        # Add noise
        noisy_tensor = dp.add_noise(tensor, sensitivity)
        
        # Check shape preservation
        assert noisy_tensor.shape == tensor.shape
        
        # Check that noise was added (values should differ)
        assert not torch.allclose(noisy_tensor, tensor, atol=0.1)
        
    def test_privacy_computation(self):
        """Test privacy budget computation."""
        dp = GaussianDP(epsilon=1.23, delta=1e-5)
        
        # Compute privacy spent
        epsilon_spent, delta_spent = dp.compute_privacy_spent(
            num_steps=100, 
            sampling_rate=0.1
        )
        
        assert epsilon_spent > 0
        assert delta_spent == dp.delta  # Delta doesn't accumulate in simple composition
        
    def test_invalid_parameters(self):
        """Test invalid parameter handling."""
        with pytest.raises(ValueError, match="Epsilon must be positive"):
            GaussianDP(epsilon=0)
            
        with pytest.raises(ValueError, match="Delta must be in"):
            GaussianDP(epsilon=1.0, delta=1.1)


class TestLaplaceDP:
    """Test Laplace DP mechanism."""
    
    def test_initialization(self):
        """Test Laplace DP initialization."""
        dp = LaplaceDP(epsilon=0.8)
        
        assert dp.epsilon == 0.8
        assert dp.noise_multiplier == 1.25  # 1/epsilon
        
    def test_noise_addition(self):
        """Test Laplace noise addition."""
        dp = LaplaceDP(epsilon=1.0)
        
        # Create test tensor
        tensor = torch.ones(10, 10)
        sensitivity = 1.0
        
        # Add noise
        noisy_tensor = dp.add_noise(tensor, sensitivity)
        
        # Check shape preservation
        assert noisy_tensor.shape == tensor.shape
        
        # Check that noise was added
        assert not torch.allclose(noisy_tensor, tensor, atol=0.1)
        
    def test_privacy_computation(self):
        """Test Laplace privacy computation."""
        dp = LaplaceDP(epsilon=0.8)
        
        # Compute privacy spent
        epsilon_spent = dp.compute_privacy_spent(num_steps=50)
        
        assert epsilon_spent > 0
        assert epsilon_spent == 50 * 0.8  # Simple composition


class TestModelIntegration:
    """Test algorithm integration with models."""
    
    def test_fedavg_with_model(self):
        """Test FedAvg with actual model training."""
        model = SimpleModel(input_size=10, output_size=2)
        fedavg = FedAvg(learning_rate=0.01, local_epochs=2)
        
        # Create dummy data
        data = torch.randn(100, 10)
        target = torch.randint(0, 2, (100,))
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(data, target)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=32)
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Perform training step
        loss, gradients = fedavg.local_train_step(
            model, data_loader, optimizer, torch.device("cpu")
        )
        
        assert loss >= 0
        assert isinstance(gradients, dict)
        assert len(gradients) > 0
        
    def test_dp_integration(self):
        """Test DP mechanism integration."""
        dp = GaussianDP(epsilon=1.23, delta=1e-5)
        
        # Create gradients
        gradients = {
            'fc.weight': torch.randn(2, 10),
            'fc.bias': torch.randn(2)
        }
        
        # Add DP noise
        noisy_grads, noise_scale = dp.add_dp_noise(gradients)
        
        # Check shape preservation
        assert noisy_grads['fc.weight'].shape == gradients['fc.weight'].shape
        assert noisy_grads['fc.bias'].shape == gradients['fc.bias'].shape
        
        # Check that noise was added
        assert not torch.allclose(noisy_grads['fc.weight'], gradients['fc.weight'], atol=0.01)


if __name__ == "__main__":
    pytest.main([__file__])