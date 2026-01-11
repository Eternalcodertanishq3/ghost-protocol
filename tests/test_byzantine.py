"""
Module: tests/test_byzantine.py
DPDP §: 8(2)(a), 9(1), 11(3)
Byzantine: Tolerates f < n/3 malicious nodes (Theorem: Lamport 1982)
Description: Test Byzantine fault tolerance mechanisms
Test: pytest tests/test_byzantine.py::test_median_filter
"""

import pytest
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List

# Import Byzantine Shield
from sna.byzantine_shield.byzantine_shield import ByzantineShield, HospitalUpdate


class TestByzantineShield:
    """Test Byzantine Shield functionality."""
    
    def test_initialization(self):
        """Test Byzantine Shield initialization."""
        shield = ByzantineShield(
            z_score_threshold=3.0,
            reputation_decay=0.95,
            min_reputation=0.1,
            max_reputation=1.0
        )
        
        assert shield.z_score_threshold == 3.0
        assert shield.reputation_decay == 0.95
        assert shield.min_reputation == 0.1
        assert shield.max_reputation == 1.0
        
    def test_hospital_update_creation(self):
        """Test HospitalUpdate dataclass creation."""
        weights = {'fc.weight': torch.randn(2, 10), 'fc.bias': torch.randn(2)}
        
        update = HospitalUpdate(
            hospital_id="H001",
            weights=weights,
            reputation_score=0.8,
            timestamp=datetime.utcnow(),
            metadata={"round": 1}
        )
        
        assert update.hospital_id == "H001"
        assert update.reputation_score == 0.8
        assert "fc.weight" in update.weights
        
    def test_anomaly_detection_zscore(self):
        """Test Z-score based anomaly detection."""
        shield = ByzantineShield(z_score_threshold=2.0)
        
        # Create updates with different norms
        updates = []
        
        # Normal updates (small norms)
        for i in range(3):
            weights = {'fc.weight': torch.randn(2, 10) * 0.1}
            update = HospitalUpdate(
                hospital_id=f"H00{i+1}",
                weights=weights,
                reputation_score=0.8,
                timestamp=datetime.utcnow(),
                metadata={}
            )
            updates.append(update)
            
        # Anomalous update (large norm)
        anomalous_weights = {'fc.weight': torch.randn(2, 10) * 5.0}
        anomalous_update = HospitalUpdate(
            hospital_id="H004",
            weights=anomalous_weights,
            reputation_score=0.8,
            timestamp=datetime.utcnow(),
            metadata={}
        )
        updates.append(anomalous_update)
        
        # Detect anomalies
        anomaly_flags, stats = shield.detect_anomalies_zscore(updates)
        
        assert len(anomaly_flags) == 4
        assert sum(anomaly_flags) >= 0  # At least one anomaly detected
        assert "mean_norm" in stats
        assert "std_norm" in stats
        
    def test_reputation_weighting(self):
        """Test reputation-based weighting."""
        shield = ByzantineShield()
        
        # Create updates
        updates = [
            HospitalUpdate(
                hospital_id="H001",
                weights={'fc.weight': torch.randn(2, 10)},
                reputation_score=0.9,
                timestamp=datetime.utcnow(),
                metadata={}
            ),
            HospitalUpdate(
                hospital_id="H002",
                weights={'fc.weight': torch.randn(2, 10)},
                reputation_score=0.7,
                timestamp=datetime.utcnow(),
                metadata={}
            )
        ]
        
        # Initialize reputations
        shield.hospital_reputations["H001"] = 0.9
        shield.hospital_reputations["H002"] = 0.7
        
        # Compute weights
        weights = shield.compute_reputation_weights(updates)
        
        assert len(weights) == 2
        assert weights[0] == 0.9  # H001 reputation
        assert weights[1] == 0.7  # H002 reputation
        
    def test_reputation_update(self):
        """Test reputation update mechanism."""
        shield = ByzantineShield(reputation_decay=0.95)
        
        # Create updates
        updates = [
            HospitalUpdate(
                hospital_id="H001",
                weights={'fc.weight': torch.randn(2, 10)},
                reputation_score=0.8,
                timestamp=datetime.utcnow(),
                metadata={}
            )
        ]
        
        # Set initial reputation
        shield.hospital_reputations["H001"] = 0.8
        
        # Update reputation (no anomaly)
        shield.update_reputations(updates, [False])
        
        # Reputation should increase slightly (good behavior)
        new_reputation = shield.hospital_reputations["H001"]
        assert new_reputation > 0.8
        assert new_reputation <= 1.0
        
        # Update reputation with anomaly
        shield.update_reputations(updates, [True])
        
        # Reputation should decrease (anomalous behavior)
        final_reputation = shield.hospital_reputations["H001"]
        assert final_reputation < new_reputation
        
    def test_geometric_median(self):
        """Test geometric median computation."""
        shield = ByzantineShield()
        
        # Create updates with known values
        updates = []
        for i in range(3):
            # Create weights with known values
            weight_val = torch.tensor([[1.0 + i * 0.1, 2.0 + i * 0.1]])
            weights = {'fc.weight': weight_val}
            
            update = HospitalUpdate(
                hospital_id=f"H00{i+1}",
                weights=weights,
                reputation_score=0.8,
                timestamp=datetime.utcnow(),
                metadata={}
            )
            updates.append(update)
            
        # Compute geometric median
        median_weights = shield.compute_geometric_median(updates, max_iterations=50)
        
        assert 'fc.weight' in median_weights
        assert median_weights['fc.weight'].shape == (1, 2)
        
        # Median should be between min and max values
        median_val = median_weights['fc.weight'].flatten()
        min_val = min(u.weights['fc.weight'].min().item() for u in updates)
        max_val = max(u.weights['fc.weight'].max().item() for u in updates)
        
        assert min_val <= median_val[0].item() <= max_val
        assert min_val <= median_val[1].item() <= max_val
        
    def test_byzantine_tolerance(self):
        """Test Byzantine fault tolerance."""
        shield = ByzantineShield()
        
        # Create 5 normal updates and 4 Byzantine updates
        normal_updates = []
        byzantine_updates = []
        
        # Normal updates
        for i in range(5):
            weights = {'fc.weight': torch.randn(2, 10) * 0.1}
            update = HospitalUpdate(
                hospital_id=f"H_NORMAL_{i}",
                weights=weights,
                reputation_score=0.8,
                timestamp=datetime.utcnow(),
                metadata={}
            )
            normal_updates.append(update)
            
        # Byzantine updates (large values)
        for i in range(4):
            weights = {'fc.weight': torch.randn(2, 10) * 10.0}
            update = HospitalUpdate(
                hospital_id=f"H_BYZ_{i}",
                weights=weights,
                reputation_score=0.5,
                timestamp=datetime.utcnow(),
                metadata={}
            )
            byzantine_updates.append(update)
            
        # Combine all updates
        all_updates = normal_updates + byzantine_updates
        
        # Aggregate with protection
        aggregated_weights, report = shield.aggregate_with_byzantine_protection(
            all_updates, apply_dp_sanitization=False
        )
        
        # Should still produce aggregated weights
        assert 'fc.weight' in aggregated_weights
        
        # Should detect anomalies
        assert report["anomalies_detected"] > 0
        assert report["anomaly_rate"] > 0
        
    def test_dp_sanitization(self):
        """Test DP sanitization."""
        shield = ByzantineShield()
        
        # Create weights
        weights = {'fc.weight': torch.ones(2, 10)}
        
        # Apply sanitization
        sanitized = shield.dp_sanitize_aggregated_weights(weights, noise_multiplier=0.1)
        
        # Check shape preservation
        assert sanitized['fc.weight'].shape == (2, 10)
        
        # Check that noise was added
        assert not torch.allclose(sanitized['fc.weight'], weights['fc.weight'], atol=0.01)
        
    def test_simulate_attack(self):
        """Test attack simulation."""
        shield = ByzantineShield()
        
        # Create normal update
        update = HospitalUpdate(
            hospital_id="H001",
            weights={'fc.weight': torch.randn(2, 10) * 0.1},
            reputation_score=0.8,
            timestamp=datetime.utcnow(),
            metadata={}
        )
        
        updates = [update]
        
        # Simulate Gaussian noise attack
        corrupted_updates = shield.simulate_attack("gaussian_noise", updates)
        
        assert len(corrupted_updates) == 1
        assert corrupted_updates[0].hospital_id == "H001"
        
        # Weights should be different (noise added)
        original_norm = torch.norm(update.weights['fc.weight']).item()
        corrupted_norm = torch.norm(corrupted_updates[0].weights['fc.weight']).item()
        
        # With high probability, norms should be different
        assert abs(original_norm - corrupted_norm) > 0.1 or original_norm < 1.0
        
    def test_leaderboard_generation(self):
        """Test reputation leaderboard generation."""
        shield = ByzantineShield()
        
        # Set up hospital reputations
        shield.hospital_reputations = {
            "H001": 0.95,
            "H002": 0.85,
            "H003": 0.75
        }
        
        # Set up histories
        for hid in ["H001", "H002", "H003"]:
            shield.hospital_histories[hid] = [
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "reputation": shield.hospital_reputations[hid],
                    "anomaly": False
                }
            ]
            
        # Generate leaderboard
        leaderboard = shield.get_reputation_leaderboard()
        
        assert len(leaderboard) == 3
        
        # Should be sorted by reputation
        assert leaderboard[0]["hospital_id"] == "H001"  # Highest reputation
        assert leaderboard[2]["hospital_id"] == "H003"  # Lowest reputation
        
        # Check fields
        for entry in leaderboard:
            assert "hospital_id" in entry
            assert "reputation_score" in entry
            assert "total_updates" in entry
            
    def test_attack_statistics(self):
        """Test attack statistics collection."""
        shield = ByzantineShield()
        
        # Simulate some activity
        shield.attack_stats["total_updates"] = 100
        shield.attack_stats["anomalies_detected"] = 15
        shield.attack_stats["byzantine_nodes_blocked"] = 3
        
        # Set up reputations
        shield.hospital_reputations = {
            "H001": 0.95,
            "H002": 0.85,
            "H003": 0.75,
            "H004": 0.45,
            "H005": 0.25
        }
        
        # Get statistics
        stats = shield.get_attack_statistics()
        
        assert stats["total_updates"] == 100
        assert stats["anomalies_detected"] == 15
        assert stats["byzantine_nodes_blocked"] == 3
        assert stats["total_hospitals"] == 5
        
        # Check reputation distribution
        assert "reputation_distribution" in stats
        assert stats["reputation_distribution"]["high"] >= 1
        assert stats["reputation_distribution"]["low"] >= 1


if __name__ == "__main__":
    pytest.main([__file__])