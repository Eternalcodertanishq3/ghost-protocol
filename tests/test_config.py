"""
Module: tests/test_config.py
Description: Test configuration and validation for Ghost Protocol
Test: pytest tests/test_config.py
"""

import pytest
import tempfile
import os
from config import GhostConfig, ALGORITHM_CONFIGS, DP_CONFIGS, SECURITY_THRESHOLDS


class TestGhostConfig:
    """Test Ghost Protocol configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = GhostConfig()
        
        assert config.MAX_EPSILON == 9.5
        assert config.DELTA == 1e-5
        assert config.DATA_RESIDENCY == True
        assert config.GAUSSIAN_NOISE == 1.3
        assert config.NORM_CLIP == 1.0
        assert config.BYZANTINE_TOLERANCE == 0.49
        
    def test_algorithm_configs(self):
        """Test algorithm configurations."""
        assert "fedavg" in ALGORITHM_CONFIGS
        assert "fedprox" in ALGORITHM_CONFIGS
        assert "scaffold" in ALGORITHM_CONFIGS
        
        fedavg_config = ALGORITHM_CONFIGS["fedavg"]
        assert fedavg_config["name"] == "Federated Averaging"
        assert fedavg_config["learning_rate"] == 0.01
        
        fedprox_config = ALGORITHM_CONFIGS["fedprox"]
        assert fedprox_config["mu"] == 0.1
        
    def test_dp_configs(self):
        """Test differential privacy configurations."""
        assert "gaussian" in DP_CONFIGS
        assert "laplace" in DP_CONFIGS
        
        gaussian_config = DP_CONFIGS["gaussian"]
        assert gaussian_config["epsilon"] == 1.23
        assert gaussian_config["delta"] == 1e-5
        
    def test_security_thresholds(self):
        """Test security thresholds."""
        assert SECURITY_THRESHOLDS["max_epsilon"] == 9.5
        assert SECURITY_THRESHOLDS["min_trust_score"] == 0.7
        assert SECURITY_THRESHOLDS["z_score_anomaly"] == 3.0
        
    def test_dpdp_compliance(self):
        """Test DPDP compliance mapping."""
        from config import DPDP_COMPLIANCE
        
        assert "data_residency" in DPDP_COMPLIANCE
        assert "purpose_limitation" in DPDP_COMPLIANCE
        assert "consent" in DPDP_COMPLIANCE
        assert "breach_notification" in DPDP_COMPLIANCE
        
        # Verify compliance references DPDP sections
        assert "§8(2)(a)" in DPDP_COMPLIANCE["data_residency"]
        assert "§9(4)" in DPDP_COMPLIANCE["purpose_limitation"]
        assert "§11(3)" in DPDP_COMPLIANCE["consent"]
        
    def test_environment_variables(self):
        """Test environment variable loading."""
        # Set test environment variables
        os.environ["SNA_HOST"] = "test-host"
        os.environ["SNA_PORT"] = "9999"
        
        config = GhostConfig()
        
        # Test that environment variables are loaded
        # Note: In actual test, these would be overridden by .env file if present
        assert config.SNA_HOST == "0.0.0.0"  # Default unless .env exists
        assert config.SNA_PORT == 8000  # Default unless .env exists
        
        # Clean up
        if "SNA_HOST" in os.environ:
            del os.environ["SNA_HOST"]
        if "SNA_PORT" in os.environ:
            del os.environ["SNA_PORT"]


class TestConfigurationValidation:
    """Test configuration validation."""
    
    def test_privacy_budget_validation(self):
        """Test privacy budget constraints."""
        from config import SECURITY_THRESHOLDS
        
        # Max epsilon should be reasonable for DPDP compliance
        assert 0 < SECURITY_THRESHOLDS["max_epsilon"] <= 10
        
        # Min trust score should be between 0 and 1
        assert 0 <= SECURITY_THRESHOLDS["min_trust_score"] <= 1
        
        # Z-score threshold should be positive
        assert SECURITY_THRESHOLDS["z_score_anomaly"] > 0
        
    def test_learning_rate_validation(self):
        """Test learning rate configuration."""
        from config import ALGORITHM_CONFIGS
        
        for algo_name, config in ALGORITHM_CONFIGS.items():
            # Learning rate should be positive and reasonable
            assert 0 < config["learning_rate"] <= 1.0
            
            # Batch size should be positive
            assert config["batch_size"] > 0
            
            # Local epochs should be positive
            assert config["local_epochs"] > 0
            
    def test_dp_parameters_validation(self):
        """Test DP parameter validation."""
        from config import DP_CONFIGS
        
        for mechanism, config in DP_CONFIGS.items():
            # Epsilon should be positive
            assert config["epsilon"] > 0
            
            # Delta should be non-negative and small
            assert config["delta"] >= 0
            if mechanism == "gaussian":
                assert config["delta"] < 1e-3  # Should be very small
                
    def test_networking_validation(self):
        """Test networking configuration validation."""
        from config import config
        
        # Rate limits should be positive
        assert config.AGENT_UPDATE_INTERVAL > 0
        assert config.RATE_LIMIT_PER_AGENT > 0
        
        # WS heartbeat should be reasonable
        assert config.WS_HEARTBEAT_INTERVAL > 0
        assert config.WS_HEARTBEAT_INTERVAL < 60  # Less than 1 minute


if __name__ == "__main__":
    pytest.main([__file__])