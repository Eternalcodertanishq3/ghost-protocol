"""
Module: config.py
DPDP §: 8(2)(a), 9(4), 11(3)
Description: Global configuration for Ghost Protocol
"""

import os
from typing import Dict, Any
from pydantic import Field
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


class GhostConfig(BaseSettings):
    """Ghost Protocol Configuration - DPDP Compliant"""
    
    # DPDP Compliance Settings
    MAX_EPSILON: float = 9.5  # Hard stop for privacy budget (DPDP §9(4))
    DELTA: float = 1e-5  # DP failure probability
    DATA_RESIDENCY: bool = True  # Hospital data never leaves LAN (DPDP §8(2)(a))
    
    # Privacy Engine Settings
    GAUSSIAN_NOISE: float = 1.3  # Standard deviation for Gaussian DP
    LAPLACE_SCALE: float = 0.5  # Scale for Laplace mechanism
    NORM_CLIP: float = 1.0  # Gradient clipping threshold
    SPARSITY_TOP_PCT: float = 0.01  # Top 1% sparsity
    
    # Byzantine Shield Settings
    BYZANTINE_TOLERANCE: float = 0.49  # Tolerate up to 49% malicious nodes
    GEOMETRIC_MEDIAN_MAX_ITER: int = 100
    REPUTATION_DECAY: float = 0.95  # Shapley value decay rate
    Z_SCORE_THRESHOLD: float = 3.0  # Anomaly detection threshold
    
    # Federated Learning Settings
    LEARNING_RATE: float = 0.01
    BATCH_SIZE: int = 32
    LOCAL_EPOCHS: int = 5
    GLOBAL_ROUNDS: int = 1000
    
    # FedProx Settings
    FEDPROX_MU: float = 0.1
    
    # Clustered FL Settings
    CLUSTERED_FL_K: int = 5
    
    # Communication Settings
    AGENT_UPDATE_INTERVAL: int = 30  # seconds
    RATE_LIMIT_PER_AGENT: int = 1  # update per interval
    WS_HEARTBEAT_INTERVAL: int = 1  # second
    
    # Security Settings
    TLS_VERSION: str = "1.3"
    CERT_ROTATION_DAYS: int = 90
    ECDSA_CURVE: str = "P-256"
    VAULT_ADDR: str = Field(default="http://localhost:8200")
    VAULT_TOKEN: str = Field(default="ghost-root-token")
    
    # Network Settings
    SNA_HOST: str = Field(default="0.0.0.0")
    SNA_PORT: int = Field(default=8000)
    AGENT_HOST: str = Field(default="0.0.0.0")
    AGENT_PORT: int = Field(default=8001)
    REDIS_URL: str = Field(default="redis://localhost:6379")
    
    # Model Settings
    MODEL_INPUT_SIZE: int = 784  # MNIST example
    MODEL_HIDDEN_SIZE: int = 256
    MODEL_OUTPUT_SIZE: int = 10
    
    # HealthToken Settings
    HEALTHTOKEN_DECIMALS: int = 18
    MIN_TRUST_SCORE: float = 0.7  # Minimum for rewards
    
    # Attack Simulation
    SIMULATION_ENABLED: bool = False  # Disable in production
    DEMO_MODE: bool = False  # Set to True only via .env for demos
    
    # Cryptography
    SALT: str = Field(default="ghost-protocol-salt-v1")  # Override in .env for production
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global configuration instance
config = GhostConfig()


# Algorithm configurations
ALGORITHM_CONFIGS: Dict[str, Dict[str, Any]] = {
    "fedavg": {
        "name": "Federated Averaging",
        "learning_rate": 0.01,
        "batch_size": 32,
        "local_epochs": 5
    },
    "fedprox": {
        "name": "FedProx",
        "learning_rate": 0.01,
        "batch_size": 32,
        "local_epochs": 5,
        "mu": 0.1  # Proximal term coefficient
    },
    "scaffold": {
        "name": "SCAFFOLD",
        "learning_rate": 0.01,
        "batch_size": 32,
        "local_epochs": 5,
        "control_variate_lr": 0.1
    },
    "clustered_fl": {
        "name": "Clustered Federated Learning",
        "learning_rate": 0.01,
        "batch_size": 32,
        "local_epochs": 5,
        "n_clusters": 5
    }
}


# DP Mechanism configurations
DP_CONFIGS: Dict[str, Dict[str, float]] = {
    "gaussian": {
        "noise_multiplier": 1.3,
        "epsilon": 1.23,
        "delta": 1e-5
    },
    "laplace": {
        "scale": 0.5,
        "epsilon": 0.8,
        "delta": 0.0  # Pure DP
    },
    "renyi": {
        "alpha": 10.0,
        "epsilon": 1.5,
        "delta": 1e-5
    }
}


# Security thresholds
SECURITY_THRESHOLDS: Dict[str, float] = {
    "max_epsilon": 9.5,
    "min_trust_score": 0.7,
    "max_latency_ms": 2000,
    "max_bandwidth_kb": 500,
    "max_accuracy_drop_pct": 5.0,
    "max_privacy_leakage_pct": 10.0,
    "z_score_anomaly": 3.0
}


# DPDP Legal compliance mapping
DPDP_COMPLIANCE: Dict[str, str] = {
    "data_residency": "§8(2)(a) - Data stored and processed within India",
    "purpose_limitation": "§9(4) - Gradients encrypted, not readable",
    "consent": "§11(3) - Local opt-in required before training",
    "breach_notification": "§25 - Auto-alert if ε > 9.5",
    "sovereignty": "§7(1) - SNA hosted on NIC Cloud India only",
    "right_to_forget": "§15 - Hospital can purge model updates"
}