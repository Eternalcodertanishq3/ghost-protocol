"""
Module: models/registry.py
Description: Shared Model Registry - Single Source of Truth for Model Architectures

Ultra-Advanced Features:
- Centralized model registration and versioning
- Architecture hash for compatibility checking
- Automatic weight shape validation
- Model serialization/deserialization
- Multi-model support for different prediction tasks
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Type

import torch
import torch.nn as nn

logger = logging.getLogger("ghost.models")


@dataclass
class ModelMetadata:
    """Metadata for a registered model."""
    name: str
    version: str
    input_size: int
    hidden_size: int
    output_size: int
    architecture_hash: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "architecture_hash": self.architecture_hash,
            "created_at": self.created_at.isoformat(),
            "description": self.description
        }


# ============================================================
# Model Definitions - Single Source of Truth
# ============================================================

class DiabetesPredictionModel(nn.Module):
    """
    Primary model for diabetes risk prediction.
    
    Architecture:
    - Input: 8 features (age, BP, glucose, etc.)
    - Hidden: 64 -> 32 neurons with ReLU, Dropout
    - Output: 1 (sigmoid for binary classification)
    
    This is the SINGLE SOURCE OF TRUTH for this model.
    Import from here in both SNA and hospital agents.
    """
    
    def __init__(self, input_size: int = 8, hidden_size: int = 64):
        super(DiabetesPredictionModel, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


class ReadmissionPredictionModel(nn.Module):
    """
    Extended model for hospital readmission prediction.
    
    Architecture:
    - Input: 16 features (demographics, vitals, diagnoses, etc.)
    - Hidden: 128 -> 64 -> 32 neurons with ReLU, BatchNorm, Dropout
    - Output: 1 (sigmoid for binary classification)
    """
    
    def __init__(self, input_size: int = 16, hidden_size: int = 128):
        super(ReadmissionPredictionModel, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.1)
        
        self.fc4 = nn.Linear(hidden_size // 4, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x


# Backwards compatibility aliases
SimpleNN = DiabetesPredictionModel


# ============================================================
# Model Registry
# ============================================================

class ModelRegistry:
    """
    Centralized Model Registry - Single Source of Truth.
    
    Solves the model duplication problem by providing:
    - Single definition of each model architecture
    - Version tracking and compatibility checking
    - Automatic hash-based architecture validation
    - Factory pattern for model instantiation
    """
    
    _instance: Optional["ModelRegistry"] = None
    _models: Dict[str, Type[nn.Module]] = {}
    _metadata: Dict[str, ModelMetadata] = {}
    _factories: Dict[str, Callable[..., nn.Module]] = {}
    _initialized: bool = False
    
    def __new__(cls):
        """Singleton pattern for global registry."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def _ensure_initialized(cls):
        """Ensure models are registered on first access."""
        if not cls._initialized:
            cls._register_default_models()
            cls._initialized = True
    
    @classmethod
    def _register_default_models(cls):
        """Register the default models."""
        # Register DiabetesPredictionModel
        cls._register_model(
            model_cls=DiabetesPredictionModel,
            name="diabetes_prediction",
            version="1.0.0",
            input_size=8,
            hidden_size=64,
            output_size=1,
            description="Binary classification for diabetes risk prediction"
        )
        
        # Register ReadmissionPredictionModel
        cls._register_model(
            model_cls=ReadmissionPredictionModel,
            name="readmission_prediction",
            version="1.0.0",
            input_size=16,
            hidden_size=128,
            output_size=1,
            description="Hospital readmission risk prediction with extended features"
        )
    
    @classmethod
    def _register_model(
        cls,
        model_cls: Type[nn.Module],
        name: str,
        version: str,
        input_size: int,
        hidden_size: int,
        output_size: int,
        description: str = ""
    ):
        """Internal method to register a model."""
        # Create instance to compute architecture hash
        try:
            instance = model_cls(input_size=input_size, hidden_size=hidden_size)
            arch_hash = cls._compute_architecture_hash(instance)
        except TypeError:
            instance = model_cls()
            arch_hash = cls._compute_architecture_hash(instance)
        
        # Store registration
        cls._models[name] = model_cls
        cls._metadata[name] = ModelMetadata(
            name=name,
            version=version,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            architecture_hash=arch_hash,
            description=description
        )
        
        logger.debug(f"Registered model: {name} v{version} (hash: {arch_hash[:8]}...)")
    
    @classmethod
    def register_factory(cls, name: str, factory: Callable[..., nn.Module]):
        """Register a factory function for creating models."""
        cls._factories[name] = factory
    
    @classmethod
    def _compute_architecture_hash(cls, model: nn.Module) -> str:
        """Compute SHA-256 hash of model architecture."""
        arch_str = ""
        for name, param in model.named_parameters():
            arch_str += f"{name}:{list(param.shape)};"
        
        return hashlib.sha256(arch_str.encode()).hexdigest()
    
    @classmethod
    def get(cls, name: str, **kwargs) -> nn.Module:
        """
        Get a model instance by name.
        
        Args:
            name: Registered model name
            **kwargs: Arguments to pass to model constructor
            
        Returns:
            Instantiated model
        """
        cls._ensure_initialized()
        
        if name in cls._factories:
            return cls._factories[name](**kwargs)
        
        if name not in cls._models:
            available = list(cls._models.keys())
            raise ValueError(f"Model '{name}' not registered. Available: {available}")
        
        model_cls = cls._models[name]
        metadata = cls._metadata[name]
        
        # Use default values from metadata if not provided
        final_kwargs = {
            "input_size": kwargs.get("input_size", metadata.input_size),
            "hidden_size": kwargs.get("hidden_size", metadata.hidden_size)
        }
        
        return model_cls(**final_kwargs)
    
    @classmethod
    def get_metadata(cls, name: str) -> Optional[ModelMetadata]:
        """Get metadata for a registered model."""
        cls._ensure_initialized()
        return cls._metadata.get(name)
    
    @classmethod
    def validate_weights(cls, name: str, weights: Dict[str, torch.Tensor]) -> bool:
        """
        Validate that weights are compatible with a model.
        
        Args:
            name: Model name
            weights: Dictionary of weight tensors
            
        Returns:
            True if weights are compatible
        """
        cls._ensure_initialized()
        
        if name not in cls._models:
            return False
        
        # Create reference instance
        model = cls.get(name)
        model_state = model.state_dict()
        
        # Check all keys match
        if set(weights.keys()) != set(model_state.keys()):
            logger.warning(
                f"Weight keys mismatch for {name}: "
                f"expected {set(model_state.keys())}, got {set(weights.keys())}"
            )
            return False
        
        # Check shapes match
        for key in weights:
            if weights[key].shape != model_state[key].shape:
                logger.warning(
                    f"Shape mismatch for {name}.{key}: "
                    f"expected {model_state[key].shape}, got {weights[key].shape}"
                )
                return False
        
        return True
    
    @classmethod
    def check_compatibility(cls, name: str, architecture_hash: str) -> bool:
        """Check if weights from another system are compatible by hash."""
        cls._ensure_initialized()
        metadata = cls._metadata.get(name)
        if not metadata:
            return False
        return metadata.architecture_hash == architecture_hash
    
    @classmethod
    def list_models(cls) -> List[Dict[str, Any]]:
        """List all registered models with metadata."""
        cls._ensure_initialized()
        return [metadata.to_dict() for metadata in cls._metadata.values()]
    
    @classmethod
    def export_model(cls, name: str, model: nn.Module) -> Dict[str, Any]:
        """Export model with metadata for sharing."""
        cls._ensure_initialized()
        metadata = cls._metadata.get(name)
        if not metadata:
            raise ValueError(f"Model '{name}' not registered")
        
        return {
            "name": name,
            "version": metadata.version,
            "architecture_hash": metadata.architecture_hash,
            "weights": {k: v.tolist() for k, v in model.state_dict().items()},
            "exported_at": datetime.utcnow().isoformat()
        }
    
    @classmethod
    def import_model(cls, data: Dict[str, Any]) -> nn.Module:
        """Import model from exported data with validation."""
        cls._ensure_initialized()
        name = data["name"]
        arch_hash = data["architecture_hash"]
        
        # Validate compatibility
        if not cls.check_compatibility(name, arch_hash):
            metadata = cls._metadata.get(name)
            if metadata:
                raise ValueError(
                    f"Architecture mismatch for {name}: "
                    f"expected {metadata.architecture_hash[:16]}..., "
                    f"got {arch_hash[:16]}..."
                )
            raise ValueError(f"Model '{name}' not registered")
        
        # Create model and load weights
        model = cls.get(name)
        weights = {k: torch.tensor(v) for k, v in data["weights"].items()}
        model.load_state_dict(weights)
        
        return model


# Convenience function
def get_model(name: str, **kwargs) -> nn.Module:
    """Convenience function to get a model from the registry."""
    return ModelRegistry.get(name, **kwargs)
