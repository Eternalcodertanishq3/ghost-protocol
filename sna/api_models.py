"""
Module: sna/api_models.py
Description: Pydantic Models for API Request/Response Validation

Ultra-Advanced Features:
- Strict type validation with custom validators
- Comprehensive error messages
- Serialization/deserialization for complex types (torch tensors)
- OpenAPI schema generation
- Request sanitization
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


# ============================================================
# Base Models
# ============================================================

class BaseAPIModel(BaseModel):
    """Base model with common configuration."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        str_min_length=1,
        validate_assignment=True,
        extra='forbid'  # Reject unknown fields
    )


# ============================================================
# Hospital Update Models
# ============================================================

class WeightTensor(BaseModel):
    """Serialized tensor for weight transfer."""
    data: List[float] = Field(..., description="Flattened tensor data")
    shape: List[int] = Field(..., description="Original tensor shape")
    dtype: str = Field(default="float32", description="Data type")
    
    @field_validator('shape')
    @classmethod
    def validate_shape(cls, v: List[int]) -> List[int]:
        if not v:
            raise ValueError("Shape cannot be empty")
        if any(s <= 0 for s in v):
            raise ValueError("Shape dimensions must be positive")
        return v
    
    @field_validator('data')
    @classmethod
    def validate_data(cls, v: List[float]) -> List[float]:
        if not v:
            raise ValueError("Data cannot be empty")
        # Check for NaN/Inf
        import math
        for i, val in enumerate(v[:100]):  # Check first 100 for performance
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"Data contains NaN or Inf at index {i}")
        return v
    
    @model_validator(mode='after')
    def validate_data_shape_match(self):
        """Ensure data length matches shape."""
        expected_size = 1
        for s in self.shape:
            expected_size *= s
        if len(self.data) != expected_size:
            raise ValueError(
                f"Data length {len(self.data)} doesn't match shape {self.shape} "
                f"(expected {expected_size})"
            )
        return self


class HospitalUpdateRequest(BaseAPIModel):
    """Request model for hospital update submission."""
    hospital_id: str = Field(
        ...,
        min_length=3,
        max_length=100,
        description="Unique hospital identifier"
    )
    weights: Dict[str, WeightTensor] = Field(
        ...,
        description="Model weights keyed by layer name"
    )
    round_number: int = Field(
        default=0,
        ge=0,
        description="Training round number"
    )
    local_auc: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Local AUC score"
    )
    gradient_norm: float = Field(
        default=0.0,
        ge=0.0,
        description="Gradient norm from training"
    )
    epsilon_spent: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Differential privacy epsilon spent this round"
    )
    signature: Optional[str] = Field(
        default=None,
        description="Digital signature for authenticity"
    )
    
    @field_validator('hospital_id')
    @classmethod
    def validate_hospital_id(cls, v: str) -> str:
        # Allow alphanumeric, underscore, hyphen only
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError(
                "hospital_id must contain only alphanumeric characters, "
                "underscores, and hyphens"
            )
        return v
    
    @field_validator('weights')
    @classmethod
    def validate_weights(cls, v: Dict[str, WeightTensor]) -> Dict[str, WeightTensor]:
        required_keys = {'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'fc3.weight', 'fc3.bias'}
        provided_keys = set(v.keys())
        
        missing = required_keys - provided_keys
        if missing:
            raise ValueError(f"Missing required weight layers: {missing}")
        
        return v


class HospitalUpdateResponse(BaseAPIModel):
    """Response model for hospital update submission."""
    accepted: bool = Field(..., description="Whether update was accepted")
    message: str = Field(..., description="Status message")
    round_number: int = Field(default=0, description="Current training round")
    tokens_earned: float = Field(default=0.0, description="HealthTokens earned")
    byzantine_detected: bool = Field(default=False, description="Byzantine anomaly flag")
    queue_position: Optional[int] = Field(default=None, description="Position in aggregation queue")


# ============================================================
# Status Models
# ============================================================

class SystemStatusResponse(BaseAPIModel):
    """Response model for system status."""
    service: str = Field(default="Ghost Protocol SNA")
    status: str = Field(default="active")
    dpdp_compliant: bool = Field(default=True)
    sovereignty: str = Field(default="NIC Cloud India")
    current_round: int = Field(default=0)
    active_hospitals: int = Field(default=0)
    model_performance: float = Field(default=0.0)
    pending_updates: int = Field(default=0)
    privacy_budget_remaining: float = Field(default=9.5)
    uptime_seconds: int = Field(default=0)
    version: str = Field(default="1.0.0")


class HealthCheckResponse(BaseAPIModel):
    """Response model for health checks."""
    healthy: bool
    checks: Dict[str, bool] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    details: Optional[Dict[str, str]] = None


# ============================================================
# Leaderboard Models
# ============================================================

class LeaderboardEntry(BaseAPIModel):
    """Single entry in the leaderboard."""
    rank: int = Field(..., ge=1)
    hospital_id: str
    tokens: float = Field(default=0.0)
    reputation: float = Field(default=1.0, ge=0.0, le=1.0)
    updates_submitted: int = Field(default=0)
    last_active: Optional[datetime] = None


class LeaderboardResponse(BaseAPIModel):
    """Response model for leaderboard."""
    entries: List[LeaderboardEntry] = Field(default_factory=list)
    total_hospitals: int = Field(default=0)
    total_tokens_distributed: float = Field(default=0.0)
    as_of_round: int = Field(default=0)


# ============================================================
# Model Download Models
# ============================================================

class GlobalModelResponse(BaseAPIModel):
    """Response model for global model download."""
    weights: Dict[str, WeightTensor] = Field(
        ...,
        description="Global model weights"
    )
    round_number: int = Field(..., description="Round when model was created")
    architecture_hash: str = Field(..., description="Model architecture hash for validation")
    performance: float = Field(default=0.0, description="Current model performance")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================================
# Authentication Models
# ============================================================

class HospitalRegistrationRequest(BaseAPIModel):
    """Request model for hospital registration."""
    hospital_id: str = Field(..., min_length=3, max_length=100)
    hospital_name: str = Field(..., min_length=3, max_length=200)
    admin_email: str = Field(..., pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    public_key: Optional[str] = Field(default=None, description="PEM-encoded public key for mTLS")
    
    @field_validator('hospital_id')
    @classmethod
    def validate_hospital_id(cls, v: str) -> str:
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("Invalid hospital_id format")
        return v


class HospitalRegistrationResponse(BaseAPIModel):
    """Response model for hospital registration."""
    hospital_id: str
    api_key: str = Field(..., description="Secret API key - store securely")
    refresh_token: str = Field(..., description="Token for refreshing access")
    expires_at: datetime
    scopes: List[str] = Field(default_factory=lambda: ["submit_update", "get_model"])


class TokenRefreshRequest(BaseAPIModel):
    """Request model for token refresh."""
    refresh_token: str = Field(..., min_length=20)
    hospital_id: str = Field(..., min_length=3)


class TokenRefreshResponse(BaseAPIModel):
    """Response model for token refresh."""
    access_token: str
    expires_at: datetime
    token_type: str = Field(default="Bearer")


# ============================================================
# Privacy Models
# ============================================================

class PrivacyBudgetRequest(BaseAPIModel):
    """Request model for privacy budget query."""
    hospital_id: str = Field(..., min_length=3)


class PrivacyBudgetResponse(BaseAPIModel):
    """Response model for privacy budget."""
    hospital_id: str
    epsilon_spent: float = Field(ge=0.0)
    epsilon_remaining: float = Field(ge=0.0)
    hard_limit: float = Field(default=9.5)
    warning_threshold: float = Field(default=8.0)
    status: str = Field(default="compliant")  # compliant, warning, halt
    rounds_participated: int = Field(default=0)


# ============================================================
# Error Models
# ============================================================

class ErrorDetail(BaseAPIModel):
    """Detailed error information."""
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable message")
    field: Optional[str] = Field(default=None, description="Field that caused error")
    suggestion: Optional[str] = Field(default=None, description="How to fix")


class ErrorResponse(BaseAPIModel):
    """Standard error response."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: List[ErrorDetail] = Field(default_factory=list)
    request_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
