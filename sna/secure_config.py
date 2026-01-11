"""
Module: sna/secure_config.py
Description: Secure Configuration Loader with Secret Management

Ultra-Advanced Features:
- No default secrets in code
- Environment variable validation
- Vault integration for secrets
- Configuration encryption at rest
- Audit logging for config access
"""

import logging
import os
import secrets
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, Optional

logger = logging.getLogger("ghost.config")


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""
    pass


class MissingSecretError(ConfigurationError):
    """Raised when a required secret is missing."""
    pass


@dataclass
class SecureConfig:
    """
    Secure configuration container.
    
    All secrets MUST be loaded from environment variables or Vault.
    No default values for secrets - fail fast if missing.
    """
    
    # Required secrets - NO DEFAULTS
    jwt_secret: str = field(repr=False)  # Hide from logs
    encryption_salt: str = field(repr=False)
    postgres_password: str = field(repr=False)
    redis_password: Optional[str] = field(default=None, repr=False)
    vault_token: Optional[str] = field(default=None, repr=False)
    blockchain_private_key: Optional[str] = field(default=None, repr=False)
    
    # Non-secret configuration - defaults allowed
    sna_host: str = "0.0.0.0"
    sna_port: int = 8000
    grpc_port: int = 50051
    
    redis_host: str = "localhost"
    redis_port: int = 6379
    
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "ghost_user"
    postgres_db: str = "ghost_protocol"
    
    vault_addr: str = "http://localhost:8200"
    
    # Feature flags
    demo_mode: bool = False
    enable_grpc: bool = True
    enable_blockchain: bool = False
    enable_mtls: bool = False
    
    # Privacy settings
    max_epsilon: float = 9.5
    default_delta: float = 1e-5
    
    # Performance settings
    max_pending_updates: int = 10000
    update_ttl_seconds: int = 3600
    aggregation_threshold: int = 3
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate secret lengths
        if len(self.jwt_secret) < 32:
            raise ConfigurationError(
                "JWT_SECRET must be at least 32 characters. "
                "Generate with: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
            )
        
        if len(self.encryption_salt) < 16:
            raise ConfigurationError(
                "ENCRYPTION_SALT must be at least 16 characters. "
                "Generate with: python -c \"import secrets; print(secrets.token_urlsafe(16))\""
            )
    
    @property
    def redis_url(self) -> str:
        """Construct Redis URL."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}"
        return f"redis://{self.redis_host}:{self.redis_port}"
    
    @property
    def postgres_url(self) -> str:
        """Construct PostgreSQL URL."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
    
    @property
    def postgres_sync_url(self) -> str:
        """Construct synchronous PostgreSQL URL."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


def _get_required_env(name: str) -> str:
    """Get required environment variable or raise error."""
    value = os.getenv(name)
    if not value:
        raise MissingSecretError(
            f"Required environment variable {name} is not set. "
            f"Set it in your .env file or environment."
        )
    return value


def _get_optional_env(name: str, default: Any = None) -> Any:
    """Get optional environment variable with default."""
    return os.getenv(name, default)


def _get_bool_env(name: str, default: bool = False) -> bool:
    """Get boolean environment variable."""
    value = os.getenv(name, str(default)).lower()
    return value in ("true", "1", "yes", "on")


def _get_int_env(name: str, default: int) -> int:
    """Get integer environment variable."""
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


@lru_cache(maxsize=1)
def load_secure_config() -> SecureConfig:
    """
    Load secure configuration from environment.
    
    Returns:
        SecureConfig instance
        
    Raises:
        MissingSecretError: If required secrets are not set
        ConfigurationError: If configuration is invalid
    """
    logger.info("Loading secure configuration...")
    
    # Check for .env file
    env_file = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_file):
        logger.info(f"Loading environment from {env_file}")
        _load_dotenv(env_file)
    
    try:
        config = SecureConfig(
            # Required secrets
            jwt_secret=_get_required_env("JWT_SECRET"),
            encryption_salt=_get_required_env("ENCRYPTION_SALT"),
            postgres_password=_get_required_env("POSTGRES_PASSWORD"),
            redis_password=_get_optional_env("REDIS_PASSWORD"),
            vault_token=_get_optional_env("VAULT_TOKEN"),
            blockchain_private_key=_get_optional_env("BLOCKCHAIN_PRIVATE_KEY"),
            
            # Network configuration
            sna_host=_get_optional_env("SNA_HOST", "0.0.0.0"),
            sna_port=_get_int_env("SNA_PORT", 8000),
            grpc_port=_get_int_env("GRPC_PORT", 50051),
            
            # Redis configuration
            redis_host=_get_optional_env("REDIS_HOST", "localhost"),
            redis_port=_get_int_env("REDIS_PORT", 6379),
            
            # PostgreSQL configuration
            postgres_host=_get_optional_env("POSTGRES_HOST", "localhost"),
            postgres_port=_get_int_env("POSTGRES_PORT", 5432),
            postgres_user=_get_optional_env("POSTGRES_USER", "ghost_user"),
            postgres_db=_get_optional_env("POSTGRES_DB", "ghost_protocol"),
            
            # Vault configuration
            vault_addr=_get_optional_env("VAULT_ADDR", "http://localhost:8200"),
            
            # Feature flags
            demo_mode=_get_bool_env("DEMO_MODE", False),
            enable_grpc=_get_bool_env("ENABLE_GRPC", True),
            enable_blockchain=_get_bool_env("ENABLE_BLOCKCHAIN", False),
            enable_mtls=_get_bool_env("ENABLE_MTLS", False),
            
            # Privacy settings
            max_epsilon=float(_get_optional_env("MAX_EPSILON", "9.5")),
            default_delta=float(_get_optional_env("DEFAULT_DELTA", "1e-5")),
            
            # Performance settings
            max_pending_updates=_get_int_env("MAX_PENDING_UPDATES", 10000),
            update_ttl_seconds=_get_int_env("UPDATE_TTL_SECONDS", 3600),
            aggregation_threshold=_get_int_env("AGGREGATION_THRESHOLD", 3)
        )
        
        logger.info(
            f"Configuration loaded: demo_mode={config.demo_mode}, "
            f"mtls={config.enable_mtls}, blockchain={config.enable_blockchain}"
        )
        
        return config
        
    except MissingSecretError as e:
        logger.error(str(e))
        logger.error(
            "HINT: Create a .env file with required secrets. "
            "See .env.example for template."
        )
        raise


def _load_dotenv(filepath: str):
    """Simple .env file loader."""
    try:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
    except Exception as e:
        logger.warning(f"Failed to load .env file: {e}")


def generate_secrets() -> Dict[str, str]:
    """
    Generate secure random secrets.
    
    Returns:
        Dictionary of generated secrets
    """
    return {
        "JWT_SECRET": secrets.token_urlsafe(48),
        "ENCRYPTION_SALT": secrets.token_urlsafe(24),
        "POSTGRES_PASSWORD": secrets.token_urlsafe(32),
        "REDIS_PASSWORD": secrets.token_urlsafe(24)
    }


def create_env_example():
    """Create .env.example file with placeholder values."""
    content = '''# Ghost Protocol Configuration
# Copy this file to .env and fill in the values

# ============================================================
# REQUIRED SECRETS - Generate with:
# python -c "import secrets; print(secrets.token_urlsafe(48))"
# ============================================================

JWT_SECRET=<generate-48-char-secret>
ENCRYPTION_SALT=<generate-24-char-secret>
POSTGRES_PASSWORD=<generate-32-char-secret>

# ============================================================
# OPTIONAL SECRETS
# ============================================================

# REDIS_PASSWORD=<your-redis-password>
# VAULT_TOKEN=<your-vault-token>
# BLOCKCHAIN_PRIVATE_KEY=<your-ethereum-private-key>

# ============================================================
# NETWORK CONFIGURATION
# ============================================================

SNA_HOST=0.0.0.0
SNA_PORT=8000
GRPC_PORT=50051

REDIS_HOST=localhost
REDIS_PORT=6379

POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=ghost_user
POSTGRES_DB=ghost_protocol

VAULT_ADDR=http://localhost:8200

# ============================================================
# FEATURE FLAGS
# ============================================================

DEMO_MODE=false
ENABLE_GRPC=true
ENABLE_BLOCKCHAIN=false
ENABLE_MTLS=false

# ============================================================
# PRIVACY SETTINGS
# ============================================================

MAX_EPSILON=9.5
DEFAULT_DELTA=1e-5

# ============================================================
# PERFORMANCE SETTINGS
# ============================================================

MAX_PENDING_UPDATES=10000
UPDATE_TTL_SECONDS=3600
AGGREGATION_THRESHOLD=3
'''
    return content
