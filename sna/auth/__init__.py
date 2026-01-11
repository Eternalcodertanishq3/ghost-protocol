"""
Module: sna/auth/__init__.py
DPDP §: 7(1) - Sovereignty, §8(2)(a) - Security Safeguards
Description: JWT + mTLS-ready Authentication for Ghost Protocol

Ultra-Advanced Features:
- JWT-based token authentication with RS256/EdDSA
- Hospital registration with certificate issuance
- Rate limiting per hospital
- mTLS-ready infrastructure (certificate validation hooks)
- Token rotation and revocation
"""

from .auth import (
    HospitalAuthenticator,
    AuthenticationError,
    create_access_token,
    decode_access_token,
    get_current_hospital,
    require_auth
)

__all__ = [
    "HospitalAuthenticator",
    "AuthenticationError",
    "create_access_token",
    "decode_access_token",
    "get_current_hospital",
    "require_auth"
]
