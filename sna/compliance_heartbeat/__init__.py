"""
Module: sna/compliance_heartbeat/__init__.py
DPDP §: 25 - Continuous compliance monitoring and breach notification
Description: Real-time compliance heartbeat monitoring system
Byzantine: Heartbeat consensus with fault tolerance (tolerates f < n/3 failures)
Test: pytest tests/test_compliance.py::test_heartbeat_monitoring
"""

from .compliance_heartbeat import ComplianceHeartbeat

__all__ = ["ComplianceHeartbeat"]