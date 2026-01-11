"""
Module: ghost_agent/emr_wrapper/__init__.py
DPDP §: 8(2)(a) - Data residency compliance
Description: Universal EMR wrapper for HL7/FHIR to NumPy conversion
"""

from .emr_wrapper import EMRWrapper

__all__ = ["EMRWrapper"]