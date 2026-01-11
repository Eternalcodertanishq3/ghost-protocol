"""
Ghost Agent Adapters Module
EMR adapters for FHIR, HL7, and database integration
"""

from .emr_adapters import EMRAdapter, FHIRAdapter, HL7Adapter, DatabaseAdapter

__all__ = ['EMRAdapter', 'FHIRAdapter', 'HL7Adapter', 'DatabaseAdapter']
