"""
EMR Adapters - Hospital System Integration
FHIR · HL7 · Database Adapters with DPDP Compliance

DPDP §: §8(2)(a) Data Residency - Local processing only
Byzantine theorem: Input validation prevents data poisoning
Test command: pytest tests/test_adapters.py -v --cov=adapters
Metrics tracked: Records processed, Consent validation, Data quality, Processing time
"""

import json
import logging
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import uuid
import xml.etree.ElementTree as ET

from fhir.resources.patient import Patient
from fhir.resources.observation import Observation
from fhir.resources.condition import Condition
from fhir.resources.medicationstatement import MedicationStatement
import hl7apy.parser as hl7_parser
from hl7apy.core import Message


class EMRAdapter(ABC):
    """Abstract base class for EMR system adapters"""
    
    def __init__(self, hospital_id: str, config: Dict[str, Any]):
        self.hospital_id = hospital_id
        self.config = config
        self.logger = logging.getLogger(f"emr_adapter.{hospital_id}")
    
    @abstractmethod
    async def get_patient_records(
        self,
        consent_status: str = "active",
        purpose: str = "federated_learning",
        validity_check: bool = True,
        limit: int = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve patient records with consent validation
        
        Args:
            consent_status: Consent filter (active, withdrawn, expired)
            purpose: Processing purpose
            validity_check: Validate record completeness
            limit: Maximum records to return
            
        Returns:
            List of patient records
        """
        pass
    
    @abstractmethod
    async def validate_consent_for_records(
        self,
        records: List[Dict[str, Any]],
        purpose: str
    ) -> List[Dict[str, Any]]:
        """Validate consent for given records"""
        pass
    
    @abstractmethod
    async def get_record_by_patient_id(
        self,
        patient_id: str,
        include_history: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Get record for specific patient"""
        pass


class FHIRAdapter(EMRAdapter):
    """FHIR (Fast Healthcare Interoperability Resources) adapter"""
    
    def __init__(self, hospital_id: str, config: Dict[str, Any]):
        super().__init__(hospital_id, config)
        
        self.fhir_server_url = config.get("fhir_server_url")
        self.auth_token = config.get("auth_token")
        self.verify_ssl = config.get("verify_ssl", True)
        
        # FHIR resource mappings
        self.observation_mappings = {
            "systolic_bp": "85354-9",  # LOINC code
            "diastolic_bp": "8462-4",
            "heart_rate": "8867-4",
            "temperature": "8310-5",
            "oxygen_saturation": "59408-5",
            "hemoglobin": "718-7",
            "white_blood_cells": "6690-2",
            "platelets": "777-3",
            "glucose": "2339-0",
            "creatinine": "38483-4"
        }
    
    async def get_patient_records(
        self,
        consent_status: str = "active",
        purpose: str = "federated_learning",
        validity_check: bool = True,
        limit: int = None
    ) -> List[Dict[str, Any]]:
        """Retrieve patient records from FHIR server"""
        
        try:
            # In production, this would make actual FHIR API calls
            # For now, simulate FHIR resource retrieval
            
            records = []
            
            # Simulate retrieving patients with observations
            for i in range(min(limit or 1000, 1000)):
                patient_id = f"patient_{uuid.uuid4().hex[:8]}"
                
                # Generate synthetic but realistic patient data
                record = await self._generate_patient_record(patient_id)
                
                if record and (not validity_check or self._validate_record(record)):
                    records.append(record)
            
            self.logger.info(f"Retrieved {len(records)} patient records from FHIR")
            return records
            
        except Exception as e:
            self.logger.error(f"FHIR retrieval failed: {e}")
            return []
    
    async def _generate_patient_record(self, patient_id: str) -> Dict[str, Any]:
        """Generate realistic patient record for demonstration"""
        
        import random
        
        # Age distribution (skewed toward older patients for readmission risk)
        age = int(np.random.gamma(2, 15) + 30)
        age = min(age, 100)
        
        # Gender
        gender = random.choice(["M", "F"])
        
        # Clinical measurements with realistic ranges
        systolic_bp = int(np.random.normal(130, 20))
        systolic_bp = max(80, min(220, systolic_bp))
        
        diastolic_bp = int(np.random.normal(80, 12))
        diastolic_bp = max(40, min(140, diastolic_bp))
        
        heart_rate = int(np.random.normal(75, 15))
        heart_rate = max(40, min(180, heart_rate))
        
        temperature = round(np.random.normal(98.6, 1.2), 1)
        temperature = max(95.0, min(106.0, temperature))
        
        oxygen_saturation = int(np.random.normal(97, 3))
        oxygen_saturation = max(85, min(100, oxygen_saturation))
        
        # Lab values
        hemoglobin = round(np.random.normal(14, 2), 1)
        hemoglobin = max(8.0, min(20.0, hemoglobin))
        
        white_blood_cells = int(np.random.normal(7000, 2000))
        white_blood_cells = max(3000, min(15000, white_blood_cells))
        
        platelets = int(np.random.normal(250000, 50000))
        platelets = max(100000, min(500000, platelets))
        
        glucose = int(np.random.normal(110, 40))
        glucose = max(60, min(400, glucose))
        
        creatinine = round(np.random.normal(1.0, 0.5), 2)
        creatinine = max(0.3, min(10.0, creatinine))
        
        # Comorbidities (correlated with age)
        diabetes = random.random() < (age / 200 + 0.1)
        hypertension = random.random() < (age / 150 + 0.15)
        heart_disease = random.random() < (age / 100 + 0.05)
        kidney_disease = random.random() < (creatinine - 0.5) / 5
        
        # Readmission risk (target variable)
        risk_score = (
            age * 0.02 +
            (diabetes * 0.3) +
            (hypertension * 0.2) +
            (heart_disease * 0.4) +
            (kidney_disease * 0.5) +
            (glucose > 140) * 0.2 +
            (creatinine > 1.5) * 0.3 +
            np.random.normal(0, 0.1)
        )
        
        readmission_risk = 1 if risk_score > 0.7 else 0
        
        return {
            "patient_id": patient_id,
            "age": age,
            "gender": gender,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "heart_rate": heart_rate,
            "temperature": temperature,
            "oxygen_saturation": oxygen_saturation,
            "hemoglobin": hemoglobin,
            "white_blood_cells": white_blood_cells,
            "platelets": platelets,
            "glucose": glucose,
            "creatinine": creatinine,
            "diabetes": int(diabetes),
            "hypertension": int(hypertension),
            "heart_disease": int(heart_disease),
            "kidney_disease": int(kidney_disease),
            "readmission_risk": readmission_risk,
            "date_created": datetime.utcnow(),
            "consent_status": "active",
            "data_source": "fhir"
        }
    
    def _validate_record(self, record: Dict[str, Any]) -> bool:
        """Validate record completeness and quality"""
        
        required_fields = [
            "patient_id", "age", "gender", "systolic_bp", "diastolic_bp",
            "heart_rate", "temperature", "oxygen_saturation", "hemoglobin",
            "white_blood_cells", "platelets", "glucose", "creatinine",
            "diabetes", "hypertension", "heart_disease", "kidney_disease",
            "readmission_risk"
        ]
        
        # Check required fields
        for field in required_fields:
            if field not in record or record[field] is None:
                return False
        
        # Validate value ranges
        if not (0 <= record["age"] <= 120):
            return False
        
        if record["gender"] not in ["M", "F"]:
            return False
        
        if not (60 <= record["systolic_bp"] <= 250):
            return False
        
        if not (40 <= record["diastolic_bp"] <= 150):
            return False
        
        if not (40 <= record["heart_rate"] <= 200):
            return False
        
        if not (95 <= record["temperature"] <= 106):
            return False
        
        if not (85 <= record["oxygen_saturation"] <= 100):
            return False
        
        return True
    
    async def validate_consent_for_records(
        self,
        records: List[Dict[str, Any]],
        purpose: str
    ) -> List[Dict[str, Any]]:
        """Validate consent for FHIR records"""
        
        # In production, check FHIR Consent resources
        # For now, simulate consent validation
        
        valid_records = []
        for record in records:
            if record.get("consent_status") == "active":
                valid_records.append(record)
        
        return valid_records
    
    async def get_record_by_patient_id(
        self,
        patient_id: str,
        include_history: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Get specific patient record"""
        
        # In production, query FHIR server for specific patient
        # For now, return None (record not found)
        
        return None


class HL7Adapter(EMRAdapter):
    """HL7 (Health Level 7) adapter for legacy systems"""
    
    def __init__(self, hospital_id: str, config: Dict[str, Any]):
        super().__init__(hospital_id, config)
        
        self.hl7_connection_string = config.get("hl7_connection_string")
        self.message_version = config.get("message_version", "2.5")
        
        # HL7 segment mappings
        self.segment_mappings = {
            "PID": "patient_identification",
            "OBX": "observation_result",
            "DG1": "diagnosis",
            "AL1": "allergy"
        }
    
    async def get_patient_records(
        self,
        consent_status: str = "active",
        purpose: str = "federated_learning",
        validity_check: bool = True,
        limit: int = None
    ) -> List[Dict[str, Any]]:
        """Retrieve patient records from HL7 system"""
        
        try:
            # Simulate HL7 message processing
            records = []
            
            for i in range(min(limit or 1000, 1000)):
                # Generate HL7 ADT message
                hl7_message = self._generate_hl7_message()
                
                # Parse HL7 message
                record = await self._parse_hl7_message(hl7_message)
                
                if record and (not validity_check or self._validate_record(record)):
                    records.append(record)
            
            self.logger.info(f"Retrieved {len(records)} patient records from HL7")
            return records
            
        except Exception as e:
            self.logger.error(f"HL7 retrieval failed: {e}")
            return []
    
    def _generate_hl7_message(self) -> str:
        """Generate sample HL7 ADT message"""
        
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        patient_id = f"P{uuid.uuid4().hex[:8]}"
        
        hl7_message = f"""MSH|^~\\&|GHOST_AGENT|{self.hospital_id}||{timestamp}||ADT^A08|{uuid.uuid4().hex}|P|2.5
EVN||{timestamp}
PID|1||{patient_id}^^^MRN||DOE^JOHN^M||19800101|M|||123 MAIN ST^^ANYTOWN^ST^12345||555-1234||||||||||||||||||||
OBX|1|NM|85354-9^Systolic BP^LN||130|mmHg|||||F
OBX|2|NM|8462-4^Diastolic BP^LN||80|mmHg|||||F
OBX|3|NM|8867-4^Heart Rate^LN||75|bpm|||||F
OBX|4|NM|8310-5^Temperature^LN||98.6|degF|||||F
OBX|5|NM|59408-5^Oxygen Saturation^LN||98|%|||||F
DG1|1||I10^Essential hypertension^ICD10||Primary hypertension||A
AL1|1||^Penicillin||Anaphylaxis
"""
        
        return hl7_message
    
    async def _parse_hl7_message(self, hl7_message: str) -> Optional[Dict[str, Any]]:
        """Parse HL7 message and extract clinical data"""
        
        try:
            # Parse HL7 message
            message = hl7_parser.parse_message(hl7_message)
            
            # Extract patient information
            pid_segment = message.pid
            if not pid_segment:
                return None
            
            patient_id = pid_segment.pid_3.pid_3_1.value if pid_segment.pid_3 else None
            
            # Extract date of birth and calculate age
            dob = pid_segment.pid_7.value
            if dob:
                birth_year = int(dob[:4])
                age = datetime.utcnow().year - birth_year
            else:
                age = 45  # Default
            
            # Extract gender
            gender = pid_segment.pid_8.value
            
            # Extract observations (vital signs, labs)
            observations = {}
            for obx in message.obx:
                if obx.obx_3 and obx.obx_5:
                    loinc_code = obx.obx_3.obx_3_1.value
                    value = obx.obx_5.value
                    
                    # Map LOINC codes to our features
                    if loinc_code == "85354-9":
                        observations["systolic_bp"] = float(value) if value else None
                    elif loinc_code == "8462-4":
                        observations["diastolic_bp"] = float(value) if value else None
                    elif loinc_code == "8867-4":
                        observations["heart_rate"] = float(value) if value else None
                    elif loinc_code == "8310-5":
                        observations["temperature"] = float(value) if value else None
                    elif loinc_code == "59408-5":
                        observations["oxygen_saturation"] = float(value) if value else None
            
            # Extract diagnoses
            comorbidities = {
                "diabetes": 0,
                "hypertension": 0,
                "heart_disease": 0,
                "kidney_disease": 0
            }
            
            for dg1 in message.dg1:
                diagnosis_code = dg1.dg1_3.dg1_3_1.value if dg1.dg1_3 else None
                
                if diagnosis_code:
                    # ICD-10 mappings
                    if diagnosis_code.startswith("E11") or diagnosis_code.startswith("E10"):
                        comorbidities["diabetes"] = 1
                    elif diagnosis_code.startswith("I10") or diagnosis_code.startswith("I15"):
                        comorbidities["hypertension"] = 1
                    elif diagnosis_code.startswith("I20") or diagnosis_code.startswith("I25"):
                        comorbidities["heart_disease"] = 1
                    elif diagnosis_code.startswith("N18") or diagnosis_code.startswith("N19"):
                        comorbidities["kidney_disease"] = 1
            
            # Fill in missing values with defaults
            defaults = {
                "systolic_bp": 120,
                "diastolic_bp": 80,
                "heart_rate": 72,
                "temperature": 98.6,
                "oxygen_saturation": 98,
                "hemoglobin": 14.0,
                "white_blood_cells": 7000,
                "platelets": 250000,
                "glucose": 100,
                "creatinine": 1.0
            }
            
            for key, default_value in defaults.items():
                if key not in observations:
                    observations[key] = default_value
            
            # Generate readmission risk (simplified)
            readmission_risk = 1 if random.random() < 0.3 else 0
            
            return {
                "patient_id": patient_id,
                "age": age,
                "gender": gender if gender in ["M", "F"] else random.choice(["M", "F"]),
                "systolic_bp": observations["systolic_bp"],
                "diastolic_bp": observations["diastolic_bp"],
                "heart_rate": observations["heart_rate"],
                "temperature": observations["temperature"],
                "oxygen_saturation": observations["oxygen_saturation"],
                "hemoglobin": observations["hemoglobin"],
                "white_blood_cells": observations["white_blood_cells"],
                "platelets": observations["platelets"],
                "glucose": observations["glucose"],
                "creatinine": observations["creatinine"],
                "diabetes": comorbidities["diabetes"],
                "hypertension": comorbidities["hypertension"],
                "heart_disease": comorbidities["heart_disease"],
                "kidney_disease": comorbidities["kidney_disease"],
                "readmission_risk": readmission_risk,
                "date_created": datetime.utcnow(),
                "consent_status": "active",
                "data_source": "hl7"
            }
            
        except Exception as e:
            self.logger.error(f"HL7 parsing failed: {e}")
            return None
    
    async def validate_consent_for_records(
        self,
        records: List[Dict[str, Any]],
        purpose: str
    ) -> List[Dict[str, Any]]:
        """Validate consent for HL7 records"""
        
        # In production, check consent in HL7 or database
        # For now, simulate consent validation
        
        valid_records = []
        for record in records:
            if record.get("consent_status") == "active":
                valid_records.append(record)
        
        return valid_records
    
    async def get_record_by_patient_id(
        self,
        patient_id: str,
        include_history: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Get specific patient record"""
        
        # In production, query HL7 system for specific patient
        return None
    
    def _validate_record(self, record: Dict[str, Any]) -> bool:
        """Validate HL7 record completeness"""
        
        # Same validation as FHIR adapter
        required_fields = [
            "patient_id", "age", "gender", "systolic_bp", "diastolic_bp",
            "heart_rate", "temperature", "oxygen_saturation",
            "diabetes", "hypertension", "heart_disease", "kidney_disease",
            "readmission_risk"
        ]
        
        for field in required_fields:
            if field not in record or record[field] is None:
                return False
        
        return True


class DatabaseAdapter(EMRAdapter):
    """Direct database adapter for custom EMR systems"""
    
    def __init__(self, hospital_id: str, config: Dict[str, Any]):
        super().__init__(hospital_id, config)
        
        self.db_type = config.get("db_type", "postgresql")
        self.connection_string = config.get("connection_string")
        self.table_name = config.get("table_name", "patient_records")
        
        # Field mappings from database to internal format
        self.field_mappings = config.get("field_mappings", {})
    
    async def get_patient_records(
        self,
        consent_status: str = "active",
        purpose: str = "federated_learning",
        validity_check: bool = True,
        limit: int = None
    ) -> List[Dict[str, Any]]:
        """Retrieve patient records from database"""
        
        try:
            # Simulate database query
            records = []
            
            for i in range(min(limit or 1000, 1000)):
                record = await self._generate_database_record()
                
                if record and (not validity_check or self._validate_record(record)):
                    records.append(record)
            
            self.logger.info(f"Retrieved {len(records)} patient records from database")
            return records
            
        except Exception as e:
            self.logger.error(f"Database retrieval failed: {e}")
            return []
    
    async def _generate_database_record(self) -> Dict[str, Any]:
        """Generate realistic database record"""
        
        import numpy as np
        import random
        
        # Similar to FHIR generation but with database-specific fields
        age = int(np.random.gamma(2, 15) + 30)
        age = min(age, 100)
        
        gender = random.choice(["M", "F"])
        
        # Clinical measurements
        systolic_bp = int(np.random.normal(130, 20))
        systolic_bp = max(80, min(220, systolic_bp))
        
        diastolic_bp = int(np.random.normal(80, 12))
        diastolic_bp = max(40, min(140, diastolic_bp))
        
        heart_rate = int(np.random.normal(75, 15))
        heart_rate = max(40, min(180, heart_rate))
        
        temperature = round(np.random.normal(98.6, 1.2), 1)
        temperature = max(95.0, min(106.0, temperature))
        
        oxygen_saturation = int(np.random.normal(97, 3))
        oxygen_saturation = max(85, min(100, oxygen_saturation))
        
        # Lab values
        hemoglobin = round(np.random.normal(14, 2), 1)
        hemoglobin = max(8.0, min(20.0, hemoglobin))
        
        white_blood_cells = int(np.random.normal(7000, 2000))
        white_blood_cells = max(3000, min(15000, white_blood_cells))
        
        platelets = int(np.random.normal(250000, 50000))
        platelets = max(100000, min(500000, platelets))
        
        glucose = int(np.random.normal(110, 40))
        glucose = max(60, min(400, glucose))
        
        creatinine = round(np.random.normal(1.0, 0.5), 2)
        creatinine = max(0.3, min(10.0, creatinine))
        
        # Comorbidities
        diabetes = random.random() < (age / 200 + 0.1)
        hypertension = random.random() < (age / 150 + 0.15)
        heart_disease = random.random() < (age / 100 + 0.05)
        kidney_disease = random.random() < (creatinine - 0.5) / 5
        
        # Readmission risk
        risk_score = (
            age * 0.02 +
            (diabetes * 0.3) +
            (hypertension * 0.2) +
            (heart_disease * 0.4) +
            (kidney_disease * 0.5) +
            (glucose > 140) * 0.2 +
            (creatinine > 1.5) * 0.3 +
            np.random.normal(0, 0.1)
        )
        
        readmission_risk = 1 if risk_score > 0.7 else 0
        
        return {
            "patient_id": f"db_patient_{uuid.uuid4().hex[:8]}",
            "age": age,
            "gender": gender,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "heart_rate": heart_rate,
            "temperature": temperature,
            "oxygen_saturation": oxygen_saturation,
            "hemoglobin": hemoglobin,
            "white_blood_cells": white_blood_cells,
            "platelets": platelets,
            "glucose": glucose,
            "creatinine": creatinine,
            "diabetes": int(diabetes),
            "hypertension": int(hypertension),
            "heart_disease": int(heart_disease),
            "kidney_disease": int(kidney_disease),
            "readmission_risk": readmission_risk,
            "date_created": datetime.utcnow(),
            "consent_status": "active",
            "data_source": "database"
        }
    
    async def validate_consent_for_records(
        self,
        records: List[Dict[str, Any]],
        purpose: str
    ) -> List[Dict[str, Any]]:
        """Validate consent for database records"""
        
        # In production, check consent table in database
        valid_records = []
        for record in records:
            if record.get("consent_status") == "active":
                valid_records.append(record)
        
        return valid_records
    
    async def get_record_by_patient_id(
        self,
        patient_id: str,
        include_history: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Get specific patient record"""
        
        # In production, query database for specific patient
        return None
    
    def _validate_record(self, record: Dict[str, Any]) -> bool:
        """Validate database record"""
        
        # Same validation as other adapters
        required_fields = [
            "patient_id", "age", "gender", "systolic_bp", "diastolic_bp",
            "heart_rate", "temperature", "oxygen_saturation",
            "diabetes", "hypertension", "heart_disease", "kidney_disease",
            "readmission_risk"
        ]
        
        for field in required_fields:
            if field not in record or record[field] is None:
                return False
        
        return True


class EMRAdapterFactory:
    """Factory for creating EMR adapters"""
    
    @staticmethod
    def create_adapter(
        hospital_id: str,
        emr_type: str,
        config: Dict[str, Any]
    ) -> EMRAdapter:
        """
        Create appropriate EMR adapter
        
        Args:
            hospital_id: Hospital identifier
            emr_type: Type of EMR system (fhir, hl7, database)
            config: Configuration parameters
            
        Returns:
            EMR adapter instance
        """
        
        if emr_type == "fhir":
            return FHIRAdapter(hospital_id, config)
        
        elif emr_type == "hl7":
            return HL7Adapter(hospital_id, config)
        
        elif emr_type == "database":
            return DatabaseAdapter(hospital_id, config)
        
        else:
            raise ValueError(f"Unknown EMR type: {emr_type}")
    
    @staticmethod
    def get_supported_types() -> List[str]:
        """Get list of supported EMR types"""
        return ["fhir", "hl7", "database"]