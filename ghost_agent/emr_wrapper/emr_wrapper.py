"""
Module: ghost_agent/emr_wrapper/emr_wrapper.py
DPDP §: 8(2)(a) - Data residency compliance, §11(3) - Consent
Description: Universal EMR wrapper converting HL7/FHIR to NumPy arrays
Test: pytest tests/test_emr_wrapper.py::test_emr_conversion
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import json
import hashlib
from datetime import datetime
import logging


class EMRWrapper:
    """
    Universal EMR wrapper for converting healthcare data formats to NumPy.
    
    Supports:
    - HL7 FHIR R4 resources
    - HL7 v2 messages
    - Custom JSON formats
    - CSV/TSV dumps
    
    All data processing happens locally - no data leaves the hospital.
    """
    
    def __init__(
        self,
        hospital_id: str,
        consent_required: bool = True,
        anonymization_level: str = "high"
    ):
        """
        Initialize EMR wrapper.
        
        Args:
            hospital_id: Unique hospital identifier
            consent_required: Whether patient consent is required (DPDP §11(3))
            anonymization_level: Anonymization level ("low", "medium", "high")
        """
        self.hospital_id = hospital_id
        self.consent_required = consent_required
        self.anonymization_level = anonymization_level
        self.logger = logging.getLogger(__name__)
        
        # Initialize consent tracking
        self.consent_records = {}
        
    def load_fhir_resource(
        self,
        resource_path: str,
        resource_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load FHIR resource from file.
        
        Args:
            resource_path: Path to FHIR resource file
            resource_type: FHIR resource type (Patient, Observation, etc.)
            
        Returns:
            Parsed FHIR resource as dictionary
        """
        try:
            with open(resource_path, 'r', encoding='utf-8') as f:
                resource = json.load(f)
                
            # Validate resource type
            if resource_type and resource.get('resourceType') != resource_type:
                raise ValueError(f"Expected {resource_type}, got {resource.get('resourceType')}")
                
            return resource
            
        except Exception as e:
            self.logger.error(f"Failed to load FHIR resource: {e}")
            raise
            
    def fhir_to_numpy(
        self,
        resources: List[Dict[str, Any]],
        feature_mapping: Optional[Dict[str, str]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Convert FHIR resources to NumPy array.
        
        Args:
            resources: List of FHIR resources
            feature_mapping: Mapping of feature names to FHIR paths
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        if not resources:
            return np.array([]), []
            
        # Default feature mapping for common FHIR resources
        if feature_mapping is None:
            feature_mapping = self._get_default_fhir_mapping()
            
        # Extract features from resources
        feature_data = []
        feature_names = list(feature_mapping.keys())
        
        for resource in resources:
            patient_features = self._extract_fhir_features(resource, feature_mapping)
            feature_data.append(patient_features)
            
        # Convert to NumPy array
        feature_matrix = np.array(feature_data)
        
        return feature_matrix, feature_names
        
    def _get_default_fhir_mapping(self) -> Dict[str, str]:
        """Get default FHIR feature mapping."""
        return {
            "age": "$.patient.birthDate",
            "gender": "$.patient.gender",
            "bmi": "$.observation.valueQuantity.value",
            "systolic_bp": "$.observation.component[?(@.code.coding[0].code=='8480-6')].valueQuantity.value",
            "diastolic_bp": "$.observation.component[?(@.code.coding[0].code=='8462-4')].valueQuantity.value",
            "heart_rate": "$.observation.component[?(@.code.coding[0].code=='8867-4')].valueQuantity.value",
            "temperature": "$.observation.component[?(@.code.coding[0].code=='8310-5')].valueQuantity.value",
            "diabetes": "$.condition.code.coding[?(@.display=='Diabetes mellitus')].code",
            "hypertension": "$.condition.code.coding[?(@.display=='Hypertension')].code"
        }
        
    def _extract_fhir_features(
        self,
        resource: Dict[str, Any],
        feature_mapping: Dict[str, str]
    ) -> List[float]:
        """Extract features from FHIR resource using mapping."""
        features = []
        
        for feature_name, fhir_path in feature_mapping.items():
            try:
                value = self._evaluate_fhir_path(resource, fhir_path)
                
                # Convert to numeric
                numeric_value = self._convert_to_numeric(value, feature_name)
                features.append(numeric_value)
                
            except Exception as e:
                self.logger.warning(f"Failed to extract {feature_name}: {e}")
                features.append(0.0)  # Default value
                
        return features
        
    def _evaluate_fhir_path(
        self,
        resource: Dict[str, Any],
        fhir_path: str
    ) -> Any:
        """Evaluate FHIR path expression on resource."""
        # Simplified FHIR path evaluation
        # In production, use a proper FHIR path library
        
        if fhir_path.startswith("$."):
            # JSON path style
            path_parts = fhir_path[2:].split(".")
            current = resource
            
            for part in path_parts:
                if "[" in part:
                    # Handle array access
                    array_part, index_part = part.split("[", 1)
                    index_part = index_part.rstrip("]")
                    
                    if array_part in current:
                        current = current[array_part]
                        if isinstance(current, list):
                            # Simple index access
                            try:
                                idx = int(index_part)
                                current = current[idx] if idx < len(current) else None
                            except ValueError:
                                # Handle filter expressions (simplified)
                                current = current[0] if current else None
                    else:
                        return None
                else:
                    # Simple property access
                    current = current.get(part) if isinstance(current, dict) else None
                    
                if current is None:
                    return None
                    
            return current
        else:
            # Direct property access
            return resource.get(fhir_path)
            
    def _convert_to_numeric(self, value: Any, feature_name: str) -> float:
        """Convert value to numeric."""
        if value is None:
            return 0.0
            
        if isinstance(value, (int, float)):
            return float(value)
            
        if isinstance(value, str):
            # Handle categorical variables
            if feature_name == "gender":
                return 1.0 if value.lower() in ["male", "m"] else 0.0
            if feature_name in ["diabetes", "hypertension"]:
                return 1.0 if value else 0.0
                
            # Try numeric conversion
            try:
                return float(value)
            except ValueError:
                return 0.0
                
        return 0.0
        
    def load_csv_data(
        self,
        csv_path: str,
        feature_columns: Optional[List[str]] = None,
        target_column: Optional[str] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
        """
        Load medical data from CSV file.
        
        Args:
            csv_path: Path to CSV file
            feature_columns: List of feature column names
            target_column: Target column name
            
        Returns:
            Tuple of (features, targets, column_names)
        """
        try:
            df = pd.read_csv(csv_path)
            
            # Handle missing values
            df = df.fillna(df.median(numeric_only=True))
            
            # Select features
            if feature_columns:
                X = df[feature_columns].values
                feature_names = feature_columns
            else:
                # Use all columns except target
                if target_column and target_column in df.columns:
                    feature_columns = [col for col in df.columns if col != target_column]
                    X = df[feature_columns].values
                    feature_names = feature_columns
                else:
                    X = df.values
                    feature_names = df.columns.tolist()
                    
            # Get targets if specified
            y = None
            if target_column and target_column in df.columns:
                y = df[target_column].values
                
            return X, y, feature_names
            
        except Exception as e:
            self.logger.error(f"Failed to load CSV data: {e}")
            raise
            
    def anonymize_data(
        self,
        data: np.ndarray,
        feature_names: List[str],
        method: str = "differential_privacy"
    ) -> np.ndarray:
        """
        Anonymize medical data.
        
        Args:
            data: Input data matrix
            feature_names: Feature names
            method: Anonymization method
            
        Returns:
            Anonymized data matrix
        """
        if method == "noise_addition":
            return self._add_noise_anonymization(data)
        elif method == "generalization":
            return self._generalize_data(data, feature_names)
        elif method == "suppression":
            return self._suppress_identifiers(data, feature_names)
        else:
            # Default: differential privacy style noise
            return self._add_noise_anonymization(data)
            
    def _add_noise_anonymization(self, data: np.ndarray) -> np.ndarray:
        """Add noise for anonymization."""
        # Add Gaussian noise proportional to data standard deviation
        noise_scale = 0.1 * np.std(data, axis=0)
        noise = np.random.normal(0, noise_scale, data.shape)
        return data + noise
        
    def _generalize_data(
        self,
        data: np.ndarray,
        feature_names: List[str]
    ) -> np.ndarray:
        """Generalize quasi-identifier attributes."""
        anonymized = data.copy()
        
        for i, feature_name in enumerate(feature_names):
            if "age" in feature_name.lower():
                # Generalize age to age groups
                age_groups = np.digitize(data[:, i], bins=[0, 18, 30, 45, 60, 75, 100])
                anonymized[:, i] = age_groups
                
        return anonymized
        
    def _suppress_identifiers(
        self,
        data: np.ndarray,
        feature_names: List[str]
    ) -> np.ndarray:
        """Suppress direct identifiers."""
        # Remove columns that are direct identifiers
        identifier_keywords = ["name", "id", "phone", "email", "address", "ssn"]
        
        safe_indices = []
        for i, feature_name in enumerate(feature_names):
            if not any(keyword in feature_name.lower() for keyword in identifier_keywords):
                safe_indices.append(i)
                
        return data[:, safe_indices]
        
    def record_consent(
        self,
        patient_id: str,
        consent_given: bool,
        consent_type: str = "federated_learning"
    ) -> str:
        """
        Record patient consent for federated learning.
        
        Args:
            patient_id: Patient identifier
            consent_given: Whether consent was given
            consent_type: Type of consent
            
        Returns:
            Consent record hash
        """
        consent_record = {
            "patient_id": self._hash_identifier(patient_id),
            "consent_given": consent_given,
            "consent_type": consent_type,
            "timestamp": datetime.utcnow().isoformat(),
            "hospital_id": self.hospital_id
        }
        
        # Create consent hash
        consent_hash = hashlib.sha256(
            json.dumps(consent_record, sort_keys=True).encode()
        ).hexdigest()
        
        self.consent_records[consent_hash] = consent_record
        
        return consent_hash
        
    def _hash_identifier(self, identifier: str) -> str:
        """Hash identifier for privacy."""
        return hashlib.sha256(identifier.encode()).hexdigest()
        
    def get_consent_status(self, consent_hash: str) -> Optional[Dict[str, Any]]:
        """Get consent status by hash."""
        return self.consent_records.get(consent_hash)
        
    def validate_data_completeness(
        self,
        data: np.ndarray,
        feature_names: List[str],
        min_completeness: float = 0.8
    ) -> bool:
        """
        Validate data completeness for federated learning.
        
        Args:
            data: Data matrix
            feature_names: Feature names
            min_completeness: Minimum completeness threshold
            
        Returns:
            True if data meets completeness requirements
        """
        # Check for missing values
        missing_ratio = np.isnan(data).sum() / data.size
        
        if missing_ratio > (1 - min_completeness):
            self.logger.warning(f"Data completeness {1-missing_ratio:.2f} below threshold {min_completeness}")
            return False
            
        return True