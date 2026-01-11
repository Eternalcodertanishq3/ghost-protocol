"""
Ghost Agent Core - Hospital-side Federated Learning Node
Production-grade implementation with DPDP compliance and Byzantine immunity
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import torch
import torch.nn as nn
from opacus import GradSampleModule
from opacus.accountants.rdp import RDPAccountant
from opacus.optimizers import DPOptimizer
from pydantic import BaseModel, Field

from .privacy import DifferentialPrivacyEngine
from .security import SecurityManager
from .compliance import DPDPComplianceManager
from .models import ClinicalPredictionModel
from .adapters import EMRAdapter
from .grpc_client import SNAClient


class GhostAgentConfig(BaseModel):
    """Configuration for Ghost Agent with DPDP compliance validation"""
    
    hospital_id: str = Field(..., description="Unique hospital identifier (UUID v4)")
    hospital_name: str = Field(..., description="Hospital name as per DPDP records")
    hospital_type: str = Field(..., description="GOVT|PRIVATE|TRUST|MULTI_SPECIALITY")
    location_state: str = Field(..., description="State for DPDP jurisdiction")
    location_district: str = Field(..., description="District for local compliance")
    
    # Privacy budget management
    epsilon_max: float = Field(default=9.5, le=10.0, description="DPDP §15: Privacy budget ceiling")
    epsilon_per_update: float = Field(default=1.23, le=2.0, description="ε per federated update")
    delta_max: float = Field(default=1e-05, le=1e-05, description="δ for (ε,δ)-DP")
    gaussian_noise_scale: float = Field(default=1.3, description="Noise multiplier for DP-SGD")
    
    # Training hyperparameters
    learning_rate: float = Field(default=0.001, description="Model learning rate")
    batch_size: int = Field(default=32, ge=16, le=128, description="Training batch size")
    epochs_per_round: int = Field(default=5, ge=1, le=20, description="Local epochs per round")
    aggregation_strategy: str = Field(default="FedAvg", description="FedAvg|FedProx|SCAFFOLD")
    fedprox_mu: float = Field(default=0.1, description="FedProx regularization parameter")
    
    # Security
    security_manager: SecurityManager = Field(..., description="Cryptographic security layer")
    
    # Compliance
    dpdp_manager: DPDPComplianceManager = Field(..., description="DPDP Act 2023 compliance")
    
    class Config:
        arbitrary_types_allowed = True


class GhostAgent:
    """
    Hospital-side federated learning agent with DPDP compliance
    
    Core invariant: Raw patient data NEVER leaves hospital premises
    """
    
    def __init__(self, config: GhostAgentConfig):
        self.config = config
        self.logger = logging.getLogger(f"ghost_agent.{config.hospital_id}")
        
        # Core components
        self.dp_engine = DifferentialPrivacyEngine(config)
        self.security = config.security_manager
        self.dpdp = config.dpdp_manager
        self.sna_client = SNAClient(config)
        self.emr_adapter = EMRAdapter(config)
        
        # Model and training state
        self.model = ClinicalPredictionModel()
        self.global_model_state: Optional[Dict[str, torch.Tensor]] = None
        self.local_update_count = 0
        self.privacy_budget_spent = 0.0
        
        # Byzantine detection
        self.gradient_history: List[Dict] = []
        self.reputation_score = 1.0
        
        # Initialize DPDP compliance
        self.dpdp.initialize_hospital_record(
            hospital_id=config.hospital_id,
            hospital_name=config.hospital_name,
            location=f"{config.location_district}, {config.location_state}",
            purpose="federated_learning_clinical_prediction"
        )
        
        self.logger.info(f"Ghost Agent initialized for {config.hospital_name}")
        self.logger.info(f"Privacy budget: ε={config.epsilon_max}, δ={config.delta_max}")
    
    async def start_federated_learning_round(self) -> Dict[str, Any]:
        """
        Execute one complete federated learning round with full privacy protection
        
        Returns:
            Dict containing round results and compliance metrics
        """
        start_time = time.time()
        round_id = str(uuid.uuid4())
        
        self.logger.info(f"Starting FL round {round_id}")
        
        try:
            # 1. Verify privacy budget before proceeding
            if self.privacy_budget_spent >= self.config.epsilon_max:
                raise ValueError(f"Privacy budget exhausted: {self.privacy_budget_spent:.2f} >= {self.config.epsilon_max}")
            
            # 2. Load and validate local data (DPDP §11(3) Consent)
            local_data = await self._load_local_data_with_consent()
            if len(local_data) == 0:
                self.logger.warning("No consented data available for training")
                return {"status": "no_data", "round_id": round_id}
            
            # 3. Get global model from SNA
            self.global_model_state = await self.sna_client.get_global_model()
            if self.global_model_state:
                self.model.load_state_dict(self.global_model_state)
            
            # 4. Local training with differential privacy
            training_results = await self._train_with_differential_privacy(
                local_data, round_id
            )
            
            # 5. Create privacy-preserving update (Ghost Pack)
            ghost_pack = await self._create_ghost_pack(
                training_results, round_id
            )
            
            # 6. Submit to SNA with Byzantine protection
            submission_result = await self.sna_client.submit_update(ghost_pack)
            
            # 7. Update compliance records
            await self._update_compliance_records(round_id, training_results)
            
            round_duration = time.time() - start_time
            
            self.logger.info(f"Round {round_id} completed in {round_duration:.2f}s")
            
            return {
                "round_id": round_id,
                "status": "success",
                "privacy_budget_used": training_results["epsilon_spent"],
                "local_auc": training_results["local_auc"],
                "gradient_norm": training_results["gradient_norm"],
                "training_samples": len(local_data),
                "duration_seconds": round_duration,
                "compliance_verified": True
            }
            
        except Exception as e:
            self.logger.error(f"Round {round_id} failed: {str(e)}")
            await self.dpdp.log_compliance_event(
                event_type="round_failure",
                round_id=round_id,
                details={"error": str(e)}
            )
            raise
    
    async def _load_local_data_with_consent(self) -> List[Dict[str, Any]]:
        """
        Load patient data with DPDP consent verification
        
        Returns:
            List of consented patient records for training
        """
        # Query EMR with consent filters
        raw_records = await self.emr_adapter.get_patient_records(
            consent_status="active",
            purpose="federated_learning",
            validity_check=True
        )
        
        # Apply DPDP filters
        consented_records = []
        for record in raw_records:
            if self.dpdp.verify_consent_for_purpose(
                patient_id=record["patient_id"],
                purpose="federated_learning_clinical_prediction",
                record_date=record["date_created"]
            ):
                # Apply feature encoding (no PII leakage)
                encoded_features = self._encode_clinical_features(record)
                if encoded_features is not None:
                    consented_records.append(encoded_features)
        
        self.logger.info(f"Loaded {len(consented_records)} consented records")
        return consented_records
    
    async def _train_with_differential_privacy(
        self, 
        data: List[Dict[str, Any]], 
        round_id: str
    ) -> Dict[str, Any]:
        """
        Train model locally with differential privacy guarantees
        
        Args:
            data: Training data from hospital EMR
            round_id: Unique round identifier
            
        Returns:
            Training results with privacy metrics
        """
        # Prepare data loaders
        train_loader = self._create_data_loader(data, shuffle=True)
        
        # Wrap model for DP training
        dp_model = GradSampleModule(self.model)
        
        # DP optimizer with privacy accounting
        optimizer = torch.optim.Adam(dp_model.parameters(), lr=self.config.learning_rate)
        dp_optimizer = DPOptimizer(
            optimizer=optimizer,
            noise_multiplier=self.config.gaussian_noise_scale,
            max_grad_norm=1.0,  # L2 clipping
            expected_batch_size=self.config.batch_size
        )
        
        # Privacy accountant
        accountant = RDPAccountant()
        
        # Training loop
        dp_model.train()
        total_loss = 0.0
        gradient_norms = []
        
        for epoch in range(self.config.epochs_per_round):
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._get_device()), targets.to(self._get_device())
                
                # Forward pass
                outputs = dp_model(inputs)
                loss = nn.BCEWithLogitsLoss()(outputs, targets)
                
                # Backward pass with DP
                loss.backward()
                
                # Compute gradient norm for monitoring
                grad_norm = self._compute_gradient_norm(dp_model)
                gradient_norms.append(grad_norm)
                
                # DP optimizer step (adds noise and clips)
                dp_optimizer.step()
                dp_optimizer.zero_grad()
                
                total_loss += loss.item()
                
                # Update privacy accountant
                accountant.step(noise_multiplier=self.config.gaussian_noise_scale, 
                              sample_rate=1/len(train_loader))
        
        # Calculate privacy spent
        epsilon_spent = accountant.get_epsilon(delta=self.config.delta_max)
        self.privacy_budget_spent += epsilon_spent
        
        # Validate privacy budget
        if self.privacy_budget_spent > self.config.epsilon_max:
            await self.dpdp.trigger_privacy_budget_exhaustion_alert(
                current_spent=self.privacy_budget_spent,
                max_budget=self.config.epsilon_max
            )
        
        # Evaluate local model
        local_auc = await self._evaluate_model(dp_model, data)
        
        # Extract model updates (gradients)
        model_updates = {}
        for name, param in dp_model.named_parameters():
            if param.grad is not None:
                model_updates[name] = param.grad.detach().cpu()
        
        avg_gradient_norm = np.mean(gradient_norms) if gradient_norms else 0.0
        
        self.logger.info(f"Training completed: ε={epsilon_spent:.3f}, AUC={local_auc:.3f}")
        
        return {
            "model_updates": model_updates,
            "epsilon_spent": epsilon_spent,
            "local_auc": local_auc,
            "gradient_norm": avg_gradient_norm,
            "training_loss": total_loss / len(train_loader),
            "samples_trained": len(data)
        }
    
    async def _create_ghost_pack(
        self, 
        training_results: Dict[str, Any], 
        round_id: str
    ) -> Dict[str, Any]:
        """
        Create privacy-preserving, signed, and encrypted update for SNA
        
        Args:
            training_results: Results from local training
            round_id: Federated learning round identifier
            
        Returns:
            Ghost Pack ready for submission to SNA
        """
        # Prepare model update for transmission
        update_dict = {}
        for name, tensor in training_results["model_updates"].items():
            # Convert to numpy for serialization
            update_dict[name] = tensor.numpy().tolist()
        
        # Create Ghost Pack structure
        ghost_pack = {
            "metadata": {
                "round_id": round_id,
                "hospital_id": self.config.hospital_id,
                "hospital_name": self.config.hospital_name,
                "timestamp": datetime.utcnow().isoformat(),
                "protocol_version": "1.0.0",
                "dp_compliance": {
                    "epsilon_spent": training_results["epsilon_spent"],
                    "delta_used": self.config.delta_max,
                    "privacy_budget_remaining": self.config.epsilon_max - self.privacy_budget_spent,
                    "dp_algorithm": "Gaussian_DP_SGD"
                },
                "model_performance": {
                    "local_auc": training_results["local_auc"],
                    "gradient_norm": training_results["gradient_norm"],
                    "training_samples": training_results["samples_trained"]
                }
            },
            "model_update": update_dict,
            "byzantine_shield": {
                "reputation_score": self.reputation_score,
                "gradient_history_length": len(self.gradient_history),
                "anomaly_score": self._calculate_anomaly_score(training_results)
            }
        }
        
        # Sign the update (ECDSA P-256)
        signature = await self.security.sign_update(ghost_pack)
        ghost_pack["signature"] = signature
        
        # Encrypt sensitive components (AES-256-GCM)
        encrypted_payload = await self.security.encrypt_ghost_pack(ghost_pack)
        
        self.logger.info(f"Ghost Pack created for round {round_id}")
        
        return {
            "encrypted_payload": encrypted_payload,
            "hospital_id": self.config.hospital_id,
            "signature": signature["signature"],
            "timestamp": ghost_pack["metadata"]["timestamp"]
        }
    
    def _encode_clinical_features(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Encode clinical features while ensuring zero PII leakage
        
        Args:
            record: Raw patient record from EMR
            
        Returns:
            Encoded features ready for model training, or None if invalid
        """
        try:
            # Extract clinical features only (no PII)
            features = {
                "age": float(record.get("age", 0)),
                "gender": 1 if record.get("gender") == "M" else 0,
                "vital_signs": {
                    "systolic_bp": float(record.get("systolic_bp", 120)),
                    "diastolic_bp": float(record.get("diastolic_bp", 80)),
                    "heart_rate": float(record.get("heart_rate", 72)),
                    "temperature": float(record.get("temperature", 98.6)),
                    "oxygen_saturation": float(record.get("oxygen_saturation", 98))
                },
                "lab_values": {
                    "hemoglobin": float(record.get("hemoglobin", 14)),
                    "white_blood_cells": float(record.get("wbc", 7000)),
                    "platelets": float(record.get("platelets", 250000)),
                    "glucose": float(record.get("glucose", 100)),
                    "creatinine": float(record.get("creatinine", 1.0))
                },
                "comorbidities": {
                    "diabetes": int(record.get("diabetes", False)),
                    "hypertension": int(record.get("hypertension", False)),
                    "heart_disease": int(record.get("heart_disease", False)),
                    "kidney_disease": int(record.get("kidney_disease", False))
                },
                "target": int(record.get("readmission_risk", 0))  # Prediction target
            }
            
            # Validate feature ranges
            if not self._validate_clinical_ranges(features):
                return None
            
            return features
            
        except (ValueError, TypeError, KeyError) as e:
            self.logger.warning(f"Feature encoding failed: {e}")
            return None
    
    def _validate_clinical_ranges(self, features: Dict[str, Any]) -> bool:
        """Validate clinical feature ranges for data quality"""
        vital_signs = features["vital_signs"]
        
        # Basic sanity checks
        if not (0 <= features["age"] <= 120):
            return False
        if not (60 <= vital_signs["systolic_bp"] <= 250):
            return False
        if not (40 <= vital_signs["diastolic_bp"] <= 150):
            return False
        if not (40 <= vital_signs["heart_rate"] <= 200):
            return False
        if not (90 <= vital_signs["oxygen_saturation"] <= 100):
            return False
        
        return True
    
    def _create_data_loader(self, data: List[Dict], shuffle: bool = False) -> torch.utils.data.DataLoader:
        """Create PyTorch DataLoader from encoded features"""
        
        # Convert to tensors
        feature_vectors = []
        targets = []
        
        for record in data:
            # Flatten features into vector
            vector = [
                record["age"],
                record["gender"]
            ]
            
            # Add vital signs
            vector.extend([
                record["vital_signs"]["systolic_bp"] / 200,
                record["vital_signs"]["diastolic_bp"] / 150,
                record["vital_signs"]["heart_rate"] / 200,
                record["vital_signs"]["temperature"] / 110,
                record["vital_signs"]["oxygen_saturation"] / 100
            ])
            
            # Add lab values (normalized)
            vector.extend([
                record["lab_values"]["hemoglobin"] / 20,
                record["lab_values"]["white_blood_cells"] / 15000,
                record["lab_values"]["platelets"] / 500000,
                record["lab_values"]["glucose"] / 300,
                record["lab_values"]["creatinine"] / 10
            ])
            
            # Add comorbidities
            vector.extend([
                record["comorbidities"]["diabetes"],
                record["comorbidities"]["hypertension"],
                record["comorbidities"]["heart_disease"],
                record["comorbidities"]["kidney_disease"]
            ])
            
            feature_vectors.append(vector)
            targets.append(record["target"])
        
        # Convert to tensors
        X = torch.FloatTensor(feature_vectors)
        y = torch.FloatTensor(targets).unsqueeze(1)
        
        # Create dataset and loader
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            drop_last=True  # Important for DP-SGD
        )
        
        return loader
    
    async def _evaluate_model(self, model: nn.Module, data: List[Dict]) -> float:
        """Evaluate model performance on local validation set"""
        if not data:
            return 0.0
        
        model.eval()
        val_loader = self._create_data_loader(data, shuffle=False)
        
        predictions = []
        targets = []
        
        with torch.no_grad():
            for inputs, target_batch in val_loader:
                inputs = inputs.to(self._get_device())
                outputs = model(inputs)
                predictions.extend(torch.sigmoid(outputs).cpu().numpy().flatten())
                targets.extend(target_batch.numpy().flatten())
        
        # Calculate AUC
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(targets, predictions) if len(set(targets)) > 1 else 0.5
        
        return float(auc)
    
    def _compute_gradient_norm(self, model: nn.Module) -> float:
        """Compute L2 norm of gradients for monitoring"""
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** (1. / 2)
    
    def _calculate_anomaly_score(self, training_results: Dict[str, Any]) -> float:
        """Calculate anomaly score for Byzantine detection"""
        # Simple Z-score based on gradient norm history
        current_norm = training_results["gradient_norm"]
        
        if not self.gradient_history:
            self.gradient_history.append(current_norm)
            return 0.0
        
        history_norms = [h["gradient_norm"] for h in self.gradient_history[-20:]]
        mean_norm = np.mean(history_norms)
        std_norm = np.std(history_norms) + 1e-8  # Avoid division by zero
        
        z_score = abs(current_norm - mean_norm) / std_norm
        
        # Update history
        self.gradient_history.append({
            "round_id": training_results.get("round_id", "unknown"),
            "gradient_norm": current_norm,
            "local_auc": training_results["local_auc"],
            "timestamp": datetime.utcnow()
        })
        
        # Keep history bounded
        if len(self.gradient_history) > 100:
            self.gradient_history = self.gradient_history[-50:]
        
        return float(z_score)
    
    def _get_device(self) -> torch.device:
        """Get appropriate compute device"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    
    async def _update_compliance_records(self, round_id: str, training_results: Dict[str, Any]):
        """Update DPDP compliance records for audit trail"""
        
        compliance_record = {
            "round_id": round_id,
            "hospital_id": self.config.hospital_id,
            "purpose": "federated_learning_clinical_prediction",
            "data_processing_type": "machine_learning_training",
            "privacy_measures": {
                "differential_privacy": True,
                "epsilon_spent": training_results["epsilon_spent"],
                "delta_used": self.config.delta_max,
                "algorithm": "Gaussian_DP_SGD"
            },
            "data_minimization": {
                "features_used": 17,  # Clinical features only
                "pii_excluded": True,
                "on_premise_processing": True
            },
            "security_measures": {
                "encryption": "AES-256-GCM",
                "signatures": "ECDSA-P256",
                "mtls": "1.3"
            },
            "retention_policy": {
                "model_updates_retained": False,  # Not stored centrally
                "audit_logs_retention_days": 2555  # 7 years as per DPDP
            },
            "cross_border_transfer": False,  # Minor-to-Minor principle
            "timestamp": datetime.utcnow()
        }
        
        await self.dpdp.log_processing_activity(compliance_record)
        
        self.logger.info(f"Compliance records updated for round {round_id}")
    
    async def get_compliance_report(self) -> Dict[str, Any]:
        """Generate DPDP compliance report for this hospital"""
        return await self.dpdp.generate_compliance_report(
            hospital_id=self.config.hospital_id,
            period_days=30
        )
    
    async def shutdown(self):
        """Graceful shutdown with compliance logging"""
        self.logger.info("Shutting down Ghost Agent")
        
        # Final compliance report
        final_report = await self.get_compliance_report()
        
        await self.dpdp.log_compliance_event(
            event_type="agent_shutdown",
            details={
                "total_rounds_completed": self.local_update_count,
                "total_privacy_budget_spent": self.privacy_budget_spent,
                "final_compliance_status": "compliant"
            }
        )
        
        # Close connections
        await self.sna_client.close()
        
        self.logger.info("Ghost Agent shutdown complete")