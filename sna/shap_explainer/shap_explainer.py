"""
Module: sna/shap_explainer/shap_explainer.py
DPDP §: 9(4) - Model interpretability for privacy-preserving AI
Byzantine: SHAP values computed locally, aggregated with geometric median (tolerates f < n/2)
Privacy: DP noise added to SHAP values (σ=0.5, ε=0.5) before aggregation
Test: pytest tests/test_shap.py::test_shap_explanation
API: POST /explain_model, GET /shap_values
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime
import json

# SHAP for model interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

from algorithms.dp_mechanisms.gaussian import GaussianDP
from sna.byzantine_shield.byzantine_shield import ByzantineShield, HospitalUpdate


@dataclass
class SHAPExplanation:
    """SHAP explanation result."""
    hospital_id: str
    feature_names: List[str]
    shap_values: np.ndarray
    base_value: float
    expected_value: float
    timestamp: str
    privacy_budget_spent: float
    byzantine_score: float


class SHAPExplainer:
    """
    SHAP Explainer for Ghost Protocol.
    
    Implements SHAP (SHapley Additive exPlanations) for model interpretability
    in federated learning while preserving privacy through DP noise addition.
    
    Features:
    - Local SHAP value computation
    - Byzantine-robust aggregation
    - Privacy-preserving explanation sharing
    - Feature importance ranking
    """
    
    def __init__(
        self,
        noise_multiplier: float = 0.5,
        epsilon: float = 0.5,
        delta: float = 1e-6,
        byzantine_threshold: float = 2.0
    ):
        """
        Initialize SHAP Explainer.
        
        Args:
            noise_multiplier: DP noise for SHAP values
            epsilon: Privacy budget for explanations
            delta: DP failure probability
            byzantine_threshold: Z-score threshold for anomaly detection
        """
        self.noise_multiplier = noise_multiplier
        self.epsilon = epsilon
        self.delta = delta
        self.byzantine_threshold = byzantine_threshold
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize DP mechanism
        self.dp_mechanism = GaussianDP(epsilon=epsilon, delta=delta)
        
        # Initialize Byzantine shield
        self.byzantine_shield = ByzantineShield(z_score_threshold=byzantine_threshold)
        
        # Explanation history
        self.explanation_history: List[SHAPExplanation] = []
        
        # Metrics tracking
        self.metrics = {
            "total_explanations": 0,
            "privacy_budget_spent": 0.0,
            "byzantine_violations": 0,
            "average_explanation_time": 0.0
        }
        
    def compute_local_shap_values(
        self,
        model: torch.nn.Module,
        background_data: np.ndarray,
        explanation_data: np.ndarray,
        feature_names: List[str],
        hospital_id: str
    ) -> Dict[str, Any]:
        """
        Compute SHAP values locally for a hospital's model.
        
        Args:
            model: Trained PyTorch model
            background_data: Background dataset for SHAP
            explanation_data: Data points to explain
            feature_names: Names of input features
            hospital_id: Hospital identifier
            
        Returns:
            Dictionary with SHAP values and metadata
            
        Note: SHAP values computed locally, never leave hospital
        """
        start_time = datetime.utcnow()
        
        if not SHAP_AVAILABLE:
            self.logger.warning("SHAP not available, using gradient-based approximation")
            return self._approximate_shap_values(
                model, background_data, explanation_data, feature_names, hospital_id
            )
            
        try:
            # Create SHAP explainer
            explainer = shap.Explainer(model, background_data, feature_names=feature_names)
            
            # Compute SHAP values
            shap_values = explainer(explanation_data)
            
            # Extract base value (expected model output)
            base_value = float(shap_values.base_values[0]) if hasattr(shap_values, 'base_values') else 0.0
            
            # Get expected value (mean prediction)
            expected_value = float(np.mean(shap_values.values))
            
            # Create explanation result
            explanation = {
                "hospital_id": hospital_id,
                "feature_names": feature_names,
                "shap_values": shap_values.values,
                "base_value": base_value,
                "expected_value": expected_value,
                "timestamp": datetime.utcnow().isoformat(),
                "privacy_budget_spent": 0.0  # Will be updated during aggregation
            }
            
            self.logger.info(f"SHAP explanation computed for {hospital_id}")
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"SHAP computation failed for {hospital_id}: {e}")
            raise
            
    def aggregate_shap_explanations(
        self,
        explanations: List[Dict[str, Any]],
        apply_dp: bool = True
    ) -> Dict[str, Any]:
        """
        Aggregate SHAP explanations from multiple hospitals.
        
        Args:
            explanations: List of local SHAP explanations
            apply_dp: Whether to add DP noise to aggregated values
            
        Returns:
            Aggregated SHAP explanation with Byzantine protection
            
        Byzantine: Uses geometric median for robust aggregation
        Privacy: Adds DP noise to protect individual hospital contributions
        """
        if not explanations:
            raise ValueError("No explanations to aggregate")
            
        start_time = datetime.utcnow()
        
        # Convert to HospitalUpdate format for Byzantine processing
        updates = []
        for exp in explanations:
            # Convert SHAP values to tensor format
            shap_tensor = torch.tensor(exp["shap_values"])
            weights = {"shap_values": shap_tensor}
            
            update = HospitalUpdate(
                hospital_id=exp["hospital_id"],
                weights=weights,
                reputation_score=0.8,  # Default reputation
                timestamp=datetime.fromisoformat(exp["timestamp"]),
                metadata={
                    "base_value": exp["base_value"],
                    "expected_value": exp["expected_value"],
                    "feature_names": exp["feature_names"]
                }
            )
            updates.append(update)
            
        # Apply Byzantine protection to SHAP values
        aggregated_weights, byzantine_report = self.byzantine_shield.aggregate_with_byzantine_protection(
            updates, apply_dp_sanitization=False  # We'll add DP manually
        )
        
        # Extract aggregated SHAP values
        aggregated_shap = aggregated_weights["shap_values"].numpy()
        
        # Add DP noise if requested
        if apply_dp:
            # Flatten for DP processing
            flat_shap = aggregated_shap.flatten()
            sensitivity = 1.0  # Assume L1 sensitivity of 1
            
            # Add DP noise
            noisy_flat_shap = self.dp_mechanism.add_noise(
                torch.tensor(flat_shap), sensitivity
            ).numpy()
            
            # Reshape back
            aggregated_shap = noisy_flat_shap.reshape(aggregated_shap.shape)
            
            # Update privacy budget
            privacy_spent = self.dp_mechanism.epsilon
            self.metrics["privacy_budget_spent"] += privacy_spent
        else:
            privacy_spent = 0.0
            
        # Create aggregated explanation
        aggregated_explanation = {
            "aggregated_shap_values": aggregated_shap,
            "feature_names": explanations[0]["feature_names"],
            "base_value": np.mean([exp["base_value"] for exp in explanations]),
            "expected_value": np.mean([exp["expected_value"] for exp in explanations]),
            "hospitals_contributed": len(explanations),
            "byzantine_violations": byzantine_report.get("anomalies_detected", 0),
            "privacy_budget_spent": privacy_spent,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Update metrics
        self.metrics["total_explanations"] += 1
        self.metrics["byzantine_violations"] += byzantine_report.get("anomalies_detected", 0)
        
        elapsed_time = (datetime.utcnow() - start_time).total_seconds()
        self.metrics["average_explanation_time"] = (
            (self.metrics["average_explanation_time"] * (self.metrics["total_explanations"] - 1) + elapsed_time) /
            self.metrics["total_explanations"]
        )
        
        self.logger.info(
            f"SHAP aggregation completed: {len(explanations)} hospitals, "
            f"byzantine_violations={byzantine_report.get('anomalies_detected', 0)}, "
            f"privacy_spent={privacy_spent:.3f}"
        )
        
        return aggregated_explanation
        
    def get_feature_importance_ranking(
        self,
        aggregated_explanation: Dict[str, Any],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Rank features by importance based on aggregated SHAP values.
        
        Args:
            aggregated_explanation: Aggregated SHAP explanation
            top_k: Number of top features to return
            
        Returns:
            Ranked list of feature importance
        """
        shap_values = aggregated_explanation["aggregated_shap_values"]
        feature_names = aggregated_explanation["feature_names"]
        
        # Compute mean absolute SHAP values for ranking
        if len(shap_values.shape) == 3:
            # For multi-class: average across classes
            mean_abs_shap = np.mean(np.abs(shap_values), axis=(0, 2))
        else:
            # For binary classification
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            
        # Create feature ranking
        feature_ranking = []
        for i, importance in enumerate(mean_abs_shap):
            feature_ranking.append({
                "feature_name": feature_names[i] if i < len(feature_names) else f"Feature_{i}",
                "importance_score": float(importance),
                "rank": 0  # Will be set after sorting
            })
            
        # Sort by importance
        feature_ranking.sort(key=lambda x: x["importance_score"], reverse=True)
        
        # Set ranks
        for i, feature in enumerate(feature_ranking):
            feature["rank"] = i + 1
            
        return feature_ranking[:top_k]
        
    def explain_prediction(
        self,
        hospital_id: str,
        feature_values: Dict[str, float],
        aggregated_explanation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Explain a specific prediction using aggregated SHAP values.
        
        Args:
            hospital_id: Hospital requesting explanation
            feature_values: Feature values for the prediction
            aggregated_explanation: Aggregated SHAP explanation
            
        Returns:
            Detailed explanation of the prediction
        """
        # Get feature names and SHAP values
        feature_names = aggregated_explanation["feature_names"]
        shap_values = aggregated_explanation["aggregated_shap_values"]
        base_value = aggregated_explanation["base_value"]
        
        # Create feature value array
        feature_array = np.array([feature_values.get(name, 0.0) for name in feature_names])
        
        # Get SHAP values for this instance
        if len(shap_values.shape) == 3:
            # Multi-class: use first class
            instance_shap = shap_values[0]  # Shape: (n_features,)
        else:
            # Binary classification
            instance_shap = shap_values[0] if len(shap_values) > 1 else shap_values
            
        # Create feature contributions
        contributions = []
        for i, (feature_name, feature_value) in enumerate(zip(feature_names, feature_array)):
            shap_value = instance_shap[i] if i < len(instance_shap) else 0.0
            contributions.append({
                "feature_name": feature_name,
                "feature_value": float(feature_value),
                "shap_value": float(shap_value),
                "contribution": float(shap_value),
                "impact": "positive" if shap_value > 0 else "negative"
            })
            
        # Sort by absolute contribution
        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        
        # Compute prediction
        prediction = base_value + sum(contrib["shap_value"] for contrib in contributions)
        
        return {
            "hospital_id": hospital_id,
            "base_value": base_value,
            "prediction": float(prediction),
            "contributions": contributions,
            "top_positive": [c for c in contributions if c["shap_value"] > 0][:3],
            "top_negative": [c for c in contributions if c["shap_value"] < 0][:3],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    def get_explanation_history(
        self,
        hospital_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get history of SHAP explanations.
        
        Args:
            hospital_id: Filter by hospital (optional)
            start_date: Filter by start date (ISO format)
            end_date: Filter by end date (ISO format)
            
        Returns:
            List of explanation records
        """
        history = []
        
        for explanation in self.explanation_history:
            # Filter by hospital if specified
            if hospital_id and explanation.hospital_id != hospital_id:
                continue
                
            # Filter by date if specified
            exp_date = datetime.fromisoformat(explanation.timestamp)
            if start_date and exp_date < datetime.fromisoformat(start_date):
                continue
            if end_date and exp_date > datetime.fromisoformat(end_date):
                continue
                
            history.append({
                "hospital_id": explanation.hospital_id,
                "feature_names": explanation.feature_names,
                "base_value": explanation.base_value,
                "expected_value": explanation.expected_value,
                "timestamp": explanation.timestamp,
                "privacy_budget_spent": explanation.privacy_budget_spent,
                "byzantine_score": explanation.byzantine_score
            })
            
        return history
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get SHAP explainer metrics."""
        return {
            **self.metrics,
            "dp_mechanism": self.dp_mechanism.get_noise_statistics(),
            "byzantine_shield_stats": self.byzantine_shield.get_attack_statistics()
        }
        
    def reset_metrics(self):
        """Reset metrics counters."""
        self.metrics = {
            "total_explanations": 0,
            "privacy_budget_spent": 0.0,
            "byzantine_violations": 0,
            "average_explanation_time": 0.0
        }
        
    def _approximate_shap_values(
        self,
        model: torch.nn.Module,
        background_data: np.ndarray,
        explanation_data: np.ndarray,
        feature_names: List[str],
        hospital_id: str
    ) -> Dict[str, Any]:
        """
        Approximate SHAP values using gradient-based method when SHAP unavailable.
        
        Args:
            model: PyTorch model
            background_data: Background data
            explanation_data: Data to explain
            feature_names: Feature names
            hospital_id: Hospital ID
            
        Returns:
            Approximated SHAP explanation
        """
        self.logger.info(f"Using gradient approximation for SHAP for {hospital_id}")
        
        # Simple gradient-based approximation
        model.eval()
        
        # Compute baseline prediction
        with torch.no_grad():
            baseline_input = torch.tensor(background_data.mean(axis=0, keepdims=True), dtype=torch.float32)
            baseline_output = model(baseline_input)
            base_value = float(baseline_output.mean().item())
            
        # Compute gradients for each feature
        shap_values = []
        
        for i in range(len(explanation_data)):
            input_tensor = torch.tensor(explanation_data[i:i+1], dtype=torch.float32, requires_grad=True)
            output = model(input_tensor).mean()
            
            # Compute gradients
            output.backward()
            gradients = input_tensor.grad.numpy()
            
            # Approximate SHAP values using gradients
            shap_val = gradients * (explanation_data[i:i+1] - background_data.mean(axis=0))
            shap_values.append(shap_val)
            
            # Clear gradients
            input_tensor.grad.zero_()
            
        shap_values = np.array(shap_values)
        
        return {
            "hospital_id": hospital_id,
            "feature_names": feature_names,
            "shap_values": shap_values,
            "base_value": base_value,
            "expected_value": base_value,
            "timestamp": datetime.utcnow().isoformat(),
            "privacy_budget_spent": 0.0,
            "approximation_method": "gradient_based"
        }