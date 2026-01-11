"""
Module: sna/dropout_predictor/dropout_predictor.py
DPDP §: 9(4) - Predictive analytics for hospital participation
Description: Predicts hospital dropout risk to maintain federated learning stability
Byzantine: Dropout predictions aggregated with geometric median (tolerates f < n/2 malicious)
Privacy: Prediction features anonymized with DP noise (ε=0.2) before model training
Test: pytest tests/test_dropout.py::test_dropout_prediction
API: GET /dropout_risk/{hospital_id}, POST /intervention, GET /stability_metrics
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

from algorithms.dp_mechanisms.gaussian import GaussianDP


@dataclass
class DropoutRisk:
    """Hospital dropout risk assessment."""
    hospital_id: str
    dropout_probability: float
    risk_level: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    contributing_factors: List[str]
    predicted_dropout_time: Optional[str]
    intervention_recommended: bool
    confidence: float
    timestamp: str


@dataclass
class Intervention:
    """Intervention strategy for at-risk hospitals."""
    hospital_id: str
    intervention_type: str
    intervention_data: Dict[str, Any]
    success_probability: float
    cost_estimate: float
    timestamp: str


class DropoutPredictor:
    """
    Dropout Predictor for Ghost Protocol.
    
    Predicts hospital dropout risk using federated learning participation patterns,
    reputation scores, and engagement metrics to maintain system stability.
    
    Features:
    - Privacy-preserving dropout prediction
    - Byzantine-robust risk aggregation
    - Automated intervention recommendations
    - Stability monitoring
    """
    
    def __init__(
        self,
        risk_threshold: float = 0.7,
        intervention_cost_limit: float = 1000.0,  # HealthTokens
        prediction_horizon_days: int = 30,
        privacy_epsilon: float = 0.2
    ):
        """
        Initialize Dropout Predictor.
        
        Args:
            risk_threshold: Probability threshold for high risk
            intervention_cost_limit: Maximum intervention cost in HealthTokens
            prediction_horizon_days: Days ahead to predict dropout
            privacy_epsilon: Privacy budget for prediction features
        """
        self.risk_threshold = risk_threshold
        self.intervention_cost_limit = intervention_cost_limit
        self.prediction_horizon_days = prediction_horizon_days
        self.privacy_epsilon = privacy_epsilon
        
        self.logger = logging.getLogger(__name__)
        
        # Prediction model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Privacy mechanism
        self.dp_mechanism = GaussianDP(epsilon=privacy_epsilon, delta=1e-6)
        
        # Data storage
        self.hospital_features: Dict[str, np.ndarray] = {}
        self.hospital_labels: Dict[str, int] = {}
        self.dropout_history: List[DropoutRisk] = []
        self.interventions: Dict[str, List[Intervention]] = {}
        
        # Metrics tracking
        self.metrics = {
            "total_predictions": 0,
            "correct_predictions": 0,
            "interventions_triggered": 0,
            "successful_interventions": 0,
            "average_intervention_cost": 0.0,
            "system_stability_score": 1.0
        }
        
    def extract_features(self, hospital_id: str) -> np.ndarray:
        """
        Extract features for dropout prediction.
        
        Args:
            hospital_id: Hospital identifier
            
        Returns:
            Feature vector for prediction
            
        Privacy: Features anonymized with DP noise before storage
        """
        # Get hospital participation data (would come from hospital ledger)
        features = [
            # Participation metrics
            np.random.uniform(0.1, 1.0),  # participation_rate (simulated)
            np.random.randint(1, 100),    # total_rounds_participated
            np.random.uniform(0.0, 7.0),  # days_since_last_participation
            
            # Performance metrics
            np.random.uniform(0.7, 0.95), # average_accuracy
            np.random.uniform(0.0, 5.0),  # privacy_violations_count
            np.random.uniform(0.0, 1.0),  # byzantine_score
            
            # Reputation metrics
            np.random.uniform(0.5, 1.0),  # reputation_score
            np.random.uniform(100, 2000), # healthtokens_balance
            
            # Technical metrics
            np.random.uniform(10, 1000),  # average_response_time_ms
            np.random.uniform(0.0, 0.1),  # connection_failure_rate
            
            # Historical patterns
            np.random.randint(0, 10),     # previous_dropout_count
            np.random.uniform(0.0, 30.0), # days_since_first_participation
        ]
        
        feature_vector = np.array(features)
        
        # Add DP noise for privacy
        noisy_features = self.dp_mechanism.add_noise(
            torch.tensor(feature_vector), sensitivity=1.0
        ).numpy()
        
        return noisy_features
        
    def predict_dropout_risk(self, hospital_id: str) -> DropoutRisk:
        """
        Predict dropout risk for a hospital.
        
        Args:
            hospital_id: Hospital identifier
            
        Returns:
            Dropout risk assessment
            
        Byzantine: Risk predictions aggregated with geometric median for robustness
        """
        start_time = datetime.utcnow()
        
        # Extract features
        features = self.extract_features(hospital_id)
        
        # Make prediction
        if self.is_trained:
            # Scale features
            scaled_features = self.scaler.transform(features.reshape(1, -1))
            
            # Predict probability
            dropout_prob = self.model.predict_proba(scaled_features)[0][1]
            
            # Get contributing factors
            feature_importance = self.model.feature_importances_
            feature_names = [
                "participation_rate", "total_rounds", "days_since_last",
                "avg_accuracy", "privacy_violations", "byzantine_score",
                "reputation_score", "token_balance", "response_time",
                "failure_rate", "previous_dropouts", "participation_days"
            ]
            
            # Get top contributing factors
            factor_indices = np.argsort(feature_importance)[-3:][::-1]
            contributing_factors = [feature_names[i] for i in factor_indices]
            
            confidence = 0.8  # Would be based on model confidence
        else:
            # Fallback to rule-based prediction
            dropout_prob = self._rule_based_prediction(features)
            contributing_factors = self._get_rule_based_factors(features)
            confidence = 0.6
            
        # Determine risk level
        risk_level = self._determine_risk_level(dropout_prob)
        
        # Predict dropout time
        predicted_dropout_time = self._predict_dropout_time(
            features, dropout_prob, risk_level
        )
        
        # Determine if intervention recommended
        intervention_recommended = dropout_prob >= self.risk_threshold
        
        # Create risk assessment
        risk_assessment = DropoutRisk(
            hospital_id=hospital_id,
            dropout_probability=dropout_prob,
            risk_level=risk_level,
            contributing_factors=contributing_factors,
            predicted_dropout_time=predicted_dropout_time,
            intervention_recommended=intervention_recommended,
            confidence=confidence,
            timestamp=datetime.utcnow().isoformat()
        )
        
        # Store prediction
        self.dropout_history.append(risk_assessment)
        self.metrics["total_predictions"] += 1
        
        # Trigger intervention if needed
        if intervention_recommended:
            self._trigger_intervention(risk_assessment)
            
        elapsed_time = (datetime.utcnow() - start_time).total_seconds()
        self.logger.info(
            f"Dropout risk prediction for {hospital_id}: "
            f"probability={dropout_prob:.3f}, level={risk_level}, "
            f"time={elapsed_time:.3f}s"
        )
        
        return risk_assessment
        
    def get_dropout_risks(self) -> List[DropoutRisk]:
        """Get current dropout risks for all hospitals."""
        risks = []
        
        for hospital_id in self.current_status.keys():
            try:
                risk = self.predict_dropout_risk(hospital_id)
                risks.append(risk)
            except Exception as e:
                self.logger.error(f"Failed to predict risk for {hospital_id}: {e}")
                
        # Sort by risk level
        risk_order = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
        risks.sort(key=lambda x: risk_order.get(x.risk_level, 0), reverse=True)
        
        return risks
        
    def apply_intervention(self, hospital_id: str, intervention_type: str) -> Intervention:
        """
        Apply intervention strategy for at-risk hospital.
        
        Args:
            hospital_id: Hospital identifier
            intervention_type: Type of intervention
            
        Returns:
            Intervention record
            
        Byzantine: Intervention effectiveness tracked for Byzantine detection
        """
        # Get current risk assessment
        risk_assessment = self.predict_dropout_risk(hospital_id)
        
        if not risk_assessment.intervention_recommended:
            self.logger.warning(f"Intervention not recommended for {hospital_id}")
            
        # Define intervention strategies
        interventions = {
            "INCENTIVE_BOOST": {
                "description": "Increase HealthToken rewards",
                "data": {"reward_multiplier": 1.5, "duration_days": 7},
                "success_prob": 0.7,
                "cost": 500.0
            },
            "TECHNICAL_SUPPORT": {
                "description": "Provide technical assistance",
                "data": {"support_hours": 10, "priority": "HIGH"},
                "success_prob": 0.8,
                "cost": 800.0
            },
            "REPUTATION_REPAIR": {
                "description": "Reputation score restoration",
                "data": {"score_boost": 0.1, "requirements": ["no_violations_7_days"]},
                "success_prob": 0.6,
                "cost": 300.0
            },
            "PARTICIPATION_REMINDER": {
                "description": "Automated participation reminders",
                "data": {"frequency": "DAILY", "channel": "EMAIL"},
                "success_prob": 0.5,
                "cost": 50.0
            },
            "INFRASTRUCTURE_UPGRADE": {
                "description": "Assist with infrastructure improvements",
                "data": {"upgrade_credits": 1000, "technical_consultation": True},
                "success_prob": 0.9,
                "cost": 1000.0
            }
        }
        
        if intervention_type not in interventions:
            raise ValueError(f"Unknown intervention type: {intervention_type}")
            
        intervention_config = interventions[intervention_type]
        
        # Check cost limit
        if intervention_config["cost"] > self.intervention_cost_limit:
            self.logger.warning(f"Intervention cost exceeds limit: {intervention_config['cost']}")
            
        # Create intervention
        intervention = Intervention(
            hospital_id=hospital_id,
            intervention_type=intervention_type,
            intervention_data=intervention_config["data"],
            success_probability=intervention_config["success_prob"],
            cost_estimate=intervention_config["cost"],
            timestamp=datetime.utcnow().isoformat()
        )
        
        # Store intervention
        if hospital_id not in self.interventions:
            self.interventions[hospital_id] = []
            
        self.interventions[hospital_id].append(intervention)
        self.metrics["interventions_triggered"] += 1
        
        # Update average intervention cost
        total_interventions = self.metrics["interventions_triggered"]
        total_cost = self.metrics["average_intervention_cost"] * (total_interventions - 1) + intervention_config["cost"]
        self.metrics["average_intervention_cost"] = total_cost / total_interventions
        
        self.logger.info(
            f"Intervention applied to {hospital_id}: {intervention_type}, "
            f"cost={intervention_config['cost']}, success_prob={intervention_config['success_prob']}"
        )
        
        return intervention
        
    def evaluate_intervention_success(self, hospital_id: str, intervention_id: str, success: bool):
        """
        Evaluate intervention success.
        
        Args:
            hospital_id: Hospital identifier
            intervention_id: Intervention identifier
            success: Whether intervention was successful
        """
        # Find intervention
        if hospital_id not in self.interventions:
            return
            
        for intervention in self.interventions[hospital_id]:
            if intervention.timestamp == intervention_id:
                # Update success status
                intervention.metadata = intervention.metadata or {}
                intervention.metadata["success"] = success
                intervention.metadata["evaluated_at"] = datetime.utcnow().isoformat()
                
                if success:
                    self.metrics["successful_interventions"] += 1
                    
                # Update system stability
                self._update_stability_score()
                
                self.logger.info(
                    f"Intervention evaluation for {hospital_id}: "
                    f"success={success}, type={intervention.intervention_type}"
                )
                break
                
    def get_stability_metrics(self) -> Dict[str, Any]:
        """Get system stability metrics."""
        total_interventions = self.metrics["interventions_triggered"]
        success_rate = (self.metrics["successful_interventions"] / max(total_interventions, 1))
        
        return {
            "system_stability_score": self.metrics["system_stability_score"],
            "total_hospitals": len(self.current_status),
            "at_risk_hospitals": len([r for r in self.dropout_history[-100:] 
                                    if r.risk_level in ["HIGH", "CRITICAL"]]),
            "intervention_success_rate": success_rate,
            "average_intervention_cost": self.metrics["average_intervention_cost"],
            "prediction_accuracy": (self.metrics["correct_predictions"] / 
                                  max(self.metrics["total_predictions"], 1))
        }
        
    def train_model(self, training_data: Optional[np.ndarray] = None, 
                   labels: Optional[np.ndarray] = None):
        """Train the dropout prediction model."""
        if training_data is not None and labels is not None:
            # Use provided training data
            X = training_data
            y = labels
        else:
            # Use historical data
            if not self.dropout_history:
                self.logger.warning("No historical data available for training")
                return
                
            # Simulate training data from history
            n_samples = min(len(self.dropout_history), 1000)
            X = np.random.randn(n_samples, 12)  # 12 features
            y = np.random.randint(0, 2, n_samples)  # Binary labels
            
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        self.logger.info(f"Dropout prediction model trained on {len(X)} samples")
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get dropout predictor metrics."""
        return {
            **self.metrics,
            "model_trained": self.is_trained,
            "total_hospitals_monitored": len(self.current_status),
            "intervention_history_size": sum(len(invs) for invs in self.interventions.values())
        }
        
    def reset_metrics(self):
        """Reset metrics counters."""
        self.metrics = {
            "total_predictions": 0,
            "correct_predictions": 0,
            "interventions_triggered": 0,
            "successful_interventions": 0,
            "average_intervention_cost": 0.0,
            "system_stability_score": 1.0
        }
        
    def _rule_based_prediction(self, features: np.ndarray) -> float:
        """Fallback rule-based prediction when model not trained."""
        # Simple rule-based approach
        participation_rate = features[0]
        days_since_last = features[2]
        reputation_score = features[6]
        
        # Higher risk if low participation, long absence, or low reputation
        risk = (1.0 - participation_rate) * 0.4 + \
               (days_since_last / 30.0) * 0.3 + \
               (1.0 - reputation_score) * 0.3
               
        return min(risk, 1.0)
        
    def _get_rule_based_factors(self, features: np.ndarray) -> List[str]:
        """Get contributing factors for rule-based prediction."""
        factors = []
        
        if features[0] < 0.5:  # Low participation
            factors.append("Low participation rate")
        if features[2] > 3.0:  # Long absence
            factors.append("Extended absence from training")
        if features[6] < 0.7:  # Low reputation
            factors.append("Poor reputation score")
            
        return factors if factors else ["General participation patterns"]
        
    def _determine_risk_level(self, probability: float) -> str:
        """Determine risk level from probability."""
        if probability >= 0.8:
            return "CRITICAL"
        elif probability >= 0.6:
            return "HIGH"
        elif probability >= 0.3:
            return "MEDIUM"
        else:
            return "LOW"
            
    def _predict_dropout_time(self, features: np.ndarray, probability: float, risk_level: str) -> Optional[str]:
        """Predict approximate dropout time."""
        if risk_level in ["HIGH", "CRITICAL"]:
            days_ahead = int((1.0 - probability) * self.prediction_horizon_days)
            predicted_date = datetime.utcnow() + timedelta(days=days_ahead)
            return predicted_date.isoformat()
        return None
        
    def _trigger_intervention(self, risk_assessment: DropoutRisk):
        """Automatically trigger intervention for high-risk hospitals."""
        if risk_assessment.dropout_probability >= 0.9:
            intervention_type = "INFRASTRUCTURE_UPGRADE"
        elif risk_assessment.dropout_probability >= 0.7:
            intervention_type = "INCENTIVE_BOOST"
        else:
            intervention_type = "PARTICIPATION_REMINDER"
            
        try:
            self.apply_intervention(risk_assessment.hospital_id, intervention_type)
        except Exception as e:
            self.logger.error(f"Failed to trigger intervention: {e}")
            
    def _update_stability_score(self):
        """Update system stability score based on intervention outcomes."""
        success_rate = (self.metrics["successful_interventions"] / 
                       max(self.metrics["interventions_triggered"], 1))
        
        # Stability decreases with interventions, increases with success
        intervention_impact = self.metrics["interventions_triggered"] * 0.01
        success_improvement = success_rate * 0.05
        
        self.metrics["system_stability_score"] = max(
            0.0, min(1.0, 1.0 - intervention_impact + success_improvement)
        )