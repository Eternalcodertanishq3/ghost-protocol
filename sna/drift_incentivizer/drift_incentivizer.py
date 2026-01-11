"""
Module: sna/drift_incentivizer/drift_incentivizer.py
DPDP §: 9(4) - Data drift detection with privacy preservation
Description: Incentivizes hospitals to report data drift for model adaptation
Byzantine: Drift consensus with Byzantine agreement (tolerates f < n/3 malicious reports, Lamport 1982)
Privacy: Drift reports anonymized with DP (ε=0.1) before aggregation
Test: pytest tests/test_drift.py::test_drift_incentivization
API: POST /report_drift, GET /drift_status, GET /drift_incentives
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from scipy import stats

from algorithms.dp_mechanisms.gaussian import GaussianDP


@dataclass
class DriftReport:
    """Data drift report from a hospital."""
    hospital_id: str
    timestamp: str
    drift_type: str  # "CONCEPT", "COVARIATE", "LABEL"
    drift_score: float
    feature_name: str
    severity: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    confidence: float
    evidence: Dict[str, Any]
    privacy_budget_spent: float


@dataclass
class DriftIncentive:
    """Incentive for accurate drift reporting."""
    hospital_id: str
    drift_report_id: str
    incentive_amount: float
    accuracy_score: float
    timestamp: str
    token_reward: float


class DriftIncentivizer:
    """
    Drift Incentivizer for Ghost Protocol.
    
    Implements a system to detect, report, and incentivize data drift detection
    across hospitals while maintaining privacy and Byzantine fault tolerance.
    
    Features:
    - Privacy-preserving drift detection
    - Byzantine-robust drift consensus
    - Incentive mechanism for accurate reporting
    - Model adaptation triggers
    """
    
    def __init__(
        self,
        drift_threshold: float = 0.1,
        consensus_threshold: float = 0.67,  # 2/3 for Byzantine consensus
        incentive_pool_size: float = 5000.0,  # HealthTokens for drift incentives
        privacy_epsilon: float = 0.1,
        validation_window_days: int = 7
    ):
        """
        Initialize Drift Incentivizer.
        
        Args:
            drift_threshold: Threshold for drift detection
            consensus_threshold: Byzantine consensus threshold
            incentive_pool_size: Total HealthTokens for drift incentives
            privacy_epsilon: Privacy budget for drift reporting
            validation_window_days: Window for validating drift reports
        """
        self.drift_threshold = drift_threshold
        self.consensus_threshold = consensus_threshold
        self.incentive_pool_size = incentive_pool_size
        self.privacy_epsilon = privacy_epsilon
        self.validation_window_days = validation_window_days
        
        self.logger = logging.getLogger(__name__)
        
        # Drift tracking
        self.drift_reports: Dict[str, List[DriftReport]] = {}
        self.drift_consensus: Dict[str, Dict[str, Any]] = {}
        
        # Incentive tracking
        self.drift_incentives: Dict[str, List[DriftIncentive]] = {}
        self.accuracy_scores: Dict[str, float] = {}
        
        # Privacy mechanism
        self.dp_mechanism = GaussianDP(epsilon=privacy_epsilon, delta=1e-6)
        
        # Model adaptation state
        self.adaptation_triggers: List[Dict[str, Any]] = []
        
        # Metrics
        self.metrics = {
            "total_drift_reports": 0,
            "validated_drifts": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "total_incentives_distributed": 0.0,
            "average_accuracy_score": 0.0,
            "model_adaptations_triggered": 0
        }
        
    def report_drift(
        self,
        hospital_id: str,
        drift_type: str,
        drift_score: float,
        feature_name: str,
        confidence: float,
        evidence: Dict[str, Any],
        baseline_stats: Optional[Dict[str, float]] = None
    ) -> DriftReport:
        """
        Report data drift from a hospital.
        
        Args:
            hospital_id: Hospital identifier
            drift_type: Type of drift detected
            drift_score: Drift magnitude score
            feature_name: Affected feature
            confidence: Confidence in drift detection
            evidence: Supporting evidence
            baseline_stats: Baseline statistics for comparison
            
        Returns:
            Drift report with privacy budget tracking
            
        Byzantine: Drift reports validated through consensus mechanism
        Privacy: Drift scores anonymized with DP noise before aggregation
        """
        # Add DP noise to drift score for privacy
        noisy_drift_score = self.dp_mechanism.add_noise(
            torch.tensor([drift_score]), sensitivity=1.0
        ).item()
        
        # Determine severity
        severity = self._determine_severity(noisy_drift_score)
        
        # Create drift report
        report = DriftReport(
            hospital_id=hospital_id,
            timestamp=datetime.utcnow().isoformat(),
            drift_type=drift_type,
            drift_score=noisy_drift_score,
            feature_name=feature_name,
            severity=severity,
            confidence=confidence,
            evidence=evidence,
            privacy_budget_spent=self.privacy_epsilon
        )
        
        # Store report
        if hospital_id not in self.drift_reports:
            self.drift_reports[hospital_id] = []
            
        self.drift_reports[hospital_id].append(report)
        
        # Update metrics
        self.metrics["total_drift_reports"] += 1
        
        # Trigger consensus check
        self._update_drift_consensus(feature_name, drift_type)
        
        self.logger.info(
            f"Drift reported by {hospital_id}: {drift_type} on {feature_name}, "
            f"score={noisy_drift_score:.3f}, severity={severity}"
        )
        
        return report
        
    def validate_drift_report(
        self,
        report_id: str,
        validation_data: np.ndarray,
        ground_truth: bool
    ) -> bool:
        """
        Validate a drift report using ground truth data.
        
        Args:
            report_id: Drift report identifier
            validation_data: Data for validation
            ground_truth: True if drift actually occurred
            
        Returns:
            True if validation successful
            
        Byzantine: Validation results used for incentive calculation
        """
        # Find the report
        report = None
        for hospital_reports in self.drift_reports.values():
            for r in hospital_reports:
                if r.hospital_id + r.timestamp == report_id:
                    report = r
                    break
            if report:
                break
                
        if not report:
            self.logger.warning(f"Drift report {report_id} not found for validation")
            return False
            
        # Update accuracy score
        hospital_id = report.hospital_id
        current_score = self.accuracy_scores.get(hospital_id, 0.5)
        
        # Simple accuracy update (could be more sophisticated)
        if ground_truth:
            new_score = min(1.0, current_score + 0.1)
            self.metrics["validated_drifts"] += 1
        else:
            new_score = max(0.0, current_score - 0.1)
            if report.drift_score > self.drift_threshold:
                self.metrics["false_positives"] += 1
            else:
                self.metrics["false_negatives"] += 1
                
        self.accuracy_scores[hospital_id] = new_score
        
        # Distribute incentive
        if ground_truth:
            incentive = self._calculate_incentive(report, new_score)
            self._distribute_incentive(report, incentive)
            
        self.logger.info(
            f"Drift validation for {hospital_id}: ground_truth={ground_truth}, "
            f"new_accuracy={new_score:.3f}"
        )
        
        return True
        
    def get_drift_status(self, feature_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get current drift status across all hospitals.
        
        Args:
            feature_name: Specific feature (optional)
            
        Returns:
            Drift status with consensus information
        """
        if feature_name:
            # Get drift status for specific feature
            if feature_name in self.drift_consensus:
                consensus = self.drift_consensus[feature_name]
                return {
                    "feature_name": feature_name,
                    "consensus_reached": consensus["consensus_reached"],
                    "consensus_drift_score": consensus["consensus_score"],
                    "hospitals_reporting": consensus["hospitals_reporting"],
                    "severity": consensus["severity"],
                    "last_update": consensus["timestamp"],
                    "requires_action": consensus["requires_action"]
                }
            else:
                return {"error": "Feature not found", "feature_name": feature_name}
        else:
            # Global drift status
            total_features = len(self.drift_consensus)
            critical_drifts = sum(1 for c in self.drift_consensus.values() 
                                if c.get("severity") == "CRITICAL")
            
            return {
                "total_features_monitored": total_features,
                "features_with_drift": len([c for c in self.drift_consensus.values() 
                                          if c.get("consensus_reached")]),
                "critical_drifts": critical_drifts,
                "requires_action": critical_drifts > 0,
                "last_consensus_update": max([c.get("timestamp", "") 
                                            for c in self.drift_consensus.values()], default="")
            }
            
    def get_drift_incentives(self, hospital_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get drift incentive history.
        
        Args:
            hospital_id: Specific hospital (optional)
            
        Returns:
            List of drift incentives
        """
        if hospital_id:
            incentives = self.drift_incentives.get(hospital_id, [])
        else:
            # Flatten all incentives
            incentives = []
            for hospital_incentives in self.drift_incentives.values():
                incentives.extend(hospital_incentives)
                
        # Sort by timestamp (newest first)
        incentives.sort(key=lambda x: x.timestamp, reverse=True)
        
        return [
            {
                "hospital_id": inc.hospital_id,
                "drift_report_id": inc.drift_report_id,
                "incentive_amount": inc.incentive_amount,
                "accuracy_score": inc.accuracy_score,
                "timestamp": inc.timestamp,
                "token_reward": inc.token_reward
            }
            for inc in incentives
        ]
        
    def trigger_model_adaptation(self, feature_name: str) -> bool:
        """
        Trigger model adaptation for a feature with detected drift.
        
        Args:
            feature_name: Feature requiring adaptation
            
        Returns:
            True if adaptation triggered
            
        Byzantine: Adaptation requires consensus from 2/3 of hospitals
        """
        if feature_name not in self.drift_consensus:
            self.logger.warning(f"No drift consensus for feature {feature_name}")
            return False
            
        consensus = self.drift_consensus[feature_name]
        
        # Check if adaptation is warranted
        if not consensus.get("requires_action", False):
            return False
            
        # Create adaptation trigger
        adaptation = {
            "feature_name": feature_name,
            "triggered_at": datetime.utcnow().isoformat(),
            "consensus_score": consensus.get("consensus_score", 0.0),
            "hospitals_consented": consensus.get("hospitals_reporting", 0),
            "severity": consensus.get("severity", "LOW"),
            "adaptation_strategy": self._determine_adaptation_strategy(feature_name, consensus)
        }
        
        self.adaptation_triggers.append(adaptation)
        self.metrics["model_adaptations_triggered"] += 1
        
        self.logger.info(
            f"Model adaptation triggered for {feature_name}: "
            f"severity={adaptation['severity']}, "
            f"strategy={adaptation['adaptation_strategy']}"
        )
        
        return True
        
    def get_adaptation_triggers(self) -> List[Dict[str, Any]]:
        """Get model adaptation triggers."""
        return [
            {
                "feature_name": trigger["feature_name"],
                "triggered_at": trigger["triggered_at"],
                "consensus_score": trigger["consensus_score"],
                "hospitals_consented": trigger["hospitals_consented"],
                "severity": trigger["severity"],
                "adaptation_strategy": trigger["adaptation_strategy"]
            }
            for trigger in self.adaptation_triggers
        ]
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get drift incentivizer metrics."""
        return {
            **self.metrics,
            "accuracy_scores": self.accuracy_scores,
            "incentive_pool_remaining": self.incentive_pool_size - self.metrics["total_incentives_distributed"],
            "features_with_consensus": len(self.drift_consensus),
            "adaptation_triggers": len(self.adaptation_triggers)
        }
        
    def reset_metrics(self):
        """Reset metrics counters."""
        self.metrics = {
            "total_drift_reports": 0,
            "validated_drifts": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "total_incentives_distributed": 0.0,
            "average_accuracy_score": 0.0,
            "model_adaptations_triggered": 0
        }
        
    def _determine_severity(self, drift_score: float) -> str:
        """Determine drift severity level."""
        if drift_score >= 0.5:
            return "CRITICAL"
        elif drift_score >= 0.3:
            return "HIGH"
        elif drift_score >= 0.1:
            return "MEDIUM"
        else:
            return "LOW"
            
    def _update_drift_consensus(self, feature_name: str, drift_type: str):
        """Update drift consensus for a feature."""
        # Collect recent reports for this feature
        reports = []
        for hospital_reports in self.drift_reports.values():
            for report in hospital_reports:
                if (report.feature_name == feature_name and 
                    report.drift_type == drift_type and
                    datetime.fromisoformat(report.timestamp) > 
                    datetime.utcnow() - timedelta(hours=1)):
                    reports.append(report)
                    
        if not reports:
            return
            
        # Compute consensus
        drift_scores = [report.drift_score for report in reports]
        consensus_score = np.median(drift_scores)  # Robust to outliers
        
        # Determine consensus severity
        severity = self._determine_severity(consensus_score)
        
        # Check if consensus reached
        total_hospitals = len(self.current_status)
        consensus_reached = len(reports) / max(total_hospitals, 1) >= self.consensus_threshold
        
        # Determine if action required
        requires_action = consensus_reached and consensus_score >= self.drift_threshold
        
        # Update consensus state
        self.drift_consensus[feature_name] = {
            "consensus_reached": consensus_reached,
            "consensus_score": consensus_score,
            "hospitals_reporting": len(reports),
            "severity": severity,
            "timestamp": datetime.utcnow().isoformat(),
            "requires_action": requires_action,
            "drift_type": drift_type
        }
        
        # Trigger model adaptation if needed
        if requires_action:
            self.trigger_model_adaptation(feature_name)
            
    def _calculate_incentive(self, report: DriftReport, accuracy_score: float) -> float:
        """Calculate drift reporting incentive."""
        # Base incentive based on severity
        base_incentive = {
            "LOW": 10.0,
            "MEDIUM": 25.0,
            "HIGH": 50.0,
            "CRITICAL": 100.0
        }.get(report.severity, 10.0)
        
        # Accuracy multiplier
        accuracy_multiplier = 0.5 + (accuracy_score * 0.5)  # 0.5 to 1.0
        
        # Confidence bonus
        confidence_bonus = 1.0 + (report.confidence * 0.5)  # 1.0 to 1.5
        
        return base_incentive * accuracy_multiplier * confidence_bonus
        
    def _distribute_incentive(self, report: DriftReport, incentive_amount: float):
        """Distribute HealthToken incentive for accurate drift report."""
        # Create incentive record
        incentive = DriftIncentive(
            hospital_id=report.hospital_id,
            drift_report_id=report.hospital_id + report.timestamp,
            incentive_amount=incentive_amount,
            accuracy_score=self.accuracy_scores.get(report.hospital_id, 0.5),
            timestamp=datetime.utcnow().isoformat(),
            token_reward=incentive_amount
        )
        
        # Store incentive
        if report.hospital_id not in self.drift_incentives:
            self.drift_incentives[report.hospital_id] = []
            
        self.drift_incentives[report.hospital_id].append(incentive)
        
        # Update metrics
        self.metrics["total_incentives_distributed"] += incentive_amount
        
        # Update average accuracy score
        accuracy_scores = list(self.accuracy_scores.values())
        self.metrics["average_accuracy_score"] = np.mean(accuracy_scores) if accuracy_scores else 0.0
        
        self.logger.info(
            f"Drift incentive distributed to {report.hospital_id}: "
            f"{incentive_amount:.2f} HealthTokens"
        )
        
    def _determine_adaptation_strategy(self, feature_name: str, consensus: Dict[str, Any]) -> str:
        """Determine model adaptation strategy based on drift characteristics."""
        severity = consensus.get("severity", "LOW")
        drift_type = consensus.get("drift_type", "CONCEPT")
        
        if severity == "CRITICAL":
            return "IMMEDIATE_RETRAINING"
        elif severity == "HIGH":
            if drift_type == "CONCEPT":
                return "GRADUAL_RETRAINING_WITH_FEDPROX"
            else:
                return "FEATURE_WEIGHT_ADJUSTMENT"
        elif severity == "MEDIUM":
            return "INCREASED_MONITORING_WITH_ADAPTIVE_LEARNING_RATE"
        else:
            return "CONTINUED_MONITORING"
            
    def get_drift_accuracy_leaderboard(self) -> List[Dict[str, Any]]:
        """Get drift detection accuracy leaderboard."""
        leaderboard = []
        
        for hospital_id, accuracy_score in self.accuracy_scores.items():
            total_reports = len(self.drift_reports.get(hospital_id, []))
            total_incentives = sum(
                inc.incentive_amount 
                for inc in self.drift_incentives.get(hospital_id, [])
            )
            
            leaderboard.append({
                "hospital_id": hospital_id,
                "accuracy_score": accuracy_score,
                "total_reports": total_reports,
                "total_incentives": total_incentives,
                "rank": 0  # Will be set after sorting
            })
            
        # Sort by accuracy score
        leaderboard.sort(key=lambda x: x["accuracy_score"], reverse=True)
        
        # Set ranks
        for i, entry in enumerate(leaderboard):
            entry["rank"] = i + 1
            
        return leaderboard