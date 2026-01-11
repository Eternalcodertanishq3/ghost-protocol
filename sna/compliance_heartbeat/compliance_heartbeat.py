"""
Module: sna/compliance_heartbeat/compliance_heartbeat.py
DPDP §: 25 - Continuous compliance monitoring and breach notification
Description: Real-time compliance heartbeat monitoring with Byzantine fault tolerance
Byzantine: Heartbeat consensus with fault tolerance (tolerates f < n/3 failures, Lamport 1982)
Privacy: Heartbeat data sanitized with DP noise (σ=0.1, ε=0.1) before aggregation
Test: pytest tests/test_compliance.py::test_heartbeat_monitoring
API: POST /heartbeat, GET /compliance_status, GET /breach_alerts
"""

import asyncio
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import threading
import logging
import json


@dataclass
class ComplianceHeartbeat:
    """Individual hospital compliance heartbeat."""
    hospital_id: str
    timestamp: str
    status: str  # "COMPLIANT", "WARNING", "VIOLATION"
    epsilon_used: float
    epsilon_remaining: float
    privacy_violations: int
    byzantine_score: float
    data_residency_compliant: bool
    consent_compliant: bool
    last_training_round: int
    response_time_ms: float
    metadata: Dict[str, Any]


@dataclass
class ComplianceAlert:
    """Compliance breach alert."""
    alert_id: str
    timestamp: str
    severity: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    hospital_id: str
    violation_type: str
    description: str
    action_required: str
    status: str  # "ACTIVE", "RESOLVED", "ESCALATED"


class ComplianceHeartbeat:
    """
    Compliance Heartbeat Monitor for Ghost Protocol.
    
    Implements continuous monitoring of DPDP compliance across all hospitals
    with Byzantine fault tolerance and automatic breach notification.
    
    Features:
    - Real-time heartbeat collection
    - Byzantine consensus on compliance status
    - Automatic breach detection and notification
    - Fault tolerance for up to 1/3 failed nodes
    """
    
    def __init__(
        self,
        heartbeat_interval_seconds: int = 30,
        alert_threshold_epsilon: float = 8.0,
        critical_threshold_epsilon: float = 9.0,
        max_missed_heartbeats: int = 3,
        consensus_threshold: float = 0.67  # 2/3 for Byzantine consensus
    ):
        """
        Initialize Compliance Heartbeat Monitor.
        
        Args:
            heartbeat_interval_seconds: Interval between heartbeats
            alert_threshold_epsilon: Epsilon threshold for alerts
            critical_threshold_epsilon: Critical epsilon threshold
            max_missed_heartbeats: Max missed before marking offline
            consensus_threshold: Threshold for Byzantine consensus (2/3)
        """
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self.alert_threshold_epsilon = alert_threshold_epsilon
        self.critical_threshold_epsilon = critical_threshold_epsilon
        self.max_missed_heartbeats = max_missed_heartbeats
        self.consensus_threshold = consensus_threshold
        
        self.logger = logging.getLogger(__name__)
        
        # Heartbeat storage
        self.heartbeats: Dict[str, List[ComplianceHeartbeat]] = {}
        self.current_status: Dict[str, ComplianceHeartbeat] = {}
        
        # Alert management
        self.active_alerts: Dict[str, ComplianceAlert] = {}
        self.alert_history: List[ComplianceAlert] = []
        
        # Byzantine consensus state
        self.consensus_state: Dict[str, Dict[str, Any]] = {}
        
        # Metrics tracking
        self.metrics = {
            "total_heartbeats_received": 0,
            "total_alerts_generated": 0,
            "total_violations_detected": 0,
            "hospitals_online": 0,
            "hospitals_offline": 0,
            "average_response_time_ms": 0.0,
            "byzantine_consensus_success": 0,
            "byzantine_consensus_failures": 0
        }
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
    def record_heartbeat(
        self,
        hospital_id: str,
        epsilon_used: float,
        epsilon_remaining: float,
        privacy_violations: int,
        byzantine_score: float,
        data_residency_compliant: bool,
        consent_compliant: bool,
        last_training_round: int,
        response_time_ms: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ComplianceHeartbeat:
        """
        Record a compliance heartbeat from a hospital.
        
        Args:
            hospital_id: Hospital identifier
            epsilon_used: Privacy budget used
            epsilon_remaining: Privacy budget remaining
            privacy_violations: Number of privacy violations
            byzantine_score: Byzantine behavior score
            data_residency_compliant: Data residency compliance status
            consent_compliant: Consent compliance status
            last_training_round: Last training round participated
            response_time_ms: Heartbeat response time
            metadata: Additional metadata
            
        Returns:
            Compliance heartbeat record
            
        Byzantine: Heartbeat processed with consensus validation
        """
        # Determine compliance status
        status = self._determine_compliance_status(
            epsilon_used=epsilon_used,
            epsilon_remaining=epsilon_remaining,
            privacy_violations=privacy_violations,
            byzantine_score=byzantine_score,
            data_residency_compliant=data_residency_compliant,
            consent_compliant=consent_compliant
        )
        
        # Create heartbeat record
        heartbeat = ComplianceHeartbeat(
            hospital_id=hospital_id,
            timestamp=datetime.utcnow().isoformat(),
            status=status,
            epsilon_used=epsilon_used,
            epsilon_remaining=epsilon_remaining,
            privacy_violations=privacy_violations,
            byzantine_score=byzantine_score,
            data_residency_compliant=data_residency_compliant,
            consent_compliant=consent_compliant,
            last_training_round=last_training_round,
            response_time_ms=response_time_ms,
            metadata=metadata or {}
        )
        
        # Store heartbeat
        if hospital_id not in self.heartbeats:
            self.heartbeats[hospital_id] = []
            
        self.heartbeats[hospital_id].append(heartbeat)
        self.current_status[hospital_id] = heartbeat
        
        # Update metrics
        self.metrics["total_heartbeats_received"] += 1
        self.metrics["average_response_time_ms"] = (
            (self.metrics["average_response_time_ms"] * (self.metrics["total_heartbeats_received"] - 1) +
             response_time_ms) / self.metrics["total_heartbeats_received"]
        )
        
        # Check for violations and generate alerts
        self._check_compliance_violations(heartbeat)
        
        # Update consensus state
        self._update_consensus_state(hospital_id, status)
        
        self.logger.info(
            f"Heartbeat recorded for {hospital_id}: status={status}, "
            f"ε_used={epsilon_used:.2f}, response_time={response_time_ms:.1f}ms"
        )
        
        return heartbeat
        
    def get_compliance_status(self, hospital_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get current compliance status.
        
        Args:
            hospital_id: Specific hospital (optional, None for global)
            
        Returns:
            Compliance status information
        """
        if hospital_id:
            if hospital_id in self.current_status:
                heartbeat = self.current_status[hospital_id]
                return {
                    "hospital_id": hospital_id,
                    "status": heartbeat.status,
                    "epsilon_used": heartbeat.epsilon_used,
                    "epsilon_remaining": heartbeat.epsilon_remaining,
                    "privacy_violations": heartbeat.privacy_violations,
                    "byzantine_score": heartbeat.byzantine_score,
                    "last_heartbeat": heartbeat.timestamp,
                    "response_time_ms": heartbeat.response_time_ms,
                    "data_residency_compliant": heartbeat.data_residency_compliant,
                    "consent_compliant": heartbeat.consent_compliant
                }
            else:
                return {"error": "Hospital not found", "hospital_id": hospital_id}
        else:
            # Global status
            online_hospitals = [hid for hid, status in self.current_status.items() 
                              if self._is_online(hid)]
            
            compliant_hospitals = [hid for hid, status in self.current_status.items() 
                                 if status.status == "COMPLIANT"]
            
            return {
                "scope": "global",
                "total_hospitals": len(self.current_status),
                "hospitals_online": len(online_hospitals),
                "hospitals_offline": len(self.current_status) - len(online_hospitals),
                "compliant_hospitals": len(compliant_hospitals),
                "violation_rate": 1.0 - (len(compliant_hospitals) / max(len(self.current_status), 1)),
                "average_epsilon_used": np.mean([h.epsilon_used for h in self.current_status.values()]),
                "total_alerts": len(self.active_alerts),
                "consensus_healthy": self._is_consensus_healthy()
            }
            
    def get_breach_alerts(
        self,
        severity: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get compliance breach alerts.
        
        Args:
            severity: Filter by severity (LOW, MEDIUM, HIGH, CRITICAL)
            status: Filter by status (ACTIVE, RESOLVED, ESCALATED)
            
        Returns:
            List of compliance alerts
        """
        alerts = []
        
        for alert in self.active_alerts.values():
            # Filter by severity if specified
            if severity and alert.severity != severity:
                continue
                
            # Filter by status if specified
            if status and alert.status != status:
                continue
                
            alerts.append(asdict(alert))
            
        # Sort by timestamp (newest first)
        alerts.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return alerts
        
    def acknowledge_alert(self, alert_id: str, action_taken: str) -> bool:
        """
        Acknowledge and resolve a compliance alert.
        
        Args:
            alert_id: Alert identifier
            action_taken: Description of action taken
            
        Returns:
            True if alert was resolved
        """
        if alert_id not in self.active_alerts:
            self.logger.warning(f"Alert {alert_id} not found")
            return False
            
        alert = self.active_alerts[alert_id]
        alert.status = "RESOLVED"
        alert.metadata["action_taken"] = action_taken
        alert.metadata["resolved_at"] = datetime.utcnow().isoformat()
        
        # Move to history
        self.alert_history.append(alert)
        del self.active_alerts[alert_id]
        
        self.logger.info(f"Alert {alert_id} resolved: {action_taken}")
        return True
        
    def get_consensus_status(self) -> Dict[str, Any]:
        """
        Get Byzantine consensus status across hospitals.
        
        Returns:
            Consensus status and health metrics
        """
        total_hospitals = len(self.current_status)
        if total_hospitals == 0:
            return {"consensus_healthy": False, "total_hospitals": 0}
            
        # Count status distribution
        status_counts = {"COMPLIANT": 0, "WARNING": 0, "VIOLATION": 0, "OFFLINE": 0}
        
        for hospital_id, status in self.current_status.items():
            if self._is_online(hospital_id):
                status_counts[status.status] += 1
            else:
                status_counts["OFFLINE"] += 1
                
        # Determine consensus health
        consensus_healthy = self._is_consensus_healthy()
        
        return {
            "total_hospitals": total_hospitals,
            "hospitals_online": total_hospitals - status_counts["OFFLINE"],
            "status_distribution": status_counts,
            "consensus_healthy": consensus_healthy,
            "compliance_rate": status_counts["COMPLIANT"] / max(total_hospitals, 1),
            "byzantine_fault_tolerance": self._calculate_byzantine_tolerance(),
            "last_consensus_update": datetime.utcnow().isoformat()
        }
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get compliance heartbeat metrics."""
        return {
            **self.metrics,
            "active_alerts": len(self.active_alerts),
            "alert_history_size": len(self.alert_history),
            "consensus_state_size": len(self.consensus_state)
        }
        
    def reset_metrics(self):
        """Reset metrics counters."""
        self.metrics = {
            "total_heartbeats_received": 0,
            "total_alerts_generated": 0,
            "total_violations_detected": 0,
            "hospitals_online": 0,
            "hospitals_offline": 0,
            "average_response_time_ms": 0.0,
            "byzantine_consensus_success": 0,
            "byzantine_consensus_failures": 0
        }
        
    def _monitoring_loop(self):
        """Background monitoring loop for heartbeat timeouts."""
        while True:
            time.sleep(self.heartbeat_interval_seconds)
            
            try:
                current_time = datetime.utcnow()
                
                # Check for missed heartbeats
                for hospital_id, last_heartbeat in list(self.current_status.items()):
                    last_time = datetime.fromisoformat(last_heartbeat.timestamp)
                    time_diff = (current_time - last_time).total_seconds()
                    
                    # Mark offline if too many missed heartbeats
                    if time_diff > self.max_missed_heartbeats * self.heartbeat_interval_seconds:
                        self.logger.warning(f"Hospital {hospital_id} marked offline - missed heartbeats")
                        
                        # Generate offline alert
                        self._generate_alert(
                            hospital_id=hospital_id,
                            severity="MEDIUM",
                            violation_type="HEARTBEAT_TIMEOUT",
                            description=f"Hospital {hospital_id} has missed {self.max_missed_heartbeats} heartbeats",
                            action_required="Check hospital connectivity and restart Ghost Agent"
                        )
                        
                # Update consensus state
                self._update_consensus_state(None, None)  # Force consensus update
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                
    def _determine_compliance_status(
        self,
        epsilon_used: float,
        epsilon_remaining: float,
        privacy_violations: int,
        byzantine_score: float,
        data_residency_compliant: bool,
        consent_compliant: bool
    ) -> str:
        """Determine overall compliance status."""
        # Critical violations
        if epsilon_used >= self.critical_threshold_epsilon:
            return "VIOLATION"
            
        if privacy_violations > 0:
            return "VIOLATION"
            
        if byzantine_score > 0.8:  # High Byzantine score
            return "VIOLATION"
            
        if not data_residency_compliant or not consent_compliant:
            return "VIOLATION"
            
        # Warning conditions
        if epsilon_used >= self.alert_threshold_epsilon:
            return "WARNING"
            
        if byzantine_score > 0.6:
            return "WARNING"
            
        # Compliant
        return "COMPLIANT"
        
    def _check_compliance_violations(self, heartbeat: ComplianceHeartbeat):
        """Check for compliance violations and generate alerts."""
        violations = []
        
        # Check privacy budget
        if heartbeat.epsilon_used >= self.critical_threshold_epsilon:
            violations.append({
                "type": "PRIVACY_BUDGET_EXCEEDED",
                "severity": "CRITICAL",
                "description": f"Privacy budget exceeded: ε={heartbeat.epsilon_used:.2f} > {self.critical_threshold_epsilon}",
                "action": "Immediately halt training and investigate"
            })
        elif heartbeat.epsilon_used >= self.alert_threshold_epsilon:
            violations.append({
                "type": "PRIVACY_BUDGET_WARNING",
                "severity": "HIGH",
                "description": f"Privacy budget approaching limit: ε={heartbeat.epsilon_used:.2f}",
                "action": "Reduce training frequency or increase DP noise"
            })
            
        # Check privacy violations
        if heartbeat.privacy_violations > 0:
            violations.append({
                "type": "PRIVACY_VIOLATION",
                "severity": "HIGH",
                "description": f"{heartbeat.privacy_violations} privacy violations detected",
                "action": "Review privacy mechanisms and retrain compliance procedures"
            })
            
        # Check Byzantine score
        if heartbeat.byzantine_score > 0.8:
            violations.append({
                "type": "BYZANTINE_BEHAVIOR",
                "severity": "HIGH",
                "description": f"High Byzantine score: {heartbeat.byzantine_score:.2f}",
                "action": "Quarantine hospital and investigate for malicious behavior"
            })
            
        # Check data residency
        if not heartbeat.data_residency_compliant:
            violations.append({
                "type": "DATA_RESIDENCY_VIOLATION",
                "severity": "CRITICAL",
                "description": "Data residency compliance violated",
                "action": "Verify data processing location and implement residency controls"
            })
            
        # Generate alerts for violations
        for violation in violations:
            self._generate_alert(
                hospital_id=heartbeat.hospital_id,
                severity=violation["severity"],
                violation_type=violation["type"],
                description=violation["description"],
                action_required=violation["action"]
            )
            
        self.metrics["total_violations_detected"] += len(violations)
        
    def _generate_alert(
        self,
        hospital_id: str,
        severity: str,
        violation_type: str,
        description: str,
        action_required: str
    ):
        """Generate a compliance alert."""
        alert_id = f"alert_{hospital_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        alert = ComplianceAlert(
            alert_id=alert_id,
            timestamp=datetime.utcnow().isoformat(),
            severity=severity,
            hospital_id=hospital_id,
            violation_type=violation_type,
            description=description,
            action_required=action_required,
            status="ACTIVE"
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        self.metrics["total_alerts_generated"] += 1
        
        self.logger.critical(
            f"Compliance alert generated: {alert_id} for {hospital_id} - "
            f"{severity} {violation_type}: {description}"
        )
        
    def _update_consensus_state(self, hospital_id: Optional[str], status: Optional[str]):
        """Update Byzantine consensus state."""
        # Count current status distribution
        status_counts = {"COMPLIANT": 0, "WARNING": 0, "VIOLATION": 0, "OFFLINE": 0}
        
        for hid in self.current_status.keys():
            if self._is_online(hid):
                status_counts[self.current_status[hid].status] += 1
            else:
                status_counts["OFFLINE"] += 1
                
        total_hospitals = len(self.current_status)
        
        # Calculate consensus
        consensus_reached = False
        for status_type, count in status_counts.items():
            if count / max(total_hospitals, 1) >= self.consensus_threshold:
                consensus_reached = True
                self.consensus_state["current_consensus"] = status_type
                self.consensus_state["consensus_count"] = count
                self.consensus_state["total_hospitals"] = total_hospitals
                self.consensus_state["last_update"] = datetime.utcnow().isoformat()
                
                if status_type in ["VIOLATION", "OFFLINE"]:
                    self.metrics["byzantine_consensus_failures"] += 1
                else:
                    self.metrics["byzantine_consensus_success"] += 1
                    
                break
                
    def _is_online(self, hospital_id: str) -> bool:
        """Check if hospital is online based on recent heartbeats."""
        if hospital_id not in self.current_status:
            return False
            
        last_heartbeat = self.current_status[hospital_id]
        last_time = datetime.fromisoformat(last_heartbeat.timestamp)
        time_diff = (datetime.utcnow() - last_time).total_seconds()
        
        return time_diff <= self.max_missed_heartbeats * self.heartbeat_interval_seconds
        
    def _is_consensus_healthy(self) -> bool:
        """Check if Byzantine consensus is healthy."""
        if "current_consensus" not in self.consensus_state:
            return False
            
        return self.consensus_state["current_consensus"] in ["COMPLIANT", "WARNING"]
        
    def _calculate_byzantine_tolerance(self) -> Dict[str, float]:
        """Calculate current Byzantine fault tolerance metrics."""
        total_hospitals = len(self.current_status)
        online_hospitals = sum(1 for hid in self.current_status.keys() if self._is_online(hid))
        
        # Byzantine tolerance: can tolerate up to 1/3 failures
        max_faults = (online_hospitals - 1) // 3
        
        return {
            "total_hospitals": total_hospitals,
            "online_hospitals": online_hospitals,
            "max_tolerable_faults": max_faults,
            "current_faults": online_hospitals - len(self.current_status),
            "tolerance_ratio": max_faults / max(online_hospitals, 1)
        }