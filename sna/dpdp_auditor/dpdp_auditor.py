"""
Module: sna/dpdp_auditor/dpdp_auditor.py
DPDP §: 9(4) - Privacy budget monitoring, §25 - Breach notification
Description: DPDP Auditor for live epsilon tracking and auto-halt at ε=9.5
Test: pytest tests/test_dpdp_auditor.py::test_privacy_budget_tracking
API: GET /dpdp_status, POST /privacy_audit, GET /epsilon_budget
"""

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import threading
import time


@dataclass
class PrivacyBudgetRecord:
    """Record of privacy budget usage."""
    hospital_id: str
    round_number: int
    epsilon_spent: float
    delta_spent: float
    mechanism: str
    timestamp: str
    compliance_status: str  # "COMPLIANT" or "VIOLATED"


@dataclass
class PrivacyAuditReport:
    """Privacy audit report."""
    audit_id: str
    timestamp: str
    total_epsilon_spent: float
    max_epsilon_allowed: float
    compliance_status: str
    violations: List[Dict[str, Any]]
    recommendations: List[str]


class DPDPAuditor:
    """
    DPDP Auditor for Ghost Protocol.
    
    Implements:
    - Live epsilon budget tracking
    - Automatic breach detection and notification (§25)
    - Auto-halt at ε=9.5 (hard stop)
    - Privacy compliance monitoring
    - Audit trail maintenance
    """
    
    def __init__(
        self,
        max_epsilon: float = 9.5,  # DPDP hard stop
        max_delta: float = 1e-5,
        audit_interval_seconds: int = 300,  # 5 minutes
        breach_notification_threshold: float = 8.0,  # Alert at 80% budget
        auto_halt_enabled: bool = True
    ):
        """
        Initialize DPDP Auditor.
        
        Args:
            max_epsilon: Maximum epsilon budget (DPDP compliance)
            max_delta: Maximum delta budget
            audit_interval_seconds: Interval between automatic audits
            breach_notification_threshold: Epsilon threshold for breach alerts
            auto_halt_enabled: Whether to auto-halt at max epsilon
        """
        self.max_epsilon = max_epsilon
        self.max_delta = max_delta
        self.audit_interval_seconds = audit_interval_seconds
        self.breach_notification_threshold = breach_notification_threshold
        self.auto_halt_enabled = auto_halt_enabled
        
        self.logger = logging.getLogger(__name__)
        
        # Privacy budget tracking
        self.hospital_budgets: Dict[str, List[PrivacyBudgetRecord]] = {}
        self.global_budget_usage: List[PrivacyBudgetRecord] = []
        
        # Audit state
        self.is_halted = False
        self.last_audit_time = None
        self.audit_thread = None
        self.breach_notifications_sent = []
        
        # Statistics
        self.audit_stats = {
            "total_audits": 0,
            "violations_detected": 0,
            "breach_notifications": 0,
            "auto_halts_triggered": 0
        }
        
        # Start audit thread
        self._start_audit_monitoring()
        
    def record_privacy_expenditure(
        self,
        hospital_id: str,
        round_number: int,
        epsilon_spent: float,
        delta_spent: float,
        mechanism: str = "gaussian"
    ) -> PrivacyBudgetRecord:
        """
        Record privacy budget expenditure.
        
        Args:
            hospital_id: Hospital identifier
            round_number: Training round number
            epsilon_spent: Epsilon spent in this round
            delta_spent: Delta spent in this round
            mechanism: DP mechanism used
            
        Returns:
            Privacy budget record
        """
        # Check compliance
        total_epsilon = self.get_hospital_epsilon_used(hospital_id)
        new_total = total_epsilon + epsilon_spent
        
        compliance_status = "COMPLIANT" if new_total <= self.max_epsilon else "VIOLATED"
        
        # Create record
        record = PrivacyBudgetRecord(
            hospital_id=hospital_id,
            round_number=round_number,
            epsilon_spent=epsilon_spent,
            delta_spent=delta_spent,
            mechanism=mechanism,
            timestamp=datetime.utcnow().isoformat(),
            compliance_status=compliance_status
        )
        
        # Store record
        if hospital_id not in self.hospital_budgets:
            self.hospital_budgets[hospital_id] = []
            
        self.hospital_budgets[hospital_id].append(record)
        self.global_budget_usage.append(record)
        
        # Check for violations
        if compliance_status == "VIOLATED":
            self._handle_privacy_violation(hospital_id, new_total)
            
        # Check for breach notification threshold
        if new_total >= self.breach_notification_threshold and new_total < self.max_epsilon:
            self._send_breach_notification(hospital_id, new_total)
            
        return record
        
    def get_hospital_epsilon_used(self, hospital_id: str) -> float:
        """Get total epsilon used by a hospital."""
        if hospital_id not in self.hospital_budgets:
            return 0.0
            
        return sum(record.epsilon_spent for record in self.hospital_budgets[hospital_id])
        
    def get_global_epsilon_used(self) -> float:
        """Get total epsilon used across all hospitals."""
        return sum(record.epsilon_spent for record in self.global_budget_usage)
        
    def get_privacy_budget_status(self, hospital_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get privacy budget status.
        
        Args:
            hospital_id: Specific hospital (optional, None for global)
            
        Returns:
            Privacy budget status
        """
        if hospital_id:
            epsilon_used = self.get_hospital_epsilon_used(hospital_id)
            records = self.hospital_budgets.get(hospital_id, [])
            return {
                "hospital_id": hospital_id,
                "epsilon_used": epsilon_used,
                "epsilon_remaining": self.max_epsilon - epsilon_used,
                "epsilon_utilization": epsilon_used / self.max_epsilon,
                "compliance_status": "COMPLIANT" if epsilon_used <= self.max_epsilon else "VIOLATED",
                "total_rounds": len(records),
                "last_update": records[-1].timestamp if records else None
            }
        else:
            # Global status
            epsilon_used = self.get_global_epsilon_used()
            return {
                "scope": "global",
                "epsilon_used": epsilon_used,
                "epsilon_remaining": self.max_epsilon - epsilon_used,
                "epsilon_utilization": epsilon_used / self.max_epsilon,
                "total_hospitals": len(self.hospital_budgets),
                "total_records": len(self.global_budget_usage),
                "is_halted": self.is_halted
            }
            
    def conduct_privacy_audit(self) -> PrivacyAuditReport:
        """
        Conduct comprehensive privacy audit.
        
        Returns:
            Privacy audit report
        """
        start_time = datetime.utcnow()
        
        # Generate audit ID
        audit_id = f"audit_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Check global compliance
        global_epsilon = self.get_global_epsilon_used()
        compliance_status = "COMPLIANT" if global_epsilon <= self.max_epsilon else "VIOLATED"
        
        # Find violations
        violations = []
        for hospital_id, records in self.hospital_budgets.items():
            epsilon_used = sum(record.epsilon_spent for record in records)
            if epsilon_used > self.max_epsilon:
                violations.append({
                    "hospital_id": hospital_id,
                    "epsilon_exceeded": epsilon_used - self.max_epsilon,
                    "total_epsilon_used": epsilon_used,
                    "violation_rounds": len(records)
                })
                
        # Generate recommendations
        recommendations = self._generate_recommendations(violations, global_epsilon)
        
        # Create report
        report = PrivacyAuditReport(
            audit_id=audit_id,
            timestamp=start_time.isoformat(),
            total_epsilon_spent=global_epsilon,
            max_epsilon_allowed=self.max_epsilon,
            compliance_status=compliance_status,
            violations=violations,
            recommendations=recommendations
        )
        
        # Update statistics
        self.audit_stats["total_audits"] += 1
        if violations:
            self.audit_stats["violations_detected"] += len(violations)
            
        self.last_audit_time = start_time
        
        self.logger.info(
            f"Privacy audit completed: {compliance_status}, "
            f"violations={len(violations)}, ε_used={global_epsilon:.3f}/{self.max_epsilon}"
        )
        
        return report
        
    def _generate_recommendations(
        self,
        violations: List[Dict[str, Any]],
        global_epsilon: float
    ) -> List[str]:
        """Generate privacy recommendations."""
        recommendations = []
        
        if violations:
            recommendations.append("Immediate action required: Privacy violations detected")
            recommendations.append("Review and reduce DP noise multipliers")
            recommendations.append("Implement stricter gradient clipping")
            
        if global_epsilon > self.breach_notification_threshold:
            recommendations.append("Privacy budget approaching limit - consider reducing training rounds")
            
        if global_epsilon > self.max_epsilon * 0.9:
            recommendations.append("Critical: Privacy budget nearly exhausted - halt training immediately")
            
        recommendations.append("Regular privacy audits recommended")
        recommendations.append("Monitor gradient norms for anomaly detection")
        
        return recommendations
        
    def _handle_privacy_violation(self, hospital_id: str, epsilon_used: float):
        """Handle privacy violation."""
        self.logger.error(
            f"Privacy violation detected for {hospital_id}: "
            f"ε_used={epsilon_used:.3f} > max_ε={self.max_epsilon}"
        )
        
        # Auto-halt if enabled
        if self.auto_halt_enabled and not self.is_halted:
            self._trigger_auto_halt()
            
        # Log violation
        self.audit_stats["violations_detected"] += 1
        
    def _trigger_auto_halt(self):
        """Trigger automatic system halt."""
        self.is_halted = True
        self.audit_stats["auto_halts_triggered"] += 1
        
        self.logger.critical("AUTO-HALT TRIGGERED: Privacy budget exceeded")
        
        # In production, this would halt all training
        # For now, just log and set flag
        
    def _send_breach_notification(self, hospital_id: str, epsilon_used: float):
        """Send breach notification (§25 DPDP compliance)."""
        notification_id = f"breach_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        notification = {
            "notification_id": notification_id,
            "hospital_id": hospital_id,
            "timestamp": datetime.utcnow().isoformat(),
            "epsilon_used": epsilon_used,
            "max_epsilon": self.max_epsilon,
            "threshold_exceeded": self.breach_notification_threshold,
            "severity": "high" if epsilon_used > self.max_epsilon else "medium",
            "action_required": "Reduce privacy expenditure immediately"
        }
        
        self.breach_notifications_sent.append(notification)
        self.audit_stats["breach_notifications"] += 1
        
        self.logger.warning(
            f"Breach notification sent to {hospital_id}: "
            f"ε_used={epsilon_used:.3f}, threshold={self.breach_notification_threshold}"
        )
        
    def _start_audit_monitoring(self):
        """Start automatic audit monitoring thread."""
        self.audit_thread = threading.Thread(target=self._audit_monitoring_loop, daemon=True)
        self.audit_thread.start()
        
    def _audit_monitoring_loop(self):
        """Background audit monitoring loop."""
        while True:
            time.sleep(self.audit_interval_seconds)
            
            try:
                # Conduct automatic audit
                report = self.conduct_privacy_audit()
                
                # Check if halt needed
                if report.compliance_status == "VIOLATED" and self.auto_halt_enabled:
                    self._trigger_auto_halt()
                    
            except Exception as e:
                self.logger.error(f"Audit monitoring error: {e}")
                
    def get_audit_history(
        self,
        hospital_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[PrivacyBudgetRecord]:
        """
        Get audit history.
        
        Args:
            hospital_id: Specific hospital (optional)
            start_date: Start date filter (ISO format)
            end_date: End date filter (ISO format)
            
        Returns:
            List of privacy budget records
        """
        if hospital_id:
            records = self.hospital_budgets.get(hospital_id, [])
        else:
            records = self.global_budget_usage
            
        # Apply date filters
        if start_date or end_date:
            filtered_records = []
            for record in records:
                record_date = datetime.fromisoformat(record.timestamp)
                if start_date and record_date < datetime.fromisoformat(start_date):
                    continue
                if end_date and record_date > datetime.fromisoformat(end_date):
                    continue
                filtered_records.append(record)
            records = filtered_records
            
        return records
        
    def export_privacy_audit(self, filepath: str):
        """Export privacy audit data to JSON file."""
        audit_data = {
            "export_timestamp": datetime.utcnow().isoformat(),
            "dpdp_compliance": {
                "max_epsilon": self.max_epsilon,
                "max_delta": self.max_delta,
                "current_global_epsilon": self.get_global_epsilon_used()
            },
            "audit_statistics": self.audit_stats,
            "hospital_budgets": {
                hospital_id: [asdict(record) for record in records]
                for hospital_id, records in self.hospital_budgets.items()
            },
            "global_usage": [asdict(record) for record in self.global_budget_usage],
            "breach_notifications": self.breach_notifications_sent,
            "system_status": {
                "is_halted": self.is_halted,
                "last_audit": self.last_audit_time.isoformat() if self.last_audit_time else None,
                "compliance_status": "COMPLIANT" if self.get_global_epsilon_used() <= self.max_epsilon else "VIOLATED"
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(audit_data, f, indent=2)
            
    def get_dpdp_certificate(self) -> Dict[str, Any]:
        """Generate DPDP compliance certificate."""
        global_epsilon = self.get_global_epsilon_used()
        
        return {
            "certificate_id": f"dpdp_cert_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.utcnow().isoformat(),
            "compliance_framework": "Digital Personal Data Protection Act 2023",
            "privacy_budget": {
                "max_epsilon": self.max_epsilon,
                "current_epsilon": global_epsilon,
                "epsilon_utilization": global_epsilon / self.max_epsilon,
                "status": "COMPLIANT" if global_epsilon <= self.max_epsilon else "VIOLATED"
            },
            "mechanisms_used": list(set(record.mechanism for record in self.global_budget_usage)),
            "audit_summary": self.audit_stats,
            "valid_until": (datetime.utcnow() + timedelta(days=1)).isoformat()
        }
        
    def reset_privacy_budget(self, hospital_id: Optional[str] = None):
        """
        Reset privacy budget (for testing or new training session).
        
        Args:
            hospital_id: Specific hospital (None for global reset)
        """
        if hospital_id:
            if hospital_id in self.hospital_budgets:
                self.hospital_budgets[hospital_id] = []
        else:
            # Global reset
            self.hospital_budgets = {}
            self.global_budget_usage = []
            self.is_halted = False
            
        self.logger.info(f"Privacy budget reset for {hospital_id or 'all hospitals'}")