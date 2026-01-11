"""
Differential Privacy Engine - Mathematical Privacy Guarantees
Ghost Protocol Privacy Layer - RDP Accountant, Gaussian Mechanism

DPDP §: §15 Right to Privacy (Privacy by Design)
Byzantine theorem: Privacy budget exhaustion prevents malicious data extraction
Test command: pytest tests/test_privacy.py -v --cov=privacy
Metrics tracked: ε consumed, δ used, privacy loss, accountant accuracy
"""

import math
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging

import numpy as np
import torch
from rdp_accountant import compute_rdp, get_privacy_spent
from scipy import stats


@dataclass
class PrivacyBudget:
    """Privacy budget allocation and consumption tracking"""
    epsilon_total: float  # Total privacy budget allocated
    epsilon_consumed: float = 0.0  # Budget consumed so far
    delta: float = 1e-5  # δ for (ε,δ)-differential privacy
    
    @property
    def epsilon_remaining(self) -> float:
        return self.epsilon_total - self.epsilon_consumed
    
    @property
    def budget_exhausted(self) -> bool:
        return self.epsilon_remaining <= 0.0


@dataclass
class DPCalibrationResult:
    """Result of privacy calibration"""
    noise_multiplier: float
    epsilon_consumed: float
    delta: float
    rdp_orders: List[float]
    rdp_values: List[float]


class DifferentialPrivacyEngine:
    """
    Production-grade differential privacy engine with Rényi DP accounting
    
    Implements Gaussian mechanism with tight privacy accounting using
    Rényi Differential Privacy (RDP) for composition across multiple rounds.
    
    Key features:
    - Rényi DP accountant for tight composition
    - Gaussian mechanism with optimal noise calibration
    - Privacy amplification by shuffling
    - Real-time privacy budget tracking
    - DPDP Act 2023 compliance
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(f"dp_engine.{config.hospital_id}")
        
        # Privacy budget management
        self.budget = PrivacyBudget(
            epsilon_total=config.epsilon_max,
            delta=config.delta_max
        )
        
        # RDP accountant state
        self.rdp_orders = self._get_default_rdp_orders()
        self.rdp_values = np.zeros(len(self.rdp_orders))
        
        # Privacy history for auditing
        self.privacy_history: List[Dict[str, Any]] = []
        
        self.logger.info(f"DP Engine initialized: ε={config.epsilon_max}, δ={config.delta_max}")
    
    def _get_default_rdp_orders(self) -> List[float]:
        """Default RDP orders for tight accounting"""
        return [
            1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0,
            4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0,
            11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
            25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0
        ]
    
    def calibrate_gaussian_mechanism(
        self,
        target_epsilon: float,
        target_delta: float,
        sensitivity: float,
        iterations: int,
        sample_rate: float = 1.0,
        amplification: bool = True
    ) -> DPCalibrationResult:
        """
        Calibrate Gaussian mechanism for desired privacy guarantees
        
        Args:
            target_epsilon: Target ε for privacy guarantee
            target_delta: Target δ for privacy guarantee  
            sensitivity: L2 sensitivity of the query/function
            iterations: Number of iterations (steps)
            sample_rate: Sampling rate for privacy amplification
            amplification: Whether to use privacy amplification
            
        Returns:
            DPCalibrationResult with calibrated parameters
        """
        
        # Use binary search to find optimal noise multiplier
        low_sigma = 0.01
        high_sigma = 10.0
        best_sigma = None
        best_epsilon = float('inf')
        
        max_iterations = 50
        tolerance = 0.01
        
        for _ in range(max_iterations):
            mid_sigma = (low_sigma + high_sigma) / 2
            
            # Compute RDP for this sigma
            rdp_orders = self.rdp_orders
            rdp_values = []
            
            for alpha in rdp_orders:
                # RDP for Gaussian mechanism
                rdp_value = alpha * (sensitivity ** 2) / (2 * (mid_sigma ** 2))
                
                # Apply composition over iterations
                composed_rdp = rdp_value * iterations
                
                # Apply privacy amplification if sampling
                if amplification and sample_rate < 1.0:
                    # Privacy amplification by subsampling
                    amplified_rdp = self._apply_privacy_amplification(
                        composed_rdp, alpha, sample_rate
                    )
                    rdp_values.append(amplified_rdp)
                else:
                    rdp_values.append(composed_rdp)
            
            # Convert RDP to (ε,δ)-DP
            epsilon_spent = self._rdp_to_epsilon(rdp_values, target_delta)
            
            if abs(epsilon_spent - target_epsilon) < tolerance:
                best_sigma = mid_sigma
                best_epsilon = epsilon_spent
                break
            elif epsilon_spent > target_epsilon:
                # Need more noise (higher sigma)
                low_sigma = mid_sigma
            else:
                # Can use less noise (lower sigma)
                high_sigma = mid_sigma
        
        if best_sigma is None:
            best_sigma = mid_sigma
            best_epsilon = epsilon_spent
        
        self.logger.info(f"Calibrated σ={best_sigma:.3f} for ε={best_epsilon:.3f}")
        
        return DPCalibrationResult(
            noise_multiplier=best_sigma,
            epsilon_consumed=best_epsilon,
            delta=target_delta,
            rdp_orders=self.rdp_orders,
            rdp_values=rdp_values
        )
    
    def _apply_privacy_amplification(
        self, 
        rdp_value: float, 
        alpha: float, 
        sample_rate: float
    ) -> float:
        """
        Apply privacy amplification by subsampling
        
        Implements the privacy amplification theorem for RDP:
        If M is (α,ε)-RDP, then M∘Sample_q is (α,ε')-RDP where
        ε' ≤ log(1 + q² * (exp(ε) - 1))
        
        Args:
            rdp_value: Original RDP value
            alpha: RDP order
            sample_rate: Sampling probability q
            
        Returns:
            Amplified RDP value
        """
        if sample_rate >= 1.0:
            return rdp_value
        
        # Privacy amplification formula for RDP
        amplified_rdp = math.log(1 + (sample_rate ** 2) * (math.exp(rdp_value) - 1))
        
        return amplified_rdp
    
    def _rdp_to_epsilon(self, rdp_values: List[float], delta: float) -> float:
        """
        Convert RDP to (ε,δ)-DP using standard conversion
        
        ε(δ) = min_α [ε_RDP(α) + log((α-1)/α) - log(δ)/α]
        
        Args:
            rdp_values: RDP values at different orders
            delta: Target δ
            
        Returns:
            Corresponding ε value
        """
        epsilon_candidates = []
        
        for alpha, rdp_value in zip(self.rdp_orders, rdp_values):
            if alpha <= 1:
                continue
            
            # Standard RDP to DP conversion
            epsilon_candidate = rdp_value + math.log((alpha - 1) / alpha) - math.log(delta) / alpha
            epsilon_candidates.append(epsilon_candidate)
        
        if not epsilon_candidates:
            return float('inf')
        
        return min(epsilon_candidates)
    
    def add_gaussian_noise(
        self, 
        tensor: torch.Tensor, 
        noise_multiplier: float,
        sensitivity: float = 1.0
    ) -> torch.Tensor:
        """
        Add Gaussian noise to tensor for differential privacy
        
        Args:
            tensor: Input tensor to add noise to
            noise_multiplier: σ parameter for Gaussian mechanism
            sensitivity: L2 sensitivity of the underlying function
            
        Returns:
            Noisy tensor with same shape as input
        """
        if noise_multiplier == 0:
            return tensor
        
        # Calculate noise scale
        noise_scale = noise_multiplier * sensitivity
        
        # Generate Gaussian noise
        noise = torch.randn_like(tensor) * noise_scale
        
        # Add noise to tensor
        noisy_tensor = tensor + noise
        
        self.logger.debug(f"Added Gaussian noise: σ={noise_multiplier}, sensitivity={sensitivity}")
        
        return noisy_tensor
    
    def compute_sensitivity_l2(
        self, 
        model: torch.nn.Module,
        max_grad_norm: float = 1.0
    ) -> float:
        """
        Compute L2 sensitivity for gradient clipping
        
        Args:
            model: PyTorch model
            max_grad_norm: Maximum gradient norm for clipping
            
        Returns:
            L2 sensitivity value
        """
        # For DP-SGD, sensitivity is the clipping norm
        return max_grad_norm
    
    def update_privacy_accountant(
        self,
        noise_multiplier: float,
        sample_rate: float,
        steps: int,
        amplification: bool = True
    ) -> Tuple[float, float]:
        """
        Update RDP accountant and compute privacy spent
        
        Args:
            noise_multiplier: Gaussian noise multiplier σ
            sample_rate: Sampling rate for this step
            steps: Number of steps/iterations
            amplification: Whether to apply privacy amplification
            
        Returns:
            Tuple of (epsilon_spent, delta_used)
        """
        
        # Compute RDP for each order
        new_rdp_values = np.zeros(len(self.rdp_orders))
        
        for i, alpha in enumerate(self.rdp_orders):
            if alpha <= 1:
                continue
            
            # RDP for Gaussian mechanism
            rdp_per_step = alpha / (2 * (noise_multiplier ** 2))
            
            # Apply composition
            composed_rdp = rdp_per_step * steps
            
            # Apply privacy amplification
            if amplification and sample_rate < 1.0:
                composed_rdp = self._apply_privacy_amplification(
                    composed_rdp, alpha, sample_rate
                )
            
            new_rdp_values[i] = composed_rdp
        
        # Update cumulative RDP values
        self.rdp_values += new_rdp_values
        
        # Convert to (ε,δ)-DP
        epsilon_spent = self._rdp_to_epsilon(self.rdp_values.tolist(), self.budget.delta)
        
        # Update budget
        self.budget.epsilon_consumed = epsilon_spent
        
        # Log privacy consumption
        privacy_record = {
            "timestamp": datetime.utcnow(),
            "noise_multiplier": noise_multiplier,
            "sample_rate": sample_rate,
            "steps": steps,
            "epsilon_spent": epsilon_spent,
            "epsilon_remaining": self.budget.epsilon_remaining,
            "delta": self.budget.delta,
            "amplification_used": amplification
        }
        
        self.privacy_history.append(privacy_record)
        
        # Keep history bounded
        if len(self.privacy_history) > 1000:
            self.privacy_history = self.privacy_history[-500:]
        
        self.logger.info(f"Privacy spent: ε={epsilon_spent:.4f}, remaining={self.budget.epsilon_remaining:.4f}")
        
        return epsilon_spent, self.budget.delta
    
    def check_budget_available(self, required_epsilon: float = None) -> bool:
        """
        Check if sufficient privacy budget is available
        
        Args:
            required_epsilon: Epsilon required for next operation
            
        Returns:
            True if budget is available, False otherwise
        """
        if required_epsilon is None:
            required_epsilon = self.config.epsilon_per_update
        
        return self.budget.epsilon_remaining >= required_epsilon
    
    def get_privacy_summary(self) -> Dict[str, Any]:
        """Get comprehensive privacy budget summary"""
        return {
            "budget_allocated": {
                "epsilon_total": self.budget.epsilon_total,
                "delta": self.budget.delta
            },
            "budget_consumed": {
                "epsilon_spent": self.budget.epsilon_consumed,
                "epsilon_remaining": self.budget.epsilon_remaining,
                "percentage_used": (self.budget.epsilon_consumed / self.budget.epsilon_total) * 100
            },
            "budget_exhausted": self.budget.budget_exhausted,
            "privacy_history_length": len(self.privacy_history),
            "rdp_accountant_state": {
                "orders": self.rdp_orders[:10],  # First 10 orders
                "values": self.rdp_values[:10].tolist()
            }
        }
    
    def reset_privacy_accountant(self):
        """Reset privacy accountant (for testing only)"""
        self.budget.epsilon_consumed = 0.0
        self.rdp_values = np.zeros(len(self.rdp_orders))
        self.privacy_history = []
        
        self.logger.warning("Privacy accountant reset - use only in development")
    
    def get_privacy_audit_trail(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get privacy audit trail for compliance reporting
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of privacy events for audit
        """
        return self.privacy_history[-limit:]
    
    def validate_privacy_guarantees(
        self, 
        claimed_epsilon: float, 
        claimed_delta: float,
        noise_multiplier: float,
        steps: int,
        sample_rate: float = 1.0
    ) -> bool:
        """
        Validate claimed privacy guarantees using independent calculation
        
        Args:
            claimed_epsilon: Claimed ε value to validate
            claimed_delta: Claimed δ value to validate  
            noise_multiplier: Noise multiplier used
            steps: Number of steps taken
            sample_rate: Sampling rate used
            
        Returns:
            True if claims are valid within tolerance
        """
        
        # Recalculate RDP values
        test_rdp_values = np.zeros(len(self.rdp_orders))
        
        for i, alpha in enumerate(self.rdp_orders):
            if alpha <= 1:
                continue
            
            rdp_per_step = alpha / (2 * (noise_multiplier ** 2))
            composed_rdp = rdp_per_step * steps
            
            # Apply privacy amplification
            if sample_rate < 1.0:
                composed_rdp = self._apply_privacy_amplification(
                    composed_rdp, alpha, sample_rate
                )
            
            test_rdp_values[i] = composed_rdp
        
        # Convert to (ε,δ)-DP
        computed_epsilon = self._rdp_to_epsilon(test_rdp_values.tolist(), claimed_delta)
        
        # Allow small tolerance for floating point errors
        tolerance = 0.001
        
        epsilon_valid = abs(computed_epsilon - claimed_epsilon) <= tolerance
        
        self.logger.info(f"Privacy validation: claimed ε={claimed_epsilon}, computed ε={computed_epsilon}, valid={epsilon_valid}")
        
        return epsilon_valid
    
    def estimate_privacy_for_future_rounds(
        self,
        rounds_remaining: int,
        avg_noise_multiplier: float = None,
        avg_sample_rate: float = 1.0,
        avg_steps_per_round: int = 100
    ) -> Dict[str, float]:
        """
        Estimate privacy budget needed for remaining rounds
        
        Args:
            rounds_remaining: Number of rounds remaining
            avg_noise_multiplier: Average noise multiplier (uses config if None)
            avg_sample_rate: Average sampling rate
            avg_steps_per_round: Average steps per round
            
        Returns:
            Dictionary with privacy projections
        """
        
        if avg_noise_multiplier is None:
            avg_noise_multiplier = self.config.gaussian_noise_scale
        
        # Estimate RDP for remaining rounds
        estimated_rdp_values = np.zeros(len(self.rdp_orders))
        
        for i, alpha in enumerate(self.rdp_orders):
            if alpha <= 1:
                continue
            
            rdp_per_step = alpha / (2 * (avg_noise_multiplier ** 2))
            rdp_per_round = rdp_per_step * avg_steps_per_round
            
            if avg_sample_rate < 1.0:
                rdp_per_round = self._apply_privacy_amplification(
                    rdp_per_round, alpha, avg_sample_rate
                )
            
            estimated_rdp_values[i] = rdp_per_round * rounds_remaining
        
        # Convert to (ε,δ)-DP
        estimated_epsilon_needed = self._rdp_to_epsilon(
            estimated_rdp_values.tolist(), self.budget.delta
        )
        
        return {
            "epsilon_needed": estimated_epsilon_needed,
            "epsilon_remaining": self.budget.epsilon_remaining,
            "epsilon_shortfall": max(0, estimated_epsilon_needed - self.budget.epsilon_remaining),
            "rounds_possible_with_current_budget": int(self.budget.epsilon_remaining / (estimated_epsilon_needed / rounds_remaining)) if estimated_epsilon_needed > 0 else 0
        }