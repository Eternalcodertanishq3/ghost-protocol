"""
Module: algorithms/dp_mechanisms/gaussian.py
DPDP §: 9(4) - Privacy preservation through differential privacy
Description: Gaussian mechanism for (ε,δ)-differential privacy
Test: pytest tests/test_dp.py::test_gaussian_dp
"""

import torch
import numpy as np
from typing import Optional, Tuple
from scipy.stats import norm


class GaussianDP:
    """
    Gaussian mechanism for (ε,δ)-differential privacy.
    
    Adds Gaussian noise to gradients/privatized values:
    noise ~ N(0, σ²) where σ = s * √(2*ln(1.25/δ)) / ε
    
    s = L2 sensitivity of the query (typically gradient clipping norm)
    """
    
    def __init__(
        self,
        epsilon: float = 1.23,
        delta: float = 1e-5,
        noise_multiplier: Optional[float] = None
    ):
        """
        Initialize Gaussian DP mechanism.
        
        Args:
            epsilon: Privacy budget (ε)
            delta: Failure probability (δ)
            noise_multiplier: Direct noise multiplier (optional)
        """
        self.epsilon = epsilon
        self.delta = delta
        
        if noise_multiplier is not None:
            self.noise_multiplier = noise_multiplier
        else:
            # Compute noise multiplier from ε,δ
            self.noise_multiplier = self._compute_noise_multiplier(epsilon, delta)
            
    def _compute_noise_multiplier(
        self,
        epsilon: float,
        delta: float
    ) -> float:
        """Compute noise multiplier for given ε,δ."""
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        if delta <= 0 or delta >= 1:
            raise ValueError("Delta must be in (0,1)")
            
        # Standard Gaussian mechanism calibration
        # σ = √(2*ln(1.25/δ)) / ε
        sigma = np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        return sigma
    
    def add_noise(
        self,
        tensor: torch.Tensor,
        sensitivity: float = 1.0
    ) -> torch.Tensor:
        """
        Add Gaussian noise to tensor.
        
        Args:
            tensor: Input tensor
            sensitivity: L2 sensitivity (typically gradient clipping norm)
            
        Returns:
            Noisy tensor with same shape as input
        """
        if sensitivity <= 0:
            raise ValueError("Sensitivity must be positive")
            
        noise_scale = self.noise_multiplier * sensitivity
        
        # Generate Gaussian noise
        noise = torch.randn_like(tensor) * noise_scale
        
        # Add noise to original tensor
        noisy_tensor = tensor + noise
        
        return noisy_tensor
    
    def privatize_gradients(
        self,
        gradients: Dict[str, torch.Tensor],
        clipping_norm: float = 1.0
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        """
        Privatize gradients using Gaussian mechanism.
        
        Args:
            gradients: Dictionary of gradient tensors
            clipping_norm: Gradient clipping norm
            
        Returns:
            Tuple of (privatized gradients, noise scale used)
        """
        privatized_grads = {}
        
        for name, grad in gradients.items():
            # Add Gaussian noise to gradients
            noisy_grad = self.add_noise(grad, clipping_norm)
            privatized_grads[name] = noisy_grad
            
        noise_scale = self.noise_multiplier * clipping_norm
        
        return privatized_grads, noise_scale
    
    def compute_privacy_spent(
        self,
        num_steps: int,
        sampling_rate: float = 1.0,
        mechanism: str = "rdp"
    ) -> Tuple[float, float]:
        """
        Compute privacy spent after multiple steps.
        
        Args:
            num_steps: Number of optimization steps
            sampling_rate: Subsampling rate (q = batch_size/dataset_size)
            mechanism: Privacy accounting mechanism ("rdp" or "moments")
            
        Returns:
            Tuple of (epsilon, delta) privacy spent
        """
        if mechanism == "rdp":
            return self._compute_rdp_privacy(num_steps, sampling_rate)
        else:
            # Simple composition (conservative)
            epsilon_total = num_steps * self.epsilon
            delta_total = num_steps * self.delta
            return epsilon_total, delta_total
    
    def _compute_rdp_privacy(
        self,
        num_steps: int,
        sampling_rate: float
    ) -> Tuple[float, float]:
        """Compute privacy using Rényi DP composition."""
        # Rényi DP parameters
        alpha = 10  # Rényi order
        
        # RDP for Gaussian mechanism
        # ε_RDP(α) = α / (2 * σ²)
        rdp_epsilon = alpha / (2 * self.noise_multiplier ** 2)
        
        # Composition over steps
        composed_rdp = num_steps * rdp_epsilon
        
        # Convert back to (ε,δ)-DP
        # ε = composed_rdp + log(1/δ)/(α-1)
        epsilon = composed_rdp + np.log(1 / self.delta) / (alpha - 1)
        
        return epsilon, self.delta
    
    def calibrate_noise(
        self,
        target_epsilon: float,
        target_delta: float,
        num_steps: int,
        sampling_rate: float = 1.0
    ) -> float:
        """
        Calibrate noise multiplier for target privacy budget.
        
        Args:
            target_epsilon: Target epsilon
            target_delta: Target delta
            num_steps: Number of steps
            sampling_rate: Subsampling rate
            
        Returns:
            Required noise multiplier
        """
        # Binary search for optimal noise multiplier
        low = 0.1
        high = 10.0
        
        for _ in range(50):  # Binary search iterations
            mid = (low + high) / 2
            
            # Create temporary mechanism
            temp_mechanism = GaussianDP(
                epsilon=target_epsilon,
                delta=target_delta,
                noise_multiplier=mid
            )
            
            # Compute privacy spent
            spent_epsilon, _ = temp_mechanism.compute_privacy_spent(
                num_steps, sampling_rate
            )
            
            if spent_epsilon <= target_epsilon:
                # Privacy budget sufficient, try less noise
                high = mid
            else:
                # Need more noise
                low = mid
                
        return (low + high) / 2
    
    def get_noise_statistics(self) -> Dict[str, float]:
        """Get noise mechanism statistics."""
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "noise_multiplier": self.noise_multiplier,
            "mechanism": "Gaussian"
        }