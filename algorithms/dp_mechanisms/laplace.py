"""
Module: algorithms/dp_mechanisms/laplace.py
DPDP §: 9(4) - Privacy preservation through differential privacy
Description: Laplace mechanism for ε-differential privacy
Test: pytest tests/test_dp.py::test_laplace_dp
"""

import torch
import numpy as np
from typing import Optional


class LaplaceDP:
    """
    Laplace mechanism for ε-differential privacy (pure DP).
    
    Adds Laplace noise to gradients/privatized values:
    noise ~ Lap(0, b) where b = s / ε
    
    s = L1 sensitivity of the query
    Provides stronger privacy guarantees (δ = 0) but potentially more noise.
    """
    
    def __init__(
        self,
        epsilon: float = 0.8,
        noise_multiplier: Optional[float] = None
    ):
        """
        Initialize Laplace DP mechanism.
        
        Args:
            epsilon: Privacy budget (ε)
            noise_multiplier: Direct noise multiplier (optional)
        """
        self.epsilon = epsilon
        
        if noise_multiplier is not None:
            self.noise_multiplier = noise_multiplier
        else:
            # For Laplace: b = 1/ε (assuming sensitivity = 1)
            self.noise_multiplier = 1.0 / epsilon
            
    def add_noise(
        self,
        tensor: torch.Tensor,
        sensitivity: float = 1.0
    ) -> torch.Tensor:
        """
        Add Laplace noise to tensor.
        
        Args:
            tensor: Input tensor
            sensitivity: L1 sensitivity
            
        Returns:
            Noisy tensor with same shape as input
        """
        if sensitivity <= 0:
            raise ValueError("Sensitivity must be positive")
            
        noise_scale = self.noise_multiplier * sensitivity
        
        # Generate Laplace noise using exponential distribution
        # Laplace(0,b) = Exponential(1/b) - Exponential(1/b)
        exponential_rate = 1.0 / noise_scale
        
        # Generate uniform random variables
        u1 = torch.rand_like(tensor)
        u2 = torch.rand_like(tensor)
        
        # Transform to exponential: -ln(1-u)/rate
        # Using log(u) directly gives exponential distribution
        noise1 = -torch.log(u1) / exponential_rate
        noise2 = -torch.log(u2) / exponential_rate
        
        # Laplace noise = noise1 - noise2
        laplace_noise = noise1 - noise2
        
        # Add noise to original tensor
        noisy_tensor = tensor + laplace_noise
        
        return noisy_tensor
    
    def privatize_gradients(
        self,
        gradients: Dict[str, torch.Tensor],
        clipping_norm: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Privatize gradients using Laplace mechanism.
        
        Args:
            gradients: Dictionary of gradient tensors
            clipping_norm: Gradient clipping norm (L1 sensitivity)
            
        Returns:
            Privatized gradients
        """
        privatized_grads = {}
        
        for name, grad in gradients.items():
            # Add Laplace noise to gradients
            noisy_grad = self.add_noise(grad, clipping_norm)
            privatized_grads[name] = noisy_grad
            
        return privatized_grads
    
    def compute_privacy_spent(
        self,
        num_steps: int,
        composition: str = "basic"
    ) -> float:
        """
        Compute privacy spent after multiple steps.
        
        Args:
            num_steps: Number of optimization steps
            composition: Composition method ("basic" or "advanced")
            
        Returns:
            Total epsilon spent (delta = 0 for Laplace)
        """
        if composition == "basic":
            # Simple composition: ε_total = k * ε
            return num_steps * self.epsilon
        else:
            # Advanced composition for Laplace is complex
            # Using basic composition as conservative estimate
            return num_steps * self.epsilon
    
    def calibrate_noise(
        self,
        target_epsilon: float,
        num_steps: int,
        composition: str = "basic"
    ) -> float:
        """
        Calibrate noise multiplier for target privacy budget.
        
        Args:
            target_epsilon: Target epsilon
            num_steps: Number of steps
            composition: Composition method
            
        Returns:
            Required noise multiplier
        """
        if composition == "basic":
            # ε_total = k * ε => ε = ε_total / k
            per_step_epsilon = target_epsilon / num_steps
            return 1.0 / per_step_epsilon
        else:
            # Conservative estimate
            return 1.0 / (target_epsilon / num_steps)
    
    def add_noise_sparse(
        self,
        tensor: torch.Tensor,
        sensitivity: float = 1.0,
        sparsity_threshold: float = 0.01
    ) -> torch.Tensor:
        """
        Add Laplace noise with sparsity preservation.
        
        Only adds noise to significant values (above threshold).
        
        Args:
            tensor: Input tensor
            sensitivity: L1 sensitivity
            sparsity_threshold: Threshold for preserving sparsity
            
        Returns:
            Noisy tensor with sparsity preserved
        """
        if sensitivity <= 0:
            raise ValueError("Sensitivity must be positive")
            
        noise_scale = self.noise_multiplier * sensitivity
        
        # Create mask for significant values
        mask = torch.abs(tensor) > sparsity_threshold
        
        # Generate noise only for significant values
        noise = torch.zeros_like(tensor)
        if mask.any():
            # Generate Laplace noise for significant entries
            significant_count = mask.sum().item()
            
            u1 = torch.rand(significant_count, device=tensor.device)
            u2 = torch.rand(significant_count, device=tensor.device)
            
            exponential_rate = 1.0 / noise_scale
            noise1 = -torch.log(u1) / exponential_rate
            noise2 = -torch.log(u2) / exponential_rate
            
            laplace_noise = noise1 - noise2
            noise[mask] = laplace_noise
            
        # Add noise to original tensor
        noisy_tensor = tensor + noise
        
        return noisy_tensor
    
    def get_noise_statistics(self) -> Dict[str, float]:
        """Get noise mechanism statistics."""
        return {
            "epsilon": self.epsilon,
            "delta": 0.0,  # Pure differential privacy
            "noise_multiplier": self.noise_multiplier,
            "mechanism": "Laplace"
        }