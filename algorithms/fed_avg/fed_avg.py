"""
Module: algorithms/fed_avg/fed_avg.py
DPDP §: 9(4) - Purpose limitation through encrypted gradients
Description: Federated Averaging algorithm implementation
Test: pytest tests/test_algorithms.py::test_fed_avg
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
import numpy as np


class FedAvg:
    """
    Federated Averaging (FedAvg) implementation.
    
    Implements the classic FedAvg algorithm with support for:
    - Weighted aggregation based on dataset sizes
    - Gradient clipping for privacy
    - Byzantine-robust aggregation
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        local_epochs: int = 5,
        gradient_clip: float = 1.0
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.gradient_clip = gradient_clip
        
    def aggregate(
        self,
        local_weights: List[Dict[str, torch.Tensor]],
        client_sizes: List[int],
        global_weights: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate local model weights using FedAvg.
        
        Args:
            local_weights: List of local model weights from clients
            client_sizes: List of dataset sizes for each client
            global_weights: Current global weights (optional)
            
        Returns:
            Aggregated global weights
            
        Note:
            Weighted aggregation: w_global = Σ(n_i/n_total) * w_i
            where n_i is the number of samples at client i
        """
        if not local_weights:
            raise ValueError("No local weights to aggregate")
            
        if len(local_weights) != len(client_sizes):
            raise ValueError("Mismatch between weights and client sizes")
            
        total_samples = sum(client_sizes)
        if total_samples == 0:
            raise ValueError("Total samples cannot be zero")
            
        # Initialize aggregated weights
        aggregated = {}
        first_weights = local_weights[0]
        
        for key in first_weights.keys():
            # Weighted sum of parameters
            weighted_sum = torch.zeros_like(first_weights[key])
            
            for i, local_w in enumerate(local_weights):
                weight = client_sizes[i] / total_samples
                weighted_sum += weight * local_w[key]
                
            aggregated[key] = weighted_sum
            
        return aggregated
    
    def local_train_step(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device
    ) -> Tuple[float, Dict[str, torch.Tensor]]:
        """
        Perform one local training step.
        
        Args:
            model: Local model instance
            data_loader: Local data loader
            optimizer: Optimizer instance
            device: Device to run on
            
        Returns:
            Tuple of (loss, model gradients)
        """
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = nn.functional.cross_entropy(output, target)
            loss.backward()
            
            # Gradient clipping for privacy
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)
                
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        # Extract gradients for privacy processing
        gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone().detach()
                
        return avg_loss, gradients
    
    def compute_update_norms(
        self,
        local_weights: List[Dict[str, torch.Tensor]],
        global_weights: Dict[str, torch.Tensor]
    ) -> List[float]:
        """
        Compute L2 norms of weight updates for anomaly detection.
        
        Args:
            local_weights: List of local model weights
            global_weights: Global model weights
            
        Returns:
            List of L2 norms for each client
        """
        norms = []
        
        for local_w in local_weights:
            total_norm_sq = 0
            
            for key in global_weights.keys():
                if key in local_w:
                    diff = local_w[key] - global_weights[key]
                    total_norm_sq += torch.sum(diff ** 2).item()
                    
            norms.append(torch.sqrt(torch.tensor(total_norm_sq)).item())
            
        return norms
    
    def detect_anomalies(
        self,
        update_norms: List[float],
        z_threshold: float = 3.0
    ) -> List[bool]:
        """
        Detect anomalous updates using Z-score.
        
        Args:
            update_norms: List of update norms
            z_threshold: Z-score threshold for anomaly detection
            
        Returns:
            List of boolean flags indicating anomalies
        """
        if len(update_norms) <= 1:
            return [False] * len(update_norms)
            
        norms_array = np.array(update_norms)
        mean_norm = np.mean(norms_array)
        std_norm = np.std(norms_array)
        
        if std_norm == 0:
            return [False] * len(update_norms)
            
        z_scores = (norms_array - mean_norm) / std_norm
        anomalies = np.abs(z_scores) > z_threshold
        
        return anomalies.tolist()
    
    def aggregate_with_byzantine_protection(
        self,
        local_weights: List[Dict[str, torch.Tensor]],
        client_sizes: List[int],
        global_weights: Optional[Dict[str, torch.Tensor]] = None,
        z_threshold: float = 3.0
    ) -> Tuple[Dict[str, torch.Tensor], List[bool]]:
        """
        Aggregate with Byzantine fault tolerance.
        
        Args:
            local_weights: List of local model weights
            client_sizes: List of dataset sizes
            global_weights: Current global weights
            z_threshold: Z-score threshold
            
        Returns:
            Tuple of (aggregated weights, anomaly flags)
        """
        if global_weights is None:
            # First round, no Byzantine protection yet
            aggregated = self.aggregate(local_weights, client_sizes)
            return aggregated, [False] * len(local_weights)
            
        # Compute update norms
        update_norms = self.compute_update_norms(local_weights, global_weights)
        
        # Detect anomalies
        anomalies = self.detect_anomalies(update_norms, z_threshold)
        
        # Filter out anomalous updates
        filtered_weights = []
        filtered_sizes = []
        
        for i, (weights, size) in enumerate(zip(local_weights, client_sizes)):
            if not anomalies[i]:
                filtered_weights.append(weights)
                filtered_sizes.append(size)
                
        if not filtered_weights:
            # All updates were anomalous, return current global weights
            return global_weights, anomalies
            
        # Aggregate filtered weights
        aggregated = self.aggregate(filtered_weights, filtered_sizes)
        
        return aggregated, anomalies