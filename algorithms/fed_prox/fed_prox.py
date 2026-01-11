"""
Module: algorithms/fed_prox/fed_prox.py
DPDP §: 9(4) - Purpose limitation through encrypted gradients
Description: FedProx algorithm for heterogeneous federated learning
Test: pytest tests/test_algorithms.py::test_fed_prox
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from ..fed_avg.fed_avg import FedAvg


class FedProx(FedAvg):
    """
    FedProx: Federated Optimization in Heterogeneous Networks.
    
    Adds a proximal term to the local objective to handle:
    - Non-IID data distributions
    - System heterogeneity
    - Stragglers and dropouts
    
    The local objective becomes:
    L_local(w) = L_task(w) + (μ/2) * ||w - w_global||²
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        local_epochs: int = 5,
        gradient_clip: float = 1.0,
        mu: float = 0.1  # Proximal coefficient
    ):
        super().__init__(learning_rate, batch_size, local_epochs, gradient_clip)
        self.mu = mu
        
    def local_train_step(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        global_weights: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[float, Dict[str, torch.Tensor]]:
        """
        Perform local training with proximal term.
        
        Args:
            model: Local model instance
            data_loader: Local data loader
            optimizer: Optimizer instance
            device: Device to run on
            global_weights: Global model weights for proximal term
            
        Returns:
            Tuple of (loss, model gradients)
        """
        model.train()
        total_loss = 0
        num_batches = 0
        
        # Store global weights for proximal term
        if global_weights is not None:
            global_dict = {}
            for name, param in model.named_parameters():
                if name in global_weights:
                    global_dict[name] = global_weights[name].to(device)
        else:
            global_dict = {}
            for name, param in model.named_parameters():
                global_dict[name] = param.data.clone()
                
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            task_loss = nn.functional.cross_entropy(output, target)
            
            # Add proximal term
            proximal_loss = 0
            for name, param in model.named_parameters():
                if name in global_dict:
                    diff = param - global_dict[name]
                    proximal_loss += torch.sum(diff ** 2)
                    
            total_batch_loss = task_loss + (self.mu / 2) * proximal_loss
            total_batch_loss.backward()
            
            # Gradient clipping for privacy
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)
                
            optimizer.step()
            
            total_loss += total_batch_loss.item()
            num_batches += 1
            
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        # Extract gradients for privacy processing
        gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone().detach()
                
        return avg_loss, gradients
    
    def aggregate_with_proximal_adjustment(
        self,
        local_weights: List[Dict[str, torch.Tensor]],
        client_sizes: List[int],
        global_weights: Dict[str, torch.Tensor],
        proximal_weights: Optional[List[float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate with proximal term weighting.
        
        Args:
            local_weights: List of local model weights
            client_sizes: List of dataset sizes
            global_weights: Global model weights
            proximal_weights: Optional proximal weights for each client
            
        Returns:
            Aggregated weights with proximal adjustment
        """
        if proximal_weights is None:
            # Use standard FedAvg aggregation
            return self.aggregate(local_weights, client_sizes)
            
        if len(local_weights) != len(proximal_weights):
            raise ValueError("Mismatch between weights and proximal weights")
            
        total_samples = sum(client_sizes)
        if total_samples == 0:
            raise ValueError("Total samples cannot be zero")
            
        # Initialize aggregated weights
        aggregated = {}
        first_weights = local_weights[0]
        
        for key in first_weights.keys():
            weighted_sum = torch.zeros_like(first_weights[key])
            
            for i, local_w in enumerate(local_weights):
                # Combine dataset size weighting with proximal weighting
                data_weight = client_sizes[i] / total_samples
                prox_weight = proximal_weights[i]
                combined_weight = data_weight * prox_weight
                
                weighted_sum += combined_weight * local_w[key]
                
            aggregated[key] = weighted_sum
            
        return aggregated
    
    def compute_proximal_weights(
        self,
        local_weights: List[Dict[str, torch.Tensor]],
        global_weights: Dict[str, torch.Tensor]
    ) -> List[float]:
        """
        Compute proximal weights based on distance from global model.
        
        Clients closer to the global model get higher weights.
        
        Args:
            local_weights: List of local model weights
            global_weights: Global model weights
            
        Returns:
            List of proximal weights for each client
        """
        distances = []
        
        for local_w in local_weights:
            total_dist_sq = 0
            
            for key in global_weights.keys():
                if key in local_w:
                    diff = local_w[key] - global_weights[key]
                    total_dist_sq += torch.sum(diff ** 2).item()
                    
            distances.append(torch.sqrt(torch.tensor(total_dist_sq)).item())
            
        # Convert distances to weights (closer = higher weight)
        # Use exponential decay: weight = exp(-distance / median_distance)
        if not distances or all(d == 0 for d in distances):
            return [1.0] * len(local_weights)
            
        median_dist = sorted(distances)[len(distances) // 2]
        if median_dist == 0:
            median_dist = max(distances) * 0.1  # Avoid division by zero
            
        weights = [torch.exp(-d / median_dist).item() for d in distances]
        
        # Normalize to sum to 1
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
            
        return weights