"""
Module: ghost_agent/local_training/local_trainer.py
DPDP §: 9(4) - Purpose limitation through encrypted gradients
Description: Local trainer with privacy-preserving federated learning
Test: pytest tests/test_local_training.py::test_local_trainer
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime

# Import algorithms
from algorithms.fed_avg.fed_avg import FedAvg
from algorithms.fed_prox.fed_prox import FedProx
from ..privacy_engine.privacy_engine import PrivacyEngine


class LocalTrainer:
    """
    Local Trainer for Ghost Protocol.
    
    Implements local training with:
    - Privacy-preserving gradient processing
    - Multiple FL algorithms (FedAvg, FedProx)
    - Byzantine-robust gradient clipping
    - DPDP compliance monitoring
    """
    
    def __init__(
        self,
        hospital_id: str,
        model: nn.Module,
        algorithm: str = "fedavg",
        learning_rate: float = 0.01,
        batch_size: int = 32,
        local_epochs: int = 5,
        device: Optional[torch.device] = None
    ):
        """
        Initialize Local Trainer.
        
        Args:
            hospital_id: Unique hospital identifier
            model: Neural network model
            algorithm: FL algorithm ("fedavg", "fedprox")
            learning_rate: Learning rate
            batch_size: Batch size
            local_epochs: Number of local epochs
            device: Device to run on
        """
        self.hospital_id = hospital_id
        self.model = model
        self.algorithm = algorithm
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.logger = logging.getLogger(__name__)
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize FL algorithm
        if algorithm == "fedavg":
            self.fl_algorithm = FedAvg(
                learning_rate=learning_rate,
                batch_size=batch_size,
                local_epochs=local_epochs
            )
        elif algorithm == "fedprox":
            self.fl_algorithm = FedProx(
                learning_rate=learning_rate,
                batch_size=batch_size,
                local_epochs=local_epochs
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
            
        # Initialize privacy engine
        self.privacy_engine = PrivacyEngine()
        
        # Training statistics
        self.stats = {
            "total_epochs": 0,
            "total_batches": 0,
            "avg_loss": 0.0,
            "privacy_violations": 0
        }
        
    def prepare_data(
        self,
        features: np.ndarray,
        targets: Optional[np.ndarray] = None,
        validation_split: float = 0.2
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Prepare data loaders for training.
        
        Args:
            features: Feature matrix
            targets: Target vector (optional for unsupervised)
            validation_split: Validation split ratio
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Convert to tensors
        features_tensor = torch.FloatTensor(features)
        
        if targets is not None:
            targets_tensor = torch.LongTensor(targets) if len(np.unique(targets)) < 10 else torch.FloatTensor(targets)
            dataset = TensorDataset(features_tensor, targets_tensor)
        else:
            dataset = TensorDataset(features_tensor)
            
        # Split dataset
        dataset_size = len(dataset)
        val_size = int(validation_split * dataset_size)
        train_size = dataset_size - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )
        
        if val_size > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False
            )
        else:
            val_loader = None
            
        return train_loader, val_loader
        
    def local_train_round(
        self,
        train_loader: DataLoader,
        global_weights: Optional[Dict[str, torch.Tensor]] = None,
        round_num: int = 0
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Perform one round of local training.
        
        Args:
            train_loader: Training data loader
            global_weights: Global model weights (for FedProx)
            round_num: Current round number
            
        Returns:
            Tuple of (local_weights, training_stats)
        """
        self.model.train()
        
        # Initialize optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Track training metrics
        total_loss = 0
        num_batches = 0
        gradient_norms = []
        
        # Training loop
        for epoch in range(self.local_epochs):
            epoch_loss = 0
            
            for batch_idx, batch_data in enumerate(train_loader):
                # Prepare batch
                if len(batch_data) == 2:
                    data, target = batch_data
                    data, target = data.to(self.device), target.to(self.device)
                else:
                    data = batch_data[0].to(self.device)
                    target = None
                    
                # Forward pass
                optimizer.zero_grad()
                output = self.model(data)
                
                # Compute loss
                if target is not None:
                    loss = nn.functional.cross_entropy(output, target)
                else:
                    # Unsupervised loss (e.g., reconstruction)
                    loss = nn.functional.mse_loss(output, data)
                    
                # Backward pass
                loss.backward()
                
                # Extract gradients before processing
                gradients = {}
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        gradients[name] = param.grad.clone().detach()
                        
                # Apply privacy processing
                try:
                    privatized_grads, privacy_stats = self.privacy_engine.privatize_gradients(
                        gradients, step_num=batch_idx
                    )
                    
                    # Apply privatized gradients
                    for name, param in self.model.named_parameters():
                        if name in privatized_grads and param.grad is not None:
                            param.grad.data = privatized_grads[name]
                            
                except Exception as e:
                    self.logger.warning(f"Privacy processing failed: {e}")
                    self.stats["privacy_violations"] += 1
                    
                # Optimizer step
                optimizer.step()
                
                # Track metrics
                total_loss += loss.item()
                epoch_loss += loss.item()
                num_batches += 1
                
                # Track gradient norms for monitoring
                total_norm = 0
                for param in self.model.parameters():
                    if param.grad is not None:
                        total_norm += torch.sum(param.grad ** 2).item()
                gradient_norms.append(np.sqrt(total_norm))
                
                # Log progress
                if batch_idx % 10 == 0:
                    self.logger.debug(
                        f"Round {round_num}, Epoch {epoch}, Batch {batch_idx}: "
                        f"Loss={loss.item():.4f}"
                    )
                    
        # Compute final statistics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_gradient_norm = np.mean(gradient_norms) if gradient_norms else 0
        
        # Get privacy report
        privacy_report = self.privacy_engine.get_privacy_report()
        
        # Extract local weights
        local_weights = {}
        for name, param in self.model.named_parameters():
            local_weights[name] = param.data.clone().detach()
            
        # Training statistics
        training_stats = {
            "round_num": round_num,
            "avg_loss": avg_loss,
            "num_batches": num_batches,
            "num_epochs": self.local_epochs,
            "avg_gradient_norm": avg_gradient_norm,
            "privacy_report": privacy_report,
            "dp_compliant": privacy_report["dp_compliance_status"] == "COMPLIANT",
            "epsilon_spent": privacy_report["current_epsilon_spent"]
        }
        
        # Update global stats
        self.stats["total_epochs"] += self.local_epochs
        self.stats["total_batches"] += num_batches
        self.stats["avg_loss"] = avg_loss
        
        self.logger.info(
            f"Local training round {round_num} completed: "
            f"loss={avg_loss:.4f}, ε_spent={privacy_report['current_epsilon_spent']:.3f}"
        )
        
        return local_weights, training_stats
        
    def validate_model(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Validate model on validation set.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Validation metrics
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                if len(batch_data) == 2:
                    data, target = batch_data
                    data, target = data.to(self.device), target.to(self.device)
                else:
                    continue  # Skip unsupervised validation
                    
                output = self.model(data)
                loss = nn.functional.cross_entropy(output, target)
                total_loss += loss.item()
                
                # Compute accuracy
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0
        accuracy = correct / total if total > 0 else 0
        
        return {
            "val_loss": avg_loss,
            "val_accuracy": accuracy
        }
        
    def get_model_weights(self) -> Dict[str, torch.Tensor]:
        """Get current model weights."""
        weights = {}
        for name, param in self.model.named_parameters():
            weights[name] = param.data.clone().detach()
        return weights
        
    def set_model_weights(self, weights: Dict[str, torch.Tensor]):
        """Set model weights."""
        for name, param in self.model.named_parameters():
            if name in weights:
                param.data = weights[name].clone().detach()
                
    def get_training_stats(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        privacy_report = self.privacy_engine.get_privacy_report()
        
        return {
            "hospital_id": self.hospital_id,
            "algorithm": self.algorithm,
            "total_epochs": self.stats["total_epochs"],
            "total_batches": self.stats["total_batches"],
            "avg_loss": self.stats["avg_loss"],
            "privacy_violations": self.stats["privacy_violations"],
            "privacy_report": privacy_report,
            "dp_compliant": privacy_report["dp_compliance_status"] == "COMPLIANT",
            "model_parameters": sum(p.numel() for p in self.model.parameters()),
            "device": str(self.device)
        }
        
    def export_privacy_log(self, filepath: str):
        """Export privacy log to file."""
        self.privacy_engine.export_privacy_log(filepath)
        
    def reset_privacy_engine(self):
        """Reset privacy engine for new training session."""
        self.privacy_engine.reset_privacy_budget()
        
    def detect_anomalies(
        self,
        gradients: Dict[str, torch.Tensor],
        z_threshold: float = 3.0
    ) -> bool:
        """
        Detect anomalous gradients.
        
        Args:
            gradients: Gradient dictionary
            z_threshold: Z-score threshold
            
        Returns:
            True if anomalies detected
        """
        # Compute gradient norms
        norms = []
        for grad in gradients.values():
            norm = torch.norm(grad).item()
            norms.append(norm)
            
        # Compute statistics
        mean_norm = np.mean(norms)
        std_norm = np.std(norms)
        
        if std_norm == 0:
            return False
            
        # Check for anomalies
        z_scores = [(norm - mean_norm) / std_norm for norm in norms]
        anomalies = any(abs(z) > z_threshold for z in z_scores)
        
        return anomalies
        
    def apply_byzantine_protection(
        self,
        gradients: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Apply Byzantine fault tolerance measures.
        
        Args:
            gradients: Gradient dictionary
            
        Returns:
            Protected gradients
        """
        # For single hospital, just clip extreme values
        protected_grads = {}
        
        for name, grad in gradients.items():
            # Clip extreme values
            clip_value = 10.0  # Conservative clipping
            protected_grad = torch.clamp(grad, -clip_value, clip_value)
            protected_grads[name] = protected_grad
            
        return protected_grads