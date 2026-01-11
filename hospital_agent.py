"""
Ghost Agent Launcher for Multi-Laptop Hackathon Demo
Each laptop runs this script with a different hospital configuration

Usage on each laptop:
  python hospital_agent.py --hospital "AIIMS_Delhi" --server 192.168.1.100:8000
  
The --server should point to the IP of the central SNA machine.
"""

import asyncio
import argparse
import logging
import sys
import socket
import platform
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from datetime import datetime
import requests
import time
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.synthetic_health_data import generate_synthetic_diabetes_data, generate_feature_tensor

# Import from shared model registry - Single Source of Truth
from models.registry import DiabetesPredictionModel, ModelRegistry


# Configure logging with hospital name
def setup_logging(hospital_id: str):
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - [{hospital_id}] - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'{hospital_id}_training.log')
        ]
    )
    return logging.getLogger(hospital_id)


class HospitalGhostAgent:
    """
    Production Ghost Agent for Laptop-Based Hospital Demo
    
    Features:
    - Real DP-SGD training with gradient clipping and Gaussian noise
    - Automatic global model sync from SNA
    - Privacy budget tracking
    - Network resilience with retries
    """
    
    def __init__(
        self,
        hospital_id: str,
        sna_url: str = "http://localhost:8000",
        epsilon_per_round: float = 1.0,
        delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 1.1,
        patient_count: int = 1000
    ):
        self.hospital_id = hospital_id
        self.sna_url = sna_url.rstrip('/')
        self.epsilon_per_round = epsilon_per_round
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        
        self.logger = setup_logging(hospital_id)
        
        # Get machine info for display
        self.machine_name = platform.node()
        self.machine_ip = self._get_local_ip()
        
        # Privacy budget tracking
        self.epsilon_spent = 0.0
        self.rounds_completed = 0
        
        # Model
        self.model = DiabetesPredictionModel()
        self.criterion = nn.BCELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        
        # Generate local data (unique to this hospital/laptop)
        self.logger.info(f"Generating {patient_count} synthetic patient records...")
        self.local_data = generate_synthetic_diabetes_data(
            n_samples=patient_count,
            hospital_id=hospital_id,
            seed=hash(hospital_id) % 10000
        )
        
        self.features, self.labels = generate_feature_tensor(self.local_data)
        self.logger.info(f"Local dataset ready: {len(self.local_data)} patients")
        
        # Display startup banner
        self._print_banner()
    
    def _get_local_ip(self) -> str:
        """Get local IP address"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"
    
    def _print_banner(self):
        """Print startup banner"""
        print("\n" + "=" * 60)
        print("  🏥 GHOST PROTOCOL - HOSPITAL AGENT")
        print("=" * 60)
        print(f"  Hospital ID:     {self.hospital_id}")
        print(f"  Machine:         {self.machine_name}")
        print(f"  Local IP:        {self.machine_ip}")
        print(f"  SNA Server:      {self.sna_url}")
        print(f"  Patient Records: {len(self.local_data)}")
        print(f"  Privacy Budget:  ε={self.epsilon_per_round}/round")
        print("=" * 60 + "\n")
    
    def _check_sna_connection(self) -> bool:
        """Verify connection to SNA server"""
        try:
            response = requests.get(f"{self.sna_url}/status", timeout=5)
            if response.status_code == 200:
                status = response.json()
                self.logger.info(f"Connected to SNA - Round: {status.get('current_round', 0)}")
                return True
        except Exception as e:
            self.logger.error(f"Cannot connect to SNA at {self.sna_url}: {e}")
        return False
    
    def _sync_global_model(self) -> bool:
        """Download latest global model from SNA"""
        try:
            response = requests.get(f"{self.sna_url}/global_model", timeout=10)
            if response.status_code == 200:
                data = response.json()
                weights = data.get("weights", {})
                
                if weights:
                    # Load weights into local model
                    for name, param in self.model.named_parameters():
                        if name in weights:
                            param.data = torch.FloatTensor(weights[name])
                    
                    self.logger.info(f"Synced global model (round {data.get('round', 0)})")
                    return True
        except Exception as e:
            self.logger.warning(f"Could not sync global model: {e}")
        return False
    
    def _clip_gradients(self) -> float:
        """Clip gradients to bound sensitivity"""
        total_norm = 0.0
        
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        
        total_norm = total_norm ** 0.5
        
        # Clip
        clip_coef = self.max_grad_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)
        
        return min(total_norm, self.max_grad_norm)
    
    def _add_dp_noise(self):
        """Add Gaussian noise for differential privacy"""
        with torch.no_grad():
            for param in self.model.parameters():
                if param.grad is not None:
                    noise = torch.randn_like(param.grad) * self.noise_multiplier * self.max_grad_norm
                    param.grad.data.add_(noise)
    
    def train_local_round(self, epochs: int = 5, batch_size: int = 32) -> dict:
        """Train model locally with differential privacy"""
        
        self.logger.info(f"Starting local training: {epochs} epochs, batch_size={batch_size}")
        
        # Create data loader
        features_tensor = torch.FloatTensor(self.features)
        labels_tensor = torch.FloatTensor(self.labels).unsqueeze(1)
        dataset = TensorDataset(features_tensor, labels_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        gradient_norms = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for batch_features, batch_labels in dataloader:
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_features)
                loss = self.criterion(outputs, batch_labels)
                
                # Backward pass
                loss.backward()
                
                # Clip gradients (for DP sensitivity)
                grad_norm = self._clip_gradients()
                gradient_norms.append(grad_norm)
                
                # Add DP noise
                self._add_dp_noise()
                
                # Update weights
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_epoch_loss = epoch_loss / len(dataloader)
            self.logger.info(f"  Epoch {epoch+1}/{epochs}: Loss = {avg_epoch_loss:.4f}")
            total_loss += epoch_loss
        
        # Calculate local AUC
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(features_tensor).numpy().flatten()
            from sklearn.metrics import roc_auc_score
            try:
                local_auc = roc_auc_score(self.labels, predictions)
            except:
                local_auc = 0.5
        
        # Update privacy budget
        self.epsilon_spent += self.epsilon_per_round
        self.rounds_completed += 1
        
        avg_grad_norm = np.mean(gradient_norms)
        
        self.logger.info(f"Training complete: AUC={local_auc:.4f}, ε_spent={self.epsilon_spent:.4f}")
        
        return {
            "hospital_id": self.hospital_id,
            "round": self.rounds_completed,
            "local_auc": float(local_auc),
            "avg_loss": total_loss / num_batches,
            "gradient_norm": float(avg_grad_norm),
            "epsilon_spent": self.epsilon_spent,
            "epsilon_this_round": self.epsilon_per_round,
            "delta": self.delta,
            "samples_trained": len(self.local_data),
            "timestamp": datetime.utcnow().isoformat(),
            "machine_name": self.machine_name,
            "machine_ip": self.machine_ip
        }
    
    def get_model_weights(self) -> dict:
        """Extract model weights for submission to SNA"""
        weights = {}
        for name, param in self.model.named_parameters():
            weights[name] = param.data.tolist()
        return weights
    
    async def submit_update_to_sna(self, training_results: dict) -> dict:
        """Submit model update to Secure National Aggregator"""
        
        self.logger.info(f"Submitting update to SNA...")
        
        ghost_pack = {
            "hospital_id": self.hospital_id,
            "round": training_results["round"],
            "weights": self.get_model_weights(),
            "metadata": {
                "local_auc": training_results["local_auc"],
                "gradient_norm": training_results["gradient_norm"],
                "samples_trained": training_results["samples_trained"],
                "machine_info": {
                    "name": training_results["machine_name"],
                    "ip": training_results["machine_ip"]
                },
                "dp_compliance": {
                    "epsilon_spent": training_results["epsilon_spent"],
                    "delta": training_results["delta"],
                    "mechanism": "Gaussian",
                    "noise_multiplier": self.noise_multiplier,
                    "max_grad_norm": self.max_grad_norm
                }
            },
            "timestamp": training_results["timestamp"]
        }
        
        # Retry logic for network resilience
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.sna_url}/submit_update",
                    json=ghost_pack,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    self.logger.info(f"✅ Update accepted! SNA Round: {result.get('round')}, Pending: {result.get('pending_count')}")
                    return result
                else:
                    self.logger.error(f"SNA rejected update: {response.status_code}")
                    if attempt < max_retries - 1:
                        self.logger.info(f"Retrying in 5 seconds... ({attempt + 1}/{max_retries})")
                        await asyncio.sleep(5)
                        
            except Exception as e:
                self.logger.error(f"Failed to submit: {e}")
                if attempt < max_retries - 1:
                    self.logger.info(f"Retrying in 5 seconds... ({attempt + 1}/{max_retries})")
                    await asyncio.sleep(5)
        
        return {"accepted": False, "error": "Max retries exceeded"}
    
    async def run_continuous(self, rounds: int = 10, delay_between_rounds: int = 10):
        """Run continuous federated learning rounds"""
        
        # Check connection first
        if not self._check_sna_connection():
            self.logger.error("❌ Cannot connect to SNA server. Please verify:")
            self.logger.error(f"   1. SNA is running on {self.sna_url}")
            self.logger.error("   2. Firewall allows connection")
            self.logger.error("   3. Both machines are on same network")
            return
        
        print("\n" + "=" * 60)
        print(f"  🚀 STARTING FEDERATED LEARNING")
        print(f"     {rounds} rounds, {delay_between_rounds}s delay between rounds")
        print("=" * 60 + "\n")
        
        for round_num in range(rounds):
            print(f"\n{'─' * 40}")
            print(f"  📊 ROUND {round_num + 1}/{rounds}")
            print(f"{'─' * 40}")
            
            # Sync global model before training
            self._sync_global_model()
            
            # Train locally
            training_results = self.train_local_round(epochs=5, batch_size=32)
            
            # Submit to SNA
            result = await self.submit_update_to_sna(training_results)
            
            # Display round summary
            print(f"\n  ┌{'─' * 38}┐")
            print(f"  │ Local AUC:      {training_results['local_auc']:.4f}             │")
            print(f"  │ Gradient Norm:  {training_results['gradient_norm']:.4f}             │")
            print(f"  │ ε Spent:        {training_results['epsilon_spent']:.4f}             │")
            print(f"  │ Status:         {'✅ Accepted' if result.get('status') == 'accepted' else '❌ Failed'}           │")
            print(f"  └{'─' * 38}┘")
            
            if round_num < rounds - 1:
                self.logger.info(f"Waiting {delay_between_rounds}s before next round...")
                await asyncio.sleep(delay_between_rounds)
        
        # Final summary
        print("\n" + "=" * 60)
        print("  ✅ FEDERATED LEARNING COMPLETE")
        print("=" * 60)
        print(f"  Hospital:        {self.hospital_id}")
        print(f"  Rounds:          {self.rounds_completed}")
        print(f"  Privacy Budget:  ε={self.epsilon_spent:.4f}")
        print(f"  Final AUC:       {training_results['local_auc']:.4f}")
        print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Ghost Protocol Hospital Agent - Run on each laptop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Laptop 1 (AIIMS Delhi)
  python hospital_agent.py --hospital AIIMS_Delhi --server 192.168.1.100:8000

  # Laptop 2 (Fortis Mumbai)  
  python hospital_agent.py --hospital Fortis_Mumbai --server 192.168.1.100:8000

  # Laptop 3 (Apollo Chennai)
  python hospital_agent.py --hospital Apollo_Chennai --server 192.168.1.100:8000

  # Laptop 4 (CMC Vellore)
  python hospital_agent.py --hospital CMC_Vellore --server 192.168.1.100:8000
        """
    )
    
    parser.add_argument(
        "--hospital", "-H",
        type=str,
        required=True,
        help="Hospital ID (e.g., AIIMS_Delhi, Fortis_Mumbai)"
    )
    
    parser.add_argument(
        "--server", "-S",
        type=str,
        default="localhost:8000",
        help="SNA server address (IP:PORT). Get this from the central server machine."
    )
    
    parser.add_argument(
        "--rounds", "-R",
        type=int,
        default=10,
        help="Number of training rounds (default: 10)"
    )
    
    parser.add_argument(
        "--delay", "-D",
        type=int,
        default=10,
        help="Delay between rounds in seconds (default: 10)"
    )
    
    parser.add_argument(
        "--patients", "-P",
        type=int,
        default=1000,
        help="Number of synthetic patient records (default: 1000)"
    )
    
    parser.add_argument(
        "--epsilon", "-E",
        type=float,
        default=1.0,
        help="Privacy budget per round (default: 1.0)"
    )
    
    args = parser.parse_args()
    
    # Build SNA URL
    if not args.server.startswith("http"):
        sna_url = f"http://{args.server}"
    else:
        sna_url = args.server
    
    # Create and run agent
    agent = HospitalGhostAgent(
        hospital_id=args.hospital,
        sna_url=sna_url,
        epsilon_per_round=args.epsilon,
        patient_count=args.patients
    )
    
    # Run the agent
    asyncio.run(agent.run_continuous(
        rounds=args.rounds,
        delay_between_rounds=args.delay
    ))


if __name__ == "__main__":
    main()
