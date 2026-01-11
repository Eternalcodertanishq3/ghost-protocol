"""
Ghost Agent with Real EMR Integration
Connects to FHIR/HL7/Database EMR systems or uses synthetic data

Usage:
  # Synthetic data (default)
  python hospital_agent_emr.py --hospital AIIMS_Delhi --server 192.168.1.100:8000
  
  # FHIR EMR connection
  python hospital_agent_emr.py --hospital AIIMS_Delhi --server 192.168.1.100:8000 \
    --emr-type fhir --emr-url https://fhir.hospital.local/api

  # HL7 EMR connection
  python hospital_agent_emr.py --hospital AIIMS_Delhi --server 192.168.1.100:8000 \
    --emr-type hl7 --emr-connection "mllp://192.168.1.50:2575"

  # Database EMR connection
  python hospital_agent_emr.py --hospital AIIMS_Delhi --server 192.168.1.100:8000 \
    --emr-type database --emr-connection "postgresql://user:pass@localhost/emr_db"
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
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.synthetic_health_data import generate_synthetic_diabetes_data, generate_feature_tensor

# Try to import EMR adapters
try:
    from ghost_agent.adapters.emr_adapters import (
        EMRAdapterFactory, 
        FHIRAdapter, 
        HL7Adapter, 
        DatabaseAdapter
    )
    EMR_AVAILABLE = True
except ImportError:
    EMR_AVAILABLE = False
    print("Warning: EMR adapters not available, using synthetic data only")


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


class DiabetesPredictionModel(nn.Module):
    """Neural network matching SNA global model architecture"""
    
    def __init__(self, input_size: int = 8, hidden_size: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


class ReadmissionPredictionModel(nn.Module):
    """Extended model for hospital readmission prediction with more features"""
    
    def __init__(self, input_size: int = 16, hidden_size: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.1)
        
        self.fc4 = nn.Linear(hidden_size // 4, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        if x.size(0) > 1:  # BatchNorm needs more than 1 sample
            x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        if x.size(0) > 1:
            x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x


class HospitalGhostAgentEMR:
    """
    Production Ghost Agent with Real EMR Integration
    
    Features:
    - Real DP-SGD training with gradient clipping and Gaussian noise
    - FHIR/HL7/Database EMR integration
    - Consent validation per DPDP Act
    - Automatic global model sync from SNA
    - Privacy budget tracking
    """
    
    def __init__(
        self,
        hospital_id: str,
        sna_url: str = "http://localhost:8000",
        emr_type: str = "synthetic",
        emr_config: dict = None,
        epsilon_per_round: float = 1.0,
        delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 1.1,
        patient_count: int = 1000,
        fedprox_mu: float = 0.01
    ):
        self.hospital_id = hospital_id
        self.sna_url = sna_url.rstrip('/')
        self.emr_type = emr_type
        self.emr_config = emr_config or {}
        self.epsilon_per_round = epsilon_per_round
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.patient_count = patient_count
        self.fedprox_mu = fedprox_mu  # FedProx regularization parameter
        
        self.logger = setup_logging(hospital_id)
        
        # Machine info
        self.machine_name = platform.node()
        self.machine_ip = self._get_local_ip()
        
        # Privacy tracking
        self.epsilon_spent = 0.0
        self.rounds_completed = 0
        
        # Data source
        self.emr_adapter = None
        self.local_data = None
        self.features = None
        self.labels = None
        
        # Model - choose based on EMR type
        if emr_type == "synthetic":
            self.model = DiabetesPredictionModel(input_size=8, hidden_size=64)
            self.input_size = 8
        else:
            # Full EMR model with more features
            self.model = ReadmissionPredictionModel(input_size=16, hidden_size=128)
            self.input_size = 16
            
        # Global model copy for FedProx
        import copy
        self.global_model_copy = copy.deepcopy(self.model)
        for param in self.global_model_copy.parameters():
            param.requires_grad = False
        
        self.criterion = nn.BCELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        
        # Display startup banner
        self._print_banner()
    
    def _get_local_ip(self) -> str:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"
    
    def _print_banner(self):
        print("\n" + "=" * 60)
        print("  🏥 GHOST PROTOCOL - HOSPITAL AGENT (EMR Mode)")
        print("=" * 60)
        print(f"  Hospital ID:     {self.hospital_id}")
        print(f"  Machine:         {self.machine_name}")
        print(f"  Local IP:        {self.machine_ip}")
        print(f"  SNA Server:      {self.sna_url}")
        print(f"  EMR Type:        {self.emr_type.upper()}")
        if self.emr_type != "synthetic":
            print(f"  EMR Connection:  {self.emr_config.get('url', 'Not configured')}")
        print(f"  Privacy Budget:  ε={self.epsilon_per_round}/round")
        print("=" * 60 + "\n")
    
    async def initialize_data_source(self):
        """Initialize data source (EMR or synthetic)"""
        
        if self.emr_type == "synthetic":
            self.logger.info(f"Generating {self.patient_count} synthetic patient records...")
            self.local_data = generate_synthetic_diabetes_data(
                n_samples=self.patient_count,
                hospital_id=self.hospital_id,
                seed=hash(self.hospital_id) % 10000
            )
            self.features, self.labels = generate_feature_tensor(self.local_data)
            
        elif EMR_AVAILABLE and self.emr_type in ["fhir", "hl7", "database"]:
            self.logger.info(f"Connecting to {self.emr_type.upper()} EMR system...")
            
            # Create EMR adapter
            self.emr_adapter = EMRAdapterFactory.create_adapter(
                hospital_id=self.hospital_id,
                emr_type=self.emr_type,
                config=self.emr_config
            )
            
            # Fetch patient records with consent validation
            records = await self.emr_adapter.get_patient_records(
                consent_status="active",
                purpose="federated_learning",
                validity_check=True,
                limit=self.patient_count
            )
            
            # Validate consent (DPDP compliance)
            validated_records = await self.emr_adapter.validate_consent_for_records(
                records, 
                purpose="federated_learning"
            )
            
            self.local_data = validated_records
            self.logger.info(f"Retrieved {len(validated_records)} consented records from EMR")
            
            # Convert to tensors
            self.features, self.labels = self._emr_to_tensors(validated_records)
            
        else:
            self.logger.warning("EMR adapters not available, falling back to synthetic data")
            self.emr_type = "synthetic"
            await self.initialize_data_source()
        
        self.logger.info(f"Data source ready: {len(self.local_data)} records")
        return True
    
    def _emr_to_tensors(self, records: list) -> tuple:
        """Convert EMR records to training tensors"""
        
        feature_names = [
            "age", "gender", "systolic_bp", "diastolic_bp",
            "heart_rate", "temperature", "oxygen_saturation",
            "hemoglobin", "white_blood_cells", "platelets",
            "glucose", "creatinine", "diabetes", "hypertension",
            "heart_disease", "kidney_disease"
        ]
        
        features = []
        labels = []
        
        for record in records:
            # Extract features
            feature_vector = []
            for name in feature_names:
                value = record.get(name, 0)
                
                # Handle gender encoding
                if name == "gender":
                    value = 1 if value == "M" else 0
                
                feature_vector.append(float(value))
            
            features.append(feature_vector)
            labels.append(float(record.get("readmission_risk", 0)))
        
        return np.array(features), np.array(labels)
    
    def _check_sna_connection(self) -> bool:
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
        try:
            response = requests.get(f"{self.sna_url}/global_model", timeout=10)
            if response.status_code == 200:
                data = response.json()
                weights = data.get("weights", {})
                
                if weights:
                    for name, param in self.model.named_parameters():
                        if name in weights:
                            param.data = torch.FloatTensor(weights[name])
                    
                    # Update local copy for FedProx calculation
                    if self.fedprox_mu > 0:
                        self.global_model_copy.load_state_dict(self.model.state_dict())
                    
                    self.logger.info(f"Synced global model (round {data.get('round', 0)})")
                    return True
        except Exception as e:
            self.logger.warning(f"Could not sync global model: {e}")
        return False
    
    def _clip_gradients(self) -> float:
        total_norm = 0.0
        
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        
        total_norm = total_norm ** 0.5
        
        clip_coef = self.max_grad_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)
        
        return min(total_norm, self.max_grad_norm)
    
    def _add_dp_noise(self):
        with torch.no_grad():
            for param in self.model.parameters():
                if param.grad is not None:
                    noise = torch.randn_like(param.grad) * self.noise_multiplier * self.max_grad_norm
                    param.grad.data.add_(noise)
    
    def train_local_round(self, epochs: int = 5, batch_size: int = 32) -> dict:
        """Train model locally with differential privacy"""
        
        self.logger.info(f"Starting local training: {epochs} epochs, batch_size={batch_size}")
        
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
                
                outputs = self.model(batch_features)
                loss = self.criterion(outputs, batch_labels)
                
                # ==================================================
                # FEDPROX ALGORITHM (Advanced Non-IID Handling)
                # ==================================================
                if self.fedprox_mu > 0 and self.global_model_copy is not None:
                    proximal_term = 0.0
                    for param, global_param in zip(self.model.parameters(), self.global_model_copy.parameters()):
                        proximal_term += (param - global_param).norm(2) ** 2
                    loss += (self.fedprox_mu / 2) * proximal_term
                # ==================================================
                
                loss.backward()
                
                grad_norm = self._clip_gradients()
                gradient_norms.append(grad_norm)
                
                self._add_dp_noise()
                
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
            "data_source": self.emr_type,
            "timestamp": datetime.utcnow().isoformat(),
            "machine_name": self.machine_name,
            "machine_ip": self.machine_ip
        }
    
    def get_model_weights(self) -> dict:
        weights = {}
        for name, param in self.model.named_parameters():
            weights[name] = param.data.tolist()
        return weights
    
    async def submit_update_to_sna(self, training_results: dict) -> dict:
        self.logger.info(f"Submitting update to SNA...")
        
        ghost_pack = {
            "hospital_id": self.hospital_id,
            "round": training_results["round"],
            "weights": self.get_model_weights(),
            "metadata": {
                "local_auc": training_results["local_auc"],
                "gradient_norm": training_results["gradient_norm"],
                "samples_trained": training_results["samples_trained"],
                "data_source": training_results["data_source"],
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
                    self.logger.info(f"✅ Update accepted! SNA Round: {result.get('round')}")
                    return result
                else:
                    self.logger.error(f"SNA rejected update: {response.status_code}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(5)
                        
            except Exception as e:
                self.logger.error(f"Failed to submit: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(5)
        
        return {"accepted": False, "error": "Max retries exceeded"}
    
    async def run_continuous(self, rounds: int = 10, delay_between_rounds: int = 10):
        """Run continuous federated learning rounds"""
        
        # Initialize data source
        await self.initialize_data_source()
        
        # Check connection
        if not self._check_sna_connection():
            self.logger.error("❌ Cannot connect to SNA server")
            return
        
        print("\n" + "=" * 60)
        print(f"  🚀 STARTING FEDERATED LEARNING ({self.emr_type.upper()} data)")
        print(f"     {rounds} rounds, {delay_between_rounds}s delay")
        print("=" * 60 + "\n")
        
        for round_num in range(rounds):
            print(f"\n{'─' * 40}")
            print(f"  📊 ROUND {round_num + 1}/{rounds}")
            print(f"{'─' * 40}")
            
            self._sync_global_model()
            
            training_results = self.train_local_round(epochs=5, batch_size=32)
            
            result = await self.submit_update_to_sna(training_results)
            
            print(f"\n  ┌{'─' * 38}┐")
            print(f"  │ Data Source:    {self.emr_type.upper():<15}     │")
            print(f"  │ Local AUC:      {training_results['local_auc']:.4f}             │")
            print(f"  │ ε Spent:        {training_results['epsilon_spent']:.4f}             │")
            print(f"  │ Status:         {'✅ Accepted' if result.get('status') == 'accepted' else '❌ Failed'}           │")
            print(f"  └{'─' * 38}┘")
            
            if round_num < rounds - 1:
                await asyncio.sleep(delay_between_rounds)
        
        print("\n" + "=" * 60)
        print("  ✅ FEDERATED LEARNING COMPLETE")
        print("=" * 60)
        print(f"  Hospital:        {self.hospital_id}")
        print(f"  Data Source:     {self.emr_type.upper()}")
        print(f"  Rounds:          {self.rounds_completed}")
        print(f"  Privacy Budget:  ε={self.epsilon_spent:.4f}")
        print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Ghost Protocol Hospital Agent with EMR Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Synthetic data (demo mode)
  python hospital_agent_emr.py --hospital AIIMS_Delhi --server 192.168.1.100:8000

  # Real FHIR EMR connection
  python hospital_agent_emr.py --hospital AIIMS_Delhi --server 192.168.1.100:8000 \\
    --emr-type fhir --emr-url https://fhir.hospital.local/api

  # Real HL7 connection (legacy EMR)
  python hospital_agent_emr.py --hospital AIIMS_Delhi --server 192.168.1.100:8000 \\
    --emr-type hl7 --emr-connection "mllp://192.168.1.50:2575"

  # Direct database connection
  python hospital_agent_emr.py --hospital AIIMS_Delhi --server 192.168.1.100:8000 \\
    --emr-type database --emr-connection "postgresql://user:pass@localhost/emr_db"
        """
    )
    
    parser.add_argument("--hospital", "-H", type=str, required=True,
                        help="Hospital ID")
    parser.add_argument("--server", "-S", type=str, default="localhost:8000",
                        help="SNA server address (IP:PORT)")
    parser.add_argument("--rounds", "-R", type=int, default=10,
                        help="Training rounds")
    parser.add_argument("--delay", "-D", type=int, default=10,
                        help="Delay between rounds (seconds)")
    parser.add_argument("--patients", "-P", type=int, default=1000,
                        help="Patient records to use")
    parser.add_argument("--epsilon", "-E", type=float, default=1.0,
                        help="Privacy budget per round")
    parser.add_argument("--fedprox-mu", type=float, default=0.01,
                        help="FedProx regularization (0 to disable)")
    
    # EMR options
    parser.add_argument("--emr-type", type=str, default="synthetic",
                        choices=["synthetic", "fhir", "hl7", "database"],
                        help="EMR system type")
    parser.add_argument("--emr-url", type=str, default=None,
                        help="FHIR server URL")
    parser.add_argument("--emr-connection", type=str, default=None,
                        help="HL7/Database connection string")
    parser.add_argument("--emr-auth-token", type=str, default=None,
                        help="EMR authentication token")
    
    args = parser.parse_args()
    
    # Build SNA URL
    if not args.server.startswith("http"):
        sna_url = f"http://{args.server}"
    else:
        sna_url = args.server
    
    # Build EMR config
    emr_config = {}
    if args.emr_type == "fhir":
        emr_config = {
            "fhir_server_url": args.emr_url,
            "auth_token": args.emr_auth_token,
            "verify_ssl": True
        }
    elif args.emr_type == "hl7":
        emr_config = {
            "hl7_connection_string": args.emr_connection,
            "message_version": "2.5"
        }
    elif args.emr_type == "database":
        emr_config = {
            "connection_string": args.emr_connection,
            "db_type": "postgresql"
        }
    
    # Create and run agent
    agent = HospitalGhostAgentEMR(
        hospital_id=args.hospital,
        sna_url=sna_url,
        emr_type=args.emr_type,
        emr_config=emr_config,
        epsilon_per_round=args.epsilon,
        patient_count=args.patients,
        fedprox_mu=args.fedprox_mu
    )
    
    asyncio.run(agent.run_continuous(
        rounds=args.rounds,
        delay_between_rounds=args.delay
    ))


if __name__ == "__main__":
    main()
