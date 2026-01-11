"""
Module: ghost_agent/main.py
DPDP §: 8(2)(a) - Data residency compliance, §11(3) - Consent
Description: Main Ghost Agent for hospital-side federated learning
API: POST /update, GET /status, GET /privacy_report
"""

import asyncio
import logging
import signal
import sys
from typing import Dict, Any, Optional
from datetime import datetime
import json
import os

import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Ghost Agent components
from .emr_wrapper import EMRWrapper
from .local_training import LocalTrainer
from .privacy_engine import PrivacyEngine
from .ghost_pack import GhostPack

# Configuration
from config import config


class SimpleNN(nn.Module):
    """Simple neural network for medical data."""
    
    def __init__(self, input_size: int = 784, hidden_size: int = 256, num_classes: int = 10):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class GhostAgent:
    """
    Ghost Agent - Hospital-side federated learning agent.
    
    Implements:
    - EMR data loading and preprocessing
    - Local training with privacy preservation
    - Secure model update packaging
    - DPDP compliance monitoring
    """
    
    def __init__(
        self,
        hospital_id: str,
        agent_port: int = 8001,
        sna_url: str = "http://localhost:8000"
    ):
        """
        Initialize Ghost Agent.
        
        Args:
            hospital_id: Unique hospital identifier
            agent_port: Port for agent API
            sna_url: SNA (Secure National Aggregator) URL
        """
        self.hospital_id = hospital_id
        self.agent_port = agent_port
        self.sna_url = sna_url
        
        # Initialize components
        self.emr_wrapper = EMRWrapper(hospital_id=hospital_id)
        self.ghost_pack = GhostPack(hospital_id=hospital_id)
        
        # Initialize model
        self.model = SimpleNN(
            input_size=config.MODEL_INPUT_SIZE,
            hidden_size=config.MODEL_HIDDEN_SIZE,
            num_classes=config.MODEL_OUTPUT_SIZE
        )
        
        # Initialize local trainer
        self.local_trainer = LocalTrainer(
            hospital_id=hospital_id,
            model=self.model,
            algorithm=config.ALGORITHM_CONFIGS["fedavg"]["name"].lower(),
            learning_rate=config.LEARNING_RATE,
            batch_size=config.BATCH_SIZE,
            local_epochs=config.LOCAL_EPOCHS
        )
        
        # Training state
        self.is_training = False
        self.current_round = 0
        self.training_stats = {}
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title=f"Ghost Agent - {hospital_id}",
            description="DPDP-Safe Federated Learning Agent",
            version="1.0.0"
        )
        
        # Configure CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging for Ghost Agent."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(f"GhostAgent-{self.hospital_id}")
        
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/")
        async def root():
            return {
                "agent": "Ghost Protocol Agent",
                "hospital_id": self.hospital_id,
                "status": "active",
                "dpdp_compliant": True
            }
            
        @self.app.get("/health")
        async def health():
            return {
                "status": "healthy",
                "hospital_id": self.hospital_id,
                "training": self.is_training,
                "privacy_compliant": True
            }
            
        @self.app.post("/load_data")
        async def load_data(data_config: Dict[str, Any]):
            """Load EMR data for training."""
            try:
                # Load data based on configuration
                data_type = data_config.get("type", "csv")
                
                if data_type == "csv":
                    csv_path = data_config.get("path", "/app/data/medical_data.csv")
                    features, targets, feature_names = self.emr_wrapper.load_csv_data(
                        csv_path,
                        feature_columns=data_config.get("features"),
                        target_column=data_config.get("target")
                    )
                elif data_type == "fhir":
                    # Load FHIR resources
                    resources = []
                    for resource_path in data_config.get("resources", []):
                        resource = self.emr_wrapper.load_fhir_resource(resource_path)
                        resources.append(resource)
                        
                    features, feature_names = self.emr_wrapper.fhir_to_numpy(
                        resources,
                        data_config.get("feature_mapping")
                    )
                    targets = None  # FHIR may not have targets
                else:
                    raise ValueError(f"Unknown data type: {data_type}")
                    
                # Store data
                self.features = features
                self.targets = targets
                self.feature_names = feature_names
                
                # Prepare data loaders
                self.train_loader, self.val_loader = self.local_trainer.prepare_data(
                    features, targets
                )
                
                return {
                    "status": "success",
                    "samples": len(features),
                    "features": len(feature_names),
                    "feature_names": feature_names
                }
                
            except Exception as e:
                self.logger.error(f"Failed to load data: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.post("/train_round")
        async def train_round(round_config: Dict[str, Any]):
            """Execute one round of local training."""
            if not hasattr(self, 'train_loader'):
                raise HTTPException(status_code=400, detail="No training data loaded")
                
            try:
                self.is_training = True
                round_num = round_config.get("round_num", self.current_round)
                
                # Get global weights if available
                global_weights = round_config.get("global_weights")
                if global_weights:
                    # Convert back to tensors
                    global_weights_tensors = {}
                    for name, weight_list in global_weights.items():
                        global_weights_tensors[name] = torch.FloatTensor(weight_list)
                    
                    self.local_trainer.set_model_weights(global_weights_tensors)
                    
                # Train locally
                local_weights, training_stats = self.local_trainer.local_train_round(
                    self.train_loader,
                    global_weights=global_weights_tensors if global_weights else None,
                    round_num=round_num
                )
                
                # Update current round
                self.current_round = round_num + 1
                self.training_stats = training_stats
                
                # Convert weights to lists for JSON serialization
                serializable_weights = {}
                for name, weight_tensor in local_weights.items():
                    serializable_weights[name] = weight_tensor.tolist()
                    
                # Create secure package
                metadata = self.ghost_pack.create_secure_metadata(
                    model_version=f"round_{round_num}",
                    training_stats=training_stats,
                    privacy_stats=training_stats.get("privacy_report", {})
                )
                
                # Pack model update
                secure_package = self.ghost_pack.pack_model_update(
                    serializable_weights,
                    metadata=metadata,
                    compress=True
                )
                
                self.is_training = False
                
                return {
                    "status": "success",
                    "round": round_num,
                    "weights": serializable_weights,  # For demo, in production send secure_package
                    "training_stats": training_stats,
                    "package_hash": self.ghost_pack.compute_integrity_hash(serializable_weights)
                }
                
            except Exception as e:
                self.is_training = False
                self.logger.error(f"Training failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.get("/status")
        async def get_status():
            """Get agent status."""
            privacy_report = self.local_trainer.privacy_engine.get_privacy_report()
            
            return {
                "hospital_id": self.hospital_id,
                "status": "training" if self.is_training else "idle",
                "current_round": self.current_round,
                "dpdp_compliant": privacy_report["dp_compliance_status"] == "COMPLIANT",
                "epsilon_spent": privacy_report["current_epsilon_spent"],
                "max_epsilon": privacy_report["max_epsilon_allowed"],
                "training_stats": self.training_stats
            }
            
        @self.app.get("/privacy_report")
        async def get_privacy_report():
            """Get detailed privacy report."""
            return self.local_trainer.privacy_engine.get_privacy_report()
            
        @self.app.get("/model_info")
        async def get_model_info():
            """Get model information."""
            return {
                "model_type": self.model.__class__.__name__,
                "parameters": sum(p.numel() for p in self.model.parameters()),
                "input_size": config.MODEL_INPUT_SIZE,
                "hidden_size": config.MODEL_HIDDEN_SIZE,
                "output_size": config.MODEL_OUTPUT_SIZE
            }
            
        @self.app.post("/update_global_weights")
        async def update_global_weights(weights_update: Dict[str, Any]):
            """Update model with new global weights."""
            try:
                # Convert weights to tensors
                global_weights = {}
                for name, weight_list in weights_update.get("weights", {}).items():
                    global_weights[name] = torch.FloatTensor(weight_list)
                    
                # Update model
                self.local_trainer.set_model_weights(global_weights)
                
                return {
                    "status": "success",
                    "weights_updated": len(global_weights)
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
                
    def run(self, host: str = "0.0.0.0", port: Optional[int] = None):
        """Run the Ghost Agent."""
        port = port or self.agent_port
        
        self.logger.info(f"Starting Ghost Agent for {self.hospital_id} on {host}:{port}")
        
        # Setup signal handlers
        def signal_handler(sig, frame):
            self.logger.info("Shutting down Ghost Agent...")
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Run server
        uvicorn.run(self.app, host=host, port=port)
        
    def load_sample_data(self):
        """Load sample medical data for testing."""
        # Generate synthetic medical data
        np.random.seed(42)
        n_samples = 1000
        n_features = config.MODEL_INPUT_SIZE
        
        # Simulate medical features
        features = np.random.randn(n_samples, n_features)
        
        # Simulate targets (binary classification)
        targets = np.random.randint(0, 2, n_samples)
        
        # Store data
        self.features = features
        self.targets = targets
        self.feature_names = [f"feature_{i}" for i in range(n_features)]
        
        # Prepare data loaders
        self.train_loader, self.val_loader = self.local_trainer.prepare_data(
            features, targets
        )
        
        print(f"Loaded sample data: {n_samples} samples, {n_features} features")


def main():
    """Main entry point for Ghost Agent."""
    # Get configuration from environment
    hospital_id = os.getenv("HOSPITAL_ID", "H001")
    agent_port = int(os.getenv("AGENT_PORT", "8001"))
    sna_url = os.getenv("SNA_URL", "http://localhost:8000")
    
    # Create and run Ghost Agent
    agent = GhostAgent(
        hospital_id=hospital_id,
        agent_port=agent_port,
        sna_url=sna_url
    )
    
    # Load sample data for demo
    agent.load_sample_data()
    
    # Run agent
    agent.run()


if __name__ == "__main__":
    main()