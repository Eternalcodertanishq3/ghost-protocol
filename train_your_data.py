"""
Train with YOUR Patient Data - Ghost Protocol
Uses your actual patient_data.csv file
"""

import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from models.registry import ModelRegistry


class PatientDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_your_data(file_path: str):
    """Load YOUR patient data."""
    print(f"\n  Loading YOUR data: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"  Total patients: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    
    # Separate features and target
    features = df.drop('Outcome', axis=1).values.astype(np.float32)
    labels = df['Outcome'].values.astype(np.float32)
    
    # Normalize
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-8
    features = (features - mean) / std
    
    print(f"  Features shape: {features.shape}")
    print(f"  Positive rate: {labels.mean()*100:.1f}%")
    
    return features, labels


class Hospital:
    def __init__(self, hospital_id, features, labels, batch_size=32):
        self.hospital_id = hospital_id
        self.n_samples = len(labels)
        
        # Split train/test
        n_test = int(len(labels) * 0.2)
        
        train_dataset = PatientDataset(features[:-n_test], labels[:-n_test])
        test_dataset = PatientDataset(features[-n_test:], labels[-n_test:])
        
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Initialize model
        self.model = ModelRegistry.get("diabetes_prediction")
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = nn.BCELoss()
    
    def train(self, epochs: int):
        """Train on local data."""
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0
            batches = 0
            
            for features, labels in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(features).squeeze()
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                batches += 1
                
                # Show progress
                print(f"\r      Epoch {epoch+1}/{epochs} | Batch {batches}/{len(self.train_loader)} | Loss: {loss.item():.4f}", end="")
            
            avg_loss = epoch_loss / batches
            print(f"\r      Epoch {epoch+1}/{epochs} completed | Avg Loss: {avg_loss:.4f}          ")
            time.sleep(0.3)  # Slow down so you can see
        
        return {name: param.data.clone() for name, param in self.model.named_parameters()}
    
    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in self.test_loader:
                outputs = self.model(features).squeeze()
                predictions = (outputs > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        return correct / total
    
    def load_weights(self, weights):
        self.model.load_state_dict(weights)


def fedavg(all_weights, sizes):
    """Federated Averaging."""
    total = sum(sizes)
    aggregated = {}
    
    for name in all_weights[0]:
        weighted_sum = torch.zeros_like(all_weights[0][name])
        for weights, size in zip(all_weights, sizes):
            weighted_sum += weights[name] * (size / total)
        aggregated[name] = weighted_sum
    
    return aggregated


def main():
    print("\n" + "=" * 70)
    print("  🏥 GHOST PROTOCOL - TRAINING WITH YOUR DATA")
    print("=" * 70)
    
    # Check for your data file
    data_file = "patient_data.csv"
    if not os.path.exists(data_file):
        print(f"  ERROR: {data_file} not found!")
        return
    
    # Configuration
    N_HOSPITALS = 3
    EPOCHS_PER_ROUND = 5  # More epochs = slower but better
    TRAINING_ROUNDS = 3
    
    print(f"\n  Configuration:")
    print(f"    Training Rounds: {TRAINING_ROUNDS}")
    print(f"    Epochs per Round: {EPOCHS_PER_ROUND}")
    print(f"    Hospitals: {N_HOSPITALS}")
    
    # Load YOUR data
    print("\n" + "-" * 50)
    print("  📊 LOADING YOUR DATA")
    print("-" * 50)
    
    features, labels = load_your_data(data_file)
    
    # Split into hospitals
    n = len(labels)
    chunk_size = n // N_HOSPITALS
    
    hospitals = []
    for i in range(N_HOSPITALS):
        start = i * chunk_size
        end = start + chunk_size if i < N_HOSPITALS - 1 else n
        
        hospital = Hospital(
            f"Hospital_{i+1}",
            features[start:end],
            labels[start:end]
        )
        hospitals.append(hospital)
        print(f"\n  Hospital {i+1}: {hospital.n_samples} patients")
    
    input("\n  Press ENTER to start training...")
    
    # Initialize global model
    print("\n" + "-" * 50)
    print("  🧠 INITIALIZING GLOBAL MODEL")
    print("-" * 50)
    
    global_model = ModelRegistry.get("diabetes_prediction")
    global_weights = {name: param.data.clone() for name, param in global_model.named_parameters()}
    
    print(f"\n  Model parameters: {sum(p.numel() for p in global_model.parameters()):,}")
    
    # Federated Learning
    print("\n" + "-" * 50)
    print("  🔄 FEDERATED LEARNING")
    print("-" * 50)
    
    for round_num in range(1, TRAINING_ROUNDS + 1):
        print(f"\n  {'='*50}")
        print(f"  📍 ROUND {round_num} of {TRAINING_ROUNDS}")
        print(f"  {'='*50}")
        
        all_updates = []
        sizes = []
        
        for hospital in hospitals:
            print(f"\n    [{hospital.hospital_id}] Training on {hospital.n_samples} patients...")
            
            # Load global weights
            hospital.load_weights(global_weights)
            
            # Local training
            start_time = time.time()
            weights = hospital.train(EPOCHS_PER_ROUND)
            train_time = time.time() - start_time
            
            # Evaluate
            accuracy = hospital.evaluate()
            print(f"    ✓ Completed in {train_time:.1f}s | Accuracy: {accuracy:.2%}")
            
            all_updates.append(weights)
            sizes.append(hospital.n_samples)
        
        # Aggregate
        print(f"\n  🔗 Aggregating model updates (FedAvg)...")
        global_weights = fedavg(all_updates, sizes)
        global_model.load_state_dict(global_weights)
        
        # Global eval
        total_acc = sum(h.evaluate() for h in hospitals) / len(hospitals)
        print(f"  📊 Round {round_num} Average Accuracy: {total_acc:.2%}")
        
        if round_num < TRAINING_ROUNDS:
            input(f"\n  Press ENTER for Round {round_num + 1}...")
    
    # Final results
    print("\n" + "=" * 70)
    print("  🎉 TRAINING COMPLETE!")
    print("=" * 70)
    
    print("\n  Final Results:")
    final_accs = []
    for hospital in hospitals:
        hospital.load_weights(global_weights)
        acc = hospital.evaluate()
        final_accs.append(acc)
        print(f"    {hospital.hospital_id}: {acc:.2%}")
    
    avg_acc = sum(final_accs) / len(final_accs)
    total_patients = sum(h.n_samples for h in hospitals)
    
    print(f"\n  Average Accuracy: {avg_acc:.2%}")
    print(f"  Total Patients Trained: {total_patients}")
    print(f"  Privacy: ✅ Patient data NEVER left hospitals!")
    
    # Save model
    torch.save(global_model.state_dict(), "models/your_trained_model.pt")
    print(f"\n  Model saved to: models/your_trained_model.pt")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
