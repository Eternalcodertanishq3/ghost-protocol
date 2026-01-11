"""
Synthetic Patient Data Generator for Ghost Protocol Testing
Generates realistic diabetes prediction datasets for federated learning simulation.
"""

import numpy as np
import pandas as pd
import os
from typing import List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader


class DiabetesDataset(Dataset):
    """PyTorch Dataset for diabetes prediction."""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def generate_synthetic_diabetes_data(
    n_samples: int = 1000,
    noise_level: float = 0.1,
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic diabetes patient data.
    
    Features (8 total):
    1. Pregnancies (0-17)
    2. Glucose (0-200 mg/dL)
    3. Blood Pressure (0-130 mm Hg)
    4. Skin Thickness (0-100 mm)
    5. Insulin (0-900 mu U/ml)
    6. BMI (0-70)
    7. Diabetes Pedigree Function (0-2.5)
    8. Age (21-81)
    
    Returns:
        features: (n_samples, 8) array
        labels: (n_samples,) binary array (0=no diabetes, 1=diabetes)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate realistic feature distributions
    pregnancies = np.random.randint(0, 18, n_samples)
    glucose = np.random.normal(120, 30, n_samples).clip(44, 200)
    blood_pressure = np.random.normal(72, 12, n_samples).clip(24, 130)
    skin_thickness = np.random.normal(29, 10, n_samples).clip(0, 100)
    insulin = np.random.exponential(80, n_samples).clip(0, 900)
    bmi = np.random.normal(32, 8, n_samples).clip(18, 70)
    diabetes_pedigree = np.random.exponential(0.5, n_samples).clip(0.08, 2.5)
    age = np.random.normal(33, 12, n_samples).clip(21, 81).astype(int)
    
    # Combine features
    features = np.column_stack([
        pregnancies, glucose, blood_pressure, skin_thickness,
        insulin, bmi, diabetes_pedigree, age
    ])
    
    # Normalize features
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
    
    # Generate labels based on realistic risk factors
    risk_score = (
        0.3 * (glucose > 140).astype(float) +
        0.25 * (bmi > 30).astype(float) +
        0.15 * (age > 45).astype(float) +
        0.1 * (pregnancies > 4).astype(float) +
        0.1 * (diabetes_pedigree > 0.5).astype(float) +
        0.1 * (blood_pressure > 80).astype(float) +
        noise_level * np.random.randn(n_samples)
    )
    
    labels = (risk_score > 0.5).astype(float)
    
    return features.astype(np.float32), labels.astype(np.float32)


def generate_hospital_datasets(
    n_hospitals: int = 3,
    samples_per_hospital: int = 500,
    heterogeneity: float = 0.3,
    output_dir: str = "data/hospitals"
) -> List[str]:
    """
    Generate heterogeneous datasets for multiple hospitals.
    
    Args:
        n_hospitals: Number of hospitals
        samples_per_hospital: Samples per hospital
        heterogeneity: How different each hospital's data distribution is (0-1)
        output_dir: Directory to save datasets
        
    Returns:
        List of dataset file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    file_paths = []
    
    print(f"\n{'='*60}")
    print("GENERATING SYNTHETIC PATIENT DATA")
    print(f"{'='*60}")
    
    for i in range(n_hospitals):
        # Add hospital-specific bias to simulate non-IID data
        seed = 42 + i * 1000
        np.random.seed(seed)
        
        # Generate base data
        features, labels = generate_synthetic_diabetes_data(
            n_samples=samples_per_hospital,
            noise_level=0.1 + heterogeneity * (i / n_hospitals),
            seed=seed
        )
        
        # Add hospital-specific feature bias (non-IID)
        if heterogeneity > 0:
            bias = np.random.randn(8) * heterogeneity * 0.5
            features = features + bias
        
        # Create DataFrame
        df = pd.DataFrame(features, columns=[
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigree', 'Age'
        ])
        df['Outcome'] = labels
        
        # Save to CSV
        file_path = os.path.join(output_dir, f"hospital_{i+1}_data.csv")
        df.to_csv(file_path, index=False)
        file_paths.append(file_path)
        
        # Print stats
        positive_rate = labels.mean() * 100
        print(f"\nHospital {i+1}:")
        print(f"  File: {file_path}")
        print(f"  Samples: {samples_per_hospital}")
        print(f"  Diabetes Rate: {positive_rate:.1f}%")
    
    print(f"\n{'='*60}")
    print(f"Generated {n_hospitals} hospital datasets")
    print(f"Total patients: {n_hospitals * samples_per_hospital}")
    print(f"{'='*60}\n")
    
    return file_paths


def create_data_loaders(
    file_paths: List[str],
    batch_size: int = 32,
    test_split: float = 0.2
) -> List[Tuple[DataLoader, DataLoader]]:
    """
    Create train/test DataLoaders for each hospital.
    
    Returns:
        List of (train_loader, test_loader) tuples
    """
    loaders = []
    
    for path in file_paths:
        df = pd.read_csv(path)
        
        features = df.drop('Outcome', axis=1).values.astype(np.float32)
        labels = df['Outcome'].values.astype(np.float32)
        
        # Split into train/test
        n_test = int(len(labels) * test_split)
        
        train_features, test_features = features[:-n_test], features[-n_test:]
        train_labels, test_labels = labels[:-n_test], labels[-n_test:]
        
        # Create datasets
        train_dataset = DiabetesDataset(train_features, train_labels)
        test_dataset = DiabetesDataset(test_features, test_labels)
        
        # Create loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        loaders.append((train_loader, test_loader))
    
    return loaders


if __name__ == "__main__":
    # Generate sample data for 3 hospitals
    file_paths = generate_hospital_datasets(
        n_hospitals=3,
        samples_per_hospital=500,
        heterogeneity=0.3
    )
    
    print("Synthetic data generation complete!")
    print("\nGenerated files:")
    for path in file_paths:
        print(f"  - {path}")
