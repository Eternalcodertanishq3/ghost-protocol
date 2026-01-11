"""
Synthetic Health Data Generator for Ghost Protocol
Generates realistic diabetes prediction dataset for FL training

DPDP §: Uses synthetic data - no real patient data involved
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import random


def generate_synthetic_diabetes_data(
    n_samples: int = 1000,
    hospital_id: str = "H001",
    seed: int = None
) -> pd.DataFrame:
    """
    Generate synthetic diabetes prediction dataset
    
    Features based on Pima Indians Diabetes Dataset structure:
    - Pregnancies: Number of times pregnant
    - Glucose: Plasma glucose concentration
    - BloodPressure: Diastolic blood pressure (mm Hg)
    - SkinThickness: Triceps skin fold thickness (mm)
    - Insulin: 2-Hour serum insulin (mu U/ml)
    - BMI: Body mass index (weight in kg/(height in m)^2)
    - DiabetesPedigreeFunction: Diabetes pedigree function
    - Age: Age (years)
    - Outcome: Class variable (0 or 1) - diabetes diagnosis
    
    Args:
        n_samples: Number of samples to generate
        hospital_id: Hospital identifier for tracking
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic patient records
    """
    
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Generate features with realistic distributions
    data = {
        'patient_id': [f"{hospital_id}_P{i:05d}" for i in range(n_samples)],
        'hospital_id': [hospital_id] * n_samples,
        
        # Pregnancies: 0-17, higher for older patients
        'pregnancies': np.clip(np.random.poisson(3.8, n_samples), 0, 17),
        
        # Glucose: 70-200 mg/dL, normally distributed around 120
        'glucose': np.clip(np.random.normal(120, 30, n_samples), 70, 200).astype(int),
        
        # Blood Pressure: 40-130 mm Hg
        'blood_pressure': np.clip(np.random.normal(72, 12, n_samples), 40, 130).astype(int),
        
        # Skin Thickness: 10-60 mm
        'skin_thickness': np.clip(np.random.normal(29, 10, n_samples), 10, 60).astype(int),
        
        # Insulin: 14-846 mu U/ml
        'insulin': np.clip(np.random.exponential(80, n_samples) + 14, 14, 846).astype(int),
        
        # BMI: 18-67 kg/m^2
        'bmi': np.clip(np.random.normal(32, 7, n_samples), 18, 67).round(1),
        
        # Diabetes Pedigree Function: 0.08-2.42
        'diabetes_pedigree': np.clip(np.random.exponential(0.47, n_samples) + 0.08, 0.08, 2.42).round(3),
        
        # Age: 21-81 years
        'age': np.clip(np.random.normal(33, 12, n_samples), 21, 81).astype(int),
    }
    
    df = pd.DataFrame(data)
    
    # Generate outcome based on risk factors (realistic correlation)
    # Higher glucose, BMI, age, and pedigree function increase diabetes risk
    risk_score = (
        0.02 * (df['glucose'] - 100) +
        0.03 * (df['bmi'] - 25) +
        0.5 * df['diabetes_pedigree'] +
        0.02 * (df['age'] - 30) +
        0.05 * df['pregnancies'] +
        np.random.normal(0, 0.5, n_samples)  # Random noise
    )
    
    # Convert to probability using sigmoid
    probability = 1 / (1 + np.exp(-risk_score))
    
    # Generate outcome based on probability
    df['outcome'] = (probability > np.random.random(n_samples)).astype(int)
    
    # Add timestamp for audit trail
    df['generated_at'] = pd.Timestamp.now().isoformat()
    
    # Add consent flag (all synthetic data has consent)
    df['consent_given'] = True
    df['consent_purpose'] = 'federated_learning_research'
    
    return df


def generate_feature_tensor(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert DataFrame to feature tensor for model training
    
    Args:
        df: DataFrame with synthetic patient records
        
    Returns:
        Tuple of (features, labels) as numpy arrays
    """
    
    feature_columns = [
        'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
        'insulin', 'bmi', 'diabetes_pedigree', 'age'
    ]
    
    features = df[feature_columns].values.astype(np.float32)
    labels = df['outcome'].values.astype(np.float32)
    
    # Normalize features (z-score normalization)
    means = features.mean(axis=0)
    stds = features.std(axis=0) + 1e-8  # Avoid division by zero
    features_normalized = (features - means) / stds
    
    return features_normalized, labels


def generate_hospital_datasets(
    hospital_ids: List[str],
    samples_per_hospital: int = 1000,
    heterogeneous: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Generate datasets for multiple hospitals with optional heterogeneity
    
    Args:
        hospital_ids: List of hospital identifiers
        samples_per_hospital: Number of samples per hospital
        heterogeneous: If True, hospitals have different data distributions
        
    Returns:
        Dictionary mapping hospital_id to DataFrame
    """
    
    datasets = {}
    
    for i, hospital_id in enumerate(hospital_ids):
        # Create heterogeneous data distributions across hospitals
        if heterogeneous:
            # Each hospital has slightly different patient demographics
            seed = 42 + i * 100  # Reproducible but different
            n_samples = samples_per_hospital + random.randint(-200, 200)
        else:
            seed = 42
            n_samples = samples_per_hospital
        
        datasets[hospital_id] = generate_synthetic_diabetes_data(
            n_samples=max(500, n_samples),
            hospital_id=hospital_id,
            seed=seed
        )
        
        print(f"Generated {len(datasets[hospital_id])} records for {hospital_id}")
    
    return datasets


if __name__ == "__main__":
    # Demo: Generate data for 5 hospitals
    hospital_ids = ["AIIMS_Delhi", "Fortis_Mumbai", "Apollo_Chennai", "CMC_Vellore", "PGIMER_Chandigarh"]
    
    datasets = generate_hospital_datasets(hospital_ids, samples_per_hospital=1000)
    
    # Print statistics
    print("\n=== Dataset Statistics ===")
    for hospital_id, df in datasets.items():
        diabetes_rate = df['outcome'].mean() * 100
        print(f"{hospital_id}: {len(df)} patients, {diabetes_rate:.1f}% diabetes rate")
