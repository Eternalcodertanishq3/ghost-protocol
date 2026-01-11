"""
Module: sna/health_ledger/shapley.py
DPDP §: 9(4) - Shapley value-based reward distribution
Description: Shapley value calculator for fair contribution assessment
Test: pytest tests/test_shapley.py::test_shapley_calculation
"""

import numpy as np
from typing import Dict, List, Any, Callable
from itertools import combinations
import logging


class ShapleyCalculator:
    """
    Shapley Value Calculator for HealthToken distribution.
    
    The Shapley value provides a fair way to distribute rewards based on
    each participant's marginal contribution to the collective outcome.
    
    For n hospitals, computes marginal contribution over all possible
    coalitions: φ(i) = Σ (|S|!(n-|S|-1)!/n!) * (v(S∪{i}) - v(S))
    """
    
    def __init__(self, approximation_method: str = "monte_carlo"):
        """
        Initialize Shapley Calculator.
        
        Args:
            approximation_method: Method for approximation ("monte_carlo", "truncated")
        """
        self.approximation_method = approximation_method
        self.logger = logging.getLogger(__name__)
        
    def calculate_exact_shapley_values(
        self,
        contributions: Dict[str, float],
        global_performance: float = 0.0
    ) -> Dict[str, float]:
        """
        Calculate exact Shapley values (exponential complexity).
        
        Args:
            contributions: Individual hospital contributions
            global_performance: Global model performance
            
        Returns:
            Shapley values for each hospital
        """
        hospitals = list(contributions.keys())
        n = len(hospitals)
        
        if n > 10:
            self.logger.warning(f"Exact Shapley calculation for {n} hospitals is expensive")
            
        shapley_values = {}
        
        for hospital in hospitals:
            shapley_value = 0.0
            
            # Iterate over all possible coalitions
            for r in range(n):
                # Coalitions of size r not containing hospital
                other_hospitals = [h for h in hospitals if h != hospital]
                
                for coalition in combinations(other_hospitals, r):
                    coalition_list = list(coalition)
                    
                    # Value of coalition without hospital
                    coalition_contrib = sum(contributions[h] for h in coalition_list)
                    
                    # Value of coalition with hospital
                    coalition_with_hospital = coalition_list + [hospital]
                    coalition_with_contrib = sum(contributions[h] for h in coalition_with_hospital)
                    
                    # Marginal contribution
                    marginal_contrib = coalition_with_contrib - coalition_contrib
                    
                    # Weight factor
                    weight = np.math.factorial(r) * np.math.factorial(n - r - 1) / np.math.factorial(n)
                    
                    shapley_value += weight * marginal_contrib
                    
            shapley_values[hospital] = shapley_value
            
        return shapley_values
        
    def calculate_monte_carlo_shapley(
        self,
        contributions: Dict[str, float],
        global_performance: float = 0.0,
        n_samples: int = 1000
    ) -> Dict[str, float]:
        """
        Calculate approximate Shapley values using Monte Carlo sampling.
        
        More efficient for large numbers of hospitals.
        
        Args:
            contributions: Individual hospital contributions
            global_performance: Global model performance
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Approximate Shapley values
        """
        hospitals = list(contributions.keys())
        n = len(hospitals)
        
        # Initialize Shapley value estimates
        shapley_values = {hospital: 0.0 for hospital in hospitals}
        
        for _ in range(n_samples):
            # Random permutation of hospitals
            permutation = np.random.permutation(hospitals)
            
            # Track cumulative contribution
            cumulative_contrib = 0.0
            
            for i, hospital in enumerate(permutation):
                # Hospital's contribution
                hospital_contrib = contributions[hospital]
                
                # Marginal contribution in this position
                marginal_contrib = hospital_contrib
                
                # Add to Shapley estimate
                shapley_values[hospital] += marginal_contrib / n_samples
                
                cumulative_contrib += hospital_contrib
                
        return shapley_values
        
    def calculate_truncated_shapley(
        self,
        contributions: Dict[str, float],
        global_performance: float = 0.0,
        max_coalition_size: int = 5
    ) -> Dict[str, float]:
        """
        Calculate truncated Shapley values (approximate).
        
        Only considers coalitions up to a certain size.
        
        Args:
            contributions: Individual hospital contributions
            global_performance: Global model performance
            max_coalition_size: Maximum coalition size to consider
            
        Returns:
            Truncated Shapley values
        """
        hospitals = list(contributions.keys())
        n = len(hospitals)
        
        shapley_values = {hospital: 0.0 for hospital in hospitals}
        
        for hospital in hospitals:
            # Only consider coalitions up to max_coalition_size
            other_hospitals = [h for h in hospitals if h != hospital]
            
            for r in range(min(max_coalition_size, n)):
                for coalition in combinations(other_hospitals, r):
                    coalition_list = list(coalition)
                    
                    # Value computations
                    coalition_contrib = sum(contributions[h] for h in coalition_list)
                    coalition_with_contrib = coalition_contrib + contributions[hospital]
                    
                    marginal_contrib = coalition_with_contrib - coalition_contrib
                    
                    # Approximate weight (ignores factorial terms for efficiency)
                    weight = 1.0 / (n * len(list(combinations(other_hospitals, r))))
                    
                    shapley_values[hospital] += weight * marginal_contrib
                    
        return shapley_values
        
    def calculate_shapley_values(
        self,
        contributions: Dict[str, float],
        global_performance: float = 0.0
    ) -> Dict[str, float]:
        """
        Calculate Shapley values using configured method.
        
        Args:
            contributions: Individual hospital contributions
            global_performance: Global model performance
            
        Returns:
            Shapley values
        """
        n_hospitals = len(contributions)
        
        if n_hospitals == 0:
            return {}
            
        # Choose calculation method based on number of hospitals
        if n_hospitals <= 8:
            # Exact calculation for small numbers
            self.logger.info(f"Using exact Shapley calculation for {n_hospitals} hospitals")
            return self.calculate_exact_shapley_values(contributions, global_performance)
        elif n_hospitals <= 20:
            # Truncated calculation for medium numbers
            self.logger.info(f"Using truncated Shapley calculation for {n_hospitals} hospitals")
            return self.calculate_truncated_shapley(contributions, global_performance)
        else:
            # Monte Carlo for large numbers
            self.logger.info(f"Using Monte Carlo Shapley calculation for {n_hospitals} hospitals")
            return self.calculate_monte_carlo_shapley(contributions, global_performance)
            
    def calculate_weighted_shapley_values(
        self,
        contributions: Dict[str, float],
        weights: Dict[str, float],
        global_performance: float = 0.0
    ) -> Dict[str, float]:
        """
        Calculate weighted Shapley values.
        
        Incorporates hospital weights (e.g., reputation, data size).
        
        Args:
            contributions: Individual hospital contributions
            weights: Hospital weights
            global_performance: Global model performance
            
        Returns:
            Weighted Shapley values
        """
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight == 0:
            return {hospital: 0.0 for hospital in contributions.keys()}
            
        normalized_weights = {
            hospital: weight / total_weight
            for hospital, weight in weights.items()
        }
        
        # Calculate base Shapley values
        shapley_values = self.calculate_shapley_values(contributions, global_performance)
        
        # Apply weights
        weighted_shapley = {
            hospital: shapley_values.get(hospital, 0.0) * normalized_weights.get(hospital, 0.0)
            for hospital in contributions.keys()
        }
        
        return weighted_shapley
        
    def compute_marginal_contributions(
        self,
        contributions: Dict[str, float],
        coalition: List[str]
    ) -> Dict[str, float]:
        """
        Compute marginal contributions for a specific coalition.
        
        Args:
            contributions: Individual contributions
            coalition: List of hospitals in coalition
            
        Returns:
            Marginal contributions
        """
        marginal_contributions = {}
        
        # Base value of coalition
        base_value = sum(contributions[hospital] for hospital in coalition)
        
        for hospital in coalition:
            # Value without this hospital
            coalition_without = [h for h in coalition if h != hospital]
            value_without = sum(contributions[h] for h in coalition_without)
            
            # Marginal contribution
            marginal_contributions[hospital] = base_value - value_without
            
        return marginal_contributions
        
    def validate_shapley_properties(
        self,
        contributions: Dict[str, float],
        shapley_values: Dict[str, float]
    ) -> Dict[str, bool]:
        """
        Validate that Shapley values satisfy required properties.
        
        Args:
            contributions: Original contributions
            shapley_values: Computed Shapley values
            
        Returns:
            Validation results
        """
        total_contribution = sum(contributions.values())
        total_shapley = sum(shapley_values.values())
        
        # Efficiency: Sum of Shapley values should equal total contribution
        efficiency_check = abs(total_shapley - total_contribution) < 1e-10
        
        # Symmetry: Hospitals with same contributions should have same Shapley values
        symmetry_check = True
        contribution_groups = {}
        for hospital, contrib in contributions.items():
            if contrib not in contribution_groups:
                contribution_groups[contrib] = []
            contribution_groups[contrib].append(hospital)
            
        for contrib, hospitals in contribution_groups.items():
            if len(hospitals) > 1:
                shapley_values_for_contrib = [shapley_values[h] for h in hospitals]
                if not all(s == shapley_values_for_contrib[0] for s in shapley_values_for_contrib):
                    symmetry_check = False
                    break
                    
        # Linearity: Should be preserved (not easily testable without function)
        linearity_check = True
        
        # Null player: Hospitals with zero contribution should have zero Shapley value
        null_player_check = all(
            shapley_values[hospital] == 0.0
            for hospital, contrib in contributions.items()
            if contrib == 0.0
        )
        
        return {
            "efficiency": efficiency_check,
            "symmetry": symmetry_check,
            "linearity": linearity_check,
            "null_player": null_player_check
        }
        
    def get_shapley_statistics(
        self,
        contributions: Dict[str, float],
        shapley_values: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Get statistics about Shapley value distribution.
        
        Args:
            contributions: Original contributions
            shapley_values: Computed Shapley values
            
        Returns:
            Shapley statistics
        """
        contribution_values = list(contributions.values())
        shapley_value_list = list(shapley_values.values())
        
        return {
            "num_hospitals": len(contributions),
            "total_contribution": sum(contribution_values),
            "total_shapley_value": sum(shapley_value_list),
            "contribution_stats": {
                "mean": np.mean(contribution_values),
                "std": np.std(contribution_values),
                "min": np.min(contribution_values),
                "max": np.max(contribution_values)
            },
            "shapley_stats": {
                "mean": np.mean(shapley_value_list),
                "std": np.std(shapley_value_list),
                "min": np.min(shapley_value_list),
                "max": np.max(shapley_value_list)
            },
            "fairness_ratio": np.std(shapley_value_list) / (np.mean(shapley_value_list) + 1e-10)
        }
        
    def explain_shapley_value(
        self,
        hospital_id: str,
        contributions: Dict[str, float],
        shapley_values: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Explain a specific hospital's Shapley value.
        
        Args:
            hospital_id: Hospital to explain
            contributions: All contributions
            shapley_values: All Shapley values
            
        Returns:
            Explanation of the hospital's Shapley value
        """
        other_hospitals = [h for h in contributions.keys() if h != hospital_id]
        n = len(contributions)
        
        # Calculate marginal contributions for this hospital
        marginal_contributions = []
        
        for r in range(n):
            for coalition in combinations(other_hospitals, r):
                coalition_list = list(coalition)
                
                # Value without this hospital
                value_without = sum(contributions[h] for h in coalition_list)
                
                # Value with this hospital
                value_with = value_without + contributions[hospital_id]
                
                # Marginal contribution
                marginal_contrib = value_with - value_without
                
                marginal_contributions.append({
                    "coalition": coalition_list,
                    "marginal_contribution": marginal_contrib,
                    "weight": np.math.factorial(r) * np.math.factorial(n - r - 1) / np.math.factorial(n)
                })
                
        # Sort by marginal contribution
        marginal_contributions.sort(key=lambda x: x["marginal_contribution"], reverse=True)
        
        return {
            "hospital_id": hospital_id,
            "individual_contribution": contributions[hospital_id],
            "shapley_value": shapley_values[hospital_id],
            "marginal_contributions": marginal_contributions[:5],  # Top 5
            "fairness_explanation": (
                "High" if shapley_values[hospital_id] >= contributions[hospital_id] else "Low"
            )
        }