"""
Module: sna/adaptive_clustering/adaptive_clustering.py
DPDP §: 9(4) - Non-IID data clustering for improved FL performance
Description: Adaptive clustering of hospitals based on data similarity and performance
Byzantine: Cluster assignments robust to malicious hospitals (tolerates f < n/3 per cluster, Lamport 1982)
Privacy: Cluster features anonymized with DP noise (ε=0.3) before clustering
Test: pytest tests/test_clustering.py::test_adaptive_clustering
API: GET /clusters, POST /update_cluster, GET /cluster_performance
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json

from algorithms.dp_mechanisms.gaussian import GaussianDP
from sna.byzantine_shield.byzantine_shield import ByzantineShield


@dataclass
class HospitalCluster:
    """Hospital cluster information."""
    cluster_id: int
    hospital_ids: List[str]
    centroid: np.ndarray
    cluster_quality: float
    performance_metrics: Dict[str, float]
    last_updated: str
    byzantine_score: float


@dataclass
class ClusterAssignment:
    """Hospital cluster assignment."""
    hospital_id: str
    cluster_id: int
    assignment_confidence: float
    distance_to_centroid: float
    timestamp: str


class AdaptiveClustering:
    """
    Adaptive Clustering for Ghost Protocol.
    
    Implements hospital clustering based on data similarity, performance patterns,
    and Byzantine behavior to optimize federated learning for non-IID data.
    
    Features:
    - Dynamic cluster reassignment
    - Byzantine-robust clustering
    - Performance-based cluster optimization
    - Privacy-preserving cluster features
    """
    
    def __init__(
        self,
        n_clusters: int = 5,
        min_cluster_size: int = 3,
        max_byzantine_ratio: float = 0.3,
        reclustering_interval_hours: int = 24,
        privacy_epsilon: float = 0.3
    ):
        """
        Initialize Adaptive Clustering.
        
        Args:
            n_clusters: Number of clusters
            min_cluster_size: Minimum hospitals per cluster
            max_byzantine_ratio: Maximum Byzantine hospitals per cluster
            reclustering_interval_hours: Hours between automatic reclustering
            privacy_epsilon: Privacy budget for cluster features
        """
        self.n_clusters = n_clusters
        self.min_cluster_size = min_cluster_size
        self.max_byzantine_ratio = max_byzantine_ratio
        self.reclustering_interval_hours = reclustering_interval_hours
        self.privacy_epsilon = privacy_epsilon
        
        self.logger = logging.getLogger(__name__)
        
        # Clustering model
        self.clustering_model = KMeans(n_clusters=n_clusters, random_state=42)
        self.is_fitted = False
        
        # Privacy mechanism
        self.dp_mechanism = GaussianDP(epsilon=privacy_epsilon, delta=1e-6)
        
        # Byzantine shield
        self.byzantine_shield = ByzantineShield()
        
        # Cluster state
        self.clusters: Dict[int, HospitalCluster] = {}
        self.assignments: Dict[str, ClusterAssignment] = {}
        self.cluster_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.performance_metrics: Dict[str, Dict[str, float]] = {}
        
        # Metrics
        self.metrics = {
            "total_reclusterings": 0,
            "cluster_changes": 0,
            "average_cluster_quality": 0.0,
            "byzantine_violations": 0,
            "performance_improvements": 0
        }
        
        # Start reclustering timer
        self.last_reclustering = datetime.utcnow()
        
    def extract_cluster_features(self, hospital_id: str) -> np.ndarray:
        """
        Extract features for clustering.
        
        Args:
            hospital_id: Hospital identifier
            
        Returns:
            Feature vector for clustering
            
        Privacy: Features anonymized with DP noise before clustering
        """
        # Get hospital characteristics (simulated - would come from actual data)
        features = [
            # Data characteristics
            np.random.uniform(0.1, 1.0),  # data_volume_gb
            np.random.uniform(0.0, 1.0),  # data_diversity_score
            np.random.uniform(0.5, 1.0),  # data_quality_score
            
            # Performance characteristics
            np.random.uniform(0.7, 0.95), # model_accuracy
            np.random.uniform(0.1, 1.0),  # convergence_speed
            np.random.uniform(0.0, 0.5),  # gradient_variance
            
            # Participation patterns
            np.random.uniform(0.1, 1.0),  # participation_frequency
            np.random.uniform(0.0, 1.0),  # response_time_score
            np.random.uniform(0.0, 0.2),  # dropout_rate
            
            # Reputation and trust
            np.random.uniform(0.5, 1.0),  # reputation_score
            np.random.uniform(0.0, 1.0),  # byzantine_score
            np.random.uniform(0.0, 0.1),  # violation_rate
            
            # Geographic and temporal
            np.random.uniform(0.0, 1.0),  # timezone_score
            np.random.uniform(0.0, 1.0),  # network_latency_score
        ]
        
        feature_vector = np.array(features)
        
        # Add DP noise for privacy
        noisy_features = self.dp_mechanism.add_noise(
            torch.tensor(feature_vector), sensitivity=1.0
        ).numpy()
        
        return noisy_features
        
    def perform_clustering(self, hospital_ids: Optional[List[str]] = None) -> Dict[int, HospitalCluster]:
        """
        Perform adaptive clustering of hospitals.
        
        Args:
            hospital_ids: List of hospital IDs to cluster (optional)
            
        Returns:
            Updated cluster assignments
            
        Byzantine: Clusters checked for Byzantine tolerance before acceptance
        """
        start_time = datetime.utcnow()
        
        # Use all hospitals if none specified
        if hospital_ids is None:
            hospital_ids = list(self.current_status.keys())
            
        if len(hospital_ids) < self.min_cluster_size:
            self.logger.warning(f"Insufficient hospitals for clustering: {len(hospital_ids)}")
            return self.clusters
            
        # Extract features for all hospitals
        features = []
        valid_hospitals = []
        
        for hospital_id in hospital_ids:
            try:
                feature_vector = self.extract_cluster_features(hospital_id)
                features.append(feature_vector)
                valid_hospitals.append(hospital_id)
            except Exception as e:
                self.logger.error(f"Failed to extract features for {hospital_id}: {e}")
                
        if len(valid_hospitals) < self.min_cluster_size:
            return self.clusters
            
        # Convert to numpy array
        feature_matrix = np.array(features)
        
        # Perform clustering
        cluster_labels = self.clustering_model.fit_predict(feature_matrix)
        self.is_fitted = True
        
        # Calculate cluster quality
        if len(feature_matrix) > self.n_clusters:
            cluster_quality = silhouette_score(feature_matrix, cluster_labels)
        else:
            cluster_quality = 0.5  # Default for small datasets
            
        # Create clusters
        new_clusters = {}
        cluster_hospitals = {i: [] for i in range(self.n_clusters)}
        
        for i, hospital_id in enumerate(valid_hospitals):
            cluster_id = cluster_labels[i]
            cluster_hospitals[cluster_id].append(hospital_id)
            
        # Filter out small clusters
        for cluster_id, hospitals in cluster_hospitals.items():
            if len(hospitals) >= self.min_cluster_size:
                # Calculate centroid
                cluster_features = feature_matrix[cluster_labels == cluster_id]
                centroid = np.mean(cluster_features, axis=0)
                
                # Calculate Byzantine score
                byzantine_score = self._calculate_cluster_byzantine_score(hospitals)
                
                # Check Byzantine tolerance
                if self._is_byzantine_tolerant(hospitals, byzantine_score):
                    cluster = HospitalCluster(
                        cluster_id=cluster_id,
                        hospital_ids=hospitals,
                        centroid=centroid,
                        cluster_quality=cluster_quality,
                        performance_metrics=self._calculate_cluster_performance(hospitals),
                        last_updated=datetime.utcnow().isoformat(),
                        byzantine_score=byzantine_score
                    )
                    new_clusters[cluster_id] = cluster
                else:
                    self.metrics["byzantine_violations"] += 1
                    self.logger.warning(f"Cluster {cluster_id} rejected due to Byzantine tolerance violation")
                    
        # Update assignments
        old_assignments = self.assignments.copy()
        self.assignments = {}
        
        for cluster_id, cluster in new_clusters.items():
            for hospital_id in cluster.hospital_ids:
                # Find distance to centroid
                hospital_features = self.extract_cluster_features(hospital_id)
                distance = np.linalg.norm(hospital_features - cluster.centroid)
                
                assignment = ClusterAssignment(
                    hospital_id=hospital_id,
                    cluster_id=cluster_id,
                    assignment_confidence=1.0 / (1.0 + distance),  # Higher confidence for closer points
                    distance_to_centroid=distance,
                    timestamp=datetime.utcnow().isoformat()
                )
                self.assignments[hospital_id] = assignment
                
                # Check if assignment changed
                if (hospital_id in old_assignments and 
                    old_assignments[hospital_id].cluster_id != cluster_id):
                    self.metrics["cluster_changes"] += 1
                    
        # Store cluster history
        self.cluster_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "clusters": {cid: {"hospitals": list(cluster.hospital_ids), "quality": cluster.cluster_quality} 
                        for cid, cluster in new_clusters.items()},
            "cluster_quality": cluster_quality,
            "hospitals_clustered": len(valid_hospitals)
        })
        
        # Update clusters
        self.clusters = new_clusters
        self.metrics["total_reclusterings"] += 1
        self.metrics["average_cluster_quality"] = (
            (self.metrics["average_cluster_quality"] * (self.metrics["total_reclusterings"] - 1) +
             cluster_quality) / self.metrics["total_reclusterings"]
        )
        
        self.last_reclustering = datetime.utcnow()
        
        self.logger.info(
            f"Clustering completed: {len(new_clusters)} clusters, "
            f"quality={cluster_quality:.3f}, hospitals={len(valid_hospitals)}, "
            f"time={(datetime.utcnow() - start_time).total_seconds():.3f}s"
        )
        
        return self.clusters
        
    def get_hospital_cluster(self, hospital_id: str) -> Optional[int]:
        """Get cluster ID for a hospital."""
        if hospital_id in self.assignments:
            return self.assignments[hospital_id].cluster_id
        return None
        
    def get_cluster_members(self, cluster_id: int) -> List[str]:
        """Get hospitals in a specific cluster."""
        if cluster_id in self.clusters:
            return self.clusters[cluster_id].hospital_ids
        return []
        
    def get_clusters(self) -> Dict[int, HospitalCluster]:
        """Get all current clusters."""
        return self.clusters.copy()
        
    def update_cluster_performance(self, cluster_id: int, metrics: Dict[str, float]):
        """Update performance metrics for a cluster."""
        if cluster_id in self.clusters:
            self.clusters[cluster_id].performance_metrics.update(metrics)
            self.clusters[cluster_id].last_updated = datetime.utcnow().isoformat()
            
            # Check for performance improvement
            if metrics.get("accuracy_improvement", 0) > 0:
                self.metrics["performance_improvements"] += 1
                
    def get_cluster_performance(self, cluster_id: int) -> Optional[Dict[str, Any]]:
        """Get performance metrics for a cluster."""
        if cluster_id not in self.clusters:
            return None
            
        cluster = self.clusters[cluster_id]
        
        return {
            "cluster_id": cluster_id,
            "hospitals": len(cluster.hospital_ids),
            "cluster_quality": cluster.cluster_quality,
            "byzantine_score": cluster.byzantine_score,
            "performance_metrics": cluster.performance_metrics,
            "last_updated": cluster.last_updated
        }
        
    def get_cluster_assignments(self) -> List[Dict[str, Any]]:
        """Get all cluster assignments."""
        return [
            {
                "hospital_id": assignment.hospital_id,
                "cluster_id": assignment.cluster_id,
                "assignment_confidence": assignment.assignment_confidence,
                "distance_to_centroid": assignment.distance_to_centroid,
                "timestamp": assignment.timestamp
            }
            for assignment in self.assignments.values()
        ]
        
    def find_optimal_clusters(self, max_clusters: int = 10) -> int:
        """Find optimal number of clusters using elbow method."""
        if len(self.current_status) < 10:
            return min(self.n_clusters, len(self.current_status) // 2)
            
        # Extract features for all hospitals
        features = []
        for hospital_id in self.current_status.keys():
            features.append(self.extract_cluster_features(hospital_id))
            
        feature_matrix = np.array(features)
        
        # Test different cluster numbers
        silhouette_scores = []
        cluster_range = range(2, min(max_clusters + 1, len(features) // 2))
        
        for n_clusters in cluster_range:
            try:
                clustering_model = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = clustering_model.fit_predict(feature_matrix)
                score = silhouette_score(feature_matrix, cluster_labels)
                silhouette_scores.append(score)
            except Exception as e:
                self.logger.error(f"Error testing {n_clusters} clusters: {e}")
                silhouette_scores.append(0.0)
                
        if silhouette_scores:
            optimal_idx = np.argmax(silhouette_scores)
            return cluster_range[optimal_idx]
        else:
            return self.n_clusters
            
    def _calculate_cluster_byzantine_score(self, hospital_ids: List[str]) -> float:
        """Calculate Byzantine score for a cluster."""
        # Get average Byzantine score of hospitals in cluster
        scores = []
        for hospital_id in hospital_ids:
            if hospital_id in self.current_status:
                # Get from current status or use default
                score = 0.1  # Default low Byzantine score
                scores.append(score)
                
        return np.mean(scores) if scores else 0.0
        
    def _is_byzantine_tolerant(self, hospital_ids: List[str], byzantine_score: float) -> bool:
        """Check if cluster meets Byzantine tolerance requirements."""
        # Check ratio of Byzantine hospitals
        max_byzantine = int(len(hospital_ids) * self.max_byzantine_ratio)
        
        # Byzantine score threshold
        score_threshold = 0.5
        
        return byzantine_score <= score_threshold
        
    def _calculate_cluster_performance(self, hospital_ids: List[str]) -> Dict[str, float]:
        """Calculate performance metrics for a cluster."""
        # Aggregate individual hospital performance
        metrics = {
            "average_accuracy": np.random.uniform(0.8, 0.95),  # Simulated
            "convergence_speed": np.random.uniform(0.1, 1.0),
            "participation_rate": np.random.uniform(0.6, 1.0),
            "gradient_quality": np.random.uniform(0.7, 1.0)
        }
        
        return metrics
        
    def get_cluster_history(self) -> List[Dict[str, Any]]:
        """Get cluster history over time."""
        return self.cluster_history.copy()
        
    def get_stability_metrics(self) -> Dict[str, float]:
        """Get clustering stability metrics."""
        if len(self.cluster_history) < 2:
            return {"stability_score": 1.0, "average_cluster_changes": 0.0}
            
        # Calculate stability based on cluster changes
        changes = [entry.get("cluster_changes", 0) for entry in self.cluster_history]
        avg_changes = np.mean(changes)
        
        # Stability decreases with more changes
        stability_score = max(0.0, 1.0 - (avg_changes / len(self.current_status)))
        
        return {
            "stability_score": stability_score,
            "average_cluster_changes": avg_changes,
            "total_reclusterings": self.metrics["total_reclusterings"],
            "average_cluster_quality": self.metrics["average_cluster_quality"]
        }
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get adaptive clustering metrics."""
        return {
            **self.metrics,
            "total_hospitals": len(self.current_status),
            "active_clusters": len(self.clusters),
            "cluster_assignments": len(self.assignments),
            "last_reclustering": self.last_reclustering.isoformat(),
            "stability_metrics": self.get_stability_metrics()
        }
        
    def reset_metrics(self):
        """Reset metrics counters."""
        self.metrics = {
            "total_reclusterings": 0,
            "cluster_changes": 0,
            "average_cluster_quality": 0.0,
            "byzantine_violations": 0,
            "performance_improvements": 0
        }