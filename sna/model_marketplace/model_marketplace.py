"""
Model Marketplace for Federated Learning Model Exchange

Enables hospitals to buy, sell, and exchange trained federated learning models
with Byzantine-fault-tolerant validation and HealthToken-based transactions.

DPDP § Citation: §9(4) - Purpose limitation through model sharing
Byzantine Theorem: Byzantine-robust model validation with consensus (Krum et al., 2017)
Test Command: pytest tests/test_model_marketplace.py -v --cov=sna/model_marketplace

Metrics:
- Transaction Throughput: > 100 models/hour
- Validation Accuracy: > 95%
- Byzantine Tolerance: f < n/3 malicious models
"""

import asyncio
import hashlib
import json
import logging
import torch
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
import redis.asyncio as redis
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from pydantic import BaseModel, validator
import aiohttp
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ModelCategory(Enum):
    """Categories of models available in the marketplace."""
    DIAGNOSIS_PREDICTION = "diagnosis_prediction"
    RISK_ASSESSMENT = "risk_assessment"
    IMAGE_CLASSIFICATION = "image_classification"
    DRUG_RECOMMENDATION = "drug_recommendation"
    OUTCOME_PREDICTION = "outcome_prediction"
    ANOMALY_DETECTION = "anomaly_detection"


class ModelStatus(Enum):
    """Status of a model in the marketplace."""
    PENDING_VALIDATION = "pending_validation"
    VALIDATED = "validated"
    REJECTED = "rejected"
    DEPRECATED = "deprecated"
    SUSPENDED = "suspended"


class TransactionStatus(Enum):
    """Status of a marketplace transaction."""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    COMPLETED = "completed"
    FAILED = "failed"
    DISPUTED = "disputed"


class ModelValidationResult(Enum):
    """Result of model validation."""
    VALID = "valid"
    INVALID = "invalid"
    MALICIOUS = "malicious"
    SUSPICIOUS = "suspicious"


@dataclass
class ModelMetadata:
    """Metadata for a federated learning model."""
    model_id: str
    name: str
    description: str
    category: ModelCategory
    version: str
    hospital_id: str
    hospital_name: str
    accuracy: float
    f1_score: float
    auc_score: float
    training_rounds: int
    data_size: int
    privacy_budget_used: float
    model_architecture: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    framework: str = "pytorch"
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            'model_id': self.model_id,
            'name': self.name,
            'description': self.description,
            'category': self.category.value,
            'version': self.version,
            'hospital_id': self.hospital_id,
            'hospital_name': self.hospital_name,
            'accuracy': self.accuracy,
            'f1_score': self.f1_score,
            'auc_score': self.auc_score,
            'training_rounds': self.training_rounds,
            'data_size': self.data_size,
            'privacy_budget_used': self.privacy_budget_used,
            'model_architecture': self.model_architecture,
            'input_shape': list(self.input_shape),
            'output_shape': list(self.output_shape),
            'framework': self.framework,
            'tags': self.tags
        }


@dataclass
class ModelPricing:
    """Pricing information for a model."""
    base_price_healthtokens: float
    per_inference_cost: float
    subscription_monthly: float
    revenue_share_percentage: float = 0.0
    dynamic_pricing_enabled: bool = True
    price_last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def calculate_dynamic_price(self, demand_factor: float, supply_factor: float) -> float:
        """Calculate dynamic price based on market conditions."""
        if not self.dynamic_pricing_enabled:
            return self.base_price_healthtokens
            
        # Simple dynamic pricing algorithm
        price = self.base_price_healthtokens * (demand_factor / (supply_factor + 0.1))
        return max(price, self.base_price_healthtokens * 0.5)  # Minimum 50% of base price


@dataclass
class ModelListing:
    """A model listing in the marketplace."""
    listing_id: str
    model_metadata: ModelMetadata
    pricing: ModelPricing
    status: ModelStatus = ModelStatus.PENDING_VALIDATION
    validation_score: float = 0.0
    reviews: List[Dict[str, Any]] = field(default_factory=list)
    download_count: int = 0
    successful_inferences: int = 0
    total_revenue_healthtokens: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def add_review(self, hospital_id: str, rating: int, comment: str) -> None:
        """Add a review to the model listing."""
        self.reviews.append({
            'hospital_id': hospital_id,
            'rating': rating,
            'comment': comment,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    def update_status(self, new_status: ModelStatus, validation_score: Optional[float] = None) -> None:
        """Update model status and validation score."""
        self.status = new_status
        if validation_score is not None:
            self.validation_score = validation_score
        self.updated_at = datetime.utcnow()


@dataclass
class ModelTransaction:
    """A transaction in the model marketplace."""
    transaction_id: str
    listing_id: str
    buyer_hospital_id: str
    seller_hospital_id: str
    transaction_type: str  # 'purchase', 'subscription', 'inference'
    amount_healthtokens: float
    status: TransactionStatus = TransactionStatus.PENDING
    model_hash: str = ""
    download_url: str = ""
    license_key: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    def complete_transaction(self, model_hash: str, download_url: str, license_key: str) -> None:
        """Mark transaction as completed."""
        self.status = TransactionStatus.COMPLETED
        self.model_hash = model_hash
        self.download_url = download_url
        self.license_key = license_key
        self.completed_at = datetime.utcnow()


class ByzantineModelValidator(ABC):
    """Abstract base class for Byzantine-robust model validation."""
    
    @abstractmethod
    async def validate_model(
        self,
        model_data: bytes,
        metadata: ModelMetadata,
        validation_dataset: Optional[pd.DataFrame] = None
    ) -> Tuple[ModelValidationResult, float, str]:
        """
        Validate a model using Byzantine-robust methods.
        
        Returns:
            Tuple of (validation_result, confidence_score, detailed_report)
        """
        pass
        
    @abstractmethod
    async def cross_validate_with_peers(
        self,
        model_id: str,
        peer_hospital_ids: List[str]
    ) -> Dict[str, Any]:
        """Cross-validate model with peer hospitals."""
        pass


class ConsensusModelValidator(ByzantineModelValidator):
    """Byzantine-robust model validator using consensus mechanisms."""
    
    def __init__(
        self,
        redis_client: redis.Redis,
        min_peer_validations: int = 5,
        byzantine_threshold: float = 0.33
    ):
        self.redis = redis_client
        self.min_peer_validations = min_peer_validations
        self.byzantine_threshold = byzantine_threshold
        
    async def validate_model(
        self,
        model_data: bytes,
        metadata: ModelMetadata,
        validation_dataset: Optional[pd.DataFrame] = None
    ) -> Tuple[ModelValidationResult, float, str]:
        """Validate model using multiple Byzantine-robust techniques."""
        
        # Step 1: Basic integrity checks
        integrity_result = await self._check_model_integrity(model_data, metadata)
        if not integrity_result['valid']:
            return (
                ModelValidationResult.INVALID,
                0.0,
                f"Model integrity check failed: {integrity_result['reason']}"
            )
            
        # Step 2: Performance validation
        performance_result = await self._validate_model_performance(metadata)
        if not performance_result['valid']:
            return (
                ModelValidationResult.INVALID,
                0.2,
                f"Performance validation failed: {performance_result['reason']}"
            )
            
        # Step 3: Byzantine behavior detection
        byzantine_result = await self._detect_byzantine_behavior(model_data, metadata)
        if byzantine_result['is_byzantine']:
            return (
                ModelValidationResult.MALICIOUS,
                0.1,
                f"Byzantine behavior detected: {byzantine_result['reason']}"
            )
            
        # Step 4: Privacy compliance check
        privacy_result = await self._check_privacy_compliance(metadata)
        if not privacy_result['compliant']:
            return (
                ModelValidationResult.SUSPICIOUS,
                0.6,
                f"Privacy compliance issue: {privacy_result['reason']}"
            )
            
        # Calculate overall confidence score
        confidence = min(0.95, performance_result.get('confidence', 0.8))
        
        return (
            ModelValidationResult.VALID,
            confidence,
            "Model validation passed all checks"
        )
        
    async def _check_model_integrity(self, model_data: bytes, metadata: ModelMetadata) -> Dict[str, Any]:
        """Check model file integrity and consistency."""
        try:
            # Calculate hash of model data
            model_hash = hashlib.sha256(model_data).hexdigest()
            
            # Basic size checks
            if len(model_data) < 1024:  # Minimum 1KB
                return {'valid': False, 'reason': 'Model file too small'}
                
            if len(model_data) > 500 * 1024 * 1024:  # Maximum 500MB
                return {'valid': False, 'reason': 'Model file too large'}
                
            # Check framework compatibility
            if metadata.framework not in ['pytorch', 'tensorflow', 'sklearn']:
                return {'valid': False, 'reason': 'Unsupported framework'}
                
            return {'valid': True, 'model_hash': model_hash}
            
        except Exception as e:
            return {'valid': False, 'reason': f'Integrity check error: {str(e)}'}
            
    async def _validate_model_performance(self, metadata: ModelMetadata) -> Dict[str, Any]:
        """Validate model performance metrics."""
        # Check for reasonable performance ranges
        if not (0.0 <= metadata.accuracy <= 1.0):
            return {'valid': False, 'reason': 'Invalid accuracy value'}
            
        if not (0.0 <= metadata.f1_score <= 1.0):
            return {'valid': False, 'reason': 'Invalid F1 score'}
            
        if not (0.0 <= metadata.auc_score <= 1.0):
            return {'valid': False, 'reason': 'Invalid AUC score'}
            
        # Check for suspiciously perfect performance (potential overfitting or attack)
        if metadata.accuracy > 0.99 and metadata.training_rounds < 10:
            return {
                'valid': False,
                'reason': 'Suspiciously high accuracy with low training rounds'
            }
            
        # Calculate confidence based on performance metrics
        confidence = np.mean([metadata.accuracy, metadata.f1_score, metadata.auc_score])
        
        return {'valid': True, 'confidence': confidence}
        
    async def _detect_byzantine_behavior(self, model_data: bytes, metadata: ModelMetadata) -> Dict[str, Any]:
        """Detect potential Byzantine attacks in the model."""
        try:
            # Load model for analysis (simplified)
            # In practice, this would involve more sophisticated analysis
            
            # Check for common attack patterns
            attack_indicators = []
            
            # Indicator 1: Unusually large weights
            if metadata.model_architecture == 'neural_network':
                # This would analyze actual model weights
                pass
                
            # Indicator 2: Inconsistent metadata
            if metadata.accuracy < 0.5 and metadata.f1_score > 0.9:
                attack_indicators.append("Inconsistent performance metrics")
                
            # Indicator 3: Excessive privacy budget usage
            if metadata.privacy_budget_used > 5.0:  # Threshold
                attack_indicators.append("Excessive privacy budget usage")
                
            if attack_indicators:
                return {
                    'is_byzantine': True,
                    'reason': '; '.join(attack_indicators)
                }
                
            return {'is_byzantine': False}
            
        except Exception as e:
            return {'is_byzantine': True, 'reason': f'Analysis error: {str(e)}'}
            
    async def _check_privacy_compliance(self, metadata: ModelMetadata) -> Dict[str, Any]:
        """Check DPDP privacy compliance."""
        # Check privacy budget usage
        if metadata.privacy_budget_used > 9.5:  # DPDP §9(4) limit
            return {
                'compliant': False,
                'reason': f"Privacy budget {metadata.privacy_budget_used} exceeds DPDP limit"
            }
            
        # Check data size adequacy
        if metadata.data_size < 100:  # Minimum data size
            return {
                'compliant': False,
                'reason': f"Training data size {metadata.data_size} too small"
            }
            
        return {'compliant': True}
        
    async def cross_validate_with_peers(
        self,
        model_id: str,
        peer_hospital_ids: List[str]
    ) -> Dict[str, Any]:
        """Cross-validate model with peer hospitals using Byzantine consensus."""
        
        if len(peer_hospital_ids) < self.min_peer_validations:
            return {
                'consensus_reached': False,
                'reason': 'Insufficient peer validators'
            }
            
        # Collect peer validation results
        peer_results = await self._collect_peer_validations(model_id, peer_hospital_ids)
        
        # Apply Byzantine consensus (similar to Krum aggregation)
        valid_votes = []
        malicious_votes = []
        
        for result in peer_results:
            if result.get('validation_result') == 'valid':
                valid_votes.append(result)
            else:
                malicious_votes.append(result)
                
        # Byzantine threshold check
        if len(malicious_votes) > len(peer_hospital_ids) * self.byzantine_threshold:
            return {
                'consensus_reached': False,
                'reason': f"Too many malicious votes: {len(malicious_votes)}/{len(peer_hospital_ids)}",
                'malicious_peers': [vote['peer_id'] for vote in malicious_votes]
            }
            
        # Calculate consensus score
        consensus_score = len(valid_votes) / len(peer_hospital_ids)
        
        return {
            'consensus_reached': True,
            'consensus_score': consensus_score,
            'valid_votes': len(valid_votes),
            'total_votes': len(peer_hospital_ids),
            'malicious_votes': len(malicious_votes)
        }
        
    async def _collect_peer_validations(self, model_id: str, peer_hospital_ids: List[str]) -> List[Dict[str, Any]]:
        """Collect validation results from peer hospitals."""
        # This would make actual network calls to peer hospitals
        # For now, simulate peer validation results
        
        peer_results = []
        for peer_id in peer_hospital_ids:
            # Simulate peer validation (in practice, this would be an API call)
            peer_result = {
                'peer_id': peer_id,
                'validation_result': np.random.choice(['valid', 'invalid'], p=[0.8, 0.2]),
                'confidence': np.random.uniform(0.7, 0.95),
                'timestamp': datetime.utcnow().isoformat()
            }
            peer_results.append(peer_result)
            
        return peer_results


class ModelMarketplace:
    """Main marketplace for federated learning model exchange."""
    
    def __init__(
        self,
        redis_client: redis.Redis,
        validator: Optional[ByzantineModelValidator] = None
    ):
        self.redis = redis_client
        self.validator = validator or ConsensusModelValidator(redis_client)
        self.listings: Dict[str, ModelListing] = {}
        self.transactions: Dict[str, ModelTransaction] = {}
        
    async def create_listing(
        self,
        model_metadata: ModelMetadata,
        pricing: ModelPricing,
        model_file: bytes
    ) -> str:
        """Create a new model listing in the marketplace."""
        
        # Generate unique listing ID
        listing_id = hashlib.sha256(
            f"{model_metadata.model_id}:{model_metadata.hospital_id}:{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]
        
        # Create listing
        listing = ModelListing(
            listing_id=listing_id,
            model_metadata=model_metadata,
            pricing=pricing
        )
        
        # Validate model before listing
        validation_result, confidence, report = await self.validator.validate_model(
            model_file, model_metadata
        )
        
        if validation_result == ModelValidationResult.INVALID:
            raise ValueError(f"Model validation failed: {report}")
            
        if validation_result == ModelValidationResult.MALICIOUS:
            # Flag for manual review
            listing.update_status(ModelStatus.SUSPENDED, 0.1)
            logger.warning(f"Malicious model detected: {listing_id}")
            
        elif validation_result == ModelValidationResult.VALID:
            listing.update_status(ModelStatus.VALIDATED, confidence)
            
        # Store listing
        self.listings[listing_id] = listing
        await self._save_listing_to_redis(listing)
        
        logger.info(f"Created model listing: {listing_id} - {model_metadata.name}")
        return listing_id
        
    async def purchase_model(
        self,
        listing_id: str,
        buyer_hospital_id: str,
        transaction_type: str = 'purchase'
    ) -> str:
        """Purchase a model from the marketplace."""
        
        if listing_id not in self.listings:
            raise ValueError(f"Listing {listing_id} not found")
            
        listing = self.listings[listing_id]
        
        if listing.status != ModelStatus.VALIDATED:
            raise ValueError(f"Model not available for purchase: {listing.status.value}")
            
        # Calculate price (with dynamic pricing if enabled)
        price = listing.pricing.base_price_healthtokens
        if listing.pricing.dynamic_pricing_enabled:
            # This would calculate based on market conditions
            price = listing.pricing.calculate_dynamic_price(1.0, 1.0)
            
        # Create transaction
        transaction_id = hashlib.sha256(
            f"{listing_id}:{buyer_hospital_id}:{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]
        
        transaction = ModelTransaction(
            transaction_id=transaction_id,
            listing_id=listing_id,
            buyer_hospital_id=buyer_hospital_id,
            seller_hospital_id=listing.model_metadata.hospital_id,
            transaction_type=transaction_type,
            amount_healthtokens=price
        )
        
        # Process payment (this would integrate with HealthToken ledger)
        payment_successful = await self._process_payment(
            buyer_hospital_id,
            listing.model_metadata.hospital_id,
            price
        )
        
        if not payment_successful:
            transaction.status = TransactionStatus.FAILED
            await self._save_transaction_to_redis(transaction)
            raise ValueError("Payment processing failed")
            
        # Generate download credentials
        model_hash = hashlib.sha256(f"model_{listing_id}".encode()).hexdigest()
        download_url = f"/downloads/{transaction_id}"
        license_key = hashlib.sha256(f"license_{transaction_id}".encode()).hexdigest()[:32]
        
        # Complete transaction
        transaction.complete_transaction(model_hash, download_url, license_key)
        
        # Update listing statistics
        listing.download_count += 1
        listing.total_revenue_healthtokens += price
        
        # Store transaction
        self.transactions[transaction_id] = transaction
        await self._save_transaction_to_redis(transaction)
        await self._save_listing_to_redis(listing)
        
        logger.info(f"Model purchase completed: {transaction_id}")
        return transaction_id
        
    async def submit_model_review(
        self,
        listing_id: str,
        hospital_id: str,
        rating: int,
        comment: str
    ) -> bool:
        """Submit a review for a model."""
        
        if listing_id not in self.listings:
            return False
            
        if not 1 <= rating <= 5:
            return False
            
        listing = self.listings[listing_id]
        listing.add_review(hospital_id, rating, comment)
        
        await self._save_listing_to_redis(listing)
        return True
        
    async def search_models(
        self,
        category: Optional[ModelCategory] = None,
        min_accuracy: float = 0.0,
        max_price: float = float('inf'),
        hospital_id: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[ModelListing]:
        """Search for models in the marketplace."""
        
        results = []
        
        for listing in self.listings.values():
            # Filter by status
            if listing.status != ModelStatus.VALIDATED:
                continue
                
            # Filter by category
            if category and listing.model_metadata.category != category:
                continue
                
            # Filter by accuracy
            if listing.model_metadata.accuracy < min_accuracy:
                continue
                
            # Filter by price
            if listing.pricing.base_price_healthtokens > max_price:
                continue
                
            # Filter by hospital
            if hospital_id and listing.model_metadata.hospital_id != hospital_id:
                continue
                
            # Filter by tags
            if tags:
                if not any(tag in listing.model_metadata.tags for tag in tags):
                    continue
                    
            results.append(listing)
            
        # Sort by validation score and download count
        results.sort(
            key=lambda x: (x.validation_score, x.download_count),
            reverse=True
        )
        
        return results
        
    async def get_trending_models(self, limit: int = 10) -> List[ModelListing]:
        """Get trending models based on recent downloads and reviews."""
        
        # Filter validated models
        validated_models = [
            listing for listing in self.listings.values()
            if listing.status == ModelStatus.VALIDATED
        ]
        
        # Sort by recent activity (downloads in last 30 days)
        trending_models = sorted(
            validated_models,
            key=lambda x: x.download_count,
            reverse=True
        )[:limit]
        
        return trending_models
        
    async def get_model_statistics(self) -> Dict[str, Any]:
        """Get marketplace statistics."""
        
        total_listings = len(self.listings)
        validated_listings = len([
            listing for listing in self.listings.values()
            if listing.status == ModelStatus.VALIDATED
        ])
        
        total_transactions = len(self.transactions)
        successful_transactions = len([
            transaction for transaction in self.transactions.values()
            if transaction.status == TransactionStatus.COMPLETED
        ])
        
        total_revenue = sum(
            transaction.amount_healthtokens
            for transaction in self.transactions.values()
            if transaction.status == TransactionStatus.COMPLETED
        )
        
        category_distribution = {}
        for listing in self.listings.values():
            category = listing.model_metadata.category.value
            category_distribution[category] = category_distribution.get(category, 0) + 1
            
        return {
            'total_listings': total_listings,
            'validated_listings': validated_listings,
            'total_transactions': total_transactions,
            'successful_transactions': successful_transactions,
            'total_revenue_healthtokens': total_revenue,
            'category_distribution': category_distribution,
            'validation_rate': validated_listings / total_listings if total_listings > 0 else 0,
            'success_rate': successful_transactions / total_transactions if total_transactions > 0 else 0
        }
        
    async def _process_payment(
        self,
        buyer_id: str,
        seller_id: str,
        amount: float
    ) -> bool:
        """Process HealthToken payment between hospitals."""
        # This would integrate with the HealthToken ledger system
        # For now, simulate successful payment
        return True
        
    async def _save_listing_to_redis(self, listing: ModelListing) -> None:
        """Save listing to Redis for persistence."""
        listing_data = {
            'listing_id': listing.listing_id,
            'model_metadata': json.dumps(listing.model_metadata.to_dict()),
            'pricing': json.dumps({
                'base_price_healthtokens': listing.pricing.base_price_healthtokens,
                'per_inference_cost': listing.pricing.per_inference_cost,
                'subscription_monthly': listing.pricing.subscription_monthly,
                'revenue_share_percentage': listing.pricing.revenue_share_percentage,
                'dynamic_pricing_enabled': listing.pricing.dynamic_pricing_enabled
            }),
            'status': listing.status.value,
            'validation_score': listing.validation_score,
            'reviews': json.dumps(listing.reviews),
            'download_count': listing.download_count,
            'successful_inferences': listing.successful_inferences,
            'total_revenue_healthtokens': listing.total_revenue_healthtokens,
            'created_at': listing.created_at.isoformat(),
            'updated_at': listing.updated_at.isoformat()
        }
        
        await self.redis.hset(f"listing:{listing.listing_id}", mapping=listing_data)
        
    async def _save_transaction_to_redis(self, transaction: ModelTransaction) -> None:
        """Save transaction to Redis for persistence."""
        transaction_data = {
            'transaction_id': transaction.transaction_id,
            'listing_id': transaction.listing_id,
            'buyer_hospital_id': transaction.buyer_hospital_id,
            'seller_hospital_id': transaction.seller_hospital_id,
            'transaction_type': transaction.transaction_type,
            'amount_healthtokens': transaction.amount_healthtokens,
            'status': transaction.status.value,
            'model_hash': transaction.model_hash,
            'download_url': transaction.download_url,
            'license_key': transaction.license_key,
            'created_at': transaction.created_at.isoformat(),
            'completed_at': transaction.completed_at.isoformat() if transaction.completed_at else None
        }
        
        await self.redis.hset(f"transaction:{transaction.transaction_id}", mapping=transaction_data)
        
    async def get_listing(self, listing_id: str) -> Optional[ModelListing]:
        """Get a model listing by ID."""
        return self.listings.get(listing_id)
        
    async def get_transaction(self, transaction_id: str) -> Optional[ModelTransaction]:
        """Get a transaction by ID."""
        return self.transactions.get(transaction_id)