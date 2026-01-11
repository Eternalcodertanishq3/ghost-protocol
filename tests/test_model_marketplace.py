"""
Test suite for Model Marketplace

Tests Byzantine-robust model exchange with HealthToken-based transactions.

DPDP § Citation: §9(4) - Purpose limitation through model sharing
Byzantine Theorem: Byzantine-robust model validation with consensus (Krum et al., 2017)

Test Command: pytest tests/test_model_marketplace.py -v --cov=sna/model_marketplace

Metrics:
- Transaction Throughput: > 100 models/hour
- Validation Accuracy: > 95%
- Byzantine Tolerance: f < n/3 malicious models
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import redis.asyncio as redis

from sna.model_marketplace import (
    ModelMarketplace,
    ModelListing,
    ModelTransaction,
    ModelMetadata,
    ModelPricing,
    ModelCategory,
    ModelStatus,
    TransactionStatus,
    ModelValidationResult,
    ConsensusModelValidator,
    ByzantineModelValidator
)


@pytest.fixture
async def redis_client():
    """Create a mock Redis client."""
    client = Mock(spec=redis.Redis)
    client.hset = AsyncMock(return_value=1)
    client.hgetall = AsyncMock(return_value={})
    client.lrange = AsyncMock(return_value=[])
    client.lpush = AsyncMock(return_value=1)
    return client


@pytest.fixture
async def marketplace(redis_client):
    """Create a model marketplace instance."""
    validator = Mock(spec=ConsensusModelValidator)
    marketplace = ModelMarketplace(redis_client, validator)
    return marketplace


@pytest.fixture
def sample_model_metadata():
    """Create sample model metadata for testing."""
    return ModelMetadata(
        model_id="model_123",
        name="Diabetes Risk Predictor",
        description="Predicts diabetes risk using patient demographics and lab results",
        category=ModelCategory.RISK_ASSESSMENT,
        version="1.0.0",
        hospital_id="hospital_001",
        hospital_name="AI Medical Center",
        accuracy=0.87,
        f1_score=0.85,
        auc_score=0.91,
        training_rounds=50,
        data_size=10000,
        privacy_budget_used=2.5,
        model_architecture="neural_network",
        input_shape=(10,),
        output_shape=(2,),
        framework="pytorch",
        tags=["diabetes", "risk_prediction", "neural_network"]
    )


@pytest.fixture
def sample_model_pricing():
    """Create sample model pricing for testing."""
    return ModelPricing(
        base_price_healthtokens=1000.0,
        per_inference_cost=1.0,
        subscription_monthly=100.0,
        revenue_share_percentage=10.0,
        dynamic_pricing_enabled=True
    )


@pytest.fixture
def sample_model_file():
    """Create a sample model file for testing."""
    return b"mock_model_data_" + b"x" * 10000  # 10KB mock model


class TestModelMetadata:
    """Test model metadata handling."""
    
    def test_model_metadata_creation(self, sample_model_metadata):
        """Test creating model metadata."""
        assert sample_model_metadata.model_id == "model_123"
        assert sample_model_metadata.name == "Diabetes Risk Predictor"
        assert sample_model_metadata.accuracy == 0.87
        assert sample_model_metadata.category == ModelCategory.RISK_ASSESSMENT
        
    def test_model_metadata_to_dict(self, sample_model_metadata):
        """Test converting metadata to dictionary."""
        metadata_dict = sample_model_metadata.to_dict()
        
        assert metadata_dict['model_id'] == "model_123"
        assert metadata_dict['category'] == "risk_assessment"
        assert metadata_dict['accuracy'] == 0.87
        assert metadata_dict['framework'] == "pytorch"
        assert len(metadata_dict['tags']) == 3


class TestModelPricing:
    """Test model pricing calculations."""
    
    def test_static_pricing(self, sample_model_pricing):
        """Test static pricing calculation."""
        price = sample_model_pricing.calculate_dynamic_price(1.0, 1.0)
        assert price == 1000.0  # Base price when dynamic pricing is disabled
        
    def test_dynamic_pricing(self, sample_model_pricing):
        """Test dynamic pricing calculation."""
        # Enable dynamic pricing
        sample_model_pricing.dynamic_pricing_enabled = True
        
        # High demand, low supply
        price = sample_model_pricing.calculate_dynamic_price(2.0, 0.5)
        assert price > 1000.0
        
        # Low demand, high supply
        price = sample_model_pricing.calculate_dynamic_price(0.5, 2.0)
        assert price < 1000.0
        assert price >= 500.0  # Minimum 50% of base price
        
    def test_pricing_bounds(self, sample_model_pricing):
        """Test that pricing respects minimum bounds."""
        sample_model_pricing.dynamic_pricing_enabled = True
        
        # Extreme low demand
        price = sample_model_pricing.calculate_dynamic_price(0.1, 10.0)
        assert price == 500.0  # Minimum price


class TestModelListing:
    """Test model listing functionality."""
    
    def test_model_listing_creation(self, sample_model_metadata, sample_model_pricing):
        """Test creating a model listing."""
        listing = ModelListing(
            listing_id="listing_001",
            model_metadata=sample_model_metadata,
            pricing=sample_model_pricing
        )
        
        assert listing.listing_id == "listing_001"
        assert listing.model_metadata == sample_model_metadata
        assert listing.status == ModelStatus.PENDING_VALIDATION
        assert listing.validation_score == 0.0
        assert listing.download_count == 0
        
    def test_add_review(self, sample_model_metadata, sample_model_pricing):
        """Test adding reviews to model listing."""
        listing = ModelListing(
            listing_id="listing_001",
            model_metadata=sample_model_metadata,
            pricing=sample_model_pricing
        )
        
        listing.add_review("hospital_002", 5, "Excellent model, highly accurate!")
        listing.add_review("hospital_003", 4, "Good model but could be faster")
        
        assert len(listing.reviews) == 2
        assert listing.reviews[0]['rating'] == 5
        assert listing.reviews[1]['rating'] == 4
        
    def test_update_status(self, sample_model_metadata, sample_model_pricing):
        """Test updating model status."""
        listing = ModelListing(
            listing_id="listing_001",
            model_metadata=sample_model_metadata,
            pricing=sample_model_pricing
        )
        
        original_updated_at = listing.updated_at
        
        # Update status
        listing.update_status(ModelStatus.VALIDATED, 0.95)
        
        assert listing.status == ModelStatus.VALIDATED
        assert listing.validation_score == 0.95
        assert listing.updated_at > original_updated_at


class TestModelTransaction:
    """Test model transaction functionality."""
    
    def test_transaction_creation(self, sample_model_metadata, sample_model_pricing):
        """Test creating a model transaction."""
        transaction = ModelTransaction(
            transaction_id="tx_001",
            listing_id="listing_001",
            buyer_hospital_id="hospital_002",
            seller_hospital_id="hospital_001",
            transaction_type="purchase",
            amount_healthtokens=1000.0
        )
        
        assert transaction.transaction_id == "tx_001"
        assert transaction.buyer_hospital_id == "hospital_002"
        assert transaction.seller_hospital_id == "hospital_001"
        assert transaction.status == TransactionStatus.PENDING
        
    def test_complete_transaction(self, sample_model_metadata, sample_model_pricing):
        """Test completing a model transaction."""
        transaction = ModelTransaction(
            transaction_id="tx_001",
            listing_id="listing_001",
            buyer_hospital_id="hospital_002",
            seller_hospital_id="hospital_001",
            transaction_type="purchase",
            amount_healthtokens=1000.0
        )
        
        original_created_at = transaction.created_at
        
        # Complete transaction
        transaction.complete_transaction(
            model_hash="abc123hash",
            download_url="/downloads/tx_001",
            license_key="license_key_123"
        )
        
        assert transaction.status == TransactionStatus.COMPLETED
        assert transaction.model_hash == "abc123hash"
        assert transaction.download_url == "/downloads/tx_001"
        assert transaction.license_key == "license_key_123"
        assert transaction.completed_at is not None
        assert transaction.completed_at > original_created_at


class TestConsensusModelValidator:
    """Test Byzantine-robust model validation."""
    
    @pytest.mark.asyncio
    async def test_valid_model_validation(self, redis_client, sample_model_metadata, sample_model_file):
        """Test validation of a valid model."""
        validator = ConsensusModelValidator(redis_client)
        
        # Mock the validation methods to return success
        with patch.object(validator, '_check_model_integrity', return_value={
            'valid': True, 'model_hash': 'abc123'
        }):
            with patch.object(validator, '_validate_model_performance', return_value={
                'valid': True, 'confidence': 0.9
            }):
                with patch.object(validator, '_detect_byzantine_behavior', return_value={
                    'is_byzantine': False
                }):
                    with patch.object(validator, '_check_privacy_compliance', return_value={
                        'compliant': True
                    }):
                        
                        result, confidence, report = await validator.validate_model(
                            sample_model_file, sample_model_metadata
                        )
                        
                        assert result == ModelValidationResult.VALID
                        assert confidence == 0.9
                        assert "passed all checks" in report.lower()
                        
    @pytest.mark.asyncio
    async def test_invalid_model_validation(self, redis_client, sample_model_metadata, sample_model_file):
        """Test validation of an invalid model."""
        validator = ConsensusModelValidator(redis_client)
        
        # Mock integrity check to fail
        with patch.object(validator, '_check_model_integrity', return_value={
            'valid': False, 'reason': 'Model file too small'
        }):
            result, confidence, report = await validator.validate_model(
                sample_model_file, sample_model_metadata
            )
            
            assert result == ModelValidationResult.INVALID
            assert confidence == 0.0
            assert "model integrity check failed" in report.lower()
            
    @pytest.mark.asyncio
    async def test_malicious_model_detection(self, redis_client, sample_model_metadata, sample_model_file):
        """Test detection of malicious models."""
        validator = ConsensusModelValidator(redis_client)
        
        # Mock Byzantine behavior detection
        with patch.object(validator, '_check_model_integrity', return_value={
            'valid': True, 'model_hash': 'abc123'
        }):
            with patch.object(validator, '_validate_model_performance', return_value={
                'valid': True, 'confidence': 0.8
            }):
                with patch.object(validator, '_detect_byzantine_behavior', return_value={
                    'is_byzantine': True, 'reason': 'Suspicious weight patterns detected'
                }):
                    with patch.object(validator, '_check_privacy_compliance', return_value={
                        'compliant': True
                    }):
                        
                        result, confidence, report = await validator.validate_model(
                            sample_model_file, sample_model_metadata
                        )
                        
                        assert result == ModelValidationResult.MALICIOUS
                        assert confidence == 0.1
                        assert "byzantine behavior detected" in report.lower()
                        
    @pytest.mark.asyncio
    async def test_privacy_compliance_check(self, redis_client, sample_model_metadata, sample_model_file):
        """Test privacy compliance validation."""
        validator = ConsensusModelValidator(redis_client)
        
        # Create metadata with excessive privacy budget usage
        malicious_metadata = ModelMetadata(
            model_id="model_malicious",
            name="Malicious Model",
            description="Model with excessive privacy budget usage",
            category=ModelCategory.RISK_ASSESSMENT,
            version="1.0.0",
            hospital_id="hospital_malicious",
            hospital_name="Malicious Hospital",
            accuracy=0.95,
            f1_score=0.93,
            auc_score=0.97,
            training_rounds=5,
            data_size=50,
            privacy_budget_used=15.0,  # Exceeds DPDP limit
            model_architecture="neural_network",
            input_shape=(10,),
            output_shape=(2,)
        )
        
        with patch.object(validator, '_check_model_integrity', return_value={
            'valid': True
        }):
            with patch.object(validator, '_validate_model_performance', return_value={
                'valid': True, 'confidence': 0.95
            }):
                with patch.object(validator, '_detect_byzantine_behavior', return_value={
                    'is_byzantine': False
                }):
                    with patch.object(validator, '_check_privacy_compliance', return_value={
                        'compliant': False,
                        'reason': 'Privacy budget 15.0 exceeds DPDP limit'
                    }):
                        
                        result, confidence, report = await validator.validate_model(
                            sample_model_file, malicious_metadata
                        )
                        
                        assert result == ModelValidationResult.SUSPICIOUS
                        assert confidence == 0.6
                        assert "privacy compliance issue" in report.lower()
                        
    @pytest.mark.asyncio
    async def test_peer_cross_validation(self, redis_client):
        """Test cross-validation with peer hospitals."""
        validator = ConsensusModelValidator(redis_client, min_peer_validations=3)
        
        # Mock peer validation collection
        with patch.object(validator, '_collect_peer_validations', return_value=[
            {'peer_id': 'hospital_1', 'validation_result': 'valid', 'confidence': 0.9},
            {'peer_id': 'hospital_2', 'validation_result': 'valid', 'confidence': 0.85},
            {'peer_id': 'hospital_3', 'validation_result': 'valid', 'confidence': 0.88},
        ]):
            
            result = await validator.cross_validate_with_peers(
                "model_123", ['hospital_1', 'hospital_2', 'hospital_3']
            )
            
            assert result['consensus_reached'] is True
            assert result['consensus_score'] == 1.0
            assert result['valid_votes'] == 3
            assert result['total_votes'] == 3
            assert result['malicious_votes'] == 0
            
    @pytest.mark.asyncio
    async def test_byzantine_peer_consensus(self, redis_client):
        """Test Byzantine peer consensus with malicious votes."""
        validator = ConsensusModelValidator(redis_client, min_peer_validations=5)
        
        # Mock peer validation with Byzantine nodes
        with patch.object(validator, '_collect_peer_validations', return_value=[
            {'peer_id': 'hospital_1', 'validation_result': 'valid', 'confidence': 0.9},
            {'peer_id': 'hospital_2', 'validation_result': 'valid', 'confidence': 0.85},
            {'peer_id': 'hospital_3', 'validation_result': 'invalid', 'confidence': 0.3},  # Byzantine
            {'peer_id': 'hospital_4', 'validation_result': 'valid', 'confidence': 0.88},
            {'peer_id': 'hospital_5', 'validation_result': 'invalid', 'confidence': 0.2},  # Byzantine
            {'peer_id': 'hospital_6', 'validation_result': 'valid', 'confidence': 0.92},
        ]):
            
            result = await validator.cross_validate_with_peers(
                "model_123", [f'hospital_{i}' for i in range(1, 7)]
            )
            
            # Should still reach consensus (4 valid vs 2 malicious < 33% threshold)
            assert result['consensus_reached'] is True
            assert result['consensus_score'] == 2/3  # 4/6
            assert result['valid_votes'] == 4
            assert result['malicious_votes'] == 2
            
    @pytest.mark.asyncio
    async def test_insufficient_peer_validations(self, redis_client):
        """Test validation with insufficient peers."""
        validator = ConsensusModelValidator(redis_client, min_peer_validations=5)
        
        result = await validator.cross_validate_with_peers(
            "model_123", ['hospital_1', 'hospital_2']  # Only 2 peers
        )
        
        assert result['consensus_reached'] is False
        assert 'insufficient peer validators' in result['reason'].lower()


class TestModelMarketplace:
    """Test main marketplace functionality."""
    
    @pytest.mark.asyncio
    async def test_create_listing(self, marketplace, sample_model_metadata, sample_model_pricing, sample_model_file):
        """Test creating a model listing."""
        # Mock validator to return valid result
        marketplace.validator.validate_model = AsyncMock(return_value=(
            ModelValidationResult.VALID, 0.9, "Model validation passed"
        ))
        
        listing_id = await marketplace.create_listing(
            sample_model_metadata, sample_model_pricing, sample_model_file
        )
        
        assert listing_id is not None
        assert len(listing_id) == 16
        
        # Verify listing was created
        listing = await marketplace.get_listing(listing_id)
        assert listing is not None
        assert listing.status == ModelStatus.VALIDATED
        assert listing.validation_score == 0.9
        
    @pytest.mark.asyncio
    async def test_create_listing_malicious_model(self, marketplace, sample_model_metadata, sample_model_pricing, sample_model_file):
        """Test creating listing for malicious model."""
        # Mock validator to detect malicious model
        marketplace.validator.validate_model = AsyncMock(return_value=(
            ModelValidationResult.MALICIOUS, 0.1, "Byzantine behavior detected"
        ))
        
        listing_id = await marketplace.create_listing(
            sample_model_metadata, sample_model_pricing, sample_model_file
        )
        
        # Listing should be created but suspended
        listing = await marketplace.get_listing(listing_id)
        assert listing.status == ModelStatus.SUSPENDED
        assert listing.validation_score == 0.1
        
    @pytest.mark.asyncio
    async def test_purchase_model(self, marketplace, sample_model_metadata, sample_model_pricing, sample_model_file):
        """Test purchasing a model from the marketplace."""
        # Create a listing first
        marketplace.validator.validate_model = AsyncMock(return_value=(
            ModelValidationResult.VALID, 0.9, "Model validation passed"
        ))
        
        listing_id = await marketplace.create_listing(
            sample_model_metadata, sample_model_pricing, sample_model_file
        )
        
        # Mock payment processing
        marketplace._process_payment = AsyncMock(return_value=True)
        
        # Purchase the model
        transaction_id = await marketplace.purchase_model(
            listing_id, "hospital_002", "purchase"
        )
        
        assert transaction_id is not None
        assert len(transaction_id) == 16
        
        # Verify transaction was created
        transaction = await marketplace.get_transaction(transaction_id)
        assert transaction is not None
        assert transaction.status == TransactionStatus.COMPLETED
        assert transaction.amount_healthtokens == 1000.0
        
        # Verify listing was updated
        listing = await marketplace.get_listing(listing_id)
        assert listing.download_count == 1
        assert listing.total_revenue_healthtokens == 1000.0
        
    @pytest.mark.asyncio
    async def test_purchase_model_payment_failed(self, marketplace, sample_model_metadata, sample_model_pricing, sample_model_file):
        """Test purchasing model with failed payment."""
        # Create listing
        marketplace.validator.validate_model = AsyncMock(return_value=(
            ModelValidationResult.VALID, 0.9, "Model validation passed"
        ))
        
        listing_id = await marketplace.create_listing(
            sample_model_metadata, sample_model_pricing, sample_model_file
        )
        
        # Mock payment failure
        marketplace._process_payment = AsyncMock(return_value=False)
        
        # Attempt purchase
        with pytest.raises(ValueError, match="Payment processing failed"):
            await marketplace.purchase_model(listing_id, "hospital_002")
            
    @pytest.mark.asyncio
    async def test_submit_model_review(self, marketplace, sample_model_metadata, sample_model_pricing, sample_model_file):
        """Test submitting model reviews."""
        # Create listing
        marketplace.validator.validate_model = AsyncMock(return_value=(
            ModelValidationResult.VALID, 0.9, "Model validation passed"
        ))
        
        listing_id = await marketplace.create_listing(
            sample_model_metadata, sample_model_pricing, sample_model_file
        )
        
        # Submit reviews
        result1 = await marketplace.submit_model_review(listing_id, "hospital_002", 5, "Excellent!")
        result2 = await marketplace.submit_model_review(listing_id, "hospital_003", 4, "Good model")
        result3 = await marketplace.submit_model_review(listing_id, "hospital_004", 6, "Invalid rating")
        
        assert result1 is True
        assert result2 is True
        assert result3 is False  # Invalid rating
        
        # Verify reviews were added
        listing = await marketplace.get_listing(listing_id)
        assert len(listing.reviews) == 2
        
    @pytest.mark.asyncio
    async def test_search_models(self, marketplace, sample_model_metadata, sample_model_pricing, sample_model_file):
        """Test searching for models in the marketplace."""
        # Create multiple listings
        marketplace.validator.validate_model = AsyncMock(return_value=(
            ModelValidationResult.VALID, 0.9, "Model validation passed"
        ))
        
        # Create models with different characteristics
        for i in range(5):
            metadata = ModelMetadata(
                model_id=f"model_{i}",
                name=f"Model {i}",
                description=f"Test model {i}",
                category=ModelCategory.DIAGNOSIS_PREDICTION if i % 2 == 0 else ModelCategory.RISK_ASSESSMENT,
                version="1.0.0",
                hospital_id=f"hospital_{i}",
                hospital_name=f"Hospital {i}",
                accuracy=0.8 + i * 0.03,  # Varying accuracy
                f1_score=0.8 + i * 0.02,
                auc_score=0.8 + i * 0.025,
                training_rounds=50,
                data_size=10000,
                privacy_budget_used=2.5,
                model_architecture="neural_network",
                input_shape=(10,),
                output_shape=(2,),
                tags=["test"] if i < 3 else ["production"]
            )
            
            pricing = ModelPricing(
                base_price_healthtokens=1000.0 + i * 100,
                per_inference_cost=1.0,
                subscription_monthly=100.0
            )
            
            await marketplace.create_listing(metadata, pricing, b"mock_model_data")
            
        # Test search by category
        diagnosis_models = await marketplace.search_models(
            category=ModelCategory.DIAGNOSIS_PREDICTION
        )
        assert len(diagnosis_models) == 3
        
        # Test search by minimum accuracy
        high_accuracy_models = await marketplace.search_models(min_accuracy=0.88)
        assert len(high_accuracy_models) >= 2
        
        # Test search by maximum price
        cheap_models = await marketplace.search_models(max_price=1100.0)
        assert len(cheap_models) >= 1
        
        # Test search by tags
        test_models = await marketplace.search_models(tags=["test"])
        assert len(test_models) == 3
        
    @pytest.mark.asyncio
    async def test_get_trending_models(self, marketplace, sample_model_metadata, sample_model_pricing, sample_model_file):
        """Test getting trending models."""
        # Create multiple listings and simulate downloads
        marketplace.validator.validate_model = AsyncMock(return_value=(
            ModelValidationResult.VALID, 0.9, "Model validation passed"
        ))
        
        listing_ids = []
        for i in range(5):
            metadata = ModelMetadata(
                model_id=f"model_{i}",
                name=f"Model {i}",
                description=f"Test model {i}",
                category=ModelCategory.DIAGNOSIS_PREDICTION,
                version="1.0.0",
                hospital_id=f"hospital_{i}",
                hospital_name=f"Hospital {i}",
                accuracy=0.85,
                f1_score=0.83,
                auc_score=0.87,
                training_rounds=50,
                data_size=10000,
                privacy_budget_used=2.5,
                model_architecture="neural_network",
                input_shape=(10,),
                output_shape=(2,)
            )
            
            listing_id = await marketplace.create_listing(
                metadata, sample_model_pricing, b"mock_model_data"
            )
            listing_ids.append(listing_id)
            
            # Simulate different download counts
            listing = await marketplace.get_listing(listing_id)
            listing.download_count = i * 10  # 0, 10, 20, 30, 40 downloads
            
        trending_models = await marketplace.get_trending_models(limit=3)
        
        assert len(trending_models) == 3
        
        # Should return models with highest download counts
        download_counts = [model.download_count for model in trending_models]
        assert download_counts == sorted(download_counts, reverse=True)
        
    @pytest.mark.asyncio
    async def test_get_model_statistics(self, marketplace, sample_model_metadata, sample_model_pricing, sample_model_file):
        """Test getting marketplace statistics."""
        # Create some listings and transactions
        marketplace.validator.validate_model = AsyncMock(return_value=(
            ModelValidationResult.VALID, 0.9, "Model validation passed"
        ))
        
        # Create listings
        for i in range(3):
            metadata = ModelMetadata(
                model_id=f"model_{i}",
                name=f"Model {i}",
                description=f"Test model {i}",
                category=ModelCategory.DIAGNOSIS_PREDICTION,
                version="1.0.0",
                hospital_id=f"hospital_{i}",
                hospital_name=f"Hospital {i}",
                accuracy=0.85,
                f1_score=0.83,
                auc_score=0.87,
                training_rounds=50,
                data_size=10000,
                privacy_budget_used=2.5,
                model_architecture="neural_network",
                input_shape=(10,),
                output_shape=(2,)
            )
            
            await marketplace.create_listing(metadata, sample_model_pricing, b"mock_model_data")
            
        # Create some transactions
        marketplace._process_payment = AsyncMock(return_value=True)
        
        listing_ids = list(marketplace.listings.keys())
        for i, listing_id in enumerate(listing_ids):
            await marketplace.purchase_model(listing_id, f"buyer_{i}")
            
        # Get statistics
        stats = await marketplace.get_model_statistics()
        
        assert stats['total_listings'] == 3
        assert stats['validated_listings'] == 3
        assert stats['total_transactions'] == 3
        assert stats['successful_transactions'] == 3
        assert stats['total_revenue_healthtokens'] == 3000.0  # 3 * 1000
        assert 'diagnosis_prediction' in stats['category_distribution']
        assert stats['validation_rate'] == 1.0
        assert stats['success_rate'] == 1.0