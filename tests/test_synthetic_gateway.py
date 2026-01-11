"""
Test suite for Synthetic Gateway

Tests privacy-preserving synthetic data generation with differential privacy guarantees.

DPDP § Citation: §9(4) - Purpose limitation through synthetic data generation
Byzantine Theorem: GAN-based synthesis with Byzantine-robust discriminator

Test Command: pytest tests/test_synthetic_gateway.py -v --cov=sna/synthetic_gateway

Metrics:
- Privacy Loss: ε ≤ 1.0
- Statistical Similarity: > 95%
- Generation Throughput: > 1000 records/second
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import redis.asyncio as redis

from sna.synthetic_gateway import (
    SyntheticGateway,
    SyntheticDataGenerator,
    GaussianCopulaSynthesizer,
    CTGANSynthesizer,
    SynthesisConfig,
    SynthesisMethod,
    DataType,
    PrivacyBudget,
    QualityMetrics
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
async def gateway(redis_client):
    """Create a synthetic gateway instance."""
    generator = Mock(spec=SyntheticDataGenerator)
    gateway = SyntheticGateway(redis_client, generator)
    return gateway


@pytest.fixture
def sample_medical_data():
    """Create sample medical data for testing."""
    np.random.seed(42)
    
    data = pd.DataFrame({
        'patient_id': [f'P{i:03d}' for i in range(100)],
        'age': np.random.randint(18, 90, 100),
        'gender': np.random.choice(['M', 'F'], 100),
        'diagnosis_code': np.random.choice(['D1', 'D2', 'D3', 'D4'], 100),
        'lab_value_1': np.random.normal(100, 15, 100),
        'lab_value_2': np.random.normal(50, 8, 100),
        'temperature': np.random.normal(98.6, 1.2, 100),
        'blood_pressure_systolic': np.random.randint(110, 180, 100),
        'blood_pressure_diastolic': np.random.randint(70, 110, 100),
        'heart_rate': np.random.randint(60, 100, 100)
    })
    
    return data


@pytest.fixture
def synthesis_config():
    """Create a synthesis configuration for testing."""
    return SynthesisConfig(
        method=SynthesisMethod.GAUSSIAN_COPULA,
        data_type=DataType.LAB_RESULTS,
        num_records=1000,
        privacy_budget=PrivacyBudget(epsilon=1.0, delta=1e-6),
        preserve_correlations=True,
        random_seed=42
    )


class TestPrivacyBudget:
    """Test privacy budget management."""
    
    def test_valid_privacy_budget(self):
        """Test creating a valid privacy budget."""
        budget = PrivacyBudget(epsilon=1.0, delta=1e-6, mechanism="gaussian")
        
        assert budget.epsilon == 1.0
        assert budget.delta == 1e-6
        assert budget.mechanism == "gaussian"
        
    def test_invalid_epsilon(self):
        """Test that negative epsilon raises error."""
        with pytest.raises(ValueError, match="Epsilon must be positive"):
            PrivacyBudget(epsilon=-1.0, delta=1e-6)
            
    def test_invalid_delta(self):
        """Test that negative delta raises error."""
        with pytest.raises(ValueError, match="Delta must be positive"):
            PrivacyBudget(epsilon=1.0, delta=-1e-6)


class TestSynthesisConfig:
    """Test synthesis configuration validation."""
    
    def test_valid_config(self):
        """Test creating a valid synthesis configuration."""
        config = SynthesisConfig(
            method=SynthesisMethod.CTGAN,
            data_type=DataType.VITAL_SIGNS,
            num_records=5000,
            privacy_budget=PrivacyBudget(epsilon=0.5, delta=1e-6),
            categorical_threshold=20
        )
        
        assert config.method == SynthesisMethod.CTGAN
        assert config.data_type == DataType.VITAL_SIGNS
        assert config.num_records == 5000
        
    def test_invalid_num_records(self):
        """Test that invalid number of records raises error."""
        with pytest.raises(ValueError, match="Number of records must be positive"):
            SynthesisConfig(
                method=SynthesisMethod.GAUSSIAN_COPULA,
                data_type=DataType.DIAGNOSIS,
                num_records=-100
            )
            
    def test_default_privacy_budget(self):
        """Test default privacy budget creation."""
        config = SynthesisConfig(
            method=SynthesisMethod.GAUSSIAN_COPULA,
            data_type=DataType.DIAGNOSIS,
            num_records=1000
        )
        
        assert config.privacy_budget.epsilon == 1.0  # Default value
        assert config.privacy_budget.delta == 1e-6   # Default value


class TestQualityMetrics:
    """Test quality metrics evaluation."""
    
    def test_quality_metrics_creation(self):
        """Test creating quality metrics."""
        metrics = QualityMetrics(
            statistical_similarity=0.95,
            correlation_preservation=0.90,
            privacy_loss=0.1,
            utility_score=0.92,
            ks_test_p_value=0.8,
            categorical_accuracy=0.88
        )
        
        assert metrics.statistical_similarity == 0.95
        assert metrics.correlation_preservation == 0.90
        assert metrics.utility_score == 0.92
        
    def test_is_acceptable_default_threshold(self):
        """Test quality acceptability with default threshold."""
        metrics = QualityMetrics(
            statistical_similarity=0.95,
            correlation_preservation=0.90,
            privacy_loss=0.1,
            utility_score=0.92,
            ks_test_p_value=0.8,
            categorical_accuracy=0.88
        )
        
        assert metrics.is_acceptable() is True
        
    def test_is_acceptable_fails(self):
        """Test quality acceptability with failing metrics."""
        metrics = QualityMetrics(
            statistical_similarity=0.70,  # Below threshold
            correlation_preservation=0.90,
            privacy_loss=0.1,
            utility_score=0.92,
            ks_test_p_value=0.8,
            categorical_accuracy=0.88
        )
        
        assert metrics.is_acceptable() is False
        
    def test_is_acceptable_custom_threshold(self):
        """Test quality acceptability with custom threshold."""
        metrics = QualityMetrics(
            statistical_similarity=0.80,
            correlation_preservation=0.85,
            privacy_loss=0.1,
            utility_score=0.82,
            ks_test_p_value=0.8,
            categorical_accuracy=0.88
        )
        
        assert metrics.is_acceptable(threshold=0.75) is True
        assert metrics.is_acceptable(threshold=0.90) is False


class TestGaussianCopulaSynthesizer:
    """Test Gaussian copula-based synthesis."""
    
    @pytest.mark.asyncio
    async def test_fit_and_generate(self, synthesis_config, sample_medical_data):
        """Test fitting and generating data with Gaussian copula."""
        synthesizer = GaussianCopulaSynthesizer(synthesis_config)
        
        # Fit to real data
        await synthesizer.fit(sample_medical_data)
        
        # Generate synthetic data
        synthetic_data = await synthesizer.generate(500)
        
        assert len(synthetic_data) == 500
        assert list(synthetic_data.columns) == list(sample_medical_data.columns)
        
        # Check data types are preserved
        for column in sample_medical_data.columns:
            if pd.api.types.is_numeric_dtype(sample_medical_data[column]):
                assert pd.api.types.is_numeric_dtype(synthetic_data[column])
                
    @pytest.mark.asyncio
    async def test_fit_empty_data_raises_error(self, synthesis_config):
        """Test that fitting empty data raises error."""
        synthesizer = GaussianCopulaSynthesizer(synthesis_config)
        empty_data = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Cannot fit to empty dataset"):
            await synthesizer.fit(empty_data)
            
    @pytest.mark.asyncio
    async def test_generate_before_fit_raises_error(self, synthesis_config):
        """Test that generating before fit raises error."""
        synthesizer = GaussianCopulaSynthesizer(synthesis_config)
        
        with pytest.raises(ValueError, match="Synthesizer must be fitted before generation"):
            await synthesizer.generate(100)
            
    @pytest.mark.asyncio
    async def test_evaluate_quality(self, synthesis_config, sample_medical_data):
        """Test quality evaluation of synthetic data."""
        synthesizer = GaussianCopulaSynthesizer(synthesis_config)
        await synthesizer.fit(sample_medical_data)
        
        synthetic_data = await synthesizer.generate(100)
        metrics = await synthesizer.evaluate_quality(synthetic_data, sample_medical_data)
        
        assert isinstance(metrics, QualityMetrics)
        assert 0 <= metrics.statistical_similarity <= 1
        assert 0 <= metrics.correlation_preservation <= 1
        assert 0 <= metrics.utility_score <= 1
        assert 0 <= metrics.privacy_loss <= 1
        
    @pytest.mark.asyncio
    async def test_privacy_preservation(self, synthesis_config, sample_medical_data):
        """Test that privacy is preserved through DP noise."""
        config_high_privacy = SynthesisConfig(
            method=SynthesisMethod.GAUSSIAN_COPULA,
            data_type=DataType.LAB_RESULTS,
            num_records=1000,
            privacy_budget=PrivacyBudget(epsilon=0.1, delta=1e-6),  # High privacy
            random_seed=42
        )
        
        synthesizer_high = GaussianCopulaSynthesizer(config_high_privacy)
        synthesizer_low = GaussianCopulaSynthesizer(synthesis_config)  # Lower privacy
        
        await synthesizer_high.fit(sample_medical_data)
        await synthesizer_low.fit(sample_medical_data)
        
        synth_high = await synthesizer_high.generate(100)
        synth_low = await synthesizer_low.generate(100)
        
        metrics_high = await synthesizer_high.evaluate_quality(synth_high, sample_medical_data)
        metrics_low = await synthesizer_low.evaluate_quality(synth_low, sample_medical_data)
        
        # Higher privacy should result in lower utility (privacy-utility tradeoff)
        assert metrics_high.utility_score <= metrics_low.utility_score


class TestCTGANSynthesizer:
    """Test CTGAN-based synthesis."""
    
    @pytest.mark.asyncio
    async def test_ctgan_fit_and_generate(self, synthesis_config, sample_medical_data):
        """Test CTGAN fitting and generation."""
        config = SynthesisConfig(
            method=SynthesisMethod.CTGAN,
            data_type=DataType.VITAL_SIGNS,
            num_records=200,
            privacy_budget=PrivacyBudget(epsilon=1.0, delta=1e-6)
        )
        
        synthesizer = CTGANSynthesizer(config)
        await synthesizer.fit(sample_medical_data)
        
        synthetic_data = await synthesizer.generate(100)
        
        assert len(synthetic_data) == 100
        assert len(synthetic_data.columns) == len(sample_medical_data.columns)
        
    @pytest.mark.asyncio
    async def test_ctgan_quality_evaluation(self, synthesis_config, sample_medical_data):
        """Test CTGAN quality evaluation."""
        config = SynthesisConfig(
            method=SynthesisMethod.CTGAN,
            data_type=DataType.VITAL_SIGNS,
            num_records=200,
            privacy_budget=PrivacyBudget(epsilon=1.0, delta=1e-6)
        )
        
        synthesizer = CTGANSynthesizer(config)
        await synthesizer.fit(sample_medical_data)
        
        synthetic_data = await synthesizer.generate(100)
        metrics = await synthesizer.evaluate_quality(synthetic_data, sample_medical_data)
        
        assert isinstance(metrics, QualityMetrics)
        assert metrics.statistical_similarity >= 0.0
        assert metrics.statistical_similarity <= 1.0


class TestSyntheticDataGenerator:
    """Test synthetic data generation coordinator."""
    
    @pytest.mark.asyncio
    async def test_create_synthetic_dataset(self, redis_client, sample_medical_data):
        """Test creating synthetic dataset with quality guarantees."""
        generator = SyntheticDataGenerator(redis_client)
        
        config = SynthesisConfig(
            method=SynthesisMethod.GAUSSIAN_COPULA,
            data_type=DataType.LAB_RESULTS,
            num_records=500,
            privacy_budget=PrivacyBudget(epsilon=0.5, delta=1e-6)
        )
        
        synthetic_data, metrics = await generator.create_synthetic_dataset(
            config, sample_medical_data
        )
        
        assert len(synthetic_data) == 500
        assert isinstance(metrics, QualityMetrics)
        
        # Quality should be acceptable
        assert metrics.utility_score >= 0.5  # Reasonable baseline
        
    @pytest.mark.asyncio
    async def test_quality_retry_mechanism(self, redis_client):
        """Test retry mechanism when quality is insufficient."""
        generator = SyntheticDataGenerator(redis_client)
        
        # Create a synthesizer that will initially fail quality check
        class FailingSynthesizer:
            def __init__(self):
                self.attempts = 0
                
            async def fit(self, data):
                pass
                
            async def generate(self, num_records):
                return pd.DataFrame({'col1': range(num_records)})
                
            async def evaluate_quality(self, synthetic_data, real_data):
                self.attempts += 1
                # Fail first two attempts, succeed on third
                if self.attempts <= 2:
                    return QualityMetrics(
                        statistical_similarity=0.5,  # Below threshold
                        correlation_preservation=0.5,
                        privacy_loss=0.1,
                        utility_score=0.5,
                        ks_test_p_value=0.1,
                        categorical_accuracy=0.5
                    )
                else:
                    return QualityMetrics(
                        statistical_similarity=0.95,
                        correlation_preservation=0.90,
                        privacy_loss=0.1,
                        utility_score=0.92,
                        ks_test_p_value=0.8,
                        categorical_accuracy=0.88
                    )
                    
        real_data = pd.DataFrame({'col1': range(100)})
        
        # Mock the synthesizer creation
        with patch.object(generator, '_get_synthesizer', return_value=FailingSynthesizer()):
            config = SynthesisConfig(
                method=SynthesisMethod.GAUSSIAN_COPULA,
                data_type=DataType.DIAGNOSIS,
                num_records=200,
                privacy_budget=PrivacyBudget(epsilon=1.0, delta=1e-6)
            )
            
            synthetic_data, metrics = await generator.create_synthetic_dataset(
                config, real_data
            )
            
            # Should succeed after retries
            assert metrics.is_acceptable() is True
            
    @pytest.mark.asyncio
    async def test_synthesis_history_tracking(self, redis_client):
        """Test synthesis history tracking."""
        generator = SyntheticDataGenerator(redis_client)
        
        # Add some history
        await generator._store_synthesis_metadata(
            SynthesisConfig(
                method=SynthesisMethod.GAUSSIAN_COPULA,
                data_type=DataType.DIAGNOSIS,
                num_records=100
            ),
            QualityMetrics(
                statistical_similarity=0.95,
                correlation_preservation=0.90,
                privacy_loss=0.1,
                utility_score=0.92,
                ks_test_p_value=0.8,
                categorical_accuracy=0.88
            )
        )
        
        history = await generator.get_synthesis_history(limit=10)
        
        assert len(history) == 1
        assert 'timestamp' in history[0]
        assert 'config' in history[0]
        assert 'metrics' in history[0]


class TestSyntheticGateway:
    """Test main synthetic gateway service."""
    
    @pytest.mark.asyncio
    async def test_generate_test_data(self, redis_client):
        """Test generating test data for different data types."""
        gateway = SyntheticGateway(redis_client)
        
        # Mock the generator to return predictable data
        mock_data = pd.DataFrame({
            'patient_id': [f'P{i:03d}' for i in range(10)],
            'diagnosis_code': ['D1'] * 10
        })
        
        mock_metrics = QualityMetrics(
            statistical_similarity=0.95,
            correlation_preservation=0.90,
            privacy_loss=0.1,
            utility_score=0.92,
            ks_test_p_value=0.8,
            categorical_accuracy=0.88
        )
        
        gateway.generator.create_synthetic_dataset = AsyncMock(
            return_value=(mock_data, mock_metrics)
        )
        
        # Test different data types
        data_types = [DataType.DIAGNOSIS, DataType.VITAL_SIGNS, DataType.LAB_RESULTS]
        
        for data_type in data_types:
            synthetic_data, metrics = await gateway.generate_test_data(
                data_type=data_type,
                num_records=10
            )
            
            assert len(synthetic_data) == 10
            assert isinstance(metrics, QualityMetrics)
            assert metrics.utility_score >= 0.9
            
    @pytest.mark.asyncio
    async def test_validate_synthetic_data(self, redis_client, sample_medical_data):
        """Test synthetic data validation."""
        gateway = SyntheticGateway(redis_client)
        
        # Create some synthetic data
        synthetic_data = sample_medical_data.copy()
        synthetic_data['age'] = synthetic_data['age'] + np.random.normal(0, 2, len(synthetic_data))
        
        metrics = await gateway.validate_synthetic_data(
            synthetic_data, sample_medical_data
        )
        
        assert isinstance(metrics, QualityMetrics)
        assert 0 <= metrics.statistical_similarity <= 1
        
    @pytest.mark.asyncio
    async def test_privacy_budget_status(self, redis_client):
        """Test privacy budget status reporting."""
        gateway = SyntheticGateway(redis_client)
        
        status = await gateway.get_privacy_budget_status()
        
        assert 'total_budget_allocated' in status
        assert 'budget_used' in status
        assert 'budget_remaining' in status
        assert 'active_synthesizers' in status
        assert 'privacy_loss_rate' in status
        
        assert status['budget_remaining'] == status['total_budget_allocated'] - status['budget_used']
        
    @pytest.mark.asyncio
    async def test_cleanup_expired_synthesizers(self, redis_client):
        """Test cleanup of expired synthesizer instances."""
        gateway = SyntheticGateway(redis_client)
        
        # Add some synthesizers
        gateway.generator.synthesizers = {
            'synth1': Mock(),
            'synth2': Mock(),
            'synth3': Mock()
        }
        
        cleaned_count = await gateway.cleanup_expired_synthesizers()
        
        # Should clean up all synthesizers
        assert len(gateway.generator.synthesizers) == 0
        
    @pytest.mark.asyncio
    async def test_schema_generation(self, redis_client):
        """Test schema generation for different data types."""
        gateway = SyntheticGateway(redis_client)
        
        # Test each data type has a schema
        data_types = [
            DataType.DIAGNOSIS,
            DataType.LAB_RESULTS,
            DataType.VITAL_SIGNS,
            DataType.MEDICATIONS,
            DataType.DEMOGRAPHICS
        ]
        
        for data_type in data_types:
            schema = gateway._get_schema_for_data_type(data_type)
            
            assert isinstance(schema, dict)
            assert len(schema) > 0  # Should have at least one field
            
            # Schema should have common medical fields
            has_patient_id = any('patient' in field.lower() for field in schema.keys())
            assert has_patient_id or data_type == DataType.DEMOGRAPHICS  # Demographics might not have patient_id


class TestPerformanceMetrics:
    """Test performance and throughput metrics."""
    
    @pytest.mark.asyncio
    async def test_generation_throughput(self, synthesis_config, sample_medical_data):
        """Test that generation throughput meets requirements."""
        synthesizer = GaussianCopulaSynthesizer(synthesis_config)
        await synthesizer.fit(sample_medical_data)
        
        import time
        
        start_time = time.time()
        synthetic_data = await synthesizer.generate(1000)
        end_time = time.time()
        
        generation_time = end_time - start_time
        throughput = len(synthetic_data) / generation_time
        
        # Should achieve at least 1000 records/second
        assert throughput >= 1000, f"Throughput {throughput} below requirement"
        
    @pytest.mark.asyncio
    async def test_privacy_loss_bounds(self, synthesis_config, sample_medical_data):
        """Test that privacy loss stays within bounds."""
        config = SynthesisConfig(
            method=SynthesisMethod.GAUSSIAN_COPULA,
            data_type=DataType.LAB_RESULTS,
            num_records=1000,
            privacy_budget=PrivacyBudget(epsilon=0.5, delta=1e-6)  # Strict privacy
        )
        
        synthesizer = GaussianCopulaSynthesizer(config)
        await synthesizer.fit(sample_medical_data)
        
        synthetic_data = await synthesizer.generate(1000)
        metrics = await synthesizer.evaluate_quality(synthetic_data, sample_medical_data)
        
        # Privacy loss should be within expected bounds
        assert metrics.privacy_loss <= 1.0
        assert metrics.privacy_loss <= config.privacy_budget.epsilon


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.mark.asyncio
    async def test_single_column_data(self, redis_client):
        """Test synthesis with single column data."""
        single_col_data = pd.DataFrame({'age': range(100)})
        
        config = SynthesisConfig(
            method=SynthesisMethod.GAUSSIAN_COPULA,
            data_type=DataType.DEMOGRAPHICS,
            num_records=200
        )
        
        synthesizer = GaussianCopulaSynthesizer(config)
        await synthesizer.fit(single_col_data)
        
        synthetic_data = await synthesizer.generate(200)
        
        assert len(synthetic_data) == 200
        assert list(synthetic_data.columns) == ['age']
        
    @pytest.mark.asyncio
    async def test_all_categorical_data(self, redis_client):
        """Test synthesis with all categorical data."""
        categorical_data = pd.DataFrame({
            'gender': np.random.choice(['M', 'F'], 100),
            'diagnosis': np.random.choice(['D1', 'D2', 'D3'], 100),
            'status': np.random.choice(['A', 'B', 'C'], 100)
        })
        
        config = SynthesisConfig(
            method=SynthesisMethod.GAUSSIAN_COPULA,
            data_type=DataType.DIAGNOSIS,
            num_records=200
        )
        
        synthesizer = GaussianCopulaSynthesizer(config)
        await synthesizer.fit(categorical_data)
        
        synthetic_data = await synthesizer.generate(200)
        
        assert len(synthetic_data) == 200
        assert set(synthetic_data.columns) == set(categorical_data.columns)
        
        # All values should be from original categories
        for column in categorical_data.columns:
            original_categories = set(categorical_data[column].unique())
            synthetic_categories = set(synthetic_data[column].unique())
            assert synthetic_categories.issubset(original_categories)
            
    @pytest.mark.asyncio
    async def test_correlation_preservation(self, redis_client):
        """Test that correlations are preserved in synthetic data."""
        # Create correlated data
        np.random.seed(42)
        x = np.random.normal(0, 1, 1000)
        y = 2 * x + np.random.normal(0, 0.1, 1000)  # Strong correlation
        
        correlated_data = pd.DataFrame({'x': x, 'y': y})
        
        config = SynthesisConfig(
            method=SynthesisMethod.GAUSSIAN_COPULA,
            data_type=DataType.LAB_RESULTS,
            num_records=1000,
            preserve_correlations=True
        )
        
        synthesizer = GaussianCopulaSynthesizer(config)
        await synthesizer.fit(correlated_data)
        
        synthetic_data = await synthesizer.generate(1000)
        
        # Check correlation preservation
        real_corr = correlated_data['x'].corr(correlated_data['y'])
        synth_corr = synthetic_data['x'].corr(synthetic_data['y'])
        
        # Correlation should be reasonably preserved
        assert abs(real_corr - synth_corr) < 0.3  # Within reasonable bounds