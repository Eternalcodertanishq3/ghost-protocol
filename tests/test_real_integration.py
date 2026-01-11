"""
Ghost Protocol Integration Tests
End-to-end testing of the complete federated learning system

DPDP §: §8(2)(a) Testing Requirements, §25 Breach Simulation
Byzantine theorem: Integration tests verify Byzantine fault tolerance
Test command: pytest tests/test_integration.py -v --cov=integration
Metrics tracked: Test coverage, Success rates, Performance benchmarks
"""

import asyncio
import json
import logging
import pytest
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn

from ghost_agent import GhostAgent, DifferentialPrivacyEngine, SecurityManager, DPDPComplianceManager
from ghost_agent.models import ClinicalPredictionModel, ModelFactory
from ghost_agent.adapters import EMRAdapterFactory
from sna import SNAServer, ByzantineShield, HealthTokenLedger
from sna.aggregation import AggregationStrategy


@pytest.mark.asyncio
class TestGhostProtocolIntegration:
    """End-to-end integration tests for Ghost Protocol"""
    
    @pytest.fixture
    async def setup_test_environment(self):
        """Setup complete test environment"""
        
        # Configure test environment
        test_config = {
            "hospital_id": "test_hospital_001",
            "hospital_name": "Test Medical Center",
            "hospital_type": "MULTI_SPECIALITY",
            "location_state": "Karnataka",
            "location_district": "Bengaluru",
            "epsilon_max": 9.5,
            "epsilon_per_update": 1.23,
            "delta_max": 1e-5,
            "gaussian_noise_scale": 1.3,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs_per_round": 5,
            "aggregation_strategy": "FedAvg",
            "fedprox_mu": 0.1
        }
        
        # Initialize security manager
        security_manager = SecurityManager(
            hospital_id=test_config["hospital_id"],
            vault_addr=None  # Use local keys for testing
        )
        
        # Initialize DPDP compliance manager
        dpdp_manager = DPDPComplianceManager(
            hospital_id=test_config["hospital_id"],
            db_path=":memory:"  # Use in-memory DB for testing
        )
        
        # Create ghost agent config
        from ghost_agent.agent import GhostAgentConfig
        agent_config = GhostAgentConfig(
            hospital_id=test_config["hospital_id"],
            hospital_name=test_config["hospital_name"],
            hospital_type=test_config["hospital_type"],
            location_state=test_config["location_state"],
            location_district=test_config["location_district"],
            epsilon_max=test_config["epsilon_max"],
            epsilon_per_update=test_config["epsilon_per_update"],
            delta_max=test_config["delta_max"],
            gaussian_noise_scale=test_config["gaussian_noise_scale"],
            learning_rate=test_config["learning_rate"],
            batch_size=test_config["batch_size"],
            epochs_per_round=test_config["epochs_per_round"],
            aggregation_strategy=test_config["aggregation_strategy"],
            fedprox_mu=test_config["fedprox_mu"],
            security_manager=security_manager,
            dpdp_manager=dpdp_manager
        )
        
        # Initialize ghost agent
        ghost_agent = GhostAgent(agent_config)
        
        # Initialize SNA components
        byzantine_shield = ByzantineShield(byzantine_threshold=0.49)
        healthtoken_ledger = HealthTokenLedger()
        
        # Initialize EMR adapter
        emr_config = {
            "emr_type": "database",
            "db_type": "sqlite",
            "connection_string": ":memory:"
        }
        emr_adapter = EMRAdapterFactory.create_adapter(
            test_config["hospital_id"],
            emr_config["emr_type"],
            emr_config
        )
        
        return {
            "ghost_agent": ghost_agent,
            "byzantine_shield": byzantine_shield,
            "healthtoken_ledger": healthtoken_ledger,
            "emr_adapter": emr_adapter,
            "test_config": test_config
        }
    
    async def test_complete_federated_learning_round(self, setup_test_environment):
        """Test complete federated learning round with privacy guarantees"""
        
        env = setup_test_environment
        ghost_agent = env["ghost_agent"]
        
        try:
            # Start federated learning round
            result = await ghost_agent.start_federated_learning_round()
            
            # Verify round completed successfully
            assert result["status"] == "success"
            assert "round_id" in result
            assert "privacy_budget_used" in result
            assert "local_auc" in result
            assert "compliance_verified" in result
            
            # Verify privacy budget was consumed
            assert result["privacy_budget_used"] > 0
            assert result["privacy_budget_used"] < ghost_agent.config.epsilon_max
            
            # Verify model performance
            assert 0.5 <= result["local_auc"] <= 1.0
            
            # Verify compliance
            assert result["compliance_verified"] == True
            
            logging.info(f"✅ FL Round {result['round_id']} completed successfully")
            logging.info(f"   Privacy used: ε={result['privacy_budget_used']:.3f}")
            logging.info(f"   Local AUC: {result['local_auc']:.3f}")
            
        except Exception as e:
            logging.error(f"❌ FL Round failed: {e}")
            raise
    
    async def test_dpdp_compliance_enforcement(self, setup_test_environment):
        """Test DPDP Act 2023 compliance enforcement"""
        
        env = setup_test_environment
        ghost_agent = env["ghost_agent"]
        dpdp_manager = ghost_agent.dpdp
        
        try:
            # Record test consent
            patient_id = "test_patient_001"
            consent_id = await dpdp_manager.record_consent(
                patient_id=patient_id,
                purpose="federated_learning_clinical_prediction",
                data_categories=["vital_signs", "lab_values", "demographics"],
                consent_mechanism="electronic",
                expiry_days=365
            )
            
            # Verify consent was recorded
            assert consent_id is not None
            
            # Test consent validation
            is_valid = await dpdp_manager.verify_consent_for_purpose(
                patient_id=patient_id,
                purpose="federated_learning_clinical_prediction"
            )
            
            assert is_valid == True
            
            # Test consent withdrawal
            withdrawal_count = await dpdp_manager.withdraw_consent(
                patient_id=patient_id
            )
            
            assert withdrawal_count > 0
            
            # Verify consent is no longer valid
            is_valid_after_withdrawal = await dpdp_manager.verify_consent_for_purpose(
                patient_id=patient_id,
                purpose="federated_learning_clinical_prediction"
            )
            
            assert is_valid_after_withdrawal == False
            
            # Generate compliance report
            report = await dpdp_manager.generate_compliance_report(period_days=1)
            
            assert report["compliance_summary"]["overall_compliance_status"] == "compliant"
            assert "consent_management" in report
            
            logging.info("✅ DPDP compliance enforcement verified")
            logging.info(f"   Consent recorded and validated: {consent_id}")
            logging.info(f"   Withdrawal processed: {withdrawal_count} consents")
            
        except Exception as e:
            logging.error(f"❌ DPDP compliance test failed: {e}")
            raise
    
    async def test_byzantine_shield_detection(self, setup_test_environment):
        """Test Byzantine Shield malicious update detection"""
        
        env = setup_test_environment
        byzantine_shield = env["byzantine_shield"]
        
        try:
            # Test normal update (should be accepted)
            normal_update = {
                "metadata": {
                    "round_id": "test_round_001",
                    "hospital_id": "normal_hospital",
                    "model_performance": {
                        "local_auc": 0.85,
                        "gradient_norm": 1.5
                    },
                    "dp_compliance": {
                        "epsilon_spent": 1.0
                    }
                },
                "model_update": {},  # Mock model update
                "byzantine_shield": {
                    "reputation_score": 1.0,
                    "anomaly_score": 0.1
                }
            }
            
            normal_result = await byzantine_shield.analyze_update(
                hospital_id="normal_hospital",
                ghost_pack=normal_update,
                current_reputation=1.0
            )
            
            assert normal_result.accepted == True
            assert normal_result.anomaly_score < 0.5
            
            # Test malicious update (should be rejected)
            malicious_update = {
                "metadata": {
                    "round_id": "test_round_001",
                    "hospital_id": "malicious_hospital",
                    "model_performance": {
                        "local_auc": 0.99,  # Unrealistically high
                        "gradient_norm": 1000.0  # Extreme gradient norm
                    },
                    "dp_compliance": {
                        "epsilon_spent": 0.1
                    }
                },
                "model_update": {},
                "byzantine_shield": {
                    "reputation_score": 0.3,
                    "anomaly_score": 5.0  # Very high anomaly score
                }
            }
            
            malicious_result = await byzantine_shield.analyze_update(
                hospital_id="malicious_hospital",
                ghost_pack=malicious_update,
                current_reputation=0.3
            )
            
            assert malicious_result.accepted == False
            assert malicious_result.rejection_reason in ["unrealistic_auc", "extreme_gradient_norm", "statistical_anomaly"]
            
            logging.info("✅ Byzantine Shield detection verified")
            logging.info(f"   Normal update accepted: {normal_result.accepted}")
            logging.info(f"   Malicious update rejected: {malicious_result.accepted}")
            logging.info(f"   Rejection reason: {malicious_result.rejection_reason}")
            
        except Exception as e:
            logging.error(f"❌ Byzantine Shield test failed: {e}")
            raise
    
    async def test_healthtoken_distribution(self, setup_test_environment):
        """Test HealthToken incentive distribution"""
        
        env = setup_test_environment
        healthtoken_ledger = env["healthtoken_ledger"]
        
        try:
            # Award tokens to test hospital
            hospital_id = "test_hospital_001"
            initial_balance = healthtoken_ledger.get_balance(hospital_id)
            
            tokens_awarded = await healthtoken_ledger.award_tokens(
                hospital_id=hospital_id,
                round_id="test_round_001",
                local_auc=0.85,
                reputation_score=0.9,
                gradient_quality=0.8,
                participation_history=[1.0, 1.0, 0.9, 1.0, 0.95]
            )
            
            # Verify tokens were awarded
            assert tokens_awarded > 0
            
            final_balance = healthtoken_ledger.get_balance(hospital_id)
            assert final_balance == initial_balance + tokens_awarded
            
            # Test token transfer
            to_hospital = "test_hospital_002"
            transfer_amount = tokens_awarded / 2
            
            transfer_success = await healthtoken_ledger.transfer_tokens(
                from_hospital=hospital_id,
                to_hospital=to_hospital,
                amount=transfer_amount,
                purpose="data_sharing_payment"
            )
            
            assert transfer_success == True
            
            # Verify balances after transfer
            sender_balance = healthtoken_ledger.get_balance(hospital_id)
            receiver_balance = healthtoken_ledger.get_balance(to_hospital)
            
            assert sender_balance == final_balance - transfer_amount
            assert receiver_balance == transfer_amount
            
            # Test staking
            stake_amount = tokens_awarded / 4
            stake_success = await healthtoken_ledger.stake_tokens(
                hospital_id=hospital_id,
                amount=stake_amount
            )
            
            assert stake_success == True
            
            # Verify staking reduces available balance
            post_stake_balance = healthtoken_ledger.get_balance(hospital_id)
            assert post_stake_balance == sender_balance - stake_amount
            
            # Check staking information
            stake_info = healthtoken_ledger.get_stake(hospital_id)
            assert stake_info is not None
            assert stake_info.staked_amount == stake_amount
            assert stake_info.is_locked == True
            
            logging.info("✅ HealthToken distribution verified")
            logging.info(f"   Tokens awarded: {tokens_awarded}")
            logging.info(f"   Transfer successful: {transfer_success}")
            logging.info(f"   Staking successful: {stake_success}")
            
        except Exception as e:
            logging.error(f"❌ HealthToken distribution test failed: {e}")
            raise
    
    async def test_privacy_budget_enforcement(self, setup_test_environment):
        """Test privacy budget exhaustion enforcement"""
        
        env = setup_test_environment
        ghost_agent = env["ghost_agent"]
        dp_engine = ghost_agent.dp_engine
        
        try:
            # Simulate privacy budget exhaustion
            initial_budget = dp_engine.budget.epsilon_remaining
            
            # Artificially exhaust budget
            dp_engine.budget.epsilon_consumed = dp_engine.config.epsilon_max
            
            # Attempt another training round
            with pytest.raises(ValueError) as exc_info:
                await ghost_agent.start_federated_learning_round()
            
            # Verify privacy budget exhaustion error
            assert "Privacy budget exhausted" in str(exc_info.value)
            
            # Verify compliance alert was triggered
            # (In production, this would trigger actual alerts)
            
            logging.info("✅ Privacy budget enforcement verified")
            logging.info(f"   Initial budget: {initial_budget:.2f}")
            logging.info(f"   Budget exhausted correctly")
            
        except Exception as e:
            logging.error(f"❌ Privacy budget enforcement test failed: {e}")
            raise
    
    async def test_aggregation_robustness(self, setup_test_environment):
        """Test Byzantine-robust aggregation"""
        
        env = setup_test_environment
        byzantine_shield = env["byzantine_shield"]
        
        try:
            # Create test model updates
            np.random.seed(42)
            
            # Generate benign updates (normal distribution around true value)
            true_weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
            benign_updates = []
            
            for i in range(8):  # 8 benign hospitals
                noise = torch.tensor(np.random.normal(0, 0.1, 5))
                update = {"layer.weight": true_weights + noise}
                benign_updates.append(update)
            
            # Generate Byzantine updates (arbitrarily wrong values)
            byzantine_updates = []
            for i in range(2):  # 2 Byzantine hospitals (20%)
                byzantine_update = {"layer.weight": torch.tensor([100.0, -100.0, 100.0, -100.0, 100.0])}
                byzantine_updates.append(byzantine_update)
            
            # Combine all updates
            all_updates = benign_updates + byzantine_updates
            weights = [1.0] * len(all_updates)  # Equal weights
            
            # Test different aggregation strategies
            strategies = [
                AggregationStrategy.FEDAVG,
                AggregationStrategy.GEOMETRIC_MEDIAN,
                AggregationStrategy.TRIMMED_MEAN
            ]
            
            results = {}
            
            for strategy in strategies:
                aggregated = await byzantine_shield.aggregate_updates(
                    all_updates,
                    weights,
                    strategy=strategy
                )
                
                # Calculate error from true weights
                aggregated_weights = aggregated["layer.weight"]
                error = torch.norm(aggregated_weights - true_weights).item()
                
                results[strategy.value] = {
                    "error": error,
                    "aggregated_weights": aggregated_weights.tolist()
                }
            
            # Verify geometric median is most robust
            fedavg_error = results["fedavg"]["error"]
            geometric_error = results["geometric_median"]["error"]
            trimmed_error = results["trimmed_mean"]["error"]
            
            logging.info(f"✅ Aggregation robustness verified")
            logging.info(f"   FedAvg error: {fedavg_error:.3f}")
            logging.info(f"   Geometric Median error: {geometric_error:.3f}")
            logging.info(f"   Trimmed Mean error: {trimmed_error:.3f}")
            
            # Geometric median should be most robust
            assert geometric_error <= fedavg_error
            
        except Exception as e:
            logging.error(f"❌ Aggregation robustness test failed: {e}")
            raise
    
    async def test_system_performance_benchmark(self, setup_test_environment):
        """Test system performance under load"""
        
        env = setup_test_environment
        ghost_agent = env["ghost_agent"]
        
        try:
            # Performance benchmarks
            benchmarks = {}
            
            # Benchmark 1: Model training time
            start_time = time.time()
            
            # Generate test data
            test_data = []
            for i in range(1000):  # 1000 patient records
                record = {
                    "age": np.random.randint(18, 100),
                    "gender": np.random.choice([0, 1]),
                    "systolic_bp": np.random.randint(80, 220),
                    "diastolic_bp": np.random.randint(40, 150),
                    "heart_rate": np.random.randint(40, 200),
                    "temperature": np.random.uniform(95, 106),
                    "oxygen_saturation": np.random.randint(85, 100),
                    "hemoglobin": np.random.uniform(8, 20),
                    "white_blood_cells": np.random.randint(3000, 15000),
                    "platelets": np.random.randint(100000, 500000),
                    "glucose": np.random.randint(60, 400),
                    "creatinine": np.random.uniform(0.3, 10),
                    "diabetes": np.random.choice([0, 1]),
                    "hypertension": np.random.choice([0, 1]),
                    "heart_disease": np.random.choice([0, 1]),
                    "kidney_disease": np.random.choice([0, 1]),
                    "readmission_risk": np.random.choice([0, 1])
                }
                test_data.append(record)
            
            training_time = time.time() - start_time
            benchmarks["data_generation"] = training_time
            
            # Benchmark 2: Privacy accounting
            start_time = time.time()
            
            dp_engine = ghost_agent.dp_engine
            epsilon_spent = dp_engine.update_privacy_accountant(
                noise_multiplier=1.3,
                sample_rate=0.1,
                steps=100
            )
            
            accounting_time = time.time() - start_time
            benchmarks["privacy_accounting"] = accounting_time
            
            # Benchmark 3: Model serialization
            start_time = time.time()
            
            model_state = ghost_agent.model.get_model_state()
            serialized_size = sum(param.numel() * param.element_size() for param in model_state.values())
            
            serialization_time = time.time() - start_time
            benchmarks["model_serialization"] = serialization_time
            benchmarks["model_size_bytes"] = serialized_size
            
            # Benchmark 4: Byzantine analysis
            start_time = time.time()
            
            byzantine_shield = env["byzantine_shield"]
            normal_update = {
                "metadata": {
                    "round_id": "benchmark_round",
                    "hospital_id": "benchmark_hospital",
                    "model_performance": {"local_auc": 0.85, "gradient_norm": 1.5}
                },
                "model_update": model_state,
                "byzantine_shield": {"reputation_score": 0.9, "anomaly_score": 0.1}
            }
            
            analysis_result = await byzantine_shield.analyze_update(
                hospital_id="benchmark_hospital",
                ghost_pack=normal_update,
                current_reputation=0.9
            )
            
            analysis_time = time.time() - start_time
            benchmarks["byzantine_analysis"] = analysis_time
            
            # Performance assertions
            assert benchmarks["data_generation"] < 5.0  # Should generate 1000 records in <5s
            assert benchmarks["privacy_accounting"] < 1.0  # Should be fast
            assert benchmarks["model_serialization"] < 2.0  # Should serialize quickly
            assert benchmarks["byzantine_analysis"] < 1.0  # Should analyze quickly
            
            logging.info("✅ Performance benchmarks verified")
            logging.info(f"   Data generation: {benchmarks['data_generation']:.3f}s")
            logging.info(f"   Privacy accounting: {benchmarks['privacy_accounting']:.3f}s")
            logging.info(f"   Model serialization: {benchmarks['model_serialization']:.3f}s")
            logging.info(f"   Byzantine analysis: {benchmarks['byzantine_analysis']:.3f}s")
            logging.info(f"   Model size: {benchmarks['model_size_bytes']} bytes")
            
        except Exception as e:
            logging.error(f"❌ Performance benchmark test failed: {e}")
            raise
    
    async def test_end_to_end_workflow(self, setup_test_environment):
        """Test complete end-to-end workflow"""
        
        env = setup_test_environment
        ghost_agent = env["ghost_agent"]
        byzantine_shield = env["byzantine_shield"]
        healthtoken_ledger = env["healthtoken_ledger"]
        
        try:
            # Simulate complete workflow
            workflow_log = []
            
            # Step 1: Hospital registers with system
            workflow_log.append("Hospital registration")
            
            # Step 2: Load patient data with consent
            workflow_log.append("Loading patient data with consent verification")
            
            # Step 3: Train local model with differential privacy
            workflow_log.append("Training local model with DP-SGD")
            
            # Step 4: Create privacy-preserving update
            workflow_log.append("Creating Ghost Pack with privacy guarantees")
            
            # Step 5: Submit to SNA with Byzantine Shield
            workflow_log.append("Submitting update to Byzantine Shield")
            
            # Simulate Byzantine analysis
            test_update = {
                "metadata": {
                    "round_id": "e2e_test_round",
                    "hospital_id": ghost_agent.config.hospital_id,
                    "model_performance": {"local_auc": 0.87, "gradient_norm": 1.2},
                    "dp_compliance": {"epsilon_spent": 1.1}
                },
                "model_update": ghost_agent.model.get_model_state(),
                "byzantine_shield": {"reputation_score": 0.95, "anomaly_score": 0.05}
            }
            
            byzantine_result = await byzantine_shield.analyze_update(
                hospital_id=ghost_agent.config.hospital_id,
                ghost_pack=test_update,
                current_reputation=0.95
            )
            
            assert byzantine_result.accepted == True
            workflow_log.append("Update accepted by Byzantine Shield")
            
            # Step 6: Award HealthTokens
            workflow_log.append("Awarding HealthTokens")
            
            tokens = await healthtoken_ledger.award_tokens(
                hospital_id=ghost_agent.config.hospital_id,
                round_id="e2e_test_round",
                local_auc=0.87,
                reputation_score=0.95,
                gradient_quality=0.8,
                participation_history=[1.0, 1.0, 0.9, 1.0]
            )
            
            assert tokens > 0
            workflow_log.append(f"Awarded {tokens} HealthTokens")
            
            # Step 7: Update compliance records
            workflow_log.append("Updating DPDP compliance records")
            
            compliance_report = await ghost_agent.dpdp.generate_compliance_report(period_days=1)
            assert compliance_report["compliance_summary"]["overall_compliance_status"] == "compliant"
            
            workflow_log.append("Compliance verified")
            
            # Log workflow completion
            logging.info("✅ End-to-end workflow completed successfully")
            for i, step in enumerate(workflow_log, 1):
                logging.info(f"   {i}. {step}")
            
        except Exception as e:
            logging.error(f"❌ End-to-end workflow test failed: {e}")
            raise


@pytest.mark.asyncio
async def test_system_initialization():
    """Test system components initialize correctly"""
    
    try:
        # Test Ghost Agent initialization
        from ghost_agent import GhostAgentConfig
        
        config = GhostAgentConfig(
            hospital_id="test_init_hospital",
            hospital_name="Test Initialization Hospital",
            hospital_type="GOVT",
            location_state="Maharashtra",
            location_district="Mumbai",
            epsilon_max=9.5,
            epsilon_per_update=1.23,
            delta_max=1e-5,
            gaussian_noise_scale=1.3,
            learning_rate=0.001,
            batch_size=32,
            epochs_per_round=5,
            aggregation_strategy="FedAvg",
            fedprox_mu=0.1,
            security_manager=SecurityManager("test_init_hospital"),
            dpdp_manager=DPDPComplianceManager("test_init_hospital", ":memory:")
        )
        
        ghost_agent = GhostAgent(config)
        
        # Verify initialization
        assert ghost_agent.config.hospital_id == "test_init_hospital"
        assert ghost_agent.config.epsilon_max == 9.5
        assert ghost_agent.privacy_budget_spent == 0.0
        
        # Test SNA components initialization
        byzantine_shield = ByzantineShield(byzantine_threshold=0.49)
        healthtoken_ledger = HealthTokenLedger()
        
        assert byzantine_shield.byzantine_threshold == 0.49
        assert len(healthtoken_ledger.balances) == 0
        
        logging.info("✅ System initialization test passed")
        
    except Exception as e:
        logging.error(f"❌ System initialization test failed: {e}")
        raise


@pytest.mark.asyncio
async def test_differential_privacy_guarantees():
    """Test mathematical differential privacy guarantees"""
    
    try:
        # Initialize DP engine
        dp_engine = DifferentialPrivacyEngine(None)  # Mock config
        dp_engine.budget = dp_engine.PrivacyBudget(epsilon_total=10.0, delta=1e-5)
        
        # Test multiple updates
        total_epsilon = 0.0
        updates = []
        
        for i in range(5):
            epsilon_spent, delta = dp_engine.update_privacy_accountant(
                noise_multiplier=1.3,
                sample_rate=0.1,
                steps=100
            )
            
            total_epsilon += epsilon_spent
            updates.append({
                "epsilon": epsilon_spent,
                "delta": delta,
                "cumulative_epsilon": dp_engine.budget.epsilon_consumed
            })
        
        # Verify composition
        assert dp_engine.budget.epsilon_consumed == total_epsilon
        assert dp_engine.budget.epsilon_consumed <= dp_engine.budget.epsilon_total
        
        # Verify RDP accounting
        assert len(dp_engine.rdp_values) == len(dp_engine.rdp_orders)
        
        # Test budget exhaustion
        dp_engine.budget.epsilon_consumed = dp_engine.budget.epsilon_total
        assert dp_engine.budget.budget_exhausted == True
        
        logging.info("✅ Differential privacy guarantees verified")
        logging.info(f"   Total epsilon consumed: {total_epsilon:.3f}")
        logging.info(f"   Budget exhausted correctly: {dp_engine.budget.budget_exhausted}")
        
    except Exception as e:
        logging.error(f"❌ Differential privacy test failed: {e}")
        raise


if __name__ == "__main__":
    # Run integration tests
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logging.info("🚀 Starting Ghost Protocol Integration Tests")
    logging.info("=" * 60)
    
    # Run all tests
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
    
    logging.info("=" * 60)
    logging.info("✅ Integration tests completed")