#!/usr/bin/env python3
"""
Ghost Protocol - Main Entry Point
DPDP-Safe Federated Learning Infrastructure for India's Healthcare System

Usage:
    python main.py --mode [agent|sna|test] --config config.yaml

DPDP §: Complete Act compliance with automated enforcement
Byzantine theorem: Production-grade Byzantine fault tolerance
Metrics tracked: System health, Performance metrics, Compliance status
"""

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from ghost_agent import GhostAgent, GhostAgentConfig
from ghost_agent.security import SecurityManager
from ghost_agent.compliance import DPDPComplianceManager
from sna import SNAServer, ByzantineShield, HealthTokenLedger


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging for Ghost Protocol"""
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('/var/log/ghost/ghost_protocol.log')
        ]
    )
    
    # Suppress verbose logs from libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('web3').setLevel(logging.WARNING)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load Ghost Protocol configuration"""
    
    import yaml
    
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


async def run_ghost_agent(config: Dict[str, Any]) -> None:
    """Run Ghost Agent (hospital-side)"""
    
    logging.info("🚀 Starting Ghost Agent - Hospital-side Federated Learning Node")
    
    # Initialize security manager
    security_manager = SecurityManager(
        hospital_id=config["hospital"]["id"],
        vault_addr=config.get("vault", {}).get("address")
    )
    
    # Initialize DPDP compliance manager
    dpdp_manager = DPDPComplianceManager(
        hospital_id=config["hospital"]["id"],
        db_path=config["compliance"]["db_path"]
    )
    
    # Create agent configuration
    agent_config = GhostAgentConfig(
        hospital_id=config["hospital"]["id"],
        hospital_name=config["hospital"]["name"],
        hospital_type=config["hospital"]["type"],
        location_state=config["hospital"]["location"]["state"],
        location_district=config["hospital"]["location"]["district"],
        epsilon_max=config["privacy"]["epsilon_max"],
        epsilon_per_update=config["privacy"]["epsilon_per_update"],
        delta_max=config["privacy"]["delta_max"],
        gaussian_noise_scale=config["privacy"]["gaussian_noise_scale"],
        learning_rate=config["training"]["learning_rate"],
        batch_size=config["training"]["batch_size"],
        epochs_per_round=config["training"]["epochs_per_round"],
        aggregation_strategy=config["training"]["aggregation_strategy"],
        fedprox_mu=config["training"].get("fedprox_mu", 0.1),
        security_manager=security_manager,
        dpdp_manager=dpdp_manager
    )
    
    # Initialize Ghost Agent
    ghost_agent = GhostAgent(agent_config)
    
    # Setup graceful shutdown
    def signal_handler(signum, frame):
        logging.info("Received shutdown signal, stopping Ghost Agent...")
        asyncio.create_task(ghost_agent.shutdown())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start federated learning loop
        round_count = 0
        while True:
            logging.info(f"Starting federated learning round {round_count + 1}")
            
            try:
                result = await ghost_agent.start_federated_learning_round()
                
                if result["status"] == "success":
                    logging.info(f"✅ Round {result['round_id']} completed successfully")
                    logging.info(f"   Privacy used: ε={result['privacy_budget_used']:.3f}")
                    logging.info(f"   Local AUC: {result['local_auc']:.3f}")
                    logging.info(f"   Training samples: {result['training_samples']}")
                    round_count += 1
                else:
                    logging.warning(f"⚠️ Round completed with status: {result['status']}")
                
            except Exception as e:
                logging.error(f"❌ Round failed: {e}")
                # Continue to next round despite failure
            
            # Wait before next round (configurable)
            await asyncio.sleep(config["training"].get("round_interval_seconds", 300))
            
    except Exception as e:
        logging.error(f"Ghost Agent failed: {e}")
        raise


async def run_sna_server(config: Dict[str, Any]) -> None:
    """Run Secure National Aggregator (central server)"""
    
    logging.info("🚀 Starting Secure National Aggregator (SNA)")
    
    # Initialize SNA components
    byzantine_shield = ByzantineShield(
        byzantine_threshold=config["byzantine_shield"]["threshold"],
        z_score_threshold=config["byzantine_shield"]["z_score_threshold"]
    )
    
    healthtoken_ledger = HealthTokenLedger(
        polygon_rpc_url=config.get("blockchain", {}).get("polygon_rpc_url"),
        contract_address=config.get("blockchain", {}).get("contract_address"),
        private_key=config.get("blockchain", {}).get("private_key")
    )
    
    # Start ledger background processes
    await healthtoken_ledger.start()
    
    # Initialize SNA server
    sna_server = SNAServer(
        host=config["sna"]["host"],
        port=config["sna"]["port"],
        max_hospitals=config["sna"]["max_hospitals"],
        aggregation_strategy=AggregationStrategy(config["sna"]["aggregation_strategy"]),
        byzantine_threshold=config["byzantine_shield"]["threshold"]
    )
    
    # Setup graceful shutdown
    def signal_handler(signum, frame):
        logging.info("Received shutdown signal, stopping SNA...")
        asyncio.create_task(sna_server.stop())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start SNA server
        await sna_server.start()
        
    except Exception as e:
        logging.error(f"SNA Server failed: {e}")
        raise


async def run_tests(config: Dict[str, Any]) -> None:
    """Run comprehensive test suite"""
    
    logging.info("🧪 Starting Ghost Protocol Test Suite")
    
    try:
        # Import and run integration tests
        from tests.test_integration import TestGhostProtocolIntegration
        
        test_suite = TestGhostProtocolIntegration()
        
        # Setup test environment
        test_env = await test_suite.setup_test_environment()
        
        # Run all integration tests
        tests = [
            ("Complete FL Round", test_suite.test_complete_federated_learning_round),
            ("DPDP Compliance", test_suite.test_dpdp_compliance_enforcement),
            ("Byzantine Shield Detection", test_suite.test_byzantine_shield_detection),
            ("HealthToken Distribution", test_suite.test_healthtoken_distribution),
            ("Privacy Budget Enforcement", test_suite.test_privacy_budget_enforcement),
            ("Aggregation Robustness", test_suite.test_aggregation_robustness),
            ("Performance Benchmark", test_suite.test_system_performance_benchmark),
            ("End-to-End Workflow", test_suite.test_end_to_end_workflow)
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            try:
                logging.info(f"Running test: {test_name}")
                await test_func(test_env)
                logging.info(f"✅ {test_name} PASSED")
                passed += 1
                
            except Exception as e:
                logging.error(f"❌ {test_name} FAILED: {e}")
                failed += 1
        
        # Print summary
        logging.info("=" * 60)
        logging.info(f"Test Summary: {passed} passed, {failed} failed")
        
        if failed == 0:
            logging.info("🎉 All tests passed! Ghost Protocol is ready for production.")
        else:
            logging.warning(f"⚠️  {failed} tests failed. Please review before deployment.")
            sys.exit(1)
            
    except Exception as e:
        logging.error(f"Test suite failed: {e}")
        sys.exit(1)


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description="Ghost Protocol - DPDP-Safe Federated Learning Infrastructure"
    )
    
    parser.add_argument(
        "--mode",
        choices=["agent", "sna", "test"],
        required=True,
        help="Operation mode: agent (hospital-side), sna (central server), test (validation)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Log startup
    logging.info("=" * 60)
    logging.info("Ghost Protocol v1.0 - DPDP-Safe Federated Learning")
    logging.info("PhD: Dr. Arya Verma - Privacy-Preserving Federated Learning")
    logging.info("Compliance: DPDP Act 2023")
    logging.info("Scale: 50,000+ hospitals, 1.4B+ patient records")
    logging.info("=" * 60)
    
    # Run based on mode
    try:
        if args.mode == "agent":
            asyncio.run(run_ghost_agent(config))
            
        elif args.mode == "sna":
            asyncio.run(run_sna_server(config))
            
        elif args.mode == "test":
            asyncio.run(run_tests(config))
            
    except KeyboardInterrupt:
        logging.info("Received keyboard interrupt, shutting down...")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()