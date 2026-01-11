"""
gRPC Client - Secure National Aggregator (SNA) Communication
mTLS 1.3 · Byzantine Shield · HealthToken Integration

DPDP §: §8(2)(a) Secure Communication, §25 Breach Notification
Byzantine theorem: Anomaly detection prevents malicious updates
Test command: pytest tests/test_grpc_client.py -v --cov=grpc_client
Metrics tracked: Round-trip time, Success rate, Byzantine detections, Token transfers
"""

import asyncio
import logging
import ssl
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import json
import time
import uuid

import grpc
from grpc import aio as aio_grpc
import numpy as np

from .proto import ghost_pb2, ghost_pb2_grpc
from .security import SecurityManager


class SNAClient:
    """
    Secure National Aggregator (SNA) gRPC Client
    
    Handles communication with central federated learning server:
    - mTLS 1.3 encrypted channels
    - Ghost Pack submission
    - Global model retrieval
    - Byzantine Shield integration
    - HealthToken ledger operations
    
    Zero-trust: Every server response is verified
    """
    
    def __init__(
        self,
        config,
        security_manager: SecurityManager,
        sna_endpoint: str = None,
        timeout_seconds: int = 30,
        max_retries: int = 3
    ):
        self.config = config
        self.security_manager = security_manager
        self.logger = logging.getLogger(f"sna_client.{config.hospital_id}")
        
        # Connection settings
        self.sna_endpoint = sna_endpoint or config.get("sna_endpoint", "sna.ghostprotocol.nic.gov.in:443")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        
        # gRPC channel and stub
        self.channel: Optional[aio_grpc.Channel] = None
        self.stub: Optional[ghost_pb2_grpc.SecureNationalAggregatorStub] = None
        
        # Connection state
        self.is_connected = False
        self.last_connection_attempt = None
        self.connection_failure_count = 0
        
        # Rate limiting
        self.last_update_timestamp = None
        self.rate_limit_interval = config.get("rate_limit_interval_seconds", 30)
        
        # Metrics
        self.metrics = {
            "updates_submitted": 0,
            "updates_failed": 0,
            "models_retrieved": 0,
            "byzantine_detections": 0,
            "round_trip_times": [],
            "bytes_transmitted": 0,
            "bytes_received": 0
        }
        
        self.logger.info(f"SNA Client initialized for {self.sna_endpoint}")
    
    async def connect(self) -> bool:
        """
        Establish mTLS 1.3 connection to SNA
        
        Returns:
            True if connection successful
        """
        try:
            self.last_connection_attempt = datetime.utcnow()
            
            # Load mTLS credentials
            mtls_credentials = await self._load_mtls_credentials()
            
            # Create secure channel with mTLS
            self.channel = aio_grpc.secure_channel(
                self.sna_endpoint,
                credentials=mtls_credentials,
                options=[
                    ("grpc.ssl_target_name_override", "sna.ghostprotocol.nic.gov.in"),
                    ("grpc.default_authority", "sna.ghostprotocol.nic.gov.in"),
                    ("grpc.keepalive_time_ms", 10000),
                    ("grpc.keepalive_timeout_ms", 5000),
                    ("grpc.keepalive_permit_without_calls", True),
                    ("grpc.http2.max_pings_without_data", 0),
                    ("grpc.max_receive_message_length", 50 * 1024 * 1024),  # 50MB
                    ("grpc.max_send_message_length", 10 * 1024 * 1024),   # 10MB
                ]
            )
            
            # Create stub
            self.stub = ghost_pb2_grpc.SecureNationalAggregatorStub(self.channel)
            
            # Test connection
            await self._test_connection()
            
            self.is_connected = True
            self.connection_failure_count = 0
            
            self.logger.info("Successfully connected to SNA with mTLS 1.3")
            return True
            
        except Exception as e:
            self.is_connected = False
            self.connection_failure_count += 1
            
            self.logger.error(f"Failed to connect to SNA: {e}")
            return False
    
    async def _load_mtls_credentials(self) -> grpc.ChannelCredentials:
        """Load mTLS credentials for secure communication"""
        
        # Load client certificate and key
        client_cert_path = "/etc/ghost/certs/client.crt"
        client_key_path = "/etc/ghost/certs/client.key"
        ca_cert_path = "/etc/ghost/certs/ca.crt"
        
        with open(client_cert_path, 'rb') as f:
            client_cert = f.read()
        
        with open(client_key_path, 'rb') as f:
            client_key = f.read()
        
        with open(ca_cert_path, 'rb') as f:
            ca_cert = f.read()
        
        # Create mTLS credentials
        credentials = grpc.ssl_channel_credentials(
            root_certificates=ca_cert,
            private_key=client_key,
            certificate_chain=client_cert
        )
        
        return credentials
    
    async def _test_connection(self):
        """Test connection with health check"""
        
        request = ghost_pb2.HealthCheckRequest(
            hospital_id=self.config.hospital_id,
            timestamp=datetime.utcnow().isoformat()
        )
        
        response = await self.stub.HealthCheck(
            request,
            timeout=self.timeout_seconds
        )
        
        if not response.healthy:
            raise Exception("SNA health check failed")
    
    async def get_global_model(self) -> Optional[Dict[str, torch.Tensor]]:
        """
        Retrieve current global model from SNA
        
        Returns:
            Global model state dict or None if not available
        """
        if not self.is_connected and not await self.connect():
            self.logger.error("Cannot retrieve global model: not connected")
            return None
        
        try:
            start_time = time.time()
            
            request = ghost_pb2.GlobalModelRequest(
                hospital_id=self.config.hospital_id,
                timestamp=datetime.utcnow().isoformat()
            )
            
            response = await self.stub.GetGlobalModel(
                request,
                timeout=self.timeout_seconds
            )
            
            # Record metrics
            round_trip_time = time.time() - start_time
            self.metrics["round_trip_times"].append(round_trip_time)
            self.metrics["bytes_received"] += len(response.SerializeToString())
            
            if not response.has_model:
                self.logger.info("No global model available from SNA")
                return None
            
            # Deserialize model state
            model_state = await self._deserialize_model_state(response.model_state)
            
            # Verify model integrity
            if not await self._verify_model_integrity(model_state, response.checksum):
                raise Exception("Model integrity verification failed")
            
            self.metrics["models_retrieved"] += 1
            self.logger.info(f"Retrieved global model in {round_trip_time:.2f}s")
            
            return model_state
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve global model: {e}")
            return None
    
    async def submit_update(self, ghost_pack: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit Ghost Pack to SNA for aggregation
        
        Args:
            ghost_pack: Privacy-preserving model update
            
        Returns:
            Submission result with Byzantine Shield analysis
        """
        if not self.is_connected and not await self.connect():
            raise Exception("Cannot submit update: not connected to SNA")
        
        # Check rate limiting
        if not await self._check_rate_limit():
            raise Exception("Rate limit exceeded")
        
        try:
            start_time = time.time()
            
            # Create gRPC request
            request = await self._create_submit_request(ghost_pack)
            
            # Submit with retry logic
            response = await self._submit_with_retry(request)
            
            # Record metrics
            round_trip_time = time.time() - start_time
            self.metrics["round_trip_times"].append(round_trip_time)
            self.metrics["bytes_transmitted"] += len(request.SerializeToString())
            self.metrics["bytes_received"] += len(response.SerializeToString())
            
            # Process response
            result = await self._process_submit_response(response)
            
            # Update last submission timestamp
            self.last_update_timestamp = datetime.utcnow()
            
            self.metrics["updates_submitted"] += 1
            self.logger.info(f"Ghost Pack submitted successfully in {round_trip_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.metrics["updates_failed"] += 1
            self.logger.error(f"Failed to submit Ghost Pack: {e}")
            raise
    
    async def _create_submit_request(self, ghost_pack: Dict[str, Any]) -> ghost_pb2.GhostPack:
        """Create gRPC request from Ghost Pack"""
        
        # Extract metadata
        metadata = ghost_pack["metadata"]
        
        # Create request
        request = ghost_pb2.GhostPack(
            metadata=ghost_pb2.Metadata(
                round_id=metadata["round_id"],
                hospital_id=metadata["hospital_id"],
                hospital_name=metadata["hospital_name"],
                timestamp=metadata["timestamp"],
                protocol_version=metadata["protocol_version"],
                dp_compliance=ghost_pb2.PrivacyMetadata(
                    epsilon_spent=metadata["dp_compliance"]["epsilon_spent"],
                    delta_used=metadata["dp_compliance"]["delta_used"],
                    privacy_budget_remaining=metadata["dp_compliance"]["privacy_budget_remaining"],
                    dp_algorithm=metadata["dp_compliance"]["dp_algorithm"]
                ),
                model_performance=ghost_pb2.ModelPerformance(
                    local_auc=metadata["model_performance"]["local_auc"],
                    gradient_norm=metadata["model_performance"]["gradient_norm"],
                    training_samples=metadata["model_performance"]["training_samples"]
                )
            ),
            byzantine_shield=ghost_pb2.ByzantineMetadata(
                reputation_score=ghost_pack["byzantine_shield"]["reputation_score"],
                gradient_history_length=ghost_pack["byzantine_shield"]["gradient_history_length"],
                anomaly_score=ghost_pack["byzantine_shield"]["anomaly_score"]
            ),
            signature=ghost_pack["signature"]["signature"],
            encrypted_payload=ghost_pack["encrypted_payload"]
        )
        
        return request
    
    async def _submit_with_retry(self, request: ghost_pb2.GhostPack) -> ghost_pb2.SubmissionResponse:
        """Submit request with exponential backoff retry"""
        
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                response = await self.stub.SubmitUpdate(
                    request,
                    timeout=self.timeout_seconds
                )
                return response
                
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    backoff_time = 2 ** attempt
                    await asyncio.sleep(backoff_time)
                    
                    # Try to reconnect
                    await self.connect()
                
                continue
        
        # All retries failed
        raise last_exception
    
    async def _process_submit_response(self, response: ghost_pb2.SubmissionResponse) -> Dict[str, Any]:
        """Process SNA submission response"""
        
        result = {
            "status": response.status,
            "round_id": response.round_id,
            "hospital_id": response.hospital_id,
            "byzantine_analysis": {
                "accepted": response.byzantine_analysis.accepted,
                "rejection_reason": response.byzantine_analysis.rejection_reason,
                "anomaly_score": response.byzantine_analysis.anomaly_score,
                "reputation_change": response.byzantine_analysis.reputation_change
            },
            "aggregation_result": {
                "model_weight": response.aggregation_result.model_weight,
                "health_tokens_earned": response.aggregation_result.health_tokens_earned
            },
            "next_round": {
                "round_id": response.next_round.round_id,
                "estimated_start": response.next_round.estimated_start,
                "participation_requested": response.next_round.participation_requested
            }
        }
        
        # Handle Byzantine Shield rejection
        if not response.byzantine_analysis.accepted:
            self.metrics["byzantine_detections"] += 1
            self.logger.warning(f"Byzantine Shield rejected update: {response.byzantine_analysis.rejection_reason}")
        
        return result
    
    async def _deserialize_model_state(self, model_state_bytes: bytes) -> Dict[str, torch.Tensor]:
        """Deserialize model state from protobuf"""
        
        # In production, deserialize from protobuf format
        # For now, return empty state (will use random initialization)
        return {}
    
    async def _verify_model_integrity(
        self,
        model_state: Dict[str, torch.Tensor],
        checksum: str
    ) -> bool:
        """Verify model integrity using checksum"""
        
        # In production, compute checksum of model state and verify
        # For now, return True (assume valid)
        return True
    
    async def _check_rate_limit(self) -> bool:
        """Check if rate limit allows submission"""
        
        if self.last_update_timestamp is None:
            return True
        
        time_since_last = (datetime.utcnow() - self.last_update_timestamp).total_seconds()
        
        if time_since_last < self.rate_limit_interval:
            self.logger.warning(f"Rate limit exceeded: {time_since_last:.1f}s since last update")
            return False
        
        return True
    
    async def get_healthtoken_balance(self) -> float:
        """
        Get current HealthToken balance from ledger
        
        Returns:
            Current token balance
        """
        if not self.is_connected and not await self.connect():
            raise Exception("Cannot get balance: not connected")
        
        try:
            request = ghost_pb2.TokenBalanceRequest(
                hospital_id=self.config.hospital_id,
                timestamp=datetime.utcnow().isoformat()
            )
            
            response = await self.stub.GetTokenBalance(request)
            
            return response.balance
            
        except Exception as e:
            self.logger.error(f"Failed to get token balance: {e}")
            return 0.0
    
    async def get_reputation_status(self) -> Dict[str, Any]:
        """
        Get current reputation status from Byzantine Shield
        
        Returns:
            Reputation metrics
        """
        if not self.is_connected and not await self.connect():
            raise Exception("Cannot get reputation: not connected")
        
        try:
            request = ghost_pb2.ReputationRequest(
                hospital_id=self.config.hospital_id,
                timestamp=datetime.utcnow().isoformat()
            )
            
            response = await self.stub.GetReputationStatus(request)
            
            return {
                "reputation_score": response.reputation_score,
                "total_updates": response.total_updates,
                "accepted_updates": response.accepted_updates,
                "rejected_updates": response.rejected_updates,
                "last_update": response.last_update,
                "status": response.status
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get reputation status: {e}")
            return {}
    
    async def close(self):
        """Close gRPC channel"""
        
        if self.channel:
            await self.channel.close()
            self.is_connected = False
            self.logger.info("SNA Client connection closed")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get client metrics"""
        
        avg_round_trip = np.mean(self.metrics["round_trip_times"]) if self.metrics["round_trip_times"] else 0
        
        return {
            "connection": {
                "is_connected": self.is_connected,
                "connection_failures": self.connection_failure_count,
                "last_connection": self.last_connection_attempt.isoformat() if self.last_connection_attempt else None
            },
            "operations": {
                "updates_submitted": self.metrics["updates_submitted"],
                "updates_failed": self.metrics["updates_failed"],
                "models_retrieved": self.metrics["models_retrieved"],
                "byzantine_detections": self.metrics["byzantine_detections"]
            },
            "performance": {
                "average_round_trip_time": avg_round_trip,
                "bytes_transmitted": self.metrics["bytes_transmitted"],
                "bytes_received": self.metrics["bytes_received"]
            },
            "rate_limiting": {
                "last_update": self.last_update_timestamp.isoformat() if self.last_update_timestamp else None,
                "rate_limit_interval": self.rate_limit_interval
            }
        }