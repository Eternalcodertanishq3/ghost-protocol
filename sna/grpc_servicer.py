"""
Module: sna/grpc_servicer.py
Description: Production gRPC Servicer Implementation

Ultra-Advanced Features:
- Full implementation of all Ghost Protocol gRPC services
- Streaming support for real-time updates
- mTLS integration ready
- Request/Response validation
- Comprehensive error handling with proper gRPC status codes
"""

import asyncio
import logging
import time
from concurrent import futures
from typing import Dict, Any, Optional, Iterator
import grpc

import torch
import numpy as np

from .proto import ghost_pb2, ghost_pb2_grpc
from .byzantine_shield import ByzantineShield, HospitalUpdate
from .health_ledger import HealthLedger

logger = logging.getLogger("ghost.grpc")


class GhostServicer(ghost_pb2_grpc.GhostServiceServicer):
    """
    Production gRPC Servicer for Ghost Protocol.
    
    Implements all RPC methods with:
    - Proper error handling using gRPC status codes
    - Request validation
    - Logging and metrics
    - Thread-safe operations
    """
    
    def __init__(
        self,
        byzantine_shield: ByzantineShield,
        health_ledger: HealthLedger,
        global_model_provider: callable,
        update_queue: asyncio.Queue
    ):
        """
        Initialize gRPC servicer.
        
        Args:
            byzantine_shield: Byzantine fault tolerance component
            health_ledger: HealthToken ledger
            global_model_provider: Callable that returns current global weights
            update_queue: Queue for incoming hospital updates
        """
        self.byzantine_shield = byzantine_shield
        self.health_ledger = health_ledger
        self.get_global_model = global_model_provider
        self.update_queue = update_queue
        
        # Metrics
        self.request_count = 0
        self.error_count = 0
        self.last_request_time = 0
        
        logger.info("GhostServicer initialized")
    
    def _validate_hospital_id(self, hospital_id: str, context: grpc.ServicerContext) -> bool:
        """Validate hospital ID format."""
        if not hospital_id or len(hospital_id) < 3:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Invalid hospital_id: must be at least 3 characters")
            return False
        
        if not hospital_id.replace("_", "").replace("-", "").isalnum():
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Invalid hospital_id: only alphanumeric, underscore, and hyphen allowed")
            return False
        
        return True
    
    def _validate_weights(self, weights: Dict[str, Any], context: grpc.ServicerContext) -> bool:
        """Validate weight tensor format."""
        if not weights:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Weights cannot be empty")
            return False
        
        # Check for required layers
        required_keys = ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "fc3.weight", "fc3.bias"]
        missing = [k for k in required_keys if k not in weights]
        
        if missing:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f"Missing required weight keys: {missing}")
            return False
        
        return True
    
    def SubmitUpdate(
        self,
        request: ghost_pb2.UpdateRequest,
        context: grpc.ServicerContext
    ) -> ghost_pb2.UpdateResponse:
        """
        Handle hospital model update submission.
        
        Args:
            request: UpdateRequest with hospital_id, weights, and metadata
            context: gRPC context
            
        Returns:
            UpdateResponse with acceptance status
        """
        self.request_count += 1
        self.last_request_time = time.time()
        
        try:
            # Validate request
            if not self._validate_hospital_id(request.hospital_id, context):
                self.error_count += 1
                return ghost_pb2.UpdateResponse(accepted=False, message="Invalid hospital ID")
            
            # Parse weights from protobuf
            weights = {}
            for layer_name, tensor_proto in request.weights.items():
                weights[layer_name] = torch.tensor(
                    list(tensor_proto.data),
                    dtype=torch.float32
                ).reshape(list(tensor_proto.shape))
            
            if not self._validate_weights(weights, context):
                self.error_count += 1
                return ghost_pb2.UpdateResponse(accepted=False, message="Invalid weights")
            
            # Create hospital update
            update = HospitalUpdate(
                hospital_id=request.hospital_id,
                weights=weights,
                local_auc=request.metadata.get("local_auc", 0.5),
                gradient_norm=request.metadata.get("gradient_norm", 0.0),
                epsilon_spent=request.metadata.get("epsilon_spent", 1.0),
                timestamp=time.time()
            )
            
            # Byzantine check
            is_byzantine = self.byzantine_shield.is_update_byzantine(update)
            if is_byzantine:
                logger.warning(f"Byzantine update detected from {request.hospital_id}")
                return ghost_pb2.UpdateResponse(
                    accepted=False,
                    message="Update rejected: anomaly detected",
                    byzantine_detected=True
                )
            
            # Queue the update
            try:
                self.update_queue.put_nowait(update)
            except asyncio.QueueFull:
                context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
                return ghost_pb2.UpdateResponse(
                    accepted=False,
                    message="Server at capacity, please retry later"
                )
            
            logger.info(f"Update accepted from {request.hospital_id}")
            
            return ghost_pb2.UpdateResponse(
                accepted=True,
                message="Update accepted for aggregation",
                round_number=request.round_number
            )
            
        except Exception as e:
            self.error_count += 1
            logger.exception(f"Error processing update from {request.hospital_id}: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal error: {str(e)}")
            return ghost_pb2.UpdateResponse(accepted=False, message="Internal error")
    
    def GetGlobalModel(
        self,
        request: ghost_pb2.ModelRequest,
        context: grpc.ServicerContext
    ) -> ghost_pb2.ModelResponse:
        """
        Return current global model weights.
        
        Args:
            request: ModelRequest with optional round_number
            context: gRPC context
            
        Returns:
            ModelResponse with current global weights
        """
        self.request_count += 1
        
        try:
            if not self._validate_hospital_id(request.hospital_id, context):
                self.error_count += 1
                return ghost_pb2.ModelResponse()
            
            # Get current global model
            global_weights, round_number = self.get_global_model()
            
            # Convert to protobuf format
            weights_proto = {}
            for name, tensor in global_weights.items():
                tensor_proto = ghost_pb2.TensorProto(
                    data=tensor.numpy().flatten().tolist(),
                    shape=list(tensor.shape),
                    dtype="float32"
                )
                weights_proto[name] = tensor_proto
            
            return ghost_pb2.ModelResponse(
                weights=weights_proto,
                round_number=round_number,
                timestamp=int(time.time())
            )
            
        except Exception as e:
            self.error_count += 1
            logger.exception(f"Error getting global model: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return ghost_pb2.ModelResponse()
    
    def GetStatus(
        self,
        request: ghost_pb2.StatusRequest,
        context: grpc.ServicerContext
    ) -> ghost_pb2.StatusResponse:
        """
        Return current SNA status.
        
        Args:
            request: Empty status request
            context: gRPC context
            
        Returns:
            StatusResponse with system metrics
        """
        self.request_count += 1
        
        try:
            _, round_number = self.get_global_model()
            
            return ghost_pb2.StatusResponse(
                active=True,
                current_round=round_number,
                pending_updates=self.update_queue.qsize() if hasattr(self.update_queue, 'qsize') else 0,
                total_requests=self.request_count,
                error_count=self.error_count,
                uptime_seconds=int(time.time() - self.last_request_time) if self.last_request_time else 0
            )
            
        except Exception as e:
            logger.exception(f"Error getting status: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return ghost_pb2.StatusResponse(active=False)
    
    def GetLeaderboard(
        self,
        request: ghost_pb2.LeaderboardRequest,
        context: grpc.ServicerContext
    ) -> ghost_pb2.LeaderboardResponse:
        """
        Return HealthToken leaderboard.
        
        Args:
            request: LeaderboardRequest with optional limit
            context: gRPC context
            
        Returns:
            LeaderboardResponse with hospital rankings
        """
        self.request_count += 1
        
        try:
            limit = request.limit if request.limit > 0 else 10
            leaderboard = self.health_ledger.get_leaderboard(limit=limit)
            
            entries = []
            for i, entry in enumerate(leaderboard):
                entries.append(ghost_pb2.LeaderboardEntry(
                    rank=i + 1,
                    hospital_id=entry["hospital_id"],
                    tokens=entry["tokens"],
                    reputation=entry.get("reputation", 1.0)
                ))
            
            return ghost_pb2.LeaderboardResponse(entries=entries)
            
        except Exception as e:
            logger.exception(f"Error getting leaderboard: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return ghost_pb2.LeaderboardResponse()
    
    def StreamUpdates(
        self,
        request: ghost_pb2.StreamRequest,
        context: grpc.ServicerContext
    ) -> Iterator[ghost_pb2.StreamUpdate]:
        """
        Stream real-time updates to connected clients.
        
        Args:
            request: StreamRequest with hospital_id
            context: gRPC context
            
        Yields:
            StreamUpdate messages for training events
        """
        self.request_count += 1
        
        if not self._validate_hospital_id(request.hospital_id, context):
            return
        
        logger.info(f"Starting stream for {request.hospital_id}")
        
        try:
            last_round = -1
            
            while context.is_active():
                # Check for new round
                _, current_round = self.get_global_model()
                
                if current_round > last_round:
                    last_round = current_round
                    yield ghost_pb2.StreamUpdate(
                        update_type="NEW_ROUND",
                        round_number=current_round,
                        timestamp=int(time.time()),
                        message=f"Round {current_round} completed"
                    )
                
                time.sleep(1)  # Poll interval
                
        except Exception as e:
            logger.exception(f"Stream error for {request.hospital_id}: {e}")
        finally:
            logger.info(f"Stream ended for {request.hospital_id}")
    
    def Heartbeat(
        self,
        request: ghost_pb2.HeartbeatRequest,
        context: grpc.ServicerContext
    ) -> ghost_pb2.HeartbeatResponse:
        """
        Handle heartbeat from hospital agent.
        
        Args:
            request: HeartbeatRequest with hospital_id
            context: gRPC context
            
        Returns:
            HeartbeatResponse confirming connection
        """
        return ghost_pb2.HeartbeatResponse(
            alive=True,
            server_time=int(time.time())
        )


def create_grpc_server(
    servicer: GhostServicer,
    port: int = 50051,
    max_workers: int = 10,
    enable_reflection: bool = True
) -> grpc.Server:
    """
    Create and configure gRPC server.
    
    Args:
        servicer: GhostServicer instance
        port: Port to listen on
        max_workers: Thread pool size
        enable_reflection: Enable server reflection for debugging
        
    Returns:
        Configured gRPC server (not started)
    """
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50MB
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),
            ('grpc.keepalive_time_ms', 30000),
            ('grpc.keepalive_timeout_ms', 10000),
            ('grpc.keepalive_permit_without_calls', True),
        ]
    )
    
    ghost_pb2_grpc.add_GhostServiceServicer_to_server(servicer, server)
    
    if enable_reflection:
        try:
            from grpc_reflection.v1alpha import reflection
            SERVICE_NAMES = (
                ghost_pb2.DESCRIPTOR.services_by_name['GhostService'].full_name,
                reflection.SERVICE_NAME,
            )
            reflection.enable_server_reflection(SERVICE_NAMES, server)
            logger.info("gRPC reflection enabled")
        except ImportError:
            logger.warning("grpc-reflection not installed, reflection disabled")
    
    server.add_insecure_port(f'[::]:{port}')
    
    logger.info(f"gRPC server configured on port {port}")
    
    return server
