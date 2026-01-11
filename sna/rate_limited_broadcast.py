"""
Module: sna/rate_limited_broadcast.py
Description: Rate-Limited WebSocket Broadcast System

Ultra-Advanced Features:
- Message batching and throttling
- Per-client send queue with backpressure
- Dead client detection and cleanup
- Message priority support
- Comprehensive metrics
"""

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Any, Dict, List, Optional, Set

from fastapi import WebSocket

logger = logging.getLogger("ghost.broadcast")


class MessagePriority(IntEnum):
    """Message priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class BroadcastMessage:
    """Message queued for broadcast."""
    data: Dict[str, Any]
    priority: MessagePriority = MessagePriority.NORMAL
    created_at: float = field(default_factory=time.time)
    message_id: int = 0


@dataclass
class ClientState:
    """State for a connected client."""
    websocket: WebSocket
    client_id: str
    connected_at: float = field(default_factory=time.time)
    messages_sent: int = 0
    messages_dropped: int = 0
    last_send_time: float = 0
    is_slow: bool = False
    pending_queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=100))


class RateLimitedBroadcaster:
    """
    Ultra-Advanced Rate-Limited WebSocket Broadcaster.
    
    Solves broadcast overload by:
    - Throttling messages per client
    - Batching rapid updates
    - Detecting and handling slow clients
    - Priority-based message ordering
    """
    
    def __init__(
        self,
        max_messages_per_second: int = 10,
        batch_interval_ms: int = 100,
        max_pending_per_client: int = 100,
        slow_client_threshold_ms: int = 1000
    ):
        self.max_messages_per_second = max_messages_per_second
        self.batch_interval_ms = batch_interval_ms
        self.max_pending_per_client = max_pending_per_client
        self.slow_client_threshold_ms = slow_client_threshold_ms
        
        # Client management
        self._clients: Dict[str, ClientState] = {}
        self._lock = asyncio.Lock()
        
        # Message batching
        self._message_buffer: List[BroadcastMessage] = []
        self._message_counter = 0
        
        # Background tasks
        self._broadcast_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Metrics
        self.total_messages_broadcast = 0
        self.total_messages_dropped = 0
        self.total_clients_connected = 0
        self.total_clients_disconnected = 0
        
        logger.info(
            f"RateLimitedBroadcaster initialized: "
            f"max_mps={max_messages_per_second}, batch_interval={batch_interval_ms}ms"
        )
    
    async def start(self):
        """Start background broadcast and cleanup tasks."""
        if self._broadcast_task is None:
            self._broadcast_task = asyncio.create_task(self._broadcast_loop())
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Broadcaster started")
    
    async def stop(self):
        """Stop background tasks and disconnect all clients."""
        if self._broadcast_task:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass
            self._broadcast_task = None
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
        
        # Disconnect all clients
        async with self._lock:
            for client in list(self._clients.values()):
                try:
                    await client.websocket.close(code=1001, reason="Server shutdown")
                except Exception:
                    pass
            self._clients.clear()
        
        logger.info("Broadcaster stopped")
    
    async def add_client(self, websocket: WebSocket, client_id: str):
        """Add a new client connection."""
        async with self._lock:
            if client_id in self._clients:
                # Replace existing connection
                old_client = self._clients[client_id]
                try:
                    await old_client.websocket.close(code=1000, reason="New connection")
                except Exception:
                    pass
            
            self._clients[client_id] = ClientState(
                websocket=websocket,
                client_id=client_id
            )
            self.total_clients_connected += 1
            
        logger.info(f"Client {client_id} connected (total: {len(self._clients)})")
    
    async def remove_client(self, client_id: str):
        """Remove a client connection."""
        async with self._lock:
            if client_id in self._clients:
                del self._clients[client_id]
                self.total_clients_disconnected += 1
                
        logger.info(f"Client {client_id} disconnected (remaining: {len(self._clients)})")
    
    async def broadcast(
        self,
        message: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        exclude: Optional[Set[str]] = None
    ):
        """
        Queue a message for broadcast to all clients.
        
        Args:
            message: Data to broadcast
            priority: Message priority
            exclude: Client IDs to exclude
        """
        self._message_counter += 1
        
        async with self._lock:
            self._message_buffer.append(BroadcastMessage(
                data=message,
                priority=priority,
                message_id=self._message_counter
            ))
            
            # If critical priority, flush immediately
            if priority == MessagePriority.CRITICAL:
                await self._flush_buffer(exclude)
    
    async def send_to_client(
        self,
        client_id: str,
        message: Dict[str, Any]
    ) -> bool:
        """
        Send a message to a specific client.
        
        Args:
            client_id: Target client ID
            message: Data to send
            
        Returns:
            True if message was sent/queued
        """
        async with self._lock:
            client = self._clients.get(client_id)
            if not client:
                return False
            
            try:
                # Try to queue message
                client.pending_queue.put_nowait(message)
                return True
            except asyncio.QueueFull:
                client.messages_dropped += 1
                self.total_messages_dropped += 1
                return False
    
    async def _broadcast_loop(self):
        """Background loop for batched broadcasting."""
        while True:
            try:
                await asyncio.sleep(self.batch_interval_ms / 1000)
                await self._flush_buffer()
                await self._send_pending_messages()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Broadcast loop error: {e}")
    
    async def _flush_buffer(self, exclude: Optional[Set[str]] = None):
        """Flush message buffer to all clients."""
        if not self._message_buffer:
            return
        
        # Get and clear buffer
        messages = self._message_buffer.copy()
        self._message_buffer.clear()
        
        # Sort by priority (highest first)
        messages.sort(key=lambda m: m.priority, reverse=True)
        
        # Deduplicate if needed (keep latest of same type)
        seen_types = set()
        unique_messages = []
        for msg in messages:
            msg_type = msg.data.get("type", "unknown")
            if msg_type not in seen_types:
                unique_messages.append(msg)
                seen_types.add(msg_type)
        
        # Queue to each client
        async with self._lock:
            for client in self._clients.values():
                if exclude and client.client_id in exclude:
                    continue
                
                for msg in unique_messages:
                    try:
                        client.pending_queue.put_nowait(msg.data)
                    except asyncio.QueueFull:
                        client.messages_dropped += 1
                        client.is_slow = True
    
    async def _send_pending_messages(self):
        """Send pending messages to each client."""
        async with self._lock:
            clients = list(self._clients.values())
        
        for client in clients:
            await self._send_client_pending(client)
    
    async def _send_client_pending(self, client: ClientState):
        """Send pending messages to a specific client."""
        # Rate limit per client
        now = time.time()
        min_interval = 1.0 / self.max_messages_per_second
        
        messages_sent = 0
        max_per_batch = min(10, self.max_messages_per_second)
        
        while not client.pending_queue.empty() and messages_sent < max_per_batch:
            # Check rate limit
            if client.last_send_time and (now - client.last_send_time) < min_interval:
                break
            
            try:
                message = client.pending_queue.get_nowait()
                
                # Send with timeout
                start = time.time()
                try:
                    await asyncio.wait_for(
                        client.websocket.send_json(message),
                        timeout=self.slow_client_threshold_ms / 1000
                    )
                    
                    elapsed = (time.time() - start) * 1000
                    if elapsed > self.slow_client_threshold_ms / 2:
                        client.is_slow = True
                    else:
                        client.is_slow = False
                    
                    client.messages_sent += 1
                    client.last_send_time = time.time()
                    self.total_messages_broadcast += 1
                    messages_sent += 1
                    
                except asyncio.TimeoutError:
                    client.is_slow = True
                    client.messages_dropped += 1
                    logger.warning(f"Slow client {client.client_id}, message dropped")
                    
            except asyncio.QueueEmpty:
                break
            except Exception as e:
                # Client likely disconnected
                logger.debug(f"Error sending to {client.client_id}: {e}")
                break
    
    async def _cleanup_loop(self):
        """Background loop to clean up dead clients."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                async with self._lock:
                    dead_clients = []
                    
                    for client_id, client in self._clients.items():
                        # Check if client is responsive
                        try:
                            # WebSocket ping/pong is handled at protocol level
                            # Just check if connection is still open
                            if client.websocket.client_state.name != "CONNECTED":
                                dead_clients.append(client_id)
                        except Exception:
                            dead_clients.append(client_id)
                    
                    for client_id in dead_clients:
                        del self._clients[client_id]
                        self.total_clients_disconnected += 1
                        logger.info(f"Cleaned up dead client: {client_id}")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Cleanup loop error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get broadcaster statistics."""
        return {
            "connected_clients": len(self._clients),
            "total_connected": self.total_clients_connected,
            "total_disconnected": self.total_clients_disconnected,
            "total_broadcast": self.total_messages_broadcast,
            "total_dropped": self.total_messages_dropped,
            "pending_buffer": len(self._message_buffer),
            "slow_clients": sum(1 for c in self._clients.values() if c.is_slow)
        }
    
    @property
    def client_count(self) -> int:
        """Get current client count."""
        return len(self._clients)
