"""
Module: sna/bounded_queue.py
Description: Bounded Update Queue with TTL and Backpressure

Ultra-Advanced Features:
- Maximum size limit to prevent memory exhaustion
- Time-to-live (TTL) for automatic stale update eviction
- Backpressure signaling when queue is full
- Priority queue support based on hospital reputation
- Comprehensive metrics and monitoring
"""

import asyncio
import heapq
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Generic, List, Optional, TypeVar

logger = logging.getLogger("ghost.queue")

T = TypeVar("T")


@dataclass
class QueueMetrics:
    """Metrics for bounded queue operations."""
    total_enqueued: int = 0
    total_dequeued: int = 0
    total_evicted_ttl: int = 0
    total_evicted_capacity: int = 0
    total_rejected: int = 0
    max_queue_size_observed: int = 0
    average_latency_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_enqueued": self.total_enqueued,
            "total_dequeued": self.total_dequeued,
            "total_evicted_ttl": self.total_evicted_ttl,
            "total_evicted_capacity": self.total_evicted_capacity,
            "total_rejected": self.total_rejected,
            "max_queue_size_observed": self.max_queue_size_observed,
            "average_latency_ms": round(self.average_latency_ms, 2)
        }


@dataclass(order=True)
class PriorityItem:
    """Wrapper for priority queue items."""
    priority: float
    timestamp: float = field(compare=False)
    item: Any = field(compare=False)


class BoundedUpdateQueue(Generic[T]):
    """
    Ultra-Advanced Bounded Queue for Hospital Updates.
    
    Solves memory leak problem by:
    - Enforcing maximum capacity
    - Auto-evicting stale updates (TTL-based)
    - Providing backpressure signals
    - Supporting priority-based ordering
    
    Thread-safe for async operations.
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        ttl_seconds: int = 3600,  # 1 hour default
        use_priority: bool = False,
        eviction_strategy: str = "oldest"  # "oldest" or "lowest_priority"
    ):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.use_priority = use_priority
        self.eviction_strategy = eviction_strategy
        
        # Storage
        if use_priority:
            self._queue: List[PriorityItem] = []  # Heap
        else:
            self._queue: deque = deque(maxlen=None)  # We manage size manually
        
        # Metadata for TTL tracking
        self._timestamps: Dict[int, float] = {}  # id(item) -> enqueue time
        
        # Concurrency control
        self._lock = asyncio.Lock()
        
        # Metrics
        self.metrics = QueueMetrics()
        
        # Background cleanup
        self._cleanup_task: Optional[asyncio.Task] = None
        
        logger.info(f"BoundedUpdateQueue initialized: max_size={max_size}, ttl={ttl_seconds}s")
    
    async def start(self):
        """Start background cleanup task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.debug("Queue cleanup task started")
    
    async def stop(self):
        """Stop background cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
    
    async def enqueue(
        self,
        item: T,
        priority: float = 0.0
    ) -> bool:
        """
        Add item to queue.
        
        Args:
            item: Item to enqueue
            priority: Priority (higher = more important, used if use_priority=True)
            
        Returns:
            True if enqueued, False if rejected (queue full + no evictable items)
        """
        async with self._lock:
            now = time.time()
            
            # Evict expired items first
            evicted = await self._evict_expired_internal(now)
            
            # Check capacity
            current_size = len(self._queue)
            
            if current_size >= self.max_size:
                # Try to evict based on strategy
                if self.eviction_strategy == "oldest":
                    evicted_item = await self._evict_oldest_internal()
                elif self.eviction_strategy == "lowest_priority":
                    evicted_item = await self._evict_lowest_priority_internal()
                else:
                    evicted_item = None
                
                if evicted_item is None:
                    # Cannot evict anything, reject
                    self.metrics.total_rejected += 1
                    logger.warning("Queue full, rejecting new item (backpressure)")
                    return False
                
                self.metrics.total_evicted_capacity += 1
            
            # Enqueue
            if self.use_priority:
                # Max-heap: negate priority so highest priority is first
                heapq.heappush(
                    self._queue,
                    PriorityItem(priority=-priority, timestamp=now, item=item)
                )
            else:
                self._queue.append({
                    "item": item,
                    "timestamp": now,
                    "priority": priority
                })
            
            self._timestamps[id(item)] = now
            self.metrics.total_enqueued += 1
            self.metrics.max_queue_size_observed = max(
                self.metrics.max_queue_size_observed,
                len(self._queue)
            )
            
            return True
    
    async def dequeue(self) -> Optional[T]:
        """
        Remove and return the next item from queue.
        
        Returns:
            Item or None if queue is empty
        """
        async with self._lock:
            if not self._queue:
                return None
            
            if self.use_priority:
                priority_item = heapq.heappop(self._queue)
                item = priority_item.item
                enqueue_time = priority_item.timestamp
            else:
                entry = self._queue.popleft()
                item = entry["item"]
                enqueue_time = entry["timestamp"]
            
            # Calculate latency
            latency = (time.time() - enqueue_time) * 1000  # ms
            total = self.metrics.total_dequeued
            self.metrics.average_latency_ms = (
                (self.metrics.average_latency_ms * total + latency) / (total + 1)
            )
            
            self.metrics.total_dequeued += 1
            
            # Cleanup timestamp tracking
            if id(item) in self._timestamps:
                del self._timestamps[id(item)]
            
            return item
    
    async def dequeue_batch(self, max_items: int) -> List[T]:
        """
        Dequeue multiple items at once.
        
        Args:
            max_items: Maximum number of items to dequeue
            
        Returns:
            List of items (may be fewer than max_items)
        """
        items = []
        for _ in range(max_items):
            item = await self.dequeue()
            if item is None:
                break
            items.append(item)
        return items
    
    async def peek(self) -> Optional[T]:
        """Peek at next item without removing it."""
        async with self._lock:
            if not self._queue:
                return None
            
            if self.use_priority:
                return self._queue[0].item
            else:
                return self._queue[0]["item"]
    
    async def _evict_expired_internal(self, now: float) -> int:
        """Internal method to evict expired items (must hold lock)."""
        if not self._queue:
            return 0
        
        cutoff = now - self.ttl_seconds
        evicted_count = 0
        
        if self.use_priority:
            # For priority queue, we need to rebuild without expired items
            valid_items = []
            for priority_item in self._queue:
                if priority_item.timestamp >= cutoff:
                    valid_items.append(priority_item)
                else:
                    evicted_count += 1
            
            if evicted_count > 0:
                self._queue = valid_items
                heapq.heapify(self._queue)
        else:
            # For deque, remove from front while expired
            while self._queue and self._queue[0]["timestamp"] < cutoff:
                entry = self._queue.popleft()
                if id(entry["item"]) in self._timestamps:
                    del self._timestamps[id(entry["item"])]
                evicted_count += 1
        
        self.metrics.total_evicted_ttl += evicted_count
        
        if evicted_count > 0:
            logger.debug(f"Evicted {evicted_count} expired items from queue")
        
        return evicted_count
    
    async def _evict_oldest_internal(self) -> Optional[T]:
        """Evict oldest item (must hold lock)."""
        if not self._queue:
            return None
        
        if self.use_priority:
            # Find and remove oldest by timestamp
            if not self._queue:
                return None
            oldest_idx = min(range(len(self._queue)), key=lambda i: self._queue[i].timestamp)
            oldest = self._queue.pop(oldest_idx)
            heapq.heapify(self._queue)
            return oldest.item
        else:
            entry = self._queue.popleft()
            return entry["item"]
    
    async def _evict_lowest_priority_internal(self) -> Optional[T]:
        """Evict lowest priority item (must hold lock)."""
        if not self._queue:
            return None
        
        if self.use_priority:
            # In max-heap, lowest priority is at the end (after sorting)
            # This is expensive but used rarely
            if not self._queue:
                return None
            lowest_idx = max(range(len(self._queue)), key=lambda i: self._queue[i].priority)
            lowest = self._queue.pop(lowest_idx)
            heapq.heapify(self._queue)
            return lowest.item
        else:
            # Find lowest priority
            if not self._queue:
                return None
            lowest_idx = min(range(len(self._queue)), key=lambda i: self._queue[i]["priority"])
            entry = self._queue[lowest_idx]
            del self._queue[lowest_idx]
            return entry["item"]
    
    async def _cleanup_loop(self):
        """Background loop to evict expired items periodically."""
        while True:
            try:
                await asyncio.sleep(60)  # Cleanup every minute
                
                async with self._lock:
                    await self._evict_expired_internal(time.time())
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Queue cleanup error: {e}")
    
    def __len__(self) -> int:
        """Return current queue size."""
        return len(self._queue)
    
    @property
    def is_full(self) -> bool:
        """Check if queue is at capacity."""
        return len(self._queue) >= self.max_size
    
    @property
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return len(self._queue) == 0
    
    def get_status(self) -> Dict[str, Any]:
        """Get queue status and metrics."""
        return {
            "current_size": len(self._queue),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "is_full": self.is_full,
            "is_empty": self.is_empty,
            "use_priority": self.use_priority,
            "eviction_strategy": self.eviction_strategy,
            "metrics": self.metrics.to_dict()
        }
    
    async def clear(self):
        """Clear all items from queue."""
        async with self._lock:
            if self.use_priority:
                self._queue = []
            else:
                self._queue.clear()
            self._timestamps.clear()
            logger.info("Queue cleared")
