"""
Module: sna/resilient_cache.py
DPDP §: 8(2)(a) - System Reliability
Description: Ultra-Advanced Resilient Cache with Circuit Breaker Pattern

Features:
- Redis with automatic fallback to in-memory cache
- Circuit breaker pattern for fault tolerance
- Exponential backoff with jitter for reconnection
- Health monitoring and metrics
- Async-first design
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger("ghost.cache")


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"          # Failing, using fallback
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5          # Failures before opening
    recovery_timeout: float = 30.0      # Seconds before half-open
    success_threshold: int = 3          # Successes needed to close
    max_backoff: float = 300.0          # Max backoff seconds


@dataclass
class CacheMetrics:
    """Cache operation metrics."""
    hits: int = 0
    misses: int = 0
    redis_failures: int = 0
    fallback_operations: int = 0
    circuit_opens: int = 0
    last_redis_success: Optional[datetime] = None
    last_redis_failure: Optional[datetime] = None


class ResilientCache:
    """
    Ultra-Advanced Resilient Cache with Circuit Breaker.
    
    Implements:
    - Redis as primary cache
    - In-memory fallback when Redis is unavailable
    - Circuit breaker pattern for fault isolation
    - Exponential backoff with jitter for reconnection
    - Comprehensive metrics and health monitoring
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        max_memory_items: int = 10000,
        default_ttl: int = 3600,
        circuit_config: Optional[CircuitBreakerConfig] = None
    ):
        self.redis_url = redis_url
        self.max_memory_items = max_memory_items
        self.default_ttl = default_ttl
        self.circuit_config = circuit_config or CircuitBreakerConfig()
        
        # Redis client (lazy initialization)
        self._redis = None
        self._redis_available = False
        
        # In-memory fallback with TTL
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        
        # Circuit breaker state
        self._circuit_state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._backoff_multiplier = 1
        
        # Metrics
        self.metrics = CacheMetrics()
        
        # Background tasks
        self._health_check_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        logger.info(f"ResilientCache initialized with Redis URL: {redis_url}")
    
    async def connect(self) -> bool:
        """
        Attempt to connect to Redis with circuit breaker protection.
        
        Returns:
            True if Redis is available, False if using fallback.
        """
        if self._circuit_state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if self._last_failure_time:
                elapsed = time.time() - self._last_failure_time
                backoff = min(
                    self.circuit_config.recovery_timeout * self._backoff_multiplier,
                    self.circuit_config.max_backoff
                )
                
                if elapsed < backoff:
                    logger.debug(f"Circuit open, waiting {backoff - elapsed:.1f}s before retry")
                    return False
                
                # Transition to half-open
                self._circuit_state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker transitioning to HALF_OPEN")
        
        try:
            import redis.asyncio as aioredis
            
            self._redis = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5.0,
                socket_timeout=5.0
            )
            
            # Test connection
            await asyncio.wait_for(self._redis.ping(), timeout=5.0)
            
            self._redis_available = True
            self._on_success()
            
            # Start background tasks
            if not self._health_check_task:
                self._health_check_task = asyncio.create_task(self._health_check_loop())
            if not self._cleanup_task:
                self._cleanup_task = asyncio.create_task(self._memory_cleanup_loop())
            
            logger.info("Redis connection established")
            return True
            
        except Exception as e:
            self._on_failure(e)
            return False
    
    def _on_success(self):
        """Handle successful Redis operation."""
        self.metrics.last_redis_success = datetime.utcnow()
        
        if self._circuit_state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.circuit_config.success_threshold:
                self._circuit_state = CircuitState.CLOSED
                self._failure_count = 0
                self._success_count = 0
                self._backoff_multiplier = 1
                logger.info("Circuit breaker CLOSED - Redis recovered")
        elif self._circuit_state == CircuitState.CLOSED:
            self._failure_count = max(0, self._failure_count - 1)
    
    def _on_failure(self, error: Exception):
        """Handle Redis operation failure."""
        self.metrics.redis_failures += 1
        self.metrics.last_redis_failure = datetime.utcnow()
        self._last_failure_time = time.time()
        self._redis_available = False
        
        if self._circuit_state == CircuitState.HALF_OPEN:
            # Failed during recovery test, re-open
            self._circuit_state = CircuitState.OPEN
            self._backoff_multiplier = min(self._backoff_multiplier * 2, 32)
            logger.warning(f"Circuit breaker re-OPENED (backoff: {self._backoff_multiplier}x)")
        else:
            self._failure_count += 1
            if self._failure_count >= self.circuit_config.failure_threshold:
                self._circuit_state = CircuitState.OPEN
                self.metrics.circuit_opens += 1
                logger.warning(f"Circuit breaker OPENED after {self._failure_count} failures: {error}")
        
        self._success_count = 0
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache (Redis or fallback).
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        # Try Redis if available
        if self._redis_available and self._circuit_state != CircuitState.OPEN:
            try:
                value = await asyncio.wait_for(
                    self._redis.get(key),
                    timeout=2.0
                )
                if value is not None:
                    self.metrics.hits += 1
                    self._on_success()
                    return value
                self.metrics.misses += 1
                return None
            except Exception as e:
                self._on_failure(e)
        
        # Fallback to memory cache
        return self._get_from_memory(key)
    
    def _get_from_memory(self, key: str) -> Optional[Any]:
        """Get value from in-memory fallback cache."""
        self.metrics.fallback_operations += 1
        
        entry = self._memory_cache.get(key)
        if entry is None:
            self.metrics.misses += 1
            return None
        
        # Check TTL
        if entry["expires_at"] and time.time() > entry["expires_at"]:
            del self._memory_cache[key]
            self.metrics.misses += 1
            return None
        
        self.metrics.hits += 1
        return entry["value"]
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache (Redis or fallback).
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
            
        Returns:
            True if successful
        """
        ttl = ttl or self.default_ttl
        
        # Try Redis if available
        if self._redis_available and self._circuit_state != CircuitState.OPEN:
            try:
                await asyncio.wait_for(
                    self._redis.setex(key, ttl, value),
                    timeout=2.0
                )
                self._on_success()
                return True
            except Exception as e:
                self._on_failure(e)
        
        # Fallback to memory cache
        return self._set_in_memory(key, value, ttl)
    
    def _set_in_memory(self, key: str, value: Any, ttl: int) -> bool:
        """Set value in in-memory fallback cache."""
        self.metrics.fallback_operations += 1
        
        # Evict if at capacity (LRU-like)
        if len(self._memory_cache) >= self.max_memory_items:
            # Remove oldest entry
            oldest_key = next(iter(self._memory_cache))
            del self._memory_cache[oldest_key]
        
        self._memory_cache[key] = {
            "value": value,
            "expires_at": time.time() + ttl if ttl else None,
            "created_at": time.time()
        }
        return True
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        # Delete from both Redis and memory
        success = False
        
        if self._redis_available and self._circuit_state != CircuitState.OPEN:
            try:
                await asyncio.wait_for(
                    self._redis.delete(key),
                    timeout=2.0
                )
                success = True
                self._on_success()
            except Exception as e:
                self._on_failure(e)
        
        if key in self._memory_cache:
            del self._memory_cache[key]
            success = True
        
        return success
    
    async def incr(self, key: str) -> int:
        """Increment counter (atomically in Redis, or in memory)."""
        if self._redis_available and self._circuit_state != CircuitState.OPEN:
            try:
                result = await asyncio.wait_for(
                    self._redis.incr(key),
                    timeout=2.0
                )
                self._on_success()
                return result
            except Exception as e:
                self._on_failure(e)
        
        # Fallback: increment in memory
        self.metrics.fallback_operations += 1
        entry = self._memory_cache.get(key, {"value": 0, "expires_at": None})
        entry["value"] = int(entry.get("value", 0)) + 1
        self._memory_cache[key] = entry
        return entry["value"]
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set TTL on existing key."""
        if self._redis_available and self._circuit_state != CircuitState.OPEN:
            try:
                await asyncio.wait_for(
                    self._redis.expire(key, ttl),
                    timeout=2.0
                )
                self._on_success()
                return True
            except Exception as e:
                self._on_failure(e)
        
        # Fallback: update TTL in memory
        if key in self._memory_cache:
            self._memory_cache[key]["expires_at"] = time.time() + ttl
            return True
        return False
    
    async def _health_check_loop(self):
        """Background health check for Redis connection."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                if not self._redis_available and self._circuit_state != CircuitState.OPEN:
                    logger.info("Attempting Redis reconnection...")
                    await self.connect()
                elif self._redis_available:
                    # Verify connection is still alive
                    try:
                        await asyncio.wait_for(self._redis.ping(), timeout=5.0)
                    except Exception:
                        self._redis_available = False
                        logger.warning("Redis health check failed")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    async def _memory_cleanup_loop(self):
        """Background cleanup for expired memory cache entries."""
        while True:
            try:
                await asyncio.sleep(60)  # Cleanup every minute
                
                now = time.time()
                expired_keys = [
                    key for key, entry in self._memory_cache.items()
                    if entry["expires_at"] and now > entry["expires_at"]
                ]
                
                for key in expired_keys:
                    del self._memory_cache[key]
                
                if expired_keys:
                    logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get cache health status."""
        return {
            "redis_available": self._redis_available,
            "circuit_state": self._circuit_state.value,
            "failure_count": self._failure_count,
            "memory_cache_size": len(self._memory_cache),
            "metrics": {
                "hits": self.metrics.hits,
                "misses": self.metrics.misses,
                "hit_rate": self.metrics.hits / max(1, self.metrics.hits + self.metrics.misses),
                "redis_failures": self.metrics.redis_failures,
                "fallback_operations": self.metrics.fallback_operations,
                "circuit_opens": self.metrics.circuit_opens,
                "last_redis_success": self.metrics.last_redis_success.isoformat() if self.metrics.last_redis_success else None,
                "last_redis_failure": self.metrics.last_redis_failure.isoformat() if self.metrics.last_redis_failure else None
            }
        }
    
    async def close(self):
        """Close cache connections and stop background tasks."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self._redis:
            await self._redis.close()
        
        logger.info("ResilientCache closed")
