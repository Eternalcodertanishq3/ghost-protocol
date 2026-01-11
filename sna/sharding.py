import torch
import logging
import asyncio
from typing import List, Dict, Any
from .byzantine_shield import ByzantineShield

logger = logging.getLogger("shard_aggregator")

class HierarchicalAggregator:
    """
    Implements Hierarchical Aggregation (Sharding) to scale Ghost Protocol 
    to thousands of hospitals.
    
    Architecture:
    [Hospitals] -> [Shard/Regional Aggregators] -> [National/Global Aggregator]
    
    Instead of computing valid_median on 5000 vectors (O(N^2) or O(N log N)),
    we split into K shards of size M.
    """
    
    def __init__(self, shard_size: int = 50):
        self.shard_size = shard_size
        self.shield = ByzantineShield()
        
    async def aggregate_updates(self, updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Main entry point. Automatically decides whether to use direct or hierarchical mode.
        """
        n_updates = len(updates)
        
        # If small enough, just use direct aggregation (Base Case)
        if n_updates <= self.shard_size:
            logger.info(f"Direct aggregation for {n_updates} updates")
            return await self.shield.aggregate_updates([u['weights'] for u in updates])
            
        # Otherwise, perform Map-Reduce sharding
        n_shards = (n_updates + self.shard_size - 1) // self.shard_size
        logger.info(f"Starting hierarchical aggregation: {n_updates} updates over {n_shards} shards")
        
        # 1. Partition (Map)
        shards = []
        for i in range(0, n_updates, self.shard_size):
            shard_updates = updates[i : i + self.shard_size]
            shards.append(shard_updates)
            
        # 2. Aggregate Shards in Parallel (Reduce Phase 1)
        # We simulate regional aggregation here
        shard_tasks = [self._process_shard(i, shard) for i, shard in enumerate(shards)]
        shard_results = await asyncio.gather(*shard_tasks)
        
        # Filter out failed shards (None)
        valid_shard_results = [res for res in shard_results if res is not None]
        
        if not valid_shard_results:
            logger.error("All shards failed aggregation!")
            return None
            
        logger.info(f"Phase 1 complete. Aggregating {len(valid_shard_results)} shard representatives.")
        
        # 3. Global Aggregation (Reduce Phase 2)
        # Treat the shard results as inputs for the final aggregation
        final_weights = await self.shield.aggregate_updates(valid_shard_results)
        
        return final_weights
        
    async def _process_shard(self, shard_id: int, updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Process a single shard (simulates a Regional SNA)
        """
        try:
            # Extract weights
            weight_list = [u['weights'] for u in updates]
            
            # Aggregate using robust geometric median
            # Note: We create a new Shield instance per shard if needed, 
            # but reusing self.shield is thread-safe for stateless aggregation
            result = await self.shield.aggregate_updates(weight_list)
            return result
        except Exception as e:
            logger.error(f"Shard {shard_id} failed: {e}")
            return None
