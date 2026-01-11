"""
HealthToken Ledger - Blockchain-based Incentive System
Shapley-based Minting · Reputation-weighted Distribution · Immutable Audit Trail

DPDP §: §9(4) Purpose Limitation - Incentive system for healthcare AI participation
Byzantine theorem: Economic incentives align honest behavior
Test command: pytest tests/test_ledger.py -v --cov=ledger
Metrics tracked: Token distribution, Transaction volume, Reputation-weighted rewards, Burn events
"""

import logging
import json
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import uuid
import asyncio
from decimal import Decimal

import numpy as np
from web3 import Web3
from eth_account import Account
import requests


class TransactionType(Enum):
    """HealthToken transaction types"""
    MINT = "mint"           # New tokens created
    TRANSFER = "transfer"   # Tokens transferred
    BURN = "burn"          # Tokens destroyed
    STAKE = "stake"        # Tokens staked
    REWARD = "reward"      # Participation reward


@dataclass
class HealthTokenTransaction:
    """HealthToken transaction record - Mock Chain Mode enabled for MVP/Demo"""
    transaction_id: str
    transaction_type: TransactionType
    from_address: str
    to_address: str
    amount: Decimal
    round_id: Optional[str]
    hospital_id: str
    reputation_score: float
    shapley_contribution: float
    timestamp: datetime
    # Mock Chain Mode: Auto-generate synthetic blockchain metadata
    block_number: Optional[int] = None
    transaction_hash: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Auto-generate mock blockchain metadata if not provided"""
        if self.block_number is None:
            # Synthetic block number based on timestamp
            self.block_number = int(self.timestamp.timestamp()) % 1000000
        if self.transaction_hash is None:
            # Synthetic transaction hash
            import hashlib
            hash_input = f"{self.transaction_id}{self.timestamp.isoformat()}"
            self.transaction_hash = f"0x{hashlib.sha256(hash_input.encode()).hexdigest()[:64]}"
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["transaction_type"] = self.transaction_type.value
        data["amount"] = str(self.amount)
        data["timestamp"] = self.timestamp.isoformat()
        data["mock_chain"] = True  # Flag indicating mock chain mode
        return data



@dataclass
class HospitalStake:
    """Hospital staking information"""
    hospital_id: str
    staked_amount: Decimal
    stake_timestamp: datetime
    unlock_timestamp: Optional[datetime]
    reward_multiplier: float
    
    @property
    def is_locked(self) -> bool:
        return self.unlock_timestamp is None or datetime.utcnow() < self.unlock_timestamp


class HealthTokenLedger:
    """
    HealthToken Ledger - Blockchain-based incentive system
    
    Implements:
    - Shapley value-based reward calculation
    - Reputation-weighted token distribution
    - Staking mechanism for governance
    - Immutable audit trail
    - Smart contract integration (Polygon)
    
    Economic incentives align hospital behavior with network goals
    """
    
    def __init__(
        self,
        polygon_rpc_url: str = None,
        contract_address: str = None,
        private_key: str = None,
        stake_lock_period_days: int = 30,
        base_reward_tokens: int = 10
    ):
        self.polygon_rpc_url = polygon_rpc_url or "https://polygon-rpc.com"
        self.contract_address = contract_address
        self.private_key = private_key
        self.stake_lock_period_days = stake_lock_period_days
        self.base_reward_tokens = base_reward_tokens
        
        self.logger = logging.getLogger("healthtoken_ledger")
        
        # Web3 connection
        self.web3 = Web3(Web3.HTTPProvider(self.polygon_rpc_url))
        self.account = None
        if private_key:
            self.account = Account.from_key(private_key)
        
        # Ledger state
        self.balances: Dict[str, Decimal] = {}  # Hospital ID -> Balance
        self.stakes: Dict[str, HospitalStake] = {}  # Hospital ID -> Stake info
        self.transactions: List[HealthTokenTransaction] = []
        
        # Transaction pool (for batching)
        self.pending_transactions: List[HealthTokenTransaction] = []
        
        # Shapley value cache
        self.shapley_cache: Dict[str, float] = {}
        
        # Metrics
        self.metrics = {
            "total_minted": Decimal("0"),
            "total_burned": Decimal("0"),
            "total_transferred": Decimal("0"),
            "total_staked": Decimal("0"),
            "transactions_processed": 0,
            "hospitals_rewarded": 0,
            "average_reward": Decimal("0"),
            "blockchain_gas_cost": 0
        }
        
        # Background tasks
        self.batch_processor_task: Optional[asyncio.Task] = None
        
        self.logger.info("HealthToken Ledger initialized")
    
    async def start(self):
        """Start ledger background processes"""
        
        if self.account:
            self.batch_processor_task = asyncio.create_task(self._batch_processor())
            self.logger.info("HealthToken Ledger started with blockchain integration")
        else:
            self.logger.warning("HealthToken Ledger running in simulation mode (no private key)")
    
    async def award_tokens(
        self,
        hospital_id: str,
        round_id: str,
        local_auc: float,
        reputation_score: float,
        gradient_quality: float,
        participation_history: List[float]
    ) -> Decimal:
        """
        Award HealthTokens based on Shapley value and reputation
        
        Args:
            hospital_id: Hospital identifier
            round_id: Federated learning round
            local_auc: Local model performance
            reputation_score: Hospital reputation (0-1)
            gradient_quality: Quality of gradient update
            participation_history: Historical participation data
            
        Returns:
            Amount of tokens awarded
        """
        
        try:
            # Calculate Shapley value contribution
            shapley_value = await self._calculate_shapley_value(
                local_auc=local_auc,
                reputation_score=reputation_score,
                gradient_quality=gradient_quality,
                participation_history=participation_history
            )
            
            # Base reward
            base_reward = Decimal(str(self.base_reward_tokens))
            
            # Performance bonus
            performance_bonus = Decimal(str(local_auc * 20))  # Up to 20 bonus tokens
            
            # Reputation multiplier
            reputation_multiplier = Decimal(str(max(0.1, reputation_score)))
            
            # Shapley value bonus
            shapley_bonus = Decimal(str(shapley_value * 10))  # Up to 10 bonus tokens
            
            # Staking bonus
            stake_bonus = await self._calculate_staking_bonus(hospital_id)
            
            # Total reward
            total_reward = (base_reward + performance_bonus + shapley_bonus + stake_bonus) * reputation_multiplier
            
            # Ensure minimum reward
            total_reward = max(total_reward, Decimal("1"))  # Minimum 1 token
            
            # Mint tokens
            await self._mint_tokens(hospital_id, total_reward, round_id, shapley_value)
            
            self.metrics["hospitals_rewarded"] += 1
            self.metrics["total_minted"] += total_reward
            
            # Update average reward
            total_transactions = self.metrics["transactions_processed"]
            current_avg = self.metrics["average_reward"]
            self.metrics["average_reward"] = (current_avg * total_transactions + total_reward) / (total_transactions + 1)
            
            self.logger.info(f"Awarded {total_reward} HealthTokens to {hospital_id} for round {round_id}")
            
            return total_reward
            
        except Exception as e:
            self.logger.error(f"Failed to award tokens to {hospital_id}: {e}")
            return Decimal("0")
    
    async def _calculate_shapley_value(
        self,
        local_auc: float,
        reputation_score: float,
        gradient_quality: float,
        participation_history: List[float]
    ) -> float:
        """
        Calculate Shapley value for hospital contribution
        
        Simplified Shapley value calculation based on:
        - Model performance contribution
        - Data quality contribution
        - Participation consistency
        """
        
        cache_key = f"{local_auc}_{reputation_score}_{gradient_quality}_{len(participation_history)}"
        
        if cache_key in self.shapley_cache:
            return self.shapley_cache[cache_key]
        
        # Performance contribution
        performance_contribution = local_auc ** 2  # Quadratic to reward high performance
        
        # Data quality contribution (based on reputation and gradient quality)
        quality_contribution = reputation_score * gradient_quality
        
        # Participation contribution (consistency bonus)
        if participation_history:
            participation_rate = np.mean(participation_history)
            consistency_bonus = 1 - np.std(participation_history) if len(participation_history) > 1 else 1.0
            participation_contribution = participation_rate * consistency_bonus
        else:
            participation_contribution = 0.5  # Default for new hospitals
        
        # Weighted combination
        shapley_value = (
            0.4 * performance_contribution +
            0.4 * quality_contribution +
            0.2 * participation_contribution
        )
        
        # Cache result
        self.shapley_cache[cache_key] = shapley_value
        
        # Limit cache size
        if len(self.shapley_cache) > 10000:
            # Clear oldest entries
            keys = list(self.shapley_cache.keys())
            for key in keys[:5000]:
                del self.shapley_cache[key]
        
        return shapley_value
    
    async def _calculate_staking_bonus(self, hospital_id: str) -> Decimal:
        """Calculate bonus for staked tokens"""
        
        if hospital_id not in self.stakes:
            return Decimal("0")
        
        stake = self.stakes[hospital_id]
        
        if stake.is_locked:
            # Bonus for locked stakes
            bonus_multiplier = min(float(stake.staked_amount) / 1000, 2.0)  # Max 2x bonus
            base_bonus = Decimal("5") * Decimal(str(bonus_multiplier))
            return base_bonus
        
        return Decimal("0")
    
    async def _mint_tokens(
        self,
        hospital_id: str,
        amount: Decimal,
        round_id: str,
        shapley_contribution: float
    ):
        """Mint new HealthTokens for hospital"""
        
        # Update balance
        if hospital_id not in self.balances:
            self.balances[hospital_id] = Decimal("0")
        
        self.balances[hospital_id] += amount
        
        # Create transaction
        transaction = HealthTokenTransaction(
            transaction_id=f"mint_{uuid.uuid4().hex}",
            transaction_type=TransactionType.MINT,
            from_address="GENESIS",
            to_address=hospital_id,
            amount=amount,
            round_id=round_id,
            hospital_id=hospital_id,
            reputation_score=self._get_hospital_reputation(hospital_id),
            shapley_contribution=shapley_contribution,
            timestamp=datetime.utcnow(),
            metadata={
                "minting_reason": "federated_learning_reward",
                "local_auc": None,  # Would be filled from ghost pack
                "gradient_norm": None
            }
        )
        
        # Add to transaction history
        self.transactions.append(transaction)
        self.pending_transactions.append(transaction)
        
        self.metrics["transactions_processed"] += 1
        
        self.logger.info(f"Minted {amount} HealthTokens for {hospital_id}")
    
    async def transfer_tokens(
        self,
        from_hospital: str,
        to_hospital: str,
        amount: Decimal,
        purpose: str = "payment"
    ) -> bool:
        """
        Transfer HealthTokens between hospitals
        
        Args:
            from_hospital: Sender hospital ID
            to_hospital: Receiver hospital ID
            amount: Amount to transfer
            purpose: Transfer purpose
            
        Returns:
            True if transfer successful
        """
        
        try:
            # Check balance
            if self.balances.get(from_hospital, Decimal("0")) < amount:
                self.logger.warning(f"Insufficient balance for transfer: {from_hospital}")
                return False
            
            # Update balances
            self.balances[from_hospital] -= amount
            if to_hospital not in self.balances:
                self.balances[to_hospital] = Decimal("0")
            self.balances[to_hospital] += amount
            
            # Create transaction
            transaction = HealthTokenTransaction(
                transaction_id=f"transfer_{uuid.uuid4().hex}",
                transaction_type=TransactionType.TRANSFER,
                from_address=from_hospital,
                to_address=to_hospital,
                amount=amount,
                round_id=None,
                hospital_id=from_hospital,
                reputation_score=self._get_hospital_reputation(from_hospital),
                shapley_contribution=0.0,
                timestamp=datetime.utcnow(),
                metadata={
                    "transfer_purpose": purpose
                }
            )
            
            self.transactions.append(transaction)
            self.pending_transactions.append(transaction)
            
            self.metrics["total_transferred"] += amount
            self.metrics["transactions_processed"] += 1
            
            self.logger.info(f"Transferred {amount} HealthTokens from {from_hospital} to {to_hospital}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Transfer failed: {e}")
            return False
    
    async def stake_tokens(self, hospital_id: str, amount: Decimal) -> bool:
        """
        Stake HealthTokens for governance and bonus rewards
        
        Args:
            hospital_id: Hospital identifier
            amount: Amount to stake
            
        Returns:
            True if staking successful
        """
        
        try:
            # Check balance
            if self.balances.get(hospital_id, Decimal("0")) < amount:
                self.logger.warning(f"Insufficient balance for staking: {hospital_id}")
                return False
            
            # Update balance
            self.balances[hospital_id] -= amount
            
            # Create or update stake
            if hospital_id in self.stakes:
                # Add to existing stake
                existing_stake = self.stakes[hospital_id]
                existing_stake.staked_amount += amount
                existing_stake.stake_timestamp = datetime.utcnow()
                existing_stake.unlock_timestamp = datetime.utcnow() + timedelta(days=self.stake_lock_period_days)
            else:
                # Create new stake
                self.stakes[hospital_id] = HospitalStake(
                    hospital_id=hospital_id,
                    staked_amount=amount,
                    stake_timestamp=datetime.utcnow(),
                    unlock_timestamp=datetime.utcnow() + timedelta(days=self.stake_lock_period_days),
                    reward_multiplier=1.0
                )
            
            # Create transaction
            transaction = HealthTokenTransaction(
                transaction_id=f"stake_{uuid.uuid4().hex}",
                transaction_type=TransactionType.STAKE,
                from_address=hospital_id,
                to_address="STAKING_CONTRACT",
                amount=amount,
                round_id=None,
                hospital_id=hospital_id,
                reputation_score=self._get_hospital_reputation(hospital_id),
                shapley_contribution=0.0,
                timestamp=datetime.utcnow(),
                metadata={
                    "lock_period_days": self.stake_lock_period_days
                }
            )
            
            self.transactions.append(transaction)
            self.pending_transactions.append(transaction)
            
            self.metrics["total_staked"] += amount
            self.metrics["transactions_processed"] += 1
            
            self.logger.info(f"Staked {amount} HealthTokens for {hospital_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Staking failed: {e}")
            return False
    
    async def unstake_tokens(self, hospital_id: str, amount: Decimal) -> bool:
        """
        Unstake HealthTokens after lock period
        
        Args:
            hospital_id: Hospital identifier
            amount: Amount to unstake
            
        Returns:
            True if unstaking successful
        """
        
        try:
            if hospital_id not in self.stakes:
                self.logger.warning(f"No stake found for {hospital_id}")
                return False
            
            stake = self.stakes[hospital_id]
            
            # Check if stake is unlocked
            if stake.is_locked:
                self.logger.warning(f"Stake is still locked for {hospital_id}")
                return False
            
            # Check if sufficient staked amount
            if stake.staked_amount < amount:
                self.logger.warning(f"Insufficient staked amount for {hospital_id}")
                return False
            
            # Update stake
            stake.staked_amount -= amount
            
            # Return to balance
            self.balances[hospital_id] = self.balances.get(hospital_id, Decimal("0")) + amount
            
            # Remove stake if zero
            if stake.staked_amount == 0:
                del self.stakes[hospital_id]
            
            self.logger.info(f"Unstaked {amount} HealthTokens for {hospital_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Unstaking failed: {e}")
            return False
    
    def get_balance(self, hospital_id: str) -> Decimal:
        """Get HealthToken balance for hospital"""
        return self.balances.get(hospital_id, Decimal("0"))
    
    def get_stake(self, hospital_id: str) -> Optional[HospitalStake]:
        """Get staking information for hospital"""
        return self.stakes.get(hospital_id)
    
    def get_transaction_history(
        self,
        hospital_id: str = None,
        transaction_type: TransactionType = None,
        limit: int = 100
    ) -> List[HealthTokenTransaction]:
        """Get transaction history"""
        
        filtered_transactions = self.transactions
        
        # Filter by hospital
        if hospital_id:
            filtered_transactions = [
                tx for tx in filtered_transactions
                if tx.hospital_id == hospital_id or tx.from_address == hospital_id or tx.to_address == hospital_id
            ]
        
        # Filter by type
        if transaction_type:
            filtered_transactions = [
                tx for tx in filtered_transactions
                if tx.transaction_type == transaction_type
            ]
        
        # Sort by timestamp and limit
        filtered_transactions.sort(key=lambda x: x.timestamp, reverse=True)
        
        return filtered_transactions[:limit]
    
    def get_total_supply(self) -> Decimal:
        """Get total HealthToken supply"""
        return self.metrics["total_minted"] - self.metrics["total_burned"]
    
    def get_circulating_supply(self) -> Decimal:
        """Get circulating HealthToken supply (excluding staked)"""
        total_supply = self.get_total_supply()
        staked_amount = sum(stake.staked_amount for stake in self.stakes.values())
        return total_supply - staked_amount
    
    def get_hospital_rankings(self, limit: int = 10) -> List[Tuple[str, Decimal]]:
        """Get top hospitals by HealthToken balance"""
        
        sorted_hospitals = sorted(
            self.balances.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_hospitals[:limit]
    
    def _get_hospital_reputation(self, hospital_id: str) -> float:
        """Get hospital reputation score (simplified)"""
        # In production, integrate with Byzantine Shield reputation system
        return 1.0
    
    async def _batch_processor(self):
        """Process pending transactions in batches"""
        
        while True:
            try:
                if self.pending_transactions and self.account:
                    # Process transactions on blockchain
                    await self._process_blockchain_batch()
                
                await asyncio.sleep(60)  # Process every minute
                
            except Exception as e:
                self.logger.error(f"Batch processing failed: {e}")
                await asyncio.sleep(300)  # Wait longer on error
    
    async def _process_blockchain_batch(self):
        """Process transaction batch on blockchain"""
        
        if not self.pending_transactions:
            return
        
        # Group transactions by type
        mint_transactions = [tx for tx in self.pending_transactions if tx.transaction_type == TransactionType.MINT]
        transfer_transactions = [tx for tx in self.pending_transactions if tx.transaction_type == TransactionType.TRANSFER]
        
        try:
            # Process mint transactions
            if mint_transactions:
                await self._execute_mint_batch(mint_transactions)
            
            # Process transfer transactions
            if transfer_transactions:
                await self._execute_transfer_batch(transfer_transactions)
            
            # Clear processed transactions
            self.pending_transactions = [
                tx for tx in self.pending_transactions
                if tx not in mint_transactions and tx not in transfer_transactions
            ]
            
        except Exception as e:
            self.logger.error(f"Blockchain batch execution failed: {e}")
            raise
    
    async def _execute_mint_batch(self, transactions: List[HealthTokenTransaction]):
        """Execute mint transactions on blockchain"""
        
        # In production, interact with smart contract
        # For now, simulate blockchain transaction
        
        for tx in transactions:
            tx.transaction_hash = f"0x{uuid.uuid4().hex}"
            tx.block_number = 1
            
            self.logger.info(f"Minted {tx.amount} HT for {tx.hospital_id} (tx: {tx.transaction_hash})")
        
        # Simulate gas cost
        self.metrics["blockchain_gas_cost"] += len(transactions) * 100000
    
    async def _execute_transfer_batch(self, transactions: List[HealthTokenTransaction]):
        """Execute transfer transactions on blockchain"""
        
        # In production, interact with smart contract
        # For now, simulate blockchain transaction
        
        for tx in transactions:
            tx.transaction_hash = f"0x{uuid.uuid4().hex}"
            tx.block_number = 1
            
            self.logger.info(f"Transferred {tx.amount} HT from {tx.from_address} to {tx.to_address} (tx: {tx.transaction_hash})")
        
        # Simulate gas cost
        self.metrics["blockchain_gas_cost"] += len(transactions) * 50000
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive ledger metrics"""
        
        return {
            "supply": {
                "total_supply": str(self.get_total_supply()),
                "circulating_supply": str(self.get_circulating_supply()),
                "total_minted": str(self.metrics["total_minted"]),
                "total_burned": str(self.metrics["total_burned"])
            },
            "transactions": {
                "total_processed": self.metrics["transactions_processed"],
                "pending_transactions": len(self.pending_transactions),
                "average_reward": str(self.metrics["average_reward"])
            },
            "staking": {
                "total_staked": str(self.metrics["total_staked"]),
                "active_stakes": len(self.stakes)
            },
            "hospitals": {
                "hospitals_rewarded": self.metrics["hospitals_rewarded"],
                "unique_hospitals": len(self.balances)
            },
            "blockchain": {
                "gas_cost": self.metrics["blockchain_gas_cost"],
                "account_connected": self.account is not None
            }
        }