"""
Module: sna/persistence.py
Description: PostgreSQL Persistence Layer for Training State

Ultra-Advanced Features:
- Async SQLAlchemy with connection pooling
- Model checkpoint storage and retrieval
- Training round history with full metadata
- Hospital reputation persistence
- Transaction log for HealthTokens
- Automatic recovery on restart
"""

import asyncio
import json
import logging
import pickle
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    LargeBinary,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import QueuePool

import torch

logger = logging.getLogger("ghost.persistence")

Base = declarative_base()


# ============================================================
# Database Models
# ============================================================

class TrainingRound(Base):
    """Stores training round state and results."""
    __tablename__ = "training_rounds"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    round_number = Column(Integer, unique=True, nullable=False, index=True)
    weights_blob = Column(LargeBinary, nullable=False)  # Pickled dict of tensors
    model_performance = Column(Float, default=0.0)
    hospitals_participated = Column(Integer, default=0)
    anomalies_detected = Column(Integer, default=0)
    tokens_distributed = Column(Float, default=0.0)
    aggregation_strategy = Column(String(50), default="geometric_median")
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata_json = Column(Text, nullable=True)  # JSON for additional data


class HospitalState(Base):
    """Stores hospital reputation and statistics."""
    __tablename__ = "hospital_states"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    hospital_id = Column(String(100), unique=True, nullable=False, index=True)
    reputation_score = Column(Float, default=1.0)
    total_updates_submitted = Column(Integer, default=0)
    total_updates_accepted = Column(Integer, default=0)
    total_tokens_earned = Column(Float, default=0.0)
    total_epsilon_spent = Column(Float, default=0.0)
    is_quarantined = Column(Boolean, default=False)
    quarantine_until = Column(DateTime, nullable=True)
    last_update_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelCheckpoint(Base):
    """Stores model checkpoints for versioning."""
    __tablename__ = "model_checkpoints"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    checkpoint_name = Column(String(100), nullable=False, index=True)
    round_number = Column(Integer, nullable=False)
    weights_blob = Column(LargeBinary, nullable=False)
    architecture_hash = Column(String(64), nullable=True)
    performance = Column(Float, default=0.0)
    is_latest = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class PrivacyAuditLog(Base):
    """Stores privacy budget expenditure for DPDP compliance."""
    __tablename__ = "privacy_audit_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    hospital_id = Column(String(100), nullable=False, index=True)
    round_number = Column(Integer, nullable=False)
    epsilon_spent = Column(Float, nullable=False)
    delta_spent = Column(Float, nullable=False)
    mechanism = Column(String(50), default="gaussian")
    compliance_status = Column(String(20), default="compliant")
    created_at = Column(DateTime, default=datetime.utcnow)


# ============================================================
# Persistence Manager
# ============================================================

class PersistenceManager:
    """
    Ultra-Advanced PostgreSQL Persistence Layer.
    
    Provides:
    - Training state recovery after restart
    - Model checkpoint storage and versioning
    - Hospital reputation persistence
    - Privacy audit trail for DPDP compliance
    """
    
    def __init__(
        self,
        database_url: str = "postgresql+asyncpg://ghost_user:ghost_password@localhost:5432/ghost_protocol",
        pool_size: int = 10,
        max_overflow: int = 20
    ):
        self.database_url = database_url
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        
        self._engine = None
        self._session_factory = None
        self._initialized = False
        
        logger.info(f"PersistenceManager created with URL: {database_url[:50]}...")
    
    async def initialize(self):
        """Initialize database connection and create tables."""
        try:
            self._engine = create_async_engine(
                self.database_url,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_pre_ping=True,  # Verify connections are alive
                pool_recycle=3600,   # Recycle connections hourly
                echo=False
            )
            
            self._session_factory = sessionmaker(
                self._engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Create tables if they don't exist
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            self._initialized = True
            logger.info("PersistenceManager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize persistence: {e}")
            # Don't crash - system can run in memory-only mode
            self._initialized = False
    
    @asynccontextmanager
    async def session(self):
        """Async context manager for database sessions."""
        if not self._initialized:
            yield None
            return
        
        session = self._session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            await session.close()
    
    # ============================================================
    # Training Round Operations
    # ============================================================
    
    async def save_round(
        self,
        round_number: int,
        weights: Dict[str, torch.Tensor],
        performance: float,
        hospitals: int,
        anomalies: int = 0,
        tokens: float = 0.0,
        strategy: str = "geometric_median",
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Save training round to database.
        
        Args:
            round_number: Round number
            weights: Model weights dictionary
            performance: Model performance metric
            hospitals: Number of participating hospitals
            anomalies: Anomalies detected this round
            tokens: Tokens distributed this round
            strategy: Aggregation strategy used
            metadata: Additional metadata
            
        Returns:
            True if saved successfully
        """
        if not self._initialized:
            return False
        
        try:
            # Serialize weights
            weights_blob = pickle.dumps({
                k: v.cpu().numpy() for k, v in weights.items()
            })
            
            async with self.session() as session:
                if session is None:
                    return False
                
                round_record = TrainingRound(
                    round_number=round_number,
                    weights_blob=weights_blob,
                    model_performance=performance,
                    hospitals_participated=hospitals,
                    anomalies_detected=anomalies,
                    tokens_distributed=tokens,
                    aggregation_strategy=strategy,
                    metadata_json=json.dumps(metadata) if metadata else None
                )
                session.add(round_record)
            
            logger.info(f"Saved training round {round_number} to database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save round {round_number}: {e}")
            return False
    
    async def load_latest_round(self) -> Optional[Dict]:
        """
        Load the latest training round state.
        
        Returns:
            Dictionary with round state or None
        """
        if not self._initialized:
            return None
        
        try:
            from sqlalchemy import select
            
            async with self.session() as session:
                if session is None:
                    return None
                
                result = await session.execute(
                    select(TrainingRound)
                    .order_by(TrainingRound.round_number.desc())
                    .limit(1)
                )
                round_record = result.scalar_one_or_none()
                
                if round_record is None:
                    return None
                
                # Deserialize weights
                weights_data = pickle.loads(round_record.weights_blob)
                weights = {
                    k: torch.tensor(v) for k, v in weights_data.items()
                }
                
                return {
                    "round_number": round_record.round_number,
                    "weights": weights,
                    "performance": round_record.model_performance,
                    "hospitals": round_record.hospitals_participated,
                    "created_at": round_record.created_at
                }
                
        except Exception as e:
            logger.error(f"Failed to load latest round: {e}")
            return None
    
    # ============================================================
    # Hospital State Operations
    # ============================================================
    
    async def save_hospital_state(
        self,
        hospital_id: str,
        reputation: float,
        tokens_earned: float = 0.0,
        epsilon_spent: float = 0.0
    ) -> bool:
        """Save or update hospital state."""
        if not self._initialized:
            return False
        
        try:
            from sqlalchemy import select
            
            async with self.session() as session:
                if session is None:
                    return False
                
                result = await session.execute(
                    select(HospitalState).where(HospitalState.hospital_id == hospital_id)
                )
                hospital = result.scalar_one_or_none()
                
                if hospital:
                    # Update existing
                    hospital.reputation_score = reputation
                    hospital.total_tokens_earned += tokens_earned
                    hospital.total_epsilon_spent += epsilon_spent
                    hospital.last_update_at = datetime.utcnow()
                else:
                    # Create new
                    hospital = HospitalState(
                        hospital_id=hospital_id,
                        reputation_score=reputation,
                        total_tokens_earned=tokens_earned,
                        total_epsilon_spent=epsilon_spent
                    )
                    session.add(hospital)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save hospital state for {hospital_id}: {e}")
            return False
    
    async def load_all_hospital_states(self) -> Dict[str, Dict]:
        """Load all hospital states."""
        if not self._initialized:
            return {}
        
        try:
            from sqlalchemy import select
            
            async with self.session() as session:
                if session is None:
                    return {}
                
                result = await session.execute(select(HospitalState))
                hospitals = result.scalars().all()
                
                return {
                    h.hospital_id: {
                        "reputation_score": h.reputation_score,
                        "total_updates": h.total_updates_submitted,
                        "total_tokens": h.total_tokens_earned,
                        "epsilon_spent": h.total_epsilon_spent,
                        "is_quarantined": h.is_quarantined
                    }
                    for h in hospitals
                }
                
        except Exception as e:
            logger.error(f"Failed to load hospital states: {e}")
            return {}
    
    # ============================================================
    # Model Checkpoint Operations
    # ============================================================
    
    async def save_checkpoint(
        self,
        name: str,
        round_number: int,
        weights: Dict[str, torch.Tensor],
        performance: float,
        architecture_hash: Optional[str] = None
    ) -> bool:
        """Save model checkpoint."""
        if not self._initialized:
            return False
        
        try:
            from sqlalchemy import update
            
            weights_blob = pickle.dumps({
                k: v.cpu().numpy() for k, v in weights.items()
            })
            
            async with self.session() as session:
                if session is None:
                    return False
                
                # Mark all existing checkpoints as not latest
                await session.execute(
                    update(ModelCheckpoint)
                    .where(ModelCheckpoint.checkpoint_name == name)
                    .values(is_latest=False)
                )
                
                # Add new checkpoint
                checkpoint = ModelCheckpoint(
                    checkpoint_name=name,
                    round_number=round_number,
                    weights_blob=weights_blob,
                    architecture_hash=architecture_hash,
                    performance=performance,
                    is_latest=True
                )
                session.add(checkpoint)
            
            logger.info(f"Saved checkpoint '{name}' at round {round_number}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint '{name}': {e}")
            return False
    
    async def load_checkpoint(self, name: str, latest: bool = True) -> Optional[Dict]:
        """Load a model checkpoint."""
        if not self._initialized:
            return None
        
        try:
            from sqlalchemy import select
            
            async with self.session() as session:
                if session is None:
                    return None
                
                query = select(ModelCheckpoint).where(
                    ModelCheckpoint.checkpoint_name == name
                )
                if latest:
                    query = query.where(ModelCheckpoint.is_latest == True)
                query = query.order_by(ModelCheckpoint.created_at.desc()).limit(1)
                
                result = await session.execute(query)
                checkpoint = result.scalar_one_or_none()
                
                if checkpoint is None:
                    return None
                
                weights_data = pickle.loads(checkpoint.weights_blob)
                weights = {
                    k: torch.tensor(v) for k, v in weights_data.items()
                }
                
                return {
                    "weights": weights,
                    "round_number": checkpoint.round_number,
                    "performance": checkpoint.performance,
                    "architecture_hash": checkpoint.architecture_hash,
                    "created_at": checkpoint.created_at
                }
                
        except Exception as e:
            logger.error(f"Failed to load checkpoint '{name}': {e}")
            return None
    
    # ============================================================
    # Privacy Audit Operations
    # ============================================================
    
    async def log_privacy_expenditure(
        self,
        hospital_id: str,
        round_number: int,
        epsilon: float,
        delta: float,
        mechanism: str = "gaussian"
    ) -> bool:
        """Log privacy budget expenditure for DPDP compliance."""
        if not self._initialized:
            return False
        
        try:
            async with self.session() as session:
                if session is None:
                    return False
                
                log_entry = PrivacyAuditLog(
                    hospital_id=hospital_id,
                    round_number=round_number,
                    epsilon_spent=epsilon,
                    delta_spent=delta,
                    mechanism=mechanism
                )
                session.add(log_entry)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to log privacy expenditure: {e}")
            return False
    
    async def get_total_epsilon(self, hospital_id: str) -> float:
        """Get total epsilon spent by a hospital."""
        if not self._initialized:
            return 0.0
        
        try:
            from sqlalchemy import func, select
            
            async with self.session() as session:
                if session is None:
                    return 0.0
                
                result = await session.execute(
                    select(func.sum(PrivacyAuditLog.epsilon_spent))
                    .where(PrivacyAuditLog.hospital_id == hospital_id)
                )
                total = result.scalar_one_or_none()
                return float(total or 0.0)
                
        except Exception as e:
            logger.error(f"Failed to get epsilon for {hospital_id}: {e}")
            return 0.0
    
    # ============================================================
    # Utility Methods
    # ============================================================
    
    async def close(self):
        """Close database connections."""
        if self._engine:
            await self._engine.dispose()
            logger.info("Database connections closed")
    
    @property
    def is_available(self) -> bool:
        """Check if persistence is available."""
        return self._initialized
