"""
Test suite for Dispute Resolution System

Tests Byzantine-fault-tolerant dispute resolution with various attack scenarios.

DPDP § Citation: §15(3) - Right to rectification includes dispute resolution
Byzantine Theorem: PBFT consensus with n ≥ 3f + 1 (Castro & Liskov, 1999)

Test Command: pytest tests/test_dispute_resolution.py -v --cov=sna/dispute_resolution

Metrics:
- Test Coverage: > 90%
- Consensus Accuracy: > 95%
- Resolution Time: < 30 seconds
"""

import pytest
import asyncio
import hashlib
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import redis.asyncio as redis

from sna.dispute_resolution import (
    DisputeResolution,
    DisputeCase,
    DisputeVerdict,
    DisputeType,
    DisputeStatus,
    EvidenceType,
    Evidence
)


@pytest.fixture
async def redis_client():
    """Create a mock Redis client."""
    client = Mock(spec=redis.Redis)
    client.hset = AsyncMock(return_value=1)
    return client


@pytest.fixture
async def dispute_system(redis_client):
    """Create a dispute resolution system instance."""
    system = DisputeResolution(
        redis_client=redis_client,
        total_hospitals=100,
        byzantine_threshold=0.33,
        voting_timeout=5,  # Short timeout for testing
        evidence_timeout=10
    )
    return system


@pytest.fixture
def sample_evidence():
    """Create sample evidence for testing."""
    evidence_data = {
        'gradient_hash': 'abc123def456',
        'model_update_id': 'update_123',
        'timestamp': datetime.utcnow().isoformat()
    }
    
    evidence = Evidence(
        evidence_id='evidence_1',
        evidence_type=EvidenceType.GRADIENT_HASH,
        submitter_id='hospital_1',
        timestamp=datetime.utcnow(),
        data=evidence_data,
        hash_signature=hashlib.sha256(
            json.dumps(evidence_data, sort_keys=True).encode()
        ).hexdigest()
    )
    return evidence


class TestDisputeCaseCreation:
    """Test dispute case creation functionality."""
    
    @pytest.mark.asyncio
    async def test_create_dispute_case(self, dispute_system):
        """Test creating a new dispute case."""
        case_id = await dispute_system.create_dispute_case(
            dispute_type=DisputeType.GRADIENT_MANIPULATION,
            plaintiff_id='hospital_1',
            defendant_id='hospital_2',
            description='Suspected gradient manipulation detected'
        )
        
        assert case_id is not None
        assert len(case_id) == 16
        
        # Verify case was created
        case = await dispute_system.get_case(case_id)
        assert case is not None
        assert case.dispute_type == DisputeType.GRADIENT_MANIPULATION
        assert case.plaintiff_id == 'hospital_1'
        assert case.defendant_id == 'hospital_2'
        assert case.status == DisputeStatus.PENDING
        
    @pytest.mark.asyncio
    async def test_multiple_dispute_cases(self, dispute_system):
        """Test creating multiple dispute cases."""
        case_ids = []
        
        for i in range(5):
            case_id = await dispute_system.create_dispute_case(
                dispute_type=DisputeType.REPUTATION_ATTACK,
                plaintiff_id=f'hospital_{i}',
                defendant_id=f'hospital_{i+1}',
                description=f'Reputation attack case {i}'
            )
            case_ids.append(case_id)
            
        assert len(case_ids) == 5
        assert len(set(case_ids)) == 5  # All IDs should be unique
        
        # Verify all cases are tracked
        active_cases = await dispute_system.get_active_cases()
        assert len(active_cases) == 5


class TestEvidenceSubmission:
    """Test evidence submission and verification."""
    
    @pytest.mark.asyncio
    async def test_submit_valid_evidence(self, dispute_system, sample_evidence):
        """Test submitting valid evidence to a case."""
        # Create a case first
        case_id = await dispute_system.create_dispute_case(
            dispute_type=DisputeType.GRADIENT_MANIPULATION,
            plaintiff_id='hospital_1',
            defendant_id='hospital_2',
            description='Test case'
        )
        
        # Submit evidence
        result = await dispute_system.submit_evidence(case_id, sample_evidence)
        assert result is True
        
        # Verify evidence was added
        case = await dispute_system.get_case(case_id)
        assert len(case.evidence) == 1
        assert case.evidence[0].evidence_id == 'evidence_1'
        assert case.evidence[0].verification_status is True
        
    @pytest.mark.asyncio
    async def test_submit_evidence_triggers_voting(self, dispute_system):
        """Test that voting starts when enough evidence is submitted."""
        case_id = await dispute_system.create_dispute_case(
            dispute_type=DisputeType.PRIVACY_VIOLATION,
            plaintiff_id='hospital_1',
            defendant_id='hospital_2',
            description='Privacy violation case'
        )
        
        # Submit 3 pieces of evidence (triggers voting)
        for i in range(3):
            evidence = Evidence(
                evidence_id=f'evidence_{i}',
                evidence_type=EvidenceType.PRIVACY_BUDGET,
                submitter_id=f'hospital_{i}',
                timestamp=datetime.utcnow(),
                data={'epsilon_used': 1.0 + i},
                hash_signature=f'hash_{i}'
            )
            
            # Mock verification to return True
            with patch.object(dispute_system, '_verify_evidence', return_value=True):
                await dispute_system.submit_evidence(case_id, evidence)
                
        # Verify voting phase started
        case = await dispute_system.get_case(case_id)
        assert case.status == DisputeStatus.VOTING


class TestVotingMechanism:
    """Test the Byzantine-fault-tolerant voting mechanism."""
    
    @pytest.mark.asyncio
    async def test_cast_vote_success(self, dispute_system):
        """Test casting a vote successfully."""
        # Create case and start voting
        case_id = await dispute_system.create_dispute_case(
            dispute_type=DisputeType.REWARD_DISCREPANCY,
            plaintiff_id='hospital_1',
            defendant_id='hospital_2',
            description='Reward discrepancy'
        )
        
        # Manually set to voting status
        case = await dispute_system.get_case(case_id)
        case.status = DisputeStatus.VOTING
        
        # Cast vote
        result = await dispute_system.cast_vote(
            case_id=case_id,
            hospital_id='hospital_3',
            vote_for_plaintiff=True,
            reputation_score=0.9
        )
        
        assert result is True
        
        # Verify vote was recorded
        case = await dispute_system.get_case(case_id)
        assert 'hospital_3' in case.votes
        assert case.votes['hospital_3'] is True
        
    @pytest.mark.asyncio
    async def test_voting_consensus_plaintiff_wins(self, dispute_system):
        """Test reaching consensus in favor of plaintiff."""
        case_id = await dispute_system.create_dispute_case(
            dispute_type=DisputeType.BYZANTINE_BEHAVIOR,
            plaintiff_id='hospital_1',
            defendant_id='hospital_2',
            description='Byzantine behavior detected'
        )
        
        case = await dispute_system.get_case(case_id)
        case.status = DisputeStatus.VOTING
        
        # Cast votes: 70% for plaintiff (supermajority)
        total_voters = 70
        for i in range(total_voters):
            vote_for_plaintiff = i < 49  # 70% for plaintiff
            await dispute_system.cast_vote(
                case_id=case_id,
                hospital_id=f'hospital_{i}',
                vote_for_plaintiff=vote_for_plaintiff,
                reputation_score=0.8
            )
            
        # Wait for resolution
        await asyncio.sleep(1)
        
        # Verify verdict
        case = await dispute_system.get_case(case_id)
        assert case.status == DisputeStatus.RESOLVED
        assert case.verdict is not None
        assert case.verdict.in_favor_of_plaintiff is True
        assert case.verdict.confidence_score >= 0.66  # 2/3 majority


class TestByzantineFaultTolerance:
    """Test Byzantine fault tolerance in dispute resolution."""
    
    @pytest.mark.asyncio
    async def test_byzantine_votes_overcome(self, dispute_system):
        """Test that Byzantine votes cannot overcome honest majority."""
        case_id = await dispute_system.create_dispute_case(
            dispute_type=DisputeType.GRADIENT_MANIPULATION,
            plaintiff_id='hospital_1',
            defendant_id='hospital_2',
            description='Gradient manipulation with Byzantine attackers'
        )
        
        case = await dispute_system.get_case(case_id)
        case.status = DisputeStatus.VOTING
        
        # Simulate Byzantine attack: 30% malicious nodes voting incorrectly
        # But honest majority (70%) votes correctly
        total_voters = 100
        honest_voters = 70
        
        for i in range(total_voters):
            # Honest nodes vote for plaintiff (correct)
            # Byzantine nodes vote for defendant (incorrect)
            is_honest = i < honest_voters
            vote_for_plaintiff = is_honest
            
            await dispute_system.cast_vote(
                case_id=case_id,
                hospital_id=f'hospital_{i}',
                vote_for_plaintiff=vote_for_plaintiff,
                reputation_score=0.9 if is_honest else 0.1
            )
            
        await asyncio.sleep(1)
        
        # Despite Byzantine votes, honest majority should win
        case = await dispute_system.get_case(case_id)
        assert case.verdict is not None
        assert case.verdict.in_favor_of_plaintiff is True
        
    @pytest.mark.asyncio
    async def test_insufficient_votes_no_consensus(self, dispute_system):
        """Test that insufficient votes don't trigger resolution."""
        case_id = await dispute_system.create_dispute_case(
            dispute_type=DisputeType.MODEL_QUALITY_CLAIM,
            plaintiff_id='hospital_1',
            defendant_id='hospital_2',
            description='Model quality dispute'
        )
        
        case = await dispute_system.get_case(case_id)
        case.status = DisputeStatus.VOTING
        
        # Cast insufficient votes (less than 2f+1)
        for i in range(5):  # Too few votes
            await dispute_system.cast_vote(
                case_id=case_id,
                hospital_id=f'hospital_{i}',
                vote_for_plaintiff=True,
                reputation_score=0.8
            )
            
        # Case should remain in voting state
        case = await dispute_system.get_case(case_id)
        assert case.status == DisputeStatus.VOTING
        assert case.verdict is None


class TestVerdictExecution:
    """Test verdict execution and penalties."""
    
    @pytest.mark.asyncio
    async def test_penalty_application(self, dispute_system):
        """Test that penalties are correctly applied."""
        case_id = await dispute_system.create_dispute_case(
            dispute_type=DisputeType.GRADIENT_MANIPULATION,
            plaintiff_id='hospital_1',
            defendant_id='hospital_2',
            description='Malicious gradient manipulation'
        )
        
        # Force resolution with plaintiff win
        case = await dispute_system.get_case(case_id)
        case.status = DisputeStatus.VOTING
        
        # Add enough votes for consensus
        for i in range(70):
            await dispute_system.cast_vote(
                case_id=case_id,
                hospital_id=f'hospital_{i}',
                vote_for_plaintiff=True,
                reputation_score=0.9
            )
            
        await asyncio.sleep(1)
        
        # Verify penalties were applied
        case = await dispute_system.get_case(case_id)
        assert case.verdict is not None
        assert case.defendant_id in case.verdict.penalties
        
        penalty = case.verdict.penalties[case.defendant_id]
        assert 'reputation_penalty' in penalty
        assert 'healthtoken_penalty' in penalty
        assert penalty['reputation_penalty'] == 0.1
        assert penalty['healthtoken_penalty'] == 100
        
    @pytest.mark.asyncio
    async def test_reward_distribution(self, dispute_system):
        """Test that correct voters are rewarded."""
        case_id = await dispute_system.create_dispute_case(
            dispute_type=DisputeType.REPUTATION_ATTACK,
            plaintiff_id='hospital_1',
            defendant_id='hospital_2',
            description='Reputation attack case'
        )
        
        case = await dispute_system.get_case(case_id)
        case.status = DisputeStatus.VOTING
        
        # Voters who vote correctly (for plaintiff)
        correct_voters = ['hospital_3', 'hospital_4', 'hospital_5']
        
        for voter in correct_voters:
            await dispute_system.cast_vote(
                case_id=case_id,
                hospital_id=voter,
                vote_for_plaintiff=True,
                reputation_score=0.8
            )
            
        # Add more votes to reach consensus
        for i in range(10, 70):
            await dispute_system.cast_vote(
                case_id=case_id,
                hospital_id=f'hospital_{i}',
                vote_for_plaintiff=True,
                reputation_score=0.8
            )
            
        await asyncio.sleep(1)
        
        # Verify rewards
        case = await dispute_system.get_case(case_id)
        assert case.verdict is not None
        
        for voter in correct_voters:
            assert voter in case.verdict.rewards
            assert case.verdict.rewards[voter] == 10.0  # Correct voter reward


class TestAppealProcess:
    """Test the appeal process for disputed verdicts."""
    
    @pytest.mark.asyncio
    async def test_successful_appeal(self, dispute_system):
        """Test filing a successful appeal."""
        case_id = await dispute_system.create_dispute_case(
            dispute_type=DisputeType.PRIVACY_VIOLATION,
            plaintiff_id='hospital_1',
            defendant_id='hospital_2',
            description='Privacy violation'
        )
        
        # Force resolution
        case = await dispute_system.get_case(case_id)
        case.status = DisputeStatus.RESOLVED
        case.verdict = DisputeVerdict(
            case_id=case_id,
            verdict_timestamp=datetime.utcnow(),
            in_favor_of_plaintiff=True,
            confidence_score=0.8,
            reasoning='Test verdict',
            penalties={},
            rewards={},
            appeal_allowed=True
        )
        case.appeal_deadline = datetime.utcnow() + timedelta(days=7)
        
        # File appeal
        appeal_result = await dispute_system.appeal_case(
            case_id=case_id,
            appeal_reason='New evidence available'
        )
        
        assert appeal_result is True
        
        # Verify appeal case was created
        active_cases = await dispute_system.get_active_cases()
        assert len(active_cases) >= 2  # Original + appeal case
        
    @pytest.mark.asyncio
    async def test_appeal_after_deadline_fails(self, dispute_system):
        """Test that appeals after deadline fail."""
        case_id = await dispute_system.create_dispute_case(
            dispute_type=DisputeType.MODEL_QUALITY_CLAIM,
            plaintiff_id='hospital_1',
            defendant_id='hospital_2',
            description='Model quality dispute'
        )
        
        # Set case as resolved with expired appeal deadline
        case = await dispute_system.get_case(case_id)
        case.status = DisputeStatus.RESOLVED
        case.verdict = DisputeVerdict(
            case_id=case_id,
            verdict_timestamp=datetime.utcnow(),
            in_favor_of_plaintiff=False,
            confidence_score=0.7,
            reasoning='Test verdict',
            penalties={},
            rewards={},
            appeal_allowed=True
        )
        case.appeal_deadline = datetime.utcnow() - timedelta(days=1)  # Expired
        
        # Attempt appeal
        appeal_result = await dispute_system.appeal_case(
            case_id=case_id,
            appeal_reason='This should fail'
        )
        
        assert appeal_result is False


class TestMetricsAndMonitoring:
    """Test metrics collection and monitoring."""
    
    @pytest.mark.asyncio
    async def test_resolution_time_metric(self, dispute_system):
        """Test that resolution time is within acceptable bounds."""
        start_time = datetime.utcnow()
        
        case_id = await dispute_system.create_dispute_case(
            dispute_type=DisputeType.GRADIENT_MANIPULATION,
            plaintiff_id='hospital_1',
            defendant_id='hospital_2',
            description='Test resolution time'
        )
        
        # Fast-track to voting and resolution
        case = await dispute_system.get_case(case_id)
        case.status = DisputeStatus.VOTING
        
        # Add sufficient votes quickly
        for i in range(70):
            await dispute_system.cast_vote(
                case_id=case_id,
                hospital_id=f'hospital_{i}',
                vote_for_plaintiff=True,
                reputation_score=0.9
            )
            
        await asyncio.sleep(1)
        
        end_time = datetime.utcnow()
        resolution_time = (end_time - start_time).total_seconds()
        
        # Should resolve within timeout period (5 seconds for tests)
        assert resolution_time < 30  # Configured timeout
        
        case = await dispute_system.get_case(case_id)
        assert case.status == DisputeStatus.RESOLVED
        
    @pytest.mark.asyncio
    async def test_consensus_accuracy_metric(self, dispute_system):
        """Test consensus accuracy under various conditions."""
        test_cases = [
            (80, True, "Clear majority for plaintiff"),
            (30, False, "Clear majority for defendant"),
            (50, None, "No clear majority")
        ]
        
        for vote_ratio, expected_winner, description in test_cases:
            case_id = await dispute_system.create_dispute_case(
                dispute_type=DisputeType.REPUTATION_ATTACK,
                plaintiff_id='hospital_1',
                defendant_id='hospital_2',
                description=f'Test case: {description}'
            )
            
            case = await dispute_system.get_case(case_id)
            case.status = DisputeStatus.VOTING
            
            # Cast votes according to ratio
            total_voters = 100
            plaintiff_votes = int(total_voters * vote_ratio / 100)
            
            for i in range(total_voters):
                vote_for_plaintiff = i < plaintiff_votes
                await dispute_system.cast_vote(
                    case_id=case_id,
                    hospital_id=f'hospital_{i}',
                    vote_for_plaintiff=vote_for_plaintiff,
                    reputation_score=0.8
                )
                
            await asyncio.sleep(1)
            
            case = await dispute_system.get_case(case_id)
            
            if expected_winner is True:
                assert case.verdict.in_favor_of_plaintiff is True
            elif expected_winner is False:
                assert case.verdict.in_favor_of_plaintiff is False
            else:
                # No consensus case
                assert case.status == DisputeStatus.VOTING or case.verdict is None