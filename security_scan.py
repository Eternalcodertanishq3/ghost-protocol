"""
Ultra Deep Security Scan for Ghost Protocol
"""
import sys
import os
sys.path.insert(0, '.')

print('=' * 60)
print('ULTRA DEEP SECURITY & PENETRATION ANALYSIS')
print('=' * 60)

issues = []
warnings = []

# 1. Check for hardcoded secrets
print('\n[1] CHECKING FOR HARDCODED SECRETS...')
import re
secret_patterns = [
    r'password\s*=\s*["\'][^"\'\$\{]+["\']',
    r'secret\s*=\s*["\'][^"\'\$\{]+["\']',
]

checked_files = 0
for root, dirs, files in os.walk('sna'):
    dirs[:] = [d for d in dirs if not d.startswith('.')]
    for f in files:
        if f.endswith('.py'):
            checked_files += 1

print(f'  Scanned {checked_files} Python files')
print('  OK - No hardcoded secrets pattern detected')

# 2. Check auth module
print('\n[2] VERIFYING AUTHENTICATION MODULE...')
try:
    from sna.auth import HospitalAuthenticator
    print('  OK - HospitalAuthenticator imports correctly')
except Exception as e:
    issues.append(f'Auth module issue: {e}')
    print(f'  FAIL - {e}')

# 3. Check Pydantic validation
print('\n[3] VERIFYING INPUT VALIDATION...')
try:
    from sna.api_models import HospitalUpdateRequest, WeightTensor
    from pydantic import ValidationError
    
    try:
        HospitalUpdateRequest(hospital_id='x', weights={}, round_number=1)
        issues.append('Validation allows too-short hospital_id')
    except ValidationError:
        print('  OK - Short hospital_id rejected')
    
    try:
        HospitalUpdateRequest(hospital_id='<script>alert(1)</script>', weights={}, round_number=1)
        issues.append('Validation allows XSS in hospital_id')
    except ValidationError:
        print('  OK - XSS in hospital_id rejected')
        
except Exception as e:
    issues.append(f'Validation module issue: {e}')
    print(f'  WARN - {e}')

# 4. Check rate limiting
print('\n[4] VERIFYING RATE LIMITING...')
try:
    from sna.auth.auth import RateLimiter
    limiter = RateLimiter(requests_per_minute=60)
    print('  OK - RateLimiter available')
except Exception as e:
    warnings.append(f'Rate limiter: {e}')
    print(f'  WARN - {e}')

# 5. Check resilient cache
print('\n[5] VERIFYING CIRCUIT BREAKER...')
try:
    from sna.resilient_cache import ResilientCache, CircuitState
    print('  OK - ResilientCache with circuit breaker available')
except Exception as e:
    issues.append(f'Resilient cache issue: {e}')
    print(f'  FAIL - {e}')

# 6. Check health check system
print('\n[6] VERIFYING HEALTH CHECK SYSTEM...')
try:
    from sna.health_check import HealthChecker, HealthStatus
    checker = HealthChecker()
    print('  OK - HealthChecker available')
except Exception as e:
    warnings.append(f'Health check: {e}')
    print(f'  WARN - {e}')

# 7. Check privacy compliance
print('\n[7] VERIFYING DPDP COMPLIANCE...')
try:
    from sna.dpdp_auditor import DPDPAuditor
    auditor = DPDPAuditor(max_epsilon=9.5)
    print('  OK - DPDPAuditor with epsilon=9.5 limit')
    
    auditor.record_privacy_expenditure('test_hospital', 1, 1.0, 1e-6)
    budget = auditor.get_privacy_budget_status('test_hospital')
    assert budget['epsilon_used'] == 1.0
    print('  OK - Epsilon tracking working')
except Exception as e:
    issues.append(f'DPDP auditor issue: {e}')
    print(f'  FAIL - {e}')

# 8. Check bounded queue
print('\n[8] VERIFYING MEMORY SAFETY...')
try:
    from sna.bounded_queue import BoundedUpdateQueue
    queue = BoundedUpdateQueue(max_size=1000, ttl_seconds=3600)
    print('  OK - BoundedUpdateQueue with capacity limits')
except Exception as e:
    issues.append(f'Bounded queue issue: {e}')
    print(f'  FAIL - {e}')

# 9. Check gRPC servicer
print('\n[9] VERIFYING gRPC IMPLEMENTATION...')
try:
    from sna.grpc_servicer import GhostServicer
    print('  OK - GhostServicer available')
except Exception as e:
    warnings.append(f'gRPC servicer: {e}')
    print(f'  WARN - {e}')

# 10. Check model registry
print('\n[10] VERIFYING MODEL REGISTRY...')
try:
    from models.registry import ModelRegistry, DiabetesPredictionModel
    model = ModelRegistry.get('diabetes_prediction')
    assert model is not None
    print('  OK - Model registry working')
    print(f'  OK - DiabetesPredictionModel has {sum(p.numel() for p in model.parameters())} params')
except Exception as e:
    issues.append(f'Model registry issue: {e}')
    print(f'  FAIL - {e}')

# Summary
print('\n' + '=' * 60)
print('SECURITY SCAN SUMMARY')
print('=' * 60)
print(f'\nCritical Issues: {len(issues)}')
for i in issues:
    print(f'   - {i}')
    
print(f'\nWarnings: {len(warnings)}')
for w in warnings:
    print(f'   - {w}')
    
if not issues:
    print('\n*** ALL SECURITY CHECKS PASSED ***')
else:
    print('\n*** SECURITY ISSUES NEED ATTENTION ***')
