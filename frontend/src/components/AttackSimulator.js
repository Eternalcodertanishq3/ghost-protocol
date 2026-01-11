import React, { useState } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Grid,
  Alert,
  Chip
} from '@mui/material';
import {
  BugReport as AttackIcon,
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  Security as SecurityIcon
} from '@mui/icons-material';

const AttackSimulator = ({ onAttack }) => {
  const [open, setOpen] = useState(false);
  const [attackType, setAttackType] = useState('');
  const [targetHospital, setTargetHospital] = useState('');
  const [isSimulating, setIsSimulating] = useState(false);
  const [results, setResults] = useState(null);

  const hospitals = [
    { id: 'H001', name: 'AIIMS Delhi' },
    { id: 'H002', name: 'CMC Vellore' },
    { id: 'H003', name: 'Tata Memorial Mumbai' },
    { id: 'H004', name: 'Apollo Chennai' },
    { id: 'H005', name: 'Fortis Bangalore' },
  ];

  const attackTypes = [
    { value: 'gradient_explosion', label: 'Gradient Explosion', description: 'Submit gradients with extremely large norms' },
    { value: 'sign_flip', label: 'Sign Flip Attack', description: 'Flip signs of all gradient values' },
    { value: 'label_poisoning', label: 'Label Poisoning', description: 'Corrupt training labels' },
    { value: 'sybil', label: 'Sybil Attack', description: 'Create multiple fake hospital nodes' },
  ];

  const handleSimulateAttack = async () => {
    if (!attackType || !targetHospital) return;

    setIsSimulating(true);

    // Notify parent to trigger visual effects
    if (onAttack) {
      onAttack({
        active: true,
        targetId: targetHospital,
        type: attackType,
        timestamp: new Date().toISOString()
      });
    }

    // Simulate attack execution
    setTimeout(() => {
      const attackResults = {
        attack_type: attackType,
        target_hospital: targetHospital,
        timestamp: new Date().toISOString(),
        before_attack: {
          model_accuracy: 0.852,
          reputation_score: 0.85,
          privacy_budget: 3.2
        },
        after_attack: {
          model_accuracy: attackType === 'gradient_explosion' ? 0.734 : 0.798,
          reputation_score: attackType === 'gradient_explosion' ? 0.65 : 0.72,
          privacy_budget: 3.2
        },
        defense_triggered: true,
        byzantine_shield_action: 'Node quarantined for 5 rounds',
        damage_mitigated: true
      };

      setResults(attackResults);
      setIsSimulating(false);
    }, 3000);
  };

  const resetSimulation = () => {
    setResults(null);
    setAttackType('');
    setTargetHospital('');
  };

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
          <Typography variant="h5" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <AttackIcon />
            Attack Simulator
          </Typography>
          <Chip
            icon={<SecurityIcon />}
            label="Testing Environment"
            color="warning"
            variant="outlined"
          />
        </Box>

        <Alert severity="warning" sx={{ mb: 3 }}>
          <Typography variant="body2">
            <strong>Warning:</strong> This is a controlled testing environment.
            Simulated attacks help validate the Byzantine Shield and defense mechanisms.
          </Typography>
        </Alert>

        <Grid container spacing={3}>
          {/* Attack Selection */}
          <Grid item xs={12} md={6}>
            <Box sx={{ p: 2, background: 'rgba(255, 255, 255, 0.05)', borderRadius: '8px' }}>
              <Typography variant="h6" sx={{ mb: 2, color: '#00ff88' }}>
                Select Attack Type
              </Typography>
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Attack Type</InputLabel>
                <Select
                  value={attackType}
                  onChange={(e) => setAttackType(e.target.value)}
                  label="Attack Type"
                  sx={{ color: '#fff' }}
                >
                  {attackTypes.map(type => (
                    <MenuItem key={type.value} value={type.value}>
                      {type.label}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              {attackType && (
                <Typography variant="body2" color="text.secondary">
                  {attackTypes.find(t => t.value === attackType)?.description}
                </Typography>
              )}
            </Box>
          </Grid>

          {/* Target Selection */}
          <Grid item xs={12} md={6}>
            <Box sx={{ p: 2, background: 'rgba(255, 255, 255, 0.05)', borderRadius: '8px' }}>
              <Typography variant="h6" sx={{ mb: 2, color: '#00ff88' }}>
                Select Target Hospital
              </Typography>
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Target Hospital</InputLabel>
                <Select
                  value={targetHospital}
                  onChange={(e) => setTargetHospital(e.target.value)}
                  label="Target Hospital"
                  sx={{ color: '#fff' }}
                >
                  {hospitals.map(hospital => (
                    <MenuItem key={hospital.id} value={hospital.id}>
                      {hospital.name} ({hospital.id})
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              {targetHospital && (
                <Typography variant="body2" color="text.secondary">
                  Target: {hospitals.find(h => h.id === targetHospital)?.name}
                </Typography>
              )}
            </Box>
          </Grid>
        </Grid>

        {/* Action Buttons */}
        <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center', mt: 3 }}>
          <Button
            variant="contained"
            color="error"
            startIcon={isSimulating ? null : <PlayIcon />}
            onClick={() => setOpen(true)}
            disabled={!attackType || !targetHospital || isSimulating}
            sx={{ minWidth: '150px' }}
          >
            {isSimulating ? 'Injecting Poison...' : 'Inject Poisoned Gradient'}
          </Button>
          <Button
            variant="outlined"
            onClick={resetSimulation}
            startIcon={<StopIcon />}
            sx={{ minWidth: '120px' }}
          >
            Reset
          </Button>
        </Box>

        {/* Results Display */}
        {results && (
          <Box sx={{ mt: 3, p: 2, background: 'rgba(0, 255, 136, 0.1)', borderRadius: '8px' }}>
            <Typography variant="h6" sx={{ color: '#00ff88', mb: 2 }}>
              Attack Simulation Results
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={4}>
                <Typography variant="body2" color="text.secondary">Model Accuracy</Typography>
                <Typography variant="h6">
                  {results.before_attack.model_accuracy} → {results.after_attack.model_accuracy}
                </Typography>
                <Typography variant="caption" sx={{ color: '#ff4757' }}>
                  {(results.before_attack.model_accuracy - results.after_attack.model_accuracy).toFixed(3)} drop
                </Typography>
              </Grid>
              <Grid item xs={4}>
                <Typography variant="body2" color="text.secondary">Reputation Score</Typography>
                <Typography variant="h6">
                  {results.before_attack.reputation_score} → {results.after_attack.reputation_score}
                </Typography>
                <Typography variant="caption" sx={{ color: '#ff4757' }}>
                  {(results.before_attack.reputation_score - results.after_attack.reputation_score).toFixed(2)} drop
                </Typography>
              </Grid>
              <Grid item xs={4}>
                <Typography variant="body2" color="text.secondary">Defense Status</Typography>
                <Typography variant="h6" sx={{ color: '#00ff88' }}>
                  {results.defense_triggered ? '✓ Active' : '✗ Failed'}
                </Typography>
                <Typography variant="caption" sx={{ color: '#00ccff' }}>
                  {results.byzantine_shield_action}
                </Typography>
              </Grid>
            </Grid>
          </Box>
        )}

        {/* Confirmation Dialog */}
        <Dialog open={open} onClose={() => setOpen(false)}>
          <DialogTitle sx={{ background: '#1a1a1a', color: '#fff' }}>
            Confirm Attack Simulation
          </DialogTitle>
          <DialogContent sx={{ background: '#1a1a1a', color: '#fff' }}>
            <Typography variant="body1" sx={{ mb: 2 }}>
              You are about to simulate a {attackTypes.find(t => t.value === attackType)?.label} attack
              on {hospitals.find(h => h.id === targetHospital)?.name}.
            </Typography>
            <Alert severity="warning">
              This is a controlled test environment. The attack will help validate
              the Byzantine Shield and defense mechanisms.
            </Alert>
          </DialogContent>
          <DialogActions sx={{ background: '#1a1a1a' }}>
            <Button onClick={() => setOpen(false)} sx={{ color: '#fff' }}>
              Cancel
            </Button>
            <Button
              onClick={() => {
                setOpen(false);
                handleSimulateAttack();
              }}
              color="error"
              variant="contained"
            >
              Confirm Attack
            </Button>
          </DialogActions>
        </Dialog>
      </CardContent>
    </Card>
  );
};

export default AttackSimulator;