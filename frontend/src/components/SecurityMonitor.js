import React, { useState, useEffect } from 'react';
import { 
  Card, 
  CardContent, 
  Typography, 
  Box, 
  Grid, 
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper
} from '@mui/material';
import { 
  Security as SecurityIcon,
  Shield as ShieldIcon,
  Warning as WarningIcon,
  CheckCircle as CheckIcon,
  Error as ErrorIcon
} from '@mui/icons-material';

const SecurityMonitor = () => {
  const [securityEvents, setSecurityEvents] = useState([]);
  const [threatLevel, setThreatLevel] = useState('LOW');
  const [metrics, setMetrics] = useState({
    anomalies_detected: 0,
    byzantine_nodes_blocked: 0,
    privacy_violations: 0,
    attacks_blocked: 0
  });

  // Simulate security events
  const generateSecurityEvents = () => {
    const events = [
      {
        timestamp: new Date().toISOString(),
        type: 'ANOMALY',
        hospital_id: 'H005',
        description: 'Gradient anomaly detected - Z-score > 3.0',
        severity: 'MEDIUM',
        action: 'Update quarantined'
      },
      {
        timestamp: new Date(Date.now() - 300000).toISOString(),
        type: 'BYZANTINE',
        hospital_id: 'H003',
        description: 'Potential Byzantine attack - Sign flip detected',
        severity: 'HIGH',
        action: 'Node blocked, reputation penalized'
      },
      {
        timestamp: new Date(Date.now() - 600000).toISOString(),
        type: 'PRIVACY',
        hospital_id: 'H007',
        description: 'Privacy budget ε=8.2 approaching limit',
        severity: 'MEDIUM',
        action: 'Warning issued'
      },
      {
        timestamp: new Date(Date.now() - 900000).toISOString(),
        type: 'ATTACK',
        hospital_id: 'EXTERNAL',
        description: 'DDoS attack detected and blocked',
        severity: 'LOW',
        action: 'IP blocked, rate limit enforced'
      }
    ];

    setSecurityEvents(events);
  };

  useEffect(() => {
    generateSecurityEvents();
    
    // Update metrics
    setMetrics({
      anomalies_detected: 12,
      byzantine_nodes_blocked: 3,
      privacy_violations: 1,
      attacks_blocked: 45
    });

    // Update threat level based on recent events
    const interval = setInterval(() => {
      const threatScore = Math.random();
      if (threatScore > 0.7) {
        setThreatLevel('HIGH');
      } else if (threatScore > 0.3) {
        setThreatLevel('MEDIUM');
      } else {
        setThreatLevel('LOW');
      }
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const getThreatLevelColor = (level) => {
    switch (level) {
      case 'LOW': return '#00ff88';
      case 'MEDIUM': return '#ffa502';
      case 'HIGH': return '#ff4757';
      default: return '#666';
    }
  };

  const getSeverityIcon = (severity) => {
    switch (severity) {
      case 'LOW': return <CheckIcon sx={{ color: '#00ff88' }} />;
      case 'MEDIUM': return <WarningIcon sx={{ color: '#ffa502' }} />;
      case 'HIGH': return <ErrorIcon sx={{ color: '#ff4757' }} />;
      default: return <SecurityIcon />;
    }
  };

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'LOW': return 'success';
      case 'MEDIUM': return 'warning';
      case 'HIGH': return 'error';
      default: return 'default';
    }
  };

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
          <Typography variant="h5" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <SecurityIcon />
            Security Monitor
          </Typography>
          
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Typography variant="body2" color="text.secondary">
              Threat Level:
            </Typography>
            <Chip
              icon={<ShieldIcon />}
              label={threatLevel}
              color={getSeverityColor(threatLevel)}
              variant="outlined"
              sx={{
                backgroundColor: `${getThreatLevelColor(threatLevel)}20`,
                borderColor: getThreatLevelColor(threatLevel),
                color: getThreatLevelColor(threatLevel),
                fontWeight: 'bold'
              }}
            />
          </Box>
        </Box>

        {/* Security Metrics */}
        <Grid container spacing={3} sx={{ mb: 3 }}>
          <Grid item xs={3}>
            <Box sx={{ 
              p: 2, 
              background: 'rgba(255, 165, 2, 0.1)',
              border: '1px solid rgba(255, 165, 2, 0.3)',
              borderRadius: '8px',
              textAlign: 'center'
            }}>
              <Typography variant="h4" sx={{ color: '#ffa502', mb: 1 }}>
                {metrics.anomalies_detected}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Anomalies Detected
              </Typography>
            </Box>
          </Grid>
          <Grid item xs={3}>
            <Box sx={{ 
              p: 2, 
              background: 'rgba(255, 71, 87, 0.1)',
              border: '1px solid rgba(255, 71, 87, 0.3)',
              borderRadius: '8px',
              textAlign: 'center'
            }}>
              <Typography variant="h4" sx={{ color: '#ff4757', mb: 1 }}>
                {metrics.byzantine_nodes_blocked}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Byzantine Nodes Blocked
              </Typography>
            </Box>
          </Grid>
          <Grid item xs={3}>
            <Box sx={{ 
              p: 2, 
              background: 'rgba(0, 204, 255, 0.1)',
              border: '1px solid rgba(0, 204, 255, 0.3)',
              borderRadius: '8px',
              textAlign: 'center'
            }}>
              <Typography variant="h4" sx={{ color: '#00ccff', mb: 1 }}>
                {metrics.privacy_violations}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Privacy Violations
              </Typography>
            </Box>
          </Grid>
          <Grid item xs={3}>
            <Box sx={{ 
              p: 2, 
              background: 'rgba(0, 255, 136, 0.1)',
              border: '1px solid rgba(0, 255, 136, 0.3)',
              borderRadius: '8px',
              textAlign: 'center'
            }}>
              <Typography variant="h4" sx={{ color: '#00ff88', mb: 1 }}>
                {metrics.attacks_blocked}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Attacks Blocked
              </Typography>
            </Box>
          </Grid>
        </Grid>

        {/* Security Events Table */}
        <Typography variant="h6" sx={{ mb: 2, color: '#00ff88' }}>
          Recent Security Events
        </Typography>
        
        <TableContainer component={Paper} sx={{ 
          background: 'rgba(255, 255, 255, 0.02)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          borderRadius: '8px'
        }}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell sx={{ color: '#00ff88', fontWeight: 'bold' }}>Time</TableCell>
                <TableCell sx={{ color: '#00ff88', fontWeight: 'bold' }}>Type</TableCell>
                <TableCell sx={{ color: '#00ff88', fontWeight: 'bold' }}>Hospital</TableCell>
                <TableCell sx={{ color: '#00ff88', fontWeight: 'bold' }}>Description</TableCell>
                <TableCell sx={{ color: '#00ff88', fontWeight: 'bold' }}>Severity</TableCell>
                <TableCell sx={{ color: '#00ff88', fontWeight: 'bold' }}>Action</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {securityEvents.map((event, index) => (
                <TableRow key={index} sx={{ 
                  '&:hover': { background: 'rgba(255, 255, 255, 0.05)' }
                }}>
                  <TableCell sx={{ color: '#ffffff' }}>
                    {formatTimestamp(event.timestamp)}
                  </TableCell>
                  <TableCell sx={{ color: '#ffffff' }}>
                    <Chip 
                      label={event.type} 
                      size="small" 
                      variant="outlined"
                      sx={{ 
                        borderColor: getThreatLevelColor(event.severity),
                        color: getThreatLevelColor(event.severity)
                      }}
                    />
                  </TableCell>
                  <TableCell sx={{ color: '#ffffff' }}>
                    {event.hospital_id}
                  </TableCell>
                  <TableCell sx={{ color: '#ffffff' }}>
                    {event.description}
                  </TableCell>
                  <TableCell>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      {getSeverityIcon(event.severity)}
                      <Chip 
                        label={event.severity} 
                        size="small" 
                        color={getSeverityColor(event.severity)}
                        variant="outlined"
                      />
                    </Box>
                  </TableCell>
                  <TableCell sx={{ color: '#00ccff' }}>
                    {event.action}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>

        {/* System Status */}
        <Box sx={{ mt: 3, p: 2, background: 'rgba(0, 255, 136, 0.1)', borderRadius: '8px' }}>
          <Typography variant="h6" sx={{ color: '#00ff88', mb: 1 }}>
            System Defense Status
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={4}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <ShieldIcon sx={{ color: '#00ff88' }} />
                <Typography variant="body2">Byzantine Shield: Active</Typography>
              </Box>
            </Grid>
            <Grid item xs={4}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <SecurityIcon sx={{ color: '#00ff88' }} />
                <Typography variant="body2">Anomaly Detection: Active</Typography>
              </Box>
            </Grid>
            <Grid item xs={4}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <CheckIcon sx={{ color: '#00ff88' }} />
                <Typography variant="body2">DPDP Auditor: Compliant</Typography>
              </Box>
            </Grid>
          </Grid>
        </Box>
      </CardContent>
    </Card>
  );
};

export default SecurityMonitor;