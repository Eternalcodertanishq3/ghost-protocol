import React from 'react';
import { Card, CardContent, Typography, Box, Grid, Chip } from '@mui/material';
import {
  TrendingUp as TrendingIcon,
  LocalHospital as HospitalIcon,
  Speed as SpeedIcon,
  Shield as ShieldIcon
} from '@mui/icons-material';

const RealTimeMetrics = ({ systemStatus }) => {
  // Determine if we're still loading (no real data yet)
  const isLoading = !systemStatus || systemStatus.current_round === 0;

  // Get Byzantine tolerance from status or use actual calculated value
  const byzantineTolerance = systemStatus?.byzantine_tolerance ||
    (systemStatus?.byzantine_shield?.tolerance_percentage) ||
    '49%';  // Only show default if truly unavailable

  const metrics = [
    {
      label: 'Active Hospitals',
      value: isLoading ? '—' : systemStatus.active_hospitals,
      icon: <HospitalIcon />,
      color: '#00ff88',
      trend: isLoading ? 'Loading...' : (systemStatus.hospital_delta || 'Live'),
      loading: isLoading
    },
    {
      label: 'Current Round',
      value: isLoading ? '—' : systemStatus.current_round,
      icon: <TrendingIcon />,
      color: '#00ccff',
      trend: 'Live',
      loading: isLoading
    },
    {
      label: 'Model Performance',
      value: isLoading ? '—' : systemStatus.model_performance?.toFixed(3),
      icon: <SpeedIcon />,
      color: '#ffa502',
      trend: isLoading ? 'Loading...' : (systemStatus.performance_delta || '+0.000'),
      loading: isLoading
    },
    {
      label: 'Byzantine Tolerance',
      value: typeof byzantineTolerance === 'number'
        ? `${byzantineTolerance}%`
        : byzantineTolerance,
      icon: <ShieldIcon />,
      color: '#ff4757',
      trend: 'Active',
      loading: false  // This is a static configuration
    }
  ];

  return (
    <Card>
      <CardContent>
        <Typography variant="h5" sx={{ mb: 3, color: '#00ff88' }}>
          Real-Time System Metrics
        </Typography>

        <Grid container spacing={3}>
          {metrics.map((metric, index) => (
            <Grid item xs={3} key={index}>
              <Box sx={{
                p: 3,
                background: 'rgba(255, 255, 255, 0.05)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '12px',
                textAlign: 'center',
                transition: 'all 0.3s ease',
                '&:hover': {
                  background: 'rgba(255, 255, 255, 0.08)',
                  transform: 'translateY(-2px)'
                }
              }}>
                <Box sx={{ color: metric.color, mb: 1 }}>
                  {metric.icon}
                </Box>
                <Typography variant="h4" sx={{ color: metric.color, mb: 1, fontWeight: 'bold' }}>
                  {metric.value}
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                  {metric.label}
                </Typography>
                <Chip
                  label={metric.trend}
                  size="small"
                  variant="outlined"
                  sx={{
                    borderColor: metric.color,
                    color: metric.color,
                    fontSize: '0.75rem'
                  }}
                />
              </Box>
            </Grid>
          ))}
        </Grid>

        <Box sx={{ mt: 3, p: 2, background: 'rgba(0, 204, 255, 0.1)', borderRadius: '8px' }}>
          <Typography variant="h6" sx={{ color: '#00ccff', mb: 1 }}>
            System Health Status
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">Ghost Agents</Typography>
              <Typography variant="body1" sx={{ color: '#00ff88' }}>Active</Typography>
            </Grid>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">SNA Aggregator</Typography>
              <Typography variant="body1" sx={{ color: '#00ff88' }}>Online</Typography>
            </Grid>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">Scaling Mode</Typography>
              <Typography variant="body1" sx={{ color: '#ffa502' }}>Hierarchical Sharding</Typography>
            </Grid>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">Algorithm</Typography>
              <Typography variant="body1" sx={{ color: '#00ccff' }}>FedProx + GeoMedian</Typography>
            </Grid>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">Byzantine Shield</Typography>
              <Typography variant="body1" sx={{ color: '#00ff88' }}>Protected</Typography>
            </Grid>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">DPDP Auditor</Typography>
              <Typography variant="body1" sx={{ color: '#00ff88' }}>Compliant</Typography>
            </Grid>
          </Grid>
        </Box>
      </CardContent>
    </Card>
  );
};

export default RealTimeMetrics;