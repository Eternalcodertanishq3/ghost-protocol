import React, { useState, useEffect } from 'react';
import { 
  Card, 
  CardContent, 
  Typography, 
  Box, 
  Grid,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  Avatar
} from '@mui/material';
import { 
  Shield as ShieldIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  EmojiEvents as TrophyIcon
} from '@mui/icons-material';

const ReputationLeaderboard = () => {
  const [hospitals, setHospitals] = useState([
    { id: 'H001', name: 'AIIMS Delhi', reputation: 0.95, tokens: 1250, last_update: '2 min ago', status: 'online' },
    { id: 'H002', name: 'CMC Vellore', reputation: 0.92, tokens: 1180, last_update: '5 min ago', status: 'online' },
    { id: 'H007', name: 'PGIMER Chandigarh', reputation: 0.90, tokens: 1150, last_update: '1 min ago', status: 'training' },
    { id: 'H003', name: 'Tata Memorial Mumbai', reputation: 0.88, tokens: 1090, last_update: '8 min ago', status: 'online' },
    { id: 'H008', name: 'SCTIMST Trivandrum', reputation: 0.87, tokens: 1080, last_update: '3 min ago', status: 'online' },
    { id: 'H004', name: 'Apollo Chennai', reputation: 0.85, tokens: 1050, last_update: '12 min ago', status: 'online' },
    { id: 'H006', name: 'Kolkata Medical College', reputation: 0.82, tokens: 1020, last_update: '15 min ago', status: 'online' },
    { id: 'H005', name: 'Fortis Bangalore', reputation: 0.78, tokens: 980, last_update: '1 hour ago', status: 'offline' },
  ]);

  const getReputationColor = (reputation) => {
    if (reputation >= 0.9) return '#00ff88';
    if (reputation >= 0.8) return '#ffa502';
    return '#ff4757';
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'online': return '#00ff88';
      case 'training': return '#ffa502';
      case 'offline': return '#ff4757';
      default: return '#666';
    }
  };

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
          <Typography variant="h5" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <TrophyIcon />
            Reputation Leaderboard
          </Typography>
          <Chip 
            icon={<ShieldIcon />} 
            label="Live Rankings" 
            color="success" 
            variant="outlined" 
          />
        </Box>

        <Grid container spacing={3}>
          {/* Top 3 Hospitals */}
          <Grid item xs={12}>
            <Box sx={{ display: 'flex', justifyContent: 'center', gap: 3, mb: 3 }}>
              {hospitals.slice(0, 3).map((hospital, index) => (
                <Box 
                  key={hospital.id}
                  sx={{ 
                    p: 3, 
                    background: index === 0 ? 'rgba(255, 215, 0, 0.1)' : 
                               index === 1 ? 'rgba(192, 192, 192, 0.1)' : 
                               'rgba(205, 127, 50, 0.1)',
                    border: `1px solid ${index === 0 ? '#FFD700' : 
                                  index === 1 ? '#C0C0C0' : 
                                  '#CD7F32'}`,
                    borderRadius: '12px',
                    textAlign: 'center',
                    minWidth: '200px'
                  }}
                >
                  <Typography variant="h6" sx={{ color: index === 0 ? '#FFD700' : 
                                                      index === 1 ? '#C0C0C0' : 
                                                      '#CD7F32', mb: 1 }}>
                    #{index + 1}
                  </Typography>
                  <Avatar sx={{ bgcolor: getReputationColor(hospital.reputation), mx: 'auto', mb: 1 }}>
                    {hospital.name.charAt(0)}
                  </Avatar>
                  <Typography variant="body1" sx={{ fontWeight: 'bold', mb: 0.5 }}>
                    {hospital.name}
                  </Typography>
                  <Typography variant="h6" sx={{ color: getReputationColor(hospital.reputation) }}>
                    {(hospital.reputation * 100).toFixed(1)}%
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {hospital.tokens} HealthTokens
                  </Typography>
                </Box>
              ))}
            </Box>
          </Grid>

          {/* Full Leaderboard Table */}
          <Grid item xs={12}>
            <TableContainer component={Paper} sx={{ 
              background: 'rgba(255, 255, 255, 0.02)',
              border: '1px solid rgba(255, 255, 255, 0.1)'
            }}>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell sx={{ color: '#00ff88', fontWeight: 'bold' }}>Rank</TableCell>
                    <TableCell sx={{ color: '#00ff88', fontWeight: 'bold' }}>Hospital</TableCell>
                    <TableCell sx={{ color: '#00ff88', fontWeight: 'bold' }}>Reputation</TableCell>
                    <TableCell sx={{ color: '#00ff88', fontWeight: 'bold' }}>HealthTokens</TableCell>
                    <TableCell sx={{ color: '#00ff88', fontWeight: 'bold' }}>Status</TableCell>
                    <TableCell sx={{ color: '#00ff88', fontWeight: 'bold' }}>Last Update</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {hospitals.map((hospital, index) => (
                    <TableRow key={hospital.id} sx={{ 
                      '&:hover': { background: 'rgba(255, 255, 255, 0.05)' }
                    }}>
                      <TableCell sx={{ color: '#ffffff', fontWeight: 'bold' }}>
                        #{index + 1}
                      </TableCell>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                          <Avatar sx={{ bgcolor: getReputationColor(hospital.reputation), width: 32, height: 32 }}>
                            {hospital.name.charAt(0)}
                          </Avatar>
                          <Box>
                            <Typography variant="body1" sx={{ color: '#ffffff', fontWeight: 'bold' }}>
                              {hospital.name}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              {hospital.id}
                            </Typography>
                          </Box>
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <TrendingUpIcon sx={{ color: getReputationColor(hospital.reputation) }} />
                          <Typography variant="body1" sx={{ color: getReputationColor(hospital.reputation), fontWeight: 'bold' }}>
                            {(hospital.reputation * 100).toFixed(1)}%
                          </Typography>
                        </Box>
                      </TableCell>
                      <TableCell sx={{ color: '#00ccff', fontWeight: 'bold' }}>
                        {hospital.tokens.toLocaleString()}
                      </TableCell>
                      <TableCell>
                        <Chip 
                          label={hospital.status.toUpperCase()} 
                          size="small" 
                          sx={{ 
                            backgroundColor: getStatusColor(hospital.status),
                            color: '#000',
                            fontWeight: 'bold'
                          }}
                        />
                      </TableCell>
                      <TableCell sx={{ color: 'text.secondary' }}>
                        {hospital.last_update}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Grid>
        </Grid>

        {/* Statistics Footer */}
        <Box sx={{ mt: 3, p: 2, background: 'rgba(0, 255, 136, 0.1)', borderRadius: '8px' }}>
          <Grid container spacing={2}>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">Average Reputation</Typography>
              <Typography variant="h6" sx={{ color: '#00ff88' }}>
                {(hospitals.reduce((sum, h) => sum + h.reputation, 0) / hospitals.length * 100).toFixed(1)}%
              </Typography>
            </Grid>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">Total HealthTokens</Typography>
              <Typography variant="h6" sx={{ color: '#00ccff' }}>
                {hospitals.reduce((sum, h) => sum + h.tokens, 0).toLocaleString()}
              </Typography>
            </Grid>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">Online Hospitals</Typography>
              <Typography variant="h6" sx={{ color: '#00ff88' }}>
                {hospitals.filter(h => h.status === 'online').length}/{hospitals.length}
              </Typography>
            </Grid>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">Last Updated</Typography>
              <Typography variant="h6" sx={{ color: '#ffa502' }}>
                Live
              </Typography>
            </Grid>
          </Grid>
        </Box>
      </CardContent>
    </Card>
  );
};

export default ReputationLeaderboard;