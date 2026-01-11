import React, { useState, useEffect } from 'react';
import { Card, CardContent, Typography, Box, Grid, Chip } from '@mui/material';
import { AccountBalance as TokenIcon, TrendingUp as TrendingIcon } from '@mui/icons-material';

const HealthTokenDashboard = () => {
  const [tokenStats, setTokenStats] = useState({
    total_tokens: 50000,
    total_hospitals: 50,
    average_balance: 1000,
    rewards_distributed: 12500,
    top_holder: { name: 'AIIMS Delhi', balance: 1250 }
  });

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
          <Typography variant="h5" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <TokenIcon />
            HealthToken Economy
          </Typography>
          <Chip 
            icon={<TrendingIcon />} 
            label="Shapley Rewards Active" 
            color="success" 
            variant="outlined" 
          />
        </Box>

        <Grid container spacing={3}>
          {[
            { label: 'Total HealthTokens', value: tokenStats.total_tokens.toLocaleString(), color: '#00ff88' },
            { label: 'Active Hospitals', value: tokenStats.total_hospitals, color: '#00ccff' },
            { label: 'Average Balance', value: tokenStats.average_balance.toLocaleString(), color: '#ffa502' },
            { label: 'Rewards Distributed', value: tokenStats.rewards_distributed.toLocaleString(), color: '#ff4757' }
          ].map((stat, index) => (
            <Grid item xs={3} key={index}>
              <Box sx={{ 
                p: 2, 
                background: 'rgba(255, 255, 255, 0.05)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '8px',
                textAlign: 'center'
              }}>
                <Typography variant="h4" sx={{ color: stat.color, mb: 1, fontWeight: 'bold' }}>
                  {stat.value}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {stat.label}
                </Typography>
              </Box>
            </Grid>
          ))}
        </Grid>

        <Box sx={{ mt: 3, p: 2, background: 'rgba(0, 255, 136, 0.1)', borderRadius: '8px' }}>
          <Typography variant="h6" sx={{ color: '#00ff88', mb: 1 }}>
            Top HealthToken Holder
          </Typography>
          <Typography variant="body1">
            {tokenStats.top_holder.name} - {tokenStats.top_holder.balance.toLocaleString()} HealthTokens
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

export default HealthTokenDashboard;