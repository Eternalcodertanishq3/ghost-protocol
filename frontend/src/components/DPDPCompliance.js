import React from 'react';
import { Box, Chip, Typography } from '@mui/material';
import { Security as SecurityIcon, Warning as WarningIcon } from '@mui/icons-material';

const DPDPCompliance = ({ systemStatus }) => {
  const complianceStatus = systemStatus?.dpdp_compliant;
  const currentEpsilon = systemStatus?.epsilon_spent || 0;
  const maxEpsilon = 9.5;
  const utilization = (currentEpsilon / maxEpsilon) * 100;

  return (
    <Box sx={{
      p: 2,
      background: complianceStatus ? 'rgba(0, 255, 136, 0.1)' : 'rgba(255, 71, 87, 0.1)',
      border: `1px solid ${complianceStatus ? 'rgba(0, 255, 136, 0.3)' : 'rgba(255, 71, 87, 0.3)'}`,
      borderRadius: '8px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between'
    }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
        {complianceStatus ?
          <SecurityIcon sx={{ color: '#00ff88' }} /> :
          <WarningIcon sx={{ color: '#ff4757' }} />
        }
        <Box>
          <Typography variant="h6" sx={{ color: complianceStatus ? '#00ff88' : '#ff4757' }}>
            {complianceStatus ? '✓ DPDP Compliant' : '✗ DPDP Violation Detected'}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Privacy Budget: ε = {currentEpsilon.toFixed(2)} / {maxEpsilon}
            ({utilization.toFixed(1)}% utilized)
          </Typography>
        </Box>
      </Box>

      <Chip
        label={complianceStatus ? "Verifiable" : "Violation"}
        color={complianceStatus ? "success" : "error"}
        variant="outlined"
        sx={{
          backgroundColor: `${complianceStatus ? '#00ff88' : '#ff4757'}20`,
          borderColor: complianceStatus ? '#00ff88' : '#ff4757',
          color: complianceStatus ? '#00ff88' : '#ff4757',
          fontWeight: 'bold'
        }}
      />

      {/* Official Government Seal Animation */}
      {complianceStatus && (
        <Box
          sx={{
            position: 'absolute',
            right: 150,
            opacity: 0.2,
            transform: 'rotate(-15deg)',
            border: '4px double #00ff88',
            borderRadius: '50%',
            width: 120,
            height: 120,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexDirection: 'column',
            animation: 'stamp-entry 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards',
            pointerEvents: 'none'
          }}
        >
          <Box sx={{
            width: '90%',
            height: '90%',
            border: '1px solid #00ff88',
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexDirection: 'column'
          }}>
            <Typography variant="caption" sx={{ color: '#00ff88', fontSize: '10px', fontWeight: 'bold' }}>GOVT OF INDIA</Typography>
            <Typography variant="body2" sx={{ color: '#00ff88', fontWeight: '900', fontSize: '14px' }}>DPDP 2023</Typography>
            <Typography variant="body2" sx={{ color: '#00ff88', fontWeight: '900', fontSize: '12px' }}>COMPLIANT</Typography>
            <Typography variant="caption" sx={{ color: '#00ff88', fontSize: '8px' }}>§9(4) VERIFIED</Typography>
          </Box>
        </Box>
      )}

      <style>
        {`
          @keyframes stamp-entry {
            0% { transform: scale(3) rotate(-15deg); opacity: 0; }
            100% { transform: scale(1) rotate(-15deg); opacity: 0.3; }
          }
        `}
      </style>
    </Box>
  );
};

export default DPDPCompliance;