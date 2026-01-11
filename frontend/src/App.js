import React, { useState, useEffect } from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Container,
  Grid,
  Box,
  Card,
  CardContent,
  Chip,
  IconButton,
  ThemeProvider,
  createTheme,
  CssBaseline
} from '@mui/material';
import {
  Security as SecurityIcon,
  Dashboard as DashboardIcon,
  Map as MapIcon,
  Assessment as AssessmentIcon,
  Shield as ShieldIcon,
  AccountBalance as TokenIcon,
  Psychology as PsychologyIcon
} from '@mui/icons-material';

// Components
import HospitalMap from './components/HospitalMap';
import PrivacyAccuracyChart from './components/PrivacyAccuracyChart';
import ReputationLeaderboard from './components/ReputationLeaderboard';
import SecurityMonitor from './components/SecurityMonitor';
// AttackSimulator removed - using real Byzantine Shield instead
import HealthTokenDashboard from './components/HealthTokenDashboard';
import DPDPCompliance from './components/DPDPCompliance';
import RealTimeMetrics from './components/RealTimeMetrics';
import FedXAIDashboard from './components/fedxai/FedXAIDashboard';
import QuantumConsole from './components/QuantumConsole'; // New Console Component

// Custom theme for Ghost Protocol
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#00ff88',
    },
    secondary: {
      main: '#00ccff',
    },
    error: {
      main: '#ff4757',
    },
    warning: {
      main: '#ffa502',
    },
    background: {
      default: '#0a0a0a',
      paper: 'rgba(255, 255, 255, 0.05)',
    },
    text: {
      primary: '#ffffff',
      secondary: 'rgba(255, 255, 255, 0.7)',
    },
  },
  typography: {
    fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", "Oxygen", "Ubuntu", "Cantarell", "Fira Sans", "Droid Sans", "Helvetica Neue", sans-serif',
    h4: {
      fontWeight: 700,
      background: 'linear-gradient(45deg, #00ff88, #00ccff)',
      WebkitBackgroundClip: 'text',
      WebkitTextFillColor: 'transparent',
      backgroundClip: 'text',
    },
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          scrollbarColor: '#6b6b6b #2b2b2b',
          '&::-webkit-scrollbar, & *::-webkit-scrollbar': {
            backgroundColor: '#2b2b2b',
            width: '8px',
          },
          '&::-webkit-scrollbar-thumb, & *::-webkit-scrollbar-thumb': {
            borderRadius: 8,
            backgroundColor: '#6b6b6b',
            minHeight: 24,
          },
          '&::-webkit-scrollbar-thumb:focus, & *::-webkit-scrollbar-thumb:focus': {
            backgroundColor: '#959595',
          },
          '&::-webkit-scrollbar-thumb:active, & *::-webkit-scrollbar-thumb:active': {
            backgroundColor: '#959595',
          },
          '&::-webkit-scrollbar-thumb:hover, & *::-webkit-scrollbar-thumb:hover': {
            backgroundColor: '#959595',
          },
          '&::-webkit-scrollbar-corner, & *::-webkit-scrollbar-corner': {
            backgroundColor: '#2b2b2b',
          },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          background: 'rgba(255, 255, 255, 0.05)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          borderRadius: '12px',
          backdropFilter: 'blur(10px)',
          transition: 'all 0.3s ease',
          '&:hover': {
            background: 'rgba(255, 255, 255, 0.08)',
            borderColor: 'rgba(255, 255, 255, 0.2)',
            transform: 'translateY(-2px)',
          },
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: '8px',
          textTransform: 'none',
          fontWeight: 600,
          padding: '12px 24px',
          transition: 'all 0.3s ease',
          '&:hover': {
            transform: 'translateY(-2px)',
          },
        },
        containedPrimary: {
          background: 'linear-gradient(45deg, #00ff88, #00ccff)',
          color: '#000',
          '&:hover': {
            boxShadow: '0 8px 25px rgba(0, 255, 136, 0.3)',
          },
        },
        outlinedSecondary: {
          borderColor: 'rgba(255, 255, 255, 0.2)',
          color: '#fff',
          '&:hover': {
            background: 'rgba(255, 255, 255, 0.1)',
            borderColor: 'rgba(255, 255, 255, 0.3)',
          },
        },
      },
    },
  },
});

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [systemStatus, setSystemStatus] = useState({
    dpdp_compliant: true,
    current_round: 0,
    active_hospitals: 0,
    model_performance: 0.0,
    is_healthy: true
  });

  // Real-time attack state (set via WebSocket events from backend)
  const [attackState, setAttackState] = useState(null);
  const [logs, setLogs] = useState([
    { type: 'SYSTEM', message: 'Ghost Protocol v1.0 initialized', timestamp: new Date().toISOString() },
    { type: 'SYSTEM', message: 'Connected to SNA (Secure Network Aggregator)', timestamp: new Date().toISOString() },
    { type: 'QUANTUM', message: 'Real-time Byzantine Shield monitoring active', timestamp: new Date().toISOString() }
  ]);


  // API Configuration - Fix #10: Use environment variables instead of hardcoded URLs
  // Constants defined outside scope or memoized
  const API_URL = process.env.REACT_APP_SNA_URL || 'http://localhost:8000';
  const WS_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8000/ws';

  // Fetch system status periodically
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const response = await fetch(`${API_URL}/status`);
        const data = await response.json();
        setSystemStatus(data);
      } catch (error) {
        console.error('Failed to fetch status:', error);
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 5000); // Update every 5 seconds
    return () => clearInterval(interval);
  }, [API_URL]); // Added API_URL dependency

  // Real WebSocket connection for live training events
  useEffect(() => {
    let ws = null;
    let reconnectTimeout = null;

    const connectWebSocket = () => {
      ws = new WebSocket(WS_URL);

      ws.onopen = () => {
        console.log('WebSocket connected to SNA');
        setLogs(prev => [...prev, {
          type: 'SYSTEM',
          message: 'LIVE: Connected to Secure National Aggregator',
          timestamp: new Date().toISOString()
        }]);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          // Handle different event types from backend
          if (data.type === 'training_update') {
            setLogs(prev => [...prev, {
              type: 'QUANTUM',
              message: `Round ${data.round}: ε=${data.epsilon_spent?.toFixed(3) || 'N/A'} | AUC=${data.auc?.toFixed(3) || 'N/A'}`,
              timestamp: new Date().toISOString()
            }]);
          } else if (data.type === 'attack_detected') {
            setLogs(prev => [...prev, {
              type: 'THREAT',
              message: `BYZANTINE ATTACK from ${data.hospital_id}: ${data.attack_type}`,
              timestamp: new Date().toISOString()
            }]);
            setAttackState({ targetId: data.hospital_id, type: data.attack_type });
          } else if (data.type === 'attack_blocked') {
            setLogs(prev => [...prev, {
              type: 'SUCCESS',
              message: `BLOCKED: Node ${data.hospital_id} quarantined (reputation: ${data.reputation?.toFixed(2)})`,
              timestamp: new Date().toISOString()
            }]);
            setTimeout(() => setAttackState(null), 2000);
          } else if (data.type === 'aggregation_complete') {
            setLogs(prev => [...prev, {
              type: 'SUCCESS',
              message: `Round ${data.round} aggregated: ${data.accepted}/${data.total} updates accepted`,
              timestamp: new Date().toISOString()
            }]);
          } else if (data.type === 'hospital_connected') {
            setLogs(prev => [...prev, {
              type: 'SYSTEM',
              message: `Hospital ${data.hospital_id} joined network`,
              timestamp: new Date().toISOString()
            }]);
          } else {
            // Generic event - show in console
            setLogs(prev => [...prev, {
              type: data.level || 'SYSTEM',
              message: data.message || JSON.stringify(data),
              timestamp: new Date().toISOString()
            }]);
          }
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e);
        }
      };

      ws.onclose = () => {
        console.log('WebSocket disconnected, reconnecting in 3s...');
        reconnectTimeout = setTimeout(connectWebSocket, 3000);
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
      };
    };

    connectWebSocket();

    return () => {
      if (ws) ws.close();
      if (reconnectTimeout) clearTimeout(reconnectTimeout);
    };
  }, [WS_URL]); // Added WS_URL dependency

  const tabs = [
    { id: 'dashboard', label: 'Dashboard', icon: <DashboardIcon /> },
    { id: 'map', label: 'Hospital Map', icon: <MapIcon /> },
    { id: 'privacy', label: 'Privacy-Accuracy', icon: <AssessmentIcon /> },
    { id: 'reputation', label: 'Leaderboard', icon: <ShieldIcon /> },
    { id: 'tokens', label: 'HealthTokens', icon: <TokenIcon /> },
    { id: 'security', label: 'Security Monitor', icon: <SecurityIcon /> },

    { id: 'fedxai', label: 'FedXAI', icon: <PsychologyIcon /> },
  ];

  const renderContent = () => {
    switch (activeTab) {
      case 'dashboard':
        return <RealTimeMetrics systemStatus={systemStatus} />;
      case 'map':
        return <HospitalMap attackState={attackState} />;
      case 'privacy':
        return <PrivacyAccuracyChart />;
      case 'reputation':
        return <ReputationLeaderboard />;
      case 'tokens':
        return <HealthTokenDashboard />;
      case 'security':
        return <SecurityMonitor />;

      case 'fedxai':
        return <FedXAIDashboard />;
      default:
        return <RealTimeMetrics systemStatus={systemStatus} />;
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ flexGrow: 1 }}>
        {/* Header */}
        <AppBar
          position="static"
          sx={{
            background: 'rgba(255, 255, 255, 0.05)',
            backdropFilter: 'blur(10px)',
            borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
            boxShadow: 'none'
          }}
        >
          <Toolbar>
            <Typography variant="h4" component="div" sx={{ flexGrow: 1 }}>
              Ghost Protocol
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Chip
                icon={<SecurityIcon />}
                label={systemStatus.dpdp_compliant ? "DPDP Compliant" : "DPDP Violation"}
                color={systemStatus.dpdp_compliant ? "success" : "error"}
                variant="outlined"
              />
              <Chip
                label={`Round ${systemStatus.current_round}`}
                variant="outlined"
                sx={{ color: '#00ff88', borderColor: '#00ff88' }}
              />
            </Box>
          </Toolbar>
        </AppBar>

        {/* Navigation Tabs */}
        <Box
          sx={{
            background: 'rgba(255, 255, 255, 0.02)',
            borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
            p: 2
          }}
        >
          <Container maxWidth="xl">
            <Box sx={{ display: 'flex', gap: 1, overflowX: 'auto' }}>
              {tabs.map((tab) => (
                <IconButton
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  sx={{
                    color: activeTab === tab.id ? '#00ff88' : 'rgba(255, 255, 255, 0.7)',
                    background: activeTab === tab.id ? 'rgba(0, 255, 136, 0.1)' : 'transparent',
                    border: activeTab === tab.id ? '1px solid rgba(0, 255, 136, 0.3)' : 'none',
                    borderRadius: '8px',
                    p: 2,
                    minWidth: '120px',
                    '&:hover': {
                      background: activeTab === tab.id ? 'rgba(0, 255, 136, 0.2)' : 'rgba(255, 255, 255, 0.05)',
                    }
                  }}
                >
                  <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 1 }}>
                    {tab.icon}
                    <Typography variant="caption">{tab.label}</Typography>
                  </Box>
                </IconButton>
              ))}
            </Box>
          </Container>
        </Box>

        {/* Main Content */}
        <Container maxWidth="xl" sx={{ p: 3 }}>
          <Box sx={{ mb: 3 }}>
            <DPDPCompliance systemStatus={systemStatus} />
          </Box>

          {renderContent()}

          {/* Quantum Console Footer */}
          <Box sx={{ mt: 3 }}>
            <QuantumConsole logs={logs} />
          </Box>
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default App;