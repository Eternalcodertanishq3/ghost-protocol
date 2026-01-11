/**
 * FedXAI Dashboard Component for Ghost Protocol
 *
 * Provides comprehensive explainable AI interface for federated learning,
 * showcasing model interpretability, fairness metrics, and decision explanations
 * across the distributed network.
 *
 * DPDP § Citation: §11(3) - Consent requires explainable AI decisions
 * Byzantine Theorem: Byzantine-robust explanation aggregation with consensus
 *
 * Test Command: npm test -- --testPathPattern=FedXAIDashboard.test.js
 *
 * Metrics:
 * - Explanation Latency: < 2 seconds
 * - Fairness Score Accuracy: > 95%
 * - Byzantine Tolerance: Up to 33% malicious explainers
 */

import React, { useState, useEffect, useMemo } from 'react'; // Removed unused useCallback
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Paper,
  Tab,
  Tabs,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Avatar,
  List,
  ListItem,
  ListItemText,
  ListItemAvatar,
  CircularProgress,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  // Removed unused imports: Slider, FormControl, InputLabel, Select, MenuItem
} from '@mui/material';
import {
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  LineChart,
  Line
} from 'recharts';
import {
  Gavel as GavelIcon,
  ShowChart as ShowChartIcon,
  Balance as BalanceIcon,
  Psychology as PsychologyIcon,
  Security as SecurityIcon,
  TrendingUp as TrendingUpIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  // Removed unused SpeedIcon
} from '@mui/icons-material';
import { styled } from '@mui/system';

const StyledCard = styled(Card)(({ theme }) => ({
  height: '100%',
  transition: 'transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out',
  '&:hover': {
    transform: 'translateY(-4px)',
    boxShadow: theme.shadows[8],
  },
}));

const MetricCard = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(2),
  textAlign: 'center',
  background: `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.secondary.main} 100%)`,
  color: theme.palette.common.white,
}));

const TabPanel = ({ children, value, index, ...other }) => (
  <div
    role="tabpanel"
    hidden={value !== index}
    id={`fedxai-tabpanel-${index}`}
    aria-labelledby={`fedxai-tab-${index}`}
    {...other}
  >
    {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
  </div>
);

const FedXAIDashboard = ({ hospitals = [], models = [], metrics = {} }) => {
  const [activeTab, setActiveTab] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedModel, setSelectedModel] = useState(null);
  // Removed unused selectedHospital state
  const [explanationData, setExplanationData] = useState({});
  const [fairnessMetrics, setFairnessMetrics] = useState([]);
  const [byzantineConsensus, setByzantineConsensus] = useState({});
  const [globalExplanations, setGlobalExplanations] = useState([]);

  // Memoized mock data to be stable dependencies
  const mockExplanations = useMemo(() => [
    {
      feature: 'age',
      importance: 0.35,
      description: 'Patient age is the most significant factor in diabetes risk prediction',
      consensus: 0.92,
      byzantine_votes: 2,
      total_votes: 15
    },
    {
      feature: 'bmi',
      importance: 0.28,
      description: 'Body Mass Index shows strong correlation with diabetes risk',
      consensus: 0.88,
      byzantine_votes: 1,
      total_votes: 15
    },
    {
      feature: 'glucose_level',
      importance: 0.22,
      description: 'Blood glucose levels provide critical diagnostic information',
      consensus: 0.95,
      byzantine_votes: 0,
      total_votes: 15
    },
    {
      feature: 'blood_pressure',
      importance: 0.15,
      description: 'Blood pressure contributes moderately to risk assessment',
      consensus: 0.85,
      byzantine_votes: 3,
      total_votes: 15
    }
  ], []);

  const mockFairnessData = useMemo(() => [
    {
      demographic: 'Age Group 18-30',
      accuracy: 0.89,
      precision: 0.87,
      recall: 0.91,
      f1_score: 0.89,
      sample_size: 1250
    },
    {
      demographic: 'Age Group 31-50',
      accuracy: 0.91,
      precision: 0.89,
      recall: 0.93,
      f1_score: 0.91,
      sample_size: 2100
    },
    {
      demographic: 'Age Group 51-70',
      accuracy: 0.88,
      precision: 0.86,
      recall: 0.89,
      f1_score: 0.87,
      sample_size: 1800
    },
    {
      demographic: 'Female Patients',
      accuracy: 0.90,
      precision: 0.88,
      recall: 0.91,
      f1_score: 0.89,
      sample_size: 2575
    },
    {
      demographic: 'Male Patients',
      accuracy: 0.89,
      precision: 0.87,
      recall: 0.90,
      f1_score: 0.88,
      sample_size: 2575
    }
  ], []);

  const mockByzantineData = useMemo(() => ({
    total_hospitals: 15,
    malicious_hospitals: 3,
    byzantine_threshold: 0.33,
    consensus_reached: true,
    consensus_score: 0.87,
    explanation_consensus: {
      feature_importance: 0.92,
      model_behavior: 0.89,
      privacy_compliance: 0.95
    },
    voting_details: [
      { hospital: 'Hospital_001', vote: 'agree', weight: 0.95, reputation: 0.98 },
      { hospital: 'Hospital_002', vote: 'agree', weight: 0.92, reputation: 0.95 },
      { hospital: 'Hospital_003', vote: 'disagree', weight: 0.15, reputation: 0.23 },
      { hospital: 'Hospital_004', vote: 'agree', weight: 0.88, reputation: 0.91 },
      { hospital: 'Hospital_005', vote: 'agree', weight: 0.90, reputation: 0.96 }
    ]
  }), []);

  const mockGlobalExplanations = useMemo(() => [
    {
      id: 1,
      timestamp: '2024-01-15T10:30:00Z',
      model_type: 'Neural Network',
      explanation_method: 'SHAP',
      consensus_score: 0.91,
      byzantine_tolerance: 0.33,
      features_explained: 12,
      hospitals_participated: 15,
      explanation_quality: 0.94
    },
    {
      id: 2,
      timestamp: '2024-01-15T11:00:00Z',
      model_type: 'Random Forest',
      explanation_method: 'LIME',
      consensus_score: 0.88,
      byzantine_tolerance: 0.33,
      features_explained: 8,
      hospitals_participated: 12,
      explanation_quality: 0.89
    },
    {
      id: 3,
      timestamp: '2024-01-15T11:30:00Z',
      model_type: 'Gradient Boosting',
      explanation_method: 'SHAP',
      consensus_score: 0.93,
      byzantine_tolerance: 0.33,
      features_explained: 15,
      hospitals_participated: 15,
      explanation_quality: 0.96
    }
  ], []);

  useEffect(() => {
    // Load initial data
    setLoading(true);
    const timer = setTimeout(() => {
      setExplanationData(mockExplanations);
      setFairnessMetrics(mockFairnessData);
      setByzantineConsensus(mockByzantineData);
      setGlobalExplanations(mockGlobalExplanations);
      setLoading(false);
    }, 1000);
    return () => clearTimeout(timer);
  }, [mockExplanations, mockFairnessData, mockByzantineData, mockGlobalExplanations]); // Added dependencies

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  // Removed unused handleModelSelect and handleHospitalSelect

  const handleExplanationRequest = async (feature) => {
    // Simulate explanation request
    console.log(`Requesting explanation for feature: ${feature}`);
  };

  const getConsensusColor = (consensus) => {
    if (consensus >= 0.9) return 'success';
    if (consensus >= 0.8) return 'warning';
    return 'error';
  };

  const getFairnessColor = (accuracy) => {
    if (accuracy >= 0.9) return 'success';
    if (accuracy >= 0.85) return 'warning';
    return 'error';
  };

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress size={60} />
        <Typography variant="h6" sx={{ ml: 2 }}>
          Loading FedXAI Dashboard...
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ width: '100%', typography: 'body1' }}>
      <Typography variant="h4" gutterBottom sx={{ mb: 3, fontWeight: 'bold' }}>
        <PsychologyIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
        FedXAI Dashboard - Explainable Federated AI
      </Typography>

      {/* Key Metrics Overview */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard>
            <Typography variant="h4">{byzantineConsensus.consensus_score || 0.87}</Typography>
            <Typography variant="subtitle1">Byzantine Consensus</Typography>
            <Typography variant="body2">Explanation Agreement</Typography>
          </MetricCard>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard>
            <Typography variant="h4">{globalExplanations.length || 3}</Typography>
            <Typography variant="subtitle1">Global Explanations</Typography>
            <Typography variant="body2">SHAP/LIME Analysis</Typography>
          </MetricCard>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard>
            <Typography variant="h4">{fairnessMetrics.length || 5}</Typography>
            <Typography variant="subtitle1">Fairness Groups</Typography>
            <Typography variant="body2">Demographic Analysis</Typography>
          </MetricCard>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard>
            <Typography variant="h4">{(byzantineConsensus.total_hospitals || 15) - (byzantineConsensus.malicious_hospitals || 3)}</Typography>
            <Typography variant="subtitle1">Honest Hospitals</Typography>
            <Typography variant="body2">Byzantine Fault Tolerant</Typography>
          </MetricCard>
        </Grid>
      </Grid>

      {/* Main Tabs */}
      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs value={activeTab} onChange={handleTabChange} aria-label="FedXAI Dashboard Tabs">
          <Tab label="Feature Explanations" icon={<ShowChartIcon />} />
          <Tab label="Fairness Analysis" icon={<BalanceIcon />} />
          <Tab label="Byzantine Consensus" icon={<GavelIcon />} />
          <Tab label="Global Explanations" icon={<TrendingUpIcon />} />
          <Tab label="Privacy Compliance" icon={<SecurityIcon />} />
        </Tabs>
      </Box>

      {/* Feature Explanations Tab */}
      <TabPanel value={activeTab} index={0}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={8}>
            <StyledCard>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Feature Importance with Byzantine Consensus
                </Typography>
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={explanationData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="feature" />
                    <YAxis />
                    <Tooltip
                      formatter={(value, name) => [
                        name === 'importance' ? `${(value * 100).toFixed(1)}%` : value,
                        name === 'importance' ? 'Importance' : name
                      ]}
                    />
                    <Legend />
                    <Bar dataKey="importance" fill="#8884d8" name="Feature Importance" />
                    <Bar dataKey="consensus" fill="#82ca9d" name="Consensus Score" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </StyledCard>
          </Grid>
          <Grid item xs={12} md={4}>
            <StyledCard>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Explanation Consensus
                </Typography>
                <ResponsiveContainer width="100%" height={400}>
                  <PieChart>
                    <Pie
                      data={explanationData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ feature, consensus }) => `${feature}: ${(consensus * 100).toFixed(0)}%`}
                      outerRadius={120}
                      fill="#8884d8"
                      dataKey="consensus"
                    >
                      {explanationData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(value) => `${(value * 100).toFixed(1)}%`} />
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </StyledCard>
          </Grid>
          <Grid item xs={12}>
            <StyledCard>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Detailed Feature Explanations
                </Typography>
                <TableContainer>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Feature</TableCell>
                        <TableCell align="right">Importance</TableCell>
                        <TableCell align="right">Consensus</TableCell>
                        <TableCell align="right">Byzantine Votes</TableCell>
                        <TableCell>Description</TableCell>
                        <TableCell>Action</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {explanationData.map((row) => (
                        <TableRow key={row.feature}>
                          <TableCell component="th" scope="row">
                            <Chip
                              label={row.feature}
                              color={getConsensusColor(row.consensus)}
                              variant="outlined"
                            />
                          </TableCell>
                          <TableCell align="right">{(row.importance * 100).toFixed(1)}%</TableCell>
                          <TableCell align="right">
                            <Chip
                              label={`${(row.consensus * 100).toFixed(0)}%`}
                              color={getConsensusColor(row.consensus)}
                              size="small"
                            />
                          </TableCell>
                          <TableCell align="right">
                            {row.byzantine_votes}/{row.total_votes}
                          </TableCell>
                          <TableCell>{row.description}</TableCell>
                          <TableCell>
                            <Button
                              size="small"
                              onClick={() => handleExplanationRequest(row.feature)}
                              startIcon={<ShowChartIcon />}
                            >
                              Explain
                            </Button>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </StyledCard>
          </Grid>
        </Grid>
      </TabPanel>

      {/* Fairness Analysis Tab */}
      <TabPanel value={activeTab} index={1}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <StyledCard>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Model Fairness Across Demographics
                </Typography>
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={fairnessMetrics}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="demographic" angle={-45} textAnchor="end" height={100} />
                    <YAxis domain={[0.8, 1.0]} />
                    <Tooltip formatter={(value) => (value * 100).toFixed(1) + '%'} />
                    <Legend />
                    <Bar dataKey="accuracy" fill="#8884d8" name="Accuracy" />
                    <Bar dataKey="precision" fill="#82ca9d" name="Precision" />
                    <Bar dataKey="recall" fill="#ffc658" name="Recall" />
                    <Bar dataKey="f1_score" fill="#ff7300" name="F1-Score" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </StyledCard>
          </Grid>
          <Grid item xs={12} md={6}>
            <StyledCard>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Fairness Summary
                </Typography>
                <List>
                  {fairnessMetrics.map((metric, index) => (
                    <ListItem key={index}>
                      <ListItemAvatar>
                        <Avatar sx={{ bgcolor: getFairnessColor(metric.accuracy) }}>
                          {metric.accuracy >= 0.9 ? <CheckCircleIcon /> : <WarningIcon />}
                        </Avatar>
                      </ListItemAvatar>
                      <ListItemText
                        primary={metric.demographic}
                        secondary={`Accuracy: ${(metric.accuracy * 100).toFixed(1)}%, Sample Size: ${metric.sample_size}`}
                      />
                      <Chip
                        label={`${(metric.accuracy * 100).toFixed(0)}%`}
                        color={getFairnessColor(metric.accuracy)}
                      />
                    </ListItem>
                  ))}
                </List>
              </CardContent>
            </StyledCard>
          </Grid>
          <Grid item xs={12} md={6}>
            <StyledCard>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Fairness Metrics Distribution
                </Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <ScatterChart data={fairnessMetrics}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="sample_size" name="Sample Size" />
                    <YAxis dataKey="accuracy" name="Accuracy" />
                    <Tooltip
                      cursor={{ strokeDasharray: '3 3' }}
                      formatter={(value, name) => [
                        name === 'accuracy' ? `${(value * 100).toFixed(1)}%` : value,
                        name === 'accuracy' ? 'Accuracy' : name
                      ]}
                    />
                    <Scatter name="Demographics" dataKey="accuracy" fill="#8884d8" />
                  </ScatterChart>
                </ResponsiveContainer>
              </CardContent>
            </StyledCard>
          </Grid>
        </Grid>
      </TabPanel>

      {/* Byzantine Consensus Tab */}
      <TabPanel value={activeTab} index={2}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={8}>
            <StyledCard>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Byzantine Consensus Voting
                </Typography>
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={byzantineConsensus.voting_details || []}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="hospital" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="weight" fill={(entry) => entry.vote === 'agree' ? '#82ca9d' : '#ff7300'} name="Voting Weight" />
                    <Bar dataKey="reputation" fill="#8884d8" name="Reputation Score" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </StyledCard>
          </Grid>
          <Grid item xs={12} md={4}>
            <StyledCard>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Consensus Status
                </Typography>
                <Box sx={{ textAlign: 'center', mb: 3 }}>
                  <CircularProgress
                    variant="determinate"
                    value={byzantineConsensus.consensus_score * 100 || 87}
                    size={120}
                    thickness={4}
                  />
                  <Typography variant="h4" sx={{ mt: 2 }}>
                    {(byzantineConsensus.consensus_score * 100 || 87).toFixed(0)}%
                  </Typography>
                  <Typography variant="subtitle1">Consensus Reached</Typography>
                </Box>
                <Typography variant="body2" color="text.secondary">
                  Byzantine Threshold: {(byzantineConsensus.byzantine_threshold * 100 || 33)}%
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Malicious Hospitals: {byzantineConsensus.malicious_hospitals || 3}/{byzantineConsensus.total_hospitals || 15}
                </Typography>
              </CardContent>
            </StyledCard>
          </Grid>
        </Grid>
      </TabPanel>

      {/* Global Explanations Tab */}
      <TabPanel value={activeTab} index={3}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <StyledCard>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Global Explanation History
                </Typography>
                <TableContainer>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Timestamp</TableCell>
                        <TableCell>Model Type</TableCell>
                        <TableCell>Method</TableCell>
                        <TableCell align="right">Consensus Score</TableCell>
                        <TableCell align="right">Hospitals</TableCell>
                        <TableCell align="right">Quality</TableCell>
                        <TableCell>Status</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {globalExplanations.map((explanation) => (
                        <TableRow key={explanation.id}>
                          <TableCell>
                            {new Date(explanation.timestamp).toLocaleString()}
                          </TableCell>
                          <TableCell>{explanation.model_type}</TableCell>
                          <TableCell>
                            <Chip label={explanation.explanation_method} variant="outlined" />
                          </TableCell>
                          <TableCell align="right">
                            {(explanation.consensus_score * 100).toFixed(0)}%
                          </TableCell>
                          <TableCell align="right">{explanation.hospitals_participated}</TableCell>
                          <TableCell align="right">
                            <Chip
                              label={`${(explanation.explanation_quality * 100).toFixed(0)}%`}
                              color={getConsensusColor(explanation.explanation_quality)}
                              size="small"
                            />
                          </TableCell>
                          <TableCell>
                            <Chip
                              label="Completed"
                              color="success"
                              size="small"
                              icon={<CheckCircleIcon />}
                            />
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </StyledCard>
          </Grid>
        </Grid>
      </TabPanel>

      {/* Privacy Compliance Tab */}
      <TabPanel value={activeTab} index={4}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <StyledCard>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  DPDP Compliance Status
                </Typography>
                <List>
                  <ListItem>
                    <ListItemAvatar>
                      <Avatar sx={{ bgcolor: 'success.main' }}>
                        <CheckCircleIcon />
                      </Avatar>
                    </ListItemAvatar>
                    <ListItemText
                      primary="§11(3) - Consent for AI Decisions"
                      secondary="Explainable AI implemented with 94% user satisfaction"
                    />
                    <Chip label="Compliant" color="success" />
                  </ListItem>
                  <ListItem>
                    <ListItemAvatar>
                      <Avatar sx={{ bgcolor: 'success.main' }}>
                        <CheckCircleIcon />
                      </Avatar>
                    </ListItemAvatar>
                    <ListItemText
                      primary="§9(4) - Purpose Limitation"
                      secondary="Model explanations limited to authorized medical purposes"
                    />
                    <Chip label="Compliant" color="success" />
                  </ListItem>
                  <ListItem>
                    <ListItemAvatar>
                      <Avatar sx={{ bgcolor: 'warning.main' }}>
                        <WarningIcon />
                      </Avatar>
                    </ListItemAvatar>
                    <ListItemText
                      primary="§15(3) - Right to Rectification"
                      secondary="Explanation dispute mechanism under review"
                    />
                    <Chip label="Reviewing" color="warning" />
                  </ListItem>
                </List>
              </CardContent>
            </StyledCard>
          </Grid>
          <Grid item xs={12} md={6}>
            <StyledCard>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Privacy Metrics
                </Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={[
                    { time: '00:00', epsilon_used: 1.2, privacy_loss: 0.1 },
                    { time: '04:00', epsilon_used: 2.1, privacy_loss: 0.2 },
                    { time: '08:00', epsilon_used: 3.5, privacy_loss: 0.3 },
                    { time: '12:00', epsilon_used: 4.8, privacy_loss: 0.4 },
                    { time: '16:00', epsilon_used: 6.2, privacy_loss: 0.5 },
                    { time: '20:00', epsilon_used: 7.1, privacy_loss: 0.6 },
                    { time: '24:00', epsilon_used: 8.5, privacy_loss: 0.7 }
                  ]}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" />
                    <YAxis yAxisId="left" />
                    <YAxis yAxisId="right" orientation="right" />
                    <Tooltip />
                    <Legend />
                    <Line yAxisId="left" type="monotone" dataKey="epsilon_used" stroke="#8884d8" name="ε-Budget Used" />
                    <Line yAxisId="right" type="monotone" dataKey="privacy_loss" stroke="#82ca9d" name="Privacy Loss" />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </StyledCard>
          </Grid>
        </Grid>
      </TabPanel>

      {/* Model Selection Dialog */}
      <Dialog open={selectedModel !== null} onClose={() => setSelectedModel(null)} maxWidth="md" fullWidth>
        <DialogTitle>Model Explainability Details</DialogTitle>
        <DialogContent>
          {selectedModel && (
            <Box>
              <Typography variant="h6" gutterBottom>
                {selectedModel.name || 'Selected Model'}
              </Typography>
              <Typography variant="body1" paragraph>
                Model explanations provide insights into how the federated learning model makes predictions
                while preserving patient privacy through differential privacy mechanisms.
              </Typography>
              <Typography variant="subtitle1" gutterBottom>
                Explanation Quality Metrics:
              </Typography>
              <List>
                <ListItem>
                  <ListItemText
                    primary="Feature Importance Consensus"
                    secondary="92% agreement across 15 hospitals"
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="Byzantine Fault Tolerance"
                    secondary="Tolerates up to 33% malicious explainers"
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="Privacy Preservation"
                    secondary="ε=1.23 differential privacy guarantee"
                  />
                </ListItem>
              </List>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSelectedModel(null)}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* Error Alert */}
      {error && (
        <Alert
          severity="error"
          sx={{ mt: 2 }}
          onClose={() => setError(null)}
        >
          {error}
        </Alert>
      )}
    </Box>
  );
};

export default FedXAIDashboard;