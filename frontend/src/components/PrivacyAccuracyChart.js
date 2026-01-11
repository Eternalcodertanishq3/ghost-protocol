import React, { useState, useEffect } from 'react';
import { Card, CardContent, Typography, Box, Slider, FormControl, InputLabel, Select, MenuItem, Grid, Chip } from '@mui/material';
import { Assessment as AssessmentIcon } from '@mui/icons-material';
import Plot from 'react-plotly.js';

const PrivacyAccuracyChart = () => {
  const [epsilon, setEpsilon] = useState(1.23);
  const [chartData, setChartData] = useState([]);
  const [selectedSeries, setSelectedSeries] = useState(['centralized', 'fl_only', 'fl_dp']);

  // Generate privacy-accuracy tradeoff data
  const generateTradeoffData = () => {
    const epsilon_values = [];
    const centralized_accuracy = [];
    const fl_only_accuracy = [];
    const fl_dp_accuracy = [];

    for (let eps = 0.1; eps <= 10; eps += 0.2) {
      epsilon_values.push(eps);

      // Simulated data - in reality this would come from experiments
      // Centralized (no privacy, highest accuracy)
      centralized_accuracy.push(0.95 - Math.exp(-eps * 0.5) * 0.05);

      // FL only (some accuracy loss due to federated learning)
      fl_only_accuracy.push(0.92 - Math.exp(-eps * 0.3) * 0.08);

      // FL + DP (privacy-preserving, lower accuracy)
      fl_dp_accuracy.push(0.85 - Math.exp(-eps * 0.2) * 0.15);
    }

    return {
      epsilon_values,
      centralized_accuracy,
      fl_only_accuracy,
      fl_dp_accuracy
    };
  };

  useEffect(() => {
    const data = generateTradeoffData();

    const traces = [
      {
        x: data.epsilon_values,
        y: data.centralized_accuracy,
        type: 'scatter',
        mode: 'lines',
        name: 'Centralized (No Privacy)',
        line: {
          color: '#ff4757',
          width: 3
        },
        visible: selectedSeries.includes('centralized') ? true : 'legendonly'
      },
      {
        x: data.epsilon_values,
        y: data.fl_only_accuracy,
        type: 'scatter',
        mode: 'lines',
        name: 'Federated Learning Only',
        line: {
          color: '#ffa502',
          width: 3
        },
        visible: selectedSeries.includes('fl_only') ? true : 'legendonly'
      },
      {
        x: data.epsilon_values,
        y: data.fl_dp_accuracy,
        type: 'scatter',
        mode: 'lines',
        name: 'FL + Differential Privacy',
        line: {
          color: '#00ff88',
          width: 3
        },
        visible: selectedSeries.includes('fl_dp') ? true : 'legendonly'
      }
    ];

    // Add current epsilon marker
    const currentData = {
      x: [epsilon],
      y: [0.85 - Math.exp(-epsilon * 0.2) * 0.15], // FL + DP accuracy at current epsilon
      type: 'scatter',
      mode: 'markers',
      name: 'Current Setting',
      marker: {
        color: '#00ccff',
        size: 12,
        symbol: 'diamond',
        line: {
          color: '#ffffff',
          width: 2
        }
      }
    };

    setChartData([...traces, currentData]);
  }, [epsilon, selectedSeries]);

  const handleSeriesToggle = (series) => {
    setSelectedSeries(prev =>
      prev.includes(series)
        ? prev.filter(s => s !== series)
        : [...prev, series]
    );
  };

  const layout = {
    title: {
      text: 'Privacy-Accuracy Tradeoff Analysis',
      font: { color: '#ffffff', size: 18, family: 'Inter' }
    },
    xaxis: {
      title: 'Privacy Budget (ε)',
      color: '#ffffff',
      gridcolor: 'rgba(255, 255, 255, 0.1)',
      zerolinecolor: 'rgba(255, 255, 255, 0.2)',
      range: [0, 10]
    },
    yaxis: {
      title: 'Model Accuracy (AUC)',
      color: '#ffffff',
      gridcolor: 'rgba(255, 255, 255, 0.1)',
      zerolinecolor: 'rgba(255, 255, 255, 0.2)',
      range: [0.6, 1.0]
    },
    plot_bgcolor: 'transparent',
    paper_bgcolor: 'transparent',
    font: { color: '#ffffff', family: 'Inter' },
    legend: {
      x: 0.02,
      y: 0.98,
      bgcolor: 'rgba(0, 0, 0, 0.8)',
      bordercolor: 'rgba(255, 255, 255, 0.2)',
      borderwidth: 1
    },
    shapes: [
      // DPDP compliance threshold line
      {
        type: 'line',
        x0: 9.5,
        y0: 0.6,
        x1: 9.5,
        y1: 1.0,
        line: {
          color: '#ff4757',
          width: 2,
          dash: 'dash'
        }
      }
    ],
    annotations: [
      {
        x: 9.5,
        y: 0.95,
        text: 'DPDP Limit (ε=9.5)',
        showarrow: true,
        arrowhead: 2,
        arrowcolor: '#ff4757',
        font: { color: '#ff4757', size: 12 }
      }
    ]
  };

  const config = {
    responsive: true,
    displayModeBar: false
  };

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
          <Typography variant="h5" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <AssessmentIcon />
            Privacy-Accuracy Tradeoff
          </Typography>

          <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
            <FormControl variant="outlined" size="small" sx={{ minWidth: 120 }}>
              <InputLabel>Current ε</InputLabel>
              <Select
                value={epsilon}
                onChange={(e) => setEpsilon(e.target.value)}
                label="Current ε"
                sx={{ color: '#fff' }}
              >
                {[0.5, 1.0, 1.23, 2.0, 3.0, 5.0, 8.0].map(val => (
                  <MenuItem key={val} value={val}>{val}</MenuItem>
                ))}
              </Select>
            </FormControl>
          </Box>
        </Box>

        <Box sx={{ mb: 3 }}>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Interactive Privacy Budget Slider
          </Typography>
          <Slider
            value={epsilon}
            onChange={(e, value) => setEpsilon(value)}
            min={0.1}
            max={10}
            step={0.1}
            valueLabelDisplay="auto"
            sx={{
              color: '#00ff88',
              '& .MuiSlider-thumb': {
                backgroundColor: '#00ff88',
              },
              '& .MuiSlider-track': {
                backgroundColor: '#00ff88',
              },
              '& .MuiSlider-rail': {
                backgroundColor: 'rgba(255, 255, 255, 0.2)',
              }
            }}
          />
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
            <Typography variant="caption" color="text.secondary">ε = 0.1 (High Privacy)</Typography>
            <Typography variant="caption" color="text.secondary">ε = {epsilon.toFixed(2)}</Typography>
            <Typography variant="caption" color="text.secondary">ε = 10 (Low Privacy)</Typography>
          </Box>
        </Box>

        <Box sx={{ height: '400px' }}>
          <Plot
            data={chartData}
            layout={layout}
            config={config}
            style={{ width: '100%', height: '100%' }}
          />
        </Box>

        <Box sx={{ mt: 3, p: 2, background: 'rgba(0, 255, 136, 0.1)', borderRadius: '8px' }}>
          <Typography variant="h6" sx={{ color: '#00ff88', mb: 1 }}>
            Current Configuration Analysis
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={4}>
              <Typography variant="body2" color="text.secondary">Privacy Level</Typography>
              <Typography variant="h6">
                {epsilon <= 2 ? 'High' : epsilon <= 5 ? 'Medium' : 'Low'}
              </Typography>
            </Grid>
            <Grid item xs={4}>
              <Typography variant="body2" color="text.secondary">Expected Accuracy</Typography>
              <Typography variant="h6">
                {(0.85 - Math.exp(-epsilon * 0.2) * 0.15).toFixed(3)}
              </Typography>
            </Grid>
            <Grid item xs={4}>
              <Typography variant="body2" color="text.secondary">DPDP Compliance</Typography>
              <Typography variant="h6" sx={{ color: epsilon <= 9.5 ? '#00ff88' : '#ff4757' }}>
                {epsilon <= 9.5 ? '✓ Compliant' : '✗ Violation'}
              </Typography>
            </Grid>
          </Grid>
        </Box>

        <Box sx={{ mt: 2 }}>
          <Typography variant="subtitle2" color="text.secondary" gutterBottom>
            Series Visibility
          </Typography>
          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
            {[
              { key: 'centralized', label: 'Centralized', color: '#ff4757' },
              { key: 'fl_only', label: 'FL Only', color: '#ffa502' },
              { key: 'fl_dp', label: 'FL + DP', color: '#00ff88' }
            ].map(series => (
              <Chip
                key={series.key}
                label={series.label}
                onClick={() => handleSeriesToggle(series.key)}
                sx={{
                  backgroundColor: selectedSeries.includes(series.key) ? series.color : 'rgba(255, 255, 255, 0.1)',
                  color: selectedSeries.includes(series.key) ? '#000' : '#fff',
                  border: selectedSeries.includes(series.key) ? 'none' : '1px solid rgba(255, 255, 255, 0.2)',
                  '&:hover': {
                    backgroundColor: selectedSeries.includes(series.key) ? series.color : 'rgba(255, 255, 255, 0.2)'
                  }
                }}
              />
            ))}
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};

export default PrivacyAccuracyChart;