import React, { useEffect, useRef, useState } from 'react';
import { Card, CardContent, Typography, Box, Chip, Grid } from '@mui/material';
import { Map as MapIcon, WifiTethering, Shield } from '@mui/icons-material';

// Hospital locations (simulated across India)
const hospitals = [
  { id: 'H001', name: 'AIIMS Delhi', lat: 28.6139, lng: 77.2090, status: 'online', reputation: 0.95, tokens: 1250 },
  { id: 'H002', name: 'CMC Vellore', lat: 12.9716, lng: 79.0747, status: 'online', reputation: 0.92, tokens: 1180 },
  { id: 'H003', name: 'Tata Memorial Mumbai', lat: 19.0760, lng: 72.8777, status: 'training', reputation: 0.88, tokens: 1090 },
  { id: 'H004', name: 'Apollo Chennai', lat: 13.0827, lng: 80.2707, status: 'online', reputation: 0.85, tokens: 1050 },
  { id: 'H005', name: 'Fortis Bangalore', lat: 12.9352, lng: 77.6245, status: 'offline', reputation: 0.78, tokens: 980 },
  { id: 'H006', name: 'Kolkata Medical College', lat: 22.5726, lng: 88.3639, status: 'online', reputation: 0.82, tokens: 1020 },
  { id: 'H007', name: 'PGIMER Chandigarh', lat: 30.7333, lng: 76.7794, status: 'training', reputation: 0.90, tokens: 1150 },
  { id: 'H008', name: 'SCTIMST Trivandrum', lat: 8.5241, lng: 76.9366, status: 'online', reputation: 0.87, tokens: 1080 },
];

const HospitalMap = ({ attackState = null }) => {
  const mapRef = useRef(null);
  const mapInstance = useRef(null);
  const [selectedHospital, setSelectedHospital] = useState(null);
  const [connectionLines, setConnectionLines] = useState([]);

  useEffect(() => {
    // Initialize map
    if (!mapInstance.current && mapRef.current) {
      mapInstance.current = window.L.map(mapRef.current, {
        center: [20.5937, 78.9629], // Center of India
        zoom: 5,
        zoomControl: true,
        attributionControl: false,
      });

      // Add dark tile layer
      window.L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '',
        subdomains: 'abcd',
        maxZoom: 19,
      }).addTo(mapInstance.current);

      // Add hospital markers
      hospitals.forEach(hospital => {
        // Check if hospital is under attack
        const isUnderAttack = attackState?.active && attackState.targetId === hospital.id;

        let markerColor;
        if (isUnderAttack) {
          markerColor = '#ff0000'; // Red for attack
        } else {
          markerColor = hospital.status === 'online' ? '#00ff88' :
            hospital.status === 'training' ? '#ffa502' : '#ff4757';
        }

        let pulseClass = '';
        if (isUnderAttack) {
          pulseClass = 'attack-pulse';
        } else if (hospital.status === 'training') {
          pulseClass = 'pulse';
        }

        const marker = window.L.circleMarker([hospital.lat, hospital.lng], {
          color: markerColor,
          fillColor: markerColor,
          fillOpacity: 0.8,
          radius: isUnderAttack ? 15 : (hospital.status === 'training' ? 12 : 8),
          className: pulseClass,
        }).addTo(mapInstance.current);

        // Add popup
        const popupContent = `
          <div style="color: #000; padding: 10px; min-width: 200px;">
            <h3 style="margin: 0 0 10px 0; color: #00ff88;">${hospital.name}</h3>
            <p><strong>ID:</strong> ${hospital.id}</p>
            <p><strong>Status:</strong> 
              <span style="color: ${markerColor}; font-weight: bold;">
                ${hospital.status.toUpperCase()}
              </span>
            </p>
            <p><strong>Reputation:</strong> ${(hospital.reputation * 100).toFixed(1)}%</p>
            <p><strong>HealthTokens:</strong> ${hospital.tokens}</p>
          </div>
        `;

        marker.bindPopup(popupContent);

        // Add click handler
        marker.on('click', () => {
          setSelectedHospital(hospital);
        });
      });

      // Add animated connection lines to SNA (Delhi area)
      const snaLocation = [28.6139, 77.2090]; // SNA location

      hospitals.forEach((hospital, index) => {
        if (hospital.status === 'online') {
          setTimeout(() => {
            const line = window.L.polyline([
              [hospital.lat, hospital.lng],
              snaLocation
            ], {
              color: '#00ccff',
              weight: 2,
              opacity: 0.6,
              dashArray: '5, 10'
            }).addTo(mapInstance.current);

            // Animate the line
            let offset = 0;
            const animateLine = () => {
              offset += 2;
              line.setStyle({
                dashOffset: offset
              });
              if (offset < 100) {
                setTimeout(animateLine, 50);
              }
            };
            animateLine();
          }, index * 200);
        }
      });
    }

    return () => {
      if (mapInstance.current) {
        mapInstance.current.remove();
        mapInstance.current = null;
      }
    };
  }, [attackState]); // Re-render when attack state changes

  const getStatusColor = (status) => {
    switch (status) {
      case 'online': return '#00ff88';
      case 'training': return '#ffa502';
      case 'offline': return '#ff4757';
      default: return '#666';
    }
  };

  return (
    <Card sx={{ height: '600px' }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Typography variant="h5" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <MapIcon />
            Hospital Network Map
          </Typography>
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Chip
              icon={<WifiTethering />}
              label="SNA Active"
              color="success"
              variant="outlined"
            />
            <Chip
              icon={<Shield />}
              label="Byzantine Shield"
              color="info"
              variant="outlined"
            />
          </Box>
        </Box>

        <Box sx={{ position: 'relative', height: '500px' }}>
          <div
            ref={mapRef}
            style={{
              height: '100%',
              width: '100%',
              borderRadius: '8px',
              overflow: 'hidden'
            }}
          />

          {/* Legend */}
          <Box
            sx={{
              position: 'absolute',
              top: 10,
              right: 10,
              background: 'rgba(0, 0, 0, 0.8)',
              p: 2,
              borderRadius: '8px',
              border: '1px solid rgba(255, 255, 255, 0.1)'
            }}
          >
            <Typography variant="subtitle2" sx={{ mb: 1 }}>Hospital Status</Typography>
            {['online', 'training', 'offline'].map(status => (
              <Box key={status} sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
                <Box
                  sx={{
                    width: 12,
                    height: 12,
                    borderRadius: '50%',
                    backgroundColor: getStatusColor(status),
                    mr: 1,
                    animation: status === 'training' ? 'pulse 2s infinite' : 'none'
                  }}
                />
                <Typography variant="caption" sx={{ textTransform: 'capitalize' }}>
                  {status}
                </Typography>
              </Box>
            ))}
          </Box>
        </Box>

        {/* Hospital Details Panel */}
        {selectedHospital && (
          <Box
            sx={{
              mt: 2,
              p: 2,
              background: 'rgba(0, 255, 136, 0.1)',
              border: '1px solid rgba(0, 255, 136, 0.3)',
              borderRadius: '8px'
            }}
          >
            <Typography variant="h6" sx={{ color: '#00ff88', mb: 1 }}>
              {selectedHospital.name}
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={3}>
                <Typography variant="body2" color="text.secondary">Hospital ID</Typography>
                <Typography variant="body1">{selectedHospital.id}</Typography>
              </Grid>
              <Grid item xs={3}>
                <Typography variant="body2" color="text.secondary">Status</Typography>
                <Chip
                  label={selectedHospital.status.toUpperCase()}
                  size="small"
                  sx={{
                    backgroundColor: getStatusColor(selectedHospital.status),
                    color: '#000',
                    fontWeight: 'bold'
                  }}
                />
              </Grid>
              <Grid item xs={3}>
                <Typography variant="body2" color="text.secondary">Reputation</Typography>
                <Typography variant="body1">
                  {(selectedHospital.reputation * 100).toFixed(1)}%
                </Typography>
              </Grid>
              <Grid item xs={3}>
                <Typography variant="body2" color="text.secondary">HealthTokens</Typography>
                <Typography variant="body1">{selectedHospital.tokens}</Typography>
              </Grid>
            </Grid>
          </Box>
        )}

        {/* Stats Footer */}
        <Box sx={{ mt: 2, display: 'flex', justifyContent: 'space-between' }}>
          <Typography variant="body2" color="text.secondary">
            Connected Hospitals: {hospitals.filter(h => h.status === 'online').length}/{hospitals.length}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Last Update: {new Date().toLocaleTimeString()}
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

export default HospitalMap;

// Add CSS for attack pulse
const styles = document.createElement('style');
styles.innerHTML = `
  @keyframes attack-pulse {
    0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(255, 0, 0, 0.7); }
    70% { transform: scale(1.5); box-shadow: 0 0 0 20px rgba(255, 0, 0, 0); }
    100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(255, 0, 0, 0); }
  }
  .attack-pulse {
    animation: attack-pulse 1s infinite;
  }
`;
document.head.appendChild(styles);