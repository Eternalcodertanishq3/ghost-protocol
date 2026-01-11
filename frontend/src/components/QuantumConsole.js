import React, { useEffect, useRef, useState } from 'react';
import { Card, CardContent, Typography, Box } from '@mui/material';
import { Terminal } from '@mui/icons-material';

const QuantumConsole = ({ logs = [], height = '300px' }) => {
    const scrollRef = useRef(null);

    // Auto-scroll to bottom
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [logs]);

    const getLogColor = (type) => {
        switch (type) {
            case 'QUANTUM': return '#00ccff'; // Cyan for Quantum
            case 'SUCCESS': return '#00ff88'; // Green for Success
            case 'WARNING': return '#ffa502'; // Orange for Warning
            case 'THREAT': return '#ff4757';  // Red for Threat
            case 'SYSTEM': return '#888888';  // Gray for System
            default: return '#cccccc';
        }
    };

    return (
        <Card sx={{
            background: '#0a0a0a',
            border: '1px solid #333',
            boxShadow: '0 0 20px rgba(0, 255, 136, 0.1)'
        }}>
            <CardContent sx={{ p: 1, '&:last-child': { pb: 1 } }}>
                <Box sx={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 1,
                    borderBottom: '1px solid #333',
                    pb: 1,
                    mb: 1
                }}>
                    <Terminal sx={{ color: '#00ff88', fontSize: 20 }} />
                    <Typography variant="subtitle2" sx={{ color: '#00ff88', fontFamily: 'monospace' }}>
                        GHOST_PROTOCOL_V1.0 :: QUANTUM_VAULT_ACCESS
                    </Typography>
                </Box>

                <Box
                    ref={scrollRef}
                    sx={{
                        height: height,
                        overflowY: 'auto',
                        fontFamily: '"Consolas", "Monaco", "Courier New", monospace',
                        fontSize: '12px',
                        lineHeight: '1.5',
                        '&::-webkit-scrollbar': { width: '6px' },
                        '&::-webkit-scrollbar-track': { background: '#111' },
                        '&::-webkit-scrollbar-thumb': { background: '#333' }
                    }}
                >
                    {logs.length === 0 && (
                        <Typography sx={{ color: '#444', fontStyle: 'italic' }}>
                            &gt; System initialized. Waiting for quantum datastream...
                        </Typography>
                    )}

                    {logs.map((log, index) => (
                        <Box key={index} sx={{ mb: 0.5, display: 'flex', gap: 1 }}>
                            <Typography component="span" sx={{ color: '#444' }}>
                                [{new Date(log.timestamp).toLocaleTimeString().split(' ')[0]}]
                            </Typography>
                            <Typography component="span" sx={{ color: getLogColor(log.type) }}>
                                {log.type.padEnd(8, ' ')}
                            </Typography>
                            <Typography component="span" sx={{ color: '#ddd' }}>
                                &gt; {log.message}
                            </Typography>
                        </Box>
                    ))}
                    <Box sx={{ mt: 1, animation: 'blink 1s infinite' }}>
                        <Typography component="span" sx={{ color: '#00ff88' }}>_</Typography>
                    </Box>
                </Box>

                <style>
                    {`
            @keyframes blink {
              0%, 100% { opacity: 1; }
              50% { opacity: 0; }
            }
          `}
                </style>
            </CardContent>
        </Card>
    );
};

export default QuantumConsole;
