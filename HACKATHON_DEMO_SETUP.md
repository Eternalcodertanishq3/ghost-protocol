# 🏥 Ghost Protocol - Multi-Laptop Hackathon Demo Setup

This guide explains how to run a distributed federated learning demo with 4-5 physical laptops, each acting as a hospital.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    CENTRAL SERVER                           │
│              (Your Main Laptop/PC)                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  SNA (Secure National Aggregator)                   │   │
│  │  - Port 8000 (opens to network)                     │   │
│  │  - Frontend Dashboard (Port 3000)                   │   │
│  │  - Byzantine Shield                                  │   │
│  │  - HealthToken Ledger                               │   │
│  └─────────────────────────────────────────────────────┘   │
│              ↑        ↑        ↑        ↑                   │
└──────────────┼────────┼────────┼────────┼───────────────────┘
               │        │        │        │
    ┌──────────┴───┐  ┌─┴─────┐  ┌┴──────┐  ┌┴──────────┐
    │   LAPTOP 1   │  │LAPTOP 2│  │LAPTOP 3│  │ LAPTOP 4  │
    │  "AIIMS_Delhi"│  │"Fortis"│  │"Apollo"│  │"CMC_Vellore"│
    │  Ghost Agent │  │Ghost Ag│  │Ghost Ag│  │ Ghost Ag  │
    └──────────────┘  └────────┘  └────────┘  └───────────┘
```

---

## Step 1: Prepare the Central Server (Main Laptop)

### 1.1 Find Your IP Address

**Windows (PowerShell):**
```powershell
ipconfig | findstr /i "IPv4"
```

**Linux/Mac:**
```bash
hostname -I
```

**Example output:** `192.168.1.100` - **Note this down!**

### 1.2 Configure Firewall (Allow Port 8000)

**Windows:**
```powershell
# Run as Administrator
netsh advfirewall firewall add rule name="Ghost Protocol SNA" dir=in action=allow protocol=TCP localport=8000
```

**Linux:**
```bash
sudo ufw allow 8000
```

### 1.3 Start the SNA Server

```powershell
cd "c:\Personal Projects\Ghost Protocol\ghost-protocol"
python -m sna.main
```

You should see:
```
INFO:     Started server process on 0.0.0.0:8000
```

### 1.4 Start the Frontend Dashboard

In a new terminal:
```powershell
cd "c:\Personal Projects\Ghost Protocol\ghost-protocol\frontend"
npm start
```

Open browser: `http://localhost:3000`

---

## Step 2: Prepare Each Hospital Laptop

### 2.1 Copy Required Files to Each Laptop

Create a folder on each laptop and copy these files:
- `hospital_agent.py`
- `data/synthetic_health_data.py`
- `data/__init__.py`

Or use a USB drive / cloud share.

### 2.2 Install Dependencies on Each Laptop

```bash
pip install torch numpy pandas scikit-learn requests
```

### 2.3 Verify Network Connection

From each laptop, test connection to the central server:
```bash
ping 192.168.1.100
curl http://192.168.1.100:8000/status
```

---

## Step 3: Start Each Hospital (on Different Laptops)

### Laptop 1 - AIIMS Delhi
```bash
python hospital_agent.py --hospital AIIMS_Delhi --server 192.168.1.100:8000 --rounds 10
```

### Laptop 2 - Fortis Mumbai
```bash
python hospital_agent.py --hospital Fortis_Mumbai --server 192.168.1.100:8000 --rounds 10
```

### Laptop 3 - Apollo Chennai
```bash
python hospital_agent.py --hospital Apollo_Chennai --server 192.168.1.100:8000 --rounds 10
```

### Laptop 4 - CMC Vellore
```bash
python hospital_agent.py --hospital CMC_Vellore --server 192.168.1.100:8000 --rounds 10
```

### Laptop 5 - PGIMER Chandigarh (Optional)
```bash
python hospital_agent.py --hospital PGIMER_Chandigarh --server 192.168.1.100:8000 --rounds 10
```

---

## Step 4: Watch the Demo

### On Central Server Dashboard
1. Open `http://localhost:3000`
2. Watch hospitals connect in real-time
3. See aggregation rounds complete
4. View privacy budgets and HealthToken distributions

### On Each Laptop
Each laptop shows:
```
============================================================
  🏥 GHOST PROTOCOL - HOSPITAL AGENT
============================================================
  Hospital ID:     AIIMS_Delhi
  Machine:         LAPTOP-ABC123
  Local IP:        192.168.1.101
  SNA Server:      http://192.168.1.100:8000
  Patient Records: 1000
  Privacy Budget:  ε=1.0/round
============================================================
```

---

## Hackathon Demo Script (5 minutes)

### Minute 0-1: Introduction
- "We're demonstrating federated learning for healthcare"
- "5 hospitals, each with private patient data, training together without sharing data"

### Minute 1-2: Show Architecture
- Point to each laptop: "This is AIIMS Delhi with 1000 patients..."
- Show the central dashboard

### Minute 2-3: Start Training
- Start each laptop one by one
- Watch them connect to the central server
- Show real-time updates on dashboard

### Minute 3-4: Explain Privacy
- "Each hospital uses differential privacy - ε budget shown here"
- "Model updates are aggregated using Byzantine-robust geometric median"
- "HealthTokens are distributed as incentives"

### Minute 4-5: Show Results
- "Global model improved from X to Y accuracy"
- "No raw patient data ever left any hospital"
- "Fully DPDP compliant"

---

## Troubleshooting

### "Cannot connect to SNA server"
1. Check if central server firewall allows port 8000
2. Verify all laptops are on the same WiFi/network
3. Try pinging the central server IP

### "Connection timeout"
1. The network might be slow
2. Increase timeout: `--delay 30`

### "Module not found"
1. Make sure you copied all required files
2. Install dependencies: `pip install torch numpy pandas scikit-learn requests`

---

## Files Required on Each Laptop

```
hospital_laptop/
├── hospital_agent.py       # Main agent script
└── data/
    ├── __init__.py
    └── synthetic_health_data.py
```

---

## Command Reference

```bash
# Full options
python hospital_agent.py \
  --hospital AIIMS_Delhi \     # Hospital name
  --server 192.168.1.100:8000 \ # Central server IP:PORT
  --rounds 10 \                 # Number of training rounds
  --delay 10 \                  # Seconds between rounds
  --patients 1000 \             # Synthetic patient count
  --epsilon 1.0                 # Privacy budget per round
```

---

## What Makes This "Real"

| Component | Real Implementation |
|-----------|---------------------|
| **Training** | Real PyTorch neural network with DP-SGD |
| **Privacy** | Real gradient clipping + Gaussian noise |
| **Aggregation** | Real Byzantine-robust geometric median |
| **Network** | Real HTTP communication between machines |
| **Incentives** | Real HealthToken distribution |
| **Compliance** | Real DPDP privacy budget tracking |

The only "synthetic" part is the patient data - but that's intentional for the demo!
