# Hospital Laptop Package for Hackathon Demo

This folder contains everything needed to run a Ghost Protocol hospital agent on a laptop.

## Quick Start

1. **Install Python dependencies:**
   ```bash
   pip install torch numpy pandas scikit-learn requests
   ```

2. **Run the startup script:**
   - Windows: Double-click `START_HOSPITAL.bat`
   - Linux/Mac: `python hospital_agent.py --hospital YOUR_HOSPITAL --server CENTRAL_IP:8000`

3. **Enter when prompted:**
   - Hospital Name (e.g., `AIIMS_Delhi`, `Fortis_Mumbai`)
   - Central Server IP (get this from the main laptop)

## Files Included

- `hospital_agent.py` - Main Ghost Agent script
- `data/synthetic_health_data.py` - Synthetic patient data generator
- `data/__init__.py` - Python module init
- `START_HOSPITAL.bat` - Windows startup script
- `README.md` - This file

## What Happens

1. Agent generates synthetic patient data locally
2. Trains a neural network with differential privacy
3. Sends model updates to central server
4. Central server aggregates all hospitals
5. Global model improves without sharing raw data!

## Hospital Names to Use

- `AIIMS_Delhi`
- `Fortis_Mumbai`
- `Apollo_Chennai`
- `CMC_Vellore`
- `PGIMER_Chandigarh`
