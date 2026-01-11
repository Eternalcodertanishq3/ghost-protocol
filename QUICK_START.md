# Ghost Protocol - Quick Start Guide

## 🚀 Run the System Locally

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Federated Learning Demo
```bash
python run_local_demo.py
```

This will:
1. ✅ Generate synthetic patient data for 3 hospitals
2. ✅ Initialize a diabetes prediction model
3. ✅ Run 5 federated learning rounds
4. ✅ Aggregate models using FedAvg
5. ✅ Save the trained model

---

## 📊 What the Demo Does

```
Hospital 1 ─┐
Hospital 2 ─┼──→ [SNA] ──→ Global Model ──→ All Hospitals
Hospital 3 ─┘

Each hospital:
1. Trains on LOCAL data (never shared)
2. Sends only MODEL WEIGHTS (not patient data)
3. Receives updated global model
```

---

## 🏥 Generated Data

The demo creates synthetic patient datasets in `data/hospitals/`:
- `hospital_1_data.csv` (500 patients)
- `hospital_2_data.csv` (500 patients)
- `hospital_3_data.csv` (500 patients)

**Features (8 total):**
| Feature | Description |
|---------|-------------|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-Hour serum insulin (mu U/ml) |
| BMI | Body mass index |
| DiabetesPedigree | Diabetes pedigree function |
| Age | Patient age (years) |

**Target:** `Outcome` (0 = No Diabetes, 1 = Diabetes)

---

## 📈 Expected Results

```
Final Results:
  Rounds Completed: 5
  Participating Hospitals: 3
  Total Patients (Privacy-Preserved): 1500
  Average Accuracy: ~78%
  Privacy Budget Used: ~2.5 ε
```

---

## 🔒 Privacy Features Demonstrated

| Feature | Status |
|---------|--------|
| Data Locality | ✅ Patient data never leaves hospital |
| Model Aggregation | ✅ FedAvg with weighted averaging |
| Non-IID Handling | ✅ Heterogeneous data distribution |
| DPDP Compliance | ✅ Privacy budget tracking |

---

## 🧪 Run Tests

```bash
# Run all passing tests
python -m pytest tests/test_integration.py tests/test_e2e_real.py -v

# Run security scan
python security_scan.py
```

---

## 🎯 Next Steps

1. **Run Full SNA Server:**
   ```bash
   # Create .env file first
   cp .env.example .env
   # Edit .env with your secrets
   
   # Run SNA
   python -m sna.main
   ```

2. **Run with Docker:**
   ```bash
   docker-compose up sna redis postgres
   ```

3. **Connect Real Hospital Agents:**
   ```bash
   python hospital_agent.py --hospital-id="Hospital_1" --data-path="data/hospitals/hospital_1_data.csv"
   ```

---

## 📁 Project Structure

```
ghost-protocol/
├── run_local_demo.py      # ← Run this!
├── data/
│   └── hospitals/         # Synthetic patient data
├── models/
│   └── registry.py        # Shared model definitions
├── sna/
│   ├── main.py           # SNA server
│   ├── byzantine_shield/ # Attack protection
│   ├── dpdp_auditor/    # Privacy monitoring
│   └── health_ledger/   # HealthToken rewards
└── tests/                # Test suites
```
