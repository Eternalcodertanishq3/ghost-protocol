# How to Use Your Own Patient Data

## 📋 Step 1: Prepare Your Data

Your PDF patient data needs to be converted to CSV format. Here's the expected format:

```csv
Age,Glucose,BMI,BloodPressure,Cholesterol,HeartRate,Outcome
45,120,28.5,80,200,72,0
55,180,35.2,95,260,88,1
38,95,24.1,70,180,68,0
...
```

**Requirements:**
- One row per patient
- Feature columns (any measurements)
- One **target column** (usually `Outcome`: 0 = no disease, 1 = has disease)

---

## 📂 Step 2: Organize Data by Hospital

Create separate CSV files for each hospital:

```
data/
└── my_hospitals/
    ├── apollo_hospital.csv      (500 patients)
    ├── fortis_hospital.csv      (300 patients)
    └── aiims_hospital.csv       (700 patients)
```

---

## 🏃 Step 3: Run the Interactive Demo

### Option A: Use Your Own Data
```bash
python run_interactive_demo.py --data data/my_hospitals/apollo.csv data/my_hospitals/fortis.csv data/my_hospitals/aiims.csv
```

### Option B: With Custom Settings
```bash
python run_interactive_demo.py \
    --data data/my_hospitals/*.csv \
    --target Outcome \
    --rounds 5 \
    --epochs 10
```

### Option C: Generate Synthetic Data (for testing)
```bash
python run_interactive_demo.py --generate
```

---

## ⚙️ Command Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--data` | `-d` | Path(s) to CSV files | Auto-detect |
| `--target` | `-t` | Target column name | "Outcome" |
| `--rounds` | `-r` | Federation rounds | 3 |
| `--epochs` | `-e` | Epochs per round | 5 |
| `--generate` | | Generate synthetic data | False |

---

## 📊 Example: Different Diseases

### Diabetes Prediction
```csv
Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigree,Age,Outcome
6,148,72,35,0,33.6,0.627,50,1
1,85,66,29,0,26.6,0.351,31,0
```

### Heart Disease Prediction
```csv
Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,HeartDisease
52,1,0,125,212,0,1,168,0,1.0,0
53,1,0,140,203,1,0,155,1,3.1,1
```

### Cancer Detection
```csv
Age,TumorSize,MarginStatus,LymphNodes,Grade,HormoneReceptor,Outcome
45,2.1,0,0,2,1,0
62,4.5,1,3,3,0,1
```

---

## 🔄 Converting PDF to CSV

If you have PDF patient records, you can:

1. **Manual Entry**: Copy data into Excel, save as CSV
2. **Python Script**: Use `tabula-py` or `pdfplumber`
3. **Online Tools**: Use PDF to CSV converters

### Example Python Conversion:
```python
import tabula

# Extract tables from PDF
tables = tabula.read_pdf("patient_records.pdf", pages="all")

# Save as CSV
for i, table in enumerate(tables):
    table.to_csv(f"hospital_{i+1}_data.csv", index=False)
```

---

## 🎯 What Happens During Training

```
Round 1:
├── Hospital 1 trains locally (5 epochs)
│   └── Only model weights sent to SNA
├── Hospital 2 trains locally (5 epochs)
│   └── Only model weights sent to SNA
├── Hospital 3 trains locally (5 epochs)
│   └── Only model weights sent to SNA
└── SNA aggregates weights (FedAvg)
    └── Global model updated

Round 2:
├── All hospitals receive updated global model
├── Each trains again on their LOCAL data
└── ... repeat ...

RESULT: Better model, NO patient data shared! 🔒
```

---

## ⏱️ Training Time Explanation

| Parameter | Effect |
|-----------|--------|
| More **Rounds** | Better convergence, longer time |
| More **Epochs** | Better local learning, longer time |
| More **Patients** | Better accuracy, longer time |
| More **Features** | More complex model, longer time |

**Recommended for Testing:**
- Rounds: 3-5
- Epochs: 5-10
- Patients: 500+ per hospital

**For Production:**
- Rounds: 10-20
- Epochs: 10-20
- Patients: 1000+ per hospital
