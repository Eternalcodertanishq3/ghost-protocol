# EMR Configuration Templates for Ghost Protocol
# Use these templates based on your hospital's EMR system

# ==========================================
# OPTION 1: FHIR R4 (Modern EMR Systems)
# ==========================================
# Compatible with: Epic, Cerner, AWS HealthLake, etc.
#
# python hospital_agent_emr.py \
#   --hospital AIIMS_Delhi \
#   --server 192.168.1.100:8000 \
#   --emr-type fhir \
#   --emr-url https://fhir.hospital.local/api \
#   --emr-auth-token YOUR_OAUTH_TOKEN


# ==========================================
# OPTION 2: HL7 v2.x (Legacy EMR Systems)
# ==========================================
# Compatible with: Older EMR systems, Lab systems
#
# python hospital_agent_emr.py \
#   --hospital Fortis_Mumbai \
#   --server 192.168.1.100:8000 \
#   --emr-type hl7 \
#   --emr-connection "mllp://192.168.1.50:2575"


# ==========================================
# OPTION 3: Direct Database Connection
# ==========================================
# Compatible with: Custom EMR databases
#
# PostgreSQL:
# python hospital_agent_emr.py \
#   --hospital Apollo_Chennai \
#   --server 192.168.1.100:8000 \
#   --emr-type database \
#   --emr-connection "postgresql://user:pass@db.hospital.local/emr_db"
#
# MySQL:
# python hospital_agent_emr.py \
#   --hospital CMC_Vellore \
#   --server 192.168.1.100:8000 \
#   --emr-type database \
#   --emr-connection "mysql://user:pass@db.hospital.local/emr_db"


# ==========================================
# OPTION 4: Synthetic Data (Hackathon Demo)
# ==========================================
# No EMR needed - generates realistic patient data
#
# python hospital_agent_emr.py \
#   --hospital PGIMER_Chandigarh \
#   --server 192.168.1.100:8000 \
#   --emr-type synthetic \
#   --patients 1000


# ==========================================
# FIELD MAPPINGS (FHIR LOINC Codes)
# ==========================================
# Vital Signs:
#   85354-9  - Systolic Blood Pressure
#   8462-4   - Diastolic Blood Pressure
#   8867-4   - Heart Rate
#   8310-5   - Body Temperature
#   59408-5  - Oxygen Saturation
#
# Lab Results:
#   718-7    - Hemoglobin
#   6690-2   - White Blood Cells
#   777-3    - Platelets
#   2339-0   - Glucose
#   38483-4  - Creatinine

# ==========================================
# ICD-10 DIAGNOSIS CODES
# ==========================================
# E10-E14  - Diabetes
# I10-I15  - Hypertension
# I20-I25  - Heart Disease
# N18-N19  - Kidney Disease
