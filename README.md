# 👻 Ghost Protocol
**Privacy-Preserving AI Training on Decentralized Healthcare Data**

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)
![Privacy: Differential](https://img.shields.io/badge/Privacy-DP_ε%3D2.45-green)
![Security: Homomorphic](https://img.shields.io/badge/Security-Paillier_HE-orange)

---

## 🛑 The Truth About Privacy AI
Most collaborative learning demos are fake. They simulate distributed networks on a single CSV file and call it a day.

**Ghost Protocol is different.** 
This is a true **Federated Learning (FL)** system designed for the hostile reality of healthcare data privacy (DPDP/GDPR). It allows hospitals to collaboratively train lifesaving AI models **without a single byte of patient data ever leaving their premises.**

> *"We don't move the data to the model. We move the model to the data."*

---

## 🛠️ Key Innovations

### 1. 🔒 Real Homomorphic Encryption (NO SIMULATIONS)
We rely on **Paillier Encryption** (via `phe` library) to encrypt model gradients.
*   The "Secure National Aggregator" performs **Homomorphic Addition** on encrypted ciphertexts.
*   The server *never* sees the raw updates, only the mathematical sum.
*   *Proof:* Check `demo_cinematic.py` for the implementation.

### 2. 🧠 Differential Privacy (Opacus)
We don't just "hope" for privacy; we calculate it.
*   Strict implementation of **Opacus** (PyTorch) to inject Gaussian noise.
*   Guaranteed **(ε, δ)-Differential Privacy** budget tracking.
*   Prevents model inversion attacks (reconstructing patient faces/data from weights).

### 3. 🛡️ Byzantine Shield
Distributed systems are vulnerable to "Poisoning Attacks" (malicious nodes).
*   Our **Shapley Value** analysis automatically detects and quarantines nodes that contribute harmful gradients.

---

## 🚀 Quick Start

### Prerequisites
*   Python 3.9+
*   Redis (for message queue)

### 1. Clone & Install
```bash
git clone https://github.com/yourusername/ghost-protocol.git
cd ghost-protocol
pip install -r requirements.txt
```

### 2. Run the "Real Math" Demo
Witness the full cryptographic cycle (Encryption -> Aggregation -> Decryption) on your local machine.
```bash
python demo_cinematic.py
```
*warning: This script performs real 1024-bit encryption. It is computationally intensive.*

---

## 🏗️ Architecture

```mermaid
graph TD
    User[Start Training] -->|Init| SNA[Secure National Aggregator]
    SNA -->|Broadcast Model| H1[Hospital Node A]
    SNA -->|Broadcast Model| H2[Hospital Node B]
    
    subgraph "Secure Hospital Silo A"
    D1[(Patient Data)] -->|Train| H1
    H1 -->|Add Noise (DP)| H1
    H1 -->|Encrypt (HE)| H1
    end

    subgraph "Secure Hospital Silo B"
    D2[(Patient Data)] -->|Train| H2
    H2 -->|Add Noise (DP)| H2
    H2 -->|Encrypt (HE)| H2
    end

    H1 -->|Encrypted Update| SNA
    H2 -->|Encrypted Update| SNA
    
    SNA -->|Homomorphic Aggregation| Global[New Global Model]
```


## 🤝 Contributing & Security (The "Hybrid" Approach)
**Privacy is a moving target.** While we rely on mathematically proven libraries (`opacus`, `phe`), implementation bugs are always possible.

We believe in **"Security through Visibility"**, not obscurity.
*   **Found a bug?** Please open an Issue. We treat security reports with highest priority.
*   **Want to break it?** We invite cryptographers and engineers to audit the `demo_cinematic.py` implementation.
*   **Pull Requests:** Welcome! Help us optimize the Paillier encryption steps or different model architectures.

We are building this *with* the community, not just *for* it.

## 📜 License
This project is open-sourced under the MIT License.
Simulations are easy. Privacy is hard. We chose the hard way.