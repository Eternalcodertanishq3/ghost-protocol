# 🎬 Demo Video

Watch Ghost Protocol in action—**100% real cryptographic computation, zero simulations**.

## 📺 Watch on LinkedIn
[**Ghost Protocol: Full System Demo**](https://www.linkedin.com/in/eternalcodertanishq3/recent-activity/all/)

*(Find the video post with the title "Why this demo looks slow")*

---

## What the Demo Shows

| Step | Description | Technology |
|------|-------------|------------|
| 1 | 3 hospitals generate synthetic patient data | `data.synthetic_data` |
| 2 | 2048-bit Paillier keypair generation | `phe` library |
| 3 | Local training with privacy protection | **Opacus DP-SGD** |
| 4 | Encrypting ~2,700 neural network weights | **Paillier HE** |
| 5 | Homomorphic aggregation (FedAvg on ciphertext) | E(a) + E(b) = E(a+b) |

---

## Run It Yourself

```bash
# Clone the repo
git clone https://github.com/Eternalcodertanishq3/ghost-protocol.git
cd ghost-protocol

# Install dependencies
pip install -r requirements.txt

# Run the full system demo
python run_local_demo.py
```

**⚠️ Warning:** This demo takes ~20 minutes because it performs real 2048-bit cryptographic operations.

---

## Key Metrics from Demo

- **Privacy Budget:** ε = 2.51 per round (calculated by Opacus)
- **Parameters Encrypted:** ~2,700 per hospital
- **Encryption Time:** 3-6 minutes per hospital (real Paillier)
- **Final Accuracy:** 70%+ (varies by run)
