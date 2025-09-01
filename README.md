# 💳 Synthetic Fraud Transaction Data Generator (Rule-based + CTGAN)

This project is a **GenAI-powered Streamlit app** that generates **synthetic credit card transaction data** for **fraud detection training**.  
It supports two modes:
1. **Rule-based Generator** → Creates fake data using patterns & Faker library.  
2. **CTGAN Generator** → Uses a GAN (CTGAN) trained on a real dataset to generate realistic synthetic data.  

---

## 🚀 Features
- Generate synthetic transactions with fields like:
  - TransactionID, Timestamp, Amount, Merchant, Location, CardType, TransactionType, Fraudulent
- Choose **Rule-based** (no training required) or **CTGAN** (trained on a real dataset)
- Control number of transactions & fraud ratio
- Upload your own real dataset to train CTGAN
- Visualize:
  - Fraud vs Non-Fraud counts
  - Transaction Amount distribution
- Download generated dataset as CSV

---

## 🛠️ Tech Stack
- [Python 3.10+](https://www.python.org/downloads/)
- [Streamlit](https://streamlit.io/) – Web app framework
- [Faker](https://faker.readthedocs.io/) – Fake merchant/location/card data
- [CTGAN](https://github.com/sdv-dev/CTGAN) – GAN for tabular synthetic data
- [Matplotlib](https://matplotlib.org/) / [Plotly](https://plotly.com/python/) – Visualizations
- [Pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) – Data handling

---

## 📦 Installation

1. Clone this repo:
   ```bash
   git clone https://github.com/your-username/SyntheticDataGenerator.git
   cd SyntheticDataGenerator
