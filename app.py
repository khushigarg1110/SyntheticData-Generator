import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from faker import Faker
from ctgan import CTGAN


fake = Faker()

# ---- Rule-based generator ----
def generate_rule_based(n=1000, fraud_ratio=0.1):
    data = []
    fraud_count = int(n * fraud_ratio)

    for i in range(n):
        is_fraud = 1 if i < fraud_count else 0
        amount = round(np.random.exponential(50), 2)

        if is_fraud:
            amount = round(random.uniform(500, 5000), 2)
            merchant = random.choice(["LuxuryStore", "UnknownMerchant", "CryptoExchange"])
            location = random.choice(["Nigeria", "Russia", "DarkWeb"])
            card_type = random.choice(["VirtualCard", "StolenCard"])
        else:
            merchant = fake.company()
            location = fake.country()
            card_type = random.choice(["Visa", "MasterCard", "Amex"])

        txn = {
            "TransactionID": fake.uuid4(),
            "Timestamp": fake.date_time_this_year(),
            "Amount": amount,
            "Merchant": merchant,
            "Location": location,
            "CardType": card_type,
            "TransactionType": random.choice(["POS", "Online", "ATM"]),
            "Fraudulent": is_fraud
        }
        data.append(txn)

    df = pd.DataFrame(data)
    return df.sample(frac=1).reset_index(drop=True)


# ---- CTGAN generator ----
@st.cache_resource
def train_ctgan(real_data):
    ctgan = CTGAN(epochs=50)  # keep epochs small for demo
    ctgan.fit(real_data)
    return ctgan

def generate_ctgan(ctgan, n=1000):
    synthetic_data = ctgan.sample(n)
    return synthetic_data


# ---- Streamlit App ----
st.title("💳 Synthetic Fraud Transaction Data Generator (Rule-based + CTGAN)")

option = st.sidebar.selectbox("Generation Method", ["Rule-based", "CTGAN"])
n_samples = st.sidebar.slider("Number of Transactions", 100, 5000, 1000, 100)
fraud_ratio = st.sidebar.slider("Fraud Ratio (%) (Rule-based only)", 0, 50, 10, 5) / 100

if option == "Rule-based":
    if st.button("Generate Rule-based Data"):
        df = generate_rule_based(n=n_samples, fraud_ratio=fraud_ratio)
        st.subheader("Generated Transactions (Rule-based)")
        st.dataframe(df.head(20))

        # Visualization
        st.write("### Fraud vs Non-Fraud Count")
        st.bar_chart(df["Fraudulent"].value_counts())

        fig, ax = plt.subplots()
        ax.hist(df["Amount"], bins=30, edgecolor="black")
        ax.set_xlabel("Amount")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        # Download
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "synthetic_rulebased.csv", "text/csv")

elif option == "CTGAN":
    uploaded_file = st.sidebar.file_uploader("Upload Real Dataset (CSV)", type=["csv"])
    # if uploaded_file:
    #     real_df = pd.read_csv(uploaded_file)

    #     # Drop TransactionID/Time if present (not useful for CTGAN training)
    #     if "TransactionID" in real_df.columns: 
    #         real_df = real_df.drop(columns=["TransactionID"])
    #     if "Timestamp" in real_df.columns: 
    #         real_df = real_df.drop(columns=["Timestamp"])

    #     st.write("Training CTGAN model on uploaded dataset...")
    #     ctgan_model = train_ctgan(real_df)
    if uploaded_file:
        real_df = pd.read_csv(uploaded_file)

        # Drop ID/Time if present
        if "TransactionID" in real_df.columns: 
            real_df = real_df.drop(columns=["TransactionID"])
        if "Timestamp" in real_df.columns: 
            real_df = real_df.drop(columns=["Timestamp"])

        # Detect categorical/discrete columns
        discrete_columns = real_df.select_dtypes(include=["object", "category"]).columns.tolist()
        if "Fraudulent" in real_df.columns:
            discrete_columns.append("Fraudulent")  # treat Fraudulent as categorical too

        st.write("Training CTGAN model on uploaded dataset...")
        ctgan_model = CTGAN(epochs=50)
        ctgan_model.fit(real_df, discrete_columns=discrete_columns)

        if st.button("Generate CTGAN Data"):
            synthetic_df = generate_ctgan(ctgan_model, n=n_samples)
            st.subheader("Generated Transactions (CTGAN)")
            st.dataframe(synthetic_df.head(20))

            # Visualization
            if "Fraudulent" in synthetic_df.columns:
                st.write("### Fraud vs Non-Fraud Count")
                st.bar_chart(synthetic_df["Fraudulent"].value_counts())

            fig, ax = plt.subplots()
            ax.hist(synthetic_df["Amount"], bins=30, edgecolor="black")
            ax.set_xlabel("Amount")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

            # Download
            csv = synthetic_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv, "synthetic_ctgan.csv", "text/csv")
    else:
        st.warning("Upload a real dataset CSV to train CTGAN.")
