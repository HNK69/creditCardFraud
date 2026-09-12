import streamlit as st
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]      
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predict import predict_transaction
                                                                          #STEP2 Connect the inference pipeline

st.set_page_config(                                                       #STEP1Basic Streamlit setup
    page_title = "Credit Card Fraud Detection",
    page_icon = "💳",
    layout = "centered",
)

st.title("💳 Credit Card Fraud Detection")

st.write(
    "Enter transaction details to estimate the probability "
    "of fraud."
)

st.subheader("Transaction Details")

def load_transaction():

    test_data = pd.read_csv(
        "data/raw/creditcard.csv"
    )

    fraud_row = test_data[test_data["Class"] == 1].iloc[0]

    st.session_state["time"] = float(fraud_row["Time"])
    st.session_state["amount"] = float(fraud_row["Amount"])

    for i in range(1,29):
        st.session_state[f'V{i}'] = float(fraud_row[f"V{i}"])


st.button(
    "Load test transaction",
    on_click = load_transaction()
)

def reset_transaction():
    st.session_state["amount"] = 100.0
    st.session_state["time"] = 0.0

    for i in range(1,29):
         st.session_state[f"V{i}"] = 0.0

amount = st.number_input(                                                #STEP3 Add the first input
    "Transaction_Amount",
    min_value = 0.0,
    value = 100.0,
    step = 1.0,
    key = "amount",
)



# st.write("Amount entered: ", amount)

#STEP4 Add the Time as input
time = st.number_input(
    "Transaction Time(seconds)",
    min_value = 0.0,
    value = 0.0,
    step = 1.0,
    key = "time",
)

# st.write("Time entered:", time)

#step5 Add V1–V28 inputs efficiently

v_features = {}

st.subheader("Transaction Features")

col1, col2 = st.columns(2)

for i in range(1, 29):
    with col1 if i % 2 == 1 else col2:
        v_features[f"V{i}"] = st.number_input(
            f"V{i}",
            value = 0.0,
            format = "%.6f",
            key = f'V{i}',
        )

#step6 Build the transaction DataFrame

transaction_data = {
    "Time":time,
    **v_features,
    "Amount":amount,
}

transaction = pd.DataFrame([transaction_data])

# st.write("Input shape:", transaction.shape)

if st.button("predict transaction", type = "primary"):
    result = predict_transaction(transaction)

    st.subheader("Prediction Result")

    st.write(
        "Fraud probability"
        f"{result["fraud_probability"]: .2%}"
    )

    # st.caption(
    #     f"Decision threshold: {threshold:.2%}"
    # )

    if result["prediction"] == 1:
        st.error("⚠️ Fraudulent Transaction")
    else:
        st.success("✅ Legitimate Transaction")

st.button(
    "Reset Transaction",
    on_click = reset_transaction
)