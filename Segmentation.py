import pandas as pd
import streamlit as st
import numpy as np
import joblib

# Retain dark theme and Poppins font for visual consistency
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');
    html, body, .stApp {
        background: #15171a;
        font-family: 'Poppins', sans-serif;
        color: #f3f6fa;
    }
    .gradient-title {
        font-size: 2.6rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #4f8cff 0%, #43e97b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.2rem;
        letter-spacing: 0.5px;
    }
    .stNumberInput > div > div > input {
        background: #23242a;
        color: #f3f6fa;
        border-radius: 7px;
        border: 1.5px solid #23283b;
        font-size: 1.08rem;
        font-family: 'Poppins', sans-serif;
        padding: 0.5em 0.8em;
        margin-bottom: 0.2em;
        transition: border 0.2s;
    }
    .stNumberInput > div > div > input:focus {
        border: 2px solid #4f8cff;
        outline: none;
    }
    .stButton > button {
        background: #4f8cff;
        color: #fff;
        border-radius: 7px;
        border: none;
        font-weight: 600;
        font-size: 1.08rem;
        padding: 0.7em 2.2em;
        margin-top: 1.5em;
        box-shadow: 0 2px 8px rgba(79,140,255,0.10);
        transition: background 0.2s, transform 0.1s;
    }
    .stButton > button:hover {
        background: #3973d6;
        color: #fff;
        transform: scale(1.03);
    }
    .stSuccess {
        background: #23283b !important;
        color: #4f8cff !important;
        border-radius: 7px;
        font-size: 1.13em;
        font-weight: 600;
        margin-top: 1.5em;
    }
    label, .stNumberInput label {
        color: #bfc8e6 !important;
        font-weight: 600;
        font-size: 1.01rem;
        margin-bottom: 0.18rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load saved models
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

# Mapping cluster numbers to segment names
cluster_names = {
    0: "Premium Loyalists",
    1: "Low Income Inactives",
    2: "Engaged Spenders",
    3: "Senior Multi-Channel Users",
    4: "Wealthy Seniors",
    5: "Unengaged Budget Seekers"
}

st.set_page_config(page_title="Customer Segmentation", layout="centered")
st.markdown('<div class="gradient-title">Customer Segmentation Prediction</div>', unsafe_allow_html=True)
st.write("Fill in customer details to predict their segment.")

# Input fields with help tooltips
age = st.number_input("Customer Age", min_value=18, max_value=100, value=35, help="Age of the customer in years")
income = st.number_input("Annual Income (₹)", min_value=0, max_value=200000, value=50000, help="Yearly income in INR")
total_spending = st.number_input("Total Spending (₹)", min_value=0, max_value=5000, value=1000, help="Total spending amount")
num_web_purchases = st.number_input("Web Purchases", min_value=0, max_value=100, value=10, help="Number of online purchases")
num_store_purchases = st.number_input("Store Purchases", min_value=0, max_value=100, value=10, help="Number of in-store purchases")
num_web_visits = st.number_input("Web Visits per Month", min_value=0, max_value=60, value=4, help="Monthly web visit count")
recency = st.number_input("Recency (days)", min_value=0, max_value=365, value=30, help="Days since last purchase")

# DataFrame creation in the correct order
input_data = pd.DataFrame({
    "Age": [age],
    "Income": [income],
    "Total_Spending": [total_spending],
    "NumWebPurchases": [num_web_purchases],
    "NumStorePurchases": [num_store_purchases],
    "NumWebVisitsMonth": [num_web_visits],
    "Recency": [recency]
})

# Ensuring column order matches model training
ordered_columns = ["Age", "Income", "Total_Spending", "NumWebPurchases", "NumStorePurchases", "NumWebVisitsMonth", "Recency"]
input_data = input_data[ordered_columns]

# Prediction
if st.button("Predict Segment"):
    input_scaled = scaler.transform(input_data)
    cluster_num = kmeans.predict(input_scaled)[0]
    cluster_label = cluster_names.get(cluster_num, "Unknown Segment")
    st.success(f"Predicted Segment: {cluster_label} (Cluster {cluster_num})") 