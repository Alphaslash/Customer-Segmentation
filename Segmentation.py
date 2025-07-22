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

# Custom CSS for better card, centering, and responsiveness
st.markdown('''
    <style>
    .main-section {
        max-width: 800px;
        margin: 2rem auto 0 auto;
        padding: 2.5rem 2rem 2rem 2rem;
        background: #181a20;
        border-radius: 18px;
        box-shadow: 0 4px 32px 0 rgba(79,140,255,0.10);
    }
    .card {
        background: #23242a;
        padding: 2rem 1.5rem 1.5rem 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 12px 0 rgba(79,140,255,0.08);
        margin-bottom: 1.5rem;
    }
    @media (max-width: 900px) {
        .main-section { max-width: 98vw; padding: 1rem; }
        .stColumns { flex-direction: column !important; }
    }
    .gradient-title {
        text-align: center;
        margin-bottom: 2.5rem !important;
        margin-top: 1.5rem !important;
    }
    </style>
''', unsafe_allow_html=True)

# Add custom CSS to align the 'Press Enter to apply' message for number inputs
st.markdown('''
    <style>
    .stNumberInput .stCaption {
        margin-top: 0.2rem !important;
        margin-left: 0.2rem !important;
        padding: 0 !important;
        font-size: 0.95em !important;
        color: #bfc8e6 !important;
        text-align: left !important;
        display: block !important;
    }
    </style>
''', unsafe_allow_html=True)

# Gradient title at the top
st.markdown('<div class="gradient-title">Customer Segmentation Prediction</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.write("Fill in customer details to predict their segment.")
    with st.form("segmentation_form"):
        age = st.number_input("Customer Age", min_value=0, max_value=100, value=0, help="Age of the customer in years")
        income = st.number_input("Annual Income ($)", min_value=0, max_value=200000, value=0, help="Yearly income in USD")
        total_spending = st.number_input("Total Spending ($)", min_value=0, max_value=5000, value=0, help="Total spending amount in USD")
        num_web_purchases = st.number_input("Web Purchases", min_value=0, max_value=100, value=0, help="Number of online purchases")
        num_store_purchases = st.number_input("Store Purchases", min_value=0, max_value=100, value=0, help="Number of in-store purchases")
        num_web_visits = st.number_input("Web Visits per Month", min_value=0, max_value=60, value=0, help="Monthly web visit count")
        recency = st.number_input("Recency (days)", min_value=0, max_value=365, value=0, help="Days since last purchase")
        submit = st.form_submit_button("Predict Segment")

    DEFAULTS = {
        "age": 35,
        "income": 50000,
        "total_spending": 1000,
        "num_web_purchases": 10,
        "num_store_purchases": 10,
        "num_web_visits": 4,
        "recency": 30
    }

    if submit:
        with st.spinner("Predicting segment..."):
            age_val = age if age != 0 else DEFAULTS["age"]
            income_val = income if income != 0 else DEFAULTS["income"]
            total_spending_val = total_spending if total_spending != 0 else DEFAULTS["total_spending"]
            num_web_purchases_val = num_web_purchases if num_web_purchases != 0 else DEFAULTS["num_web_purchases"]
            num_store_purchases_val = num_store_purchases if num_store_purchases != 0 else DEFAULTS["num_store_purchases"]
            num_web_visits_val = num_web_visits if num_web_visits != 0 else DEFAULTS["num_web_visits"]
            recency_val = recency if recency != 0 else DEFAULTS["recency"]

            input_data = pd.DataFrame({
                "Age": [age_val],
                "Income": [income_val],
                "Total_Spending": [total_spending_val],
                "NumWebPurchases": [num_web_purchases_val],
                "NumStorePurchases": [num_store_purchases_val],
                "NumWebVisitsMonth": [num_web_visits_val],
                "Recency": [recency_val]
            })

            ordered_columns = ["Age", "Income", "Total_Spending", "NumWebPurchases", "NumStorePurchases", "NumWebVisitsMonth", "Recency"]
            input_data = input_data[ordered_columns]

            input_scaled = scaler.transform(input_data)
            cluster_num = kmeans.predict(input_scaled)[0]
            cluster_label = cluster_names.get(cluster_num, "Unknown Segment")
            st.success(f"Predicted Segment: {cluster_label} (Cluster {cluster_num})")

def format_inr(n):
    s = str(int(n))
    if len(s) <= 3:
        return s
    else:
        last3 = s[-3:]
        rest = s[:-3]
        parts = []
        while len(rest) > 2:
            parts.append(rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.append(rest)
        return ','.join(parts[::-1]) + ',' + last3

with col2:
    st.markdown("#### Rupee to Dollar Helper")
    st.write("Enter an amount in rupees to see the equivalent in dollars. (Exchange rate: 1 USD = 83 INR)")
    rupees = st.number_input("Amount in Rupees", min_value=0, value=0, step=1, key="rupee_helper")
    if rupees:
        dollars = rupees / 83
        st.success(f"₹{format_inr(rupees)} = ${dollars:.2f}") 