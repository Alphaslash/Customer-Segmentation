# Customer Segmentation – ML Project

This project implements a customer segmentation application using KMeans clustering on customer behavioral and demographic data. The app predicts customer segments to help businesses target marketing and improve customer engagement strategies.

---

## 📌 Objectives

- Develop a machine learning model to segment customers based on purchasing behavior and demographics.
- Build an interactive Streamlit web app to input customer details and predict segments.
- Interpret customer clusters with meaningful segment names.
- Enable businesses to understand their customer base for targeted marketing.

---

## 📁 Files Included

- `Segmentation.py` – Streamlit app source code for customer input and segmentation prediction.
- `kmeans_model.pkl` – Pre-trained KMeans clustering model serialized with joblib.
- `scaler.pkl` – StandardScaler object for feature normalization.
- `requirements.txt` – List of required Python packages.
- `README.md` – Project documentation.

---

## 🛠️ Tools & Libraries Used

- Python
- pandas
- numpy
- scikit-learn
- streamlit
- joblib

---

## 📊 Features

- User-friendly interface to enter customer data:
  - Age
  - Income
  - Total Spending
  - Number of Web Purchases
  - Number of Store Purchases
  - Number of Web Visits per Month
  - Recency (days since last purchase)
- Scales inputs using StandardScaler.
- Predicts customer segment with KMeans clustering.
- Displays segment name with a clear description.

---

## 📈 Customer Segments (Clusters)

| Cluster ID | Segment Name               |
|------------|----------------------------|
| 0          | Premium Loyalists           |
| 1          | Low Income Inactives        |
| 2          | Engaged Spenders            |
| 3          | Senior Multi-Channel Users  |
| 4          | Wealthy Seniors             |
| 5          | Unengaged Budget Seekers    |

---

## 💡 Key Insights

- Customers are segmented into distinct groups to tailor marketing efforts.
- Premium loyalists show high engagement and spending.
- Low income inactives require targeted campaigns to activate.
- Multi-channel users engage across platforms for better reach.

---

## 🚀 Future Improvements

- Add real-time dashboard with detailed cluster analytics.
- Incorporate additional features like customer lifetime value.
- Build recommendation engine based on segment profiles.
- Integrate with business CRM for seamless marketing automation.

---

## 📬 Let's Connect

This project is part of my data science journey.  
Feel free to reach out on [LinkedIn](https://www.linkedin.com/in/gitesh-garg) for feedback or collaboration.

---

🏷️ Tags: #CustomerSegmentation #KMeans #MachineLearning #Streamlit #DataScience
