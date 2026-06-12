import streamlit as st
import psycopg2
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load model from the train_model
model = pickle.load(open("model.pkl", "rb"))

# Fetch data pgadmin 
conn = psycopg2.connect(
    host="localhost",
    database="churndb",
    user="postgres",
    password="root"
)
df = pd.read_sql("SELECT * FROM customers", conn)
conn.close()

#  creation of Dashboard
st.title("Customer Churn Predictor")

# Show churn distribution
st.subheader("Churn Distribution")
fig, ax = plt.subplots()
df['churn'].value_counts().plot(kind='bar', ax=ax, color=['green','red'])
st.pyplot(fig)

# Prediction form
st.subheader("Predict Single Customer")
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
total = st.number_input("Total Charges", 0.0, 10000.0, 500.0)
senior = st.selectbox("Senior Citizen", [0, 1])
gender = st.selectbox("Gender", [0, 1])

if st.button("Predict Churn"):
    pred = model.predict([[gender, senior, tenure, monthly, total]])
    if pred[0] == 1:
        st.error("This customer is likely to CHURN")
    else:
        st.success("This customer will likely STAY")