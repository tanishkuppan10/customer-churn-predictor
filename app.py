import streamlit as st
import psycopg2
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {background-color: #f0f2f6;}
    .risk-high {color: red; font-size: 24px; font-weight: bold;}
    .risk-medium {color: orange; font-size: 24px; font-weight: bold;}
    .risk-low {color: green; font-size: 24px; font-weight: bold;}
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    return pickle.load(open("model.pkl", "rb"))

# Fetch data from PostgreSQL
@st.cache_data
def load_data():
    conn = psycopg2.connect(
        host="localhost",
        database="churndb",
        user="postgres",
        password="root"
    )
    df = pd.read_sql("SELECT * FROM customers", conn)
    conn.close()
    return df

model = load_model()
df = load_data()
df['Churn_num'] = df['churn'].map({'Yes': 1, 'No': 0})

# Sidebar navigation
st.sidebar.title("📊 Churn Predictor")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "🏠 Dashboard",
    "🔍 Single Prediction"
])

# ─────────────────────────────────────────
# PAGE 1 — DASHBOARD
# ─────────────────────────────────────────
if page == "🏠 Dashboard":
    st.title("Customer Churn Dashboard")

    # Top metrics
    total = len(df)
    churned = int(df['Churn_num'].sum())
    churn_rate = (churned / total) * 100
    avg_monthly = df['monthlycharges'].mean()
    avg_tenure = df['tenure'].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{total:,}")
    col2.metric("Churned Customers", f"{churned:,}")
    col3.metric("Churn Rate", f"{churn_rate:.1f}%")
    col4.metric("Avg Monthly Charge", f"${avg_monthly:.0f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        counts = df['churn'].value_counts()
        colors = ['#2ecc71', '#e74c3c']
        ax.pie(counts.values, labels=['No Churn', 'Churned'],
               colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title("Customer Churn Rate")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Tenure vs Churn")
        fig, ax = plt.subplots(figsize=(5, 4))
        df[df['Churn_num'] == 0]['tenure'].hist(
            bins=20, alpha=0.7, label='No Churn', color='#2ecc71', ax=ax)
        df[df['Churn_num'] == 1]['tenure'].hist(
            bins=20, alpha=0.7, label='Churned', color='#e74c3c', ax=ax)
        ax.set_xlabel("Tenure (months)")
        ax.set_ylabel("Count")
        ax.legend()
        ax.set_title("Tenure Distribution by Churn")
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("Monthly Charges by Churn Status")
    fig, ax = plt.subplots(figsize=(10, 3))
    df.boxplot(column='monthlycharges', by='churn', ax=ax)
    ax.set_xlabel("Churn")
    ax.set_ylabel("Monthly Charges ($)")
    plt.suptitle("")
    st.pyplot(fig)
    plt.close()

# ─────────────────────────────────────────
# PAGE 2 — SINGLE PREDICTION
# ─────────────────────────────────────────
elif page == "🔍 Single Prediction":
    st.title("Single Customer Churn Prediction")
    st.subheader("Enter Customer Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0)

    with col2:
        total = st.number_input("Total Charges ($)", 0.0, 10000.0, 500.0)
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])

    with col3:
        gender = st.selectbox("Gender", ["Male", "Female"])

    senior_val = 1 if senior == "Yes" else 0
    gender_val = 1 if gender == "Male" else 0

    if st.button("Predict Churn Risk", type="primary"):
        features = [[gender_val, senior_val, tenure, monthly, total]]
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1] * 100

        st.markdown("---")
        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            if probability >= 70:
                st.error("⚠️ HIGH RISK of Churn")
                st.markdown(
                    f"<p class='risk-high'>Risk Score: {probability:.1f}%</p>",
                    unsafe_allow_html=True)
                st.write("This customer is very likely to leave. Consider offering a discount or loyalty reward.")
            elif probability >= 40:
                st.warning("⚡ MEDIUM RISK of Churn")
                st.markdown(
                    f"<p class='risk-medium'>Risk Score: {probability:.1f}%</p>",
                    unsafe_allow_html=True)
                st.write("This customer may leave. Monitor their activity closely.")
            else:
                st.success("✅ LOW RISK of Churn")
                st.markdown(
                    f"<p class='risk-low'>Risk Score: {probability:.1f}%</p>",
                    unsafe_allow_html=True)
                st.write("This customer is likely to stay.")

        with col2:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.barh(["Churn Risk"], [probability],
                    color='#e74c3c' if probability >= 70
                    else '#f39c12' if probability >= 40
                    else '#2ecc71')
            ax.barh(["Churn Risk"], [100 - probability],
                    left=[probability], color='#ecf0f1')
            ax.set_xlim(0, 100)
            ax.set_xlabel("Risk %")
            ax.set_title(f"Churn Probability: {probability:.1f}%")
            st.pyplot(fig)
            plt.close()