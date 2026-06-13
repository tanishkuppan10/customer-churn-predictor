# Customer Churn Predictor 📊

## What is Customer Churn?
Customer churn means when a customer stops using a company's 
service or product. For example, if you subscribed to a streaming 
platform and then cancelled your subscription — that is churn.

For businesses, losing customers is expensive. It costs 5x more 
to acquire a new customer than to retain an existing one. That is 
why companies use Machine Learning to predict which customers are 
likely to leave — so they can take action before it happens, like 
offering a discount or a loyalty reward.

This project builds a complete end-to-end churn prediction system 
for a telecom company using real-world data.

---

## 🛠️ Tech Stack
| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PostgreSQL | Database to store customer data |
| psycopg2 | Connect Python to PostgreSQL |
| Pandas | Data cleaning and manipulation |
| Scikit-learn | Training and evaluating ML models |
| Matplotlib | Data visualization |
| Streamlit | Interactive web dashboard |

---

## 📁 Project Structure

---

## 📄 File Breakdown

### `load_data.py`
Reads the Telco Customer Churn CSV dataset, cleans it by 
handling missing values in the TotalCharges column, connects 
to a PostgreSQL database called churn_db, creates a customers 
table, and inserts all 7,043 records into it row by row.

### `train_model.py`
Fetches customer data directly from PostgreSQL using psycopg2, 
prepares features by encoding categorical columns like gender, 
splits data into training and testing sets, trains and compares 
two ML models — Logistic Regression and Random Forest Classifier 
— evaluates both using accuracy score and classification report, 
and saves the best performing model as model.pkl using pickle.

### `app.py`
A two-page interactive Streamlit dashboard:

**Page 1 — Dashboard:**
Displays key business metrics including total customers, 
churned customers, churn rate, and average monthly charge. 
Shows three charts — a pie chart of churn distribution, 
a histogram comparing tenure of churned vs retained customers, 
and a box plot of monthly charges by churn status.

**Page 2 — Single Prediction:**
Takes customer inputs including gender, senior citizen status, 
tenure, monthly charges, and total charges. Runs the trained 
Random Forest model and outputs a churn risk score with three 
levels — Low Risk (green), Medium Risk (orange), and High Risk 
(red) — along with a visual risk bar chart and a business 
recommendation.

---

## 🚀 How to Run This Project

**1. Clone the repository**
```bash
git clone https://github.com/tanishkuppan10/customer-churn-predictor.git
cd customer-churn-predictor
```

**2. Create a virtual environment**
```bash
python -m venv venv
source venv/Scripts/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up PostgreSQL**
- Install PostgreSQL and create a database called `churndb`
- Update the password in `load_data.py`, `train_model.py`, 
  and `app.py`

**5. Download the dataset**
- Download Telco Customer Churn dataset from Kaggle
- Rename it to `telco_churn.csv` and place it in the 
  project folder

**6. Load data into PostgreSQL**
```bash
python load_data.py
```

**7. Train the model**
```bash
python train_model.py
```

**8. Run the app**
```bash
streamlit run app.py
```

---

## 📊 Dataset
- **Name:** Telco Customer Churn Dataset
- **Source:** Kaggle — IBM Sample Dataset
- **Records:** 7,043 customers
- **Link:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn

This dataset contains information about customers of a telecom 
company — similar to networks like Airtel or Jio — and whether 
they cancelled their subscription or not.

### Key Features Explained
| Feature | Description |
|---|---|
| tenure | How many months the customer has been with the company. A customer with tenure 1 just joined, tenure 72 has been with the company for 6 years. |
| MonthlyCharges | How much the customer pays every month for their subscription plan. |
| TotalCharges | Total amount the customer has paid since joining. Calculated as tenure × MonthlyCharges approximately. |
| Churn | Whether the customer left the company. Yes means they cancelled, No means they are still a customer. |
| gender | Whether the customer is Male or Female. |
| SeniorCitizen | Whether the customer is a senior citizen. 1 means yes, 0 means no. |

### Why These Features Matter
A customer who just joined (low tenure) and pays a high monthly 
charge is more likely to leave if they feel the service is not 
worth the price. On the other hand, a customer who has been with 
the company for years (high tenure) is less likely to churn 
because they are already comfortable with the service.

---

## 🧠 ML Models Compared
| Model | Purpose |
|---|---|
| Logistic Regression | Baseline linear model |
| Random Forest | Final model — better accuracy |

Random Forest was selected as the final model due to its 
superior accuracy and ability to handle non-linear patterns 
in customer behaviour.

---

## 💡 Key Insight
Customers with short tenure and high monthly charges are at 
the highest risk of churning. Businesses should target these 
customers with retention offers before they leave.

---

## 👨‍💻 Author
**Tanishk Uppan**  
3rd Year Engineering Student  
[GitHub](https://github.com/tanishkuppan10) | 
[LinkedIn](https://www.linkedin.com/in/tanish-kuppan-56ab2b33b/)