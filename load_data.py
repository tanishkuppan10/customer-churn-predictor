import psycopg2
import pandas as pd

# Load CSV
df = pd.read_csv("telco_churn.csv")

# Fix TotalCharges column — it has some empty strings

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# Connect to PostgreSQL


conn = psycopg2.connect(
    host="localhost",
    database="churndb",
    user="postgres",
    password="root"
)
cur = conn.cursor()

# Create table
cur.execute("""
    CREATE TABLE IF NOT EXISTS customers (
        customerID VARCHAR(50),
        gender VARCHAR(10),
        SeniorCitizen INT,
        tenure INT,
        MonthlyCharges FLOAT,
        TotalCharges FLOAT,
        Churn VARCHAR(5)
    )
""")

# Insert data row by row

for _, row in df.iterrows():
    cur.execute("""
        INSERT INTO customers 
        (customerID, gender, SeniorCitizen, tenure, 
        MonthlyCharges, TotalCharges, Churn)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (row['customerID'], row['gender'], row['SeniorCitizen'],
          row['tenure'], row['MonthlyCharges'], 
          row['TotalCharges'], row['Churn']))

conn.commit()
cur.close()
conn.close()
print("Data loaded successfully into PostgreSQL")