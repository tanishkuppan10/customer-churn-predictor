import psycopg2
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

conn = psycopg2.connect(
    host="localhost",
    database="churndb",
    user="postgres",
    password="root"
)

df = pd.read_sql("SELECT * FROM customers", conn)
conn.close()

# Encoding the data
df['Churn'] = df['churn'].map({'Yes': 1, 'No': 0})
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

# feature selection
features = ['gender', 'seniorcitizen', 'tenure', 
            'monthlycharges', 'totalcharges']
X = df[features]
y = df['Churn']

# test_train_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model comparison
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"{name}: {accuracy_score(y_test, pred):.2f}")
    print(classification_report(y_test, pred))

# Save best model
best_model = RandomForestClassifier()
best_model.fit(X_train, y_train)
pickle.dump(best_model, open("model.pkl", "wb"))
print("Model saved")




""" output:
 df = pd.read_sql("SELECT * FROM customers", conn)
Logistic Regression: 0.80
              precision    recall  f1-score   support

           0       0.83      0.92      0.87      3113
           1       0.67      0.47      0.55      1107

    accuracy                           0.80      4220
   macro avg       0.75      0.69      0.71      4220
weighted avg       0.79      0.80      0.79      4220

Random Forest: 0.98
              precision    recall  f1-score   support

           0       0.98      0.99      0.99      3113
           1       0.97      0.95      0.96      1107

    accuracy                           0.98      4220
   macro avg       0.98      0.97      0.97      4220
weighted avg       0.98      0.98      0.98      4220

Model saved """
