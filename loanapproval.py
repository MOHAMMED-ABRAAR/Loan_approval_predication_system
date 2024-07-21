import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import shap
from flask import Flask, request, jsonify

# Load data
data = pd.read_csv('loan_prediction.csv')

# Basic data info
print(data.info())
print(data.describe())

# Exploratory Data Analysis
sns.countplot(data['Loan_Status'])
plt.xlabel('Loan Status', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Distribution of Loan Status', fontsize=15)
plt.xticks(rotation=0)
plt.show()

# Normal bar graph for Loan_Status
loan_status_counts = data['Loan_Status'].value_counts()
loan_status_counts.plot(kind='bar')
plt.xlabel('Loan Status', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Distribution of Loan Status', fontsize=15)
plt.xticks(rotation=0)
plt.show()

# Handle missing values
imputer = SimpleImputer(strategy='mean')
data['LoanAmount'] = imputer.fit_transform(data[['LoanAmount']])

# Encode categorical variables
label_encoder = LabelEncoder()
categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']

for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])

# Feature scaling
scaler = StandardScaler()
data[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']] = scaler.fit_transform(data[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']])

# Split data into train and test
X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Engineering
X_train['Total_Income'] = X_train['ApplicantIncome'] + X_train['CoapplicantIncome']
X_test['Total_Income'] = X_test['ApplicantIncome'] + X_test['CoapplicantIncome']

# Drop original income columns
X_train.drop(['ApplicantIncome', 'CoapplicantIncome'], axis=1, inplace=True)
X_test.drop(['ApplicantIncome', 'CoapplicantIncome'], axis=1, inplace=True)

# Train Logistic Regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)

# Train Random Forest model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Evaluation
print("Logistic Regression:")
print(classification_report(y_test, y_pred_logreg))
print("Accuracy:", accuracy_score(y_test, y_pred_logreg))

print("Random Forest:")
print(classification_report(y_test, y_pred_rf))
print("Accuracy:", accuracy_score(y_test, y_pred_rf))

# Model Interpretation
explainer = shap.Explainer(rf, X_train)
shap_values = explainer(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test)

# Deployment using Flask
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = rf.predict(pd.DataFrame([data]))
    return jsonify({'loan_status': 'Approved' if prediction[0] == 1 else 'Rejected'})

if __name__ == '__main__':
    app.run(debug=True)
