import os
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import joblib
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class LogTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log1p(X)

# Load model
pipeline = joblib.load("final_churn_pipeline.pkl")

# Load test data

X_test = joblib.load("X_test.pkl")



st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="💼",
    layout="wide"
)

st.title("Customer Churn Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    credit_score = st.number_input("Credit Score:", step=1, value=600)
    age = st.number_input("Age:", step=1, value=30)
    gender = st.selectbox("Gender:", options=X_test["Gender"].unique())
    is_active_member = st.selectbox("Is Active Member:", options=X_test["IsActiveMember"].unique())

with col2:
    geography = st.selectbox("Geography:", options=X_test["Geography"].unique())
    tenure = st.selectbox("Tenure (Years):", options=sorted(X_test["Tenure_x"].unique()))
    has_cr_card = st.selectbox("Has Credit Card:", options=X_test["HasCrCard"].unique())
    zero_balance = st.selectbox("Zero Balance Account:", options=X_test["Zero_Balnce"].unique())

with col3:
    estimated_salary = st.number_input("Estimated Salary:", step=1000.0, value=50000.0)
    balance = st.number_input("Account Balance:", step=1000.0, value=0.0)
    num_of_products = st.selectbox("Number of Products:", options=sorted(X_test["NumOfProducts"].unique()))

input_df = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure_x': [tenure],
    'EstimatedSalary': [estimated_salary],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Zero_Balnce': [zero_balance]
})

if st.button("Predict Churn"):
    
   

    # Directly predict using the full pipeline
    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"🚨 Customer is likely to churn! Probability: {probability:.2%}")
    else:
        st.success(f"✅ Customer is likely to stay. Probability: {probability:.2%}")