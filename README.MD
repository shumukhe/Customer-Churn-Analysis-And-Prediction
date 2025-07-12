# 🔍 Customer Churn Analysis & Prediction

This project explores historical customer data, uncovers churn patterns, trains machine learning models to predict churn risk, and proposes strategies to improve retention.

[🔗 Live Demo](https://customer-churn-analysis-and-prediction-koz7xxyydmvijcuges6mw8.streamlit.app/)

---

## 🚀 Project Overview

Customer churn (attrition) is a critical metric for subscription-based or service-oriented businesses. This project performs:

- **Exploratory Data Analysis (EDA)**: Understand factors associated with customer churn via visualizations and statistics.
- **Preprocessing & Feature Engineering**: Handle missing data, create meaningful features, balance the dataset.
- **Machine Learning Modeling**: Train models like Logistic Regression, Random Forest, XGBoost to predict churn probability.
- **Model Evaluation**: Evaluate accuracy, precision, recall, F1-score, and ROC–AUC for comparison.
- **Recommendations**: Identify key drivers of churn and propose actionable business strategies.

---

## 📂 Project Structure

```
Customer-Churn-Analysis-And-Prediction/
│
├── data/
│   ├── first_data.csv        # Initial 200 samples
│   ├── second_data.xlsx      # Extended with 2000 samples
│   └── final_data.csv        # Combined dataset
│
├── models/
│   ├── preprocessor.pkl      # Data preprocessing pipeline
│   ├── randomforest_model.pkl
│   └── best_model.pkl        # Optimized model from GridSearch
│
├── EDA.ipynb                 # Exploratory data analysis report
├── model_training_manual.ipynb
├── model_training_automation.py
├── preprocessing.py         # Data cleaning & preprocessing script
├── recommendation.py        # Recommender based on churned users
├── model_api.py             # Flask API for inference
├── requirements.txt
└── README.md
```

---

## 🛠️ Technologies & Tools

- Python (Pandas, NumPy, sklearn, XGBoost)
- Jupyter Notebooks for EDA & modeling walkthroughs
- scikit-learn, XGBoost for model training
- Joblib for saving pipeline and model
- Flask API (`model_api.py`) for real-time inference
- Plotly/Seaborn for visualizations

---

## 🧪 How to Run the Project

### ✅ Setup Environment

```bash
git clone https://github.com/shumukhe/Customer-Churn-Analysis-And-Prediction.git
cd Customer-Churn-Analysis-And-Prediction
conda create -n churnenv python=3.10 -y
conda activate churnenv
pip install -r requirements.txt
```

### 🔄 Prepare Data & Train Model

```bash
python preprocessing.py --input final_data.csv --output processed_data.csv
python model_training_automation.py --data processed_data.csv --save-model models/best_model.pkl
```

Alternatively, explore ideas and manual runs via the provided Jupyter notebooks.

### 🚀 Run the API

```bash
python model_api.py
```

Sample POST request:
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H 'Content-Type: application/json' \
  -d '{"Age":30,"Tenure":12,"Gender":"Female", ... }'
```

---

## 📈 Sample Results

The best model (e.g., Random Forest / XGBoost) achieved:
- **Accuracy**: ~82%
- **ROC–AUC**: ~0.85
- Detailed results and charts are in `model_training_manual.ipynb`.

---

## 💡 Insights & Recommendations

- The most influential churn factors: **monthly charges**, **tenure**, **payment method**
- Reduce churn by offering **early discounts**, tailored **loyalty plans**, and improved **customer support**
- Use API in production to flag high-risk accounts for proactive retention efforts

---

## 🌟 Future Enhancements

- Add model interpretability (SHAP values, feature importance plots)
- Build interactive dashboards with Streamlit or Dash
- Integrate more features (e.g., customer support logs, demographics)
- Deploy model and Flask API to Heroku, AWS, or GCP

---

## 📞 Contact

For suggestions or issues, feel free to open an issue or reach out via GitHub.

---

### 📜 License

This project is licensed under the **MIT License** — see `LICENSE` for details.

