<div align="center">

# 💳 Credit Card Fraud Detection

**A machine learning web app that flags fraudulent credit card transactions in real time.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?logo=scikitlearn&logoColor=white)
![Model](https://img.shields.io/badge/Model-Logistic%20Regression-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

</div>

---

## 📌 Overview

Credit card fraud costs banks and customers billions every year. This project trains a **Logistic Regression** classifier on the public credit card fraud dataset and wraps it in an interactive **Streamlit** app, so anyone can check a single transaction or score a whole CSV file in seconds.

The dataset is **highly imbalanced**: only **492 of 284,807** transactions (≈ 0.17%) are fraud. Because of this, the project evaluates the model with **precision and recall** instead of accuracy.

## 📑 Table of Contents
- [Features](#-features)
- [Demo](#-demo)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Installation & Usage](#-installation--usage)
- [How to Use the App](#-how-to-use-the-app)
- [Troubleshooting](#-troubleshooting)
- [Future Improvements](#-future-improvements)
- [Disclaimer](#-disclaimer)

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Single transaction check** | Pick a built-in sample or enter values manually and get an instant fraud probability. |
| 📂 **Batch CSV scoring** | Upload a file with many transactions, see flagged rows sorted by risk, and download the results. |
| 🎚️ **Adjustable threshold** | Slide between catching more fraud and avoiding false alarms. |
| 📊 **Built-in evaluation** | If your CSV has a `Class` column, the app shows precision, recall, frauds caught, false alarms and a confusion matrix. |
| ⚡ **Fast** | Model is loaded once and cached. |

## 🖼️ Demo

> Add screenshots of your app here, for example:
>
> `![Single transaction](screenshots/single.png)`
> `![Batch results](screenshots/batch.png)`

## 🛠️ Tech Stack

- **Language:** Python
- **Web app:** Streamlit
- **ML:** scikit-learn (Logistic Regression)
- **Data:** Pandas, NumPy
- **Model storage:** Joblib

## 📊 Dataset

Source: [Credit Card Fraud Detection (Kaggle, ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- **284,807** transactions by European cardholders (two days, September 2013)
- **30 input features:** `Time`, `Amount`, and `V1`–`V28` (anonymised PCA components)
- **Target:** `Class` → `1` = fraud, `0` = legitimate

> The full 150 MB `creditcard.csv` is **not** included in this repo. A 500-row `sample_transactions.csv` is provided for demos.

## 📈 Model Performance

Evaluated on the full dataset at the default threshold of 0.5:

| Metric (fraud class) | Score |
|---|---|
| Precision | **71.6%** |
| Recall | **61.6%** |
| F1-score | **0.66** |
| ROC-AUC | **0.916** |

**Confusion matrix (threshold 0.5):**

| | Predicted Legit | Predicted Fraud |
|---|---|---|
| **Actual Legit** | 284,195 | 120 |
| **Actual Fraud** | 189 | 303 |

**Effect of the threshold:**

| Threshold | Frauds caught (of 492) | False alarms |
|---|---|---|
| 0.3 | 324 | 231 |
| 0.5 | 303 | 120 |
| 0.7 | 292 | 80 |
| 0.9 | 258 | 56 |

A lower threshold catches more fraud but blocks more genuine payments, so the right value depends on the cost of each mistake.

## 📁 Project Structure

```
credit-card-fraud/
├── app.py                    # Streamlit application
├── model.pkl                 # Trained Logistic Regression model
├── sample_transactions.csv   # 500-row demo file (60 frauds)
├── requirements.txt          # Python dependencies
└── README.md
```

## 🚀 Installation & Usage

**1. Clone the repository**
```bash
git clone https://github.com/<your-username>/credit-card-fraud.git
cd credit-card-fraud
```

**2. (Optional) Create a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the app**
```bash
python -m streamlit run app.py
```
Then open **http://localhost:8501** in your browser.

## 🧭 How to Use the App

1. **Set the threshold** in the sidebar (default 0.50).
2. **Single transaction tab:** choose a sample (legit or fraud) or enter your own values, then click **Check transaction**.
3. **Batch tab:** upload a CSV with the columns `Time, V1 … V28, Amount` (optionally `Class`) and review or download the results.

## 🩺 Troubleshooting

| Problem | Fix |
|---|---|
| `streamlit` is not recognised | Run `python -m streamlit run app.py` |
| `FileNotFoundError: model.pkl` | Keep `app.py` and `model.pkl` in the same folder |
| Version warning when loading the model | Install `scikit-learn==1.6.1`, the version the model was saved with |
| Batch upload says "Missing columns" | Your CSV needs `Time, V1–V28, Amount` |

## 🔮 Future Improvements

- Try class-imbalance techniques (SMOTE, class weights) and stronger models such as Random Forest or XGBoost
- Add a precision-recall curve and feature-importance charts
- Deploy on Streamlit Community Cloud
- Add an API endpoint for real-time scoring

## ⚠️ Disclaimer

This is an educational project and is **not** intended for real-world financial decision making.

## 👤 Author

**Your Name** · [GitHub](https://github.com/studyhub-komal) · [LinkedIn](https://linkedin.com/in/<your-profile>)

⭐ If you found this project useful, consider giving it a star!
