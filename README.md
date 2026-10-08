# Integrated ML System for AML Detection & Visual Analytics
**MSc Data Science Thesis | University of Greenwich**

## 🎯 The Challenge
Financial institutions face a "needle in a haystack" problem: Fraudulent transactions represent < 0.2% of data. Traditional rule-based systems generate high false-positive rates and fail to catch evolving laundering patterns.

## 🛠️ The Solution
A hybrid machine learning pipeline that combines:
1. **Unsupervised Learning (K-Means):** To discover hidden patterns and group transactions into risk clusters.
2. **Supervised Learning (XGBoost & Random Forest):** To classify transactions with high precision.
3. **Class Balancing (SMOTE):** To address the 0.17% fraud disparity.

## 📈 Key Results
- **High AUC-ROC:** Outperformed base models through progressive enhancement.
- **Pattern Segregation:** Successfully isolated 90% of fraud cases into a specific high-risk cluster using PCA-reduced feature sets.
- **Explainability:** Integrated an interactive dashboard to make "black-box" decisions interpretable for compliance officers.

## 💻 Tech Stack
- **Python:** Scikit-Learn, XGBoost, Pandas, NumPy
- **Visuals:** Streamlit, Seaborn, Matplotlib
- **Research:** Dimensionality Reduction (PCA), Synthetic Oversampling (SMOTE)

## 🗂️ Project Structure
```
├── src/
│   ├── data.py          # loading, scaling, stratified split, SMOTE
│   ├── clustering.py    # K-means cluster feature (k=2)
│   └── models.py        # base RF, enhanced RF, XGBoost, evaluation
├── train.py             # runs the full pipeline and saves models + metrics
├── requirements.txt     # pinned dependencies (Python 3.11)
└── Anti-Money MoneyLaundering Dashbaord and Algorithm Project Source Code.ipynb
```

## ▶️ Running the Pipeline
1. Download `creditcard.csv` from Kaggle and place it at `data/creditcard.csv`.
2. Install dependencies with Python 3.11:
   ```bash
   pip install -r requirements.txt
   ```
3. Train and save the models:
   ```bash
   python train.py --data data/creditcard.csv --out models
   ```
   This writes the fitted scaler, K-means model, Random Forest and XGBoost models, and `metrics.json` to `models/`.

The notebook remains the record of the exploratory analysis and results.
