# File Utama Projek

from src.utils import load_data
from src.preprocessing import handle_missing_values, encode_categorical
from src.feature_engineering import buat_feature_baru
from src.modeling import train_logistic_regression, train_random_forest, train_xgboost
from src.evaluation import evaluate_model
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd

print("Memulai Final Project Prediksi Skor Kredit Home Credit Indonesia...\n")

# 1. Proses mengambil dataset.
df = load_data('application_train.csv')

# 2. Proses melakukan preprocessing.
df = handle_missing_values(df)
df = encode_categorical(df)

# 3. Proses melakukan feature engineering.
df = buat_feature_baru(df)

# 4. Proses pemilihan fitur dan target pada projek.
from src.config import FITUR_UTAMA
fitur = FITUR_UTAMA

X = df[fitur]
y = df['TARGET']

print(f"\nJumlah Fitur yang Digunakan: {len(fitur)}")

# 5. Proses melakukan split train dan test.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 6. Proses melakukan handling pada data yang imbalance dengan SMOTE.
print("Menangani Data Imbalance dengan SMOTE...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 7. Proses training model.
print("\n=== Mulai Training Model ===")
logreg = train_logistic_regression(X_train_res, y_train_res)
rf = train_random_forest(X_train_res, y_train_res)
xgb_model = train_xgboost(X_train_res, y_train_res) 

# 8. Hasil dan evaluasi model.
print("\n=== Evaluasi Model ===")
evaluate_model(logreg, X_test, y_test, "Logistic Regression")
evaluate_model(rf, X_test, y_test, "Random Forest")
evaluate_model(xgb_model, X_test, y_test, "XGBoost")