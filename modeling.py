# Proses Modeling

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import xgboost as xgb

def train_logistic_regression(X_train, y_train):
    print("Proses Training Logistic Regression...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    print("Proses Logistic Regression Selesai")
    return model

def train_random_forest(X_train, y_train):
    print("Proses Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=300, 
        max_depth=12, 
        random_state=42, 
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    print("Proses Random Forest Selesai")
    return model


def train_xgboost(X_train, y_train):
    print("Proses Training XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        random_state=42,
        eval_metric='auc',
        use_label_encoder=False
    )
    model.fit(X_train, y_train)
    print("Proses XGBoost Selesai")
    return model