# Proses Model Evaluation

from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test, nama_model):

    # Proses mengevaluasi model yang dipakai.
    print(f"\n=== Evaluasi {nama_model} ===")
    
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    
    print(classification_report(y_test, pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, prob):.4f}")
    
    # Proses mengerjakan confusion matrix.
    plt.figure(figsize=(6,4))
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {nama_model}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Proses menyimpan confussion matrix
    from src.utils import simpan_grafik
    simpan_grafik(f'confusion_matrix_{nama_model.lower().replace(" ", "_")}.png')
    
    plt.close('all')   
    
    print(f"Confusion Matrix disimpan ke folder output/figures/\n")
    
    return prob