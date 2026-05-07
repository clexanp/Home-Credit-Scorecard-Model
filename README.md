# Home Credit Scorecard Model

## Deskripsi Singkat

Home Credit Scorecard Model adalah sistem prediksi untuk mengevaluasi layak tidaknya pengajuan kredit nasabah berdasarkan data profil dan riwayat finansial mereka. Proyek ini mengembangkan model machine learning yang dapat memprediksi kemungkinan pembayaran kredit dengan akurasi tinggi. Model ini dirancang untuk membantu institusi keuangan dalam pengambilan keputusan supaya pemberian kredit menjadi lebih objektif dan komprehensif.

## Fitur Utama

1. **Preprocessing Data Komprehensif**
   - Penanganan missing values dengan strategi yang tepat
   - Encoding variabel kategorikal
   - Normalisasi dan scaling data numerik

2. **Feature Engineering**
   - Pembuatan fitur turunan berdasarkan domain knowledge
   - Seleksi fitur yang relevan untuk meningkatkan performa model
   - Analisis hubungan antar variabel

3. **Multiple Model Approach**
   - Logistic Regression untuk interpretabilitas tinggi
   - Random Forest untuk performa yang robust
   - XGBoost untuk akurasi maksimal

4. **Handling Class Imbalance**
   - Implementasi SMOTE (Synthetic Minority Over-sampling Technique)
   - Strategi sampling yang sesuai dengan distribusi data
   - Penggunaan metrics yang tepat untuk evaluasi (precision, recall, F1-score)

5. **Model Evaluation**
   - Confusion Matrix dan Classification Report
   - ROC-AUC Score untuk mengukur diskriminasi model
   - Cross-validation untuk validasi model yang robust

---

## Dataset

### Sumber dan Deskripsi
Dataset yang digunakan adalah data aplikasi kredit dari Home Credit dengan informasi lengkap mengenai profil nasabah dan hasil pemberian kredit. Dataset ini terdiri dari:

- **application_train.csv**: Data pelatihan dengan lebih dari 300,000 record
  - Jumlah fitur: 121+ kolom
  - Target variabel: TARGET (0 = Pembayaran lancar, 1 = Default)
  - Tipe data: Mix dari numerik, kategorikal, dan datetime

- **bureau.csv**: Data riwayat kredit sebelumnya dari biro kredit
  - Informasi tentang kredit sebelumnya di institusi lain
  - Dapat digunakan untuk feature engineering lanjutan

### Struktur Data Utama
| Kolom | Tipe | Deskripsi |
|-------|------|-----------|
| AMT_CREDIT | Numerik | Jumlah kredit yang diajukan |
| AMT_INCOME_TOTAL | Numerik | Total pendapatan tahunan |
| DAYS_BIRTH | Numerik | Usia nasabah (dalam hari) |
| CODE_GENDER | Kategorikal | Jenis kelamin |
| NAME_EDUCATION_TYPE | Kategorikal | Tipe pendidikan |
| TARGET | Kategorikal (0/1) | Status pembayaran |

### Akses Dataset
Dataset dapat diakses dari:
- Platform Kaggle: Home Credit Default Risk competition
- Folder `data/` dalam repository ini
- Catatan: File CSV berukuran besar (100+MB) dan tidak disertakan dalam repository

---

## Instalasi dan Requirements

### Persyaratan Sistem
- Python 3.8 atau lebih baru
- Disk space minimal 2GB untuk dataset
- RAM minimal 8GB untuk proses training

### Langkah-langkah Setup

1. **Clone atau download repository**
   ```bash
   cd "Dokumen Data Scientist PBI Rakamin/Home Credit Scorecard Model"
   ```

2. **Buat virtual environment (opsional tapi disarankan)**
   ```bash
   # Untuk Windows
   python -m venv venv
   venv\Scripts\activate
   
   # Untuk Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies
Proyek ini menggunakan library berikut:
- **pandas** - Manipulasi dan analisis data
- **numpy** - Komputasi numerik
- **scikit-learn** - Machine learning models dan utilities
- **imbalanced-learn** - SMOTE untuk handling class imbalance
- **matplotlib** - Visualisasi data
- **seaborn** - Statistical visualization

Lihat `requirements.txt` untuk versi spesifik dari setiap library.

---

## Cara Penggunaan

### Quick Start

1. **Pastikan semua dependencies sudah terinstall**
   ```bash
   pip install -r requirements.txt
   ```

2. **Letakkan dataset di folder `data/`**
   - Pastikan file `application_train.csv` berada di `data/application_train.csv`
   - File ini tidak termasuk dalam repository, perlu diunduh terpisah dari Kaggle

3. **Jalankan pipeline lengkap**
   ```bash
   python main.py
   ```

   Script ini akan melakukan:
   - Loading dataset
   - Preprocessing dan cleaning data
   - Feature engineering
   - Train-test split dengan ratio 80-20
   - Handling class imbalance menggunakan SMOTE
   - Training 3 model (Logistic Regression, Random Forest, XGBoost)
   - Evaluasi dan perbandingan performa model

### Workflow Detail

#### 1. Data Loading dan Preprocessing
```python
from src.utils import load_data
from src.preprocessing import handle_missing_values, encode_categorical

df = load_data('application_train.csv')
df = handle_missing_values(df)
df = encode_categorical(df)
```

#### 2. Feature Engineering
```python
from src.feature_engineering import buat_feature_baru

df = buat_feature_baru(df)
```

#### 3. Persiapan Data untuk Modeling
```python
from src.config import FITUR_UTAMA
from sklearn.model_selection import train_test_split

fitur = FITUR_UTAMA
X = df[fitur]
y = df['TARGET']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
```

#### 4. Handling Class Imbalance
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
```

#### 5. Training dan Evaluasi Model
```python
from src.modeling import train_logistic_regression, train_random_forest, train_xgboost
from src.evaluation import evaluate_model

logreg = train_logistic_regression(X_train_res, y_train_res)
rf = train_random_forest(X_train_res, y_train_res)
xgb_model = train_xgboost(X_train_res, y_train_res)

evaluate_model(logreg, X_test, y_test, model_name="Logistic Regression")
evaluate_model(rf, X_test, y_test, model_name="Random Forest")
evaluate_model(xgb_model, X_test, y_test, model_name="XGBoost")
```

### Output yang Dihasilkan
- Model yang sudah dilatih disimpan di folder `output/models/`
- Visualisasi performa model (confusion matrix, ROC curve)
- Laporan evaluasi dengan metrik lengkap (precision, recall, F1-score, AUC)
- Feature importance dari model tree-based

---

## Struktur Direktori

```
Home Credit Scorecard Model/
│
├── main.py                    # Script utama untuk menjalankan pipeline lengkap
├── config.py                  # Konfigurasi global (fitur, hyperparameter)
├── requirements.txt           # List dependencies yang diperlukan
├── README.md                  # Dokumentasi proyek (file ini)
├── .gitignore                 # File yang diabaikan Git
│
├── src/                       # Source code utama
│   ├── __init__.py
│   ├── config.py              # Konfigurasi project (fitur utama, path)
│   ├── utils.py               # Fungsi utility untuk loading dan saving data
│   ├── preprocessing.py       # Preprocessing: handling missing values, encoding
│   ├── feature_engineering.py # Feature engineering: pembuatan fitur baru
│   ├── modeling.py            # Training berbagai model ML
│   └── evaluation.py          # Evaluasi dan metrik performa model
│
├── data/                      # Folder data (tidak dalam repo)
│   ├── application_train.csv  # Dataset utama training
│   └── bureau.csv             # Data bureau (additional data)
│
├── output/                    # Output folder
│   ├── models/                # Model yang sudah dilatih (.pkl, .joblib)
│   └── figures/               # Visualisasi dan plot hasil
│
├── notebooks/                 # Jupyter notebooks untuk eksplorasi
│   └── Proses_EDA.ipynb       # Exploratory Data Analysis
│
└── Hasil Output Grafik/       # Output grafik visualisasi (optional)
```

### Penjelasan File Utama

- **main.py**: Entry point utama yang menjalankan seluruh pipeline dari data loading hingga model evaluation
- **src/utils.py**: Berisi fungsi bantuan untuk loading data, preprocessing, dan saving model
- **src/preprocessing.py**: Modul untuk cleansing data termasuk handling missing values dan encoding
- **src/feature_engineering.py**: Modul untuk pembuatan dan seleksi fitur yang lebih powerful
- **src/modeling.py**: Modul yang berisi training logic untuk Logistic Regression, Random Forest, dan XGBoost
- **src/evaluation.py**: Modul untuk evaluasi model dengan berbagai metrik dan visualisasi

---

## Lisensi

Proyek ini adalah bagian dari program pembelajaran Rakamin Academy. Dataset dan model adalah untuk keperluan pendidikan dan riset.

---

