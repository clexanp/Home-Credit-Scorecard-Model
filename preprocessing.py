# Proses Data Cleaning dan Preprocessing

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Proses handle missing value.
def handle_missing_values(df):
    print("Menangani missing values...")

    # Proses pengisian data numerik dengan median.
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    # Proses pengisian data kategorikal dengan modus.
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    print("Missing values telah ditangani")
    return df

def encode_categorical(df):

    # Proses pengubahan data kategorikal menjadi data numerik.
    print("Melakukan Label Encoding...")
    le = LabelEncoder()
    cat_cols = df.select_dtypes(include=['object']).columns
    
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    
    print(f"Label Encoding selesai untuk {len(cat_cols)} kolom")
    return df