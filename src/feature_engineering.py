# Proses Feature Engineering

def buat_feature_baru(df):
    # Proses pembuatan fitur baru yang sesuai dengan credit scoring.
    print("Memulai Proses Feature Engineering...")
    
    # Fitur rasio keuangan. 
    df['RASIO_KREDIT_PENDAPATAN'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1)
    df['RASIO_ANNUITY_PENDAPATAN'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1)
    df['RASIO_CREDIT_GOODS'] = df['AMT_CREDIT'] / (df['AMT_GOODS_PRICE'] + 1)
    
    # Fitur usia dan lama kerja.
    df['USIA'] = -df['DAYS_BIRTH'] // 365
    df['LAMA_BEKERJA_TAHUN'] = -df['DAYS_EMPLOYED'] // 365
    df['LAMA_BEKERJA_TAHUN'] = df['LAMA_BEKERJA_TAHUN'].replace(1000, 0)
    
    # Fitur rasio usia vs lama kerja.
    df['RASIO_USIA_KERJA'] = df['USIA'] / (df['LAMA_BEKERJA_TAHUN'] + 1)
    
    # Fitur jumlah tanggungan.
    df['JUMLAH_TANGGUNGAN'] = df['CNT_CHILDREN'] + df.get('CNT_FAM_MEMBERS', 1)
    
    print("Proses Feature Engineering Selesai")
    return df