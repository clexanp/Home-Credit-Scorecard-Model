# Utils atau alat bantu yang dipakai di projek ini

import pandas as pd
import os
import matplotlib.pyplot as plt

# Proses memuat dataset utama.
def load_data(filename='application_train.csv'):
    path = f'data/{filename}'
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} tidak ditemukan. Pastikan dataset ada di folder 'data/'")
    
    print(f"Sedang memuat {filename} ...")
    df = pd.read_csv(path)
    print(f"Dataset berhasil dimuat! Shape: {df.shape[0]:,} baris × {df.shape[1]} kolom")
    return df

# Proses menyimpan grafik ke folder output.
def simpan_grafik(nama_file, folder='output/figures'):
    os.makedirs(folder, exist_ok=True)
    plt.savefig(f'{folder}/{nama_file}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Grafik disimpan: {nama_file}")