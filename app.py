from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# 1. Load model dan alat preprocessing
model = joblib.load('model_ann.pkl')
scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')
le = joblib.load('label_encoder.pkl')

daftar_kota = list(le.classes_)

# 2. Baca CSV asli
file_csv = 'disparbud-od_15392_jml_ush_restoran_rumah_makan_cafe_brdsrkn_kabupate_v1_data.csv'
df = pd.read_csv(file_csv)

# Menghitung data untuk Dasbor
df_top15 = df.groupby('nama_kabupaten_kota')['jumlah_usaha'].sum().nlargest(15).reset_index()
labels_top15 = df_top15['nama_kabupaten_kota'].tolist()
values_top15 = df_top15['jumlah_usaha'].tolist()

data_tabel = [{'peringkat': i+1, 'kota': k, 'jumlah': v} for i, (k, v) in enumerate(zip(labels_top15, values_top15))]

kpi_total_daerah = len(daftar_kota)
kpi_total_usaha = int(df['jumlah_usaha'].sum())
kpi_rata_usaha = int(df['jumlah_usaha'].mean())

# BARU: Membuat daftar tahun otomatis untuk Dropdown (Tahun terakhir di data + 6 tahun ke depan)
tahun_terakhir = int(df['tahun'].max())
daftar_tahun = list(range(tahun_terakhir + 1, tahun_terakhir + 7))

@app.route('/')
def home():
    # BARU: Mengirim 'years=daftar_tahun' ke HTML
    return render_template('index.html', 
                           cities=daftar_kota, 
                           labels=labels_top15, 
                           values=values_top15,
                           data_tabel=data_tabel,
                           kpi_daerah=kpi_total_daerah,
                           kpi_total=kpi_total_usaha,
                           kpi_rata=kpi_rata_usaha,
                           years=daftar_tahun) 

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        nama_kota = request.form['kota']
        tahun = int(request.form['tahun'])

        kota_id = le.transform([nama_kota])[0]
        X_input = np.array([[kota_id, tahun]])
        X_scaled = scaler_X.transform(X_input)

        prediksi_scaled = model.predict(X_scaled)
        prediksi_asli = scaler_y.inverse_transform(prediksi_scaled.reshape(-1, 1))[0][0]
        hasil_akhir = int(round(abs(prediksi_asli)))

        return render_template('result.html', kota=nama_kota, tahun=tahun, hasil=hasil_akhir)

if __name__ == '__main__':
    print("Menyalakan server Flask Dasbor Profesional (Dengan Dropdown Tahun)...")
    app.run(debug=True)