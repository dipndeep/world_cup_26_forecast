# 🏆 FIFA World Cup 2026 Prediction Project

Proyek berbasis data untuk memprediksi hasil dan jalannya turnamen **FIFA World Cup 2026** menggunakan kombinasi **Elo Rating System**, **Machine Learning Classifiers**, dan **Simulasi Monte Carlo** dengan format baru 48 tim.

> [!WARNING]
> **Disclaimer**: Proyek ini dibuat murni untuk tujuan pembelajaran, analisis data, dan hiburan. Seluruh hasil prediksi yang disajikan berupa estimasi probabilitas statistik dan simulasi komputer, **bukanlah jaminan hasil akhir pertandingan yang sesungguhnya**. Kondisi nyata di lapangan sangat dinamis (faktor cedera, kartu merah, cuaca, dll.). **Dilarang keras menyalahgunakan hasil simulasi ini untuk keperluan taruhan atau judi.**

---

## 📂 Struktur Proyek

```text
d:\code\wc26_forecast\
├── data/                          # Dataset Historis & Informasi WC
│   ├── international_matches.csv  # ~17,700 pertandingan internasional (1872-2022)
│   ├── world_cup_matches.csv      # 901 pertandingan Piala Dunia (1930-2018)
│   ├── world_cups.csv             # Ringkasan 22 edisi Piala Dunia historis
│   ├── 2022_world_cup_groups.csv  # Grup WC 2022
│   ├── 2022_world_cup_matches.csv # Jadwal/skor WC 2022
│   ├── 2022_world_cup_squads.csv  # Skuad tim WC 2022 (untuk proksi fitur)
│   └── 2026_world_cup_groups.csv  # ⭐ Drawing grup & ranking FIFA tim WC 2026
├── notebooks/                     # Jupyter Notebook Alur Kerja (01 - 06)
│   ├── 01_eda.ipynb               # Analisis Data Eksploratif (EDA)
│   ├── 02_feature_engineering.ipynb # Ekstraksi fitur & kalkulasi Elo
│   ├── 03_modeling.ipynb          # Pelatihan model & evaluasi ML
│   ├── 04_tournament_simulation.ipynb # Simulasi turnamen Monte Carlo (10.000 iterasi)
│   ├── 05_visualization.ipynb     # Visualisasi hasil prediksi & grafik
│   └── 06_model_evaluation.ipynb  # Backtesting model pada WC 2018/2014/2010
├── src/                           # Modul Program Utama (Python)
│   ├── elo.py                     # Algoritma Elo Rating sepak bola
│   ├── features.py                # Pipeline Feature Engineering (Vectorized)
│   ├── models.py                  # Wrapper Model (LR, RF, XGBoost, Poisson, Ensemble)
│   └── simulation.py              # Simulator turnamen format 48 tim (Monte Carlo)
├── outputs/                       # Hasil Keluaran Proyek
│   ├── predictions/               # Data hasil prediksi (.csv) & model tersimpan (.pkl)
│   └── figures/                   # Grafik visualisasi (.png)
└── requirements.txt               # Dependencies library Python
```

---

## ⚡ Optimalisasi Performa
Proyek ini menggunakan pemrosesan data yang telah dioptimalkan secara penuh:
* **Vectorized Features**: Pencarian form performa tim (`calculate_recent_form`) dan head-to-head (`calculate_h2h`) di [features.py](src/features.py) dijalankan menggunakan operasi matriks Pandas (vectorized). Notebook 02 selesai dalam **~74 detik** (sebelumnya ~20 menit).
* **Fast Monte Carlo**: Simulasi fase grup Piala Dunia di [simulation.py](src/simulation.py) meniadakan pembuatan DataFrame di dalam iterasi loop. Simulasi 10.000 iterasi selesai dalam **~45 detik** (sebelumnya ~3,5 menit).

---

## 🚀 Cara Instalasi dan Penggunaan

### 1. Klon / Masuk ke Folder Proyek
Buka terminal Anda (PowerShell / Command Prompt) dan masuk ke direktori proyek:
```bash
cd d:\code\wc26_forecast
```

### 2. Membuat & Mengaktifkan Virtual Environment (venv)
```bash
# Membuat venv
python -m venv venv

# Mengaktifkan venv (PowerShell)
.\venv\Scripts\Activate.ps1

# Mengaktifkan venv (Command Prompt)
.\venv\Scripts\activate.bat
```

### 3. Menginstal Library (Dependencies)
```bash
pip install -r requirements.txt
```

> [!IMPORTANT]
> **Mengatasi Error `UnicodeDecodeError` pada Skuad 2022:**
> File skuad berisi aksen Latin pada nama pemain. Kode pemuatan telah disesuaikan menggunakan opsi `encoding='latin-1'` agar dapat dibaca tanpa error pada notebook 01 dan 02.
>
> **Mengatasi Kebijakan Keamanan Windows (AppLocker / WDAC DLL Block):**
> Jika Anda mengalami kendala `ImportError: DLL load failed` saat menjalankan notebook di dalam `venv` (karena drive `D:` dibatasi untuk eksekusi file `.pyd` scikit-learn), silakan instal library langsung ke profil pengguna komputer Anda yang telah disetujui sistem:
> ```bash
> python -m pip install --user -r requirements.txt
> ```

### 4. Menjalankan Jupyter Notebook
Jalankan Jupyter Notebook dari dalam terminal Anda:
```bash
cd notebooks
jupyter notebook
```
Buka dan jalankan notebook **secara berurutan**:
1. **`01_eda.ipynb`** — Memahami kualitas data & profil tim WC 2026.
2. **`02_feature_engineering.ipynb`** — Menghitung Elo Rating & matriks form.
3. **`03_modeling.ipynb`** — Melatih & mengevaluasi model klasifikasi ML.
4. **`04_tournament_simulation.ipynb`** — Menjalankan simulasi turnamen Monte Carlo.
5. **`05_visualization.ipynb`** — Membuat infografis probabilitas juara & peta progresi.
6. **`06_model_evaluation.ipynb`** — Menguji performa historis model pada turnamen masa lalu.

---

## 🏆 Temuan Utama & Model
* **Model Terbaik**: Klasifikasi ensemble berbasis pembobotan (Logistic Regression + Random Forest + XGBoost + Poisson) menghasilkan keseimbangan prediksi terbaik.
* **Format WC 2026**: Menggunakan bagan simulasi 48 tim (12 grup × 4 tim, lolos ke Babak 32 Besar melalui 2 teratas grup + 8 peringkat 3 terbaik).
* **Hasil Keluaran**: Grafik probabilitas juara dan diagram radar kekuatan tim dapat ditemukan langsung di folder `outputs/figures/`.
