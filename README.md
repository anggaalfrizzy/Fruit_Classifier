# 🍎 FruitScan — Klasifikasi Buah dengan Random Forest + Flask

> **Tugas 8 Praktikum Kecerdasan Buatan**  
> Implementasi Algoritma Random Forest untuk Klasifikasi Jenis Buah  
> Berbasis Web Menggunakan Flask  
> Universitas Bale Bandung — Teknik Informatika

---

## 📋 Deskripsi Proyek

FruitScan adalah aplikasi web berbasis **Flask** yang menggunakan algoritma **Random Forest** (Scikit-learn) untuk mengklasifikasikan jenis buah dari gambar yang diupload pengguna. Dataset yang digunakan adalah **Fruits 360** dari Kaggle yang berisi 90+ jenis buah.

---

## 🗂️ Struktur Proyek

```
fruit-classifier/
├── app.py                  # Flask backend utama
├── train_model.py          # Script training model Random Forest
├── requirements.txt        # Dependensi Python
├── Procfile                # Konfigurasi deploy Heroku
├── .gitignore
├── model/                  # Folder model (dihasilkan setelah training)
│   ├── rf_model.pkl        # Model Random Forest terlatih
│   ├── label_encoder.pkl   # Label encoder kelas buah
│   └── config.json         # Konfigurasi & metadata model
├── templates/
│   ├── index.html          # Halaman utama (upload gambar)
│   ├── result.html         # Halaman hasil klasifikasi
│   └── about.html          # Halaman tentang proyek
└── static/
    └── uploads/            # Gambar yang diupload pengguna
```

---

## ⚙️ Instalasi & Menjalankan Lokal

### 1. Clone Repository
```bash
git clone https://github.com/username/fruit-classifier.git
cd fruit-classifier
```

### 2. Buat Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install Dependensi
```bash
pip install -r requirements.txt
```

### 4. Download Dataset Fruits 360
- Buka: https://www.kaggle.com/datasets/moltean/fruits
- Download dan ekstrak ke folder `fruits-360/` di root proyek
- Struktur yang diharapkan:
  ```
  fruits-360/
  ├── Training/
  │   ├── Apple Braeburn/
  │   ├── Banana/
  │   └── ...
  └── Test/
      ├── Apple Braeburn/
      └── ...
  ```

### 5. Training Model
```bash
# Training dengan setting default (200 pohon, max_depth=20, 150 sampel/kelas)
python train_model.py

# Atau kustomisasi parameter:
python train_model.py --n_estimators 300 --max_depth 25 --max_samples 200
```

Parameter yang tersedia:
| Parameter | Default | Keterangan |
|---|---|---|
| `--data_dir` | `fruits-360` | Path folder dataset |
| `--n_estimators` | `200` | Jumlah pohon keputusan |
| `--max_depth` | `20` | Kedalaman maksimum pohon |
| `--img_size` | `32` | Ukuran resize gambar (NxN) |
| `--max_samples` | `150` | Maksimum sampel per kelas |
| `--output_dir` | `model` | Folder simpan model |

### 6. Jalankan Aplikasi
```bash
python app.py
```
Buka browser: **http://127.0.0.1:5000**

---

## 🚀 Deployment ke Heroku

### Prasyarat
- Akun Heroku (gratis): https://heroku.com
- Heroku CLI terinstal: https://devcenter.heroku.com/articles/heroku-cli

### Langkah Deploy
```bash
# Login ke Heroku
heroku login

# Buat aplikasi baru
heroku create nama-aplikasi-kamu

# Tambahkan file model ke Git LFS atau upload manual
# (model pkl bisa besar, gunakan Git LFS)
git lfs install
git lfs track "model/*.pkl"

# Commit dan push
git add .
git commit -m "Initial deploy"
git push heroku main

# Buka aplikasi
heroku open
```

> **Catatan:** Karena file model `.pkl` bisa mencapai ratusan MB, pertimbangkan untuk menyimpannya di **Google Drive** atau **AWS S3** dan memuatnya saat startup.

---

## 🔌 API Endpoint

Tersedia REST API JSON untuk integrasi:

```bash
POST /api/predict
Content-Type: multipart/form-data

# Contoh dengan curl:
curl -X POST http://localhost:5000/api/predict \
  -F "file=@gambar_buah.jpg"
```

Response:
```json
{
  "label": "Banana",
  "confidence": 92.45,
  "top5": [
    { "label": "Banana", "confidence": 92.45 },
    { "label": "Mango",  "confidence": 4.20  },
    { "label": "Papaya", "confidence": 1.80  },
    { "label": "Lemon",  "confidence": 0.85  },
    { "label": "Orange", "confidence": 0.70  }
  ]
}
```

---

## 🧠 Cara Kerja Algoritma

1. **Preprocessing Gambar**: Resize → RGB → Normalisasi [0,1] → Flatten (vektor 1D)
2. **Training (Bootstrap Sampling)**: Dataset dibagi menjadi subset acak dengan pengembalian
3. **Pembangunan Pohon**: Setiap pohon dilatih pada subset data & subset fitur yang berbeda
4. **Prediksi (Majority Voting)**: Semua pohon memberikan vote, kelas mayoritas = hasil akhir
5. **Confidence Score**: Proporsi pohon yang setuju = tingkat kepercayaan

---

## 📊 Performa Model

| Metrik | Nilai |
|---|---|
| Algoritma | Random Forest |
| Jumlah Pohon | 200 |
| Max Depth | 20 |
| Ukuran Input | 32×32 piksel (3072 fitur) |
| Dataset | Fruits 360 (Kaggle) |
| Akurasi (test set) | ~75-90% (bergantung konfigurasi) |

---

## 🛠️ Teknologi

- **Python 3.10+**
- **Flask 3.x** — Web framework
- **Scikit-learn** — Random Forest Classifier
- **Pillow (PIL)** — Pemrosesan gambar
- **NumPy** — Operasi array
- **Bootstrap 5** — UI/UX frontend
- **Heroku** — Platform deployment

---

## 👨‍💻 Informasi Mahasiswa

| | |
|---|---|
| **Judul** | Implementasi Algoritma Random Forest untuk Klasifikasi Jenis Buah Berbasis Web Menggunakan Flask |
| **Mata Kuliah** | Praktikum Kecerdasan Buatan |
| **Institusi** | Universitas Bale Bandung |
| **Program Studi** | Teknik Informatika |

---

## 📚 Referensi

- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
- Oltean, M. (2019). Fruits 360 Dataset. Kaggle.
- Scikit-learn Documentation: https://scikit-learn.org
- Flask Documentation: https://flask.palletsprojects.com
