# User Conversion Predictor

Streamlit app untuk prediksi konversi/checkout user berdasarkan 23 fitur perilaku
browsing (klik basket, sign in, saw checkout, device, dll), menggunakan model
`RandomForestClassifier` (scikit-learn Pipeline).

## Struktur folder
```
.
├── app.py               # Streamlit app
├── model/
│   └── best_model.pkl   # Model pipeline (preprocessing + RandomForest)
├── requirements.txt
├── Dockerfile
└── README.md
```

## 1. Jalankan lokal (tanpa Docker)
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Buka http://localhost:8501

## 2. Jalankan dengan Docker
```bash
docker build -t conversion-predictor .
docker run -p 8501:8501 conversion-predictor
```
Buka http://localhost:8501

## 3. Push ke GitHub
```bash
git init
git add .
git commit -m "Initial commit: conversion predictor app"
git branch -M main
git remote add origin https://github.com/<username>/<repo-name>.git
git push -u origin main
```
> Model `.pkl` berukuran ~1.7MB, aman untuk push langsung ke GitHub (limit GitHub
> per file 100MB). Kalau nanti model membesar (>50MB), pertimbangkan Git LFS.

## 4. Deploy ke cloud (pilih salah satu)

### A. Streamlit Community Cloud (paling gampang, gratis)
1. Push repo ke GitHub (langkah 3).
2. Buka https://share.streamlit.io → "New app" → pilih repo & branch.
3. Set "Main file path" = `app.py`.
4. Deploy. Selesai — dapat URL publik otomatis.

### B. Docker di cloud server (VPS / Railway / Render / Cloud Run / EC2)
Karena sudah ada `Dockerfile`, tinggal:
- **Railway/Render**: connect repo GitHub → mereka otomatis detect Dockerfile → deploy.
- **Google Cloud Run**:
  ```bash
  gcloud builds submit --tag gcr.io/<project-id>/conversion-predictor
  gcloud run deploy --image gcr.io/<project-id>/conversion-predictor --port 8501
  ```
- **VPS manual (misal DigitalOcean/EC2)**:
  ```bash
  git clone https://github.com/<username>/<repo-name>.git
  cd <repo-name>
  docker build -t conversion-predictor .
  docker run -d -p 80:8501 conversion-predictor
  ```

## Catatan penting
- Model dilatih dengan **scikit-learn 1.6.1**. `requirements.txt` sudah dipin ke
  versi yang sama supaya tidak ada `InconsistentVersionWarning` atau error saat
  load pickle di environment lain.
- Input harus berupa 23 kolom sesuai `FEATURES` di `app.py` (nama & urutan kolom
  harus sama dengan saat training).
