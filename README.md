# Prediksi Penumpang Kereta Api Indonesia
## Deskripsi Proyek
Proyek Data Science untuk menganalisis dan memprediksi jumlah penumpang kereta api menggunakan Machine Learning serta menampilkan hasil analisis dalam bentuk dashboard interaktif menggunakan Streamlit.

## Tujuan
- Menganalisis pola jumlah penumpang kereta api.
- Melakukan eksplorasi data (EDA).
- Membangun model prediksi jumlah penumpang.
- Mengevaluasi performa model.
- Menyediakan dashboard interaktif untuk visualisasi dan prediksi.

## Dataset
Dataset terdiri dari data operasional kereta api periode:
- 2017
- 2018
- 2019
- 2020
- 2021
- 2022
- 2023
- 2024
- 2025

Dataset hasil pembersihan disimpan pada:

```text
data/cleaned_Data_Kereta.csv
```

## Teknologi yang Digunakan
- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Plotly
- Matplotlib
- Seaborn
- Statsmodels
- Joblib

## Tahapan Proyek
### 1. Data Preparation
- Menggabungkan dataset tahunan
- Data cleaning
- Feature engineering
### 2. Exploratory Data Analysis (EDA)
- Analisis tren penumpang
- Analisis distribusi data
- Analisis korelasi fitur
### 3. Machine Learning
Model yang digunakan:
- Random Forest Regressor
### 4. Evaluasi Model
Metrik evaluasi:
- MAE
- RMSE
- MAPE
- R² Score
### 5. Dashboard Streamlit
Dashboard menyediakan:
- Visualisasi data
- Analisis tren
- Evaluasi model
- Prediksi jumlah penumpang

## Struktur Folder
```text
KeretaDashboard/
│
├── app.py
├── TubesVisual.ipynb
│
├── data/
├── model/
├── output/
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Cara Menjalankan

### Clone Repository
```bash
git clone <repository-url>
```

### Install Dependency
```bash
pip install -r requirements.txt
```

### Jalankan Dashboard
```bash
streamlit run app.py
```

Dashboard akan berjalan pada:
```text
http://localhost:8501
```

## Author
Ridho Perlinta Sembiring
Mahasiswa Telkom University Purwokerto