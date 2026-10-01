import json
from datetime import date

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────
# KONFIGURASI HALAMAN & LOKASI FILE
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard Prediksi Penumpang KAI",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "model/rf_model.joblib"
KERETA_MAP_PATH = "model/kereta_map.joblib"

DATA_PATH = "data/cleaned_Data_Kereta.csv"

EVAL_METRICS_PATH = "output/eval_metrics.json"
EVAL_RESULTS_PATH = "output/eval_results.csv"
FEATURE_IMPORTANCE_PATH = "output/feature_importance.csv"

LABEL_FITUR = {
    "tahun": "Tahun",
    "bulan": "Bulan",
    "hari": "Tanggal (1-31)",
    "hari_ke": "Hari ke- dalam Minggu",
    "is_weekend": "Akhir Pekan",
    "is_lebaran": "Periode Lebaran",
    "is_libur_sekolah": "Libur Sekolah",
    "is_libur_akhir_tahun": "Libur Akhir Tahun",
    "kapasitas_kursi": "Kapasitas Kursi",
    "gerbong_tersedia": "Gerbong Tersedia",
    "bulan_sin": "Pola Musiman (Komponen Sin)",
    "bulan_cos": "Pola Musiman (Komponen Cos)",
    "kereta_code": "Jenis/Nama Kereta",
}

# Define KAI brand colors for consistent use in Python/Plotly
KAI_BLUE_900 = "#06335c"
KAI_BLUE_700 = "#0a5599"
KAI_BLUE_500 = "#1f7bc4"
KAI_ORANGE_700 = "#c2480a"
KAI_ORANGE_600 = "#e8650f"
TEXT_PRIMARY = "#ffffff" # Changed to white for visibility
TEXT_SECONDARY = "#f3f7fc" # Changed to lighter grey

# ──────────────────────────────────────────────────────────────────────────
# CSS — TEMA WARNA KAI (BIRU, ORANYE, PUTIH) DENGAN KONTRAS YANG TERJAGA
# ──────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root{
    --kai-blue-900:#06335c; --kai-blue-700:#0a5599; --kai-blue-500:#1f7bc4; --kai-blue-100:#e8f1fa;
    --kai-orange-600:#e8650f; --kai-orange-700:#c2480a; --kai-orange-100:#fdeadb;
    --bg-page:#f4f6fa; --bg-card:#ffffff;
    --text-primary:#1e293b; --text-secondary:#64748b; --border-soft:#e6eaf0;
    --success-bg:#e7f6ec; --success-text:#0f6e37;
    --warning-bg:#fff2e0; --warning-text:#9c4d04;
    --danger-bg:#fdeaea;  --danger-text:#c0392b;
    --info-bg:#e8f1fa;    --info-text:#0a5599;
}

.stApp { background-color: var(--bg-page); }
.block-container { padding-top: 1.6rem; max-width: 1180px; }
h1,h2,h3,h4 { color: var(--text-primary); }
p, span, label, div { color: var(--text-primary); }

/* ---------- SIDEBAR ---------- */
[data-testid="stSidebar"]{
    background: linear-gradient(180deg, var(--kai-blue-900) 0%, var(--kai-blue-700) 100%);
}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3,
[data-testid="stSidebar"] .stCaption {
    color: #f3f7fc !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"]{
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 10px;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary p { color:#f3f7fc !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.2); }

.sidebar-brand{ padding: 4px 0 18px 0; }
.sidebar-brand .brand-title{ font-size:1.5rem; font-weight:800; color:#ffffff; margin:0; line-height:1.15; }
.sidebar-brand .brand-sub{ font-size:0.85rem; color:#cfe2f5; margin-top:4px; }
.sidebar-foot{ font-size:0.74rem; color:#bcd2e8; line-height:1.5; margin-top:10px; }

/* ---------- BUTTON UTAMA (ORANYE KAI) ---------- */
.stButton > button{
    background-color: var(--kai-orange-700);
    color:#ffffff;
    border:none;
    border-radius:10px;
    font-weight:700;
    padding:0.6rem 1rem;
    transition: filter 0.15s ease;
}
.stButton > button:hover{ filter: brightness(1.08); color:#ffffff; }
.stButton > button p { color:#ffffff !important; }

/* ---------- KARTU ---------- */
.kai-card{
    background: var(--bg-card);
    border: 1px solid var(--border-soft);
    border-radius: 16px;
    padding: 22px 24px;
    box-shadow: 0 2px 10px rgba(6,51,92,0.05);
    margin-bottom: 18px;
}
.kai-card-accent{ border-left: 4_px solid var(--kai-orange-600); }

/* ---------- HEADER SEKSI: aksen garis mirip rel kereta ---------- */
.kai-section-head{ display:flex; align-items:center; gap:10px; margin: 6px 0 14px 0; }
.kai-section-head .icon-badge{
    width:34px; height:34px; border-radius:9px; background:var(--kai-blue-100);
    display:flex; align-items:center; justify-content:center; font-size:1.05rem;
}
.kai-section-head h3{ margin:0; color:var(--kai-blue-900); font-size:1.18rem; }
.kai-section-head .kai-sub{ color:var(--text-secondary); font-size:0.85rem; margin:2px 0 0 44px; }

.rail-divider{
    height:10px; margin: 28px 0 22px 0; border-radius:6px;
    background-image: repeating-linear-gradient(90deg, var(--border-soft) 0 14px, transparent 14px 22px);
    background-position:center; background-repeat:repeat-x; background-size:auto 2px;
    border-top:2px solid var(--border-soft); border-bottom:2px solid var(--border-soft);
}

/* ---------- METRIC / INFO CHIP ---------- */
.metric-label{ font-size:0.78rem; color:var(--text-secondary); text-transform:uppercase; letter-spacing:0.04em; font-weight:600; }
.metric-value{ font-size:2.0rem; font-weight:800; color:var(--kai-blue-900); margin-top:2px; }
.metric-caption{ font-size:0.85rem; color:var(--text-secondary); margin-top:6px; }

.info-chip{
    background: var(--kai-blue-100); border-radius:12px; padding:12px 14px; height:100%;
}
.info-chip .info-chip-label{ font-size:0.74rem; color:var(--kai-blue-700); font-weight:700; text-transform:uppercase; letter-spacing:0.03em; }
.info-chip .info-chip-value{ font-size:1.15rem; color:var(--kai-blue-900); font-weight:700; margin-top:3px; }

/* ---------- BADGE STATUS ---------- */
.kai-badge{ display:inline-block; padding:5px 13px; border-radius:999px; font-size:0.82rem; font-weight:700; }
.badge-success{ background:var(--success-bg); color:var(--success-text); }
.badge-warning{ background:var(--warning-bg); color:var(--warning-text); }
.badge-danger { background:var(--danger-bg);  color:var(--danger-text); }
.badge-info   { background:var(--info-bg);    color:var(--info-text); }

/* ---------- PROGRESS BAR KAPASITAS ---------- */
.cap-track{ background:#eef1f5; border-radius:999px; height:14px; width:100%; overflow:hidden; }
.cap-fill{ height:100%; border-radius:999px; }

/* ---------- KARTU REKOMENDASI ---------- */
.rec-card{ background:#ffffff; border:1px solid var(--border-soft); border-radius:14px; padding:16px 18px; height:100%; }
.rec-card h4{ margin:0 0 6px 0; font-size:1.0rem; color:var(--text-primary); }
.rec-card p{ font-size:0.88rem; color:var(--text-secondary); margin:0; line-height:1.5; }
.rec-card .rec-strip{ height:4px; border-radius:4px; margin-bottom:12px; width:46px; }
.strip-success{ background:var(--success-text); }
.strip-warning{ background:var(--warning-text); }
.strip-danger{ background:var(--danger-text); }
.strip-info{ background:var(--info-text); }

/* ---------- LANDING (BELUM PREDIKSI) ---------- */
.kai-landing{
    background: linear-gradient(135deg, var(--kai-blue-900), var(--kai-blue-700));
    border-radius:18px; padding:34px 32px; color:#ffffff; margin-bottom:18px;
}
.kai-landing h2{ color:#ffffff; margin:0 0 6px 0; }
.kai-landing p{ color:#dceaf7; margin:0; font-size:0.95rem; max-width:640px; }
/* --- CSS Update untuk Kalender & Widget --- */

/* --- CSS Update untuk Kalender (Warna Putih Agar Terlihat Jelas) --- */

/* Menargetkan angka tanggal agar menjadi putih */
div[data-baseweb="calendar"] div[role="gridcell"] button div,
div[data-baseweb="calendar"] div[role="gridcell"] button {
    color: #ffffff !important;
}

/* Menargetkan hari (Su, Mo, Tu, dst) di header kalender agar putih */
div[data-baseweb="calendar"] div[role="columnheader"] {
    color: #ffffff !important;
}

/* Menargetkan teks bulan dan tahun di header agar putih */
div[data-baseweb="calendar"] [data-testid="stText"],
div[data-baseweb="calendar"] div[role="presentation"] {
    color: #ffffff !important;
}

/* Menargetkan angka tanggal yang di luar bulan (disabled) agar abu-abu terang */
div[data-baseweb="calendar"] div[role="gridcell"] button:disabled div {
    color: #888888 !important;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# FUNGSI BANTUAN UNTUK DASHBOARD (DIPINDAH DARI UTILS/FUNCTIONS.PY)
# ──────────────────────────────────────────────────────────────────────────
FEATURE_COLUMNS = [
    "tahun",
    "bulan",
    "hari",
    "hari_ke",
    "is_weekend",
    "is_lebaran",
    "is_libur_sekolah",
    "is_libur_akhir_tahun",
    "kapasitas_kursi",
    "gerbong_tersedia",
    "bulan_sin",
    "bulan_cos",
    "kereta_code",
]

HARI_INDO = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
BULAN_INDO = [
    "Januari", "Februari", "Maret", "April", "Mei", "Juni",
    "Juli", "Agustus", "September", "Oktober", "November", "Desember",
]

KURSI_PER_GERBONG = 80  # konsisten dengan asumsi pada notebook (gerbong_dibutuhkan = ceil(penumpang/80))

def nama_hari(tanggal):
    """Mengembalikan nama hari dalam Bahasa Indonesia dari objek date/datetime."""
    return HARI_INDO[tanggal.weekday()]


def format_tanggal_indo(tanggal):
    """Format tanggal menjadi 'D Bulan YYYY', contoh: 17 Agustus 2026."""
    return f"{tanggal.day} {BULAN_INDO[tanggal.month - 1]} {tanggal.year}"


def saran_kondisi_khusus(tanggal):
    """
    Memberi NILAI AWAL (default) untuk checkbox kondisi khusus berdasarkan
    aturan kalender akademik/libur yang umum berlaku di Indonesia.

    Ini hanya SARAN, bukan kebenaran mutlak, karena tanggal libur sekolah dan
    Lebaran berubah setiap tahun dan ditentukan lewat SKB Menteri/Kemenag.
    Pengguna tetap bisa mengubahnya secara manual di sidebar sebelum prediksi.
    """
    bulan, hari = tanggal.month, tanggal.day
    libur_akhir_tahun = (bulan == 12 and hari >= 20) or (bulan == 1 and hari <= 3)
    libur_sekolah = bulan in (6, 7) or (bulan == 12 and hari >= 18)
    return {
        "is_libur_akhir_tahun": libur_akhir_tahun,
        "is_libur_sekolah": libur_sekolah,
        "is_lebaran": False,
    }


def get_info_kereta(df, nama_kereta):
    """
    Mengambil informasi statis & historis satu kereta dari dataset bersih:
    kapasitas kursi, jumlah gerbong tersedia (nilai paling umum/modus),
    rata-rata historis, nilai maksimum historis, dan rentang periode data.
    """
    subset = df[df["nama_kereta"] == nama_kereta]
    if subset.empty:
        return None

    return {
        "kapasitas_kursi": int(subset["kapasitas_kursi"].mode().iloc[0]),
        "gerbong_tersedia": int(subset["gerbong_tersedia"].mode().iloc[0]),
        "rata_rata_historis": float(subset["jumlah_penumpang"].mean()),
        "maksimum_historis": float(subset["jumlah_penumpang"].max()),
        "jumlah_data": int(len(subset)),
        "periode_awal": subset["tanggal"].min(),
        "periode_akhir": subset["tanggal"].max(),
    }


def bangun_fitur(tanggal, kereta_code, kapasitas_kursi, gerbong_tersedia,
                  is_lebaran, is_libur_sekolah, is_libur_akhir_tahun):
    """
    Membangun satu baris DataFrame fitur sesuai FEATURE_COLUMNS, siap
    dimasukkan ke rf_model.predict().
    """
    hari_ke = tanggal.weekday()
    bulan = tanggal.month

    baris = {
        "tahun": tanggal.year,
        "bulan": bulan,
        "hari": tanggal.day,
        "hari_ke": hari_ke,
        "is_weekend": int(hari_ke in (5, 6)),
        "is_lebaran": int(is_lebaran),
        "is_libur_sekolah": int(is_libur_sekolah),
        "is_libur_akhir_tahun": int(is_libur_akhir_tahun),
        "kapasitas_kursi": kapasitas_kursi,
        "gerbong_tersedia": gerbong_tersedia,
        "bulan_sin": np.sin(2 * np.pi * bulan / 12),
        "bulan_cos": np.cos(2 * np.pi * bulan / 12),
        "kereta_code": kereta_code,
    }
    return pd.DataFrame([baris], columns=FEATURE_COLUMNS)


def hitung_gerbong(prediksi_penumpang, gerbong_tersedia, kursi_per_gerbong=KURSI_PER_GERBONG):
    """Menghitung jumlah gerbong yang dibutuhkan dan kekurangannya (jika ada)."""
    gerbong_dibutuhkan = int(np.ceil(prediksi_penumpang / kursi_per_gerbong))
    tambahan_gerbong = max(0, gerbong_dibutuhkan - gerbong_tersedia)
    return gerbong_dibutuhkan, tambahan_gerbong


def klasifikasi_kepadatan(load_factor):
    """
    Mengklasifikasikan tingkat kepadatan berdasarkan load factor (%).
    Mengembalikan (label, kode_warna) dengan kode_warna salah satu dari
    'success', 'warning', 'danger' yang dipetakan ke CSS di app.py.
    """
    if load_factor < 70:
        return "Normal", "success"
    elif load_factor < 90:
        return "Padat", "warning"
    return "Sangat Padat", "danger"


def buat_rekomendasi(load_factor, tambahan_gerbong, is_lebaran, is_libur_sekolah, is_libur_akhir_tahun):
    """Menyusun daftar kartu rekomendasi tindakan berbasis aturan (rule-based)."""
    rekomendasi = []

    if load_factor < 70:
        rekomendasi.append({
            "icon": "✅",
            "judul": "Kapasitas Mencukupi",
            "isi": "Prediksi jumlah penumpang masih di bawah kapasitas kursi yang tersedia. "
                   "Operasional dapat berjalan dengan skema rangkaian normal tanpa penambahan armada.",
            "level": "success",
        })
    elif load_factor < 90:
        rekomendasi.append({
            "icon": "⚠️",
            "judul": "Perlu Antisipasi",
            "isi": "Tingkat kepadatan diprediksi cukup tinggi. Disarankan menyiapkan petugas tambahan "
                   "di stasiun keberangkatan dan memantau perkembangan penjualan tiket secara berkala.",
            "level": "warning",
        })
    else:
        rekomendasi.append({
            "icon": "🚨",
            "judul": "Risiko Kepadatan Tinggi",
            "isi": "Prediksi penumpang mendekati atau melampaui kapasitas kursi yang tersedia. "
                   "Disarankan berkoordinasi untuk penambahan rangkaian gerbong dan menyiapkan "
                   "skema pembatasan penjualan tiket di loket maupun aplikasi.",
            "level": "danger",
        })

    if tambahan_gerbong > 0:
        rekomendasi.append({
            "icon": "🚃",
            "judul": "Tambahan Gerbong Disarankan",
            "isi": f"Dengan asumsi kapasitas {KURSI_PER_GERBONG} kursi per gerbong, dibutuhkan kurang "
                   f"lebih {tambahan_gerbong} gerbong tambahan dari jumlah yang saat ini tersedia.",
            "level": "warning",
        })

    if is_lebaran:
        rekomendasi.append({
            "icon": "🕌",
            "judul": "Periode Mudik Lebaran",
            "isi": "Tanggal ini termasuk periode Lebaran. Data historis menunjukkan lonjakan penumpang "
                   "yang signifikan pada periode ini, sehingga koordinasi lintas unit perlu dilakukan "
                   "lebih awal dari biasanya.",
            "level": "info",
        })

    if is_libur_akhir_tahun:
        rekomendasi.append({
            "icon": "🎄",
            "judul": "Periode Libur Akhir Tahun",
            "isi": "Tanggal ini termasuk periode libur akhir tahun yang secara historis memiliki volume "
                   "penumpang lebih tinggi dibandingkan hari biasa.",
            "level": "info",
        })

    if is_libur_sekolah:
        rekomendasi.append({
            "icon": "🎒",
            "judul": "Periode Libur Sekolah",
            "isi": "Tanggal ini termasuk periode libur sekolah, yang umumnya turut mendorong "
                   "peningkatan jumlah penumpang dibandingkan hari sekolah biasa.",
            "level": "info",
        })

    return rekomendasi


def cek_ekstrapolasi(tanggal, tahun_min, tahun_maks):
    """
    Mengecek apakah tahun yang diprediksi berada di luar rentang tahun data latih.
    RandomForestRegressor adalah model berbasis pohon yang tidak mampu melakukan
    ekstrapolasi tren di luar rentang nilai yang pernah dipelajari, sehingga
    pengguna perlu diberi tahu jika memprediksi tahun yang jauh di luar data historis.
    """
    return tanggal.year > tahun_maks or tanggal.year < tahun_min


# ──────────────────────────────────────────────────────────────────────────
# PEMUATAN ARTEFAK (MODEL, DATA, HASIL EVALUASI) — DI-CACHE AGAR CEPAT
# ──────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat model Random Forest...")
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource(show_spinner=False)
def load_kereta_map():
    return joblib.load(KERETA_MAP_PATH)


@st.cache_data(show_spinner="Memuat dataset historis...")
def load_dataset():
    df = pd.read_csv(DATA_PATH)
    df["tanggal"] = pd.to_datetime(df["tanggal"])
    return df


@st.cache_data(show_spinner=False)
def load_eval_metrics():
    with open(EVAL_METRICS_PATH) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_eval_results():
    return pd.read_csv(EVAL_RESULTS_PATH)


@st.cache_data(show_spinner=False)
def load_feature_importance():
    return pd.read_csv(FEATURE_IMPORTANCE_PATH)

try:
    rf_model = load_model()
    kereta_map = load_kereta_map()
    df = load_dataset()
    eval_metrics = load_eval_metrics()
    eval_results = load_eval_results()
    feature_importance = load_feature_importance()
except FileNotFoundError as e:
    st.error(
        "Salah satu file artefak belum ditemukan: **{}**.\n\n"
        "Pastikan rf_model.joblib, kereta_map.joblib, cleaned_Data_Kereta.csv, "
        "eval_metrics.json, eval_results.csv, dan feature_importance.csv berada "
        "di folder yang sama dengan app.py.".format(e.filename)
    )
    st.stop()

kode_kereta_map = {nama: kode for kode, nama in kereta_map.items()}
daftar_kereta = sorted(kode_kereta_map.keys())
tahun_min, tahun_maks = int(df["tahun"].min()), int(df["tahun"].max())


# ──────────────────────────────────────────────────────────────────────────
# FUNGSI BANTUAN UNTUK RENDER KOMPONEN (HTML KECIL, DIPAKAI BERULANG)
# ──────────────────────────────────────────────────────────────────────────
def section_head(icon, title, subtitle=None):
    st.markdown(f"""
    <div class="kai-section-head">
        <div class="icon-badge">{icon}</div>
        <h3>{title}</h3>
    </div>
    {f'<div class="kai-sub">{subtitle}</div>' if subtitle else ''}
    """, unsafe_allow_html=True)


def info_chip(label, value):
    st.markdown(f"""
    <div class="info-chip">
        <div class="info-chip-label">{label}</div>
        <div class="info-chip-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def badge(text, level):
    return f'<span class="kai-badge badge-{level}">{text}</span>'


def metric_card(label, value, caption=None, value_color=None):
    color_style = f"color:{value_color};" if value_color else ""
    st.markdown(f"""
    <div class="kai-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="{color_style}">{value}</div>
        {f'<div class="metric-caption">{caption}</div>' if caption else ''}
    </div>
    """, unsafe_allow_html=True)


def recommendation_card(icon, judul, isi, level):
    st.markdown(f"""
    <div class="rec-card">
        <div class="rec-strip strip-{level}"></div>
        <h4>{icon} {judul}</h4>
        <p>{isi}</p>
    </div>
    """, unsafe_allow_html=True)


def styled_fig(fig, height=380):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        # Mengatur warna font dasar menjadi hitam untuk label dan angka
        font=dict(color="#000000", size=13),
        margin=dict(l=10, r=10, t=50, b=10),
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(bgcolor="white", font_size=12),
        # Mengatur warna judul agar hitam
        title=dict(font=dict(color="#000000", size=16)),
    )

    # Memperjelas warna sumbu X dan Y agar teks label menjadi hitam
    fig.update_xaxes(
        showgrid=True, zeroline=False,
        linecolor="#000000", gridcolor="#cbd5e1",
        title_font=dict(color="#000000"), # Hitam untuk "Indeks Data Uji" / "Tingkat Kepentingan"
        tickfont=dict(color="#000000")    # Hitam untuk angka sumbu
    )
    fig.update_yaxes(
        showgrid=True, zeroline=False,
        linecolor="#000000", gridcolor="#cbd5e1",
        title_font=dict(color="#000000"), # Hitam untuk "Jumlah Penumpang"
        tickfont=dict(color="#000000")    # Hitam untuk angka sumbu
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────
# SIDEBAR — INPUT PARAMETER PREDIKSI
# ──────────────────────────────────────────────────────────────────────────
if "hasil_prediksi" not in st.session_state:
    st.session_state.hasil_prediksi = None

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <p class="brand-title">🚆 KAI Predict</p>
        <p class="brand-sub">Dashboard Prediksi Penumpang Harian</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr/>", unsafe_allow_html=True)

    st.markdown("**Parameter Prediksi**")
    tanggal_input = st.date_input(
        "Tanggal Keberangkatan",
        value=date.today(),
        help="Pilih tanggal yang ingin diprediksi jumlah penumpangnya.",
    )
    nama_kereta_input = st.selectbox(
        "Nama Kereta",
        options=daftar_kereta,
        help="Pilih kereta yang ingin diprediksi.",
    )

    # Catatan teknis: checkbox dengan parameter key hanya membaca value= pada
    # render PERTAMA kali. Supaya saran kondisi khusus ikut berubah setiap
    # tanggal diganti, session_state untuk ketiga checkbox di-refresh secara
    # manual SEBELUM widget dibuat, setiap kali tanggal_input berbeda dari
    # tanggal pada rerun sebelumnya.
    if st.session_state.get("tanggal_sebelumnya") != tanggal_input:
        saran = saran_kondisi_khusus(tanggal_input)
        st.session_state["chk_lebaran"] = saran["is_lebaran"]
        st.session_state["chk_sekolah"] = saran["is_libur_sekolah"]
        st.session_state["chk_akhirtahun"] = saran["is_libur_akhir_tahun"]
        st.session_state["tanggal_sebelumnya"] = tanggal_input

    with st.expander("⚙️ Kondisi Khusus (Opsional)"):
        st.caption("Sesuaikan dengan kalender resmi bila perlu. Nilai di bawah hanya saran awal.")
        is_lebaran = st.checkbox("Periode Lebaran", key="chk_lebaran")
        is_libur_sekolah = st.checkbox("Libur Sekolah", key="chk_sekolah")
        is_libur_akhir_tahun = st.checkbox("Libur Akhir Tahun", key="chk_akhirtahun")

    tombol_prediksi = st.button("🔍 Jalankan Prediksi", width="stretch")

    st.markdown(f"""
    <div class="sidebar-foot">
        Model: Random Forest Regressor<br/>
        Periode data latih: {tahun_min}–{tahun_maks}<br/>
        Total kereta tercatat: {len(daftar_kereta)}
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────
# PROSES PREDIKSI SAAT TOMBOL DIKLIK
# ──────────────────────────────────────────────────────────────────────────
if tombol_prediksi:
    info_kereta = get_info_kereta(df, nama_kereta_input)

    fitur = bangun_fitur(
        tanggal=tanggal_input,
        kereta_code=kode_kereta_map[nama_kereta_input],
        kapasitas_kursi=info_kereta["kapasitas_kursi"],
        gerbong_tersedia=info_kereta["gerbong_tersedia"],
        is_lebaran=is_lebaran,
        is_libur_sekolah=is_libur_sekolah,
        is_libur_akhir_tahun=is_libur_akhir_tahun,
    )
    prediksi = float(rf_model.predict(fitur)[0])
    prediksi = max(0, prediksi)

    load_factor = (prediksi / info_kereta["kapasitas_kursi"]) * 100
    label_kepadatan, level_kepadatan = klasifikasi_kepadatan(load_factor)
    gerbong_dibutuhkan, tambahan_gerbong = hitung_gerbong(prediksi, info_kereta["gerbong_tersedia"])
    rekomendasi_list = buat_rekomendasi(
        load_factor, tambahan_gerbong, is_lebaran, is_libur_sekolah, is_libur_akhir_tahun
    )

    # Cek apakah ada data aktual pada tanggal & kereta yang sama (untuk validasi, bukan acuan)
    cocok = df[(df["nama_kereta"] == nama_kereta_input) & (df["tanggal"].dt.date == tanggal_input)]
    aktual = float(cocok["jumlah_penumpang"].iloc[0]) if not cocok.empty else None

    st.session_state.hasil_prediksi = {
        "nama_kereta": nama_kereta_input,
        "tanggal": tanggal_input,
        "info_kereta": info_kereta,
        "is_lebaran": is_lebaran,
        "is_libur_sekolah": is_libur_sekolah,
        "is_libur_akhir_tahun": is_libur_akhir_tahun,
        "prediksi": prediksi,
        "load_factor": load_factor,
        "label_kepadatan": label_kepadatan,
        "level_kepadatan": level_kepadatan,
        "gerbong_dibutuhkan": gerbong_dibutuhkan,
        "tambahan_gerbong": tambahan_gerbong,
        "rekomendasi_list": rekomendasi_list,
        "aktual": aktual,
        "ekstrapolasi": cek_ekstrapolasi(tanggal_input, tahun_min, tahun_maks),
    }


# ──────────────────────────────────────────────────────────────────────────
# HEADER UTAMA (SELALU TAMPIL)
# ──────────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style="margin-bottom:2px;">🚆 Dashboard Prediksi Penumpang Kereta Api</h1>
<p style="color:var(--text-secondary); margin-top:0;">
    Estimasi jumlah penumpang harian berbasis Random Forest Regressor untuk mendukung
    perencanaan kebutuhan gerbong dan kapasitas operasional.
</p>
""", unsafe_allow_html=True)

hasil = st.session_state.hasil_prediksi

# ──────────────────────────────────────────────────────────────────────────
# TAMPILAN AWAL (BELUM ADA PREDIKSI)
# ──────────────────────────────────────────────────────────────────────────
if hasil is None:
    st.markdown(f"""
    <div class="kai-landing">
        <h2>Pilih parameter di sidebar untuk mulai</h2>
        <p>Tentukan tanggal keberangkatan dan nama kereta pada panel sebelah kiri,
        lalu klik tombol <b>Jalankan Prediksi</b> untuk melihat estimasi jumlah
        penumpang, status kepadatan, dan rekomendasi operasional.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        info_chip("Total Kereta Tercatat", f"{len(daftar_kereta)} kereta")
    with c2:
        info_chip("Periode Data Historis", f"{tahun_min}–{tahun_maks}")
    with c3:
        info_chip("Total Data Historis", f"{len(df):,} baris")

# ──────────────────────────────────────────────────────────────────────────
# HASIL PREDIKSI: 1) INFO UMUM -> 2) PREDIKSI & REKOMENDASI -> 3) EVALUASI
# ──────────────────────────────────────────────────────────────────────────
else:
    info_k = hasil["info_kereta"]
    tgl = hasil["tanggal"]

    # ===================== SEKSI 1 — INFORMASI UMUM =====================
    section_head("ℹ️", "Informasi Umum", "Ringkasan kereta dan tanggal yang dipilih")

    status_hari = "Akhir Pekan" if tgl.weekday() in (5, 6) else "Hari Kerja"
    badge_khusus = ""
    if hasil["is_lebaran"]:
        badge_khusus += badge("Periode Lebaran", "info") + " "
    if hasil["is_libur_sekolah"]:
        badge_khusus += badge("Libur Sekolah", "info") + " "
    if hasil["is_libur_akhir_tahun"]:
        badge_khusus += badge("Libur Akhir Tahun", "info") + " "

    st.markdown(f"""
    <div class="kai-card kai-card-accent">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:14px;">
            <div>
                <div style="font-size:1.4rem; font-weight:800; color:{KAI_BLUE_900};">{hasil['nama_kereta']}</div>
                <div style="color:var(--text-secondary); margin-top:2px;">
                    {format_tanggal_indo(tgl)} &middot; {nama_hari(tgl)} &middot; {status_hari}
                </div>
            </div>
            <div style="text-align:right;">{badge_khusus if badge_khusus else badge("Hari Reguler", "info")}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        info_chip("Kapasitas Kursi", f"{info_k['kapasitas_kursi']:,} kursi")
    with c2:
        info_chip("Gerbong Tersedia", f"{info_k['gerbong_tersedia']} gerbong")
    with c3:
        info_chip("Rata-rata Historis", f"{info_k['rata_rata_historis']:,.0f} pnp/hari")
    with c4:
        info_chip("Periode Data Kereta", f"{info_k['periode_awal'].year}–{info_k['periode_akhir'].year}")


    st.markdown('<div class="rail-divider"></div>', unsafe_allow_html=True)

    # ============== SEKSI 2 — HASIL PREDIKSI & REKOMENDASI ==============
    section_head("📈", "Hasil Prediksi & Rekomendasi", "Estimasi penumpang dan tindak lanjut yang disarankan")

    level_warna = {"success": "#0f6e37", "warning": "#9c4d04", "danger": "#c0392b"}[hasil["level_kepadatan"]]

    colA, colB = st.columns(2)
    with colA:
        caption_aktual = ""
        if hasil["aktual"] is not None:
            selisih = hasil["prediksi"] - hasil["aktual"]
            caption_aktual = (
                f"Data aktual tercatat: {hasil['aktual']:,.0f} penumpang "
                f"(selisih prediksi {selisih:+,.0f})"
            )
        metric_card(
            "Prediksi Jumlah Penumpang",
            f"{hasil['prediksi']:,.0f} <span style='font-size:1rem; font-weight:600;'>orang</span>",
            caption=caption_aktual if caption_aktual else f"Rata-rata historis: {info_k['rata_rata_historis']:,.0f} orang",
        )
    with colB:
        metric_card(
            "Tingkat Kepadatan (Load Factor)",
            f"{hasil['load_factor']:.1f}%",
            value_color=level_warna,
            caption=f"Status: {badge(hasil['label_kepadatan'], hasil['level_kepadatan'])}",
        )

    fill_pct = min(100, hasil["load_factor"])
    st.markdown(f"""
    <div class="kai-card">
        <div class="metric-label" style="margin-bottom:8px;">Pemakaian Kapasitas Kursi</div>
        <div class="cap-track"><div class="cap-fill" style="width:{fill_pct}%; background:{level_warna};"></div></div>
        <div style="display:flex; justify-content:space-between; margin-top:14px;">
            <div><span class="metric-label">Gerbong Tersedia</span><br/><b>{info_k['gerbong_tersedia']} gerbong</b></div>
            <div><span class="metric-label">Gerbong Dibutuhkan</span><br/><b>{hasil['gerbong_dibutuhkan']} gerbong</b></div>
            <div><span class="metric-label">Tambahan Diperlukan</span><br/>
                <b style="color:{'#c0392b' if hasil['tambahan_gerbong'] > 0 else '#0f6e37'};">
                    {hasil['tambahan_gerbong']} gerbong
                </b>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**💡 Rekomendasi Tindakan**")
    rek_cols = st.columns(len(hasil["rekomendasi_list"]))
    for col, r in zip(rek_cols, hasil["rekomendasi_list"]):
        with col:
            recommendation_card(r["icon"], r["judul"], r["isi"], r["level"])

    # Grafik pola musiman bulanan kereta terpilih, dengan titik prediksi
    df_kereta = df[df["nama_kereta"] == hasil["nama_kereta"]]
    pola_bulanan = df_kereta.groupby("bulan")["jumlah_penumpang"].mean().reindex(range(1, 13))

    fig_musiman = go.Figure()
    fig_musiman.add_trace(go.Scatter(
        x=BULAN_INDO, y=pola_bulanan.values,
        mode="lines+markers", name="Rata-rata Historis",
        line=dict(color=KAI_BLUE_700, width=3), # Use KAI_BLUE_700
        marker=dict(size=6, color=KAI_BLUE_700), # Use KAI_BLUE_700 for markers
    ))
    fig_musiman.add_trace(go.Scatter(
        x=[BULAN_INDO[tgl.month - 1]], y=[hasil["prediksi"]],
        mode="markers", name="Prediksi Saat Ini",
        marker=dict(size=15, color=KAI_ORANGE_700, symbol="star"), # Use KAI_ORANGE_700
    ))
    fig_musiman.update_layout(title="Pola Musiman Rata-rata Penumpang per Bulan vs Prediksi")
    st.plotly_chart(styled_fig(fig_musiman, height=360), width="stretch")

    st.markdown('<div class="rail-divider"></div>', unsafe_allow_html=True)

    # ================ SEKSI 3 — EVALUASI MODEL & VISUALISASI ================M
    section_head(
        "📊", "Evaluasi Model & Visualisasi",
        "Performa umum model pada data uji (test set) — bukan akurasi prediksi tanggal di atas",
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        info_chip("MAE", f"{eval_metrics['mae']:,.1f} pnp")
    with m2:
        info_chip("RMSE", f"{eval_metrics['rmse']:,.1f} pnp")
    with m3:
        info_chip("MAPE", f"{eval_metrics['mape']:.1f}%")
    with m4:
        info_chip("R² Score", f"{eval_metrics['r2']:.3f}")

    st.caption(
        "MAE dan RMSE menunjukkan rata-rata selisih antara prediksi dan nilai aktual dalam satuan "
        "jumlah penumpang (semakin kecil semakin baik). MAPE menunjukkan persentase kesalahan "
        "rata-rata, dan R² menunjukkan seberapa besar variasi data yang berhasil dijelaskan model "
        "(rentang 0 sampai 1, semakin mendekati 1 semakin baik)."
    )

    colC, colD = st.columns(2)
    with colC:
        n_tampil = min(120, len(eval_results))
        subset_eval = eval_results.tail(n_tampil).reset_index(drop=True)
        fig_eval = go.Figure()
        fig_eval.add_trace(go.Scatter(
            y=subset_eval["Actual"], mode="lines", name="Aktual",
            line=dict(color=KAI_BLUE_700, width=2.2), # Use KAI_BLUE_700
        ))
        fig_eval.add_trace(go.Scatter(
            y=subset_eval["Prediction"], mode="lines", name="Prediksi",
            line=dict(color=KAI_ORANGE_600, width=2.2, dash="dot"), # Use KAI_ORANGE_600
        ))
        fig_eval.update_layout(title=f"Aktual vs Prediksi pada {n_tampil} Data Uji Terakhir")
        fig_eval.update_xaxes(title_text="Indeks Data Uji")
        fig_eval.update_yaxes(title_text="Jumlah Penumpang")
        st.plotly_chart(styled_fig(fig_eval), width="stretch")

    with colD:
        top_n = feature_importance.sort_values("Importance", ascending=False).head(10).copy()
        top_n["Label"] = top_n["Feature"].map(lambda x: LABEL_FITUR.get(x, x))
        top_n = top_n.sort_values("Importance")
        fig_imp = go.Figure(go.Bar(
            x=top_n["Importance"], y=top_n["Label"], orientation="h",
            marker_color=KAI_BLUE_700, # Use KAI_BLUE_700
        ))
        fig_imp.update_layout(title="10 Fitur Paling Berpengaruh")
        fig_imp.update_xaxes(title_text="Tingkat Kepentingan")
        st.plotly_chart(styled_fig(fig_imp), width="stretch")

    st.markdown("<br/>", unsafe_allow_html=True)
    if st.button("🔄 Prediksi Tanggal/Kereta Lain"):
        st.session_state.hasil_prediksi = None
        st.rerun()
