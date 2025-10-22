# streamlit_app/app.py
import os
import pandas as pd
import streamlit as st
import requests

st.set_page_config(page_title="Prediksi Harga Rumah (OLX)", page_icon="🏠", layout="wide")

# ====== PENGATURAN ======
API_URL_DEFAULT = os.getenv("API_URL", "http://localhost:8000")
CSV_PATH_DEFAULT = os.getenv("CSV_PATH", "final.csv")

with st.sidebar:
    st.header("⚙️ Pengaturan")
    api_url = st.text_input(
        "FastAPI URL", 
        API_URL_DEFAULT, 
        help="Contoh: http://localhost:8000 atau http://fastapi:8000 (kalau via compose)"
    )
    timeout = st.number_input("Timeout (detik)", min_value=5, max_value=60, value=20, step=1)
    csv_path = st.text_input("CSV path", CSV_PATH_DEFAULT, help="Path ke final.csv")
    csv_upload = st.file_uploader("Atau upload final.csv di sini", type=["csv"])

@st.cache_data
def load_options_from_csv(_file_like_or_path):
    df = pd.read_csv(_file_like_or_path)
    df.columns = [c.strip() for c in df.columns]
    need = {"Provinsi", "Kota/Kab"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Kolom CSV kurang: {missing}. Pastikan ada 'Provinsi' dan 'Kota/Kab'.")
    provs = sorted(df["Provinsi"].dropna().astype(str).unique())
    mapping = (
        df.dropna(subset=["Provinsi", "Kota/Kab"])
          .astype({"Provinsi": "string", "Kota/Kab": "string"})
          .groupby("Provinsi")["Kota/Kab"]
          .apply(lambda s: sorted(s.dropna().unique().tolist()))
          .to_dict()
    )
    return provs, mapping

csv_source = csv_upload if csv_upload is not None else csv_path

provinces, prov2cities = [], {}
error_loading = None
try:
    provinces, prov2cities = load_options_from_csv(csv_source)
except Exception as e:
    error_loading = str(e)

st.title("🧮 Prediksi Harga Rumah (OLX)")

if error_loading:
    st.error(f"Gagal load opsi dari CSV: {error_loading}")
    st.info("Silakan periksa path CSV di sidebar atau upload file `final.csv`.")
    st.stop()

# ====== FORM INPUT ======
st.subheader("🏘️ Data Properti")

# --- Slider luas bangunan & tanah ---
st.markdown("### 📐 Luas Bangunan & Luas Tanah (m²)")
col_lb, col_lt = st.columns(2)
with col_lb:
    luas_bangunan = st.slider("Luas Bangunan (LB)", 10, 1000, 120, 10)
with col_lt:
    luas_tanah = st.slider("Luas Tanah (LT)", 10, 2000, 150, 10)

# --- Kamar tidur & kamar mandi ---
st.markdown("### 🛏️ Kamar Tidur & Kamar Mandi")
col_kt, col_km = st.columns(2)
with col_kt:
    kamar_tidur = st.selectbox("Kamar Tidur (KT)", [1, 2, 3, 4, 5, 6, 7], index=2)
with col_km:
    kamar_mandi = st.selectbox("Kamar Mandi (KM)", [1, 2, 3, 4, 5, 6], index=1)

# --- Provinsi dan Kota/Kab ---
st.markdown("### 🌍 Lokasi Properti")
col1, col2 = st.columns(2)
with col1:
    default_prov = provinces.index("Jawa Barat") if "Jawa Barat" in provinces else 0
    provinsi = st.selectbox("Provinsi", provinces, index=default_prov)
with col2:
    cities = prov2cities.get(provinsi, [])
    kota_kab = st.selectbox("Kota/Kab", cities, index=0 if cities else None)

# --- Tipe properti ---
tipe = st.selectbox("Tipe Properti", ["rumah", "apartemen"])

# --- Ratio otomatis ---
auto_ratio = st.checkbox("Hitung otomatis ‘ratio_bangunan ruma’ = LB / LT", value=True)
ratio_bangunan = round(luas_bangunan / luas_tanah, 3) if auto_ratio and luas_tanah else None
if ratio_bangunan is not None:
    st.caption(f"📏 ratio_bangunan ruma (auto): {ratio_bangunan}")

# ====== PREDIKSI ======
st.markdown("---")
if st.button("🔮 Prediksi Harga"):
    payload = {
        "LB": float(luas_bangunan),
        "LT": float(luas_tanah),
        "KM": int(kamar_mandi),
        "KT": int(kamar_tidur),
        "Provinsi": provinsi,
        "Kota/Kab": kota_kab,
        "Type": tipe
    }

    if ratio_bangunan is not None:
        payload["ratio_bangunan ruma"] = float(ratio_bangunan)

    try:
        resp = requests.post(f"{api_url}/predict", json=payload, timeout=timeout)
        if resp.ok:
            st.success(f"💰 Prediksi harga: Rp {resp.json().get('prediction', 0):,.0f}")
        else:
            st.error(f"API error [{resp.status_code}]: {resp.text}")
    except Exception as e:
        st.error(f"Gagal menghubungi API: {e}")
