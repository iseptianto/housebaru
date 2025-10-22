# streamlit_app/app.py
import os
import pandas as pd
import streamlit as st
import requests
from translations import TRANSLATIONS

# Page configuration
st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="wide")

# Language selector in the top right
lang_col1, lang_col2 = st.columns([4, 1])
with lang_col2:
    lang = st.selectbox(
        "",
        options=['🇺🇸 English', '🇮🇩 Indonesia'],
        index=0,
        key='language'
    )
    current_lang = 'en' if lang.startswith('🇺🇸') else 'id'

# Translation helper
def t(key):
    return TRANSLATIONS[current_lang].get(key, key)

# ====== PENGATURAN ======
API_URL_DEFAULT = os.getenv("API_URL", "http://localhost:8000")
CSV_PATH_DEFAULT = os.getenv("CSV_PATH", "final.csv")

with st.sidebar:
    st.header(f"⚙️ {t('settings')}")
    api_url = st.text_input(
        t('api_url'), 
        API_URL_DEFAULT, 
        help=t('api_url_help')
    )
    timeout = st.number_input(t('timeout'), min_value=5, max_value=60, value=20, step=1)
    csv_path = st.text_input(t('csv_path'), CSV_PATH_DEFAULT)
    csv_upload = st.file_uploader(t('csv_upload'), type=["csv"])

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
    error_loading = t('error_loading').format(str(e))

# Main title
st.title(t('title'))
st.write(t('subtitle'))

st.title("🏠 House Price Prediction")
st.markdown("A simple MLOps demonstration project for real-time house price prediction")

if error_loading:
    st.error(f"Gagal load opsi dari CSV: {error_loading}")
    st.info("Silakan periksa path CSV di sidebar atau upload file `final.csv`.")
    st.stop()

# Create two columns for input and results
col_input, col_results = st.columns([2, 3])

# ====== FORM INPUT ======
st.subheader(f"🏘️ {t('input_section')}")

# --- Building and Land Area sliders ---
st.markdown(f"### 📐 {t('square_footage')} & {t('land_area')}")
col_lb, col_lt = st.columns(2)
with col_lb:
    luas_bangunan = st.slider(t('square_footage'), 10, 1000, 120, 10)
with col_lt:
    luas_tanah = st.slider(t('land_area'), 10, 2000, 150, 10)

# --- Bedrooms and Bathrooms ---
st.markdown(f"### 🛏️ {t('bedrooms')} & {t('bathrooms')}")
col_kt, col_km = st.columns(2)
with col_kt:
    kamar_tidur = st.selectbox(t('bedrooms'), [1, 2, 3, 4, 5, 6, 7], index=2)
with col_km:
    kamar_mandi = st.selectbox(t('bathrooms'), [1, 2, 3, 4, 5, 6], index=1)

# --- Province and City/District ---
st.markdown(f"### 🌍 {t('province')} & {t('city')}")
col1, col2 = st.columns(2)
with col1:
    default_prov = provinces.index("Jawa Barat") if "Jawa Barat" in provinces else 0
    provinsi = st.selectbox(t('province'), provinces, index=default_prov)
with col2:
    cities = prov2cities.get(provinsi, [])
    kota_kab = st.selectbox(t('city'), cities, index=0 if cities else None)

# --- Property type ---
tipe = st.selectbox(t('type'), ["rumah", "apartemen"])

# --- Ratio otomatis ---
auto_ratio = st.checkbox("Hitung otomatis ‘ratio_bangunan ruma’ = LB / LT", value=True)
ratio_bangunan = round(luas_bangunan / luas_tanah, 3) if auto_ratio and luas_tanah else None
if ratio_bangunan is not None:
    st.caption(f"📏 ratio_bangunan ruma (auto): {ratio_bangunan}")

# ====== SECTION PREDIKSI ======
st.markdown("---")

# Create the results section
st.markdown("---")
st.markdown(f"### {t('prediction_results')}")

# Create two columns for metrics
col1, col2 = st.columns(2)

with col1:
    price_placeholder = st.empty()
    price_placeholder.metric(t('predicted_price'), "---")

with col2:
    confidence_placeholder = st.empty()
    confidence_placeholder.metric(t('confidence_score'), "---")

def make_prediction(data):
    """Make API call with retry logic and proper error handling."""
    for attempt in range(3):  # 3 retries
        try:
            with st.spinner(t('predicting')):
                response = requests.post(
                    f"{api_url}/predict",
                    json=data,
                    timeout=timeout
                )
                response.raise_for_status()
                result = response.json()
                
                # Update metrics with results
                price_placeholder.metric(
                    t('predicted_price'),
                    f"Rp {result['prediction']:,.0f}"
                )
                confidence_placeholder.metric(
                    t('confidence_score'),
                    f"{result.get('confidence', 92)}%"
                )
                
                # Show prediction range
                low = result['prediction'] * 0.9
                high = result['prediction'] * 1.1
                st.success(f"💰 {t('price_range')}: Rp {low:,.0f} - Rp {high:,.0f}")
                
                return result
                
        except requests.exceptions.Timeout:
            if attempt == 2:  # Last attempt
                st.error(t('error_timeout'))
                return None
            time.sleep(1)  # Wait before retry
        except requests.exceptions.RequestException as e:
            st.error(t('error_api').format(str(e)))
            return None
        except Exception as e:
            st.error(t('error_prediction').format(str(e)))
            return None
    return None

# Cache successful predictions
@st.cache_data(ttl=3600)  # Cache for 1 hour
def cached_prediction(data_key):
    return make_prediction(data_key)

# Prediction button
predict_button = st.button(f"🔮 {t('predict_button')}", use_container_width=True)

if predict_button:
    # Input validation
    if not all([luas_bangunan, luas_tanah, kamar_tidur, kamar_mandi, provinsi, kota_kab, tipe]):
        st.error(t('error_missing_inputs'))
    else:
        # Prepare request data
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
            payload["ratio_bangunan_rumah"] = float(ratio_bangunan)

        try:
            with st.spinner(t('predicting')):
                resp = requests.post(f"{api_url}/predict", json=payload, timeout=timeout)
                if resp.ok:
                    result = resp.json()
                    price_placeholder.metric(
                        t('predicted_price'),
                        f"Rp {result['prediction']:,.0f}"
                    )
                    confidence_placeholder.metric(
                        t('confidence_score'),
                        f"{result.get('confidence', 92)}%"
                    )
                    
                    # Show prediction range
                    low = result['prediction'] * 0.9
                    high = result['prediction'] * 1.1
                    st.info(f"💰 {t('price_range')}: Rp {low:,.0f} - Rp {high:,.0f}")
                else:
                    st.error(f"API error [{resp.status_code}]: {resp.text}")
        except requests.exceptions.Timeout:
            st.error(t('error_timeout'))
        except Exception as e:
            st.error(f"{t('error_api')}: {str(e)}")
