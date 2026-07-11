import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# --- Set Konfigurasi Halaman ---
st.set_page_config(page_title="Prediksi Konversi Ordered", layout="wide")

st.title("🛍️ Aplikasi Prediksi Perilaku Pengguna (Ordered vs Not Ordered)")

# --- 1. Load Data Ringkas & Train Model ---
@st.cache_resource
def train_model():
    # Menembak langsung file CSV via raw URL GitHub agar bebas dari FileNotFoundError lokal
    url = "https://raw.githubusercontent.com/Raihansatr/tugas-data-mining/main/training_sample.csv"
    
    # Membaca data langsung dari internet
    df = pd.read_csv(url, nrows=20000, sep=None, engine='python')
    df.columns = df.columns.str.strip()
    
    fitur = [
        'basket_icon_click', 'basket_add_list', 'basket_add_detail', 'sort_by',
        'image_picker', 'account_page_click', 'promo_banner_click', 'detail_wishlist_add',
        'list_size_dropdown', 'closed_minibasket_click', 'checked_delivery_detail',
        'checked_returns_detail', 'sign_in', 'saw_checkout', 'saw_sizecharts',
        'saw_delivery', 'saw_account_upgrade', 'saw_homepage', 'device_mobile',
        'device_computer', 'device_tablet', 'returning_user', 'loc_uk'
    ]
    
    fitur_fix = [f for f in fitur if f in df.columns]
    if len(fitur_fix) == 0:
        X = df.iloc[:, 1:24]
        y = df.iloc[:, 24]
        fitur_fix = list(X.columns)
    else:
        X = df[fitur_fix]
        y = df['ordered'] if 'ordered' in df.columns else df.iloc[:, -1]
    
    model = RandomForestClassifier(n_estimators=20, random_state=42, max_depth=10)
    model.fit(X, y)
    return model, fitur_fix

# Jalankan fungsi training
with st.spinner("Sedang memuat data optimasi, mohon tunggu..."):
    model, fitur_training = train_model()
st.success("Model Machine Learning siap digunakan!")

# --- 2. Membuat Form Input Pengguna ---
st.markdown("---")
st.subheader("📊 Masukkan Data Aktivitas Pengunjung")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🛒 Aktivitas Keranjang")
    basket_icon_click = st.selectbox("Klik Ikon Keranjang?", [0, 1])
    basket_add_list = st.selectbox("Tambah ke List Keranjang?", [0, 1])
    basket_add_detail = st.selectbox("Tambah dari Detail Keranjang?", [0, 1])
    closed_minibasket_click = st.selectbox("Menutup Mini-basket?", [0, 1])
    detail_wishlist_add = st.selectbox("Tambah ke Wishlist?", [0, 1])

with col2:
    st.markdown("### 📋 Aktivitas Halaman")
    sort_by = st.selectbox("Menggunakan Fitur Sortir?", [0, 1])
    image_picker = st.selectbox("Menggunakan Image Picker?", [0, 1])
    account_page_click = st.selectbox("Buka Halaman Akun?", [0, 1])
    promo_banner_click = st.selectbox("Klik Banner Promo?", [0, 1])
    list_size_dropdown = st.selectbox("Membuka Dropdown Ukuran?", [0, 1])
    sign_in = st.selectbox("Melakukan Sign In?", [0, 1])

with col3:
    st.markdown("### ⚙️ Proses Checkout & Perangkat")
    saw_checkout = st.selectbox("Melihat Halaman Checkout?", [0, 1])
    checked_delivery_detail = st.selectbox("Cek Detail Pengiriman?", [0, 1])
    saw_delivery = st.selectbox("Melihat Informasi Pengiriman?", [0, 1])
    returning_user = st.selectbox("Pengguna Lama (Returning)?", [0, 1])
    
    device = st.radio("Perangkat yang digunakan:", ["Mobile", "Computer", "Tablet"])
    device_mobile = 1 if device == "Mobile" else 0
    device_computer = 1 if device == "Computer" else 0
    device_tablet = 1 if device == "Tablet" else 0
    
    checked_returns_detail = 0
    saw_sizecharts = 0
    saw_account_upgrade = 0
    saw_homepage = 0
    loc_uk = 1

# --- 3. Proses Prediksi Ketika Tombol Diklik ---
st.markdown("---")
if st.button("🚀 Jalankan Prediksi Ordered", type="primary"):
    
    raw_input = {
        'basket_icon_click': basket_icon_click,
        'basket_add_list': basket_add_list,
        'basket_add_detail': basket_add_detail,
        'sort_by': sort_by,
        'image_picker': image_picker,
        'account_page_click': account_page_click,
        'promo_banner_click': promo_banner_click,
        'detail_wishlist_add': detail_wishlist_add,
        'list_size_dropdown': list_size_dropdown,
        'closed_minibasket_click': closed_minibasket_click,
        'checked_delivery_detail': checked_delivery_detail,
        'checked_returns_detail': checked_returns_detail,
        'sign_in': sign_in,
        'saw_checkout': saw_checkout,
        'saw_sizecharts': saw_sizecharts,
        'saw_delivery': saw_delivery,
        'saw_account_upgrade': saw_account_upgrade,
        'saw_homepage': saw_homepage,
        'device_mobile': device_mobile,
        'device_computer': device_computer,
        'device_tablet': device_tablet,
        'returning_user': returning_user,
        'loc_uk': loc_uk
    }
    
    X_input = pd.DataFrame([raw_input])
    
    for col in fitur_training:
        if col not in X_input.columns:
            X_input[col] = 0
            
    X_input = X_input[fitur_training]
    
    prediksi = model.predict(X_input)
    probabilitas = model.predict_proba(X_input)
    
    st.subheader("🏁 Hasil Analisis Prediksi")
    
    if prediksi[0] == 1:
        st.success("🎉 **Prediksi: USER BAKAL ORDERED!**")
        st.write(f"Kemungkinan melakukan pemesanan (Ordered): **{probabilitas[0][1] * 100:.2f}%**")
    else:
        st.error("❌ **Prediksi: USER TIDAK ORDERED**")
        st.write(f"Kemungkinan tidak melakukan pemesanan: **{probabilitas[0][0] * 100:.2f}%**")
