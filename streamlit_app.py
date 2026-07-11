import streamlit as st
import pandas as pd
import pickle
import os
import plotly.graph_objects as go
import plotly.express as px

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Prediksi Konversi Ordered", page_icon="🛍️", layout="wide")

# --- Load Model ---
@st.cache_resource
def load_model():
    path_model = os.path.join(os.getcwd(), 'model', 'best_model.pkl')
    with open(path_model, 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

fitur_training = [
    'basket_icon_click', 'basket_add_list', 'basket_add_detail', 'sort_by',
    'image_picker', 'account_page_click', 'promo_banner_click', 'detail_wishlist_add',
    'list_size_dropdown', 'closed_minibasket_click', 'checked_delivery_detail',
    'checked_returns_detail', 'sign_in', 'saw_checkout', 'saw_sizecharts',
    'saw_delivery', 'saw_account_upgrade', 'saw_homepage', 'device_mobile',
    'device_computer', 'device_tablet', 'returning_user', 'loc_uk'
]

label_readable = {
    'basket_icon_click': 'Klik Ikon Keranjang',
    'basket_add_list': 'Tambah ke List Keranjang',
    'basket_add_detail': 'Tambah dari Detail Produk',
    'sort_by': 'Menggunakan Fitur Sortir',
    'image_picker': 'Menggunakan Image Picker',
    'account_page_click': 'Buka Halaman Akun',
    'promo_banner_click': 'Klik Banner Promo',
    'detail_wishlist_add': 'Tambah ke Wishlist',
    'list_size_dropdown': 'Membuka Dropdown Ukuran',
    'closed_minibasket_click': 'Menutup Mini-basket',
    'checked_delivery_detail': 'Cek Detail Pengiriman',
    'checked_returns_detail': 'Cek Detail Pengembalian',
    'sign_in': 'Sign In',
    'saw_checkout': 'Melihat Halaman Checkout',
    'saw_sizecharts': 'Melihat Size Chart',
    'saw_delivery': 'Melihat Info Pengiriman',
    'saw_account_upgrade': 'Melihat Account Upgrade',
    'saw_homepage': 'Melihat Homepage',
    'device_mobile': 'Perangkat Mobile',
    'device_computer': 'Perangkat Computer',
    'device_tablet': 'Perangkat Tablet',
    'returning_user': 'Pengguna Lama (Returning)',
    'loc_uk': 'Lokasi UK',
}

# --- Skenario Contoh ---
skenario = {
    "Kosongkan Semua": {f: 0 for f in fitur_training},
    "🔥 User Sangat Berpotensi Beli": {
        'basket_icon_click': 1, 'basket_add_list': 1, 'basket_add_detail': 1,
        'sign_in': 1, 'saw_checkout': 1, 'checked_delivery_detail': 1,
        'returning_user': 1, 'device_mobile': 1, 'loc_uk': 1,
    },
    "🚶 User Hanya Lihat-lihat": {
        'saw_homepage': 1, 'sort_by': 1, 'image_picker': 1,
        'device_computer': 1, 'loc_uk': 1,
    },
}

# --- Sidebar ---
with st.sidebar:
    st.header("ℹ️ Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini memprediksi apakah seorang pengunjung website e-commerce
    akan **melakukan order (checkout)** atau tidak, berdasarkan pola
    aktivitas klik (clickstream) selama sesi kunjungan.
    """)
    st.markdown("**Model:** Random Forest Classifier")
    st.markdown("**Dataset:** E-Commerce Clickstream Session")
    st.divider()
    st.markdown("**Kelompok Data Mining**")
    st.caption("RAIHAN SATRIA RAMADHAN, RANGGA PRANA MAHENDRA PUTRA Y, MUHAMMAD THUFAEL AMRULLAH, PETRUS EFRAIM DAPA WEA")
    st.divider()
    st.subheader("🧪 Coba Skenario Cepat")
    pilihan_skenario = st.selectbox("Pilih skenario:", list(skenario.keys()))

# --- Header ---
st.title("🛍️ Aplikasi Prediksi Perilaku Pengguna")
st.subheader("Ordered vs Not Ordered — E-Commerce Session Analysis")
st.success("✅ Model Machine Learning siap digunakan")

st.markdown("---")
st.markdown("### 📊 Masukkan Data Aktivitas Pengunjung")
st.caption("Centang aktivitas yang dilakukan pengunjung selama sesi browsing.")

nilai_default = skenario[pilihan_skenario]

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 🛒 Aktivitas Keranjang")
    basket_icon_click = int(st.checkbox("Klik Ikon Keranjang", value=bool(nilai_default.get('basket_icon_click', 0))))
    basket_add_list = int(st.checkbox("Tambah ke List Keranjang", value=bool(nilai_default.get('basket_add_list', 0))))
    basket_add_detail = int(st.checkbox("Tambah dari Detail Produk", value=bool(nilai_default.get('basket_add_detail', 0))))
    closed_minibasket_click = int(st.checkbox("Menutup Mini-basket", value=bool(nilai_default.get('closed_minibasket_click', 0))))
    detail_wishlist_add = int(st.checkbox("Tambah ke Wishlist", value=bool(nilai_default.get('detail_wishlist_add', 0))))

with col2:
    st.markdown("#### 📋 Aktivitas Halaman")
    sort_by = int(st.checkbox("Menggunakan Fitur Sortir", value=bool(nilai_default.get('sort_by', 0))))
    image_picker = int(st.checkbox("Menggunakan Image Picker", value=bool(nilai_default.get('image_picker', 0))))
    account_page_click = int(st.checkbox("Buka Halaman Akun", value=bool(nilai_default.get('account_page_click', 0))))
    promo_banner_click = int(st.checkbox("Klik Banner Promo", value=bool(nilai_default.get('promo_banner_click', 0))))
    list_size_dropdown = int(st.checkbox("Membuka Dropdown Ukuran", value=bool(nilai_default.get('list_size_dropdown', 0))))
    sign_in = int(st.checkbox("Sign In", value=bool(nilai_default.get('sign_in', 0))))

with col3:
    st.markdown("#### ⚙️ Checkout & Perangkat")
    saw_checkout = int(st.checkbox("Melihat Halaman Checkout", value=bool(nilai_default.get('saw_checkout', 0))))
    checked_delivery_detail = int(st.checkbox("Cek Detail Pengiriman", value=bool(nilai_default.get('checked_delivery_detail', 0))))
    saw_delivery = int(st.checkbox("Melihat Info Pengiriman", value=bool(nilai_default.get('saw_delivery', 0))))
    returning_user = int(st.checkbox("Pengguna Lama (Returning)", value=bool(nilai_default.get('returning_user', 0))))

    device_options = ["Mobile", "Computer", "Tablet"]
    default_device = "Mobile"
    if nilai_default.get('device_computer'):
        default_device = "Computer"
    elif nilai_default.get('device_tablet'):
        default_device = "Tablet"
    device = st.radio("Perangkat yang digunakan", device_options, index=device_options.index(default_device))
    device_mobile = 1 if device == "Mobile" else 0
    device_computer = 1 if device == "Computer" else 0
    device_tablet = 1 if device == "Tablet" else 0

    checked_returns_detail = nilai_default.get('checked_returns_detail', 0)
    saw_sizecharts = nilai_default.get('saw_sizecharts', 0)
    saw_account_upgrade = nilai_default.get('saw_account_upgrade', 0)
    saw_homepage = nilai_default.get('saw_homepage', 0)
    loc_uk = 1

st.caption("💡 Beberapa fitur tambahan (cek retur, size chart, account upgrade, homepage) mengikuti nilai dari skenario yang dipilih di sidebar.")

st.markdown("---")

if st.button("🚀 Jalankan Prediksi Ordered", type="primary", use_container_width=True):

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
    prob_order = probabilitas[0][1] * 100

    st.markdown("### 🏁 Hasil Analisis Prediksi")

    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        if prediksi[0] == 1:
            st.success("🎉 **USER DIPREDIKSI AKAN ORDER!**")
        else:
            st.error("❌ **USER DIPREDIKSI TIDAK ORDER**")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_order,
            number={'suffix': "%"},
            title={'text': "Probabilitas Order"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#2E86DE" if prob_order >= 50 else "#EE5253"},
                'steps': [
                    {'range': [0, 40], 'color': "#3d1f1f"},
                    {'range': [40, 70], 'color': "#3d3a1f"},
                    {'range': [70, 100], 'color': "#1f3d24"},
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 3},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with res_col2:
        st.markdown("#### 🔍 Faktor Paling Berpengaruh")
        rf_model = model.named_steps['model']
        importances = rf_model.feature_importances_
        imp_df = pd.DataFrame({
            'Fitur': [label_readable.get(f, f) for f in fitur_training],
            'Pengaruh': importances
        }).sort_values('Pengaruh', ascending=True).tail(8)

        fig_bar = px.bar(
            imp_df, x='Pengaruh', y='Fitur', orientation='h',
            color='Pengaruh', color_continuous_scale='Blues'
        )
        fig_bar.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20), coloraxis_showscale=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.info("📌 Grafik di atas menunjukkan 8 fitur paling berpengaruh terhadap prediksi model secara umum (berdasarkan feature importance Random Forest), bukan khusus untuk input ini saja.")

else:
    st.info("👆 Atur aktivitas pengunjung di atas (atau pilih skenario cepat di sidebar), lalu klik tombol **Jalankan Prediksi Ordered**.")

st.markdown("---")
st.caption("Dibuat untuk keperluan tugas kelompok Data Mining — Universitas Indraprasta PGRI")
