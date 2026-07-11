import streamlit as st
import pandas as pd
import pickle
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "best_model.pkl")

FEATURES = [
    "basket_icon_click", "basket_add_list", "basket_add_detail", "sort_by",
    "image_picker", "account_page_click", "promo_banner_click",
    "detail_wishlist_add", "list_size_dropdown", "closed_minibasket_click",
    "checked_delivery_detail", "checked_returns_detail", "sign_in",
    "saw_checkout", "saw_sizecharts", "saw_delivery", "saw_account_upgrade",
    "saw_homepage", "device_mobile", "device_computer", "device_tablet",
    "returning_user", "loc_uk",
]

# Fitur ini biasanya bernilai 0/1 (behavior flags / device flags)
BINARY_FEATURES = [f for f in FEATURES if f not in ("sort_by",)]


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


model = load_model()

st.set_page_config(page_title="Conversion Predictor", page_icon="🛒", layout="centered")
st.title("🛒 User Conversion Predictor")
st.write(
    "Prediksi kemungkinan konversi/checkout user berdasarkan perilaku browsing "
    "di website (23 fitur)."
)

tab1, tab2 = st.tabs(["🔘 Input Manual", "📄 Upload CSV (Batch)"])

with tab1:
    st.subheader("Isi perilaku user")
    input_data = {}
    cols = st.columns(3)
    for i, feat in enumerate(FEATURES):
        with cols[i % 3]:
            if feat in BINARY_FEATURES:
                input_data[feat] = st.selectbox(feat, [0, 1], key=feat)
            else:
                input_data[feat] = st.number_input(feat, value=0, key=feat)

    if st.button("Predict", type="primary"):
        df = pd.DataFrame([input_data])[FEATURES]
        pred = model.predict(df)[0]
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)[0]

        st.success(f"Prediksi: **{pred}**")
        if proba is not None:
            st.write("Probabilitas per kelas:")
            st.bar_chart(pd.Series(proba, index=model.classes_))

with tab2:
    st.subheader("Upload file CSV")
    st.caption(f"CSV harus memiliki kolom: {', '.join(FEATURES)}")
    uploaded = st.file_uploader("Pilih file CSV", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        missing = set(FEATURES) - set(df.columns)
        if missing:
            st.error(f"Kolom berikut hilang di CSV: {missing}")
        else:
            preds = model.predict(df[FEATURES])
            df["prediction"] = preds
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(df[FEATURES])
                for idx, cls in enumerate(model.classes_):
                    df[f"proba_{cls}"] = proba[:, idx]

            st.write("Hasil prediksi:")
            st.dataframe(df)

            csv_out = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download hasil (CSV)",
                data=csv_out,
                file_name="predictions.csv",
                mime="text/csv",
            )
