"""
Dashboard Streamlit - Klasifikasi Hujan Harian Kolkata dengan XGBoost
Jalankan:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay)
from xgboost import XGBClassifier

CSV_PATH = "kolkata_weather_jan_june_2026.csv"
FITUR_NUMERIK = ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
                 "wind_speed_10m_max", "wind_gusts_10m_max",
                 "wind_direction_10m_dominant", "sunshine_duration"]
NAMA_KELAS = ["Tidak Hujan", "Hujan"]

st.set_page_config(page_title="Klasifikasi Hujan Kolkata - XGBoost", page_icon="🌧️", layout="wide")
sns.set_theme(style="whitegrid")


def buat_musim(bulan: int) -> str:
    if bulan in (12, 1, 2):
        return "Dingin"
    if bulan in (3, 4, 5):
        return "Pra-Monsun"
    if bulan in (6, 7, 8, 9):
        return "Monsun"
    return "Pasca-Monsun"


@st.cache_data
def muat_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df["time"] = pd.to_datetime(df["time"])
    df["hujan"] = (df["weather_code"] >= 51).astype(int)   # 1 = Hujan, 0 = Tidak Hujan
    df["bulan"] = df["time"].dt.month
    df["musim"] = df["bulan"].apply(buat_musim)
    return df


@st.cache_resource
def latih_model(df: pd.DataFrame):
    # Fitur TANPA weather_code, precipitation_sum, rain_sum, precipitation_hours (data leakage)
    X = pd.get_dummies(df[FITUR_NUMERIK + ["bulan", "musim"]], columns=["musim"], dtype=int)
    y = df["hujan"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5,
                          random_state=42, eval_metric="logloss")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrik = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    }
    return model, list(X.columns), y_test, y_pred, y_proba, metrik


df = muat_data()
model, kolom_fitur, y_test, y_pred, y_proba, metrik = latih_model(df)

# ------------------------------------------------------------------ Judul
st.title("🌧️ Klasifikasi Hujan Harian Kota Kolkata dengan XGBoost")
st.markdown(
    "Proyek mata kuliah **Analisis Big Data**. Model **XGBoost Classifier** memprediksi apakah "
    "suatu hari **hujan** atau **tidak hujan** dari data cuaca harian Kolkata, India "
    "(sumber: Open-Meteo, 30 Jun 2021 – 30 Jun 2026). Target dibentuk dari kode cuaca WMO "
    "(`weather_code` ≥ 51 = hujan). Kolom yang membocorkan jawaban (`weather_code`, "
    "`precipitation_sum`, `rain_sum`, `precipitation_hours`) **tidak** dipakai sebagai fitur."
)

# ------------------------------------------------------------------ Ringkasan dataset
st.header("1. Ringkasan Dataset")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Jumlah data", f"{len(df):,}")
c2.metric("Jumlah fitur", len(kolom_fitur))
c3.metric("Hari hujan", f"{(df['hujan'] == 1).sum():,} ({df['hujan'].mean():.1%})")
c4.metric("Hari tidak hujan", f"{(df['hujan'] == 0).sum():,} ({1 - df['hujan'].mean():.1%})")
with st.expander("Lihat contoh data & fitur yang dipakai"):
    st.dataframe(df.head(10))
    st.write("Fitur model:", kolom_fitur)

# ------------------------------------------------------------------ EDA
st.header("2. Eksplorasi Data (EDA)")
tab1, tab2, tab3 = st.tabs(["Proporsi kelas", "Distribusi fitur", "Korelasi"])

with tab1:
    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(x="hujan", data=df, palette=["#f4a261", "#2a9d8f"], ax=ax)
        ax.set_xticklabels(NAMA_KELAS)
        ax.set_xlabel("")
        ax.set_title("Jumlah Kelas Target")
        st.pyplot(fig)
    with col_b:
        fig, ax = plt.subplots(figsize=(5, 4))
        df["hujan"].map(dict(enumerate(NAMA_KELAS))).value_counts().plot.pie(
            autopct="%1.1f%%", colors=["#2a9d8f", "#f4a261"], ax=ax)
        ax.set_ylabel("")
        ax.set_title("Proporsi Kelas Target")
        st.pyplot(fig)

with tab2:
    fitur_pilih = st.selectbox("Pilih fitur", FITUR_NUMERIK)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(data=df, x=fitur_pilih, hue="hujan", kde=True, bins=30,
                 palette=["#f4a261", "#2a9d8f"], ax=ax)
    ax.set_title(f"Distribusi {fitur_pilih} (0 = Tidak Hujan, 1 = Hujan)")
    st.pyplot(fig)

with tab3:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[FITUR_NUMERIK + ["hujan"]].corr(), annot=True, fmt=".2f",
                cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Heatmap Korelasi")
    st.pyplot(fig)

# ------------------------------------------------------------------ Performa
st.header("3. Performa Model")
cols = st.columns(5)
for col, (nama, nilai) in zip(cols, metrik.items()):
    col.metric(nama, f"{nilai:.3f}")
st.table(pd.DataFrame({"Metrik": list(metrik.keys()),
                       "Nilai": [round(v, 4) for v in metrik.values()]}).set_index("Metrik"))
st.caption("Evaluasi pada 20% data uji (stratified split, random_state=42). "
           "Parameter: n_estimators=100, learning_rate=0.1, max_depth=5.")

# ------------------------------------------------------------------ Visualisasi hasil
st.header("4. Visualisasi Hasil")
col_a, col_b, col_c = st.columns(3)

with col_a:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                           display_labels=NAMA_KELAS).plot(cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

with col_b:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, color="#2a9d8f", lw=2, label=f"XGBoost (AUC = {metrik['ROC-AUC']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Acak")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    st.pyplot(fig)

with col_c:
    imp = pd.Series(model.feature_importances_, index=kolom_fitur).sort_values()
    fig, ax = plt.subplots(figsize=(5, 4.5))
    imp.plot.barh(color="#2a9d8f", ax=ax)
    ax.set_title("Feature Importance")
    st.pyplot(fig)

# ------------------------------------------------------------------ Form prediksi
st.header("5. Coba Prediksi Manual")
st.write("Masukkan kondisi cuaca, lalu model akan memprediksi apakah hari itu hujan.")

with st.form("form_prediksi"):
    f1_, f2_, f3_ = st.columns(3)
    with f1_:
        t_mean = st.slider("Suhu rata-rata (°C)", 10.0, 40.0, 28.0, 0.1)
        t_max = st.slider("Suhu maksimum (°C)", 15.0, 45.0, 32.0, 0.1)
        t_min = st.slider("Suhu minimum (°C)", 5.0, 35.0, 25.0, 0.1)
    with f2_:
        w_speed = st.slider("Kecepatan angin maks (km/jam)", 0.0, 50.0, 14.0, 0.1)
        w_gust = st.slider("Hembusan angin maks (km/jam)", 0.0, 90.0, 30.0, 0.1)
        w_dir = st.slider("Arah angin dominan (derajat)", 0, 360, 190)
    with f3_:
        sun_jam = st.slider("Lama sinar matahari (jam)", 0.0, 13.0, 9.0, 0.1)
        bulan = st.selectbox("Bulan", list(range(1, 13)), index=6)
    kirim = st.form_submit_button("Prediksi")

if kirim:
    musim = buat_musim(bulan)
    baris = {"temperature_2m_mean": t_mean, "temperature_2m_max": t_max,
             "temperature_2m_min": t_min, "wind_speed_10m_max": w_speed,
             "wind_gusts_10m_max": w_gust, "wind_direction_10m_dominant": w_dir,
             "sunshine_duration": sun_jam * 3600,   # dataset menyimpan detik
             "bulan": bulan, "musim": musim}
    X_in = pd.get_dummies(pd.DataFrame([baris]), columns=["musim"], dtype=int)
    X_in = X_in.reindex(columns=kolom_fitur, fill_value=0)   # samakan kolom dengan data latih

    if t_min > t_mean or t_mean > t_max:
        st.warning("Suhu minimum, rata-rata, dan maksimum sebaiknya berurutan (min ≤ rata-rata ≤ maks).")

    prob = float(model.predict_proba(X_in)[0, 1])
    if prob >= 0.5:
        st.success(f"Prediksi: **HUJAN** 🌧️ (probabilitas {prob:.1%})")
    else:
        st.info(f"Prediksi: **TIDAK HUJAN** ☀️ (probabilitas hujan {prob:.1%})")
    st.progress(prob)
    st.caption(f"Musim terdeteksi: {musim}")