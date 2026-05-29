import streamlit as st
import pandas as pd
import joblib

#Konfigurasi Halaman
st.set_page_config(page_title="MOOC Learning Predictor", layout="wide")
st.title("Prediksi Efisiensi Belajar MOOC")
st.write("Sesuaikan metrik interaksi siswa di bawah ini untuk melihat prediksi efisiensi belajar secara real-time.")

#Load Model & Template Data
@st.cache_resource
def load_assets():
    model = joblib.load('xgboost_mooc_model.pkl')
    base_data = joblib.load('base_data.pkl')
    return model, base_data

model, base_data = load_assets()

#Form Input
st.subheader("Masukkan Metrik Perilaku Siswa")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Metrik Waktu & Dedikasi**")
    total_learning_hours = st.number_input("Total Jam Belajar (Hours)", min_value=0.0, max_value=500.0, value=180.5)
    daily_app_minutes = st.number_input("Rata-rata Menit Harian", min_value=0.0, max_value=300.0, value=45.0)
    engagement_consistency = st.slider("Konsistensi Interaksi (0 - 1)", 0.0, 1.0, 0.8)

with col2:
    st.markdown("**Metrik Interaksi Sistem**")
    gamification_engagement = st.number_input("Poin Gamifikasi", min_value=0, max_value=200000, value=75000)
    forum_posts = st.number_input("Jumlah Postingan Forum", min_value=0, max_value=200, value=25)
    video_completion_pct = st.slider("Persentase Video Selesai (%)", 0.0, 100.0, 85.0)

with col3:
    st.markdown("**Metrik Akademik & Rekomendasi**")
    skill_pre_score = st.number_input("Skor Awal (Pre-score)", min_value=0.0, max_value=100.0, value=85.0)
    content_recom_followed = st.slider("Kepatuhan Rekomendasi Konten (%)", 0.0, 100.0, 75.0)

#Logika Prediksi
if st.button("Prediksi Skor Efisiensi", use_container_width=True):
    
    #Copy template base_data agar struktur kolom (termasuk dummy variable) sama persis
    input_df = base_data.copy()
    
    #Timpa nilai rata-rata di template dengan nilai input dari pengguna
    input_df['total_learning_hours'] = total_learning_hours
    input_df['daily_app_minutes'] = daily_app_minutes
    input_df['engagement_consistency'] = engagement_consistency
    input_df['gamification_engagement'] = gamification_engagement
    input_df['forum_posts'] = forum_posts
    input_df['video_completion_pct'] = video_completion_pct
    input_df['skill_pre_score'] = skill_pre_score
    input_df['content_recommendations_followed'] = content_recom_followed
    
    #Lakukan Prediksi
    prediksi = model.predict(input_df)
    skor_hasil = prediksi[0]
    
    #Tampilan Hasil
    st.divider()
    st.subheader("Hasil Prediksi Regresi XGBoost")
    
    #Menentukan status warna berdasarkan skor
    if skor_hasil >= 70.0:
        st.success(f"### Predicted Learning Efficiency: {skor_hasil:.2f} (Sangat Efisien)")
        st.info("Siswa ini memanfaatkan platform dengan maksimal. Berikan lencana (badge) penghargaan!")
    elif skor_hasil >= 40.0:
        st.warning(f"### Predicted Learning Efficiency: {skor_hasil:.2f} (Cukup)")
        st.info("Performa standar. Dorong siswa untuk mengikuti lebih banyak rekomendasi konten.")
    else:
        st.error(f"### Predicted Learning Efficiency: {skor_hasil:.2f} (Rendah / Risiko Dropout)")
        st.info("🚨 Siswa memiliki efisiensi rendah. Sistem peringatan dini harus mengirimkan notifikasi intervensi.")