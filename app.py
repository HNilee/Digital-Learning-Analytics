import streamlit as st
import pandas as pd
import joblib

#Konfigurasi Halaman
st.set_page_config(page_title="MOOC Learning Predictor", layout="wide")
st.title("Prediksi Efisiensi Belajar MOOC")
st.write("Sesuaikan metrik interaksi siswa di bawah ini untuk melihat prediksi efisiensi belajar secara real-time.")

with st.expander("ℹ️ Info Range Skor Prediksi Efisiensi"):
    st.markdown("""
    Model ini menghitung seberapa efisien seorang siswa belajar (mencapai penguasaan materi dalam waktu yang optimal). 
    Berdasarkan data 100,000 siswa, nilai rata-rata efisiensi adalah **~14.8**. Berikut adalah indikator statusnya:
    - 🟢 **Sangat Efisien (Skor $\ge$ 25.0)**: Siswa belajar dengan sangat cepat, terarah, dan mengikuti rekomendasi sistem.
    - 🟡 **Cukup / Rata-rata (Skor 15.0 - 24.9)**: Siswa belajar dengan kecepatan normal seperti kebanyakan pengguna lain.
    - 🔴 **Rendah (Skor < 15.0)**: Siswa menghabiskan terlalu banyak waktu tanpa progres nyata, atau berisiko *dropout*.
    """)

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
    gamification_engagement = st.number_input("Poin Gamifikasi (0-100)", min_value=0.0, max_value=100.0, value=39.9)
    forum_posts = st.number_input("Jumlah Postingan Forum", min_value=0, max_value=200, value=25)
    video_completion_pct = st.slider("Persentase Video Selesai (%)", 0.0, 100.0, 85.0)

with col3:
    st.markdown("**Metrik Akademik & Rekomendasi**")
    skill_pre_score = st.number_input("Skor Awal (Pre-score)", min_value=0.0, max_value=100.0, value=85.0)
    content_recom_followed = st.slider("Kepatuhan Rekomendasi Konten (%)", 0.0, 100.0, 75.0)

# Statistik Scaler (diambil dari data training)
# Note: Model XGBoost ini dilatih pada data yang sudah di-scale dengan StandardScaler
scaler_stats = {
    'total_learning_hours': {'mean': 44.954331, 'std': 40.708444},
    'daily_app_minutes': {'mean': 45.403519, 'std': 22.550576},
    'engagement_consistency': {'mean': 0.490421, 'std': 0.199641},
    'gamification_engagement': {'mean': 39.971342, 'std': 19.963302},
    'forum_posts': {'mean': 1.745746, 'std': 2.202564},
    'video_completion_pct': {'mean': 40.853429, 'std': 18.718410},
    'skill_pre_score': {'mean': 44.964121, 'std': 17.586624},
    'content_recommendations_followed': {'mean': 56.067160, 'std': 18.472460},
    'total_interactions': {'mean': 58.846994, 'std': 18.668226}
}

#Logika Prediksi
if st.button("Prediksi Skor Efisiensi", use_container_width=True):
    
    #Copy template base_data agar struktur kolom sama persis
    input_df = base_data.copy()
    
    # 1. Update Fitur Engineered (sebelum di-scale)
    # Di notebook: total_interactions = forum_posts + peer_review_given + content_recommendations_followed
    # Kita asumsikan peer_review_given adalah rata-rata (0.95)
    total_interactions = forum_posts + 0.95 + content_recom_followed
    
    # 2. Masukkan nilai ke DataFrame dan lakukan Scaling
    # Rumus StandardScaler: (x - mean) / std
    input_df['total_learning_hours'] = (total_learning_hours - scaler_stats['total_learning_hours']['mean']) / scaler_stats['total_learning_hours']['std']
    input_df['daily_app_minutes'] = (daily_app_minutes - scaler_stats['daily_app_minutes']['mean']) / scaler_stats['daily_app_minutes']['std']
    input_df['engagement_consistency'] = (engagement_consistency - scaler_stats['engagement_consistency']['mean']) / scaler_stats['engagement_consistency']['std']
    input_df['gamification_engagement'] = (gamification_engagement - scaler_stats['gamification_engagement']['mean']) / scaler_stats['gamification_engagement']['std']
    input_df['forum_posts'] = (forum_posts - scaler_stats['forum_posts']['mean']) / scaler_stats['forum_posts']['std']
    input_df['video_completion_pct'] = (video_completion_pct - scaler_stats['video_completion_pct']['mean']) / scaler_stats['video_completion_pct']['std']
    input_df['skill_pre_score'] = (skill_pre_score - scaler_stats['skill_pre_score']['mean']) / scaler_stats['skill_pre_score']['std']
    input_df['content_recommendations_followed'] = (content_recom_followed - scaler_stats['content_recommendations_followed']['mean']) / scaler_stats['content_recommendations_followed']['std']
    input_df['total_interactions'] = (total_interactions - scaler_stats['total_interactions']['mean']) / scaler_stats['total_interactions']['std']
    
    # Fitur score_improvement dibiarkan di nilai rata-rata (0 di data ter-scale)
    # karena kita tidak punya input skill_post_score di app.
    
    #Lakukan Prediksi
    prediksi = model.predict(input_df)
    skor_hasil = prediksi[0]
    
    #Tampilan Hasil
    st.divider()
    st.subheader("Hasil Prediksi Regresi XGBoost")
    
    #Menentukan status warna berdasarkan skor (Disesuaikan dengan distribusi data: Mean ~14.8)
    if skor_hasil >= 25.0:
        st.success(f"### Predicted Learning Efficiency: {skor_hasil:.2f} (Sangat Efisien)")
        st.info("Siswa ini sangat efisien: Mencapai target pembelajaran dengan waktu yang optimal. Sangat direkomendasikan untuk program akselerasi!")
    elif skor_hasil >= 15.0:
        st.warning(f"### Predicted Learning Efficiency: {skor_hasil:.2f} (Cukup)")
        st.info("Performa rata-rata. Siswa belajar dengan ritme yang stabil namun masih bisa ditingkatkan.")
    else:
        st.error(f"### Predicted Learning Efficiency: {skor_hasil:.2f} (Rendah / Risiko Dropout)")
        st.info("🚨 Efisiensi rendah: Siswa menghabiskan terlalu banyak waktu tanpa progres yang signifikan atau kurang interaksi.")