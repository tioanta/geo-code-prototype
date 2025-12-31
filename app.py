import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

# -----------------------------------------------------------------------------
# 1. KONFIGURASI HALAMAN
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Geo-Credit Strategic Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# [FIX UTAMA] Mematikan limit baris, tapi kita akan membatasi data secara manual
alt.data_transformers.disable_max_rows()

# Styling CSS
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 5px 5px 0 0; gap: 1px; padding-top: 10px; padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] { background-color: #FFFFFF; border-top: 3px solid #2e7bcf; }
    .metric-card { background-color: #ffffff; border: 1px solid #e0e0e0; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA ENGINE
# -----------------------------------------------------------------------------
@st.cache_data
def load_data_engine():
    try:
        # Load data
        df = pd.read_csv('Prototype Jawa Tengah.csv')
    except Exception as e:
        return None

    # --- CLEANING ---
    df.columns = [col.replace('potensi_wilayah_kel_podes_pdrb_sekda_current.', '') for col in df.columns]
    
    # Rename Column
    rename_map = {
        'nama_kabupaten': 'Kabupaten', 'nama_kecamatan': 'Kecamatan', 'nama_desa': 'Desa',
        'latitude_desa': 'lat', 'longitude_desa': 'lon',
        'total_pinjaman_kel': 'Total_Pinjaman', 'total_simpanan_kel': 'Total_Simpanan',
        'jumlah_keluarga_pengguna_listrik': 'Jumlah_KK',
        'attractiveness_index': 'Skor_Potensi',
        'max_tipe_usaha': 'Sektor_Dominan',
        'jumlah_lokasi_permukiman_kumuh': 'Risk_Kumuh',
        'bencana_alam': 'Risk_Bencana',
        'jumlah_perkelahian_masyarakat': 'Risk_Konflik'
    }
    # Hanya rename kolom yang ada
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
    
    # Fill NaN
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)
    
    # --- METRIC CALCULATION ---
    # Safe Division
    df['Jumlah_KK'] = df['Jumlah_KK'].replace(0, 1) 
    df['Loan_per_HH'] = (df['Total_Pinjaman'] / df['Jumlah_KK']) / 1_000_000 
    
    # Risk Scoring
    sat_risk = (df['Loan_per_HH'] / 50.0).clip(0, 1) * 100 
    max_pot = df['Skor_Potensi'].max() if df['Skor_Potensi'].max() > 0 else 1
    eco_risk = 100 - ((df['Skor_Potensi'] / max_pot) * 100)
    
    # Check flags existence before accessing
    r_kumuh = (df['Risk_Kumuh'] > 0).astype(int) if 'Risk_Kumuh' in df.columns else 0
    r_bencana = (df['Risk_Bencana'] > 0).astype(int) if 'Risk_Bencana' in df.columns else 0
    r_konflik = (df['Risk_Konflik'] > 0).astype(int) if 'Risk_Konflik' in df.columns else 0
    
    env_risk = (r_kumuh * 30) + (r_bencana * 20) + (r_konflik * 50)
               
    df['Final_Risk_Score'] = (0.3 * sat_risk) + (0.3 * eco_risk) + (0.4 * env_risk)
    
    # Clean Infinite
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    # Categorization
    def get_risk_cat(x):
        if x >= 60: return 'Critical'
        elif x >= 40: return 'High'
        elif x >= 20: return 'Medium'
        else: return 'Low'
    df['Risk_Category'] = df['Final_Risk_Score'].apply(get_risk_cat)

    # Strategy Quadrant
    avg_pot = df['Skor_Potensi'].mean()
    avg_sat = df['Loan_per_HH'].mean()
    def get_quad(row):
        if row['Skor_Potensi'] >= avg_pot and row['Loan_per_HH'] < avg_sat: return "Hidden Gem (Grow)"
        elif row['Skor_Potensi'] >= avg_pot and row['Loan_per_HH'] >= avg_sat: return "Red Ocean (Compete)"
        elif row['Skor_Potensi'] < avg_pot and row['Loan_per_HH'] >= avg_sat: return "High Risk (Stop)"
        else: return "Dormant (Monitor)"
    df['Strategy_Quadrant'] = df.apply(get_quad, axis=1)

    # Simulation Data
    np.random.seed(42)
    df['Sentiment_Score'] = np.random.uniform(3.5, 4.9, size=len(df))
    df['Review_Count'] = np.random.randint(10, 500, size=len(df))
    
    return df

# LOAD DATA
df = load_data_engine()
if df is None:
    st.error("Data 'Prototype Jawa Tengah.csv' tidak ditemukan atau corrupt.")
    st.stop()

# -----------------------------------------------------------------------------
# 3. SIDEBAR & FILTERING
# -----------------------------------------------------------------------------
st.sidebar.title("🎛️ Geo-Control")

# [FIX] Default Selectbox ke Index 0 (Satu Kabupaten) untuk load awal ringan
all_kab = sorted(df['Kabupaten'].unique())
selected_kab = st.sidebar.selectbox("Wilayah (Kabupaten)", all_kab, index=0)

# Filter Dataframe Utama
df_kab = df[df['Kabupaten'] == selected_kab]

# Filter Kecamatan
all_kec = sorted(df_kab['Kecamatan'].unique())
selected_kec = st.sidebar.multiselect("Filter Kecamatan", all_kec, default=all_kec)

if not selected_kec:
    st.warning("Mohon pilih minimal satu kecamatan.")
    st.stop()

df_filtered = df_kab[df_kab['Kecamatan'].isin(selected_kec)]

st.sidebar.markdown("---")
st.sidebar.caption(f"Data Terpilih: {len(df_filtered)} Desa")

# -----------------------------------------------------------------------------
# 4. FUNGSI KEAMANAN VISUALISASI (ANTI-CRASH)
# -----------------------------------------------------------------------------
def safe_chart_data(dataframe, required_cols, limit=4000):
    """
    1. Hanya ambil kolom yang diminta.
    2. Jika data > limit, ambil sample acak.
    """
    # Pastikan kolom ada di dataframe
    valid_cols = [c for c in required_cols if c in dataframe.columns]
    
    # Copy hanya data yang diperlukan (hemat memori)
    temp_df = dataframe[valid_cols].copy()
    
    # Sampling jika terlalu besar
    if len(temp_df) > limit:
        temp_df = temp_df.sample(limit)
        
    return temp_df

# -----------------------------------------------------------------------------
# 5. DASHBOARD TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Executive Summary", 
    "🚀 Growth Intelligence", 
    "⚖️ Saturation Deep-Dive", 
    "🛡️ Risk Guardian"
])

# ================= TAB 1: EXECUTIVE SUMMARY =================
with tab1:
    st.markdown(f"### 📋 Ringkasan Eksekutif: {selected_kab}")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Total Exposure", f"Rp {df_filtered['Total_Pinjaman'].sum()/1e9:,.1f} M")
    with c2: st.metric("Avg Risk Score", f"{df_filtered['Final_Risk_Score'].mean():.1f}/100")
    with c3: st.metric("Hidden Gems", f"{len(df_filtered[df_filtered['Strategy_Quadrant']=='Hidden Gem (Grow)'])}")
    with c4: st.metric("High Risk Areas", f"{len(df_filtered[df_filtered['Risk_Category'].isin(['High', 'Critical'])])}", delta_color="inverse")

    st.markdown("---")
    st.subheader("🌌 Peta Strategi Makro")
    
    # [FIX] Gunakan safe_chart_data
    cols_univ = ['Skor_Potensi', 'Loan_per_HH', 'Risk_Category', 'Total_Pinjaman', 'Desa', 'Kecamatan', 'Strategy_Quadrant']
    data_univ = safe_chart_data(df_filtered, cols_univ)
    
    chart_univ = alt.Chart(data_univ).mark_circle().encode(
        x=alt.X('Skor_Potensi', title='Potensi Ekonomi'),
        y=alt.Y('Loan_per_HH', title='Saturasi (Juta/KK)'),
        color=alt.Color('Risk_Category', scale=alt.Scale(domain=['Low','Medium','High','Critical'], range=['green','gold','orange','red'])),
        size=alt.Size('Total_Pinjaman', legend=None),
        tooltip=['Desa', 'Kecamatan', 'Risk_Category', 'Strategy_Quadrant']
    ).properties(height=450).interactive()
    
    st.altair_chart(chart_univ, use_container_width=True)

# ================= TAB 2: GROWTH INTELLIGENCE =================
with tab2:
    st.markdown("### 🚀 Potensi Pertumbuhan")
    col_g1, col_g2 = st.columns([2, 1])
    
    with col_g1:
        st.subheader("📍 Peta Potensi Ekonomi")
        # [FIX] Gunakan safe_chart_data
        cols_map = ['lon', 'lat', 'Skor_Potensi', 'Desa', 'Sektor_Dominan']
        data_map = safe_chart_data(df_filtered, cols_map)
        
        chart_map = alt.Chart(data_map).mark_circle(size=60).encode(
            longitude='lon', latitude='lat',
            color=alt.Color('Skor_Potensi', scale=alt.Scale(scheme='greens')),
            tooltip=['Desa', 'Skor_Potensi', 'Sektor_Dominan']
        ).project('mercator').properties(height=400)
        st.altair_chart(chart_map, use_container_width=True)
        
    with col_g2:
        st.subheader("🏭 Sektor Dominan")
        if 'Sektor_Dominan' in df_filtered.columns:
            sec_data = df_filtered['Sektor_Dominan'].value_counts().reset_index()
            sec_data.columns = ['Sektor', 'Jumlah']
            st.dataframe(sec_data, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("⭐ Market Sentiment (Top 10)")
    top_sent = df_filtered.nlargest(10, 'Sentiment_Score')
    # [FIX] Safe chart data
    data_sent = safe_chart_data(top_sent, ['Sentiment_Score', 'Desa', 'Review_Count', 'Sektor_Dominan'], limit=20)
    
    chart_sent = alt.Chart(data_sent).mark_bar().encode(
        x=alt.X('Sentiment_Score', domain=[0, 5]),
        y=alt.Y('Desa', sort='-x'),
        color='Review_Count',
        tooltip=['Sektor_Dominan', 'Sentiment_Score']
    )
    st.altair_chart(chart_sent, use_container_width=True)

# ================= TAB 3: SATURATION DEEP-DIVE =================
with tab3:
    st.markdown("### ⚖️ Analisis Saturasi")
    
    c_s1, c_s2 = st.columns(2)
    with c_s1:
        st.subheader("Matriks Persaingan")
        quad_data = df_filtered['Strategy_Quadrant'].value_counts().reset_index()
        quad_data.columns = ['Kategori', 'Jumlah']
        
        pie = alt.Chart(quad_data).mark_arc(outerRadius=120).encode(
            theta=alt.Theta("Jumlah", stack=True),
            color=alt.Color("Kategori"),
            tooltip=["Kategori", "Jumlah"]
        )
        st.altair_chart(pie, use_container_width=True)
        
    with c_s2:
        st.subheader("Top 10 Area Jenuh (Red Ocean)")
        # Filter manual untuk performa
        red_ocean = df_filtered[df_filtered['Strategy_Quadrant'].astype(str).str.contains('Red Ocean')]
        if not red_ocean.empty:
            red_ocean_sorted = red_ocean.nlargest(10, 'Loan_per_HH')
            st.dataframe(red_ocean_sorted[['Desa', 'Kecamatan', 'Loan_per_HH']], hide_index=True, use_container_width=True)
        else:
            st.info("Tidak ada area Red Ocean di wilayah ini.")

# ================= TAB 4: RISK GUARDIAN =================
with tab4:
    st.markdown("### 🛡️ Profil Risiko")
    
    col_r1, col_r2 = st.columns([2, 1])
    with col_r1:
        st.subheader("📍 Peta Zona Merah")
        # [FIX] Gunakan safe_chart_data
        cols_risk = ['lon', 'lat', 'Final_Risk_Score', 'Desa', 'Risk_Category']
        data_risk = safe_chart_data(df_filtered, cols_risk)
        
        r_map = alt.Chart(data_risk).mark_circle(size=60).encode(
            longitude='lon', latitude='lat',
            color=alt.Color('Final_Risk_Score', scale=alt.Scale(scheme='reds')),
            tooltip=['Desa', 'Final_Risk_Score', 'Risk_Category']
        ).project('mercator').properties(height=400)
        st.altair_chart(r_map, use_container_width=True)

    with col_r2:
        st.subheader("🚨 Watchlist (High Risk)")
        high_risk = df_filtered[df_filtered['Risk_Category'].isin(['High', 'Critical'])]
        if not high_risk.empty:
            high_risk_sorted = high_risk.nlargest(100, 'Final_Risk_Score')
            cols_show = ['Desa', 'Final_Risk_Score']
            if 'Risk_Konflik' in df_filtered.columns: cols_show.append('Risk_Konflik')
            if 'Risk_Bencana' in df_filtered.columns: cols_show.append('Risk_Bencana')
            
            st.dataframe(
                high_risk_sorted[cols_show], 
                hide_index=True, use_container_width=True
            )
        else:
            st.success("Tidak ada area berisiko tinggi.")

# Footer
st.markdown("---")
st.caption("Geo-Credit Intelligence v5.3 (Stable Release)")
