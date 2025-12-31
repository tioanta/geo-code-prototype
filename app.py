import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

# -----------------------------------------------------------------------------
# 1. KONFIGURASI HALAMAN & PERFORMA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Geo-Credit Strategic Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# [CRITICAL FIX] Mematikan batasan baris default Altair
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
        df = pd.read_csv('Prototype Jawa Tengah.csv')
    except FileNotFoundError:
        return None

    # --- CLEANING ---
    df.columns = [col.replace('potensi_wilayah_kel_podes_pdrb_sekda_current.', '') for col in df.columns]
    
    # Rename Column Penting Saja
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
    df.rename(columns=rename_map, inplace=True)
    
    # Fill NaN 0
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)
    
    # --- METRIC CALCULATION ---
    # 1. Saturation
    df['Jumlah_KK'] = df['Jumlah_KK'].replace(0, 1) 
    df['Loan_per_HH'] = (df['Total_Pinjaman'] / df['Jumlah_KK']) / 1_000_000 
    
    # 2. Risk Scoring
    sat_risk = (df['Loan_per_HH'] / 50.0).clip(0, 1) * 100 
    max_pot = df['Skor_Potensi'].max() if df['Skor_Potensi'].max() > 0 else 1
    eco_risk = 100 - ((df['Skor_Potensi'] / max_pot) * 100)
    
    env_risk = (df['Risk_Kumuh'] > 0).astype(int) * 30 + \
               (df['Risk_Bencana'] > 0).astype(int) * 20 + \
               (df['Risk_Konflik'] > 0).astype(int) * 50
               
    df['Final_Risk_Score'] = (0.3 * sat_risk) + (0.3 * eco_risk) + (0.4 * env_risk)
    
    # Safety: Hapus infinite value
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    # 3. Categorization
    def get_risk_cat(x):
        if x >= 60: return 'Critical'
        elif x >= 40: return 'High'
        elif x >= 20: return 'Medium'
        else: return 'Low'
    df['Risk_Category'] = df['Final_Risk_Score'].apply(get_risk_cat)

    # 4. Strategy Quadrant
    avg_pot = df['Skor_Potensi'].mean()
    avg_sat = df['Loan_per_HH'].mean()
    def get_quad(row):
        if row['Skor_Potensi'] >= avg_pot and row['Loan_per_HH'] < avg_sat: return "Hidden Gem (Grow)"
        elif row['Skor_Potensi'] >= avg_pot and row['Loan_per_HH'] >= avg_sat: return "Red Ocean (Compete)"
        elif row['Skor_Potensi'] < avg_pot and row['Loan_per_HH'] >= avg_sat: return "High Risk (Stop)"
        else: return "Dormant (Monitor)"
    df['Strategy_Quadrant'] = df.apply(get_quad, axis=1)

    # 5. Simulasi Sentiment
    np.random.seed(42)
    df['Sentiment_Score'] = np.random.uniform(3.5, 4.9, size=len(df))
    df['Review_Count'] = np.random.randint(10, 500, size=len(df))
    
    return df

df = load_data_engine()
if df is None:
    st.error("Data tidak ditemukan.")
    st.stop()

# -----------------------------------------------------------------------------
# 3. SIDEBAR
# -----------------------------------------------------------------------------
st.sidebar.title("🎛️ Geo-Control")
# Default pilih satu kabupaten agar load awal ringan
default_kab = df['Kabupaten'].unique()[0]
selected_kab = st.sidebar.selectbox("Wilayah (Kabupaten)", df['Kabupaten'].unique(), index=0)

df_kab = df[df['Kabupaten'] == selected_kab]

selected_kec = st.sidebar.multiselect("Filter Kecamatan", df_kab['Kecamatan'].unique(), default=df_kab['Kecamatan'].unique())
df_filtered = df_kab[df_kab['Kecamatan'].isin(selected_kec)]

st.sidebar.markdown("---")
st.sidebar.caption(f"Total Desa: {len(df_filtered)}")

# -----------------------------------------------------------------------------
# 4. DASHBOARD TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Executive Summary", 
    "🚀 Growth Intelligence", 
    "⚖️ Saturation Deep-Dive", 
    "🛡️ Risk Guardian"
])

# --- HELPER UNTUK MENGHINDARI ERROR MEMORI ---
def get_chart_data(dataset, cols):
    # Hanya ambil kolom yang dibutuhkan dan batasi max 5000 baris untuk visualisasi
    data = dataset[cols].copy()
    if len(data) > 5000:
        return data.sample(5000)
    return data

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
    
    # [OPTIMASI] Siapkan data minimalis untuk chart ini
    chart_cols = ['Skor_Potensi', 'Loan_per_HH', 'Risk_Category', 'Total_Pinjaman', 'Desa', 'Kecamatan', 'Strategy_Quadrant']
    univ_data = get_chart_data(df_filtered, chart_cols)
    
    chart_univ = alt.Chart(univ_data).mark_circle().encode(
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
        st.subheader("🗺️ Peta Potensi Ekonomi")
        # [OPTIMASI] Data spasial minimalis
        map_cols = ['lon', 'lat', 'Skor_Potensi', 'Desa', 'Sektor_Dominan']
        map_data = get_chart_data(df_filtered, map_cols)
        
        chart_map = alt.Chart(map_data).mark_circle(size=60).encode(
            longitude='lon', latitude='lat',
            color=alt.Color('Skor_Potensi', scale=alt.Scale(scheme='greens')),
            tooltip=['Desa', 'Skor_Potensi', 'Sektor_Dominan']
        ).project('mercator').properties(height=400)
        st.altair_chart(chart_map, use_container_width=True)
        
    with col_g2:
        st.subheader("🏭 Sektor Dominan")
        sec_data = df_filtered['Sektor_Dominan'].value_counts().reset_index()
        sec_data.columns = ['Sektor', 'Jumlah']
        st.dataframe(sec_data, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("⭐ Market Sentiment (Top 10)")
    
    # [FIX] Reset index dan pastikan data bersih
    top_sent = df_filtered.nlargest(10, 'Sentiment_Score')[['Desa', 'Sentiment_Score', 'Review_Count', 'Sektor_Dominan']].reset_index(drop=True)
    
    # [FIX] Pastikan tidak ada NaN di kolom yang digunakan
    top_sent = top_sent.dropna(subset=['Desa', 'Sentiment_Score'])
    
    if len(top_sent) > 0:
        chart_sent = alt.Chart(top_sent).mark_bar().encode(
            x=alt.X('Sentiment_Score:Q', scale=alt.Scale(domain=[0, 5]), title='Sentiment Score'),
            y=alt.Y('Desa:N', sort='-x', title='Desa'),
            color=alt.Color('Review_Count:Q', scale=alt.Scale(scheme='blues'), title='Review Count'),
            tooltip=['Desa:N', 'Sektor_Dominan:N', 'Sentiment_Score:Q', 'Review_Count:Q']
        ).properties(height=300)
        st.altair_chart(chart_sent, use_container_width=True)
    else:
        st.info("Tidak ada data sentiment tersedia untuk ditampilkan")

# ================= TAB 3: SATURATION DEEP-DIVE =================
with tab3:
    st.markdown("### ⚖️ Analisis Saturasi")
    
    c_s1, c_s2 = st.columns(2)
    with c_s1:
        st.subheader("Matriks Persaingan")
        quad_data = df_filtered['Strategy_Quadrant'].value_counts().reset_index()
        quad_data.columns = ['Kategori', 'Jumlah']
        
        pie = alt.Chart(quad_data).mark_arc(outerRadius=120).encode(
            theta=alt.Theta("Jumlah:Q", stack=True),
            color=alt.Color("Kategori:N"),
            tooltip=["Kategori:N", "Jumlah:Q"]
        ).properties(height=300)
        st.altair_chart(pie, use_container_width=True)
        
    with c_s2:
        st.subheader("Top 10 Area Jenuh (Red Ocean)")
        red_ocean = df_filtered[df_filtered['Strategy_Quadrant'].str.contains('Red Ocean')].nlargest(10, 'Loan_per_HH')
        if len(red_ocean) > 0:
            st.dataframe(red_ocean[['Desa', 'Kecamatan', 'Loan_per_HH']], hide_index=True, use_container_width=True)
        else:
            st.info("Tidak ada area Red Ocean di wilayah ini")

# ================= TAB 4: RISK GUARDIAN =================
with tab4:
    st.markdown("### 🛡️ Profil Risiko")
    
    col_r1, col_r2 = st.columns([2, 1])
    with col_r1:
        st.subheader("🗺️ Peta Zona Merah")
        # [OPTIMASI] Data spasial risiko
        risk_map_cols = ['lon', 'lat', 'Final_Risk_Score', 'Desa', 'Risk_Category']
        risk_map_data = get_chart_data(df_filtered, risk_map_cols)
        
        r_map = alt.Chart(risk_map_data).mark_circle(size=60).encode(
            longitude='lon', latitude='lat',
            color=alt.Color('Final_Risk_Score', scale=alt.Scale(scheme='reds')),
            tooltip=['Desa', 'Final_Risk_Score', 'Risk_Category']
        ).project('mercator').properties(height=400)
        st.altair_chart(r_map, use_container_width=True)

    with col_r2:
        st.subheader("🚨 Watchlist (High Risk)")
        high_risk = df_filtered[df_filtered['Risk_Category'].isin(['High', 'Critical'])].nlargest(100, 'Final_Risk_Score')
        if len(high_risk) > 0:
            st.dataframe(
                high_risk[['Desa', 'Final_Risk_Score', 'Risk_Konflik', 'Risk_Bencana']], 
                hide_index=True, use_container_width=True
            )
        else:
            st.success("Tidak ada area berisiko tinggi di wilayah ini")

# Footer
st.markdown("---")
st.caption("Geo-Credit Intelligence v5.3 (Schema Validation Fixed)")
