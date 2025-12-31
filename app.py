import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

# -----------------------------------------------------------------------------
# 1. KONFIGURASI HALAMAN & UX
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Geo-Credit Strategic Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# [CRITICAL] Mematikan limit default Altair
alt.data_transformers.disable_max_rows()

# Custom CSS: Dark Cards, White Numbers, Insight Box Hitam
st.markdown("""
<style>
    /* Styling Global */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; white-space: pre-wrap; background-color: #f1f3f4; border-radius: 8px 8px 0 0; font-size: 14px; color: #5f6368;
    }
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-top: 3px solid #1a73e8; font-weight: bold; color: #1a73e8; }
    
    /* Kartu Metrik Dark Mode */
    [data-testid="stMetric"] {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #464855;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    [data-testid="stMetricValue"] { 
        font-size: 26px; color: #FFFFFF !important; font-weight: 700;
    }
    [data-testid="stMetricLabel"] {
        color: #cfd8dc !important; font-size: 14px;
    }
    [data-testid="stMetricDelta"] svg { fill: #ffffff !important; }
    
    /* Insight Box (Teks Hitam) */
    .insight-box {
        background-color: #e8f0fe; 
        border-left: 5px solid #1a73e8; 
        padding: 15px; 
        border-radius: 5px; 
        margin-bottom: 20px;
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA ENGINE
# -----------------------------------------------------------------------------
@st.cache_data
def load_data_engine():
    try:
        df = pd.read_csv('Prototype Jawa Tengah.csv')
    except Exception as e:
        return None

    # --- CLEANING & RENAMING ---
    df.columns = [col.replace('potensi_wilayah_kel_podes_pdrb_sekda_current.', '') for col in df.columns]
    
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
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
    
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)
    
    # --- METRICS CALCULATION ---
    df['Jumlah_KK'] = df['Jumlah_KK'].replace(0, 1) 
    df['Loan_per_HH'] = (df['Total_Pinjaman'] / df['Jumlah_KK']) / 1_000_000 
    
    # Risk Scoring Logic
    sat_risk = (df['Loan_per_HH'] / 50.0).clip(0, 1) * 100 
    max_pot = df['Skor_Potensi'].max() if df['Skor_Potensi'].max() > 0 else 1
    eco_risk = 100 - ((df['Skor_Potensi'] / max_pot) * 100)
    
    r_kumuh = (df['Risk_Kumuh'] > 0).astype(int) if 'Risk_Kumuh' in df.columns else 0
    r_bencana = (df['Risk_Bencana'] > 0).astype(int) if 'Risk_Bencana' in df.columns else 0
    r_konflik = (df['Risk_Konflik'] > 0).astype(int) if 'Risk_Konflik' in df.columns else 0
    
    env_risk = (r_kumuh * 30) + (r_bencana * 20) + (r_konflik * 50)
               
    df['Final_Risk_Score'] = (0.3 * sat_risk) + (0.3 * eco_risk) + (0.4 * env_risk)
    
    # Kategori Risiko & Kuadran
    def get_risk_cat(x):
        if x >= 60: return 'Critical'
        elif x >= 40: return 'High'
        elif x >= 20: return 'Medium'
        else: return 'Low'
    df['Risk_Category'] = df['Final_Risk_Score'].apply(get_risk_cat)

    avg_pot = df['Skor_Potensi'].mean()
    avg_sat = df['Loan_per_HH'].mean()
    def get_quad(row):
        if row['Skor_Potensi'] >= avg_pot and row['Loan_per_HH'] < avg_sat: return "Hidden Gem (Grow)"
        elif row['Skor_Potensi'] >= avg_pot and row['Loan_per_HH'] >= avg_sat: return "Red Ocean (Compete)"
        elif row['Skor_Potensi'] < avg_pot and row['Loan_per_HH'] >= avg_sat: return "High Risk (Stop)"
        else: return "Dormant (Monitor)"
    df['Strategy_Quadrant'] = df.apply(get_quad, axis=1)

    # Simulasi Data Tambahan
    np.random.seed(42)
    df['Sentiment_Score'] = np.random.uniform(3.5, 4.9, size=len(df))
    df['Review_Count'] = np.random.randint(10, 500, size=len(df))
    
    return df

# LOAD DATA
df = load_data_engine()
if df is None:
    st.error("❌ Data tidak ditemukan.")
    st.stop()

# -----------------------------------------------------------------------------
# 3. SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.title("🎛️ Geo-Control Panel")
all_kab = sorted(df['Kabupaten'].unique())
selected_kab = st.sidebar.selectbox("Pilih Wilayah (Kabupaten)", all_kab, index=0)

df_kab = df[df['Kabupaten'] == selected_kab]
all_kec = sorted(df_kab['Kecamatan'].unique())
selected_kec = st.sidebar.multiselect("Filter Kecamatan", all_kec, default=all_kec)

if not selected_kec:
    st.warning("⚠️ Mohon pilih minimal satu kecamatan.")
    st.stop()

df_filtered = df_kab[df_kab['Kecamatan'].isin(selected_kec)]
st.sidebar.markdown("---")
st.sidebar.info(f"📍 **Coverage:** {len(df_filtered)} Desa")

# -----------------------------------------------------------------------------
# 4. COLOR HELPER (HEX CODES FOR st.map)
# -----------------------------------------------------------------------------
# Mengadaptasi logika warna dari app (1).py tapi menggunakan skor numerik v7

def get_hex_risk(score):
    # Hijau -> Kuning -> Merah (Risk semakin tinggi semakin merah)
    if score < 20: return '#00cc96' # Green (Low Risk)
    elif score < 40: return '#ffa15a' # Orange
    elif score < 60: return '#ef553b' # Red Orange
    else: return '#b30000' # Dark Red (Critical)

def get_hex_potential(score):
    # Abu -> Biru -> Hijau (Potensi semakin tinggi semakin hijau)
    if score > 80: return '#00cc96' # Green (High Potensi)
    elif score > 60: return '#636efa' # Blue
    elif score > 40: return '#ab63fa' # Purple
    else: return '#d3d3d3' # Grey

# Terapkan warna ke dataframe
df_filtered['color_hex_risk'] = df_filtered['Final_Risk_Score'].apply(get_hex_risk)
df_filtered['color_hex_pot'] = df_filtered['Skor_Potensi'].apply(get_hex_potential)

# -----------------------------------------------------------------------------
# 5. DASHBOARD TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Executive Summary", 
    "🚀 Growth Intelligence", 
    "⚖️ Saturation & Competition", 
    "🛡️ Risk Guardian"
])

# ================= TAB 1: EXECUTIVE SUMMARY =================
with tab1:
    st.markdown(f"### 📋 Ringkasan Strategis: {selected_kab}")
    
    col1, col2, col3, col4 = st.columns(4)
    total_loan = df_filtered['Total_Pinjaman'].sum() / 1e9
    avg_risk = df_filtered['Final_Risk_Score'].mean()
    growth_opp = len(df_filtered[df_filtered['Strategy_Quadrant']=='Hidden Gem (Grow)'])
    high_risk_count = len(df_filtered[df_filtered['Risk_Category'].isin(['High', 'Critical'])])
    
    col1.metric("Total Exposure", f"Rp {total_loan:,.1f} M")
    col2.metric("Avg Risk Score", f"{avg_risk:.1f} / 100", delta=f"{avg_risk-50:.1f} vs Threshold", delta_color="inverse")
    col3.metric("Growth Spots", f"{growth_opp} Desa", "Target Ekspansi", delta_color="normal")
    col4.metric("High Risk Areas", f"{high_risk_count} Desa", "Perlu Mitigasi", delta_color="inverse")

    st.markdown("---")
    
    # Insight Box (Hitam)
    pct_growth = (growth_opp / len(df_filtered)) * 100
    dom_sector = df_filtered['Sektor_Dominan'].mode()[0] if not df_filtered['Sektor_Dominan'].empty else "Umum"
    st.markdown(f"""
    <div class="insight-box">
        <b>💡 Automated Business Insights:</b><br>
        Wilayah <b>{selected_kab}</b> memiliki potensi pertumbuhan <b>{pct_growth:.1f}%</b> di sektor <b>{dom_sector}</b>.
        Waspadai <b>{high_risk_count} desa</b> dengan risiko tinggi (Critical Risk). Disarankan fokus akuisisi pada zona hijau di peta Growth Intelligence.
    </div>
    """, unsafe_allow_html=True)

    # Strategic Universe
    cols_univ = ['Skor_Potensi', 'Loan_per_HH', 'Risk_Category', 'Total_Pinjaman', 'Desa', 'Strategy_Quadrant']
    chart_univ = alt.Chart(df_filtered[cols_univ]).mark_circle().encode(
        x=alt.X('Skor_Potensi', title='Potensi Ekonomi'),
        y=alt.Y('Loan_per_HH', title='Tingkat Saturasi'),
        color=alt.Color('Strategy_Quadrant', scale=alt.Scale(scheme='category10')),
        size=alt.Size('Total_Pinjaman', legend=None),
        tooltip=['Desa', 'Strategy_Quadrant', 'Loan_per_HH']
    ).properties(height=350).interactive()
    st.altair_chart(chart_univ, use_container_width=True)

# ================= TAB 2: GROWTH INTELLIGENCE =================
with tab2:
    st.markdown("### 🚀 Analisis Potensi Pertumbuhan")
    
    c_g1, c_g2 = st.columns([2, 1])
    
    with c_g1:
        st.subheader("🗺️ Peta Sebaran Potensi")
        st.caption("Hijau = Potensi Tinggi | Biru = Menengah | Abu = Rendah")
        
        # --- SIMPLE MAP (st.map) ---
        # Sesuai request: tampilan seperti app (1).py
        st.map(
            df_filtered,
            latitude='lat',
            longitude='lon',
            color='color_hex_pot',
            size=30,
            zoom=10
        )
        # ---------------------------
        
    with c_g2:
        st.subheader("🏭 Matriks Kualitas Sektor")
        st.caption("Mencari Sektor 'Niche' (Kualitas Tinggi, Volume Rendah).")
        
        if 'Sektor_Dominan' in df_filtered.columns:
            sec_agg = df_filtered.groupby('Sektor_Dominan').agg({
                'Desa': 'count',
                'Skor_Potensi': 'mean',
                'Final_Risk_Score': 'mean'
            }).reset_index()
            sec_agg.columns = ['Sektor', 'Jumlah_Desa', 'Avg_Potensi', 'Avg_Risiko']
            
            chart_matrix = alt.Chart(sec_agg).mark_circle().encode(
                x=alt.X('Jumlah_Desa', title='Market Size (Jml Desa)'),
                y=alt.Y('Avg_Potensi', title='Market Quality (Avg Potensi)', scale=alt.Scale(domain=[0, 100])),
                size=alt.Size('Jumlah_Desa', legend=None),
                color=alt.Color('Avg_Risiko', scale=alt.Scale(scheme='redyellowgreen', reverse=True), title='Avg Risk'),
                tooltip=['Sektor', 'Jumlah_Desa', 'Avg_Potensi', 'Avg_Risiko']
            ).properties(height=400).interactive()
            
            st.altair_chart(chart_matrix, use_container_width=True)

    st.info("💡 **Insight:** Peta di atas menggunakan indikator warna sederhana untuk memudahkan identifikasi zona hijau (High Potential) secara cepat.")

# ================= TAB 3: SATURATION DEEP-DIVE =================
with tab3:
    st.markdown("### ⚖️ Analisis Saturasi")
    
    c_s1, c_s2 = st.columns(2)
    with c_s1:
        st.subheader("📊 Distribusi Beban Utang")
        hist = alt.Chart(df_filtered).mark_bar().encode(
            x=alt.X('Loan_per_HH', bin=alt.Bin(maxbins=20), title='Pinjaman per KK (Juta Rp)'),
            y='count()',
            color=alt.value('#ffa15a'),
            tooltip=['count()']
        ).properties(height=300)
        st.altair_chart(hist, use_container_width=True)
        
    with c_s2:
        st.subheader("⚔️ Market Share")
        quad_counts = df_filtered['Strategy_Quadrant'].value_counts().reset_index()
        quad_counts.columns = ['Kategori', 'Jumlah']
        pie = alt.Chart(quad_counts).mark_arc(outerRadius=100).encode(
            theta=alt.Theta("Jumlah", stack=True),
            color=alt.Color("Kategori", scale=alt.Scale(scheme='tableau10')),
            tooltip=["Kategori", "Jumlah"]
        )
        st.altair_chart(pie, use_container_width=True)
        
    st.markdown("### 🚨 Red Ocean (Top 10 Saturated)")
    top_sat = df_filtered.nlargest(10, 'Loan_per_HH')[['Desa', 'Kecamatan', 'Loan_per_HH', 'Total_Pinjaman']]
    st.dataframe(top_sat, hide_index=True, use_container_width=True, 
                 column_config={"Loan_per_HH": st.column_config.ProgressColumn("Saturasi", format="Rp %.1f", max_value=100)})

# ================= TAB 4: RISK GUARDIAN =================
with tab4:
    st.markdown("### 🛡️ Profil Risiko (EWS)")
    
    r1, r2 = st.columns([2, 1])
    with r1:
        st.subheader("🗺️ Peta Zona Merah (Risk Map)")
        st.caption("Merah = Risiko Tinggi. Hijau = Risiko Rendah.")
        
        # --- SIMPLE MAP (st.map) FOR RISK ---
        st.map(
            df_filtered,
            latitude='lat',
            longitude='lon',
            color='color_hex_risk',
            size=30,
            zoom=10
        )
        # ------------------------------------

    with r2:
        st.subheader("🔍 Pemicu Risiko")
        rf_data = pd.DataFrame({
            'Faktor': ['Konflik Sosial', 'Bencana Alam', 'Lingkungan Kumuh', 'Saturasi Tinggi'],
            'Jumlah': [
                (df_filtered.get('Risk_Konflik', 0) > 0).sum(),
                (df_filtered.get('Risk_Bencana', 0) > 0).sum(),
                (df_filtered.get('Risk_Kumuh', 0) > 0).sum(),
                (df_filtered['Loan_per_HH'] > 50).sum()
            ]
        })
        bar_risk = alt.Chart(rf_data).mark_bar().encode(
            x='Jumlah', y=alt.Y('Faktor', sort='-x'), color=alt.value('#d32f2f')
        )
        st.altair_chart(bar_risk, use_container_width=True)

    st.markdown("### 📋 Watchlist Critical")
    high_risk = df_filtered[df_filtered['Risk_Category'].isin(['High', 'Critical'])].nlargest(50, 'Final_Risk_Score')
    st.dataframe(high_risk[['Desa', 'Final_Risk_Score', 'Risk_Category', 'Risk_Konflik']], hide_index=True, use_container_width=True)

# Footer
st.markdown("---")
st.caption("Geo-Credit Intelligence Framework v8.0 | Simple Map Edition")
