import streamlit as st
import pandas as pd
import altair as alt
import pydeck as pdk
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

# Custom CSS
st.markdown("""
<style>
    /* Tabs & Global */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; white-space: pre-wrap; background-color: #f1f3f4; border-radius: 8px 8px 0 0; font-size: 14px; color: #5f6368;
    }
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-top: 3px solid #1a73e8; font-weight: bold; color: #1a73e8; }
    
    /* Metric Cards */
    [data-testid="stMetric"] {
        background-color: #262730; padding: 15px; border-radius: 10px; border: 1px solid #464855; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    [data-testid="stMetricValue"] { font-size: 26px; color: #FFFFFF !important; font-weight: 700; }
    [data-testid="stMetricLabel"] { color: #cfd8dc !important; font-size: 14px; }
    
    /* Insight Box */
    .insight-box {
        background-color: #e8f0fe; border-left: 5px solid #1a73e8; padding: 15px; border-radius: 5px; margin-bottom: 20px; color: #000000 !important;
    }
    
    /* Winner Box */
    .winner-box {
        background-color: #e6fffa; border: 1px solid #b2f5ea; padding: 20px; border-radius: 10px; color: #234e52;
    }
    
    /* Scoring Box */
    .score-box { background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 1px solid #ddd; color: #333; }
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
    
    # Estimasi KK Belum Pinjam
    saturation_ratio = (df['Loan_per_HH'] / 50.0).clip(0, 1)
    df['Est_Unserved_KK'] = (df['Jumlah_KK'] * (1 - saturation_ratio)).astype(int)

    # Risk Scoring Logic
    sat_risk = (df['Loan_per_HH'] / 50.0).clip(0, 1) * 100 
    max_pot = df['Skor_Potensi'].max() if df['Skor_Potensi'].max() > 0 else 1
    eco_risk = 100 - ((df['Skor_Potensi'] / max_pot) * 100)
    
    # Risk Flags
    df['Flag_Kumuh'] = (df['Risk_Kumuh'] > 0).astype(int)
    df['Flag_Bencana'] = (df['Risk_Bencana'] > 0).astype(int)
    df['Flag_Konflik'] = (df['Risk_Konflik'] > 0).astype(int)
    df['Flag_Saturasi'] = (df['Loan_per_HH'] > 50).astype(int)
    df['Risk_Trigger_Count'] = df['Flag_Kumuh'] + df['Flag_Bencana'] + df['Flag_Konflik'] + df['Flag_Saturasi']
    
    env_risk = (df['Flag_Kumuh'] * 30) + (df['Flag_Bencana'] * 20) + (df['Flag_Konflik'] * 50)
    df['Final_Risk_Score'] = (0.3 * sat_risk) + (0.3 * eco_risk) + (0.4 * env_risk)
    
    # Categories
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

    # Simulasi Market Sentiment
    np.random.seed(42) 
    df['Sentiment_Score'] = np.random.uniform(3.5, 4.9, size=len(df))
    df['Review_Count'] = np.random.randint(10, 1000, size=len(df))
    
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

# Color Helpers for Map
def get_hex_risk(score):
    if score < 20: return '#00cc96' # Green
    elif score < 40: return '#ffa15a' # Orange
    elif score < 60: return '#ef553b' # Red Orange
    else: return '#b30000' # Dark Red

def get_hex_potential(score):
    if score > 80: return '#00cc96'
    elif score > 60: return '#636efa'
    elif score > 40: return '#ab63fa'
    else: return '#d3d3d3'

def get_hex_saturation(val):
    # Logic v3.0: Merah = Jenuh/High Risk, Hijau = Low Saturation
    if val > 50: return '#ff0000' # Red (Saturated)
    elif val > 20: return '#ffa500' # Orange
    else: return '#008000' # Green (Opportunity)

df_filtered['color_hex_risk'] = df_filtered['Final_Risk_Score'].apply(get_hex_risk)
df_filtered['color_pot_hex'] = df_filtered['Skor_Potensi'].apply(get_hex_potential)
df_filtered['color_sat_hex'] = df_filtered['Loan_per_HH'].apply(get_hex_saturation)

# -----------------------------------------------------------------------------
# 5. DASHBOARD TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive Summary", 
    "🚀 Growth Intelligence", 
    "⚖️ Saturation & Insight", 
    "🛡️ Risk Guardian",
    "🧮 Scoring & Sector"
])

# ================= TAB 1: EXECUTIVE SUMMARY =================
with tab1:
    st.markdown(f"### 📋 Ringkasan Strategis: {selected_kab}")
    
    # KPI Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Exposure", f"Rp {df_filtered['Total_Pinjaman'].sum()/1e9:,.1f} M")
    c2.metric("Avg Risk Score", f"{df_filtered['Final_Risk_Score'].mean():.1f}/100")
    c3.metric("Growth Spots", f"{len(df_filtered[df_filtered['Strategy_Quadrant']=='Hidden Gem (Grow)'])} Desa", delta_color="normal")
    c4.metric("High Risk Areas", f"{len(df_filtered[df_filtered['Risk_Category'].isin(['High', 'Critical'])])} Desa", delta_color="inverse")

    st.markdown("---")
    
    # Market Sentiment Engine
    st.subheader("⭐ Market Sentiment Insight")
    pct_growth = (len(df_filtered[df_filtered['Strategy_Quadrant']=='Hidden Gem (Grow)']) / len(df_filtered)) * 100
    dom_sector = df_filtered['Sektor_Dominan'].mode()[0] if not df_filtered['Sektor_Dominan'].empty else "Umum"
    
    st.markdown(f"""
    <div class="insight-box">
        <b>💡 Automated Business Insights:</b><br>
        Wilayah <b>{selected_kab}</b> memiliki potensi pertumbuhan <b>{pct_growth:.1f}%</b> di sektor <b>{dom_sector}</b>.
        Sentimen pasar positif, namun perhatikan tingkat saturasi di zona perkotaan.
    </div>
    """, unsafe_allow_html=True)
    
    # Top Sector Analysis Box
    sent_col1, sent_col2 = st.columns([1, 2])
    sec_stats = df_filtered.groupby('Sektor_Dominan').agg({'Sentiment_Score': 'mean', 'Review_Count': 'mean'}).reset_index().sort_values('Sentiment_Score', ascending=False)
    
    if not sec_stats.empty:
        top_sector = sec_stats.iloc[0]
        with sent_col1:
            st.markdown(f"""
            <div class="winner-box">
                <h4>🏆 Top Sector Winner</h4>
                <h2>{top_sector['Sektor_Dominan']}</h2>
                <p>Rating: <b>{top_sector['Sentiment_Score']:.1f} / 5.0</b></p>
            </div>
            """, unsafe_allow_html=True)
        with sent_col2:
            chart_sent_bar = alt.Chart(sec_stats.head(5)).mark_bar().encode(
                x=alt.X('Sentiment_Score', scale=alt.Scale(domain=[3.5, 5.0])),
                y=alt.Y('Sektor_Dominan', sort='-x'),
                color=alt.Color('Sentiment_Score', scale=alt.Scale(scheme='greens'))
            ).properties(height=180)
            st.altair_chart(chart_sent_bar, use_container_width=True)

    st.markdown("---")

    # --- STRATEGIC MATRIX & SATURATION MAP (FROM V3.0) ---
    st.subheader("🎯 Matriks Strategi & Peta Saturasi")
    
    col_strat1, col_strat2 = st.columns(2)
    
    with col_strat1:
        st.markdown("**Matriks Posisi (Potensi vs Saturasi)**")
        
        # Quadrant Chart
        base = alt.Chart(df_filtered).mark_circle(size=80).encode(
            x=alt.X('Skor_Potensi', title='Potensi Ekonomi (Makin Kanan Bagus)'),
            y=alt.Y('Loan_per_HH', title='Saturasi (Pinjaman/KK)'),
            color=alt.Color('Strategy_Quadrant', scale=alt.Scale(scheme='category10'), legend=alt.Legend(orient='bottom')),
            tooltip=['Desa', 'Kecamatan', 'Strategy_Quadrant', 'Loan_per_HH']
        ).properties(height=400)
        
        # Garis Rata-rata
        rule_x = alt.Chart(df_filtered).mark_rule(color='gray', strokeDash=[3,3]).encode(x='mean(Skor_Potensi)')
        rule_y = alt.Chart(df_filtered).mark_rule(color='gray', strokeDash=[3,3]).encode(y='mean(Loan_per_HH)')
        
        st.altair_chart(base + rule_x + rule_y, use_container_width=True)
        
    with col_strat2:
        st.markdown("**Peta Panas Saturasi (Risk Map)**")
        st.caption("Merah = Sangat Jenuh (> Rp 50jt/KK). Hijau = Masih Luas.")
        
        # Simple Map Saturasi
        st.map(df_filtered, latitude='lat', longitude='lon', color='color_sat_hex', size=30, zoom=10)

    # --- TABEL DATA LENGKAP PER DESA (FROM V3.0) ---
    st.markdown("---")
    with st.expander("📋 Lihat Data Lengkap Per Desa"):
        st.dataframe(
            df_filtered[['Desa', 'Kecamatan', 'Skor_Potensi', 'Loan_per_HH', 'Total_Pinjaman', 'Strategy_Quadrant', 'Final_Risk_Score']],
            use_container_width=True,
            column_config={
                "Skor_Potensi": st.column_config.ProgressColumn("Potensi", max_value=100, format="%.1f"),
                "Loan_per_HH": st.column_config.NumberColumn("Saturasi/KK", format="Rp %.1f Jt"),
                "Total_Pinjaman": st.column_config.NumberColumn("Total Pinjaman", format="Rp %.0f")
            }
        )

# ================= TAB 2: GROWTH INTELLIGENCE =================
with tab2:
    st.markdown("### 🚀 Analisis Potensi Pertumbuhan")
    st.info("Visualisasi dan analisis detail area dengan potensi ekonomi tinggi.")

    c_g1, c_g2 = st.columns([2, 1])
    
    with c_g1:
        st.subheader("🗺️ Peta Sebaran Potensi")
        st.map(df_filtered, latitude='lat', longitude='lon', color='color_pot_hex', size=30, zoom=10)
        
    with c_g2:
        st.subheader("📊 Kategori Potensi Desa")
        def categorize_pot(x):
            if x > 70: return 'Tinggi (>70)'
            elif x > 40: return 'Sedang (40-70)'
            else: return 'Rendah (<40)'
        df_filtered['Kategori_Potensi'] = df_filtered['Skor_Potensi'].apply(categorize_pot)
        
        hist_pot = alt.Chart(df_filtered).mark_bar().encode(
            x=alt.X('Kategori_Potensi', sort=['Tinggi (>70)', 'Sedang (40-70)', 'Rendah (<40)']),
            y='count()',
            color=alt.Color('Kategori_Potensi', scale=alt.Scale(range=['#00cc96', '#636efa', '#d3d3d3']))
        ).properties(height=300)
        st.altair_chart(hist_pot, use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Detail Desa: Growth Opportunities")
    
    with st.expander("ℹ️ Definisi & Metodologi Skor Ekonomi"):
        st.markdown("**Skor Ekonomi** (0-100) mengukur daya tarik investasi berdasarkan aktivitas bisnis, infrastruktur, dan daya beli.")

    growth_cols = ['Desa', 'Kecamatan', 'Skor_Potensi', 'Sektor_Dominan', 'Jumlah_KK', 'Est_Unserved_KK']
    growth_df = df_filtered.sort_values(by='Skor_Potensi', ascending=False)[growth_cols].copy()
    
    st.dataframe(
        growth_df, hide_index=True, use_container_width=True,
        column_config={
            "Skor_Potensi": st.column_config.NumberColumn("Skor Ekonomi", format="%.1f"),
            "Est_Unserved_KK": st.column_config.NumberColumn("Est. KK Belum Pinjam")
        }
    )

# ================= TAB 3: SATURATION & INSIGHT =================
with tab3:
    st.markdown("### ⚖️ Analisis Saturasi & Beban Utang")
    
    col_sat1, col_sat2 = st.columns(2)
    with col_sat1:
        st.subheader("📊 Histogram Beban Utang")
        hist = alt.Chart(df_filtered).mark_bar().encode(
            x=alt.X('Loan_per_HH', bin=alt.Bin(maxbins=20), title='Pinjaman per KK (Juta Rp)'),
            y='count()', color=alt.value('#ffa15a')
        ).properties(height=250)
        st.altair_chart(hist, use_container_width=True)

    with col_sat2:
        st.subheader("⚔️ Komposisi Kuadran")
        quad_counts = df_filtered['Strategy_Quadrant'].value_counts().reset_index()
        quad_counts.columns = ['Kategori', 'Jumlah']
        pie = alt.Chart(quad_counts).mark_arc(outerRadius=100).encode(
            theta=alt.Theta("Jumlah", stack=True),
            color=alt.Color("Kategori", scale=alt.Scale(scheme='tableau10'))
        )
        st.altair_chart(pie, use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Kategorisasi Beban Utang Per Desa")
    
    def categorize_debt(val):
        if val < 10: return "🟢 Ringan (< 10 Juta)"
        elif val < 30: return "🟡 Menengah (10-30 Juta)"
        elif val < 50: return "🟠 Berat (30-50 Juta)"
        else: return "🔴 Sangat Berat (> 50 Juta)"
        
    sat_table = df_filtered[['Desa', 'Kecamatan', 'Loan_per_HH', 'Total_Pinjaman']].copy()
    sat_table['Kategori Beban Utang'] = sat_table['Loan_per_HH'].apply(categorize_debt)
    
    st.dataframe(
        sat_table.sort_values('Loan_per_HH', ascending=False),
        hide_index=True, use_container_width=True,
        column_config={
            "Loan_per_HH": st.column_config.NumberColumn("Rata-rata Utang/KK", format="Rp %.1f Juta"),
            "Total_Pinjaman": st.column_config.NumberColumn("Total Outstanding", format="Rp %.0f")
        }
    )

# ================= TAB 4: RISK GUARDIAN =================
with tab4:
    st.markdown("### 🛡️ Profil Risiko Wilayah")
    
    col_r1, col_r2 = st.columns([2, 1])
    with col_r1:
        st.subheader("🗺️ Peta Risiko (Heatmap)")
        st.caption("Merah = Risiko Tinggi. Hijau = Risiko Rendah.")
        st.map(df_filtered, latitude='lat', longitude='lon', color='color_hex_risk', size=30, zoom=10)

    with col_r2:
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

    st.markdown("---")
    st.subheader("📋 Interpretasi Skor Risiko")
    
    with st.expander("ℹ️ Definisi & Metodologi Skor Risiko"):
        st.markdown("**Skor Risiko** (0-100) memprediksi probabilitas default berdasarkan Saturasi (30%), Ekonomi (30%), dan Faktor Lingkungan (40%).")
    
    def interpret_score(score):
        if score > 80: return "⛔ KRITIS: Stop Lending"
        elif score > 60: return "⚠️ TINGGI: Batasi Plafond"
        elif score > 40: return "✋ SEDANG: Perlu Survey"
        else: return "✅ RENDAH: Aman"
        
    risk_table = df_filtered[['Desa', 'Final_Risk_Score', 'Risk_Trigger_Count', 'Flag_Konflik', 'Flag_Bencana']].copy()
    risk_table['Interpretasi Kebijakan'] = risk_table['Final_Risk_Score'].apply(interpret_score)
    
    st.dataframe(
        risk_table.sort_values('Final_Risk_Score', ascending=False).head(100),
        hide_index=True, use_container_width=True,
        column_config={
            "Final_Risk_Score": st.column_config.ProgressColumn("Risk Score", max_value=100, format="%.1f")
        }
    )

# ================= TAB 5: SCORING & SECTOR =================
with tab5:
    st.markdown("### 🧮 Simulasi Kredit & Benchmark Sektor")

    with st.container():
        st.markdown('<div class="score-box">', unsafe_allow_html=True)
        col_sim1, col_sim2 = st.columns(2)
        
        with col_sim1:
            st.markdown("#### 1. Input Data Usaha")
            avail_sectors = df_filtered['Sektor_Dominan'].unique()
            sim_sector = st.selectbox("Sektor Usaha Nasabah", avail_sectors)
            sim_desa = st.selectbox("Lokasi Usaha (Desa)", df_filtered['Desa'].unique())
            sim_omzet = st.number_input("Omzet Usaha Bulanan (Juta Rp)", min_value=1.0, value=15.0, step=0.5)
            sim_lama = st.slider("Lama Usaha Berjalan (Tahun)", 0, 30, 3)
            sim_jaminan = st.selectbox("Jenis Agunan", ["Tanpa Agunan", "BPKB Motor", "BPKB Mobil", "Sertifikat Tanah/Rumah"])
            
            sector_stats = df_filtered[df_filtered['Sektor_Dominan'] == sim_sector]
            avg_sec_risk = sector_stats['Final_Risk_Score'].mean()
            
            st.info(f"**Benchmark Sektor:** Rata-rata Risiko **{avg_sec_risk:.1f}/100**")

        # Calculation Engine
        desa_data = df_filtered[df_filtered['Desa'] == sim_desa].iloc[0]
        loc_score = desa_data['Skor_Potensi']
        risk_loc = desa_data['Final_Risk_Score']
        sentiment_loc = desa_data['Sentiment_Score']
        
        cap_score = min(sim_omzet * 3, 100)
        coll_score = 0
        if sim_jaminan == "Sertifikat Tanah/Rumah": coll_score = 100
        elif sim_jaminan == "BPKB Mobil": coll_score = 70
        elif sim_jaminan == "BPKB Motor": coll_score = 40
        
        final_score = (0.3 * loc_score) + (0.4 * cap_score) + (0.2 * coll_score) + (0.1 * (sentiment_loc/5)*100)
        
        if avg_sec_risk > 60: final_score -= 10
        if risk_loc > 60: final_score -= 15
        
        with col_sim2:
            st.markdown("#### 2. Hasil Keputusan")
            st.write(f"**Geo-Credit Score:** {final_score:.1f} / 100")
            
            if final_score >= 75:
                st.success("✅ **APPROVED**")
                st.metric("Plafond Disetujui", f"Rp {sim_omzet * 5:,.0f} Juta")
            elif final_score >= 50:
                st.warning("⚠️ **REVIEW (Komite)**")
                st.metric("Plafond Maksimal", f"Rp {sim_omzet * 2:,.0f} Juta")
            else:
                st.error("❌ **REJECTED**")
                st.write("Risiko terlalu tinggi.")
            
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("Geo-Credit Intelligence Framework v11.5 | Final Integrated Release")
