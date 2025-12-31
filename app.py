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
    
    # Estimasi KK Belum Pinjam (Market Size Proxy)
    # Asumsi: Jika Loan/HH rendah, maka banyak KK belum terlayani
    # Logic: Unserved = Total KK * (1 - SaturationRatio)
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

# Color Helpers for PyDeck/Map
def get_hex_risk(score):
    if score < 20: return [0, 204, 150, 200] # Green
    elif score < 40: return [255, 161, 90, 200] # Orange
    elif score < 60: return [239, 85, 59, 200] # Red Orange
    else: return [179, 0, 0, 200] # Dark Red

def get_hex_potential(score):
    if score > 80: return '#00cc96'
    elif score > 60: return '#636efa'
    elif score > 40: return '#ab63fa'
    else: return '#d3d3d3'

df_filtered['color_risk_list'] = df_filtered['Final_Risk_Score'].apply(get_hex_risk)
df_filtered['color_pot_hex'] = df_filtered['Skor_Potensi'].apply(get_hex_potential)

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
    st.subheader("Dashboard Intelegensi Kredit Mikro & Sentimen Pasar")
    
    # KPI Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Exposure", f"Rp {df_filtered['Total_Pinjaman'].sum()/1e9:,.1f} M")
    c2.metric("Avg Risk Score", f"{df_filtered['Final_Risk_Score'].mean():.1f}/100")
    c3.metric("Growth Spots", f"{len(df_filtered[df_filtered['Strategy_Quadrant']=='Hidden Gem (Grow)'])} Desa", delta_color="normal")
    c4.metric("High Risk Areas", f"{len(df_filtered[df_filtered['Risk_Category'].isin(['High', 'Critical'])])} Desa", delta_color="inverse")

    st.markdown("---")
    
    # 1. PETA POTENSI + MARKET SENTIMENT (Gaya v2.0)
    col_map, col_list = st.columns([2, 1])
    
    with col_map:
        st.subheader("🗺️ Peta Potensi Ekonomi & Sentimen")
        st.caption("Warna menunjukkan Potensi Ekonomi.")
        st.map(df_filtered, latitude='lat', longitude='lon', color='color_pot_hex', size=30, zoom=10)
    
    with col_list:
        st.subheader("🏆 Top Desa Potensial (Hidden Gems)")
        # Slider interaktif
        top_n = st.slider("Jumlah Desa:", 3, 20, 5)
        
        # Logic: Filter Hidden Gems
        gems = df_filtered[df_filtered['Strategy_Quadrant'] == 'Hidden Gem (Grow)']
        if not gems.empty:
            # Sort by Est_Unserved_KK (Permintaan User: Potensi jumlah KK yg belum pinjam)
            top_gems = gems.nlargest(top_n, 'Est_Unserved_KK')[['Desa', 'Kecamatan', 'Est_Unserved_KK', 'Skor_Potensi']]
            
            st.dataframe(
                top_gems, 
                hide_index=True, 
                use_container_width=True,
                column_config={
                    "Est_Unserved_KK": st.column_config.ProgressColumn("Potensi KK (Belum Pinjam)", format="%d KK", min_value=0, max_value=int(df_filtered['Jumlah_KK'].max())),
                    "Skor_Potensi": st.column_config.NumberColumn("Eco Score")
                }
            )
        else:
            st.warning("Belum ada area Hidden Gems di wilayah ini.")

# ================= TAB 2: GROWTH INTELLIGENCE =================
with tab2:
    st.markdown("### 🚀 Analisis Mendalam: Potensi Pertumbuhan Per Desa")
    st.info("Fokus pada karakteristik ekonomi dan infrastruktur desa untuk ekspansi.")

    # Table Detail Growth
    growth_cols = ['Desa', 'Kecamatan', 'Skor_Potensi', 'Sektor_Dominan', 'Jumlah_KK', 'Est_Unserved_KK']
    growth_df = df_filtered.sort_values(by='Skor_Potensi', ascending=False)[growth_cols].copy()
    
    # Menambahkan Kolom Analisis Tekstual
    def analyze_growth(row):
        return f"Potensi tinggi di sektor {row['Sektor_Dominan']} dengan basis {row['Jumlah_KK']} KK."
    
    growth_df['Analisis Singkat'] = growth_df.apply(analyze_growth, axis=1)
    
    st.dataframe(
        growth_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Skor_Potensi": st.column_config.NumberColumn("Skor Ekonomi (0-100)", format="%.1f"),
            "Jumlah_KK": st.column_config.NumberColumn("Total Keluarga"),
            "Est_Unserved_KK": st.column_config.NumberColumn("Est. KK Belum Pinjam", help="Estimasi jumlah KK yang belum terjangkau kredit"),
            "Analisis Singkat": st.column_config.TextColumn("Highlight Analis", width="large")
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
            y='count()', color=alt.value('#ffa15a'), tooltip=['count()']
        ).properties(height=250)
        st.altair_chart(hist, use_container_width=True)
        st.caption("Grafik yang condong ke kanan menandakan area yang mulai jenuh (over-leveraged).")

    with col_sat2:
        st.subheader("⚔️ Komposisi Kuadran")
        quad_counts = df_filtered['Strategy_Quadrant'].value_counts().reset_index()
        quad_counts.columns = ['Kategori', 'Jumlah']
        pie = alt.Chart(quad_counts).mark_arc(outerRadius=100).encode(
            theta=alt.Theta("Jumlah", stack=True),
            color=alt.Color("Kategori", scale=alt.Scale(scheme='tableau10')),
            tooltip=["Kategori", "Jumlah"]
        )
        st.altair_chart(pie, use_container_width=True)
        st.caption("Proporsi wilayah berdasarkan strategi pertumbuhan vs pertahanan.")

    st.markdown("---")
    st.subheader("📋 Kategorisasi Beban Utang Per Desa")
    
    # Logic: Categorize Debt Load (Textual)
    def categorize_debt(val):
        if val < 10: return "🟢 Ringan (< 10 Juta)"
        elif val < 30: return "🟡 Menengah (10-30 Juta)"
        elif val < 50: return "🟠 Berat (30-50 Juta)"
        else: return "🔴 Sangat Berat (> 50 Juta)"
        
    sat_table = df_filtered[['Desa', 'Kecamatan', 'Loan_per_HH', 'Total_Pinjaman']].copy()
    sat_table['Kategori Beban Utang'] = sat_table['Loan_per_HH'].apply(categorize_debt)
    
    st.dataframe(
        sat_table.sort_values('Loan_per_HH', ascending=False),
        hide_index=True,
        use_container_width=True,
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
        st.subheader("🗺️ Peta Risiko Interaktif")
        st.caption("Arahkan mouse ke titik untuk melihat detail risiko.")
        
        # PYDECK MAP FOR RISK (Hover Support)
        view_state = pdk.ViewState(
            latitude=df_filtered['lat'].mean(),
            longitude=df_filtered['lon'].mean(),
            zoom=10, pitch=0
        )
        
        layer_risk = pdk.Layer(
            "ScatterplotLayer",
            data=df_filtered,
            get_position='[lon, lat]',
            get_fill_color='color_risk_list',
            get_radius=200, pickable=True, opacity=0.8, filled=True,
            radius_min_pixels=5, radius_max_pixels=20,
        )
        
        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            initial_view_state=view_state,
            layers=[layer_risk],
            tooltip={"html": "<b>Desa:</b> {Desa}<br><b>Risk Score:</b> {Final_Risk_Score:.1f}<br><b>Jml Trigger:</b> {Risk_Trigger_Count}"}
        ))

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
    
    # Logic: Interpretasi Score Textual
    def interpret_score(score):
        if score > 80: return "⛔ KRITIS: Stop Lending, Fokus Penagihan"
        elif score > 60: return "⚠️ TINGGI: Batasi Plafond, Wajib Agunan Fisik"
        elif score > 40: return "✋ SEDANG: Perlu Survey Berlapis"
        else: return "✅ RENDAH: Aman untuk Growth"
        
    risk_table = df_filtered[['Desa', 'Final_Risk_Score', 'Risk_Trigger_Count', 'Flag_Konflik', 'Flag_Bencana']].copy()
    risk_table['Interpretasi Kebijakan'] = risk_table['Final_Risk_Score'].apply(interpret_score)
    
    st.dataframe(
        risk_table.sort_values('Final_Risk_Score', ascending=False).head(100),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Final_Risk_Score": st.column_config.ProgressColumn("Risk Score", max_value=100, format="%.1f"),
            "Risk_Trigger_Count": st.column_config.NumberColumn("Jml Pemicu"),
            "Flag_Konflik": st.column_config.CheckboxColumn("Konflik?"),
            "Flag_Bencana": st.column_config.CheckboxColumn("Bencana?")
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
            
            # Pilihan Sektor
            avail_sectors = df_filtered['Sektor_Dominan'].unique()
            sim_sector = st.selectbox("Sektor Usaha Nasabah", avail_sectors)
            
            sim_desa = st.selectbox("Lokasi Usaha (Desa)", df_filtered['Desa'].unique())
            sim_omzet = st.number_input("Omzet Usaha Bulanan (Juta Rp)", min_value=1.0, value=15.0, step=0.5)
            sim_lama = st.slider("Lama Usaha Berjalan (Tahun)", 0, 30, 3)
            sim_jaminan = st.selectbox("Jenis Agunan", ["Tanpa Agunan", "BPKB Motor", "BPKB Mobil", "Sertifikat Tanah/Rumah"])
            
            # BENCHMARK SEKTOR
            sector_stats = df_filtered[df_filtered['Sektor_Dominan'] == sim_sector]
            avg_sec_risk = sector_stats['Final_Risk_Score'].mean()
            avg_sec_pot = sector_stats['Skor_Potensi'].mean()
            
            st.info(f"""
            **Benchmark Sektor ({sim_sector}) di {selected_kab}:**
            - Rata-rata Risiko: **{avg_sec_risk:.1f}/100**
            - Rata-rata Potensi: **{avg_sec_pot:.1f}/100**
            """)

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
        
        # Penalties based on Sector Benchmark
        note_sector = "✅ Sektor Aman"
        if avg_sec_risk > 60:
            final_score -= 10
            note_sector = "⚠️ Sektor Berisiko Tinggi di Wilayah Ini"
            
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

            st.caption(f"Catatan: {note_sector}")
            
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("Geo-Credit Intelligence Framework v11.0 | Final Comprehensive Edition")
