import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

# -----------------------------------------------------------------------------
# 1. KONFIGURASI HALAMAN & PERFORMA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Geo-Credit Strategic Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# [CRITICAL] Mematikan limit default Altair
alt.data_transformers.disable_max_rows()

# Custom CSS untuk tampilan metrik yang lebih tajam
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 45px; white-space: pre-wrap; background-color: #f8f9fa; border-radius: 4px 4px 0 0; font-size: 14px;
    }
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-top: 3px solid #0056b3; font-weight: bold; color: #0056b3; }
    div[data-testid="stMetricValue"] { font-size: 20px; color: #212529; }
    .insight-box { background-color: #e3f2fd; border-left: 5px solid #2196f3; padding: 15px; border-radius: 5px; font-size: 14px; }
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

    # --- CLEANING ---
    df.columns = [col.replace('potensi_wilayah_kel_podes_pdrb_sekda_current.', '') for col in df.columns]
    
    # Rename Dictionary
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
    
    # Fill NaN
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)
    
    # --- METRICS ---
    df['Jumlah_KK'] = df['Jumlah_KK'].replace(0, 1) 
    df['Loan_per_HH'] = (df['Total_Pinjaman'] / df['Jumlah_KK']) / 1_000_000 
    
    # Risk Scoring Components
    sat_risk = (df['Loan_per_HH'] / 50.0).clip(0, 1) * 100 
    max_pot = df['Skor_Potensi'].max() if df['Skor_Potensi'].max() > 0 else 1
    eco_risk = 100 - ((df['Skor_Potensi'] / max_pot) * 100)
    
    r_kumuh = (df['Risk_Kumuh'] > 0).astype(int) if 'Risk_Kumuh' in df.columns else 0
    r_bencana = (df['Risk_Bencana'] > 0).astype(int) if 'Risk_Bencana' in df.columns else 0
    r_konflik = (df['Risk_Konflik'] > 0).astype(int) if 'Risk_Konflik' in df.columns else 0
    
    env_risk = (r_kumuh * 30) + (r_bencana * 20) + (r_konflik * 50)
               
    df['Final_Risk_Score'] = (0.3 * sat_risk) + (0.3 * eco_risk) + (0.4 * env_risk)
    
    # Clean Infinite
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    # Categories
    def get_risk_cat(x):
        if x >= 60: return 'Critical'
        elif x >= 40: return 'High'
        elif x >= 20: return 'Medium'
        else: return 'Low'
    df['Risk_Category'] = df['Final_Risk_Score'].apply(get_risk_cat)

    # Strategy Quadrants
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
    st.error("Data 'Prototype Jawa Tengah.csv' tidak ditemukan.")
    st.stop()

# -----------------------------------------------------------------------------
# 3. SIDEBAR & FILTER
# -----------------------------------------------------------------------------
st.sidebar.title("🎛️ Geo-Control")
all_kab = sorted(df['Kabupaten'].unique())
selected_kab = st.sidebar.selectbox("Wilayah (Kabupaten)", all_kab, index=0)

df_kab = df[df['Kabupaten'] == selected_kab]
all_kec = sorted(df_kab['Kecamatan'].unique())
selected_kec = st.sidebar.multiselect("Filter Kecamatan", all_kec, default=all_kec)

if not selected_kec:
    st.warning("Pilih minimal satu kecamatan.")
    st.stop()

df_filtered = df_kab[df_kab['Kecamatan'].isin(selected_kec)]
st.sidebar.markdown("---")
st.sidebar.info(f"📊 Dataset: {len(df_filtered)} Desa")

# -----------------------------------------------------------------------------
# 4. SAFETY FUNCTION (Visualisation Helper)
# -----------------------------------------------------------------------------
def safe_chart_data(dataframe, required_cols, limit=4000):
    valid_cols = [c for c in required_cols if c in dataframe.columns]
    temp_df = dataframe[valid_cols].copy()
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
    
    # 1. Metrics Bar
    col1, col2, col3, col4 = st.columns(4)
    total_loan = df_filtered['Total_Pinjaman'].sum() / 1e9
    avg_risk = df_filtered['Final_Risk_Score'].mean()
    growth_opp = len(df_filtered[df_filtered['Strategy_Quadrant']=='Hidden Gem (Grow)'])
    high_risk = len(df_filtered[df_filtered['Risk_Category'].isin(['High', 'Critical'])])
    
    col1.metric("Total Exposure", f"Rp {total_loan:,.1f} M", help="Total Outstanding Loan")
    col2.metric("Avg Risk Score", f"{avg_risk:.1f} / 100", delta=f"{avg_risk-50:.1f} vs Threshold", delta_color="inverse")
    col3.metric("Growth Opportunity", f"{growth_opp} Desa", "Target Ekspansi", delta_color="normal")
    col4.metric("High Risk Watchlist", f"{high_risk} Desa", "Perlu Mitigasi", delta_color="inverse")

    st.markdown("---")
    
    # 2. Automated Insights Generator
    st.subheader("💡 Automated Business Insights")
    with st.container():
        # Logic sederhana untuk generate insight teks
        pct_growth = (growth_opp / len(df_filtered)) * 100
        dom_sector = df_filtered['Sektor_Dominan'].mode()[0] if not df_filtered['Sektor_Dominan'].empty else "N/A"
        
        insight_html = f"""
        <div class="insight-box">
            <ul>
                <li><strong>Posisi Strategis:</strong> Sebesar <b>{pct_growth:.1f}%</b> wilayah di {selected_kab} masuk kategori 'Hidden Gems' yang siap untuk ekspansi kredit.</li>
                <li><strong>Dominasi Sektor:</strong> Perekonomian lokal didorong oleh sektor <b>{dom_sector}</b>. Disarankan membuat produk kredit spesifik untuk sektor ini.</li>
                <li><strong>Kesehatan Portofolio:</strong> Terdapat <b>{high_risk}</b> desa dengan risiko tinggi yang memerlukan restrukturisasi atau penagihan intensif.</li>
            </ul>
        </div>
        """
        st.markdown(insight_html, unsafe_allow_html=True)

    # 3. Macro Strategy Map
    st.markdown("### 🌌 Peta Strategi Makro")
    cols_univ = ['Skor_Potensi', 'Loan_per_HH', 'Risk_Category', 'Total_Pinjaman', 'Desa', 'Kecamatan', 'Strategy_Quadrant']
    data_univ = safe_chart_data(df_filtered, cols_univ)
    
    chart_univ = alt.Chart(data_univ).mark_circle().encode(
        x=alt.X('Skor_Potensi', title='Potensi Ekonomi (Makin Kanan Makin Bagus)'),
        y=alt.Y('Loan_per_HH', title='Saturasi Kredit (Makin Atas Makin Jenuh)'),
        color=alt.Color('Strategy_Quadrant', scale=alt.Scale(scheme='category10'), legend=alt.Legend(title="Kuadran Strategi")),
        size=alt.Size('Total_Pinjaman', legend=None),
        tooltip=['Desa', 'Kecamatan', 'Strategy_Quadrant', 'Loan_per_HH']
    ).properties(height=450).interactive()
    
    st.altair_chart(chart_univ, use_container_width=True)

# ================= TAB 2: GROWTH INTELLIGENCE =================
with tab2:
    st.markdown("### 🚀 Analisis Potensi Pertumbuhan")
    
    c_g1, c_g2 = st.columns([2, 1])
    
    with c_g1:
        st.subheader("📍 Geospasial Potensi Ekonomi")
        st.caption("Peta menunjukkan desa dengan skor attractiveness index tertinggi.")
        
        cols_map = ['lon', 'lat', 'Skor_Potensi', 'Desa', 'Sektor_Dominan']
        data_map = safe_chart_data(df_filtered, cols_map)
        
        chart_map = alt.Chart(data_map).mark_circle(size=80).encode(
            longitude='lon', latitude='lat',
            color=alt.Color('Skor_Potensi', scale=alt.Scale(scheme='greens'), title='Skor Potensi'),
            tooltip=['Desa', 'Skor_Potensi', 'Sektor_Dominan']
        ).project('mercator').properties(height=400)
        st.altair_chart(chart_map, use_container_width=True)
        
    with c_g2:
        st.subheader("🏭 Efisiensi Sektor")
        st.caption("Rata-rata Skor Potensi per Sektor Usaha")
        
        if 'Sektor_Dominan' in df_filtered.columns:
            # Grouping untuk melihat kualitas sektor, bukan hanya kuantitas
            sec_perf = df_filtered.groupby('Sektor_Dominan')['Skor_Potensi'].mean().reset_index()
            sec_perf = sec_perf.sort_values(by='Skor_Potensi', ascending=False).head(10)
            
            chart_bar = alt.Chart(sec_perf).mark_bar().encode(
                x=alt.X('Skor_Potensi', title='Rata-rata Potensi Wilayah'),
                y=alt.Y('Sektor_Dominan', sort='-x', title=''),
                color=alt.Color('Skor_Potensi', scale=alt.Scale(scheme='greens'), legend=None),
                tooltip=['Sektor_Dominan', 'Skor_Potensi']
            )
            st.altair_chart(chart_bar, use_container_width=True)

    # Correlation Sentiment vs Potential
    st.markdown("---")
    st.subheader("💎 The 'Sweet Spot': Sentimen Pasar vs Potensi Ekonomi")
    st.caption("Area di pojok kanan atas adalah target terbaik: Ekonomi kuat & Review bagus.")
    
    cols_scat = ['Skor_Potensi', 'Sentiment_Score', 'Desa', 'Sektor_Dominan']
    data_scat = safe_chart_data(df_filtered, cols_scat)
    
    scatter = alt.Chart(data_scat).mark_circle(size=60).encode(
        x=alt.X('Skor_Potensi', title='Potensi Ekonomi'),
        y=alt.Y('Sentiment_Score', title='Sentimen Pasar (Rating 1-5)', scale=alt.Scale(domain=[3, 5])),
        color=alt.value('#00cc96'),
        tooltip=['Desa', 'Sektor_Dominan', 'Sentiment_Score', 'Skor_Potensi']
    ).interactive()
    
    st.altair_chart(scatter, use_container_width=True)

# ================= TAB 3: SATURATION DEEP-DIVE =================
with tab3:
    st.markdown("### ⚖️ Analisis Saturasi Kredit")
    
    c_s1, c_s2 = st.columns(2)
    with c_s1:
        st.subheader("📊 Distribusi Beban Utang (Histogram)")
        st.caption("Apakah pinjaman merata atau menumpuk di sedikit desa?")
        
        data_hist = safe_chart_data(df_filtered, ['Loan_per_HH'])
        hist = alt.Chart(data_hist).mark_bar().encode(
            x=alt.X('Loan_per_HH', bin=alt.Bin(maxbins=20), title='Pinjaman per KK (Juta Rp)'),
            y=alt.Y('count()', title='Jumlah Desa'),
            color=alt.value('#ffa15a'),
            tooltip=['count()']
        )
        st.altair_chart(hist, use_container_width=True)
        
    with c_s2:
        st.subheader("⚔️ Komposisi Pasar")
        quad_data = df_filtered['Strategy_Quadrant'].value_counts().reset_index()
        quad_data.columns = ['Kategori', 'Jumlah']
        
        pie = alt.Chart(quad_data).mark_arc(outerRadius=120).encode(
            theta=alt.Theta("Jumlah", stack=True),
            color=alt.Color("Kategori", scale=alt.Scale(scheme='tableau10')),
            tooltip=["Kategori", "Jumlah"]
        )
        st.altair_chart(pie, use_container_width=True)
        
    st.markdown("### 🚨 Red Ocean Alert (Top 10 Saturated Areas)")
    st.caption("Desa-desa ini memiliki beban utang per keluarga tertinggi. Hati-hati risiko *Over-Indebtedness*.")
    
    red_ocean_cols = ['Desa', 'Kecamatan', 'Loan_per_HH', 'Total_Pinjaman', 'Jumlah_KK']
    top_sat = df_filtered.nlargest(10, 'Loan_per_HH')[red_ocean_cols]
    
    st.dataframe(
        top_sat, 
        hide_index=True, 
        use_container_width=True,
        column_config={
            "Loan_per_HH": st.column_config.ProgressColumn("Saturasi (Juta/KK)", format="Rp %.1f", min_value=0, max_value=100),
            "Total_Pinjaman": st.column_config.NumberColumn("Total Outstanding", format="Rp %.0f")
        }
    )

# ================= TAB 4: RISK GUARDIAN =================
with tab4:
    st.markdown("### 🛡️ Profil Risiko & Mitigasi")
    
    r1, r2 = st.columns([2, 1])
    with r1:
        st.subheader("🗺️ Peta Sebaran Risiko (Heatmap)")
        cols_risk = ['lon', 'lat', 'Final_Risk_Score', 'Desa', 'Risk_Category']
        data_risk = safe_chart_data(df_filtered, cols_risk)
        
        r_map = alt.Chart(data_risk).mark_circle(size=70).encode(
            longitude='lon', latitude='lat',
            color=alt.Color('Final_Risk_Score', scale=alt.Scale(scheme='reds', domain=[0, 100]), title='Risk Score'),
            tooltip=['Desa', 'Final_Risk_Score', 'Risk_Category']
        ).project('mercator').properties(height=400)
        st.altair_chart(r_map, use_container_width=True)

    with r2:
        st.subheader("🔍 Pemicu Risiko Utama")
        # Menghitung agregat faktor risiko
        rf_data = pd.DataFrame({
            'Faktor': ['Konflik Sosial', 'Bencana Alam', 'Lingkungan Kumuh', 'Saturasi Tinggi'],
            'Jumlah Desa': [
                (df_filtered.get('Risk_Konflik', 0) > 0).sum(),
                (df_filtered.get('Risk_Bencana', 0) > 0).sum(),
                (df_filtered.get('Risk_Kumuh', 0) > 0).sum(),
                (df_filtered['Loan_per_HH'] > 50).sum()
            ]
        })
        
        rf_chart = alt.Chart(rf_data).mark_bar().encode(
            x=alt.X('Jumlah Desa'),
            y=alt.Y('Faktor', sort='-x'),
            color=alt.value('#ef553b')
        )
        st.altair_chart(rf_chart, use_container_width=True)

    st.markdown("### 📋 Watchlist: Desa Risiko Kritis")
    high_risk = df_filtered[df_filtered['Risk_Category'].isin(['High', 'Critical'])]
    
    if not high_risk.empty:
        hr_sorted = high_risk.nlargest(50, 'Final_Risk_Score')
        # Dynamic columns check
        disp_cols = ['Desa', 'Kecamatan', 'Final_Risk_Score', 'Risk_Category']
        if 'Risk_Konflik' in df.columns: disp_cols.append('Risk_Konflik')
        
        st.dataframe(
            hr_sorted[disp_cols], 
            hide_index=True, 
            use_container_width=True,
            column_config={
                "Final_Risk_Score": st.column_config.NumberColumn("Score (0-100)", format="%.1f"),
                "Risk_Category": st.column_config.TextColumn("Status"),
                "Risk_Konflik": st.column_config.CheckboxColumn("Konflik?")
            }
        )
    else:
        st.success("✅ Tidak ada desa dengan status High/Critical Risk di area filter ini.")

# Footer
st.markdown("---")
st.caption("Geo-Credit Intelligence Framework v6.0 | Developed by Tio Brain")
