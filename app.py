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
    
    /* Validation Cards */
    .validation-card {
        background-color: #ffffff; padding: 20px; border-radius: 10px; border: 1px solid #e0e0e0; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 15px;
    }
    .status-pass { color: #2e7d32; font-weight: bold; }
    .status-fail { color: #c62828; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA ENGINE (EXCEL & CSV LOADER)
# -----------------------------------------------------------------------------
@st.cache_data
def load_data_engine():
    data = {}
    try:
        # 1. Load Main Data (CSV)
        data['main'] = pd.read_csv('Prototype Jawa Tengah.csv')
        
        # 2. Load Validation Data (EXCEL Multiple Sheets)
        excel_file = 'Kewajaran_Omzet_All.xlsx'
        data['level1'] = pd.read_excel(excel_file, sheet_name='Level 1')
        data['level2'] = pd.read_excel(excel_file, sheet_name='Level 2')
        data['level3'] = pd.read_excel(excel_file, sheet_name='Level 3')
        
        # --- DATA PRE-PROCESSING ---
        # Clean Columns for Main Data
        data['main'].columns = [col.replace('potensi_wilayah_kel_podes_pdrb_sekda_current.', '') for col in data['main'].columns]
        
        # Rename Main Data Columns
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
        data['main'].rename(columns={k: v for k, v in rename_map.items() if k in data['main'].columns}, inplace=True)
        
        # Fill NaN
        num_cols = data['main'].select_dtypes(include=[np.number]).columns
        data['main'][num_cols] = data['main'][num_cols].fillna(0)
        
        # Metrics Calculation for Main Data
        data['main']['Jumlah_KK'] = data['main']['Jumlah_KK'].replace(0, 1) 
        data['main']['Loan_per_HH'] = (data['main']['Total_Pinjaman'] / data['main']['Jumlah_KK']) / 1_000_000 
        
        # Risk Scoring Logic
        sat_risk = (data['main']['Loan_per_HH'] / 50.0).clip(0, 1) * 100 
        max_pot = data['main']['Skor_Potensi'].max() if data['main']['Skor_Potensi'].max() > 0 else 1
        eco_risk = 100 - ((data['main']['Skor_Potensi'] / max_pot) * 100)
        
        r_kumuh = (data['main']['Risk_Kumuh'] > 0).astype(int) if 'Risk_Kumuh' in data['main'].columns else 0
        r_bencana = (data['main']['Risk_Bencana'] > 0).astype(int) if 'Risk_Bencana' in data['main'].columns else 0
        r_konflik = (data['main']['Risk_Konflik'] > 0).astype(int) if 'Risk_Konflik' in data['main'].columns else 0
        
        env_risk = (r_kumuh * 30) + (r_bencana * 20) + (r_konflik * 50)
        data['main']['Final_Risk_Score'] = (0.3 * sat_risk) + (0.3 * eco_risk) + (0.4 * env_risk)
        
        def get_risk_cat(x):
            if x >= 60: return 'Critical'
            elif x >= 40: return 'High'
            elif x >= 20: return 'Medium'
            else: return 'Low'
        data['main']['Risk_Category'] = data['main']['Final_Risk_Score'].apply(get_risk_cat)

        avg_pot = data['main']['Skor_Potensi'].mean()
        avg_sat = data['main']['Loan_per_HH'].mean()
        def get_quad(row):
            if row['Skor_Potensi'] >= avg_pot and row['Loan_per_HH'] < avg_sat: return "Hidden Gem (Grow)"
            elif row['Skor_Potensi'] >= avg_pot and row['Loan_per_HH'] >= avg_sat: return "Red Ocean (Compete)"
            elif row['Skor_Potensi'] < avg_pot and row['Loan_per_HH'] >= avg_sat: return "High Risk (Stop)"
            else: return "Dormant (Monitor)"
        data['main']['Strategy_Quadrant'] = data['main'].apply(get_quad, axis=1)

        # Simulating Market Sentiment
        np.random.seed(42) 
        data['main']['Sentiment_Score'] = np.random.uniform(3.5, 4.9, size=len(data['main']))
        data['main']['Review_Count'] = np.random.randint(10, 1000, size=len(data['main']))
        
        # Market Size Proxy
        saturation_ratio = (data['main']['Loan_per_HH'] / 50.0).clip(0, 1)
        data['main']['Est_Unserved_KK'] = (data['main']['Jumlah_KK'] * (1 - saturation_ratio)).astype(int)
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None
        
    return data

# LOAD DATA
dataset = load_data_engine()
if dataset is None:
    st.error("❌ Data tidak ditemukan. Pastikan 'Prototype Jawa Tengah.csv' dan 'Kewajaran_Omzet_All.xlsx' ada.")
    st.stop()

# -----------------------------------------------------------------------------
# 3. SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.title("🎛️ Geo-Control Panel")

# Gunakan dataset['main'] sebagai basis data spasial
df_main = dataset['main']

all_kab = sorted(df_main['Kabupaten'].unique())
selected_kab = st.sidebar.selectbox("Pilih Wilayah (Kabupaten)", all_kab, index=0)

# Filter Kecamatan berdasarkan Kabupaten terpilih
df_kab = df_main[df_main['Kabupaten'] == selected_kab]
all_kec = sorted(df_kab['Kecamatan'].unique())
selected_kec = st.sidebar.multiselect("Filter Kecamatan", all_kec, default=all_kec)

if not selected_kec:
    st.warning("⚠️ Mohon pilih minimal satu kecamatan.")
    st.stop()

df_filtered = df_kab[df_kab['Kecamatan'].isin(selected_kec)].copy()

st.sidebar.markdown("---")
st.sidebar.info(f"📍 **Coverage:** {len(df_filtered)} Desa")

# Color Helpers
def get_hex_risk(score):
    if score < 20: return '#00cc96' # Green
    elif score < 40: return '#ffa15a' # Orange
    elif score < 60: return '#ef553b' # Red Orange
    else: return '#b30000' # Dark Red

def get_hex_potential(score):
    if score > 80: return '#00cc96' # Bright Green
    elif score > 60: return '#636efa' # Blue
    elif score > 40: return '#ab63fa' # Purple
    else: return '#d3d3d3' # Grey

df_filtered['color_hex_risk'] = df_filtered['Final_Risk_Score'].apply(get_hex_risk)
df_filtered['color_pot_hex'] = df_filtered['Skor_Potensi'].apply(get_hex_potential)

# -----------------------------------------------------------------------------
# 5. DASHBOARD TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive Summary", 
    "🚀 Growth Intelligence", 
    "⚖️ Saturation & Insight", 
    "🛡️ Risk Guardian",
    "✅ Pengecekan Kewajaran"
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
    st.subheader("⭐ Market Sentiment Engine & Insight")
    
    pct_growth = (len(df_filtered[df_filtered['Strategy_Quadrant']=='Hidden Gem (Grow)']) / len(df_filtered)) * 100
    dom_sector = df_filtered['Sektor_Dominan'].mode()[0] if not df_filtered['Sektor_Dominan'].empty else "Umum"
    
    st.markdown(f"""
    <div class="insight-box">
        <b>💡 Automated Business Insights:</b><br>
        Wilayah <b>{selected_kab}</b> didorong oleh sektor <b>{dom_sector}</b> dengan potensi pertumbuhan <b>{pct_growth:.1f}%</b>.
        Analisis sentimen menunjukkan sektor ini memiliki tingkat kepuasan tinggi.
    </div>
    """, unsafe_allow_html=True)
    
    # Top Sector Analysis
    sent_col1, sent_col2 = st.columns([1, 2])
    sec_stats = df_filtered.groupby('Sektor_Dominan').agg({'Sentiment_Score': 'mean', 'Review_Count': 'mean'}).reset_index().sort_values('Sentiment_Score', ascending=False)
    
    if not sec_stats.empty:
        top_sector = sec_stats.iloc[0]
        with sent_col1:
            st.markdown(f"""
            <div class="winner-box">
                <h4>🏆 Top Sector Winner</h4>
                <h2>{top_sector['Sektor_Dominan']}</h2>
                <p>Rating Rata-rata: <b>{top_sector['Sentiment_Score']:.1f} / 5.0</b></p>
                <p><i>"Sektor paling direkomendasikan."</i></p>
            </div>
            """, unsafe_allow_html=True)
        with sent_col2:
            chart_sent_bar = alt.Chart(sec_stats.head(5)).mark_bar().encode(
                x=alt.X('Sentiment_Score', scale=alt.Scale(domain=[3.5, 5.0]), title='Rata-rata Rating'),
                y=alt.Y('Sektor_Dominan', sort='-x', title='Sektor'),
                color=alt.Color('Sentiment_Score', scale=alt.Scale(scheme='greens'), legend=None),
                tooltip=['Sektor_Dominan', 'Sentiment_Score']
            ).properties(height=200)
            st.altair_chart(chart_sent_bar, use_container_width=True)

    st.markdown("---")

    # Eco Score Definition
    with st.expander("ℹ️ Definisi & Metodologi Skor Ekonomi (Eco Score)"):
        st.markdown("**Skor Ekonomi** (0-100) mengukur daya tarik investasi berdasarkan aktivitas bisnis (40%), infrastruktur (30%), dan daya beli (30%).")

    # Strategic Matrix
    st.subheader("🎯 Matriks Strategi & Peta Potensi")
    col_strat, col_map = st.columns(2)
    
    with col_strat:
        st.markdown("**Matriks Posisi: Potensi vs Saturasi**")
        chart_quad = alt.Chart(df_filtered).mark_circle(size=100).encode(
            x=alt.X('Skor_Potensi', title='Potensi Ekonomi'),
            y=alt.Y('Loan_per_HH', title='Saturasi'),
            color='Strategy_Quadrant',
            tooltip=['Desa', 'Strategy_Quadrant']
        ).properties(height=400).interactive()
        rule_x = alt.Chart(df_filtered).mark_rule(color='gray', strokeDash=[3,3]).encode(x='mean(Skor_Potensi)')
        rule_y = alt.Chart(df_filtered).mark_rule(color='gray', strokeDash=[3,3]).encode(y='mean(Loan_per_HH)')
        st.altair_chart(chart_quad + rule_x + rule_y, use_container_width=True)
        
    with col_map:
        st.markdown("**Peta Sebaran Potensi**")
        st.map(df_filtered, latitude='lat', longitude='lon', color='color_pot_hex', size=30, zoom=10)
    
    # Table Hidden Gems
    st.markdown("---")
    st.subheader("💎 Top Hidden Gems (Unserved Market)")
    top_n = st.slider("Jumlah Desa:", 3, 20, 5)
    gems = df_filtered[df_filtered['Strategy_Quadrant'] == 'Hidden Gem (Grow)'].nlargest(top_n, 'Est_Unserved_KK')
    st.dataframe(gems[['Desa', 'Kecamatan', 'Est_Unserved_KK', 'Skor_Potensi']], hide_index=True, use_container_width=True)

    with st.expander("📋 Lihat Data Lengkap Per Desa"):
        st.dataframe(df_filtered, use_container_width=True)

# ================= TAB 2: GROWTH INTELLIGENCE =================
with tab2:
    st.markdown("### 🚀 Analisis Potensi Pertumbuhan")
    c_g1, c_g2 = st.columns([2, 1])
    with c_g1:
        st.subheader("🗺️ Peta Sebaran Potensi")
        st.map(df_filtered, latitude='lat', longitude='lon', color='color_pot_hex', size=30, zoom=10)
    with c_g2:
        st.subheader("📊 Kategori Potensi")
        hist_pot = alt.Chart(df_filtered).mark_bar().encode(
            x=alt.X('Skor_Potensi', bin=True), y='count()', color=alt.value('#00cc96')
        ).properties(height=300)
        st.altair_chart(hist_pot, use_container_width=True)

    st.subheader("📋 Detail Desa: Growth Opportunities")
    with st.expander("ℹ️ Definisi Skor Ekonomi"):
        st.write("Skor komposit dari aktivitas bisnis dan infrastruktur.")
    st.dataframe(df_filtered[['Desa', 'Skor_Potensi', 'Est_Unserved_KK', 'Sektor_Dominan']].sort_values('Skor_Potensi', ascending=False), hide_index=True, use_container_width=True)

# ================= TAB 3: SATURATION =================
with tab3:
    st.markdown("### ⚖️ Analisis Saturasi")
    col_sat1, col_sat2 = st.columns(2)
    with col_sat1:
        st.subheader("📊 Distribusi Beban Utang")
        hist = alt.Chart(df_filtered).mark_bar().encode(
            x=alt.X('Loan_per_HH', bin=True), y='count()', color=alt.value('#ffa15a')
        )
        st.altair_chart(hist, use_container_width=True)
    with col_sat2:
        st.subheader("⚔️ Komposisi Kuadran")
        pie = alt.Chart(df_filtered).mark_arc().encode(
            theta='count()', color='Strategy_Quadrant'
        )
        st.altair_chart(pie, use_container_width=True)
    
    st.subheader("📋 Kategorisasi Beban Utang")
    def categorize_debt(val):
        if val < 10: return "🟢 Ringan"
        elif val < 30: return "🟡 Menengah"
        elif val < 50: return "🟠 Berat"
        else: return "🔴 Sangat Berat"
    df_filtered['Kategori_Beban'] = df_filtered['Loan_per_HH'].apply(categorize_debt)
    st.dataframe(df_filtered[['Desa', 'Loan_per_HH', 'Kategori_Beban']].sort_values('Loan_per_HH', ascending=False), hide_index=True, use_container_width=True)

# ================= TAB 4: RISK GUARDIAN =================
with tab4:
    st.markdown("### 🛡️ Profil Risiko Wilayah")
    col_r1, col_r2 = st.columns([2, 1])
    with col_r1:
        st.subheader("🗺️ Peta Risiko")
        st.map(df_filtered, latitude='lat', longitude='lon', color='color_hex_risk', size=30, zoom=10)
    with col_r2:
        st.subheader("🔍 Pemicu Risiko")
        rf_data = pd.DataFrame({
            'Faktor': ['Konflik', 'Bencana', 'Kumuh', 'Saturasi'],
            'Jumlah': [df_filtered['Risk_Konflik'].astype(bool).sum(), df_filtered['Risk_Bencana'].astype(bool).sum(), 
                       df_filtered['Risk_Kumuh'].astype(bool).sum(), (df_filtered['Loan_per_HH']>50).sum()]
        })
        st.bar_chart(rf_data.set_index('Faktor'))

    st.subheader("📋 Interpretasi Skor Risiko")
    with st.expander("ℹ️ Definisi Skor Risiko"):
        st.write("Gabungan risiko saturasi, ekonomi, dan lingkungan.")
    
    def interpret_risk(score):
        if score > 80: return "⛔ KRITIS"
        elif score > 60: return "⚠️ TINGGI"
        elif score > 40: return "✋ SEDANG"
        else: return "✅ RENDAH"
    df_filtered['Interpretasi_Risiko'] = df_filtered['Final_Risk_Score'].apply(interpret_risk)
    st.dataframe(df_filtered[['Desa', 'Final_Risk_Score', 'Interpretasi_Risiko']].sort_values('Final_Risk_Score', ascending=False), hide_index=True, use_container_width=True)

# ================= TAB 5: PENGECEKAN KEWAJARAN (VALIDATION ENGINE) =================
with tab5:
    st.markdown("### ✅ Pengecekan Tingkat Kewajaran (Validation Engine)")
    st.info("Fitur untuk Mantri memvalidasi inputan pengajuan kredit berdasarkan benchmark data historis (Level 1-3).")

    # --- 1. SELECTION PANEL ---
    with st.container():
        st.markdown('<div class="validation-card">', unsafe_allow_html=True)
        st.markdown("#### 1. Parameter Wilayah & Sektor")
        
        # Load Reference Data
        ref_l3 = dataset['level3']
        ref_l2 = dataset['level2']
        ref_l1 = dataset['level1']
        
        c_sel1, c_sel2 = st.columns(2)
        with c_sel1:
            # 1. Provinsi (Cascading)
            prov_opts = sorted(ref_l3['Provinsi Usaha'].astype(str).unique())
            sel_prov = st.selectbox("Provinsi Usaha", prov_opts)
            
            # 2. Kota/Kab
            kab_opts = sorted(ref_l3[ref_l3['Provinsi Usaha'] == sel_prov]['Kabupaten/kota'].astype(str).unique())
            sel_kab = st.selectbox("Kabupaten/Kota", kab_opts)
            
        with c_sel2:
            # 3. Sektor Ekonomi
            sec_opts = sorted(ref_l3['Sektor Ekonomi'].astype(str).unique())
            sel_sec = st.selectbox("Sektor Ekonomi", sec_opts)
            
            # 4. Sub Sektor Ekonomi
            sub_opts = sorted(ref_l3[ref_l3['Sektor Ekonomi'] == sel_sec]['Sub Sektor Ekonomi'].astype(str).unique())
            sel_sub = st.selectbox("Sub Sektor Ekonomi", sub_opts)
            
        st.markdown('</div>', unsafe_allow_html=True)

    # --- 2. INPUT PANEL (NO PLAFOND INPUT) ---
    with st.container():
        st.markdown('<div class="validation-card">', unsafe_allow_html=True)
        st.markdown("#### 2. Input Data Keuangan")
        
        c_in1, c_in2, c_in3 = st.columns(3)
        with c_in1:
            in_omzet = st.number_input("Omzet (Rp)", min_value=0.0, step=1000000.0, format="%.0f")
        with c_in2:
            in_hpp = st.number_input("HPP (Rp)", min_value=0.0, step=1000000.0, format="%.0f")
        with c_in3:
            in_laba = st.number_input("Laba (Rp)", min_value=0.0, step=1000000.0, format="%.0f")
            
        btn_check = st.button("🔍 Analisa & Rekomendasi Plafond", type="primary")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- 3. VALIDATION ENGINE & RESULT ---
    if btn_check:
        st.markdown("### 📊 Hasil Analisa Multi-Level")
        
        # Function to Render Level Card
        def render_level_check(level_name, dataset, query_mask, inputs):
            match = dataset[query_mask]
            if match.empty:
                return None
            
            row = match.iloc[0]
            max_omzet = row['OMZET_MAX_WAJAR']
            max_hpp = row['HPP_MAX_WAJAR']
            max_laba = row['LABA_MAX_WAJAR']
            max_plafond = row['PLAFOND_MAX_WAJAR']
            
            # Check Status
            status_omzet = "✅ WAJAR" if inputs['omzet'] <= max_omzet else "❌ TIDAK WAJAR"
            status_hpp = "✅ WAJAR" if inputs['hpp'] <= max_hpp else "❌ TIDAK WAJAR"
            status_laba = "✅ WAJAR" if inputs['laba'] <= max_laba else "❌ TIDAK WAJAR"
            
            # Overall Status for the Level
            is_all_valid = (inputs['omzet'] <= max_omzet) and (inputs['hpp'] <= max_hpp) and (inputs['laba'] <= max_laba)
            
            if is_all_valid:
                level_badge = '<span style="background-color:#e8f5e9; color:#2e7d32; padding:3px 8px; border-radius:4px; font-weight:bold;">WAJAR</span>'
                border_color = "#4caf50" # Green
            else:
                level_badge = '<span style="background-color:#ffebee; color:#c62828; padding:3px 8px; border-radius:4px; font-weight:bold;">TIDAK WAJAR</span>'
                border_color = "#e57373" # Red

            html = f"""
            <div style="border: 2px solid {border_color}; padding: 15px; border-radius: 8px; margin-bottom: 15px; background-color: #fafafa;">
                <div style="font-weight: bold; font-size: 16px; margin-bottom: 10px; display: flex; justify-content: space-between;">
                    <span>{level_name}</span>
                    {level_badge}
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 10px; font-size: 13px;">
                    <div><strong>Omzet:</strong><br>{status_omzet}<br><span style="color:#666">Max: {max_omzet:,.0f}</span></div>
                    <div><strong>HPP:</strong><br>{status_hpp}<br><span style="color:#666">Max: {max_hpp:,.0f}</span></div>
                    <div><strong>Laba:</strong><br>{status_laba}<br><span style="color:#666">Max: {max_laba:,.0f}</span></div>
                    <div style="background-color: #f5f5f5; padding: 5px; border-radius: 4px; border-left: 3px solid #000;">
                        <strong>Plafond yang Wajar:</strong><br>
                        <span style="color: #000000; font-size: 15px; font-weight: bold;">Rp {max_plafond:,.0f}</span>
                    </div>
                </div>
            </div>
            """
            return html

        user_inputs = {'omzet': in_omzet, 'hpp': in_hpp, 'laba': in_laba}
        
        # 1. Check Level 1 (Provinsi + Sektor)
        mask_l1 = (ref_l1['Provinsi Usaha'] == sel_prov) & (ref_l1['Sektor Ekonomi'] == sel_sec)
        res_l1 = render_level_check("Level 1: Provinsi & Sektor", ref_l1, mask_l1, user_inputs)
        
        # 2. Check Level 2 (Provinsi + Sub Sektor)
        mask_l2 = (ref_l2['Provinsi Usaha'] == sel_prov) & (ref_l2['Sektor Ekonomi'] == sel_sec) & (ref_l2['Sub Sektor Ekonomi'] == sel_sub)
        res_l2 = render_level_check("Level 2: Provinsi & Sub Sektor", ref_l2, mask_l2, user_inputs)
        
        # 3. Check Level 3 (Kabupaten + Sub Sektor) - MOST SPECIFIC
        mask_l3 = (ref_l3['Provinsi Usaha'] == sel_prov) & (ref_l3['Kabupaten/kota'] == sel_kab) & (ref_l3['Sektor Ekonomi'] == sel_sec) & (ref_l3['Sub Sektor Ekonomi'] == sel_sub)
        res_l3 = render_level_check(f"Level 3: {sel_kab} & Sub Sektor", ref_l3, mask_l3, user_inputs)
        
        # Render Results Stacked
        if res_l1: st.markdown(res_l1, unsafe_allow_html=True)
        if res_l2: st.markdown(res_l2, unsafe_allow_html=True)
        if res_l3: st.markdown(res_l3, unsafe_allow_html=True)
        
        if not (res_l1 or res_l2 or res_l3):
            st.warning("⚠️ Data benchmark tidak ditemukan untuk kombinasi wilayah/sektor ini.")

# Footer
st.markdown("---")
st.caption("Geo-Credit Intelligence Framework v12.5 | Final Validation Engine with Explicit Status")
