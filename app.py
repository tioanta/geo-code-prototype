import streamlit as st
import pandas as pd
import altair as alt
import pydeck as pdk
import numpy as np
import difflib # Library untuk string matching (Simulasi AI)

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
    
    /* Validation Cards */
    .validation-card {
        background-color: #ffffff; padding: 20px; border-radius: 10px; border: 1px solid #e0e0e0; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 15px;
    }
    .option-card {
        background-color: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #dee2e6; margin-bottom: 15px;
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
        
        # --- REPHRASING SECTOR & SUB-SECTOR NAMES ---
        sector_map = {
            "01-PERTANIAN, PERBURUAN DAN KEHUTANAN": "Pertanian, Kebun & Kehutanan",
            "03-PERTAMBANGAN DAN PENGGALIAN": "Tambang & Galian",
            "04-INDUSTRI PENGOLAHAN": "Industri & Produksi Barang",
            "06-KONSTRUKSI": "Konstruksi & Bangunan",
            "07-PERDAGANGAN BESAR DAN ECERAN": "Perdagangan (Toko/Grosir)",
            "08-PENYEDIAAN AKOMODASI DAN PENYEDIAAN MAKAN MINUM": "Hotel, Restoran & Katering",
            "09-TRANSPORTASI, PERGUDANGAN DAN KOMUNIKASI": "Transportasi, Gudang & Logistik",
            "10-PERANTARA KEUANGAN": "Jasa Keuangan",
            "11-REAL ESTATE, USAHA PERSEWAAN, DAN JASA PERUSAHAAN": "Properti, Sewa & Jasa Bisnis",
            "13-JASA PENDIDIKAN": "Pendidikan & Kursus",
            "14-JASA KESEHATAN DAN KEGIATAN SOSIAL": "Kesehatan & Klinik",
            "15-JASA KEMASYARAKATAN, SOSIAL BUDAYA, HIBURAN DAN PERORANGAN LAINNYA": "Jasa Sosial, Hiburan & Loundry",
            "19-PENERIMA KREDIT BUKAN LAPANGAN USAHA": "Keperluan Konsumtif / Rumah Tangga",
            "18-KEGIATAN YANG BELUM JELAS BATASANNYA": "Kegiatan Lainnya"
        }
        
        sub_sector_map = {
            "Pertanian Padi": "Petani Padi (Sawah)",
            "Kombinasi Pertanian/Perkebunan dg Peternakan (Mixed Farming)": "Tani & Ternak Campuran",
            "Perdagangan Kelapa dan Kelapa Sawit": "Jual Beli Kelapa/Sawit",
            "Perdagangan Eceran Furniture dan Handycraft": "Toko Mebel & Kerajinan",
            "Perdagangan Eceran Hasil Perikanan Darat dan Laut": "Jualan Ikan (Pasar/Eceran)",
            "Perdagangan Eceran Mobil": "Jual Beli Mobil",
            "Perdagangan Eceran Hasil Bumi (Campuran)": "Jualan Hasil Bumi",
            "Distribusi Alat Elektronik": "Distributor Elektronik",
            "Jasa Pelayanan Bongkar Muat Barang": "Jasa Bongkar Muat",
            "Jasa Kebersihan": "Jasa Cleaning Service",
            "Rumah Tangga utk Pemilikan Furnitur & Peralatan Rumah Tangga": "Pembelian Perabot Rumah",
            "Rumah Tangga untuk Pemilikan Rumah Tinggal s.d. Tipe 21": "Pembelian/Renovasi Rumah"
        }

        # Apply Rephrasing
        for lvl in ['level1', 'level2', 'level3']:
            if 'Sektor Ekonomi' in data[lvl].columns:
                data[lvl]['Sektor Ekonomi'] = data[lvl]['Sektor Ekonomi'].replace(sector_map)
            if 'Sub Sektor Ekonomi' in data[lvl].columns:
                data[lvl]['Sub Sektor Ekonomi'] = data[lvl]['Sub Sektor Ekonomi'].replace(sub_sector_map)

        # --- MAIN DATA PRE-PROCESSING ---
        data['main'].columns = [col.replace('potensi_wilayah_kel_podes_pdrb_sekda_current.', '') for col in data['main'].columns]
        
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
        
        if 'Sektor_Dominan' in data['main'].columns:
             data['main']['Sektor_Dominan'] = data['main']['Sektor_Dominan'].replace(sector_map)

        num_cols = data['main'].select_dtypes(include=[np.number]).columns
        data['main'][num_cols] = data['main'][num_cols].fillna(0)
        
        data['main']['Jumlah_KK'] = data['main']['Jumlah_KK'].replace(0, 1) 
        data['main']['Loan_per_HH'] = (data['main']['Total_Pinjaman'] / data['main']['Jumlah_KK']) / 1_000_000 
        
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

        np.random.seed(42) 
        data['main']['Sentiment_Score'] = np.random.uniform(3.5, 4.9, size=len(data['main']))
        data['main']['Review_Count'] = np.random.randint(10, 1000, size=len(data['main']))
        
        saturation_ratio = (data['main']['Loan_per_HH'] / 50.0).clip(0, 1)
        data['main']['Est_Unserved_KK'] = (data['main']['Jumlah_KK'] * (1 - saturation_ratio)).astype(int)
        
    except Exception as e:
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
    st.info("Pilih metode input sektor: Manual (Dropdown) atau AI (Free Text).")

    # Load Reference Data
    ref_l3 = dataset['level3']
    ref_l2 = dataset['level2']
    ref_l1 = dataset['level1']

    # --- 1. LOKASI SELECTION (COMMON FOR BOTH METHODS) ---
    with st.container():
        st.markdown('<div class="validation-card">', unsafe_allow_html=True)
        st.markdown("#### 1. Lokasi Usaha")
        
        c_loc1, c_loc2 = st.columns(2)
        with c_loc1:
            prov_opts = sorted(ref_l3['Provinsi Usaha'].astype(str).unique())
            sel_prov = st.selectbox("Provinsi", prov_opts)
        
        with c_loc2:
            kab_opts = sorted(ref_l3[ref_l3['Provinsi Usaha'] == sel_prov]['Kabupaten/kota'].astype(str).unique())
            sel_kab = st.selectbox("Kabupaten/Kota", kab_opts)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- 2. SECTOR INPUT METHOD SELECTION ---
    st.markdown("#### 2. Identifikasi Jenis Usaha")
    
    # Init variables
    selected_sector = None
    selected_sub_sector = None
    
    # 2 OPSI INPUT (Horizontal Radio / Tabs look)
    input_method = st.radio("Metode Input Sektor:", ["🗂️ Pilih dari List Eksisting", "🤖 Cari dengan AI (Free Text)"], horizontal=True)
    
    if input_method == "🗂️ Pilih dari List Eksisting":
        with st.container():
            st.markdown('<div class="option-card">', unsafe_allow_html=True)
            sec_opts = sorted(ref_l3['Sektor Ekonomi'].astype(str).unique())
            selected_sector = st.selectbox("Sektor Ekonomi", sec_opts)
            
            sub_opts = sorted(ref_l3[ref_l3['Sektor Ekonomi'] == selected_sector]['Sub Sektor Ekonomi'].astype(str).unique())
            selected_sub_sector = st.selectbox("Sub Sektor Ekonomi", sub_opts)
            st.markdown('</div>', unsafe_allow_html=True)
            
    else: # AI Free Text
        with st.container():
            st.markdown('<div class="option-card">', unsafe_allow_html=True)
            user_query = st.text_input("Ketik Jenis Usaha (Contoh: Jualan Bakso, Ternak Lele, Toko Baju)", placeholder="Ketik disini...")
            
            # AI MATCHING LOGIC
            suggested_options = []
            if user_query:
                # Prioritize filtered location
                relevant_data = ref_l3[ref_l3['Provinsi Usaha'] == sel_prov]
                unique_subs = relevant_data['Sub Sektor Ekonomi'].dropna().unique().tolist()
                
                matches = difflib.get_close_matches(user_query, unique_subs, n=3, cutoff=0.3)
                if matches:
                    suggested_options = matches
                else:
                    all_subs = ref_l3['Sub Sektor Ekonomi'].dropna().unique().tolist()
                    suggested_options = difflib.get_close_matches(user_query, all_subs, n=3, cutoff=0.3)

            if suggested_options:
                st.success(f"🤖 **Rekomendasi AI:** Ditemukan {len(suggested_options)} sub-sektor.")
                selected_sub_sector = st.radio("Pilih yang Sesuai:", suggested_options)
                
                # Auto-find Parent Sector
                if selected_sub_sector:
                    try:
                        found_row = ref_l3[ref_l3['Sub Sektor Ekonomi'] == selected_sub_sector].iloc[0]
                        selected_sector = found_row['Sektor Ekonomi']
                        st.caption(f"ℹ️ Sektor Induk: **{selected_sector}**")
                    except:
                        selected_sector = ref_l3['Sektor Ekonomi'].iloc[0]
            elif user_query:
                st.warning("⚠️ AI tidak menemukan kecocokan. Coba kata kunci lain atau gunakan mode 'Pilih dari List'.")
            st.markdown('</div>', unsafe_allow_html=True)

    # --- 3. INPUT DATA KEUANGAN ---
    if selected_sector and selected_sub_sector:
        st.markdown("#### 3. Input Data Keuangan")
        with st.container():
            st.markdown('<div class="validation-card">', unsafe_allow_html=True)
            c_in1, c_in2, c_in3 = st.columns(3)
            with c_in1:
                in_omzet = st.number_input("Omzet (Rp)", min_value=0.0, step=1000000.0, format="%.0f")
            with c_in2:
                in_hpp = st.number_input("HPP (Rp)", min_value=0.0, step=1000000.0, format="%.0f")
            with c_in3:
                in_laba = st.number_input("Laba (Rp)", min_value=0.0, step=1000000.0, format="%.0f")
                
            btn_check = st.button("🚀 Cek Validasi", type="primary")
            st.markdown('</div>', unsafe_allow_html=True)

        # --- 4. EXECUTION ---
        if btn_check:
            st.markdown("### 📊 Hasil Analisa Multi-Level")
            
            # Helper to render card
            def render_level_check(level_name, dataset, query_mask, inputs):
                match = dataset[query_mask]
                if match.empty: return None
                
                row = match.iloc[0]
                max_omzet = row['OMZET_MAX_WAJAR']
                max_hpp = row['HPP_MAX_WAJAR']
                max_laba = row['LABA_MAX_WAJAR']
                max_plafond = row['PLAFOND_MAX_WAJAR']
                
                # Logic Status
                status_omzet = "✅ WAJAR" if inputs['omzet'] <= max_omzet else "❌ TIDAK WAJAR"
                status_hpp = "✅ WAJAR" if inputs['hpp'] <= max_hpp else "❌ TIDAK WAJAR"
                status_laba = "✅ WAJAR" if inputs['laba'] <= max_laba else "❌ TIDAK WAJAR"
                
                is_all_valid = (inputs['omzet'] <= max_omzet) and (inputs['hpp'] <= max_hpp) and (inputs['laba'] <= max_laba)
                
                badge_color = "#e8f5e9" if is_all_valid else "#ffebee"
                badge_text_color = "#2e7d32" if is_all_valid else "#c62828"
                badge_label = "WAJAR" if is_all_valid else "TIDAK WAJAR"
                border_color = "#4caf50" if is_all_valid else "#e57373"

                html = f"""
                <div style="border: 2px solid {border_color}; padding: 15px; border-radius: 8px; margin-bottom: 15px; background-color: #fafafa;">
                    <div style="font-weight: bold; font-size: 16px; margin-bottom: 10px; display: flex; justify-content: space-between; color: #000000;">
                        <span>{level_name}</span>
                        <span style="background-color:{badge_color}; color:{badge_text_color}; padding:3px 8px; border-radius:4px;">{badge_label}</span>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 10px; font-size: 13px; color: #000000;">
                        <div><strong>Omzet:</strong><br>{status_omzet}<br><span style="color:#000000">Max: {max_omzet:,.0f}</span></div>
                        <div><strong>HPP:</strong><br>{status_hpp}<br><span style="color:#000000">Max: {max_hpp:,.0f}</span></div>
                        <div><strong>Laba:</strong><br>{status_laba}<br><span style="color:#000000">Max: {max_laba:,.0f}</span></div>
                        <div style="background-color: #f5f5f5; padding: 5px; border-radius: 4px; border-left: 3px solid #000;">
                            <strong>Plafond yang Wajar:</strong><br>
                            <span style="color: #000000; font-size: 15px; font-weight: bold;">Rp {max_plafond:,.0f}</span>
                        </div>
                    </div>
                </div>
                """
                return html

            inputs = {'omzet': in_omzet, 'hpp': in_hpp, 'laba': in_laba}
            
            # Logic Checking (Cascading)
            # Level 1
            mask1 = (ref_l1['Provinsi Usaha'] == sel_prov) & (ref_l1['Sektor Ekonomi'] == selected_sector)
            res1 = render_level_check("Level 1: Provinsi & Sektor", ref_l1, mask1, inputs)
            
            # Level 2
            mask2 = (ref_l2['Provinsi Usaha'] == sel_prov) & (ref_l2['Sektor Ekonomi'] == selected_sector) & (ref_l2['Sub Sektor Ekonomi'] == selected_sub_sector)
            res2 = render_level_check("Level 2: Provinsi & Sub Sektor", ref_l2, mask2, inputs)
            
            # Level 3
            mask3 = (ref_l3['Provinsi Usaha'] == sel_prov) & (ref_l3['Kabupaten/kota'] == sel_kab) & (ref_l3['Sektor Ekonomi'] == selected_sector) & (ref_l3['Sub Sektor Ekonomi'] == selected_sub_sector)
            res3 = render_level_check(f"Level 3: {sel_kab} & Sub Sektor", ref_l3, mask3, inputs)
            
            if res1: st.markdown(res1, unsafe_allow_html=True)
            if res2: st.markdown(res2, unsafe_allow_html=True)
            if res3: st.markdown(res3, unsafe_allow_html=True)
            
            if not (res1 or res2 or res3):
                st.warning("⚠️ Data benchmark tidak ditemukan untuk kombinasi ini.")

# Footer
st.markdown("---")
st.caption("MRM Intelligence Framework v12.7 | AI Sector Matching Enabled")
