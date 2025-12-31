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
    
    /* Insight & Interpretation Box */
    .insight-box {
        background-color: #e8f0fe; border-left: 5px solid #1a73e8; padding: 15px; border-radius: 5px; margin-bottom: 20px; color: #000000 !important;
    }
    .interpret-box {
        background-color: #fff3e0; border-left: 5px solid #ff9800; padding: 10px; border-radius: 5px; margin-top: 10px; font-size: 13px; color: #333;
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
    
    # Risk Scoring Logic
    sat_risk = (df['Loan_per_HH'] / 50.0).clip(0, 1) * 100 
    max_pot = df['Skor_Potensi'].max() if df['Skor_Potensi'].max() > 0 else 1
    eco_risk = 100 - ((df['Skor_Potensi'] / max_pot) * 100)
    
    # Risk Flags (Binary)
    df['Flag_Kumuh'] = (df['Risk_Kumuh'] > 0).astype(int)
    df['Flag_Bencana'] = (df['Risk_Bencana'] > 0).astype(int)
    df['Flag_Konflik'] = (df['Risk_Konflik'] > 0).astype(int)
    df['Flag_Saturasi'] = (df['Loan_per_HH'] > 50).astype(int)
    
    # Total Pemicu Risiko (0-4)
    df['Risk_Trigger_Count'] = df['Flag_Kumuh'] + df['Flag_Bencana'] + df['Flag_Konflik'] + df['Flag_Saturasi']
    
    env_risk = (df['Flag_Kumuh'] * 30) + (df['Flag_Bencana'] * 20) + (df['Flag_Konflik'] * 50)
               
    df['Final_Risk_Score'] = (0.3 * sat_risk) + (0.3 * eco_risk) + (0.4 * env_risk)
    
    # Kategori & Kuadran
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

# Color Helpers
def get_hex_risk(score):
    if score < 20: return '#00cc96' # Green
    elif score < 40: return '#ffa15a' # Orange
    elif score < 60: return '#ef553b' # Red Orange
    else: return '#b30000' # Dark Red

def get_hex_potential(score):
    if score > 80: return '#00cc96' # Green
    elif score > 60: return '#636efa' # Blue
    elif score > 40: return '#ab63fa' # Purple
    else: return '#d3d3d3' # Grey

df_filtered['color_hex_risk'] = df_filtered['Final_Risk_Score'].apply(get_hex_risk)
df_filtered['color_hex_pot'] = df_filtered['Skor_Potensi'].apply(get_hex_potential)

# -----------------------------------------------------------------------------
# 5. DASHBOARD TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive Summary", 
    "🚀 Growth & Sentiment", 
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
    
    # 1. FITUR ADJUSTABLE TOP DESA
    st.subheader("🏆 Top Desa: Potensi vs Risiko")
    top_n = st.slider("Jumlah Desa ditampilkan:", min_value=3, max_value=20, value=5)
    
    col_top1, col_top2 = st.columns(2)
    
    with col_top1:
        st.markdown(f"#### 💎 Top {top_n} Desa Potensial (Hidden Gems)")
        gems = df_filtered[df_filtered['Strategy_Quadrant'] == 'Hidden Gem (Grow)']
        if not gems.empty:
            top_gems = gems.nlargest(top_n, 'Skor_Potensi')[['Desa', 'Kecamatan', 'Skor_Potensi', 'Sektor_Dominan']]
            st.dataframe(top_gems, hide_index=True, use_container_width=True, 
                         column_config={"Skor_Potensi": st.column_config.ProgressColumn("Potensi", max_value=100)})
        else:
            st.info("Tidak ada desa Hidden Gem ditemukan.")

    with col_top2:
        st.markdown(f"#### ⚠️ Top {top_n} Desa Berisiko NPL (Critical)")
        risks = df_filtered[df_filtered['Risk_Category'].isin(['High', 'Critical'])]
        if not risks.empty:
            top_risks = risks.nlargest(top_n, 'Final_Risk_Score')[['Desa', 'Kecamatan', 'Final_Risk_Score', 'Risk_Trigger_Count']]
            st.dataframe(top_risks, hide_index=True, use_container_width=True,
                         column_config={"Final_Risk_Score": st.column_config.ProgressColumn("Risk Score", max_value=100, format="%.1f")})
        else:
            st.success("Tidak ada desa risiko kritis.")

# ================= TAB 2: GROWTH & SENTIMENT =================
with tab2:
    st.markdown("### 🚀 Market Sentiment & Growth Intelligence")
    
    c_g1, c_g2 = st.columns([2, 1])
    with c_g1:
        st.subheader("🗺️ Peta Potensi Ekonomi")
        st.map(df_filtered, latitude='lat', longitude='lon', color='color_hex_pot', size=30, zoom=10)
    with c_g2:
        st.subheader("⭐ Sentimen Pasar (Google Maps)")
        chart_sent = alt.Chart(df_filtered).mark_circle(size=80).encode(
            x=alt.X('Skor_Potensi', title='Potensi'),
            y=alt.Y('Sentiment_Score', title='Rating (1-5)', scale=alt.Scale(domain=[3, 5])),
            color=alt.Color('Review_Count', scale=alt.Scale(scheme='tealblues')),
            tooltip=['Desa', 'Sentiment_Score']
        ).interactive()
        st.altair_chart(chart_sent, use_container_width=True)

# ================= TAB 3: SATURATION & INSIGHT =================
with tab3:
    st.markdown("### ⚖️ Analisis Saturasi Mendalam")
    
    # Visualisasi 1: Histogram
    c_s1, c_s2 = st.columns(2)
    with c_s1:
        st.subheader("📊 Distribusi Beban Utang")
        hist = alt.Chart(df_filtered).mark_bar().encode(
            x=alt.X('Loan_per_HH', bin=alt.Bin(maxbins=20), title='Pinjaman per KK (Juta Rp)'),
            y='count()', color=alt.value('#ffa15a'), tooltip=['count()']
        ).properties(height=300)
        st.altair_chart(hist, use_container_width=True)
        
        # 2. NARASI INTERPRETASI
        skewness = df_filtered['Loan_per_HH'].skew()
        if skewness > 1:
            msg = "Grafik condong ke kiri (Positif Skew). Mayoritas desa memiliki beban utang rendah, namun ada beberapa 'Outliers' dengan utang ekstrem yang perlu diawasi."
        elif skewness < -1:
            msg = "Grafik condong ke kanan (Negatif Skew). Mayoritas desa sudah memiliki beban utang tinggi. Pasar cenderung jenuh."
        else:
            msg = "Distribusi normal. Sebaran utang cukup merata di seluruh wilayah."
            
        st.markdown(f"""<div class="interpret-box"><b>💡 Interpretasi Analis:</b><br>{msg}</div>""", unsafe_allow_html=True)

    with c_s2:
        st.subheader("⚔️ Peta Persaingan (Market Share)")
        quad_counts = df_filtered['Strategy_Quadrant'].value_counts().reset_index()
        quad_counts.columns = ['Kategori', 'Jumlah']
        pie = alt.Chart(quad_counts).mark_arc(outerRadius=100).encode(
            theta=alt.Theta("Jumlah", stack=True),
            color=alt.Color("Kategori", scale=alt.Scale(scheme='tableau10')),
            tooltip=["Kategori", "Jumlah"]
        )
        st.altair_chart(pie, use_container_width=True)
        
        # NARASI INTERPRETASI
        dom_quad = quad_counts.iloc[0]['Kategori']
        st.markdown(f"""<div class="interpret-box"><b>💡 Interpretasi Analis:</b><br>Wilayah ini didominasi oleh kategori <b>{dom_quad}</b>. Strategi utama adalah {'Ekspansi Agresif' if 'Hidden Gem' in dom_quad else 'Retensi & Efisiensi'}.</div>""", unsafe_allow_html=True)

    st.markdown("---")
    # 2. DAFTAR DESA DENGAN TINGKAT SATURASI
    st.subheader("📋 Daftar Tingkat Saturasi Per Desa")
    
    sat_df = df_filtered[['Desa', 'Kecamatan', 'Loan_per_HH', 'Total_Pinjaman', 'Strategy_Quadrant']].copy()
    sat_df['Status'] = np.where(sat_df['Loan_per_HH'] > 50, '🔴 Sangat Jenuh', 
                       np.where(sat_df['Loan_per_HH'] > 20, '🟡 Moderat', '🟢 Masih Luas'))
    
    st.dataframe(sat_df.sort_values('Loan_per_HH', ascending=False), hide_index=True, use_container_width=True,
                 column_config={
                     "Loan_per_HH": st.column_config.ProgressColumn("Saturasi (Juta/KK)", format="Rp %.1f", max_value=100),
                     "Total_Pinjaman": st.column_config.NumberColumn("Total Pinjaman", format="Rp %.0f")
                 })

# ================= TAB 4: RISK GUARDIAN =================
with tab4:
    st.markdown("### 🛡️ Profil Risiko & Analisa Dampak")
    
    col_rf1, col_rf2 = st.columns([2, 1])
    
    with col_rf1:
        st.subheader("📋 Daftar Pemicu Risiko Per Desa")
        # 3. TABEL DAFTAR DESA + JUMLAH FAKTOR PENYEBAB
        risk_factors = df_filtered[['Desa', 'Kecamatan', 'Final_Risk_Score', 'Risk_Trigger_Count', 
                                    'Flag_Kumuh', 'Flag_Bencana', 'Flag_Konflik', 'Flag_Saturasi']].copy()
        
        # Rename columns for display
        risk_factors.rename(columns={
            'Flag_Kumuh': 'Faktor_Kumuh', 'Flag_Bencana': 'Faktor_Bencana', 
            'Flag_Konflik': 'Faktor_Konflik', 'Flag_Saturasi': 'Faktor_Saturasi'
        }, inplace=True)
        
        st.dataframe(
            risk_factors.sort_values('Risk_Trigger_Count', ascending=False),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Final_Risk_Score": st.column_config.NumberColumn("Score", format="%.1f"),
                "Risk_Trigger_Count": st.column_config.NumberColumn("Jml Trigger", help="Berapa banyak faktor risiko aktif?"),
                "Faktor_Kumuh": st.column_config.CheckboxColumn("Kumuh?"),
                "Faktor_Bencana": st.column_config.CheckboxColumn("Bencana?"),
                "Faktor_Konflik": st.column_config.CheckboxColumn("Konflik?"),
                "Faktor_Saturasi": st.column_config.CheckboxColumn("Jenuh?")
            }
        )
    
    with col_rf2:
        st.subheader("💡 Analisis Dampak")
        
        # Agregasi faktor risiko
        total_kumuh = risk_factors['Faktor_Kumuh'].sum()
        total_bencana = risk_factors['Faktor_Bencana'].sum()
        total_konflik = risk_factors['Faktor_Konflik'].sum()
        total_saturasi = risk_factors['Faktor_Saturasi'].sum()
        
        st.metric("🏚️ Lingkungan Kumuh", f"{total_kumuh} desa")
        st.caption("Dampak: Kesulitan kolektabilitas, risiko moral hazard meningkat")
        
        st.metric("🌪️ Rawan Bencana", f"{total_bencana} desa")
        st.caption("Dampak: Gangguan cashflow usaha, potensi force majeure")
        
        st.metric("⚔️ Konflik Sosial", f"{total_konflik} desa")
        st.caption("Dampak: Risiko tertinggi, gangguan operasional, reputasi")
        
        st.metric("📊 Saturasi Tinggi", f"{total_saturasi} desa")
        st.caption("Dampak: Kompetisi ketat, margin tipis, risiko default")
    
    st.markdown("---")
    st.info("""
    🎯 **Rekomendasi Mitigasi:**
    - **Desa dengan 3-4 faktor**: Moratorium ekspansi, monitoring intensif, restrukturisasi portofolio existing.
    - **Desa dengan 2 faktor**: Review berkala, limit exposure per nasabah, perkuat kolateral.
    """)

# ================= TAB 5: SCORING & SECTOR =================
with tab5:
    st.markdown("### 🧮 Geo-Credit Scoring & Sector Analysis")
    st.info("Simulasi persetujuan kredit dengan benchmark performa sektor usaha di wilayah ini.")

    with st.container():
        st.markdown('<div class="score-box">', unsafe_allow_html=True)
        col_sim1, col_sim2 = st.columns(2)
        
        with col_sim1:
            st.markdown("#### 1. Input Data Usaha")
            
            # 4. USER DAPAT MEMILIH SEKTOR USAHA
            avail_sectors = df_filtered['Sektor_Dominan'].unique()
            sim_sector = st.selectbox("Sektor Usaha Nasabah", avail_sectors)
            
            sim_desa = st.selectbox("Lokasi Usaha (Desa)", df_filtered['Desa'].unique())
            sim_omzet = st.number_input("Omzet Usaha Bulanan (Juta Rp)", min_value=1.0, value=15.0, step=0.5)
            sim_lama = st.slider("Lama Usaha Berjalan (Tahun)", 0, 30, 3)
            sim_jaminan = st.selectbox("Jenis Agunan", ["Tanpa Agunan", "BPKB Motor", "BPKB Mobil", "Sertifikat Tanah/Rumah"])
            
            # --- SECTOR BENCHMARK LOGIC ---
            # Hitung rata-rata sektor terpilih di wilayah ini
            sector_stats = df_filtered[df_filtered['Sektor_Dominan'] == sim_sector]
            avg_sec_risk = sector_stats['Final_Risk_Score'].mean()
            avg_sec_pot = sector_stats['Skor_Potensi'].mean()
            
            st.markdown("---")
            st.markdown(f"**📊 Benchmark Sektor: {sim_sector}**")
            st.caption(f"Di wilayah {selected_kab}, sektor ini memiliki karakteristik:")
            
            col_b1, col_b2 = st.columns(2)
            col_b1.metric("Rata-rata Risiko", f"{avg_sec_risk:.1f}", 
                          delta="High Risk" if avg_sec_risk > 50 else "Low Risk", delta_color="inverse")
            col_b2.metric("Potensi Wilayah", f"{avg_sec_pot:.1f}", 
                          delta="High Potential" if avg_sec_pot > 60 else "Avg Potential")

        # Calculation Engine
        desa_data = df_filtered[df_filtered['Desa'] == sim_desa].iloc[0]
        loc_score = desa_data['Skor_Potensi']
        risk_loc = desa_data['Final_Risk_Score']
        sentiment_loc = desa_data['Sentiment_Score']
        
        # Hitung Score
        cap_score = min(sim_omzet * 3, 100)
        coll_score = 0
        if sim_jaminan == "Sertifikat Tanah/Rumah": coll_score = 100
        elif sim_jaminan == "BPKB Mobil": coll_score = 70
        elif sim_jaminan == "BPKB Motor": coll_score = 40
        
        # Final Score Formula
        final_score = (0.3 * loc_score) + (0.4 * cap_score) + (0.2 * coll_score) + (0.1 * (sentiment_loc/5)*100)
        
        # Adjustment berdasarkan Benchmark Sektor
        # Jika sektor ini berisiko tinggi secara rata-rata, score dikurangi
        if avg_sec_risk > 60:
            final_score -= 10
            sector_note = "⚠️ Skor dikurangi karena sektor ini berisiko tinggi di wilayah ini."
        else:
            sector_note = "✅ Sektor performa baik."

        if risk_loc > 60: final_score -= 15
        
        with col_sim2:
            st.markdown("#### 2. Hasil Keputusan Kredit")
            st.write(f"**Geo-Credit Score:** {final_score:.1f} / 100")
            
            if final_score >= 75:
                st.success("✅ **APPROVED** (Excellent)")
                plafon = sim_omzet * 5 
                st.metric("Rekomendasi Limit", f"Rp {plafon:,.0f} Juta")
            elif final_score >= 50:
                st.warning("⚠️ **REVIEW NEEDED** (Marginal)")
                plafon = sim_omzet * 2
                st.metric("Rekomendasi Limit", f"Rp {plafon:,.0f} Juta")
            else:
                st.error("❌ **REJECTED** (High Risk)")
                st.write("Alasan: Skor gabungan di bawah threshold atau lokasi berisiko tinggi.")

            st.markdown("---")
            st.caption("📝 **Catatan Analis:**")
            st.write(f"- {sector_note}")
            st.write(f"- Profil Lokasi: Risk {risk_loc:.1f} | Potensi {loc_score:.1f}")
            st.write(f"- Sentimen Pasar: ⭐ {sentiment_loc:.1f}/5.0")
            
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("Geo-Credit Intelligence Framework v10.0 | Full Analytics Suite")
