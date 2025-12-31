import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Geo-Credit Risk Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Styling
st.markdown("""
<style>
    .risk-badge-high { background-color: #ffcccc; color: #cc0000; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .risk-badge-med { background-color: #fff4cc; color: #996600; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .risk-badge-low { background-color: #ccffcc; color: #006600; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    div[data-testid="stMetricValue"] { font-size: 24px; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA LOADING & ENGINEERING
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_process_data():
    try:
        df = pd.read_csv('Prototype Jawa Tengah.csv')
    except FileNotFoundError:
        return None

    # Clean Column Names
    df.columns = [col.replace('potensi_wilayah_kel_podes_pdrb_sekda_current.', '') for col in df.columns]
    
    # Rename & Select Critical Columns
    df.rename(columns={
        'nama_kabupaten': 'Kabupaten',
        'nama_kecamatan': 'Kecamatan',
        'nama_desa': 'Desa',
        'latitude_desa': 'lat',
        'longitude_desa': 'lon',
        'total_pinjaman_kel': 'Total_Pinjaman',
        'total_simpanan_kel': 'Total_Simpanan',
        'jumlah_keluarga_pengguna_listrik': 'Jumlah_KK',
        'attractiveness_index': 'Skor_Potensi',
        'max_tipe_usaha': 'Sektor_Dominan',
        # Risk Factors
        'jumlah_lokasi_permukiman_kumuh': 'Risk_Kumuh',
        'bencana_alam': 'Risk_Bencana',
        'jumlah_perkelahian_masyarakat': 'Risk_Konflik'
    }, inplace=True)
    
    # Fill NaN
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)
    
    # --- A. SATURATION METRIC ---
    df['Jumlah_KK'] = df['Jumlah_KK'].replace(0, 1) 
    df['Loan_per_HH'] = (df['Total_Pinjaman'] / df['Jumlah_KK']) / 1_000_000
    
    # --- B. COMPOSITE RISK SCORING ALGORITHM ---
    # 1. Saturation Risk (0-100) -> Higher loan/HH = Higher Risk
    # Cap at 50 Juta/KK as max risk reference
    df['Score_Saturation'] = (df['Loan_per_HH'] / 50.0).clip(0, 1) * 100 
    
    # 2. Economic Vulnerability Risk (0-100) -> Lower Potential = Higher Risk
    # Invert Attractiveness Index (Assuming max score ~100)
    max_potensi = df['Skor_Potensi'].max()
    df['Score_Economic'] = 100 - ((df['Skor_Potensi'] / max_potensi) * 100)
    
    # 3. Environmental & Social Risk (Flags)
    # Give penalty points
    df['Score_Env_Social'] = 0
    df.loc[df['Risk_Kumuh'] > 0, 'Score_Env_Social'] += 30 # High impact on collateral
    df.loc[df['Risk_Bencana'] > 0, 'Score_Env_Social'] += 20 # Business continuity risk
    df.loc[df['Risk_Konflik'] > 0, 'Score_Env_Social'] += 50 # Collection risk (Red Flag)
    
    # 4. FINAL DISBURSEMENT RISK SCORE (Weighted Average)
    # Formula: 30% Saturation + 30% Economic + 40% Env/Social
    df['Final_Risk_Score'] = (0.3 * df['Score_Saturation']) + \
                             (0.3 * df['Score_Economic']) + \
                             (0.4 * df['Score_Env_Social'])
                             
    # Classification
    def get_risk_level(x):
        if x >= 60: return '🔴 Critical Risk'
        elif x >= 40: return '🟠 High Risk'
        elif x >= 20: return '🟡 Medium Risk'
        else: return '🟢 Low Risk'
        
    df['Risk_Level'] = df['Final_Risk_Score'].apply(get_risk_level)
    
    # Create "Risk Tags" for tooltip
    def get_risk_tags(row):
        tags = []
        if row['Risk_Kumuh'] > 0: tags.append("Kumuh")
        if row['Risk_Bencana'] > 0: tags.append("Rawan Bencana")
        if row['Risk_Konflik'] > 0: tags.append("Konflik Sosial")
        if row['Score_Saturation'] > 80: tags.append("Over-Indebted")
        return ", ".join(tags) if tags else "Clean"
        
    df['Risk_Factors'] = df.apply(get_risk_tags, axis=1)

    return df

df = load_and_process_data()
if df is None:
    st.error("Data not found.")
    st.stop()

# -----------------------------------------------------------------------------
# 3. SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.title("🎛️ Risk Dashboard")
selected_kab = st.sidebar.selectbox("Wilayah Operasional", df['Kabupaten'].unique())
df_kab = df[df['Kabupaten'] == selected_kab]

selected_kec = st.sidebar.multiselect("Filter Kecamatan", df_kab['Kecamatan'].unique(), default=df_kab['Kecamatan'].unique())
df_filtered = df_kab[df_kab['Kecamatan'].isin(selected_kec)]

# -----------------------------------------------------------------------------
# 4. MAIN DASHBOARD
# -----------------------------------------------------------------------------
st.title(f"Analisis Risiko Penyaluran: {selected_kab}")
st.markdown("### Profil Risiko Kewilayahan (Disbursement Risk)")

# --- KPI RISIKO ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    high_risk_count = len(df_filtered[df_filtered['Risk_Level'].isin(['🔴 Critical Risk', '🟠 High Risk'])])
    st.metric("Desa Risiko Tinggi (Stop/Limit)", f"{high_risk_count}", delta="Mitigasi Segera", delta_color="inverse")
with col2:
    avg_risk = df_filtered['Final_Risk_Score'].mean()
    st.metric("Rata-rata Skor Risiko Wilayah", f"{avg_risk:.1f}/100", help="0=Aman, 100=Sangat Bahaya")
with col3:
    conflict_area = len(df_filtered[df_filtered['Risk_Konflik'] > 0])
    st.metric("Area Rawan Konflik (Blacklist)", f"{conflict_area} Desa")
with col4:
    disaster_area = len(df_filtered[df_filtered['Risk_Bencana'] > 0])
    st.metric("Area Rawan Bencana", f"{disaster_area} Desa")

# --- RISK MAP ---
st.markdown("---")
st.subheader("📍 Peta Zonasi Risiko Kredit")

# Color Logic
def risk_color(level):
    if 'Critical' in level: return '#ff0000' # Red
    if 'High' in level: return '#ff8c00' # Dark Orange
    if 'Medium' in level: return '#ffd700' # Gold
    return '#32cd32' # Lime Green

df_filtered['color'] = df_filtered['Risk_Level'].apply(risk_color)

col_map, col_details = st.columns([2, 1])

with col_map:
    st.map(df_filtered, latitude='lat', longitude='lon', color='color', size=30, zoom=10)
    st.caption("🔴 Merah: Critical Risk (Ada Konflik/Saturasi Ekstrem) | 🟢 Hijau: Low Risk (Safe for Lending)")

with col_details:
    st.subheader("⚠️ Top 5 Desa Paling Berisiko")
    risky_villages = df_filtered.sort_values(by='Final_Risk_Score', ascending=False).head(5)
    
    for _, row in risky_villages.iterrows():
        st.markdown(f"""
        <div style="background-color: #ffe6e6; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
            <strong>{row['Desa']}</strong> ({row['Kecamatan']})<br>
            <span style="color: red; font-size: 12px;">Risk Score: {row['Final_Risk_Score']:.1f}</span><br>
            <span style="font-size: 11px;">🚫 Faktor: {row['Risk_Factors']}</span>
        </div>
        """, unsafe_allow_html=True)

# --- DETAILED RISK TABLE ---
st.markdown("---")
st.subheader("📋 Detail Profil Risiko Per Desa")

# Styling function for dataframe
def highlight_risk(val):
    color = 'red' if 'Critical' in val or 'High' in val else 'black'
    return f'color: {color}; font-weight: bold'

display_cols = ['Desa', 'Kecamatan', 'Risk_Level', 'Final_Risk_Score', 'Risk_Factors', 'Loan_per_HH', 'Risk_Bencana', 'Risk_Konflik']

st.dataframe(
    df_filtered[display_cols].style.applymap(highlight_risk, subset=['Risk_Level']),
    use_container_width=True,
    column_config={
        "Final_Risk_Score": st.column_config.ProgressColumn("Risk Score", min_value=0, max_value=100, format="%.1f"),
        "Loan_per_HH": st.column_config.NumberColumn("Pinjaman/KK (Juta)", format="Rp %.1f"),
        "Risk_Bencana": st.column_config.NumberColumn("Idx Bencana"),
        "Risk_Konflik": st.column_config.NumberColumn("Idx Konflik"),
    }
)

# --- RECOMMENDATION ENGINE ---
st.markdown("---")
st.subheader("💡 Rekomendasi Kebijakan Penyaluran (Credit Policy)")

risk_policy = {
    "🔴 Critical Risk": "⛔ **STOP LENDING / BLACKLIST.** Risiko gagal bayar & sulit tagih sangat tinggi. Fokus pada penagihan berjalan saja.",
    "🟠 High Risk": "⚠️ **RESTRICTED.** Maksimum Plafond Rp 10 Juta. Wajib agunan fisik 150%. Hindari pinjaman tanpa agunan.",
    "🟡 Medium Risk": "✋ **CAUTION.** Lakukan survei lapis dua (Two-layer verification). Monitor ketat pembayaran 3 bulan pertama.",
    "🟢 Low Risk": "✅ **AGGRESSIVE GROWTH.** Berikan pre-approved limit. Tawarkan promo bunga kompetitif."
}

for level, policy in risk_policy.items():
    st.markdown(f"**{level}:** {policy}")

st.markdown("---")
st.caption("Geo-Credit Intelligence Framework v4.0 | Created by Tio Brain")
