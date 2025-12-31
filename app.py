import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Geo-Credit Risk & Saturation",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional UI
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #2e7bcf;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    h1, h2, h3 { color: #2c3e50; }
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
    
    # Rename critical columns
    df.rename(columns={
        'nama_kabupaten': 'Kabupaten',
        'nama_kecamatan': 'Kecamatan',
        'nama_desa': 'Desa',
        'latitude_desa': 'lat',
        'longitude_desa': 'lon',
        'total_pinjaman_kel': 'Total_Pinjaman',
        'total_simpanan_kel': 'Total_Simpanan',
        'jumlah_keluarga_pengguna_listrik': 'Jumlah_KK', # Proxy for Households
        'attractiveness_index': 'Skor_Potensi',
        'max_tipe_usaha': 'Sektor_Dominan'
    }, inplace=True)
    
    # Fill NaN
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)
    
    # --- FEATURE ENGINEERING: SATURATION METRICS ---
    # 1. Loan Density (Rata-rata Pinjaman per Keluarga)
    # Menghindari pembagian dengan nol
    df['Jumlah_KK'] = df['Jumlah_KK'].replace(0, 1) 
    df['Loan_per_HH'] = (df['Total_Pinjaman'] / df['Jumlah_KK']) / 1_000_000 # Dalam Juta Rupiah
    
    # 2. Saturation Level Classification (Dynamic Quintiles)
    # Kita bagi menjadi 3 level: Low, Optimal, Saturated berdasarkan distribusi data
    q33 = df['Loan_per_HH'].quantile(0.33)
    q66 = df['Loan_per_HH'].quantile(0.66)
    
    def classify_saturation(x):
        if x < q33: return 'Low Saturation (Underbanked)'
        elif x < q66: return 'Optimal'
        else: return 'High Saturation (Overbanked)'
        
    df['Status_Saturasi'] = df['Loan_per_HH'].apply(classify_saturation)
    
    # 3. Strategy Quadrant
    # Membandingkan Potensi (X) vs Saturasi (Y)
    avg_potensi = df['Skor_Potensi'].mean()
    avg_saturasi = df['Loan_per_HH'].mean()
    
    def get_quadrant(row):
        high_potensi = row['Skor_Potensi'] >= avg_potensi
        high_saturasi = row['Loan_per_HH'] >= avg_saturasi
        
        if high_potensi and not high_saturasi:
            return "💎 Hidden Gem (Grow)"
        elif high_potensi and high_saturasi:
            return "⚔️ Red Ocean (Compete)"
        elif not high_potensi and high_saturasi:
            return "⚠️ High Risk (Avoid)"
        else:
            return "💤 Dormant (Monitor)"

    df['Strategy_Quadrant'] = df.apply(get_quadrant, axis=1)

    return df

df = load_and_process_data()

if df is None:
    st.error("⚠️ File data tidak ditemukan.")
    st.stop()

# -----------------------------------------------------------------------------
# 3. SIDEBAR
# -----------------------------------------------------------------------------
st.sidebar.title("🛡️ Risk Control")
selected_kab = st.sidebar.selectbox("Pilih Kabupaten", df['Kabupaten'].unique())
df_kab = df[df['Kabupaten'] == selected_kab]

selected_kec = st.sidebar.multiselect("Filter Kecamatan", df_kab['Kecamatan'].unique(), default=df_kab['Kecamatan'].unique())
df_filtered = df_kab[df_kab['Kecamatan'].isin(selected_kec)]

# -----------------------------------------------------------------------------
# 4. MAIN DASHBOARD: SATURATION ANALYSIS
# -----------------------------------------------------------------------------
st.title(f"Analisis Saturasi Kredit: {selected_kab}")
st.markdown("Identifikasi area yang *Underbanked* (Peluang) vs *Overbanked* (Risiko).")

# --- KPI SATURASI ---
c1, c2, c3, c4 = st.columns(4)
with c1:
    avg_loan_hh = df_filtered['Loan_per_HH'].mean()
    st.metric("Rata-rata Pinjaman/KK", f"Rp {avg_loan_hh:,.1f} Juta")
with c2:
    saturated_count = len(df_filtered[df_filtered['Status_Saturasi'] == 'High Saturation (Overbanked)'])
    st.metric("Desa Jenuh (High Risk)", f"{saturated_count} Desa")
with c3:
    opportunity_count = len(df_filtered[df_filtered['Strategy_Quadrant'] == '💎 Hidden Gem (Grow)'])
    st.metric("Desa Peluang (Hidden Gem)", f"{opportunity_count} Desa", delta="Target Ekspansi", delta_color="normal")
with c4:
    max_loan = df_filtered['Loan_per_HH'].max()
    max_desa = df_filtered.loc[df_filtered['Loan_per_HH'].idxmax(), 'Desa']
    st.metric("Saturasi Tertinggi", f"{max_desa}", f"Rp {max_loan:,.1f} Juta/KK")

# --- QUADRANT STRATEGY CHART ---
st.markdown("---")
st.subheader("🎯 Matriks Strategi: Potensi vs Saturasi")
st.info("Grafik ini memetakan posisi setiap desa. Arahkan kursor untuk melihat detail.")

quadrant_chart = alt.Chart(df_filtered).mark_circle(size=100).encode(
    x=alt.X('Skor_Potensi', title='Skor Potensi Ekonomi (Makin Kanan Makin Bagus)'),
    y=alt.Y('Loan_per_HH', title='Tingkat Saturasi (Pinjaman Juta/KK)'),
    color=alt.Color('Strategy_Quadrant', 
                    scale=alt.Scale(domain=['💎 Hidden Gem (Grow)', '⚔️ Red Ocean (Compete)', '⚠️ High Risk (Avoid)', '💤 Dormant (Monitor)'],
                                    range=['#00CC96', '#636EFA', '#EF553B', '#B8B8B8'])),
    tooltip=['Desa', 'Kecamatan', 'Skor_Potensi', 'Loan_per_HH', 'Total_Pinjaman', 'Strategy_Quadrant']
).properties(height=500).interactive()

# Add Reference Lines (Averages)
rule_x = alt.Chart(df_filtered).mark_rule(color='gray', strokeDash=[5,5]).encode(x='mean(Skor_Potensi)')
rule_y = alt.Chart(df_filtered).mark_rule(color='gray', strokeDash=[5,5]).encode(y='mean(Loan_per_HH)')

st.altair_chart(quadrant_chart + rule_x + rule_y, use_container_width=True)

# --- SATURATION MAP ---
st.markdown("---")
col_map, col_list = st.columns([2, 1])

with col_map:
    st.subheader("📍 Peta Panas Saturasi (Risk Map)")
    
    # Color mapping for map
    def get_sat_color(val):
        if val == 'High Saturation (Overbanked)': return '#ff0000' # Red
        elif val == 'Optimal': return '#ffa500' # Orange
        else: return '#008000' # Green
        
    df_filtered['sat_color'] = df_filtered['Status_Saturasi'].apply(get_sat_color)
    
    st.map(df_filtered, latitude='lat', longitude='lon', color='sat_color', size=25, zoom=10)
    st.caption("Merah = Jenuh (Risiko Tinggi), Hijau = Masih Kosong (Peluang)")

with col_list:
    st.subheader("📋 Top 10 Desa 'Hidden Gems'")
    st.markdown("Prioritas untuk penetrasi pasar baru.")
    
    hidden_gems = df_filtered[df_filtered['Strategy_Quadrant'] == '💎 Hidden Gem (Grow)']
    top_gems = hidden_gems.nlargest(10, 'Skor_Potensi')[['Desa', 'Kecamatan', 'Skor_Potensi', 'Loan_per_HH']]
    
    st.dataframe(top_gems, hide_index=True)

# --- DETAIL DATA TABLE ---
st.markdown("---")
with st.expander("Lihat Data Lengkap Per Desa"):
    st.dataframe(df_filtered[['Desa', 'Kecamatan', 'Skor_Potensi', 'Total_Pinjaman', 'Jumlah_KK', 'Loan_per_HH', 'Status_Saturasi', 'Strategy_Quadrant']])

st.markdown("---")
st.caption("Developed by Tio Brain | Risk Modelling & Strategy Framework")
