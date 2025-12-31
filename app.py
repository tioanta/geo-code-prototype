import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Geo-Credit Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. DATA LOADING & PREPROCESSING
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    # Load dataset
    df = pd.read_csv('Prototype Jawa Tengah.csv')
    
    # Clean Column Names (Remove long prefixes)
    df.columns = [col.replace('potensi_wilayah_kel_podes_pdrb_sekda_current.', '') for col in df.columns]
    
    # Rename critical columns for easier access
    df.rename(columns={
        'nama_kabupaten': 'Kabupaten',
        'nama_kecamatan': 'Kecamatan',
        'nama_desa': 'Desa',
        'latitude_desa': 'lat',
        'longitude_desa': 'lon',
        'total_pinjaman_kel': 'Total_Pinjaman',
        'total_simpanan_kel': 'Total_Simpanan',
        'jumlah_industri_mikro': 'Industri_Mikro',
        'attractiveness_index': 'Attractiveness_Score',
        'kelas_potensi_kel': 'Kelas_Potensi'
    }, inplace=True)
    
    # Fill NaN with 0 for numerical columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)
    
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("File 'Prototype Jawa Tengah.csv' tidak ditemukan. Pastikan file ada di folder yang sama.")
    st.stop()

# -----------------------------------------------------------------------------
# 3. SIDEBAR FILTERS
# -----------------------------------------------------------------------------
st.sidebar.title("🔍 Geo-Filter")
st.sidebar.info("Filter wilayah untuk analisis spesifik.")

# Filter Kabupaten
selected_kab = st.sidebar.selectbox("Pilih Kabupaten", df['Kabupaten'].unique())
df_kab = df[df['Kabupaten'] == selected_kab]

# Filter Kecamatan
selected_kec = st.sidebar.multiselect(
    "Pilih Kecamatan", 
    options=df_kab['Kecamatan'].unique(),
    default=df_kab['Kecamatan'].unique()
)
df_filtered = df_kab[df_kab['Kecamatan'].isin(selected_kec)]

# -----------------------------------------------------------------------------
# 4. MAIN DASHBOARD
# -----------------------------------------------------------------------------
st.title(f"🌍 Geo-Credit Intelligence: {selected_kab}")
st.markdown("### Analisis Potensi Wilayah & Risiko Kredit Mikro")

# --- KPI METRICS ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Desa", f"{len(df_filtered):,}")
with col2:
    avg_score = df_filtered['Attractiveness_Score'].mean()
    st.metric("Rata-rata Skor Wilayah", f"{avg_score:.2f}")
with col3:
    total_loan = df_filtered['Total_Pinjaman'].sum() / 1e9
    st.metric("Total Exposure Kredit (Miliar)", f"Rp {total_loan:,.1f} M")
with col4:
    total_deposit = df_filtered['Total_Simpanan'].sum() / 1e9
    st.metric("Total Simpanan (Miliar)", f"Rp {total_deposit:,.1f} M")

# --- GEOSPATIAL MAP ---
st.markdown("---")
st.subheader(f"📍 Peta Sebaran Potensi di {selected_kab}")

# Color mapping logic
def get_color(val):
    if val == 'High': return '#00cc96' # Green
    elif val == 'Medium High': return '#636efa' # Blue
    elif val == 'Medium Low': return '#ffa15a' # Orange
    else: return '#ef553b' # Red

df_filtered['color'] = df_filtered['Kelas_Potensi'].apply(get_color)

# Map Visualization using Streamlit Map
st.map(df_filtered, latitude='lat', longitude='lon', color='color', size=20, zoom=10)
st.caption("Warna merepresentasikan Kelas Potensi (Hijau=High, Merah=Low)")

# --- CHARTS ---
st.markdown("---")
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("📊 Pinjaman vs Simpanan (LDR Proxy)")
    chart = alt.Chart(df_filtered).mark_circle(size=60).encode(
        x=alt.X('Total_Pinjaman', title='Total Pinjaman'),
        y=alt.Y('Total_Simpanan', title='Total Simpanan'),
        color='Kelas_Potensi',
        tooltip=['Desa', 'Kecamatan', 'Total_Pinjaman', 'Total_Simpanan', 'Industri_Mikro']
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

with col_chart2:
    st.subheader("🏭 Sebaran Industri Mikro")
    bar_chart = alt.Chart(df_filtered).mark_bar().encode(
        x=alt.X('Industri_Mikro', bin=True),
        y='count()',
        color='Kelas_Potensi'
    )
    st.altair_chart(bar_chart, use_container_width=True)

# -----------------------------------------------------------------------------
# 5. SIMULATION ENGINE (FEATURE ENGINEERING DEMO)
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("🧮 Geo-Credit Scoring Simulator")
st.write("Simulasi skor risiko untuk nasabah baru berdasarkan lokasi desa mereka.")

with st.expander("Buka Simulator"):
    sim_col1, sim_col2 = st.columns(2)
    
    with sim_col1:
        target_desa = st.selectbox("Pilih Desa Lokasi Usaha", df_filtered['Desa'].unique())
        nasabah_omzet = st.number_input("Omzet Bulanan Nasabah (Juta Rp)", min_value=1.0, value=10.0)
        
    # Retrieve Desa Data
    desa_data = df_filtered[df_filtered['Desa'] == target_desa].iloc[0]
    
    # SIMPLE SCORING LOGIC (HYBRID)
    # 1. Location Score (From Data) -> Normalized 0-100
    loc_score = desa_data['Attractiveness_Score'] 
    
    # 2. Financial Score (Dummy Logic for Prototype)
    # Asumsi: Omzet > 50 Juta skor bagus
    fin_score = min(nasabah_omzet * 2, 100) 
    
    # 3. Hybrid Score (60% Financial + 40% Location)
    final_score = (0.6 * fin_score) + (0.4 * loc_score)
    
    with sim_col2:
        st.metric("Skor Lokasi (External)", f"{loc_score:.1f}/100")
        st.metric("Skor Kapasitas (Internal)", f"{fin_score:.1f}/100")
        
        # Risk Grade
        if final_score >= 80: grade, color = "Low Risk (A)", "green"
        elif final_score >= 50: grade, color = "Medium Risk (B)", "orange"
        else: grade, color = "High Risk (C)", "red"
        
        st.markdown(f"### Final Geo-Score: :{color}[{final_score:.1f}]")
        st.markdown(f"**Rekomendasi:** {grade}")

# Footer
st.markdown("---")
st.caption("Developed by Tio Brain | Prototype v1.0")
