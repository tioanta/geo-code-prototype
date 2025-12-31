import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import random

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Geo-Credit Intelligence Pro",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Corporate Look"
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
    }
    .stMetric {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA LOADING & PREPROCESSING
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    # Load dataset
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
        'jumlah_industri_mikro': 'Industri_Mikro',
        'attractiveness_index': 'Attractiveness_Score',
        'kelas_potensi_kel': 'Kelas_Potensi',
        'max_tipe_usaha': 'Sektor_Dominan'
    }, inplace=True)
    
    # Clean Sector Names
    df['Sektor_Dominan'] = df['Sektor_Dominan'].str.replace('_', ' ').str.title()
    df['Sektor_Dominan'] = df['Sektor_Dominan'].str.replace('Dan', '&')
    
    # Fill NaN
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)
    
    return df

df = load_data()
if df is None:
    st.error("⚠️ File 'Prototype Jawa Tengah.csv' tidak ditemukan. Mohon upload file data terlebih dahulu.")
    st.stop()

# -----------------------------------------------------------------------------
# 3. HELPER FUNCTION: GOOGLE MAPS SENTIMENT SIMULATOR
# -----------------------------------------------------------------------------
def get_market_sentiment(kabupaten, sectors_list):
    """
    Simulates fetching aggregated Google Maps reviews for business sectors in a region.
    In a real production app, this would call a Google Places API.
    """
    # Seed random for consistency per session
    np.random.seed(len(kabupaten)) 
    
    sentiment_data = []
    for sector in sectors_list:
        # Simulate logic: Higher potential sectors usually have more competition & varied reviews
        base_rating = np.random.uniform(3.8, 4.8)
        review_count = np.random.randint(50, 5000)
        
        # Sentiment tags
        tags = ["Pelayanan Cepat", "Harga Murah", "Parkir Luas", "Bersih", "Produk Lengkap"]
        top_tag = np.random.choice(tags)
        
        sentiment_data.append({
            "Sektor Usaha": sector,
            "Rata-rata Rating": round(base_rating, 1),
            "Jumlah Ulasan": review_count,
            "Sentimen Positif Utama": top_tag
        })
    
    return pd.DataFrame(sentiment_data).sort_values(by="Rata-rata Rating", ascending=False)

# -----------------------------------------------------------------------------
# 4. SIDEBAR FILTERS
# -----------------------------------------------------------------------------
st.sidebar.title("🌍 Geo-Credit Control")
st.sidebar.caption("v2.0 - Market Intelligence Enabled")

# Filter Kabupaten
selected_kab = st.sidebar.selectbox("Pilih Wilayah (Kabupaten)", df['Kabupaten'].unique())
df_kab = df[df['Kabupaten'] == selected_kab]

# Filter Kecamatan
selected_kec = st.sidebar.multiselect(
    "Filter Kecamatan", 
    options=df_kab['Kecamatan'].unique(),
    default=df_kab['Kecamatan'].unique()
)
df_filtered = df_kab[df_kab['Kecamatan'].isin(selected_kec)]

# -----------------------------------------------------------------------------
# 5. MAIN DASHBOARD
# -----------------------------------------------------------------------------
st.title(f"Analisis Strategis: {selected_kab}")
st.markdown("Dashboard Intelegensi Kredit Mikro & Sentimen Pasar")

# --- A. EXECUTIVE SUMMARY (KPIs) ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Coverage Area", f"{len(df_filtered)} Desa")
with col2:
    avg_score = df_filtered['Attractiveness_Score'].mean()
    delta = avg_score - df['Attractiveness_Score'].mean() # Compare to National Avg
    st.metric("Skor Potensi Wilayah", f"{avg_score:.2f}", delta=f"{delta:.2f} vs Rata-rata Prov")
with col3:
    total_loan = df_filtered['Total_Pinjaman'].sum() / 1e9
    st.metric("Total Exposure Kredit", f"Rp {total_loan:,.1f} M")
with col4:
    ldr = (df_filtered['Total_Pinjaman'].sum() / df_filtered['Total_Simpanan'].sum()) * 100
    st.metric("Loan-to-Deposit Ratio (LDR)", f"{ldr:.1f}%")

# --- B. GOOGLE MAPS MARKET INTELLIGENCE (NEW FEATURE) ---
st.markdown("---")
st.subheader(f"⭐ Market Sentiment: Sektor Paling Direkomendasikan di {selected_kab}")
st.markdown("""
*Analisis ini menggabungkan kepadatan bisnis (Internal Data) dengan **kepuasan pelanggan (Simulasi Google Reviews)** untuk menemukan sektor 'Sweet Spot': Permintaan tinggi, Kepuasan tinggi.*
""")

# Get unique sectors in this region
available_sectors = df_filtered['Sektor_Dominan'].unique()
# Generate Insights
df_sentiment = get_market_sentiment(selected_kab, available_sectors)

# Display Layout
col_sent1, col_sent2 = st.columns([1, 2])

with col_sent1:
    # Display Top 1 Sector prominently
    top_sector = df_sentiment.iloc[0]
    st.info(f"🏆 **Top Sector Winner:**\n\n### {top_sector['Sektor Usaha']}")
    st.write(f"⭐ **Rating:** {top_sector['Rata-rata Rating']} / 5.0")
    st.write(f"🗣️ **Total Ulasan:** {top_sector['Jumlah Ulasan']:,}")
    st.write(f"👍 **Kata Kunci:** {top_sector['Sentimen Positif Utama']}")
    
    st.markdown("#### Rekomendasi Strategi:")
    if top_sector['Rata-rata Rating'] > 4.5:
        st.success("**Pertahankan:** Tawarkan produk KUR Khusus untuk ekspansi usaha di sektor ini.")
    else:
        st.warning("**Perbaiki:** Sektor ramai tapi komplain tinggi. Peluang masuk bagi pemain baru dengan layanan lebih baik.")

with col_sent2:
    # Bar Chart for Ratings
    chart_sentiment = alt.Chart(df_sentiment).mark_bar().encode(
        x=alt.X('Rata-rata Rating', scale=alt.Scale(domain=[3.5, 5.0])),
        y=alt.Y('Sektor Usaha', sort='-x'),
        color=alt.Color('Rata-rata Rating', scale=alt.Scale(scheme='greens')),
        tooltip=['Sektor Usaha', 'Rata-rata Rating', 'Jumlah Ulasan', 'Sentimen Positif Utama']
    ).properties(height=300)
    
    st.altair_chart(chart_sentiment, use_container_width=True)

# --- C. GEOSPATIAL RISK MAP ---
st.markdown("---")
st.subheader("📍 Peta Risiko & Potensi Mikro")

# Color mapping logic
def get_color(val):
    if val == 'High': return '#00cc96' # Green
    elif val == 'Medium High': return '#636efa' # Blue
    elif val == 'Medium Low': return '#ffa15a' # Orange
    else: return '#ef553b' # Red

df_filtered['color'] = df_filtered['Kelas_Potensi'].apply(get_color)

# Map Visualization
st.map(df_filtered, latitude='lat', longitude='lon', color='color', size=20, zoom=10)
st.caption("Hijau = Potensi Tinggi (Low Risk), Merah = Potensi Rendah (High Risk)")

# --- D. SCORING SIMULATOR (MONETIZATION DEMO) ---
st.markdown("---")
st.subheader("🧮 Geo-Credit Scoring Engine (Demo)")
st.info("Fitur ini dapat dijual sebagai API ke BPR atau Fintech (Pricing: Rp 5.000/hit).")

c1, c2, c3 = st.columns(3)
with c1:
    sim_desa = st.selectbox("Lokasi Calon Debitur", df_filtered['Desa'].unique())
    sim_sector = st.selectbox("Sektor Usaha", available_sectors)
with c2:
    sim_omzet = st.number_input("Omzet Bulanan (Juta Rp)", 5.0, 500.0, 15.0)
    sim_lama = st.slider("Lama Usaha (Tahun)", 1, 20, 3)

# Retrieve Data & Calculate
desa_data = df_filtered[df_filtered['Desa'] == sim_desa].iloc[0]
sector_sentiment = df_sentiment[df_sentiment['Sektor Usaha'] == sim_sector]['Rata-rata Rating'].values[0]

# Advanced Scoring Formula
# Score = (0.3 * Location_Score) + (0.3 * Financial_Capacity) + (0.2 * Sector_Health) + (0.2 * Experience)
loc_score = desa_data['Attractiveness_Score']
fin_score = min(sim_omzet * 2, 100)
sec_score = (sector_sentiment / 5.0) * 100
exp_score = min(sim_lama * 10, 100)

final_score = (0.3 * loc_score) + (0.3 * fin_score) + (0.2 * sec_score) + (0.2 * exp_score)

with c3:
    st.markdown(f"### Score: {final_score:.1f}")
    if final_score >= 75:
        st.success("✅ **RECOMMENDED** (Limit: Rp 50-100 Juta)")
    elif final_score >= 50:
        st.warning("⚠️ **REVIEW NEEDED** (Limit: Rp 10-25 Juta)")
    else:
        st.error("❌ **REJECT** (High Risk)")

st.markdown("---")
st.caption("© 2025 Tio Brain - Geo-Credit Intelligence Framework")
