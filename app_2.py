import streamlit as st
import pandas as pd

# Konfigurasi Halaman
st.set_page_config(page_title="Aplikasi Penilaian Kewajaran - BRI", layout="wide")

# --- 1. Fungsi Load Data ---
@st.cache_data
def load_data():
    # Pastikan nama file sesuai dengan yang ada di folder data Anda
    try:
        df_l1 = pd.read_csv("data/Kewajaran_Omzet_All.xlsx - Level 1.csv")
        df_l2 = pd.read_csv("data/Kewajaran_Omzet_All.xlsx - Level 2.csv")
        df_l3 = pd.read_csv("data/Kewajaran_Omzet_All.xlsx - Level 3.csv")
        
        # Bersihkan nama kolom agar seragam (menghapus spasi tambahan jika ada)
        for df in [df_l1, df_l2, df_l3]:
            df.columns = df.columns.str.strip()
            
        return df_l1, df_l2, df_l3
    except FileNotFoundError as e:
        st.error(f"File data tidak ditemukan: {e}")
        return None, None, None

df_level_1, df_level_2, df_level_3 = load_data()

if df_level_1 is not None:
    # --- 2. Sidebar: Input Data Wilayah & Sektor ---
    st.sidebar.header("📝 Input Data Nasabah")

    # A. Pilih Provinsi (Ambil dari Level 1 untuk list lengkap provinsi)
    provinsi_list = sorted(df_level_1['Provinsi Usaha'].unique())
    selected_provinsi = st.sidebar.selectbox("Pilih Provinsi", provinsi_list)

    # B. Pilih Kabupaten/Kota (Filter Level 3 berdasarkan Provinsi)
    kota_list = sorted(df_level_3[df_level_3['Provinsi Usaha'] == selected_provinsi]['Kabupaten/kota'].unique())
    selected_kota = st.sidebar.selectbox("Pilih Kabupaten/Kota", kota_list)

    # C. Pilih Sektor Ekonomi (Ambil dari Level 1)
    sektor_list = sorted(df_level_1['Sektor Ekonomi'].unique())
    selected_sektor = st.sidebar.selectbox("Pilih Sektor Ekonomi", sektor_list)

    # D. Pilih Sub Sektor Ekonomi (Filter Level 2 berdasarkan Sektor)
    # Kita ambil dari Level 2 karena Level 1 tidak punya sub sektor
    sub_sektor_list = sorted(df_level_2[df_level_2['Sektor Ekonomi'] == selected_sektor]['Sub Sektor Ekonomi'].unique())
    selected_sub_sektor = st.sidebar.selectbox("Pilih Sub Sektor Ekonomi", sub_sektor_list)

    st.sidebar.markdown("---")
    
    # --- 3. Sidebar: Input Nilai Keuangan ---
    st.sidebar.header("💰 Input Nilai Keuangan")
    input_omset = st.sidebar.number_input("Omset (Rp)", min_value=0.0, step=1000000.0)
    input_hpp = st.sidebar.number_input("HPP (Rp)", min_value=0.0, step=1000000.0)
    input_laba = st.sidebar.number_input("Laba (Rp)", min_value=0.0, step=1000000.0)
    input_plafond = st.sidebar.number_input("Plafond Pinjaman (Rp)", min_value=0.0, step=1000000.0)

    # Tombol Validasi
    cek_validasi = st.sidebar.button("🔍 Cek Kewajaran")

    # --- 4. Halaman Utama: Hasil Analisis ---
    st.title("Sistem Penilaian Kewajaran Segmen Mikro")
    st.markdown(f"**Wilayah:** {selected_provinsi}, {selected_kota} | **Usaha:** {selected_sektor} - {selected_sub_sektor}")

    if cek_validasi:
        st.subheader("Hasil Validasi")

        # --- LOGIKA PENCARIAN DATA REFERENSI ---
        
        # 1. Cari Data Level 3 (Paling Spesifik: Sektor, SubSektor, Prov, Kota)
        ref_l3 = df_level_3[
            (df_level_3['Sektor Ekonomi'] == selected_sektor) &
            (df_level_3['Sub Sektor Ekonomi'] == selected_sub_sektor) &
            (df_level_3['Provinsi Usaha'] == selected_provinsi) &
            (df_level_3['Kabupaten/kota'] == selected_kota)
        ]

        # 2. Cari Data Level 2 (Menengah: Sektor, SubSektor, Prov)
        ref_l2 = df_level_2[
            (df_level_2['Sektor Ekonomi'] == selected_sektor) &
            (df_level_2['Sub Sektor Ekonomi'] == selected_sub_sektor) &
            (df_level_2['Provinsi Usaha'] == selected_provinsi)
        ]

        # 3. Cari Data Level 1 (Umum: Sektor, Prov)
        ref_l1 = df_level_1[
            (df_level_1['Sektor Ekonomi'] == selected_sektor) &
            (df_level_1['Provinsi Usaha'] == selected_provinsi)
        ]

        # --- FUNGSI PEMBANTU UNTUK MENAMPILKAN STATUS ---
        def display_status(label, input_val, ref_df, col_max):
            if ref_df.empty:
                return "Data Tidak Tersedia", "grey", 0
            
            max_val = ref_df.iloc[0][col_max]
            
            # Logika Warna dan Status
            if input_val <= max_val:
                return "WAJAR (Pass)", "green", max_val
            else:
                return "TIDAK WAJAR (Over)", "red", max_val

        # --- TAMPILAN TABEL HASIL ---
        
        # Container untuk setiap Level
        cols = st.columns(3)
        
        # --- LEVEL 3 ---
        with cols[0]:
            st.markdown("### 🏙️ Level 3\n(Spesifik Kota/Kab)")
            if ref_l3.empty:
                st.warning("Data referensi Level 3 tidak ditemukan untuk kombinasi ini.")
            else:
                for metric, val, col_name in [
                    ("Omset", input_omset, 'OMZET_MAX_WAJAR'),
                    ("HPP", input_hpp, 'HPP_MAX_WAJAR'),
                    ("Laba", input_laba, 'LABA_MAX_WAJAR'),
                    ("Plafond", input_plafond, 'PLAFOND_MAX_WAJAR')
                ]:
                    status, color, max_limit = display_status(metric, val, ref_l3, col_name)
                    st.markdown(f"**{metric}**")
                    st.markdown(f"Input: {:,.0f}".format(val))
                    st.markdown(f"Max: {:,.0f}".format(max_limit))
                    st.markdown(f":{color}[{status}]")
                    st.divider()

        # --- LEVEL 2 ---
        with cols[1]:
            st.markdown("### 🗺️ Level 2\n(Provinsi & Sub Sektor)")
            if ref_l2.empty:
                st.warning("Data referensi Level 2 tidak ditemukan.")
            else:
                for metric, val, col_name in [
                    ("Omset", input_omset, 'OMZET_MAX_WAJAR'),
                    ("HPP", input_hpp, 'HPP_MAX_WAJAR'),
                    ("Laba", input_laba, 'LABA_MAX_WAJAR'),
                    ("Plafond", input_plafond, 'PLAFOND_MAX_WAJAR')
                ]:
                    status, color, max_limit = display_status(metric, val, ref_l2, col_name)
                    st.markdown(f"**{metric}**")
                    st.markdown(f"Input: {:,.0f}".format(val))
                    st.markdown(f"Max: {:,.0f}".format(max_limit))
                    st.markdown(f":{color}[{status}]")
                    st.divider()

        # --- LEVEL 1 ---
        with cols[2]:
            st.markdown("### 🏢 Level 1\n(Provinsi & Sektor Umum)")
            if ref_l1.empty:
                st.warning("Data referensi Level 1 tidak ditemukan.")
            else:
                for metric, val, col_name in [
                    ("Omset", input_omset, 'OMZET_MAX_WAJAR'),
                    ("HPP", input_hpp, 'HPP_MAX_WAJAR'),
                    ("Laba", input_laba, 'LABA_MAX_WAJAR'),
                    ("Plafond", input_plafond, 'PLAFOND_MAX_WAJAR')
                ]:
                    status, color, max_limit = display_status(metric, val, ref_l1, col_name)
                    st.markdown(f"**{metric}**")
                    st.markdown(f"Input: {:,.0f}".format(val))
                    st.markdown(f"Max: {:,.0f}".format(max_limit))
                    st.markdown(f":{color}[{status}]")
                    st.divider()

    else:
        st.info("Silakan pilih parameter di sidebar dan tekan 'Cek Kewajaran' untuk melihat hasil.")
