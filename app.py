import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from supabase import create_client, Client
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.signal import find_peaks
import os

# --------------------- Configuração da página ---------------------
st.set_page_config(page_title="📊 Materials Database", layout="wide")
st.title("Materials Database")

# --------------------- Conexão Supabase ---------------------
supabase_url = st.secrets["SUPABASE_URL"]
supabase_key = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(supabase_url, supabase_key)

# --------------------- Função para obter colunas válidas ---------------------
def get_table_columns(table_name):
    try:
        data = supabase.table(table_name).select("*").limit(1).execute().data
        if data:
            return list(data[0].keys())
        else:
            # Se a tabela estiver vazia, usar nomes fixos conhecidos
            predefined = {
                "samples": ["sample_name", "description"],
                "raman_spectra": ["measurement_id", "wavenumber_cm1", "intensity_a"],
                "four_point_probe_points": ["measurement_id", "current_a", "voltage_v"],
                "tensiometry_points": ["measurement_id", "t_seconds", "surface_tension_mn_m"],
                "contact_angle_points": ["measurement_id", "t_seconds", "angle_mean_deg"]
            }
            return predefined.get(table_name, [])
    except Exception as e:
        st.error(f"Erro ao obter colunas da tabela {table_name}: {e}")
        return []

# --------------------- Carregamento de amostras ---------------------
@st.cache_data(ttl=300)
def load_samples():
    try:
        data = supabase.table("samples").select("*").execute().data
        return pd.DataFrame(data) if data else pd.DataFrame()
    except Exception as e:
        st.error(f"Erro ao carregar amostras: {e}")
        return pd.DataFrame()

df_samples = load_samples()

# --------------------- Abas ---------------------
abas = st.tabs(["1 Amostras", "2 Ensaios", "3 Otimização"])

# --------------------- Aba 1: Amostras ---------------------
with abas[0]:
    st.header("1 Gerenciamento de Amostras")
    
    # Upload CSV
    st.subheader("📥 Importar nova amostra")
    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")
    
    if uploaded_file is not None:
        try:
            new_sample_df = pd.read_csv(uploaded_file)
            st.write("Pré-visualização do CSV:")
            st.dataframe(new_sample_df.head())

            if st.button("Cadastrar amostras no banco"):
                valid_columns = get_table_columns("samples")
                for _, row in new_sample_df.iterrows():
                    filtered_row = {k: v for k, v in row.to_dict().items() if k in valid_columns}
                    try:
                        supabase.table("samples").insert(filtered_row).execute()
                    except Exception as e:
                        st.error(f"Erro ao cadastrar amostra: {e}")
                st.success("✅ Amostras cadastradas com sucesso!")
                df_samples = load_samples()  # atualizar tabela
        except Exception as e:
            st.error(f"Erro ao ler CSV: {e}")

    # Tabela de amostras existentes
    st.subheader("Amostras cadastradas")
    if df_samples.empty:
        st.info("Nenhuma amostra cadastrada ainda.")
    else:
        st.dataframe(df_samples)

# --------------------- Aba 2: Ensaios ---------------------
with abas[1]:
    st.header("2 Ensaios por Amostra")
    if df_samples.empty:
        st.warning("Cadastre amostras primeiro.")
    else:
        sample_choice = st.selectbox("Escolha a amostra", df_samples["id"])
        tipo = st.radio("Tipo de experimento", ["Raman", "4 Pontas", "Tensiometria", "Ângulo de Contato"])

        try:
            measurements = supabase.table("measurements").select("*").eq("sample_id", sample_choice).execute().data
        except Exception as e:
            st.error(f"Erro ao buscar measurements: {e}")
            measurements = []

        measurement_ids = [m["id"] for m in measurements] if measurements else []

        data = []
        for mid in measurement_ids:
            try:
                table_map = {
                    "Raman": "raman_spectra",
                    "4 Pontas": "four_point_probe_points",
                    "Tensiometria": "tensiometry_points",
                    "Ângulo de Contato": "contact_angle_points"
                }
                table_name = table_map.get(tipo)
                if table_name:
                    response = supabase.table(table_name).select("*").eq("measurement_id", mid).execute()
                    if response and response.data:
                        data.extend(response.data)
            except Exception as e:
                st.error(f"Erro ao buscar dados do measurement_id {mid}: {e}")

        df = pd.DataFrame(data) if data else pd.DataFrame()
        if not df.empty:
            st.write(df.head())
            fig, ax = plt.subplots()
            if tipo == "Raman":
                ax.plot(df["wavenumber_cm1"], df["intensity_a"])
            elif tipo == "4 Pontas":
                ax.plot(df["current_a"], df["voltage_v"], 'o-')
            elif tipo == "Tensiometria":
                ax.plot(df["t_seconds"], df["surface_tension_mn_m"])
            elif tipo == "Ângulo de Contato":
                ax.plot(df["t_seconds"], df["angle_mean_deg"], "ro-")
            st.pyplot(fig)
        else:
            st.warning(f"Nenhum dado de {tipo} encontrado.")

# --------------------- Aba 3: Otimização ---------------------
with abas[2]:
    st.header("3 Otimização Automática de Ensaios")
    if df_samples.empty:
        st.warning("Cadastre amostras primeiro.")
    else:
        sample_choice = st.selectbox("Selecione a amostra para otimizar", df_samples["id"], key="opt")
        tipo = st.radio("Escolha o experimento para otimizar", ["Raman", "4 Pontas", "Tensiometria"], key="opt_tipo")

        try:
            measurements = supabase.table("measurements").select("*").eq("sample_id", sample_choice).execute().data
        except Exception as e:
            st.error(f"Erro ao buscar measurements: {e}")
            measurements = []

        measurement_ids = [m["id"] for m in measurements] if measurements else []

        data = []
        for mid in measurement_ids:
            try:
                table_map = {
                    "Raman": "raman_spectra",
                    "4 Pontas": "four_point_probe_points",
                    "Tensiometria": "tensiometry_points"
                }
                table_name = table_map.get(tipo)
                if table_name:
                    response = supabase.table(table_name).select("*").eq("measurement_id", mid).execute()
                    if response and response.data:
                        data.extend(response.data)
            except Exception as e:
                st.error(f"Erro ao buscar dados do measurement_id {mid}: {e}")

        df = pd.DataFrame(data) if data else pd.DataFrame()
        
        if not df.empty:
            if tipo == "Raman":
                peaks, _ = find_peaks(df["intensity_a"], height=np.mean(df["intensity_a"]))
                peak_positions = df["wavenumber_cm1"].iloc[peaks].tolist()
                st.write("Picos detectados:", peak_positions)
                fig, ax = plt.subplots()
                ax.plot(df["wavenumber_cm1"], df["intensity_a"], label="Espectro Raman")
                ax.plot(df["wavenumber_cm1"].iloc[peaks], df["intensity_a"].iloc[peaks], "ro")
                st.pyplot(fig)
            elif tipo == "4 Pontas":
                X = df[["current_a"]].values
                y = df["voltage_v"].values
                model = LinearRegression().fit(X, y)
                fig, ax = plt.subplots()
                ax.scatter(df["current_a"], df["voltage_v"], label="Dados")
                ax.plot(df["current_a"], model.predict(X), 'r-', label="Ajuste Linear")
                st.pyplot(fig)
            elif tipo == "Tensiometria":
                coef = np.polyfit(df["t_seconds"], df["surface_tension_mn_m"], 3)
                poly = np.poly1d(coef)
                fig, ax = plt.subplots()
                ax.plot(df["t_seconds"], df["surface_tension_mn_m"], 'bo')
                ax.plot(df["t_seconds"], poly(df["t_seconds"]), 'r-')
                st.pyplot(fig)
        else:
            st.warning(f"Nenhum dado de {tipo} encontrado.")
