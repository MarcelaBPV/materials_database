import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from supabase import create_client, Client
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from scipy.signal import find_peaks

# --------------------- Configuração da página ---------------------
st.set_page_config(page_title="📊 Materials Database", layout="wide")
st.title("Materials Database")

# --------------------- Conexão Supabase ---------------------
supabase_url = st.secrets["SUPABASE_URL"]
supabase_key = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(supabase_url, supabase_key)

# --------------------- Funções de otimização ---------------------
def optimize_raman(df, sample_id):
    """Detecta picos e aplica PCA no espectro Raman"""
    try:
        # Detectar picos
        peaks, _ = find_peaks(df["intensity_a"], height=0)
        num_peaks = len(peaks)

        # PCA
        X = df[["wavenumber_cm1", "intensity_a"]].dropna()
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        explained = pca.explained_variance_ratio_.sum()

        # Salvar resultado
        result = {
            "num_peaks": int(num_peaks),
            "peak_positions": df["wavenumber_cm1"].iloc[peaks].tolist(),
            "explained_variance": float(explained),
        }
        supabase.table("resultadosotimizacao").insert({
            "sample_id": int(sample_id),
            "tipo": "raman",
            "metricas": result
        }).execute()

        return result
    except Exception as e:
        st.error(f"Erro na otimização Raman: {e}")
        return {}

def optimize_four_point(df, sample_id):
    """Ajusta regressão linear da curva I-V"""
    try:
        X = df[["current"]].values
        y = df["voltage"].values
        model = LinearRegression().fit(X, y)
        resistencia = model.coef_[0]

        result = {
            "resistencia_linear": float(resistencia),
            "intercepto": float(model.intercept_)
        }
        supabase.table("resultadosotimizacao").insert({
            "sample_id": int(sample_id),
            "tipo": "four_pontas",
            "metricas": result
        }).execute()

        return result
    except Exception as e:
        st.error(f"Erro na otimização 4 Pontas: {e}")
        return {}

def optimize_tensiometry(df, sample_id):
    """Calcula média e ajusta polinômio"""
    try:
        media_forca = df["forca"].mean()
        coef = np.polyfit(df["tempo"], df["forca"], 3)

        result = {
            "media_forca": float(media_forca),
            "coef_poly3": coef.tolist()
        }
        supabase.table("resultadosotimizacao").insert({
            "sample_id": int(sample_id),
            "tipo": "tensiometria",
            "metricas": result
        }).execute()

        return result
    except Exception as e:
        st.error(f"Erro na otimização Tensiometria: {e}")
        return {}

# --------------------- Carregamento de dados ---------------------
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
        tipo = st.radio("Tipo de experimento", ["Raman", "4 Pontas", "Tensiometria"])

        if tipo == "Raman":
            data = supabase.table("raman_spectra").select("*").eq("sample_id", sample_choice).execute().data
            df = pd.DataFrame(data) if data else pd.DataFrame()
            if not df.empty:
                st.write(df.head())
                fig, ax = plt.subplots()
                ax.plot(df["wavenumber_cm1"], df["intensity_a"])
                st.pyplot(fig)
            else:
                st.warning("Nenhum dado Raman encontrado.")

        elif tipo == "4 Pontas":
            data = supabase.table("four_point_probe_points").select("*").eq("sample_id", sample_choice).execute().data
            df = pd.DataFrame(data) if data else pd.DataFrame()
            if not df.empty:
                st.write(df.head())
                fig, ax = plt.subplots()
                ax.plot(df["current"], df["voltage"], 'o-')
                st.pyplot(fig)
            else:
                st.warning("Nenhum dado 4 Pontas encontrado.")

        elif tipo == "Tensiometria":
            data = supabase.table("tensiometry_points").select("*").eq("sample_id", sample_choice).execute().data
            df = pd.DataFrame(data) if data else pd.DataFrame()
            if not df.empty:
                st.write(df.head())
                fig, ax = plt.subplots()
                ax.plot(df["tempo"], df["forca"])
                st.pyplot(fig)
            else:
                st.warning("Nenhum dado de Tensiometria encontrado.")

# --------------------- Aba 3: Otimização ---------------------
with abas[2]:
    st.header("3 Otimização Automática de Ensaios")
    if df_samples.empty:
        st.warning("Cadastre amostras primeiro.")
    else:
        sample_choice = st.selectbox("Selecione a amostra para otimizar", df_samples["id"], key="opt")
        tipo = st.radio("Escolha o experimento para otimizar", ["Raman", "4 Pontas", "Tensiometria"], key="opt_tipo")

        if st.button("Rodar Otimização"):
            if tipo == "Raman":
                data = supabase.table("raman_spectra").select("*").eq("sample_id", sample_choice).execute().data
                df = pd.DataFrame(data)
                if not df.empty:
                    res = optimize_raman(df, sample_choice)
                    st.success("Otimização Raman concluída e salva!")
                    st.json(res)

            elif tipo == "4 Pontas":
                data = supabase.table("four_point_probe_points").select("*").eq("sample_id", sample_choice).execute().data
                df = pd.DataFrame(data)
                if not df.empty:
                    res = optimize_four_point(df, sample_choice)
                    st.success("Otimização 4 Pontas concluída e salva!")
                    st.json(res)

            elif tipo == "Tensiometria":
                data = supabase.table("tensiometry_points").select("*").eq("sample_id", sample_choice).execute().data
                df = pd.DataFrame(data)
                if not df.empty:
                    res = optimize_tensiometry(df, sample_choice)
                    st.success("Otimização Tensiometria concluída e salva!")
                    st.json(res)
