import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from supabase import create_client, Client
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# --------------------- Configuração da página ---------------------
st.set_page_config(page_title="📊 Materials Database", layout="wide")
st.title("Materials Database")

# --------------------- Conexão Supabase ---------------------
supabase_url = st.secrets["SUPABASE_URL"]
supabase_key = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(supabase_url, supabase_key)

# --------------------- Carregamento de dados ---------------------
@st.cache_data(ttl=300)
def load_samples():
    try:
        data = supabase.table("samples").select("*").execute().data
        return pd.DataFrame(data) if data else pd.DataFrame()
    except Exception as e:
        st.error(f"Erro ao carregar amostras: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_results():
    try:
        data = supabase.table("resultadoscaracterizacao").select("*").execute().data
        return pd.DataFrame(data) if data else pd.DataFrame()
    except Exception as e:
        st.error(f"Erro ao carregar resultados: {e}")
        return pd.DataFrame()

df_samples = load_samples()
df_results = load_results()

# --------------------- Abas ---------------------
abas = st.tabs(["1 Amostras", "2 Ensaios", "3 Resultados", "4 Otimização"])

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
            if df.empty:
                st.warning("Nenhum dado Raman encontrado.")
            elif all(col in df.columns for col in ["shift", "intensity"]):
                st.write(df.head())
                fig, ax = plt.subplots()
                ax.plot(df["shift"], df["intensity"])
                ax.set_xlabel("Raman Shift (cm⁻¹)")
                ax.set_ylabel("Intensity (a.u.)")
                ax.set_title("Raman Spectrum")
                st.pyplot(fig)
            else:
                st.warning("Colunas esperadas não encontradas.")

        elif tipo == "4 Pontas":
            data = supabase.table("four_point_probe_points").select("*").eq("sample_id", sample_choice).execute().data
            df = pd.DataFrame(data) if data else pd.DataFrame()
            if df.empty:
                st.warning("Nenhum dado 4 Pontas encontrado.")
            elif all(col in df.columns for col in ["corrente", "tensao"]):
                st.write(df.head())
                fig, ax = plt.subplots()
                ax.plot(df["corrente"], df["tensao"], 'o-')
                ax.set_xlabel("Corrente (A)")
                ax.set_ylabel("Tensão (V)")
                ax.set_title("Curva 4 Pontas")
                st.pyplot(fig)
            else:
                st.warning("Colunas esperadas não encontradas.")

        elif tipo == "Tensiometria":
            data = supabase.table("tensiometry_points").select("*").eq("sample_id", sample_choice).execute().data
            df = pd.DataFrame(data) if data else pd.DataFrame()
            if df.empty:
                st.warning("Nenhum dado de Tensiometria encontrado.")
            elif all(col in df.columns for col in ["tempo", "forca"]):
                st.write(df.head())
                fig, ax = plt.subplots()
                ax.plot(df["tempo"], df["forca"])
                ax.set_xlabel("Tempo (s)")
                ax.set_ylabel("Força (mN)")
                ax.set_title("Curva de Tensiometria")
                st.pyplot(fig)
            else:
                st.warning("Colunas esperadas não encontradas.")

# --------------------- Aba 3: Resultados ---------------------
with abas[2]:
    st.header("3 Resultados de Caracterização")
    if df_results.empty:
        st.info("Nenhum resultado disponível.")
    else:
        st.dataframe(df_results)

# --------------------- Aba 4: Otimização ---------------------
with abas[3]:
    st.header("4 Otimização de Dados (Machine Learning)")
    if df_results.empty:
        st.info("Nenhum dado disponível para otimização.")
    else:
        df_ml = df_results.select_dtypes(include=[np.number]).dropna()
        st.write("Dados carregados para ML:", df_ml.head())

        if len(df_ml) > 2 and df_ml.shape[1] > 1:
            # PCA
            st.subheader("Redução de Dimensionalidade (PCA)")
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(df_ml)
            fig, ax = plt.subplots()
            ax.scatter(X_pca[:, 0], X_pca[:, 1])
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            st.pyplot(fig)

            # Clustering
            st.subheader("Agrupamento (KMeans)")
            kmeans = KMeans(n_clusters=2, n_init=10).fit(df_ml)
            df_results["cluster"] = kmeans.labels_
            st.dataframe(df_results)

            # Regressão Linear (exemplo simples)
            if df_ml.shape[1] >= 2:
                st.subheader("Regressão Linear")
                X_lin = df_ml.iloc[:, :-1]
                y_lin = df_ml.iloc[:, -1]
                model = LinearRegression().fit(X_lin, y_lin)
                st.write("Coeficientes:", model.coef_)
                st.write("Intercepto:", model.intercept_)
        else:
            st.warning("Poucos dados para aplicar Machine Learning.")
