import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from supabase import create_client, Client
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import numpy as np

# --------------------- Conexão segura com Supabase ---------------------
import streamlit as st
from supabase import create_client

supabase_url = st.secrets["SUPABASE_URL"]
supabase_key = st.secrets["SUPABASE_KEY"]

supabase = create_client(supabase_url, supabase_key)

samples = supabase.table("samples").select("*").execute().data
st.write(samples)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --------------------- Layout ---------------------
st.set_page_config(page_title="📊 Materials Database", layout="wide")
st.title("Materials Database")

# --------------------- Abas ---------------------
abas = st.tabs(["1 Amostras", "2 Ensaios", "3 Resultados", "4 Otimização"])

# --------------------- Aba 1: Amostras ---------------------
with abas[0]:
    st.header("1 Gerenciamento de Amostras")

    try:
        samples = supabase.table("samples").select("*").execute().data
        if samples:
            df_samples = pd.DataFrame(samples)
            st.dataframe(df_samples)
        else:
            st.info("Nenhuma amostra cadastrada ainda.")
    except Exception as e:
        st.error(f"Erro ao carregar amostras: {e}")

# --------------------- Aba 2: Ensaios ---------------------
with abas[1]:
    st.header("2 Ensaios por Amostra")

    # Seleciona amostra
    samples = supabase.table("samples").select("*").execute().data
    if samples:
        df_samples = pd.DataFrame(samples)
        sample_choice = st.selectbox("Escolha a amostra", df_samples["id"])
        
        # Seleciona tipo de experimento
        tipo = st.radio("Tipo de experimento", ["Raman", "4 Pontas", "Tensiometria"])

        if tipo == "Raman":
            data = supabase.table("raman_spectra").select("*").eq("sample_id", sample_choice).execute().data
            if data:
                df = pd.DataFrame(data)
                st.write(df.head())
                fig, ax = plt.subplots()
                ax.plot(df["shift"], df["intensity"])
                ax.set_xlabel("Raman Shift (cm⁻¹)")
                ax.set_ylabel("Intensity (a.u.)")
                ax.set_title("Raman Spectrum")
                st.pyplot(fig)
            else:
                st.warning("Nenhum dado Raman encontrado.")

        elif tipo == "4 Pontas":
            data = supabase.table("four_point_probe_points").select("*").eq("sample_id", sample_choice).execute().data
            if data:
                df = pd.DataFrame(data)
                st.write(df.head())
                fig, ax = plt.subplots()
                ax.plot(df["corrente"], df["tensao"], 'o-')
                ax.set_xlabel("Corrente (A)")
                ax.set_ylabel("Tensão (V)")
                ax.set_title("Curva 4 Pontas")
                st.pyplot(fig)
            else:
                st.warning("Nenhum dado 4 Pontas encontrado.")

        elif tipo == "Tensiometria":
            data = supabase.table("tensiometry_points").select("*").eq("sample_id", sample_choice).execute().data
            if data:
                df = pd.DataFrame(data)
                st.write(df.head())
                fig, ax = plt.subplots()
                ax.plot(df["tempo"], df["forca"])
                ax.set_xlabel("Tempo (s)")
                ax.set_ylabel("Força (mN)")
                ax.set_title("Curva de Tensiometria")
                st.pyplot(fig)
            else:
                st.warning("Nenhum dado de Tensiometria encontrado.")
    else:
        st.warning("Cadastre amostras primeiro.")

# --------------------- Aba 3: Resultados ---------------------
with abas[2]:
    st.header("3 Resultados de Caracterização")
    try:
        resultados = supabase.table("resultadoscaracterizacao").select("*").execute().data
        if resultados:
            df_res = pd.DataFrame(resultados)
            st.dataframe(df_res)
        else:
            st.info("Nenhum resultado disponível.")
    except Exception as e:
        st.error(f"Erro ao carregar resultados: {e}")

# --------------------- Aba 4: Otimização ---------------------
with abas[3]:
    st.header("4 Otimização de Dados (Machine Learning)")

    try:
        data = supabase.table("resultadoscaracterizacao").select("*").execute().data
        if data:
            df = pd.DataFrame(data).dropna()
            st.write("Dados carregados:", df.head())

            if len(df) > 2:
                # PCA
                st.subheader("Redução de Dimensionalidade (PCA)")
                X = df.select_dtypes(include=[np.number])
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X)
                fig, ax = plt.subplots()
                ax.scatter(X_pca[:, 0], X_pca[:, 1])
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                st.pyplot(fig)

                # Clustering
                st.subheader("Agrupamento (KMeans)")
                kmeans = KMeans(n_clusters=2, n_init=10).fit(X)
                df["cluster"] = kmeans.labels_
                st.dataframe(df)

                # Regressão Linear (exemplo simples)
                if X.shape[1] >= 2:
                    st.subheader("Regressão Linear")
                    X_lin = X.iloc[:, :-1]
                    y_lin = X.iloc[:, -1]
                    model = LinearRegression().fit(X_lin, y_lin)
                    st.write("Coeficientes:", model.coef_)
                    st.write("Intercepto:", model.intercept_)
            else:
                st.warning("Poucos dados para aplicar ML.")
        else:
            st.info("Nenhum dado disponível para otimização.")
    except Exception as e:
        st.error(f"Erro no módulo de otimização: {e}")
