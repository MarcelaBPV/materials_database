import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from supabase import create_client, Client
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.signal import find_peaks

# ============================
# Configuração Supabase
# ============================
url = st.secrets["SUPABASE_URL"]
key = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(url, key)

st.set_page_config(page_title="📊 Materials Database", layout="wide")
st.title("📊 Banco de Dados de Ensaios de Materiais")

# ============================
# Seleção de amostra
# ============================
samples = supabase.table("samples").select("id, sample_name").execute().data
if not samples:
    st.error("Nenhuma amostra encontrada na tabela 'samples'.")
    st.stop()

sample_options = {s["sample_name"]: s["id"] for s in samples}
sample_choice = st.selectbox("Escolha a amostra:", list(sample_options.keys()))
sample_id = sample_options[sample_choice]
st.success(f"Amostra selecionada: {sample_choice} (ID {sample_id})")

# ============================
# Seleção de experimento
# ============================
experiment_type = st.radio(
    "Selecione o tipo de experimento:",
    ["Raman", "4 Pontas", "Tensiometria", "Ângulo de Contato"]
)

# ============================
# Função para carregar dados
# ============================
def get_data(table_name: str, sample_id: int):
    response = supabase.table(table_name).select("*").eq("sample_id", sample_id).execute()
    return response.data

# ============================
# Funções de otimização
# ============================
def optimize_raman(df, sample_id):
    peaks, _ = find_peaks(df["intensity"], height=0)
    X = df[["wavenumber", "intensity"]].dropna()
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_.sum()
    
    result = {
        "num_peaks": int(len(peaks)),
        "peak_positions": df["wavenumber"].iloc[peaks].tolist(),
        "explained_variance": float(explained)
    }
    supabase.table("resultadosotimizacao").insert({
        "sample_id": int(sample_id),
        "tipo": "raman",
        "metricas": result
    }).execute()

    fig, ax = plt.subplots()
    ax.plot(df["wavenumber"], df["intensity"], label="Espectro Raman")
    ax.plot(df["wavenumber"].iloc[peaks], df["intensity"].iloc[peaks], "ro", label="Picos detectados")
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Intensidade (a.u.)")
    ax.legend()
    st.pyplot(fig)
    
    return result

def optimize_four_point(df, sample_id):
    X = df[["I_mA"]].values
    y = df["V_mV"].values
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

    fig, ax = plt.subplots()
    ax.scatter(df["I_mA"], df["V_mV"], label="Dados experimentais")
    ax.plot(df["I_mA"], model.predict(X), "r-", label=f"Ajuste Linear (R={resistencia:.2f})")
    ax.set_xlabel("Corrente (mA)")
    ax.set_ylabel("Tensão (mV)")
    ax.legend()
    st.pyplot(fig)
    
    return result

def optimize_tensiometry(df, sample_id):
    media_forca = df["force_N"].mean()
    coef = np.polyfit(df["time_s"], df["force_N"], 3)
    poly = np.poly1d(coef)
    
    result = {
        "media_forca": float(media_forca),
        "coef_poly3": coef.tolist()
    }
    supabase.table("resultadosotimizacao").insert({
        "sample_id": int(sample_id),
        "tipo": "tensiometria",
        "metricas": result
    }).execute()

    fig, ax = plt.subplots()
    ax.plot(df["time_s"], df["force_N"], "bo", label="Dados experimentais")
    ax.plot(df["time_s"], poly(df["time_s"]), "r-", label="Ajuste Polinomial (grau 3)")
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Força (N)")
    ax.legend()
    st.pyplot(fig)
    
    return result

# ============================
# Carregar e exibir dados + otimização
# ============================
if experiment_type == "Raman":
    st.subheader("📈 Espectroscopia Raman")
    data = get_data("raman_spectra", sample_id)

    if data:
        df = pd.DataFrame(data)
        st.dataframe(df)
        fig, ax = plt.subplots()
        ax.plot(df["wavenumber"], df["intensity"])
        ax.set_xlabel("Wavenumber (cm⁻¹)")
        ax.set_ylabel("Intensidade (a.u.)")
        st.pyplot(fig)

        if st.button("Rodar Otimização Raman"):
            res = optimize_raman(df, sample_id)
            st.success("✅ Otimização concluída!")
            st.json(res)
    else:
        st.warning("Nenhum dado Raman encontrado.")

elif experiment_type == "4 Pontas":
    st.subheader("🔌 Medida de Resistividade (4 Pontas)")
    data = get_data("four_point_probe_points", sample_id)

    if data:
        df = pd.DataFrame(data)
        st.dataframe(df)
        fig, ax = plt.subplots()
        ax.plot(df["I_mA"], df["V_mV"], 'o-')
        ax.set_xlabel("Corrente (mA)")
        ax.set_ylabel("Tensão (mV)")
        st.pyplot(fig)

        if st.button("Rodar Otimização 4 Pontas"):
            res = optimize_four_point(df, sample_id)
            st.success("✅ Otimização concluída!")
            st.json(res)
    else:
        st.warning("Nenhum dado 4 Pontas encontrado.")

elif experiment_type == "Tensiometria":
    st.subheader("💧 Tensiometria")
    data = get_data("tensiometry_points", sample_id)

    if data:
        df = pd.DataFrame(data)
        st.dataframe(df)
        fig, ax = plt.subplots()
        ax.plot(df["time_s"], df["force_N"])
        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel("Força (N)")
        st.pyplot(fig)

        if st.button("Rodar Otimização Tensiometria"):
            res = optimize_tensiometry(df, sample_id)
            st.success("✅ Otimização concluída!")
            st.json(res)
    else:
        st.warning("Nenhum dado de Tensiometria encontrado.")

elif experiment_type == "Ângulo de Contato":
    st.subheader("📐 Ângulo de Contato")
    data = get_data("contact_angle_points", sample_id)

    if data:
        df = pd.DataFrame(data)
        st.dataframe(df)
        fig, ax = plt.subplots()
        ax.plot(df["tempo_s"], df["angulo_deg"], "ro-")
        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel("Ângulo (°)")
        st.pyplot(fig)
    else:
        st.warning("Nenhum dado de Ângulo de Contato encontrado.")
