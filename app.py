import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from supabase import create_client, Client
from sklearn.decomposition import PCA
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
    peaks, _ = find_peaks(df["intensity_a"], height=0)
    num_peaks = len(peaks)
    X = df[["wavenumber_cm1", "intensity_a"]].dropna()
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_.sum()
    
    result = {
        "num_peaks": int(num_peaks),
        "peak_positions": df["wavenumber_cm1"].iloc[peaks].tolist(),
        "explained_variance": float(explained)
    }

    # Salvar otimização no Supabase
    supabase.table("resultadosotimizacao").insert({
        "id_ensaio": None,  # Se tiver id_ensaio, colocar aqui
        "parametros_otimizados": result
    }).execute()

    # Plot
    fig, ax = plt.subplots()
    ax.plot(df["wavenumber_cm1"], df["intensity_a"], label="Espectro Raman")
    ax.plot(df["wavenumber_cm1"].iloc[peaks], df["intensity_a"].iloc[peaks], "ro", label="Picos detectados")
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Intensidade (a.u.)")
    ax.legend()
    st.pyplot(fig)

    return result

def optimize_four_point(df, sample_id):
    X = df[["current_a"]].values
    y = df["voltage_v"].values
    model = LinearRegression().fit(X, y)
    resistencia = model.coef_[0]

    result = {
        "resistencia_linear": float(resistencia),
        "intercepto": float(model.intercept_)
    }
    supabase.table("resultadosotimizacao").insert({
        "id_ensaio": None,
        "parametros_otimizados": result
    }).execute()

    fig, ax = plt.subplots()
    ax.scatter(df["current_a"], df["voltage_v"], label="Dados experimentais")
    ax.plot(df["current_a"], model.predict(X), "r-", label=f"Ajuste Linear (R={resistencia:.2f})")
    ax.set_xlabel("Corrente (A)")
    ax.set_ylabel("Tensão (V)")
    ax.legend()
    st.pyplot(fig)

    return result

def optimize_tensiometry(df, sample_id):
    media_forca = df["surface_tension_mn_m"].mean()
    coef = np.polyfit(df["t_seconds"], df["surface_tension_mn_m"], 3)
    poly = np.poly1d(coef)

    result = {
        "media_forca": float(media_forca),
        "coef_poly3": coef.tolist()
    }
    supabase.table("resultadosotimizacao").insert({
        "id_ensaio": None,
        "parametros_otimizados": result
    }).execute()

    fig, ax = plt.subplots()
    ax.plot(df["t_seconds"], df["surface_tension_mn_m"], "bo", label="Dados experimentais")
    ax.plot(df["t_seconds"], poly(df["t_seconds"]), "r-", label="Ajuste Polinomial (grau 3)")
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Força (N)")
    ax.legend()
    st.pyplot(fig)

    return result

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
        tipo = st.radio("Tipo de experimento", ["Raman", "4 Pontas", "Tensiometria", "Ângulo de Contato"])

        # Pega todos os measurements da amostra
        measurements = supabase.table("measurements").select("*").eq("sample_id", sample_choice).execute().data
        measurement_ids = [m["id"] for m in measurements] if measurements else []

        data = []
        if tipo == "Raman":
            for mid in measurement_ids:
                pts = supabase.table("raman_spectra").select("*").eq("measurement_id", mid).execute().data
                if pts: data.extend(pts)
            df = pd.DataFrame(data)
            if not df.empty:
                st.write(df.head())
                fig, ax = plt.subplots()
                ax.plot(df["wavenumber_cm1"], df["intensity_a"])
                st.pyplot(fig)
            else:
                st.warning("Nenhum dado Raman encontrado.")

        elif tipo == "4 Pontas":
            for mid in measurement_ids:
                pts = supabase.table("four_point_probe_points").select("*").eq("measurement_id", mid).execute().data
                if pts: data.extend(pts)
            df = pd.DataFrame(data)
            if not df.empty:
                st.write(df.head())
                fig, ax = plt.subplots()
                ax.plot(df["current_a"], df["voltage_v"], 'o-')
                st.pyplot(fig)
            else:
                st.warning("Nenhum dado 4 Pontas encontrado.")

        elif tipo == "Tensiometria":
            for mid in measurement_ids:
                pts = supabase.table("tensiometry_points").select("*").eq("measurement_id", mid).execute().data
                if pts: data.extend(pts)
            df = pd.DataFrame(data)
            if not df.empty:
                st.write(df.head())
                fig, ax = plt.subplots()
                ax.plot(df["t_seconds"], df["surface_tension_mn_m"])
                st.pyplot(fig)
            else:
                st.warning("Nenhum dado de Tensiometria encontrado.")

        elif tipo == "Ângulo de Contato":
            for mid in measurement_ids:
                pts = supabase.table("contact_angle_points").select("*").eq("measurement_id", mid).execute().data
                if pts: data.extend(pts)
            df = pd.DataFrame(data)
            if not df.empty:
                st.write(df.head())
                fig, ax = plt.subplots()
                ax.plot(df["t_seconds"], df["angle_mean_deg"], "ro-")
                st.pyplot(fig)
            else:
                st.warning("Nenhum dado de Ângulo de Contato encontrado.")

# --------------------- Aba 3: Otimização ---------------------
with abas[2]:
    st.header("3 Otimização Automática de Ensaios")
    if df_samples.empty:
        st.warning("Cadastre amostras primeiro.")
    else:
        sample_choice = st.selectbox("Selecione a amostra para otimizar", df_samples["id"], key="opt")
        tipo = st.radio("Escolha o experimento para otimizar", ["Raman", "4 Pontas", "Tensiometria"], key="opt_tipo")

        # Pega measurements da amostra
        measurements = supabase.table("measurements").select("*").eq("sample_id", sample_choice).execute().data
        measurement_ids = [m["id"] for m in measurements] if measurements else []

        data = []
        for mid in measurement_ids:
            if tipo == "Raman":
                pts = supabase.table("raman_spectra").select("*").eq("measurement_id", mid).execute().data
            elif tipo == "4 Pontas":
                pts = supabase.table("four_point_probe_points").select("*").eq("measurement_id", mid).execute().data
            elif tipo == "Tensiometria":
                pts = supabase.table("tensiometry_points").select("*").eq("measurement_id", mid).execute().data
            else:
                pts = []
            if pts: data.extend(pts)

        if data:
            df = pd.DataFrame(data)
            if tipo == "Raman":
                res = optimize_raman(df, sample_choice)
                st.success("✅ Otimização Raman concluída e salva!")
                st.json(res)
            elif tipo == "4 Pontas":
                res = optimize_four_point(df, sample_choice)
                st.success("✅ Otimização 4 Pontas concluída e salva!")
                st.json(res)
            elif tipo == "Tensiometria":
                res = optimize_tensiometry(df, sample_choice)
                st.success("✅ Otimização Tensiometria concluída e salva!")
                st.json(res)
        else:
            st.warning(f"Nenhum dado de {tipo} encontrado para otimização.")
