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
st.title(" 📊 Materials Platform")

# --------------------- Conexão Supabase ---------------------
supabase_url = st.secrets["SUPABASE_URL"]
supabase_key = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(supabase_url, supabase_key)

# --------------------- Carregar tabela de atribuições Raman ---------------------
def load_atribuicoes():
    path = os.path.join("data", "raman_atribuicoes.csv")
    return pd.read_csv(path)

def atribuir_picos(peak_positions, tol=15):
    tabela = load_atribuicoes()
    atribuicoes = []
    for p in peak_positions:
        match = tabela[tabela["Frequência (cm⁻¹)"].between(p - tol, p + tol)]
        if not match.empty:
            for _, row in match.iterrows():
                atribuicoes.append({
                    "Pico (cm⁻¹)": round(p, 1),
                    "Atribuição Molecular": row["Atribuição Molecular"],
                    "Componente Químico": row["Componente Químico"]
                })
        else:
            atribuicoes.append({
                "Pico (cm⁻¹)": round(p, 1),
                "Atribuição Molecular": "Não identificado",
                "Componente Químico": ""
            })
    return pd.DataFrame(atribuicoes)

# --------------------- Funções de otimização ---------------------
def optimize_raman(df, sample_id):
    peaks, _ = find_peaks(df["intensity_a"], height=np.mean(df["intensity_a"]))
    peak_positions = df["wavenumber_cm1"].iloc[peaks].tolist()

    # Atribuições
    atribuicoes = atribuir_picos(peak_positions)

    # PCA
    X = df[["wavenumber_cm1", "intensity_a"]].dropna()
    pca = PCA(n_components=2)
    pca.fit(X)
    explained = pca.explained_variance_ratio_.sum()

    result = {
        "num_peaks": len(peak_positions),
        "atribuições": atribuicoes.to_dict(orient="records"),
        "explained_variance": float(explained)
    }

    try:
        supabase.table("resultadosotimizacao").insert({
            "id_ensaio": None,  # ajustar se tiver id_ensaio
            "parametros_otimizados": result
        }).execute()
    except Exception as e:
        st.error(f"Erro ao salvar otimização Raman: {e}")

    # Gráfico
    fig, ax = plt.subplots()
    ax.plot(df["wavenumber_cm1"], df["intensity_a"], label="Espectro Raman")
    ax.plot(df["wavenumber_cm1"].iloc[peaks], df["intensity_a"].iloc[peaks], "ro", label="Picos detectados")

    for _, row in atribuicoes.iterrows():
        if row["Atribuição Molecular"] != "Não identificado":
            ax.annotate(row["Atribuição Molecular"],
                        (row["Pico (cm⁻¹)"], 
                         df.loc[df["wavenumber_cm1"].sub(row["Pico (cm⁻¹)"]).abs().idxmin(), "intensity_a"]),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=8, rotation=45)

    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Intensidade (a.u.)")
    ax.legend()
    st.pyplot(fig)

    st.subheader("📋 Picos atribuídos")
    st.dataframe(atribuicoes)

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

    try:
        supabase.table("resultadosotimizacao").insert({
            "id_ensaio": None,
            "parametros_otimizados": result
        }).execute()
    except Exception as e:
        st.error(f"Erro ao salvar otimização 4 Pontas: {e}")

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

    try:
        supabase.table("resultadosotimizacao").insert({
            "id_ensaio": None,
            "parametros_otimizados": result
        }).execute()
    except Exception as e:
        st.error(f"Erro ao salvar otimização Tensiometria: {e}")

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

        try:
            measurements = supabase.table("measurements").select("*").eq("sample_id", sample_choice).execute().data
        except Exception as e:
            st.error(f"Erro ao buscar measurements: {e}")
            measurements = []
        measurement_ids = [m["id"] for m in measurements] if measurements else []

        data = []
        for mid in measurement_ids:
            try:
                if tipo == "Raman":
                    response = supabase.table("raman_spectra").select("*").eq("measurement_id", mid).execute()
                elif tipo == "4 Pontas":
                    response = supabase.table("four_point_probe_points").select("*").eq("measurement_id", mid).execute()
                elif tipo == "Tensiometria":
                    response = supabase.table("tensiometry_points").select("*").eq("measurement_id", mid).execute()
                elif tipo == "Ângulo de Contato":
                    response = supabase.table("contact_angle_points").select("*").eq("measurement_id", mid).execute()
                else:
                    response = None
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
                if tipo == "Raman":
                    response = supabase.table("raman_spectra").select("*").eq("measurement_id", mid).execute()
                elif tipo == "4 Pontas":
                    response = supabase.table("four_point_probe_points").select("*").eq("measurement_id", mid).execute()
                elif tipo == "Tensiometria":
                    response = supabase.table("tensiometry_points").select("*").eq("measurement_id", mid).execute()
                else:
                    response = None
                if response and response.data:
                    data.extend(response.data)
            except Exception as e:
                st.error(f"Erro ao buscar dados do measurement_id {mid}: {e}")

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
