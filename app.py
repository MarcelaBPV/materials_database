# -*- coding: utf-8 -*-
"""
Plataforma de Caracterização de Materiais — Versão Final
Compatível com ramanchada2 (MIT), Supabase e IA molecular.
Autor: Marcela Veiga
"""

import os
import hashlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from supabase import create_client, Client
from sklearn.linear_model import LinearRegression

# Módulos internos
from raman_processing import (
    process_raman_pipeline,
    compare_spectra,
    load_raman_dataframe,
    preprocess_spectrum,
)
from molecular_identification import identify_molecular_groups

# --------------------- Configuração ---------------------
st.set_page_config(page_title="📊 Materials Database", layout="wide")
st.title("📊 Plataforma de Caracterização de Superfícies de Materiais")

# --------------------- Sidebar ---------------------
with st.sidebar:
    st.subheader("Status")
    st.caption("Conectado ao ambiente local (Codespaces).")
    if st.button("🧹 Limpar cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cache limpo.")

# --------------------- Supabase ---------------------
def _get_secret(k: str, default=None):
    try:
        return st.secrets[k]
    except Exception:
        return os.getenv(k, default)

try:
    SUPABASE_URL = _get_secret("SUPABASE_URL")
    SUPABASE_KEY = _get_secret("SUPABASE_KEY")
except Exception as e:
    st.warning("⚠️ Supabase não configurado. Você ainda pode usar o app offline.")
    SUPABASE_URL, SUPABASE_KEY = None, None

@st.cache_resource(show_spinner=False)
def get_client(url, key) -> Client:
    if url and key:
        return create_client(url, key)
    return None

supabase: Client = get_client(SUPABASE_URL, SUPABASE_KEY)

# --------------------- Funções utilitárias ---------------------
def hash_df_fast(df: pd.DataFrame) -> str:
    h = hashlib.md5()
    h.update(str(df.shape).encode())
    if len(df):
        h.update(df.head(min(200, len(df))).to_csv(index=False).encode())
    return h.hexdigest()

def df_downsample(df, x, y, max_pts=8000):
    if len(df) <= max_pts:
        return df[[x, y]]
    idx = np.linspace(0, len(df)-1, max_pts).astype(int)
    return df.iloc[idx][[x, y]]

# --------------------- Tabs ---------------------
tabs = st.tabs(["1️⃣ Amostras", "2️⃣ Ensaios", "3️⃣ Otimização"])

# =====================================================
# 1️⃣ AMOSTRAS
# =====================================================
with tabs[0]:
    st.header("1️⃣ Gerenciamento de Amostras")

    if supabase is None:
        st.warning("⚠️ Modo offline — cadastros locais não serão salvos no banco.")
    else:
        # Importar CSV
        up_s = st.file_uploader("📥 Importar amostras via CSV", type="csv")
        if up_s is not None:
            try:
                df_new = pd.read_csv(up_s)
                st.dataframe(df_new.head(), use_container_width=True)
                if st.button("Cadastrar no banco"):
                    data = df_new.to_dict("records")
                    supabase.table("samples").insert(data).execute()
                    st.success(f"✅ {len(data)} amostras enviadas!")
            except Exception as e:
                st.error(f"Erro ao importar: {e}")

        # Cadastrar manual
        st.subheader("➕ Cadastrar manualmente")
        nome = st.text_input("Nome da amostra")
        desc = st.text_area("Descrição")
        if st.button("Salvar amostra"):
            if supabase:
                supabase.table("samples").insert(
                    {"sample_name": nome, "description": desc}
                ).execute()
                st.success("✅ Amostra salva no Supabase!")
            else:
                st.info("💾 Offline: amostra não enviada (Supabase ausente).")

# =====================================================
# 2️⃣ ENSAIOS
# =====================================================
with tabs[1]:
    st.header("2️⃣ Ensaios por Amostra")

    tipo = st.radio(
        "Tipo de experimento",
        ["Raman", "4 Pontas", "Ângulo de Contato", "Perfilometria"],
        horizontal=True,
    )

    uploaded = st.file_uploader(f"📤 Enviar arquivo de {tipo}", type=["csv", "tsv", "txt", "log"])
    if uploaded is not None:
        try:
            df_file = pd.read_csv(uploaded, sep=None, engine="python", comment="#")
            st.write("Prévia dos dados:")
            st.dataframe(df_file.head(), use_container_width=True)

            if tipo == "Raman":
                df_file.columns = ["wavenumber_cm1", "intensity_a"]
                fig, ax = plt.subplots()
                ax.plot(df_file["wavenumber_cm1"], df_file["intensity_a"])
                ax.set_xlabel("Número de onda (cm⁻¹)")
                ax.set_ylabel("Intensidade (u.a.)")
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Erro ao processar arquivo: {e}")

# =====================================================
# 3️⃣ OTIMIZAÇÃO
# =====================================================
with tabs[2]:
    st.header("3️⃣ Otimização Automática")

    tipo = st.radio("Experimento", ["Raman", "4 Pontas", "Ângulo de Contato"], horizontal=True)

    # ---------------- Raman ----------------
    if tipo == "Raman":
        st.subheader("Pipeline Raman — MIT `ramanchada2`")

        uploaded = st.file_uploader("📤 Carregar espectro Raman (CSV)", type=["csv"])
        if uploaded is not None:
            try:
                df_raman = pd.read_csv(uploaded)
                df_spec = load_raman_dataframe(df_raman)

                # Parâmetros do pipeline
                colp1, colp2, colp3 = st.columns(3)
                with colp1:
                    smooth = st.checkbox("Suavizar", True)
                with colp2:
                    baseline = st.checkbox("Remover baseline", True)
                with colp3:
                    prom = st.number_input("Prominência mínima", 0.0, 1.0, 0.05, 0.01)

                if st.button("▶️ Processar Raman"):
                    processed, peaks, fig = process_raman_pipeline(
                        df_spec, smooth=smooth, baseline=baseline, peak_prominence=prom
                    )
                    st.pyplot(fig)
                    st.subheader("📈 Picos detectados")
                    st.dataframe(peaks, use_container_width=True)

                    # Identificação molecular
                    if not peaks.empty:
                        st.subheader("🧠 Identificação Molecular Automática")
                        id_df = identify_molecular_groups(peaks, tolerance=15.0)
                        st.dataframe(id_df, use_container_width=True)

                        # Gráfico de confiança
                        fig2, ax2 = plt.subplots(figsize=(6, 3))
                        ax2.bar(id_df["Pico (cm⁻¹)"], id_df["Confiança (%)"], color="orange")
                        ax2.set_xlabel("Pico Raman (cm⁻¹)")
                        ax2.set_ylabel("Confiança (%)")
                        ax2.set_title("Probabilidade de grupos funcionais")
                        st.pyplot(fig2)

            except Exception as e:
                st.error(f"Erro ao processar Raman: {e}")

    # ---------------- 4 Pontas ----------------
    elif tipo == "4 Pontas":
        st.subheader("Ajuste Linear — Método de Quatro Pontas")
        uploaded = st.file_uploader("📤 Arquivo (corrente/tensão)", type=["csv"])
        if uploaded is not None:
            try:
                df_4p = pd.read_csv(uploaded)
                df_4p = df_4p.rename(columns={"corrente": "current_a", "tensao": "voltage_v"})
                X = df_4p[["current_a"]].values
                y = df_4p["voltage_v"].values
                model = LinearRegression().fit(X, y)

                fig, ax = plt.subplots()
                ax.scatter(X, y, label="Dados")
                ax.plot(X, model.predict(X), label="Ajuste linear")
                ax.legend()
                st.pyplot(fig)
                st.success(f"Resistência estimada: **{model.coef_[0]:.6f} Ω**")
            except Exception as e:
                st.error(f"Erro no ajuste 4 Pontas: {e}")

    # ---------------- Ângulo de Contato ----------------
    elif tipo == "Ângulo de Contato":
        st.subheader("Ajuste Polinomial — Ângulo de Contato")
        uploaded = st.file_uploader("📤 Arquivo de Ângulo de Contato", type=["csv"])
        if uploaded is not None:
            try:
                df_ang = pd.read_csv(uploaded)
                df_ang = df_ang.rename(columns={"Time": "t_seconds", "Mean": "angle_mean_deg"})
                coef = np.polyfit(df_ang["t_seconds"], df_ang["angle_mean_deg"], 3)
                poly = np.poly1d(coef)

                fig, ax = plt.subplots()
                ax.plot(df_ang["t_seconds"], df_ang["angle_mean_deg"], "o", label="Dados")
                ax.plot(df_ang["t_seconds"], poly(df_ang["t_seconds"]), label="Ajuste 3º grau")
                ax.legend()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Erro no ajuste de ângulo de contato: {e}")
