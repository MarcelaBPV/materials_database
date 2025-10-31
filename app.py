# -*- coding: utf-8 -*-
"""
Plataforma de Caracterização de Materiais — Versão final estável
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

# módulos internos
from raman_processing import process_raman_pipeline, load_raman_dataframe
from molecular_identification import identify_molecular_groups

# ---------------- CONFIG ----------------
st.set_page_config(page_title="📊 Caracterização de Materiais", layout="wide")
st.title("📊 Plataforma Inteligente para Caracterização de Materiais")

# ----------- SUPABASE ----------
def _get_secret(k, default=""):
    try: 
        return st.secrets[k]
    except: 
        return os.getenv(k, default)

SUPABASE_URL = _get_secret("SUPABASE_URL")
SUPABASE_KEY = _get_secret("SUPABASE_KEY")

@st.cache_resource
def get_client(url, key):
    if url and key:
        return create_client(url, key)
    return None

supabase = get_client(SUPABASE_URL, SUPABASE_KEY)

# ----------- FUNÇÕES AUXILIARES ----------
def smart_read(uploaded):
    return pd.read_csv(uploaded, sep=None, engine="python", comment="#")

def coluna(df, alternativas):
    for c in alternativas:
        if c in df.columns:
            return c
    return None

# ----------- TABS ----------
tabs = st.tabs(["🧾 Amostras", "🧪 Ensaios", "🤖 IA Raman"])

# =========================================================
# 1️⃣ AMOSTRAS
# =========================================================
with tabs[0]:
    st.header("Cadastro & Importação de Amostras")

    nome = st.text_input("Nome da amostra")
    desc = st.text_area("Descrição")

    if st.button("Salvar Amostra"):
        if supabase:
            supabase.table("samples").insert({"sample_name": nome, "description": desc}).execute()
            st.success("✅ Amostra salva no banco!")
        else:
            st.warning("Sem Supabase configurado — modo local.")

    up = st.file_uploader("📥 Importar CSV de amostras", type="csv")
    if up:
        df = pd.read_csv(up)
        st.dataframe(df.head())
        if st.button("Enviar para banco"):
            supabase.table("samples").insert(df.to_dict("records")).execute()
            st.success("✅ Amostras enviadas")

# =========================================================
# 2️⃣ ENSAIOS
# =========================================================
with tabs[1]:
    st.header("Envio e Processamento de Ensaios")

    tipo = st.radio("Tipo", ["Raman","4 Pontas","Ângulo de Contato","Tensiometria"], horizontal=True)

    uploaded = st.file_uploader(f"📤 Upload para {tipo}", type=["csv","txt","tsv"])
    if not uploaded: st.stop()

    df = smart_read(uploaded)
    st.dataframe(df.head(), use_container_width=True)

    # ---------------- RAMAN ----------------
    if tipo == "Raman":
        df.columns = ["wavenumber_cm1","intensity_a"]
        fig, ax = plt.subplots()
        ax.plot(df.iloc[:,0], df.iloc[:,1])
        ax.set_xlabel("cm⁻¹")
        ax.set_ylabel("Intensidade (u.a.)")
        st.pyplot(fig)

    # ---------------- 4 PONTAS ----------------
    if tipo == "Resistividade":
        c = coluna(df, ["corrente","current","I","current_a"])
        v = coluna(df, ["tensao","voltage","V","voltage_v"])

        if not c or not v:
            st.error("⚠️ Arquivo deve conter colunas de corrente e tensão.")
            st.stop()

        X = df[[c]].values
        y = df[v].values
        model = LinearRegression().fit(X, y)

        fig, ax = plt.subplots()
        ax.scatter(X,y)
        ax.plot(X, model.predict(X))
        ax.set_title("Ajuste linear — 4 Pontas")
        st.pyplot(fig)

        st.success(f"Resistência: **{model.coef_[0]:.6f} Ω**")

    # ---------------- Tensiometria ----------------
    if tipo == "Tensiometria":
        t = coluna(df, ["tempo","time","t","Time","t_seconds"])
        a = coluna(df, ["angulo","angle","Mean","theta","angle_mean_deg"])

        coef = np.polyfit(df[t], df[a], 3)
        poly = np.poly1d(coef)

        fig, ax = plt.subplots()
        ax.scatter(df[t], df[a])
        ax.plot(df[t], poly(df[t]))
        ax.set_title("Evolução do ângulo de contato")
        st.pyplot(fig)

    # ---------------- TENSIOMETRIA ----------------
    if tipo == "Tensiometria":
        f = coluna(df, ["forca","force","F"])
        d = coluna(df, ["diametro","diameter","D"])
        tens = df[f] / df[d]

        df["tensao_surface_mN_m"] = tens * 1000

        st.write("Resultado:")
        st.dataframe(df[["tensao_surface_mN_m"]].head())

        st.success("✅ Tensão superficial calculada")

# =========================================================
# 3️⃣ Otimização
# =========================================================
with tabs[2]:
    st.header("Processamento + IA de Grupos Moleculares (Raman)")

    uploaded = st.file_uploader("📤 Carregar espectro Raman", type=["csv"])
    if not uploaded: st.stop()

    df_r = pd.read_csv(uploaded)
    df_spec = load_raman_dataframe(df_r)

    prom = st.slider("Prominência mínima", 0.01, 1.0, 0.05)

    if st.button("▶️ Processar"):
        processed, peaks, fig = process_raman_pipeline(df_spec, smooth=True, baseline=True, peak_prominence=prom)
        st.pyplot(fig)

        st.subheader("Picos encontrados")
        st.dataframe(peaks)

        if len(peaks):
            st.subheader("🧠 Identificação Molecular")
            id_df = identify_molecular_groups(peaks)
            st.dataframe(id_df)
