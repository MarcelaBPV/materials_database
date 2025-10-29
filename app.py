# app.py
# -*- coding: utf-8 -*-
import os
import io
import math
import hashlib
from typing import Optional, List, Dict, Any

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from supabase import create_client, Client
from sklearn.linear_model import LinearRegression

# >>> Integração com o pipeline Raman (usa ramanchada2 dentro de raman_processing.py)
# As funções abaixo devem existir no seu raman_processing.py:
# - load_raman_dataframe(df_raw) -> pd.DataFrame[['wavenumber_cm1','intensity_a']]
# - preprocess_spectrum(df_spec, **kwargs) -> objeto compatível
# - process_raman_pipeline(df_spec, **kwargs) -> (processed_spec, peaks_df, fig)
# - compare_spectra(spec_a, spec_b) -> float
from raman_processing import (
    process_raman_pipeline,
    compare_spectra,
    load_raman_dataframe,
    preprocess_spectrum,
)

# --------------------- Configuração de página ---------------------
st.set_page_config(page_title="📊 Materials Database", layout="wide")
st.title("📊 Plataforma de Caracterização de Superfície de Materiais")

with st.sidebar:
    st.subheader("Status")
    st.caption("Conectado ao Supabase. Pronto para analisar Raman com `ramanchada2`.")
    if st.button("🧹 Limpar cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cache limpo.")

# --------------------- Conexão Supabase ---------------------
def _get_secret(k: str, default: Optional[str] = None) -> str:
    try:
        return st.secrets[k]
    except Exception:
        v = os.getenv(k, default)
        if v is None:
            raise RuntimeError(f"Secret {k} não configurado.")
        return v

try:
    SUPABASE_URL = _get_secret("SUPABASE_URL")
    SUPABASE_KEY = _get_secret("SUPABASE_KEY")
except Exception as e:
    st.error(f"⚠️ Configure `SUPABASE_URL` e `SUPABASE_KEY` em `.streamlit/secrets.toml` ou como variáveis de ambiente. Detalhe: {e}")
    st.stop()

@st.cache_resource(show_spinner=False)
def get_client(url: str, key: str) -> Client:
    return create_client(url, key)

supabase: Client = get_client(SUPABASE_URL, SUPABASE_KEY)

# --------------------- Utilidades de performance ---------------------
CHUNK_SIZE = 5000            # inserções em lotes para grandes arquivos
MAX_POINTS_PLOT = 8000       # limita pontos desenhados para não travar

def df_downsample_for_plot(df: pd.DataFrame, x: str, y: str, max_pts: int = MAX_POINTS_PLOT):
    if len(df) <= max_pts:
        return df[[x, y]]
    idx = np.linspace(0, len(df)-1, max_pts).astype(int)
    return df.iloc[idx][[x, y]]

def hash_df_fast(df: pd.DataFrame) -> str:
    h = hashlib.md5()
    h.update(str(df.shape).encode())
    if len(df):
        h.update(df.head(min(200, len(df))).to_csv(index=False).encode())
        h.update(df.tail(min(200, len(df))).to_csv(index=False).encode())
    return h.hexdigest()

def insert_in_chunks(table: str, rows: List[Dict[str, Any]], chunk_size: int = CHUNK_SIZE):
    if not rows:
        return 0
    total = 0
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i:i+chunk_size]
        supabase.table(table).insert(chunk).execute()
        total += len(chunk)
    return total

# --------------------- Esquema conhecido (para validação) ---------------------
def get_table_columns(table_name: str) -> List[str]:
    predefined = {
        "samples": ["id", "category_id", "sample_name", "description", "created_at"],
        "measurements": ["id", "sample_id", "type", "created_at"],
        "raman_spectra": ["id", "measurement_id", "wavenumber_cm1", "intensity_a"],
        "four_point_probe_points": ["id", "measurement_id", "current_a", "voltage_v"],
        "contact_angle_points": ["id", "measurement_id", "t_seconds", "angle_mean_deg"],
        "profilometry_points": ["id", "measurement_id", "position_um", "height_nm"],
        "categories": ["id", "name", "description"],
    }
    return predefined.get(table_name, [])

# --------------------- Cache de dados principais ---------------------
@st.cache_data(ttl=300, show_spinner=False)
def load_samples_df() -> pd.DataFrame:
    try:
        data = (
            supabase.table("samples")
            .select("id, sample_name, description, category_id, categories(name)")
            .execute()
            .data
        )
        return pd.DataFrame(data) if data else pd.DataFrame()
    except Exception as e:
        st.error(f"Erro ao carregar amostras: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def load_categories_df() -> pd.DataFrame:
    try:
        data = supabase.table("categories").select("id, name, description").execute().data
        return pd.DataFrame(data) if data else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def refresh_samples():
    st.session_state["samples_df"] = load_samples_df()

if "samples_df" not in st.session_state:
    st.session_state["samples_df"] = load_samples_df()

# --------------------- Funções de importação por tipo ---------------------
def get_or_create_measurement(sample_id: int, exp_type: str) -> int:
    res = (
        supabase.table("measurements")
        .select("id")
        .eq("sample_id", sample_id)
        .eq("type", exp_type)
        .limit(1)
        .execute()
    )
    if res.data:
        return res.data[0]["id"]
    new_m = supabase.table("measurements").insert({"sample_id": sample_id, "type": exp_type}).execute()
    return new_m.data[0]["id"]

def import_raman(df: pd.DataFrame, sample_id: int) -> int:
    # Tenta normalizar automaticamente formatos típicos
    df = df.rename(columns={
        "Wavenumber": "wavenumber_cm1",
        "wavenumber": "wavenumber_cm1",
        "Intensity": "intensity_a",
        "intensity": "intensity_a"
    })
    required = {"wavenumber_cm1", "intensity_a"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV/TSV precisa conter colunas: {sorted(required)}")
    measurement_id = get_or_create_measurement(sample_id, "raman")
    rows = [
        {
            "measurement_id": measurement_id,
            "wavenumber_cm1": float(r["wavenumber_cm1"]),
            "intensity_a": float(r["intensity_a"]),
        }
        for _, r in df.iterrows()
    ]
    return insert_in_chunks("raman_spectra", rows)

def import_4p(df: pd.DataFrame, sample_id: int) -> int:
    df = df.rename(columns={"corrente": "current_a", "tensao": "voltage_v"})
    required = {"current_a", "voltage_v"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV precisa conter colunas: {sorted(required)}")
    measurement_id = get_or_create_measurement(sample_id, "4_pontas")
    rows = [
        {
            "measurement_id": measurement_id,
            "current_a": float(r["current_a"]),
            "voltage_v": float(r["voltage_v"]),
        }
        for _, r in df.iterrows()
    ]
    return insert_in_chunks("four_point_probe_points", rows)

def import_contact_angle(df: pd.DataFrame, sample_id: int) -> int:
    df = df.rename(columns={"Time": "t_seconds", "Mean": "angle_mean_deg"})
    required = {"t_seconds", "angle_mean_deg"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV precisa conter colunas: {sorted(required)}")
    measurement_id = get_or_create_measurement(sample_id, "tensiometria")
    rows = [
        {
            "measurement_id": measurement_id,
            "t_seconds": float(r["t_seconds"]),
            "angle_mean_deg": float(r["angle_mean_deg"]),
        }
        for _, r in df.iterrows()
    ]
    return insert_in_chunks("contact_angle_points", rows)

def import_profilometry(df: pd.DataFrame, sample_id: int) -> int:
    df = df.rename(columns={"x_um": "position_um", "z_nm": "height_nm"})
    required = {"position_um", "height_nm"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV precisa conter colunas: {sorted(required)}")
    measurement_id = get_or_create_measurement(sample_id, "perfilometria")
    rows = [
        {
            "measurement_id": measurement_id,
            "position_um": float(r["position_um"]),
            "height_nm": float(r["height_nm"]),
        }
        for _, r in df.iterrows()
    ]
    return insert_in_chunks("profilometry_points", rows)

# --------------------- Tabs ---------------------
tabs = st.tabs(["1 Amostras", "2 Ensaios", "3 Otimização"])

# ===================== Aba 1: Amostras =====================
with tabs[0]:
    st.header("1) Gerenciamento de Amostras")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.subheader("📥 Importar amostras via CSV")
        up_s = st.file_uploader("Escolha um CSV de amostras", type="csv", key="samples_csv")
        if up_s is not None:
            try:
                df_new = pd.read_csv(up_s)
                st.write("Pré-visualização:")
                st.dataframe(df_new.head(20), use_container_width=True)
                if st.button("Cadastrar amostras no banco"):
                    valid = set(get_table_columns("samples"))
                    inserted = 0
                    rows = []
                    for _, row in df_new.iterrows():
                        payload = {k: v for k, v in row.to_dict().items() if k in valid and pd.notna(v)}
                        if payload:
                            rows.append(payload)
                    if rows:
                        inserted = insert_in_chunks("samples", rows, chunk_size=1000)
                    st.success(f"✅ Amostras cadastradas: {inserted}")
                    refresh_samples()
            except Exception as e:
                st.error(f"Erro ao ler CSV: {e}")

    with col_b:
        st.subheader("➕ Cadastrar amostra manualmente")
        sample_name = st.text_input("Nome da amostra")
        sample_desc = st.text_area("Descrição")
        # categorias (opcional)
        cats = load_categories_df()
        cat_id = None
        if not cats.empty:
            cat_label = st.selectbox("Categoria (opcional)", ["—"] + cats["name"].tolist())
            if cat_label != "—":
                cat_id = cats.loc[cats["name"] == cat_label, "id"].iloc[0]
        if st.button("Salvar amostra"):
            payload = {"sample_name": sample_name, "description": sample_desc}
            if cat_id is not None:
                payload["category_id"] = int(cat_id)
            try:
                supabase.table("samples").insert(payload).execute()
                st.success("✅ Amostra cadastrada.")
                refresh_samples()
            except Exception as e:
                st.error(f"Erro ao salvar amostra: {e}")

    st.subheader("Amostras cadastradas")
    df_samples = st.session_state["samples_df"]
    if df_samples.empty:
        st.info("Nenhuma amostra cadastrada ainda.")
    else:
        st.dataframe(df_samples, use_container_width=True)

# ===================== Aba 2: Ensaios =====================
with tabs[1]:
    st.header("2) Ensaios por Amostra")
    df_samples = st.session_state["samples_df"]
    if df_samples.empty:
        st.warning("Cadastre amostras primeiro.")
    else:
        sample_choice = st.selectbox("Escolha a amostra", df_samples["id"], key="ens_sample")
        tipo = st.radio("Tipo de experimento", ["Raman", "4 Pontas", "Ângulo de Contato", "Perfilometria"], horizontal=True)

        st.subheader(f"📤 Importar arquivo de {tipo}")
        uploaded = st.file_uploader(f"Arquivo para {tipo}", type=["csv", "tsv", "txt", "log", "LOG"])

        if uploaded is not None:
            try:
                # Detecta separador comum
                name = uploaded.name.lower()
                if name.endswith(".tsv"):
                    df_file = pd.read_csv(uploaded, sep="\t", engine="python", comment="#")
                else:
                    # muitos equipamentos exportam primeira linha de cabeçalho simples
                    try:
                        df_file = pd.read_csv(uploaded, engine="python", comment="#")
                    except Exception:
                        uploaded.seek(0)
                        df_file = pd.read_csv(uploaded, sep="\t", engine="python", comment="#")

                st.write("Prévia:")
                st.dataframe(df_file.head(20), use_container_width=True)

                if st.button("Enviar para o banco"):
                    if tipo == "Raman":
                        # padroniza para colunas esperadas; se for export de equipamento (2 colunas sem header)
                        if df_file.shape[1] == 2 and set(df_file.columns) == {0, 1}:
                            df_file.columns = ["wavenumber_cm1", "intensity_a"]
                        count = import_raman(df_file, int(sample_choice))
                        st.success(f"✅ Raman importado: {count} pontos.")
                    elif tipo == "4 Pontas":
                        count = import_4p(df_file, int(sample_choice))
                        st.success(f"✅ 4 Pontas importado: {count} pontos.")
                    elif tipo == "Ângulo de Contato":
                        count = import_contact_angle(df_file, int(sample_choice))
                        st.success(f"✅ Ângulo de Contato importado: {count} pontos.")
                    elif tipo == "Perfilometria":
                        count = import_profilometry(df_file, int(sample_choice))
                        st.success(f"✅ Perfilometria importada: {count} pontos.")
            except Exception as e:
                st.error(f"Erro ao importar: {e}")

        # ---- Visualização do que já existe para a amostra/tipo selecionados
        st.subheader(f"Dados existentes de {tipo}")
        table_map = {
            "Raman": "raman_spectra",
            "4 Pontas": "four_point_probe_points",
            "Ângulo de Contato": "contact_angle_points",
            "Perfilometria": "profilometry_points",
        }
        table_name = table_map.get(tipo)

        df_existing = pd.DataFrame()
        if table_name:
            # pega todos os measurements desta amostra deste tipo
            ms = (
                supabase.table("measurements")
                .select("id")
                .eq("sample_id", int(sample_choice))
                .eq("type", table_name.replace("_points", "").replace("raman_spectra", "raman").replace("profilometry", "perfilometria"))
                .execute()
                .data
            )
            mids = [m["id"] for m in ms] if ms else []
            rows = []
            for mid in mids:
                resp = supabase.table(table_name).select("*").eq("measurement_id", mid).execute()
                if resp and resp.data:
                    rows.extend(resp.data)
            if rows:
                df_existing = pd.DataFrame(rows)

        if df_existing.empty:
            st.info("Sem dados para exibir.")
        else:
            fig, ax = plt.subplots()
            if tipo == "Raman":
                plot_df = df_downsample_for_plot(df_existing.sort_values("wavenumber_cm1"), "wavenumber_cm1", "intensity_a")
                ax.plot(plot_df["wavenumber_cm1"], plot_df["intensity_a"])
                ax.set_xlabel("Número de onda (cm⁻¹)")
                ax.set_ylabel("Intensidade (a.u.)")
            elif tipo == "4 Pontas":
                ax.plot(df_existing["current_a"], df_existing["voltage_v"], "o-")
                ax.set_xlabel("Corrente (A)")
                ax.set_ylabel("Tensão (V)")
            elif tipo == "Ângulo de Contato":
                ax.plot(df_existing["t_seconds"], df_existing["angle_mean_deg"], "o-")
                ax.set_xlabel("Tempo (s)")
                ax.set_ylabel("Ângulo médio (°)")
            elif tipo == "Perfilometria":
                plot_df = df_downsample_for_plot(df_existing.sort_values("position_um"), "position_um", "height_nm")
                ax.plot(plot_df["position_um"], plot_df["height_nm"])
                ax.set_xlabel("Posição (µm)")
                ax.set_ylabel("Altura (nm)")
            st.pyplot(fig)
            with st.expander("Ver tabela (primeiras linhas)"):
                st.dataframe(df_existing.head(100), use_container_width=True)

# ===================== Aba 3: Otimização =====================
with tabs[2]:
    st.header("3) Otimização Automática")
    df_samples = st.session_state["samples_df"]
    if df_samples.empty:
        st.warning("Cadastre amostras primeiro.")
    else:
        col1, col2 = st.columns([1, 1])
        with col1:
            sample_choice = st.selectbox("Amostra", df_samples["id"], key="opt_sample")
        with col2:
            tipo = st.radio("Experimento", ["Raman", "4 Pontas", "Ângulo de Contato"], horizontal=True, key="opt_tipo")

        # Coleta dados do tipo escolhido para a amostra
        def collect_data_for(sample_id: int, tipo: str) -> pd.DataFrame:
            table_map = {
                "Raman": ("measurements", "raman_spectra", "raman"),
                "4 Pontas": ("measurements", "four_point_probe_points", "4_pontas"),
                "Ângulo de Contato": ("measurements", "contact_angle_points", "tensiometria"),
            }
            meas_tbl, data_tbl, meas_type = table_map[tipo]
            ms = (
                supabase.table(meas_tbl)
                .select("id")
                .eq("sample_id", int(sample_id))
                .eq("type", meas_type)
                .execute()
                .data
            )
            mids = [m["id"] for m in ms] if ms else []
            rows = []
            for mid in mids:
                r = supabase.table(data_tbl).select("*").eq("measurement_id", mid).execute()
                if r and r.data:
                    rows.extend(r.data)
            return pd.DataFrame(rows) if rows else pd.DataFrame()

        df_opt = collect_data_for(int(sample_choice), tipo)

        if df_opt.empty:
            st.warning(f"Nenhum dado de {tipo} encontrado.")
        else:
            if tipo == "Raman":
                st.subheader("Pipeline Raman (`ramanchada2`)")
                # Parâmetros opcionais do pipeline
                colp1, colp2, colp3 = st.columns(3)
                with colp1:
                    do_smooth = st.checkbox("Suavizar", value=True)
                with colp2:
                    do_baseline = st.checkbox("Remover baseline", value=True)
                with colp3:
                    peak_prom = st.number_input("Prominência mínima (picos)", min_value=0.0, value=0.0, step=0.1)

                # Execução sob demanda (evita travar UI)
                if st.button("▶️ Processar Raman"):
                    try:
                        df_spec = load_raman_dataframe(df_opt)  # normaliza colunas
                        # hash para cache
                        cache_key = f"raman_{hash_df_fast(df_spec)}_{int(do_smooth)}_{int(do_baseline)}_{peak_prom}"
                        @st.cache_data(show_spinner=True)
                        def _run_pipeline(_df, _do_smooth, _do_baseline, _prom):
                            processed, peaks, fig = process_raman_pipeline(
                                _df,
                                smooth=_do_smooth,
                                baseline=_do_baseline,
                                peak_prominence=_prom if _prom > 0 else None,
                            )
                            return processed, peaks, fig
                        processed, peaks, fig = _run_pipeline(df_spec, do_smooth, do_baseline, peak_prom)
                        st.pyplot(fig)
                        st.subheader("Picos detectados")
                        st.dataframe(peaks, use_container_width=True)

                        # ---- Comparação com outra amostra (opcional)
                        st.subheader("Comparar com outra amostra (Raman)")
                        other_sample = st.selectbox("Amostra de referência", df_samples["id"], key="opt_other_sample")
                        if st.button("Comparar espectros"):
                            df_ref = collect_data_for(int(other_sample), "Raman")
                            if df_ref.empty:
                                st.warning("A amostra de referência não possui Raman.")
                            else:
                                try:
                                    spec_ref = preprocess_spectrum(load_raman_dataframe(df_ref))
                                    similarity = compare_spectra(processed, spec_ref)
                                    st.info(f"Similaridade espectral (0–1): **{similarity:.3f}**")
                                except Exception as e:
                                    st.error(f"Erro na comparação: {e}")
                    except Exception as e:
                        st.error(f"Falha no pipeline Raman: {e}")

            elif tipo == "4 Pontas":
                st.subheader("Ajuste Linear (R ≈ V/I)")
                try:
                    df_opt = df_opt.sort_values("current_a")
                    X = df_opt[["current_a"]].values
                    y = df_opt["voltage_v"].values
                    model = LinearRegression().fit(X, y)
                    fig, ax = plt.subplots()
                    ax.scatter(df_opt["current_a"], df_opt["voltage_v"], label="Dados")
                    ax.plot(df_opt["current_a"], model.predict(X), label="Ajuste Linear")
                    ax.set_xlabel("Corrente (A)")
                    ax.set_ylabel("Tensão (V)")
                    ax.legend()
                    st.pyplot(fig)
                    st.success(f"Resistência estimada (coeficiente angular): **{model.coef_[0]:.6f} Ω**")
                except Exception as e:
                    st.error(f"Erro na otimização 4 Pontas: {e}")

            elif tipo == "Ângulo de Contato":
                st.subheader("Tendência temporal (polinômio 3º)")
                try:
                    df_opt = df_opt.sort_values("t_seconds")
                    coef = np.polyfit(df_opt["t_seconds"], df_opt["angle_mean_deg"], 3)
                    poly = np.poly1d(coef)
                    fig, ax = plt.subplots()
                    ax.plot(df_opt["t_seconds"], df_opt["angle_mean_deg"], "o", label="Dados")
                    ax.plot(df_opt["t_seconds"], poly(df_opt["t_seconds"]), label="Ajuste 3º grau")
                    ax.set_xlabel("Tempo (s)")
                    ax.set_ylabel("Ângulo médio (°)")
                    ax.legend()
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Erro na otimização de ângulo de contato: {e}")
