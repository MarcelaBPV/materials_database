# app.py
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from supabase import create_client
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ---------------------- CONFIG ----------------------
st.set_page_config(page_title="📊 Plataforma de Materiais", layout="wide")

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------------- IMPORTA MÓDULOS ----------------------
try:
    from raman_processing import process_raman_pipeline
    from molecular_identification import identify_blood_components
except ImportError:
    st.error("❌ Verifique se raman_processing.py e molecular_identification.py estão no mesmo diretório.")
    st.stop()

# ---------------------- FUNÇÕES SUPABASE ----------------------
def safe_insert(table, data):
    """Insere dados no Supabase com tratamento de erro."""
    try:
        supabase.table(table).insert(data).execute()
    except Exception as e:
        st.warning(f"⚠️ Erro ao inserir em {table}: {e}")

def create_measurement(sample_id, ensaio_type):
    """Cria um registro em measurements e retorna o ID."""
    res = supabase.table("measurements").insert({
        "sample_id": sample_id,
        "type": ensaio_type
    }).execute()
    return res.data[0]["id"]

def insert_samples(df):
    safe_insert("samples", df.to_dict(orient="records"))

def insert_raman(df, meas_id):
    df["measurement_id"] = meas_id
    safe_insert("raman_spectra", df.to_dict(orient="records"))

def insert_peaks(df_peaks, meas_id):
    df_peaks["measurement_id"] = meas_id
    safe_insert("raman_peaks", df_peaks.to_dict(orient="records"))

def insert_four_points(df, meas_id):
    df["measurement_id"] = meas_id
    safe_insert("four_point_probe_points", df.to_dict(orient="records"))

def insert_contact_angle(df, meas_id):
    df["measurement_id"] = meas_id
    safe_insert("contact_angle_points", df.to_dict(orient="records"))

def get_samples():
    return supabase.table("samples").select("*").order("id").execute().data

def get_measurements(sample_id):
    return supabase.table("measurements").select("*").eq("sample_id", sample_id).order("created_at", desc=True).execute().data


# =====================================================
# 🧱 ABA 1 — AMOSTRAS
# =====================================================
tab1, tab2, tab3 = st.tabs(["1️⃣ Amostras", "2️⃣ Ensaios", "3️⃣ Otimização (IA)"])

with tab1:
    st.header("1️⃣ Gestão de Amostras")

    samples = get_samples()
    df_samples = pd.DataFrame(samples)
    st.subheader("📋 Amostras cadastradas")
    st.dataframe(df_samples)

    st.markdown("---")
    st.subheader("📤 Upload de arquivo CSV")
    file = st.file_uploader("Selecione CSV com colunas 'sample_name' e 'description'", type=["csv"])
    if file:
        df_upload = pd.read_csv(file)
        st.write(df_upload.head())
        if st.button("Enviar para Supabase"):
            insert_samples(df_upload)
            st.success("✅ Amostras inseridas com sucesso!")
            st.experimental_rerun()

# =====================================================
# 🧪 ABA 2 — ENSAIOS
# =====================================================
with tab2:
    st.header("2️⃣ Ensaios")

    samples = get_samples()
    if not samples:
        st.warning("⚠️ Nenhuma amostra cadastrada. Adicione na aba anterior.")
    else:
        df_samples = pd.DataFrame(samples)
        sample_dict = {r["sample_name"]: r["id"] for r in samples}
        sample_choice = st.selectbox("Selecione a amostra", list(sample_dict.keys()))
        sample_id = sample_dict[sample_choice]

        ensaio_tipo = st.selectbox("Tipo de ensaio", ["Raman", "Tensiometria (Ângulo de Contato)", "Resistividade (4 Pontas)"])
        upload = st.file_uploader("📄 Upload do arquivo CSV do ensaio", type=["csv"])

        if upload:
            df_ens = pd.read_csv(upload)
            st.write("Pré-visualização dos dados:")
            st.dataframe(df_ens.head())

            # =====================================================
            # 🔬 RAMAN
            # =====================================================
            if ensaio_tipo == "Raman":
                try:
                    (x, y), peaks_df, fig = process_raman_pipeline(df_ens)
                    st.pyplot(fig)
                    st.success(f"✅ {len(peaks_df)} picos detectados.")

                    csv_peaks = peaks_df.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Baixar picos detectados (CSV)", data=csv_peaks, file_name="raman_peaks.csv", mime="text/csv")

                    # --- Identificação molecular ---
                    st.markdown("### 🧬 Identificação Molecular (Sangue + Prata + Papel)")
                    id_df = identify_blood_components(peaks_df, tolerance=10.0)

                    if not id_df.empty:
                        st.dataframe(id_df)
                        csv_id = id_df.to_csv(index=False).encode("utf-8")
                        st.download_button("⬇️ Baixar relatório molecular (CSV)", data=csv_id, file_name="molecular_identification.csv", mime="text/csv")
                    else:
                        st.info("Nenhuma correspondência molecular encontrada dentro da tolerância.")

                    if st.button("💾 Salvar espectro e picos no Supabase"):
                        meas_id = create_measurement(sample_id, "raman")
                        insert_raman(pd.DataFrame({"wavenumber_cm1": x, "intensity_a": y}), meas_id)
                        insert_peaks(peaks_df, meas_id)
                        st.success("✅ Espectro e picos salvos no banco com sucesso!")

                except Exception as e:
                    st.error(f"Erro no processamento Raman: {e}")

            # =====================================================
            # 💧 TENSIOMETRIA — Ângulo de Contato
            # =====================================================
            elif ensaio_tipo.startswith("Tensio"):
                st.subheader("💧 Análise de Ângulo de Contato (Tensiometria)")

                try:
                    if {"t_seconds", "angle_mean_deg"}.issubset(df_ens.columns):
                        st.line_chart(df_ens[["t_seconds", "angle_mean_deg"]].set_index("t_seconds"))
                    else:
                        st.info("Esperado: colunas 't_seconds' e 'angle_mean_deg'.")

                    if st.button("💾 Salvar dados de tensiometria no banco"):
                        meas_id = create_measurement(sample_id, "tensiometria")
                        insert_contact_angle(df_ens, meas_id)
                        st.success("✅ Dados de ângulo de contato salvos com sucesso!")
                except Exception as e:
                    st.error(f"Erro ao processar tensiometria: {e}")

            # =====================================================
            # ⚡ RESISTIVIDADE — 4 Pontas
            # =====================================================
            elif ensaio_tipo.startswith("Resist"):
                st.subheader("⚡ Medição de Resistividade (4 Pontas)")

                try:
                    if {"current_a", "voltage_v"}.issubset(df_ens.columns):
                        df_ens["resistance_ohm"] = df_ens["voltage_v"] / df_ens["current_a"]
                        resist_media = df_ens["resistance_ohm"].mean()
                        st.metric("Resistência Média (Ω)", f"{resist_media:.4f}")
                        st.line_chart(df_ens[["current_a", "voltage_v"]].set_index("current_a"))
                    else:
                        st.info("Esperado: colunas 'current_a' e 'voltage_v'.")

                    if st.button("💾 Salvar dados de resistividade no banco"):
                        meas_id = create_measurement(sample_id, "4_pontas")
                        insert_four_points(df_ens, meas_id)
                        st.success("✅ Dados de resistividade salvos com sucesso!")
                except Exception as e:
                    st.error(f"Erro ao processar resistividade: {e}")

        # --- Exibe medições existentes ---
        st.markdown("---")
        st.subheader("📊 Ensaios cadastrados")
        df_meas = pd.DataFrame(get_measurements(sample_id))
        st.dataframe(df_meas)

# =====================================================
# 🤖 ABA 3 — OTIMIZAÇÃO (IA)
# =====================================================
with tab3:
    st.header("3️⃣ Otimização (IA)")

    st.info("Treine um modelo de IA para classificar espectros Raman rotulados.")

    file_train = st.file_uploader("📂 CSV de treino com coluna 'label'", type=["csv"], key="train")
    if file_train:
        df_train = pd.read_csv(file_train)
        if "label" not in df_train.columns:
            st.error("O arquivo deve conter uma coluna 'label'.")
        else:
            X = df_train.drop(columns=["label"])
            y = df_train["label"]

            Xs = StandardScaler().fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, random_state=42)

            model = RandomForestClassifier(n_estimators=200, random_state=42)
            model.fit(X_train, y_train)
            acc = model.score(X_test, y_test)
            st.success(f"✅ Modelo treinado — Acurácia: {acc:.2%}")

            file_predict = st.file_uploader("📄 Envie CSV para prever", type=["csv"], key="predict")
            if file_predict:
                df_pred = pd.read_csv(file_predict)
                preds = model.predict(StandardScaler().fit_transform(df_pred))
                st.dataframe(pd.DataFrame({"Previsão": preds}))

# ---------------------- FOOTER ----------------------
st.markdown("---")
st.caption("📊 Plataforma integrada ao Supabase | Raman, Ângulo de Contato e Resistividade (4 Pontas) | Pipeline baseado no MIT (ramanchada2)")
