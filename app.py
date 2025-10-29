# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from supabase import create_client, Client
from sklearn.linear_model import LinearRegression

# >>> Integração com o pipeline Raman (usa ramanchada2 dentro de raman_processing.py)
from raman_processing import (
    process_raman_pipeline,
    compare_spectra,
    load_raman_dataframe,
    preprocess_spectrum,
)

# --------------------- Configuração da página ---------------------
st.set_page_config(page_title="📊 Materials Database", layout="wide")
st.title("📊 Plataforma de Caracterização de Superfície de Materiais")

with st.sidebar:
    st.markdown("**Status**")
    st.caption("App conectado ao Supabase e pronto para analisar espectros Raman com `ramanchada2`.")

# --------------------- Conexão Supabase ---------------------
try:
    supabase_url = st.secrets["SUPABASE_URL"]
    supabase_key = st.secrets["SUPABASE_KEY"]
except Exception:
    st.error(
        "⚠️ Configure `SUPABASE_URL` e `SUPABASE_KEY` em `.streamlit/secrets.toml`."
    )
    st.stop()

supabase: Client = create_client(supabase_url, supabase_key)

# --------------------- Função para obter colunas válidas ---------------------
def get_table_columns(table_name):
    predefined = {
        "samples": ["id", "category_id", "sample_name", "description", "created_at"],
        "measurements": ["id", "sample_id", "type", "created_at"],
        "raman_spectra": ["id", "measurement_id", "wavenumber_cm1", "intensity_a"],
        "four_point_probe_points": ["id", "measurement_id", "current_a", "voltage_v"],
        "contact_angle_points": ["id", "measurement_id", "t_seconds", "angle_mean_deg"],
        "profilometry_points": ["id", "measurement_id", "position_um", "height_nm"],
    }
    return predefined.get(table_name, [])

# --------------------- Carregamento de amostras ---------------------
@st.cache_data(ttl=300)
def load_samples():
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

df_samples = load_samples()

# --------------------- Funções de importação de ensaios ---------------------
def get_measurement_id(sample_id, exp_type):
    meas = (
        supabase.table("measurements")
        .select("id")
        .eq("sample_id", sample_id)
        .eq("type", exp_type)
        .execute()
    )
    if meas.data:
        return meas.data[0]["id"]
    new_meas = (
        supabase.table("measurements")
        .insert({"sample_id": sample_id, "type": exp_type})
        .execute()
    )
    return new_meas.data[0]["id"]

def import_raman(df, sample_id):
    # Espera colunas: wavenumber_cm1, intensity_a
    required = {"wavenumber_cm1", "intensity_a"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV precisa conter colunas: {required}")
    measurement_id = get_measurement_id(sample_id, "raman")
    rows = [
        {
            "measurement_id": measurement_id,
            "wavenumber_cm1": float(r["wavenumber_cm1"]),
            "intensity_a": float(r["intensity_a"]),
        }
        for _, r in df.iterrows()
    ]
    supabase.table("raman_spectra").insert(rows).execute()
    return len(rows)

def import_4p(df, sample_id):
    required = {"current_a", "voltage_v"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV precisa conter colunas: {required}")
    measurement_id = get_measurement_id(sample_id, "4_pontas")
    rows = [
        {
            "measurement_id": measurement_id,
            "current_a": float(r["current_a"]),
            "voltage_v": float(r["voltage_v"]),
        }
        for _, r in df.iterrows()
    ]
    supabase.table("four_point_probe_points").insert(rows).execute()
    return len(rows)

def import_contact_angle(df, sample_id):
    required = {"t_seconds", "angle_mean_deg"}
    if not required.issubset(df.columns):
        # tentar renomear formatos comuns (ex: da exportação da câmera)
        df = df.rename(columns={"Time": "t_seconds", "Mean": "angle_mean_deg"})
        if not required.issubset(df.columns):
            raise ValueError(f"CSV precisa conter colunas: {required}")
    measurement_id = get_measurement_id(sample_id, "tensiometria")
    rows = [
        {
            "measurement_id": measurement_id,
            "t_seconds": float(r["t_seconds"]),
            "angle_mean_deg": float(r["angle_mean_deg"]),
        }
        for _, r in df.iterrows()
    ]
    supabase.table("contact_angle_points").insert(rows).execute()
    return len(rows)

# --------------------- Abas ---------------------
abas = st.tabs(["1 Amostras", "2 Ensaios", "3 Otimização"])

# --------------------- Aba 1: Amostras ---------------------
with abas[0]:
    st.header("1 Gerenciamento de Amostras")

    # Upload CSV (de amostras)
    st.subheader("📥 Importar nova amostra")
    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")

    if uploaded_file is not None:
        try:
            new_sample_df = pd.read_csv(uploaded_file)
            st.write("Pré-visualização do CSV:")
            st.dataframe(new_sample_df.head())

            if st.button("Cadastrar amostras no banco"):
                valid_columns = get_table_columns("samples")
                for _, row in new_sample_df.iterrows():
                    filtered_row = {
                        k: v for k, v in row.to_dict().items() if k in valid_columns
                    }
                    try:
                        supabase.table("samples").insert(filtered_row).execute()
                    except Exception as e:
                        st.error(f"Erro ao cadastrar amostra: {e}")
                st.success("✅ Amostras cadastradas com sucesso!")
                df_samples = load_samples()  # atualizar tabela
        except Exception as e:
            st.error(f"Erro ao ler CSV: {e}")

    # Tabela de amostras existentes
    st.subheader("Amostras cadastradas")
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
        tipo = st.radio(
            "Tipo de experimento",
            ["Raman", "4 Pontas", "Ângulo de Contato", "Perfilometria"],
        )

        # Upload de arquivo de ensaio
        uploaded_file = st.file_uploader(
            f"Escolha arquivo para {tipo}", type=["csv", "txt", "LOG"]
        )
        if uploaded_file:
            df = None
            try:
                if tipo == "Raman":
                    # formatos comuns: TSV exportado do equipamento
                    df = pd.read_csv(
                        uploaded_file,
                        sep="\t",
                        names=["wavenumber_cm1", "intensity_a"],
                        skiprows=1,
                        engine="python",
                    )
                    count = import_raman(df, sample_choice)
                    st.success(f"✅ Raman importado: {count} pontos")
                elif tipo == "4 Pontas":
                    df = pd.read_csv(uploaded_file)
                    count = import_4p(df, sample_choice)
                    st.success(f"✅ 4 Pontas importado: {count} pontos")
                elif tipo == "Ângulo de Contato":
                    df = pd.read_csv(uploaded_file)
                    count = import_contact_angle(df, sample_choice)
                    st.success(f"✅ Ângulo de contato importado: {count} pontos")
                elif tipo == "Perfilometria":
                    df = pd.read_csv(uploaded_file)
                    # você pode criar importador específico depois
                    st.info("Visualizando arquivo de perfilometria (importador opcional).")
                st.write(df.head() if df is not None else "Arquivo carregado.")
            except Exception as e:
                st.error(f"Erro ao importar ensaio: {e}")

        # Visualização de dados já cadastrados
        try:
            measurements = (
                supabase.table("measurements")
                .select("*")
                .eq("sample_id", sample_choice)
                .execute()
                .data
            )
        except Exception as e:
            st.error(f"Erro ao buscar measurements: {e}")
            measurements = []

        measurement_ids = [m["id"] for m in measurements] if measurements else []
        data = []
        for mid in measurement_ids:
            table_map = {
                "Raman": "raman_spectra",
                "4 Pontas": "four_point_probe_points",
                "Ângulo de Contato": "contact_angle_points",
                "Perfilometria": "profilometry_points",
            }
            table_name = table_map.get(tipo)
            if table_name:
                resp = (
                    supabase.table(table_name)
                    .select("*")
                    .eq("measurement_id", mid)
                    .execute()
                )
                if resp and resp.data:
                    data.extend(resp.data)

        df_existing = pd.DataFrame(data) if data else pd.DataFrame()
        if not df_existing.empty:
            st.subheader(f"Dados existentes de {tipo}")
            st.write(df_existing.head())
            fig, ax = plt.subplots()
            if tipo == "Raman":
                ax.plot(df_existing["wavenumber_cm1"], df_existing["intensity_a"])
                ax.set_xlabel("Número de onda (cm⁻¹)")
                ax.set_ylabel("Intensidade (a.u.)")
            elif tipo == "4 Pontas":
                ax.plot(df_existing["current_a"], df_existing["voltage_v"], "o-")
                ax.set_xlabel("Corrente (A)")
                ax.set_ylabel("Tensão (V)")
            elif tipo == "Ângulo de Contato":
                ax.plot(df_existing["t_seconds"], df_existing["angle_mean_deg"], "ro-")
                ax.set_xlabel("Tempo (s)")
                ax.set_ylabel("Ângulo médio (°)")
            elif tipo == "Perfilometria":
                ax.plot(df_existing["position_um"], df_existing["height_nm"])
                ax.set_xlabel("Posição (µm)")
                ax.set_ylabel("Altura (nm)")
            st.pyplot(fig)

# --------------------- Aba 3: Otimização ---------------------
with abas[2]:
    st.header("3 Otimização Automática de Ensaios")
    if df_samples.empty:
        st.warning("Cadastre amostras primeiro.")
    else:
        sample_choice = st.selectbox(
            "Selecione a amostra para otimizar", df_samples["id"], key="opt"
        )
        tipo = st.radio(
            "Escolha o experimento para otimizar",
            ["Raman", "4 Pontas", "Ângulo de Contato"],
            key="opt_tipo",
        )

        try:
            measurements = (
                supabase.table("measurements")
                .select("*")
                .eq("sample_id", sample_choice)
                .execute()
                .data
            )
        except Exception as e:
            st.error(f"Erro ao buscar measurements: {e}")
            measurements = []

        measurement_ids = [m["id"] for m in measurements] if measurements else []
        data = []
        for mid in measurement_ids:
            table_map = {
                "Raman": "raman_spectra",
                "4 Pontas": "four_point_probe_points",
                "Ângulo de Contato": "contact_angle_points",
            }
            table_name = table_map.get(tipo)
            if table_name:
                resp = (
                    supabase.table(table_name)
                    .select("*")
                    .eq("measurement_id", mid)
                    .execute()
                )
                if resp and resp.data:
                    data.extend(resp.data)

        df = pd.DataFrame(data) if data else pd.DataFrame()

        if df.empty:
            st.warning(f"Nenhum dado de {tipo} encontrado.")
        else:
            if tipo == "Raman":
                # >>> Processamento completo com ramanchada2 (via raman_processing.py)
                try:
                    processed, peaks, fig = process_raman_pipeline(df)
                except Exception as e:
                    st.error(f"Falha no pipeline Raman: {e}")
                else:
                    st.pyplot(fig)
                    st.subheader("Picos detectados")
                    st.dataframe(peaks)

                    # ---- Comparação com outra amostra (opcional)
                    st.subheader("Comparar com outra amostra")
                    other_sample = st.selectbox("Amostra de referência", df_samples["id"])
                    if st.button("Comparar espectros"):
                        ref_meas = (
                            supabase.table("measurements")
                            .select("*")
                            .eq("sample_id", other_sample)
                            .eq("type", "raman")
                            .execute()
                            .data
                        )
                        if ref_meas:
                            ref_id = ref_meas[0]["id"]
                            ref_data = (
                                supabase.table("raman_spectra")
                                .select("*")
                                .eq("measurement_id", ref_id)
                                .execute()
                                .data
                            )
                            df_ref = pd.DataFrame(ref_data)
                            try:
                                spec_ref = preprocess_spectrum(load_raman_dataframe(df_ref))
                                similarity = compare_spectra(processed, spec_ref)
                                st.info(f"Similaridade espectral: **{similarity:.3f}**")
                            except Exception as e:
                                st.error(f"Erro na comparação: {e}")
                        else:
                            st.warning("Amostra de referência não possui ensaio Raman.")

            elif tipo == "4 Pontas":
                # Ajuste linear (R = V/I)
                try:
                    X = df[["current_a"]].values
                    y = df["voltage_v"].values
                    model = LinearRegression().fit(X, y)
                    fig, ax = plt.subplots()
                    ax.scatter(df["current_a"], df["voltage_v"], label="Dados")
                    ax.plot(df["current_a"], model.predict(X), "r-", label="Ajuste Linear")
                    ax.set_xlabel("Corrente (A)")
                    ax.set_ylabel("Tensão (V)")
                    ax.legend()
                    st.pyplot(fig)
                    st.success(f"Resistência estimada (slope): {model.coef_[0]:.6f} Ω")
                except Exception as e:
                    st.error(f"Erro na otimização 4 Pontas: {e}")

            elif tipo == "Ângulo de Contato":
                try:
                    coef = np.polyfit(df["t_seconds"], df["angle_mean_deg"], 3)
                    poly = np.poly1d(coef)
                    fig, ax = plt.subplots()
                    ax.plot(df["t_seconds"], df["angle_mean_deg"], "bo", label="Dados")
                    ax.plot(df["t_seconds"], poly(df["t_seconds"]), "r-", label="Ajuste 3º grau")
                    ax.set_xlabel("Tempo (s)")
                    ax.set_ylabel("Ângulo médio (°)")
                    ax.legend()
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Erro na otimização de ângulo de contato: {e}")
