import streamlit as st
import pandas as pd
import numpy as np
from supabase import create_client
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter

# ---------------------- CONFIG ----------------------
st.set_page_config(page_title="Plataforma Materiais", layout="wide")

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------------- FUNÇÕES DB ------------------
def insert_samples(df):
    records = df.to_dict(orient="records")
    supabase.table("samples").insert(records).execute()

def insert_raman(df, measurement_id):
    df['measurement_id'] = measurement_id
    records = df.to_dict(orient="records")
    supabase.table("raman_spectra").insert(records).execute()

def insert_four_points(df, measurement_id):
    df['measurement_id'] = measurement_id
    supabase.table("four_point_probe_points").insert(df.to_dict("records")).execute()

def insert_contact_angle(df, measurement_id):
    df['measurement_id'] = measurement_id
    supabase.table("contact_angle_points").insert(df.to_dict("records")).execute()

def get_samples():
    return supabase.table("samples").select("*").order("id").execute().data

def get_raman(measurement_id):
    return supabase.table("raman_spectra").select("*").eq("measurement_id", measurement_id).execute().data

# ---------------------- PREPROCESSAMENTO RAMAN ------------------
def preprocess_raman(wavenumber, intensity):
    baseline = savgol_filter(intensity, 51, 3)  # estilo MIT
    corrected = intensity - baseline
    normalized = corrected / np.max(np.abs(corrected))
    return normalized

# ---------------------- INTERFACE ---------------------
tab1, tab2, tab3 = st.tabs(["1 Amostras", "2 Ensaios", "3 Otimização (IA)"])

# =============== ABA 1: AMOSTRAS =====================
with tab1:
    st.header("1 Gestão de Amostras")

    samples = get_samples()
    df_samples = pd.DataFrame(samples)
    st.subheader("Amostras no banco")
    st.dataframe(df_samples)

    st.subheader("Upload de arquivo para cadastro (CSV)")
    file = st.file_uploader("Selecione CSV com 'sample_name' e 'description'")

    if file:
        df_upload = pd.read_csv(file)
        st.write(df_upload)

        if st.button("Enviar para Supabase"):
            insert_samples(df_upload)
            st.success("✅ Amostras enviadas!")
            st.experimental_rerun()

# =============== ABA 2: ENSAIOS =====================
with tab2:
    st.header("2 Ensaios e Gráficos")

    samples = get_samples()
    df_samples = pd.DataFrame(samples)
    sample_dict = {r['sample_name']: r['id'] for r in samples}
    sample_choice = st.selectbox("Selecione Amostra", list(sample_dict.keys()))

    ensaio_tipo = st.selectbox("Tipo de ensaio", ["raman", "tensiometria", "4_pontas"])

    upload = st.file_uploader("Upload CSV do ensaio")

    if upload:
        df_ens = pd.read_csv(upload)
        st.write(df_ens)

        if st.button("Salvar Ensaio no Supabase"):
            meas = supabase.table("measurements").insert({
                "sample_id": sample_dict[sample_choice],
                "type": ensaio_tipo
            }).execute().data[0]["id"]

            if ensaio_tipo == "raman":
                insert_raman(df_ens.rename(columns={"wavenumber": "wavenumber_cm1", "intensity": "intensity_a"}), meas)
            elif ensaio_tipo == "4_pontas":
                insert_four_points(df_ens, meas)
            else:
                insert_contact_angle(df_ens, meas)

            st.success("✅ Dados enviados")

        # ----------- PLOT RAMAN -----------
        if ensaio_tipo == "raman":
            st.subheader("📈 Raman — processamento estilo MIT")
            wn = df_ens.iloc[:,0].values
            inten = df_ens.iloc[:,1].values
            norm = preprocess_raman(wn, inten)

            fig = plt.figure()
            plt.plot(wn, norm)
            plt.title("Raman Normalizado")
            plt.gca().invert_xaxis()
            st.pyplot(fig)

# =============== ABA 3: AI OTIMIZAÇÃO ================
with tab3:
    st.header("3 Otimização — IA com Random Forest (Raman)")

    st.info("Modelo usa dados Raman para prever classe do material (demo)")

    file_model = st.file_uploader("Carregue CSV Raman com 'label' para treinar modelo", key="train")

    if file_model:
        df = pd.read_csv(file_model)

        X = df.drop(columns=["label"])
        y = df["label"]

        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        acc = model.score(X_test, y_test)
        st.success(f"✅ Modelo treinado — Acurácia: {acc:.2%}")

        file_predict = st.file_uploader("Envie CSV Raman para prever", key="predict")

        if file_predict:
            dfp = pd.read_csv(file_predict)
            Xp = StandardScaler().fit_transform(dfp)
            pred = model.predict(Xp)
            st.write(pd.DataFrame({"previsão": pred}))
