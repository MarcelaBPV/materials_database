import streamlit as st
import pandas as pd
from supabase import create_client, Client

# --------------------- Conexão segura com Supabase ---------------------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --------------------- Layout do Streamlit ---------------------
st.set_page_config(page_title="📊 Materials Database", layout="wide")
st.title("📊 Materials Database")
st.write("Visualize, filtre e insira dados diretamente no Supabase.")

# --------------------- Lista de tabelas disponíveis ---------------------
tabelas = ["samples", "experiments", "results"]  # Ajuste conforme seu projeto
tabela_selec = st.selectbox("Escolha a tabela para visualizar", tabelas)

# --------------------- Consulta e exibição de dados ---------------------
try:
    response = supabase.table(tabela_selec).select("*").execute()

    if response.data:
        st.success(f"Conexão com '{tabela_selec}' estabelecida!")
        
        # Filtro de colunas
        st.subheader("Filtrar colunas")
        colunas = list(response.data[0].keys())
        col_selec = st.multiselect("Escolha colunas para exibir", colunas, default=colunas)
        st.dataframe([{k: row[k] for k in col_selec} for row in response.data])
    else:
        st.warning("Tabela vazia.")
except Exception as e:
    st.error(f"Erro ao acessar banco: {e}")

# --------------------- Upload de novos dados ---------------------
st.subheader("Inserir novos dados")
uploaded_file = st.file_uploader("Escolha um arquivo CSV para enviar", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Pré-visualização do CSV:")
        st.dataframe(df)

        if st.button("Enviar para Supabase"):
            for _, row in df.iterrows():
                supabase.table(tabela_selec).insert(row.to_dict()).execute()
            st.success("Dados enviados com sucesso!")
    except Exception as e:
        st.error(f"Erro ao processar o arquivo CSV: {e}")
