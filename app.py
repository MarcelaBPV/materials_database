import streamlit as st
from supabase import create_client, Client

# --------------------- Conexão segura com Supabase --------------------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --------------------- Layout do Streamlit ---------------------
st.set_page_config(page_title="📊 Materials Database", layout="wide")
st.title("📊 Materials Database")
st.write("Exemplo de conexão com banco de dados no Supabase.")

# --------------------- Teste de conexão e exibição de dados --------------------
try:
    response = supabase.table("samples").select("*").execute()
    if response.data:
        st.success("Conexão estabelecida com sucesso!")
        st.dataframe(response.data)
    else:
        st.warning("Conexão OK, mas a tabela está vazia.")
except Exception as e:
    st.error(f"Erro ao acessar banco: {e}")

# --------------------- Filtros opcionais ---------------------
if response.data:
    st.subheader("Filtrar dados por colunas")
    colunas = list(response.data[0].keys())
    col_selec = st.multiselect("Escolha colunas para exibir", colunas, default=colunas)
    st.dataframe([{k: row[k] for k in col_selec} for row in response.data])
