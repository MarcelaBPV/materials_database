import streamlit as st
from supabase import create_client, Client

# ----- Conexão segura com Supabase -----
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ----- Layout do Streamlit -----
st.title("📊 Materials Database")
st.write("Exemplo de conexão com banco de dados no Supabase.")

# ----- Teste de conexão -----
try:
    data = supabase.table("samples").select("*").execute()
    st.success("Conexão estabelecida com sucesso!")
    st.dataframe(data.data)
except Exception as e:
    st.error(f"Erro ao acessar banco: {e}")
