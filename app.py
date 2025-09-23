import streamlit as st
from supabase import create_client, Client

# Conexão com Supabase usando secrets
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

st.title("📊 Materials Database")
st.write("Exemplo de conexão com banco de dados no Supabase.")

# Exemplo: listar dados da tabela "samples"
try:
    data = supabase.table("samples").select("*").execute()
    st.write(data.data)
except Exception as e:
    st.error(f"Erro ao acessar banco: {e}")
