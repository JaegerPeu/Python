pip install garmin


import streamlit as st
from garminconnect import Garmin
import pandas as pd

# Configuração do título da página e barra lateral
st.title("Preparação Maratona Fefê")
st.sidebar.title("Login Garmin")

# Entrada para o e-mail e senha
email = st.sidebar.text_input("Email", type="default")
password = st.sidebar.text_input("Password", type="password")

# Função para fazer login e pegar os dados
if st.sidebar.button("Get Activities"):
    if email and password:
        try:
            # Login na conta Garmin
            client = Garmin(email, password)
            client.login()

            # Obter atividades
            activities = client.get_activities(0, 100)

            # Converter atividades para DataFrame pandas
            df = pd.DataFrame(activities)

            # Mostrar os dados no app
            st.write("Suas Atividades:")
            st.dataframe(df)

        except Exception as e:
            st.error(f"Erro {e}")
    else:
        st.warning("Colocar email e senha.")
