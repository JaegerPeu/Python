import streamlit as st
import numpy as np
import pandas as pd

# Funções para calcular PA e PG
def progressao_aritmetica(a1, r, n):
    return a1 + (n-1) * r

def progressao_geometrica(a1, r, n):
    return a1 * r**(n-1)

# Configurações iniciais
st.title("Calculadora de Progressões Interativa")

# Barra lateral para escolher o tipo de progressão
tipo_progressao = st.sidebar.selectbox("Escolha o tipo de Progressão:", ["PA (Aritmética)", "PG (Geométrica)"])

# Abas para a interface
aba = st.sidebar.radio("Escolha a aba:", ["Soma Fixa", "Soma Variável"])

if aba == "Soma Fixa":
    
    # Ajuste dos parâmetros com sliders que permitem entrada manual
    a1 = st.number_input("Primeiro Termo (a1)", min_value=1, max_value=10000, value=10)
    r = st.number_input("Razão/Diferença (r)", min_value=0.01, max_value=1000.0, step=0.01, value=2.0)
    n = st.number_input("Número de Termos (n)", min_value=1, max_value=5000, value=5)

    # Calculando a progressão conforme a escolha
    if tipo_progressao == "PA (Aritmética)":
        termos = np.array([progressao_aritmetica(a1, r, i) for i in range(1, n+1)])
    else:
        termos = np.array([progressao_geometrica(a1, r, i) for i in range(1, n+1)])
    
    # Soma dos termos
    soma_termos = np.sum(termos)
    
    # Ajuste do parâmetro baseado na soma desejada


elif aba == "Soma Variável":
    # Ajuste dos parâmetros com sliders que permitem entrada manual
    a1 = st.number_input("Primeiro Termo (a1)", min_value=1, max_value=10000, value=10)
    r = st.number_input("Razão/Diferença (r)", min_value=0.01, max_value=1000.0, step=0.01, value=2.0)
    n = st.number_input("Número de Termos (n)", min_value=1, max_value=5000, value=5)

    # Calculando a progressão conforme a escolha
    if tipo_progressao == "PA (Aritmética)":
        termos = np.array([progressao_aritmetica(a1, r, i) for i in range(1, n+1)])
    else:
        termos = np.array([progressao_geometrica(a1, r, i) for i in range(1, n+1)])
    
    # Soma dos termos
    soma_termos = np.sum(termos)

# Exibindo o último termo da progressão
ultimo_termo = termos[-1]
st.write(f"Último Termo da Progressão: {ultimo_termo}")

# Exibindo a soma em um card
st.metric(label="Soma dos Termos", value=f"{soma_termos}")

# Gráfico da progressão
st.line_chart(termos)

# Criando a tabela com n e termos
df_termos = pd.DataFrame({
    'n': np.arange(1, n+1),
    'Termo': termos
})

# Exibindo a tabela
st.write("Tabela dos Termos da Progressão")
st.table(df_termos)
