import streamlit as st
import pandas as pd
import numpy as np

st.title("🔄 Realocador de Carteira para Alocação Ótima")

# Escolha de modo de input
modo_input = st.radio("Como você quer informar sua carteira?", ["Porcentagem (%)", "Valor (R$)"])

# Número de classes
num_classes = st.number_input("Quantas classes de ativos você tem?", 1, 20, 4)

# Inputs
classes = []
aloc_atual = []
aloc_otima = []

st.write("### Preencha os dados de cada classe")

for i in range(num_classes):
    col1, col2, col3 = st.columns([3, 2, 2])
    with col1:
        nome = st.text_input(f"Classe {i+1}", f"Classe {i+1}")
    with col2:
        atual = st.number_input(
            f"{'Atual (%)' if modo_input == 'Porcentagem (%)' else 'Atual (R$)'} - {nome}",
            min_value=0.0,
            step=0.1,
            key=f"atual_{i}")
    with col3:
        otima = st.number_input(f"Alocação Ótima (%) - {nome}", 0.0, 100.0, step=0.1, key=f"otima_{i}")
    classes.append(nome)
    aloc_atual.append(atual)
    aloc_otima.append(otima)

# Processamento dos dados
if modo_input == "Valor (R$)":
    total_valor = sum(aloc_atual)
    if total_valor == 0:
        st.warning("⚠️ O valor total da carteira deve ser maior que zero.")
        aloc_atual_pct = None
    else:
        aloc_atual_pct = [v / total_valor * 100 for v in aloc_atual]
        total_base = total_valor
else:
    soma_pct = sum(aloc_atual)
    if abs(soma_pct - 100) > 1e-3:
        st.warning("⚠️ A soma das alocações atuais deve ser 100%.")
        aloc_atual_pct = None
    else:
        aloc_atual_pct = aloc_atual
        total_base = 100.0

# Resultado
if aloc_atual_pct and st.button("📊 Calcular Realocação"):
    otima_valores = [p / 100 * total_base for p in aloc_otima]
    atual_valores = [p / 100 * total_base for p in aloc_atual_pct]
    delta = [round(otima_valores[i] - atual_valores[i], 2) for i in range(num_classes)]

    resultado = pd.DataFrame({
        "Classe": classes,
        "Atual (%)": [round(p, 2) for p in aloc_atual_pct],
        "Ótima (%)": [round(p, 2) for p in aloc_otima],
        "Sugerido (%)": [round((otima_valores[i]/total_base)*100, 2) for i in range(num_classes)],
        "Delta (%)": [round((delta[i] / total_base) * 100, 2) for i in range(num_classes)],
        "Ação": ["Comprar" if d > 0 else "Vender" if d < 0 else "Manter" for d in delta],
    })

    st.write("### 📋 Plano de Realocação")
    st.dataframe(resultado)
