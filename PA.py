import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Realocador de Carteira", layout="wide")
st.title("🔄 Realocador de Carteira para Alocação Ótima")

# Escolha de modo de input
modo_input = st.radio("Como você quer informar sua carteira?", ["Porcentagem (%)", "Valor (R$)"])

# Número de classes
num_classes = st.number_input("Quantas classes de ativos você tem?", 1, 20, 4)

classes = []
aloc_atual = []
aloc_otima = []
fixos = []
nao_vender_flags = []
minimos = []
maximos = []

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
        otima = st.number_input(f"Alocação Ótima (%) - {nome}", 0.0, 100.0, step=0.1, key=f"otima_{i}")
    with col3:
        fixo = st.number_input(f"Fixo (%) - {nome}", 0.0, 100.0, step=0.1, key=f"fixo_{i}")
        minimo = st.number_input(f"Mínimo (%) - {nome}", 0.0, 100.0, step=0.1, key=f"minimo_{i}")
        maximo = st.number_input(f"Máximo (%) - {nome}", 0.0, 100.0, step=0.1, key=f"maximo_{i}")
        nao_vender = st.checkbox(f"🔒 Não vender {nome}", key=f"nao_vender_{i}")

    classes.append(nome)
    aloc_atual.append(atual)
    aloc_otima.append(otima)
    fixos.append(fixo)
    nao_vender_flags.append(nao_vender)
    minimos.append(minimo)
    maximos.append(maximo)

# Processar input
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

# Realocação
if aloc_atual_pct and st.button("📊 Calcular Realocação"):
    resultado = []

    # Identificar classes travadas
    travado_pct = 0.0
    travado_classes = []
    for i in range(num_classes):
        if nao_vender_flags[i] and aloc_atual_pct[i] > aloc_otima[i]:
            travado_pct += aloc_atual_pct[i]
            travado_classes.append(i)

    restante_pct = 100.0 - travado_pct
    soma_otima_travada = sum([aloc_otima[j] for j in travado_classes])

    otima_ajustada = []
    for i in range(num_classes):
        if i in travado_classes:
            otima_ajustada.append(aloc_atual_pct[i])
        else:
            if (100.0 - soma_otima_travada) == 0:
                otima_ajustada.append(aloc_atual_pct[i])  # evitar divisão por zero
            else:
                otima_ajustada.append(aloc_otima[i] / (100.0 - soma_otima_travada) * restante_pct)

    for i in range(num_classes):
        atual_pct = aloc_atual_pct[i]
        otimo_pct = otima_ajustada[i]
        fixo_pct = min(fixos[i], atual_pct)

        atual_variavel = atual_pct - fixo_pct
        otimo_variavel = otimo_pct - fixo_pct

        delta_pct = otimo_variavel - atual_variavel

        if nao_vender_flags[i] and delta_pct < 0:
            delta_pct = 0
            otimo_variavel = atual_variavel

        sugerido_pct = fixo_pct + otimo_variavel
        delta_total = sugerido_pct - atual_pct

        atual_valor = atual_pct / 100 * total_base
        sugerido_valor = sugerido_pct / 100 * total_base
        delta_valor = sugerido_valor - atual_valor

        # Verificar enquadramento
        enquadrado = "Sim" if (sugerido_pct >= minimos[i] and sugerido_pct <= maximos[i]) else "Não"

        resultado.append({
            "Classe": classes[i],
            "Atual (%)": round(atual_pct, 2),
            "Ótima (%)": round(aloc_otima[i], 2),
            "Fixo (%)": round(fixo_pct, 2),
            "Sugerido (%)": round(sugerido_pct, 2),
            "Delta (%)": round(delta_total, 2),
            "Ação": "Comprar" if delta_total > 0 else "Vender" if delta_total < 0 else "Manter",
            "Min (%)": minimos[i],
            "Max (%)": maximos[i],
            "Enquadrado?": enquadrado,
            "Atual (R$)": round(atual_valor, 2),
            "Sugerido (R$)": round(sugerido_valor, 2),
            "Delta (R$)": round(delta_valor, 2)
        })

    df_resultado = pd.DataFrame(resultado)

    st.write("### 📋 Plano de Realocação com Restrições")
    st.dataframe(df_resultado)

    # Gráfico 1 — Alocação Atual vs Sugerida com Enquadramento
    st.write("### 📈 Alocação Atual vs Sugerida (com Faixa Permitida)")

    fig1 = go.Figure()
    for idx, row in df_resultado.iterrows():
        fig1.add_trace(go.Scatter(
            x=[row['Min (%)'], row['Max (%)']],
            y=[row['Classe'], row['Classe']],
            mode='lines',
            line=dict(color='lightgray', width=10),
            showlegend=False
        ))
        fig1.add_trace(go.Scatter(
            x=[row['Atual (%)']],
            y=[row['Classe']],
            mode='markers',
            marker=dict(color='red', size=12, symbol='circle'),
            name='Atual'
        ))
        fig1.add_trace(go.Scatter(
            x=[row['Sugerido (%)']],
            y=[row['Classe']],
            mode='markers',
            marker=dict(color='green', size=12, symbol='diamond'),
            name='Sugerido'
        ))

    fig1.update_layout(
        xaxis_title='Alocação (%)',
        yaxis_title='Classe de Ativo',
        yaxis=dict(categoryorder='array', categoryarray=df_resultado['Classe'][::-1]),
        height=400,
        margin=dict(l=100, r=40, t=60, b=40),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )

    st.plotly_chart(fig1, use_container_width=True)

    # Gráfico 2 — Delta com zero centralizado
    st.write("### 🔁 Variação da Alocação (Delta %)")

    fig2 = px.bar(
        df_resultado,
        x="Delta (%)",
        y="Classe",
        orientation='h',
        color="Ação",
        color_discrete_map={"Comprar": "green", "Vender": "red", "Manter": "gray"},
        title="Delta de Alocação por Classe"
    )

    fig2.update_layout(
        xaxis_title="Delta (%) (Sugerido - Atual)",
        yaxis_title="Classe",
        xaxis=dict(zeroline=True, range=[-100, 100]),
        height=400,
        margin=dict(l=80, r=40, t=50, b=40)
    )

    st.plotly_chart(fig2, use_container_width=True)
