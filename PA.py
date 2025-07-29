import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Rebalanceamento de Carteira", layout="wide")
st.title("Rebalanceamento de Carteira")



modo_input = st.radio("Como você quer informar sua carteira?", ["Porcentagem (%)", "Valor (R$)"])

num_classes = st.number_input("Determinar quantidade de Classes", 1, 20, 2, step=1, key="num_classes")

classes = []
aloc_atual = []
aloc_otima = []
fixos = []
nao_vender_flags = []
minimos = []
maximos = []

st.write("### Preencha os dados de cada classe:")
col1, col2 = st.columns([3,2])
with col1:
    st.write("#### Asset Allocation")
with col2:
    st.write("#### Suitability")

for i in range(int(num_classes)):
    col1, col2 = st.columns([3, 2])
    with col1:
        nome = st.text_input(f"Classe Nome {i+1}", value=st.session_state.get(f"nome_{i}", f"Classe {i+1}"), key=f"nome_{i}")
        atual = st.number_input(
            f"{'Atual (%)' if modo_input == 'Porcentagem (%)' else 'Atual (R$)'} - Classe {i+1}",
            min_value=0.0,
            format="%.2f",
            key=f"atual_{i}"
        )
        otima = st.number_input(f"Alocacao Otima (%) - Classe {i+1}", 0.0, 100.0, format="%.2f", key=f"otima_{i}")
    with col2:
        fixo = st.number_input(f"Fixo (%) - Classe {i+1}", 0.0, 100.0, format="%.2f", key=f"fixo_{i}")
        minimo = st.number_input(f"Minimo (%) - Classe {i+1}", 0.0, 100.0, format="%.2f", key=f"minimo_{i}")
        maximo = st.number_input(f"Maximo (%) - Classe {i+1}", 0.0, 100.0, format="%.2f", key=f"maximo_{i}")
        nao_vender = st.checkbox(f"Não vender - Classe {i+1}", key=f"nao_vender_{i}")

    classes.append(nome)
    aloc_atual.append(atual)
    aloc_otima.append(otima)
    fixos.append(fixo)
    nao_vender_flags.append(nao_vender)
    minimos.append(minimo)
    maximos.append(maximo)

# Aporte adicional (opcional)
aporte = st.number_input("Aporte adicional (R$)", min_value=0.0, format="%.2f", key="aporte")

# Validações e preparos
st.markdown("### Validacões")
aloc_atual_pct = None

if modo_input == "Porcentagem (%)":
    soma_atual_pct = sum(aloc_atual)
    st.write(f"**Soma das Alocações Informadas:** {round(soma_atual_pct, 2)}%")
    if abs(soma_atual_pct - 100) > 1e-3:
        st.warning("A soma das porcentagens alocadas não é 100%.")
    else:
        aloc_atual_pct = aloc_atual
        total_base = 100.0
else:
    total_valor = sum(aloc_atual)
    st.write(f"**Patrimônio Líquido (PL) Informado:** R$ {total_valor:,.2f}")
    if total_valor == 0:
        st.warning("O valor total da carteira deve ser maior que zero.")
    else:
        total_com_aporte = total_valor + aporte
        aloc_atual_pct = [v / total_com_aporte * 100 for v in aloc_atual]
        total_base = total_com_aporte

# Cálculo principal
if aloc_atual_pct and ((modo_input == "Valor (R$)") or (modo_input == "Porcentagem (%)" and abs(sum(aloc_atual) - 100) < 1e-3)) and st.button("Calcular Realocação"):
    resultado = []

    travado_classes = [i for i in range(int(num_classes)) if nao_vender_flags[i] and aloc_atual_pct[i] > aloc_otima[i]]
    travado_pct = sum([aloc_atual_pct[i] for i in travado_classes])
    restante_pct = 100.0 - travado_pct
    soma_otima_travada = sum([aloc_otima[i] for i in travado_classes])

    otima_ajustada = []
    for i in range(int(num_classes)):
        if i in travado_classes:
            otima_ajustada.append(aloc_atual_pct[i])
        else:
            if (100.0 - soma_otima_travada) == 0:
                otima_ajustada.append(aloc_atual_pct[i])
            else:
                otima_ajustada.append(aloc_otima[i] / (100.0 - soma_otima_travada) * restante_pct)

    for i in range(int(num_classes)):
        atual_pct = aloc_atual_pct[i]
        otimo_pct = otima_ajustada[i]
        fixo_pct = min(fixos[i], atual_pct)

        atual_variavel = atual_pct - fixo_pct
        otimo_variavel = otimo_pct - fixo_pct
        sugerido_pct = fixo_pct + otimo_variavel

        if nao_vender_flags[i] and sugerido_pct < atual_pct:
            sugerido_pct = atual_pct
            otimo_variavel = atual_pct - fixo_pct

        delta_total = sugerido_pct - atual_pct
        atual_valor = atual_pct / 100 * total_base
        sugerido_valor = sugerido_pct / 100 * total_base
        delta_valor = sugerido_valor - atual_valor

        sugerido_pct_check = round(sugerido_pct, 4)
        enquadrado = "Sim" if (sugerido_pct_check >= minimos[i] and sugerido_pct_check <= maximos[i]) else "Não"

        resultado.append({
            "Classe": classes[i],
            "Atual (%)": round(atual_pct, 2),
            "Min (%)": minimos[i],
            "Ótima (%)": round(aloc_otima[i], 2),
            "Max (%)": maximos[i],
            "Fixo (%)": round(fixo_pct, 2),
            "Sugerido (%)": round(sugerido_pct, 2),
            "Delta (%)": round(delta_total, 2),
            "Ação": "Comprar" if delta_valor > 0 else "Vender" if delta_valor < 0 else "Manter",
            "Enquadrado?": enquadrado,
            "Atual (R$)": round(atual_valor, 2),
            "Sugerido (R$)": round(sugerido_valor, 2),
            "Delta (R$)": round(delta_valor, 2)
        })

    df_resultado = pd.DataFrame(resultado)

    st.write("### Plano de Realocação com Restrições")
    st.dataframe(df_resultado, use_container_width=True)

    # Os gráficos continuam como no código anterior...

    st.write("### Alocação Atual vs Sugerida (com Faixa Permitida)")
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

    st.write("### Variação da Alocação (Delta %)")
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

    st.write("### Radar de Alocação por Classe")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatterpolar(
        r=df_resultado["Atual (%)"],
        theta=df_resultado["Classe"],
        fill='toself',
        name='Atual',
        line_color='red'
    ))
    fig3.add_trace(go.Scatterpolar(
        r=df_resultado["Sugerido (%)"],
        theta=df_resultado["Classe"],
        fill='toself',
        name='Sugerido',
        line_color='green'
    ))
    fig3.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(df_resultado["Max (%)"].max(), 100)])),
        title="Radar de Alocação Atual vs Sugerida",
        showlegend=True,
        height=500
    )
    st.plotly_chart(fig3, use_container_width=True)
