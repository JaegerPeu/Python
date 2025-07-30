import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import linprog
import plotly.graph_objects as go
import plotly.express as px



st.title("Rebalanceamento de Carteira")

# Seção de seleção de modo de input (Percentual ou Valor)
input_mode = st.radio("Modo de entrada de dados da carteira:", ["Percentual (%)", "Valores (R$)"])

# Entrada do número de classes de ativo
num_classes = st.number_input("Número de classes de ativos na carteira:", min_value=1, step=1, value=3)

# Criação de dataframe para inputs de alocação atual, target, min e max
columns = ["Classe", "Alocação Atual", "Target (%)", "Mínimo (%)", "Máximo (%)"]
data = []
for i in range(int(num_classes)):
    classe_name = f"Classe {i+1}"
    data.append([classe_name, 0.0, 0.0, 0.0, 100.0])
df_input = pd.DataFrame(data, columns=columns)

st.subheader("Dados da Carteira")
st.markdown("Preencha a alocação atual de cada classe, o target (%) e os limites mínimo/máximo (%).")

# ✅ USAR VERSÃO ESTÁVEL:
edited_df = st.data_editor(df_input, num_rows="dynamic")


# Após edição, extrair os valores preenchidos
classes = edited_df["Classe"].tolist()
current_alloc = edited_df["Alocação Atual"].astype(float).tolist()
target_pct = edited_df["Target (%)"].astype(float).tolist()
min_pct = edited_df["Mínimo (%)"].astype(float).tolist()
max_pct = edited_df["Máximo (%)"].astype(float).tolist()

# Entrada do valor de aporte (R$) - se input em % usamos base 100, mas o aporte será considerado em R$ real
aporte = st.number_input("Valor de aporte (R$):", min_value=0.0, step=1.0, value=0.0)

# Seleção do modo de rebalanceamento
mode = st.selectbox("Modo de rebalanceamento:", ["Somente aporte", "Aporte + Rebalanceamento", "Somente rebalanceamento"])

# Converter tudo para um modelo numérico compatível:
# Se input em Percentual, tratar valores atuais como percentuais de um total virtual de 100
# Verifica se usuário está usando modo percentual e quer informar patrimônio real
usar_patrimonio_real = False
valor_patrimonio = 100.0

if input_mode == "Percentual (%)":
    soma_pct = sum(current_alloc)
    if soma_pct == 0:
        st.warning("⚠️ Alocação atual está zerada. Preencha os valores antes de continuar.")
        st.stop()
    
    usar_patrimonio_real = st.checkbox("Deseja informar o valor real do seu patrimônio atual?")
    if usar_patrimonio_real:
        valor_patrimonio = st.number_input("Informe o valor total do patrimônio (R$):", min_value=0.01, step=100.0, value=100000.0)
        total_atual = valor_patrimonio
        current_values = [val * total_atual / 100.0 for val in current_alloc]
    else:
        total_atual = 100.0
        current_values = [val * total_atual / 100.0 for val in current_alloc]
else:
    total_atual = sum(current_alloc)
    if total_atual == 0:
        st.warning("⚠️ O valor total da carteira atual é zero. Preencha antes de continuar.")
        st.stop()
    current_values = current_alloc


# Total final T depende do modo e do tipo de input
if mode == "Somente rebalanceamento":
    aporte_value = 0.0
elif input_mode == "Percentual (%)" and not usar_patrimonio_real:
    # Não considerar aporte se patrimônio real não foi informado
    aporte_value = 0.0
else:
    aporte_value = aporte

T_final = total_atual + aporte_value


# Verificar restrições de limites min/max
sum_min = sum(min_pct)
sum_max = sum(max_pct)
if sum_min > 100:
    st.error(f"⚠️ A soma dos limites mínimos ({sum_min}%) excede 100%. Ajuste os mínimos para que somem no máximo 100%.")
    st.stop()
if sum_max < 100:
    st.error(f"⚠️ A soma dos limites máximos ({sum_max}%) é menor que 100%. Ajuste os máximos para cobrir pelo menos 100%.")
    st.stop()

# Convertir target_pct, min_pct, max_pct para frações (0-1) para cálculos
target_frac = [t/100.0 for t in target_pct]
min_frac = [m/100.0 for m in min_pct]
max_frac = [M/100.0 for M in max_pct]

# Caso especial: se modo somente aporte e a distribuição atual já estiver balanceada (igual ao target)
# então distribuímos o aporte direto em proporção ao target
# Verificação: comparar percentuais atuais vs target (considerando tolerância pequena)
current_pct = []
if total_atual > 0:
    current_pct = [ (cv/total_atual)*100 if total_atual>0 else 0 for cv in current_values ]
else:
    current_pct = [0 for _ in current_values]
balanced = all(abs(current_pct[i] - target_pct[i]) < 1e-6 for i in range(len(target_pct)))
if mode == "Somente aporte" and balanced and aporte_value > 0:
    # Carteira já está balanceada, distribuir aporte conforme target
    final_values = [cv + (target_frac[i] * aporte_value) for i, cv in enumerate(current_values)]
    final_pct = [ (fv / (total_atual+aporte_value)) * 100 if (total_atual+aporte_value)>0 else 0 for fv in final_values ]
    # Nenhuma venda, apenas compras proporcionais ao target
    diffs = [fv - cv for fv, cv in zip(final_values, current_values)]
else:
    # Montar o problema de programação linear e resolver com linprog
    n = len(current_values)
    # Variáveis: x[0..n-1] = alocações finais, x[n..2n-1] = desvios absolutos d_i
    Nvars = 2 * n

    # Vetor de coeficientes do objetivo (c) – minimizar soma de d_i (desvios)
    c = np.zeros(Nvars)
    c[n:2*n] = 1.0  # coef 1 para cada d_i, 0 para x_i

    # Restrições de igualdade (soma das alocações = T_final)
    A_eq = np.zeros((1, Nvars))
    A_eq[0, 0:n] = 1.0
    b_eq = np.array([T_final])

    # Restrições de desigualdade (A_ub * x <= b_ub)
    A_ub = []
    b_ub = []

    # 1. Restrições de valor absoluto: 
    #    x_i - t_i*T <= d_i  =>  (x_i) + (-d_i) <= t_i * T_final
    #    -(x_i - t_i*T) <= d_i  =>  (-x_i) + (-d_i) <= -t_i * T_final
    for i in range(n):
        # x_i - d_i <= t_i * T_final
        row1 = np.zeros(Nvars)
        row1[i] = 1.0    # x_i coef
        row1[n+i] = -1.0  # -d_i coef
        A_ub.append(row1)
        b_ub.append(target_frac[i] * T_final)
        # -x_i - d_i <= -t_i * T_final
        row2 = np.zeros(Nvars)
        row2[i] = -1.0   # -x_i
        row2[n+i] = -1.0  # -d_i
        A_ub.append(row2)
        b_ub.append(- target_frac[i] * T_final)

    # 2. Restrições de "Somente aporte" (se aplicável): x_i >= current_values_i 
    if mode == "Somente aporte":
        for i in range(n):
            # -x_i <= -current_i   (equivalente a x_i >= current_i)
            row = np.zeros(Nvars)
            row[i] = -1.0
            A_ub.append(row)
            b_ub.append(- current_values[i])

    # 3. Limites mínimos e máximos: 
    for i in range(n):
        # x_i <= max_frac[i] * T_final
        row_up = np.zeros(Nvars)
        row_up[i] = 1.0
        A_ub.append(row_up)
        b_ub.append(max_frac[i] * T_final)
        # x_i >= min_frac[i] * T_final  ->  -x_i <= -min*T
        row_low = np.zeros(Nvars)
        row_low[i] = -1.0
        A_ub.append(row_low)
        b_ub.append(- min_frac[i] * T_final)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # Definir bounds (limites) das variáveis:
    bounds = []
    # x_i bounds:
    for i in range(n):
        # x_i >= 0 sempre. 
        # Se Somente aporte, x_i >= current (mas isso já foi tratado em restrição acima),
        # aqui basta x_i >= 0. 
        lower_bound = 0.0
        upper_bound = None  # None = sem limite superior além das restrições explícitas
        bounds.append((lower_bound, upper_bound))
    # d_i bounds:
    for i in range(n):
        bounds.append((0.0, None))  # d_i >= 0, sem limite superior (pode ser até T_final no pior caso)

    # Chamar o solver de Programação Linear (Simplex através do método HiGHS)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res.success:
        # Extrair soluções
        x_solution = res.x[0:n]   # valores finais alocados ótimos
        final_values = list(x_solution)
        final_pct = [(val / T_final) * 100 if T_final>0 else 0 for val in final_values]
        # Diferenças (comprar/vender)
        diffs = [final_values[i] - current_values[i] for i in range(n)]
    else:
        st.error("Não foi possível resolver o problema de otimização (solução inviável ou erro no solver).")
        st.stop()

# Preparar DataFrame de resultado para exibição
# Preparar DataFrame de resultado para exibição
df_result = pd.DataFrame({
    "Classe": classes,
    "Atual (%)": [f"{(cv / total_atual * 100) if total_atual > 0 else 0:.2f}%" for cv in current_values],
    "Target (%)": [f"{t:.2f}%" for t in target_pct],
    "Diferença (%)": [f"{((fv - cv) / total_atual * 100) if total_atual > 0 else 0:+.2f}%" for fv, cv in zip(final_values, current_values)],
    "Final (%)": [f"{(fv / T_final * 100) if T_final > 0 else 0:.2f}%" for fv in final_values]
})

# Adicionar colunas em R$ apenas se for aplicável
if input_mode == "Valores (R$)" or (input_mode == "Percentual (%)" and usar_patrimonio_real):
    df_result.insert(2, "Atual (R$)", [f"{val:.2f}" for val in current_values])
    df_result.insert(5, "Diferença (R$)", [f"{diff:+.2f}" for diff in diffs])
    df_result["Final (R$)"] = [f"{val:.2f}" for val in final_values]


st.subheader("Resultado do Rebalanceamento")
st.dataframe(df_result, height=300)

# Interpretar ações de compra/venda a partir de diffs
actions = []
for cls, diff in zip(classes, diffs):
    if abs(diff) < 1e-6:
        action = f"{cls}: **Sem alterações** (mantido)"
    elif diff > 0:
        action = f"{cls}: Comprar/Aportar **R${diff:.2f}**"
    else:
        action = f"{cls}: Vender **R${-diff:.2f}**"
    actions.append(action)
st.markdown("**Operações sugeridas:**")
for act in actions:
    st.write("- " + act)
# Cálculo da Aderência da alocação atual vs target (%)
if len(current_values) >= 1:
    current_pct_array = np.array([(cv / total_atual * 100) if total_atual > 0 else 0 for cv in current_values])
    target_array = np.array(target_pct)

    erro_medio = np.mean(np.abs(current_pct_array - target_array))  # erro absoluto médio
    aderencia = 1 - erro_medio / 100
    aderencia_pct = max(0, min(aderencia * 100, 100))  # limitar entre 0 e 100

    st.markdown(f"**Aderência da alocação atual ao target:** {aderencia_pct:.2f}%")
else:
    st.markdown("**Não há ativos suficientes para calcular aderência.**")
    
    
# st.markdown("---")
# st.subheader("Cálculo de Aporte para Atingir Alocação Ideal")

# if input_mode == "Valores (R$)" and aporte == 0:
#     target_valores = [t / 100 for t in target_pct]
#     soma_target = sum(target_valores)
#     if soma_target > 0:
#         # Recalcular target normalizado
#         target_valores = [t / soma_target for t in target_valores]
#         atual_total = sum(current_values)
#         aportes_necessarios = []
#         for i in range(len(current_values)):
#             valor_ideal = target_valores[i] * atual_total / (1 - target_valores[i]) if target_valores[i] < 1 else float('inf')
#             aporte_ideal = max(0, valor_ideal - current_values[i])
#             aportes_necessarios.append(aporte_ideal)

#         total_aporte_necessario = sum(aportes_necessarios)
#         st.markdown(f"📌 **Aporte necessário estimado para atingir o target (sem rebalanceamento): R$ {total_aporte_necessario:,.2f}**")

#         for cls, valor in zip(classes, aportes_necessarios):
#             if valor > 0:
#                 st.write(f"- {cls}: Aportar R$ {valor:,.2f}")
#             else:
#                 st.write(f"- {cls}: Nenhum aporte necessário")
#     else:
#         st.warning("Os targets estão zerados ou inválidos.")


# Gráfico tipo velocímetro (gauge) para visualização da aderência

fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=aderencia_pct,
    domain={'x': [0, 1], 'y': [0, 1]},
    title={'text': "Aderência Atual vs Target (%)"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "green"},
        'steps': [
            {'range': [0, 50], 'color': 'red'},
            {'range': [50, 80], 'color': 'yellow'},
            {'range': [80, 100], 'color': 'lightgreen'}
        ],
        'threshold': {
            'line': {'color': "black", 'width': 4},
            'thickness': 0.75,
            'value': aderencia_pct
        }
    }
))

st.plotly_chart(fig_gauge, use_container_width=True)


# Conversão dos valores para float
df_result["Atual (%)"] = df_result["Atual (%)"].str.replace('%', '').astype(float)
df_result["Sugerido (%)"] = df_result["Final (%)"].str.replace('%', '').astype(float)
df_result["Min (%)"] = min_pct
df_result["Max (%)"] = max_pct
df_result["Ação"] = [
    "Comprar" if d > 0 else "Vender" if d < 0 else "Manter"
    for d in diffs
]
df_result["Delta (%)"] = df_result["Sugerido (%)"] - df_result["Atual (%)"]

# Gráfico 1 — Faixa de enquadramento
st.write("### 📈 Alocação Atual vs Sugerida (com Enquadramento)")
fig1 = go.Figure()
for idx, row in df_result.iterrows():
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
    yaxis=dict(categoryorder='array', categoryarray=df_result['Classe'][::-1]),
    height=400,
    margin=dict(l=100, r=40, t=60, b=40),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
)
st.plotly_chart(fig1, use_container_width=True)

# Gráfico 2 — Delta
st.write("### 🔁 Variação da Alocação (Delta %)")
fig2 = px.bar(
    df_result,
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

# Gráfico 3 — Radar
st.write("### 🧭 Radar de Asset Allocation")
fig3 = go.Figure()
fig3.add_trace(go.Scatterpolar(
    r=df_result["Atual (%)"],
    theta=df_result["Classe"],
    fill='toself',
    name='Atual',
    line_color='red'
))
fig3.add_trace(go.Scatterpolar(
    r=df_result["Sugerido (%)"],
    theta=df_result["Classe"],
    fill='toself',
    name='Sugerido',
    line_color='green'
))
fig3.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, max(df_result["Max (%)"].max(), 100)])),
    title="Radar de Alocação Atual vs Sugerida",
    showlegend=True,
    height=500
)
st.plotly_chart(fig3, use_container_width=True)
