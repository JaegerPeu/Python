import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.title("Atingir meta com Progressão Aritmética")

st.markdown(
    """
    Defina:
    - Valor inicial \(a_1\)  
    - Meta da soma \(S_n\)  
    - Número de períodos \(n\)  

    O app calcula a razão \(r\) e mostra a evolução do saldo acumulado.
    """
)

# Sliders
a1 = st.slider("Valor inicial (a₁)", min_value=0.0, max_value=1000.0, value=1000.0, step=50.0)
Sn_meta = st.slider("Meta da soma (Sₙ)", min_value=0.0, max_value=50000.0, value=50000.0, step=100.0)
n = st.slider("Número de períodos (n)", min_value=2, max_value=365, value=12, step=1)

if n <= 1:
    st.error("n deve ser maior ou igual a 2 para calcular a razão.")
else:
    # calcula razão r
    r = ((2 * Sn_meta / n) - 2 * a1) / (n - 1)

    st.subheader("Resultado do cálculo")
    st.write(f"Razão necessária r: **{r:,.4f}**")

    # sequência da PA
    k_vals = np.arange(1, n + 1)
    termos = a1 + (k_vals - 1) * r
    soma_parcial = np.cumsum(termos)

    df = pd.DataFrame({
        "Período": k_vals,
        "Termo": termos,
        "Saldo acumulado": soma_parcial
    })

    st.subheader("Tabela da progressão")
    st.dataframe(
        df.style.format(
            {"Termo": "{:,.2f}", "Saldo acumulado": "{:,.2f}"}
        ),
        use_container_width=True
    )

    st.subheader("Evolução do saldo acumulado")

    fig = px.bar(
        df,
        x="Período",
        y="Saldo acumulado",
        title="Saldo acumulado por período",
        text="Saldo acumulado"
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(xaxis_title="Período", yaxis_title="Saldo acumulado")

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Verificação")
    st.write(f"Soma final Sₙ calculada: **{soma_parcial[-1]:,.2f}** (meta: {Sn_meta:,.2f})")
