import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================================================
# CONFIGURAÇÃO GLOBAL
# =========================================================
st.set_page_config(page_title="Monte Carlo Core", layout="wide")

st.title("Monte Carlo Core – Motor de Cálculo")

# =========================================================
# PARÂMETROS GLOBAIS
# =========================================================
st.sidebar.header("Parâmetros globais")

n_sims = st.sidebar.slider("Nº simulações", 1000, 50000, 10000, 1000)
n_periodos = st.sidebar.slider("Horizonte (meses)", 12, 360, 120)

capital_inicial = st.sidebar.number_input(
    "Capital inicial",
    min_value=0.0,
    max_value=1e9,
    value=10000.0,
    step=1000.0,
)

aporte_mensal = st.sidebar.number_input(
    "Aporte mensal",
    min_value=0.0,
    max_value=1e9,
    value=1000.0,
    step=100.0,
)

st.sidebar.markdown("---")
show_mean = st.sidebar.checkbox("Mostrar linha média", True)
show_p5 = st.sidebar.checkbox("Mostrar P5", True)
show_p95 = st.sidebar.checkbox("Mostrar P95", True)

# =========================================================
# FUNÇÕES CORE
# =========================================================
def sim_ativo_unico(mu_anual, sigma_anual, n_periodos, n_sims, capital_inicial, aporte_mensal):
    """
    Monte Carlo de 1 ativo com aportes mensais.
    mu_anual e sigma_anual em base anual (ex: 0.12 = 12%).
    """
    dt = 1 / 12
    mu = mu_anual * dt
    sigma = sigma_anual * np.sqrt(dt)

    retornos = np.random.normal(mu, sigma, (n_periodos, n_sims))
    valores = np.zeros((n_periodos + 1, n_sims))
    valores[0] = capital_inicial

    for t in range(1, n_periodos + 1):
        valores[t] = (valores[t - 1] + aporte_mensal) * (1 + retornos[t - 1])

    return valores, retornos


def sim_carteira(mu_anual_vec, sigma_anual_vec, corr_matrix, pesos_vec,
                 n_periodos, n_sims, capital_inicial, aporte_mensal):
    """
    Motor robusto de Monte Carlo para carteira:

    mu_anual_vec   : (n_assets,)  retornos médios anuais
    sigma_anual_vec: (n_assets,)  volatilidades anuais
    corr_matrix    : (n_assets,n_assets) matriz de correlação
    pesos_vec      : (n_assets,) pesos da carteira (somam 1)
    """
    mu_anual_vec = np.asarray(mu_anual_vec, dtype=float)
    sigma_anual_vec = np.asarray(sigma_anual_vec, dtype=float)
    corr_matrix = np.asarray(corr_matrix, dtype=float)
    pesos_vec = np.asarray(pesos_vec, dtype=float)

    n_assets = len(mu_anual_vec)
    dt = 1 / 12

    # μ mensal
    mu_vec = mu_anual_vec * dt

    # cov anual = D * Corr * D, D = diag(sigma)
    D = np.diag(sigma_anual_vec)
    cov_anual = D @ corr_matrix @ D

    # cov mensal
    Sigma = cov_anual * dt

    # garantir matriz definida positiva para Cholesky
    eps = 1e-8
    Sigma_pd = Sigma + eps * np.eye(n_assets)
    try:
        L = np.linalg.cholesky(Sigma_pd)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(Sigma)
        eigvals_clipped = np.clip(eigvals, eps, None)
        Sigma_pd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
        L = np.linalg.cholesky(Sigma_pd)

    Z = np.random.normal(0, 1, (n_periodos, n_sims, n_assets))
    retornos_port = np.zeros((n_periodos, n_sims))
    retornos_ativos = np.zeros((n_periodos, n_sims, n_assets))

    for t in range(n_periodos):
        shocks = np.tensordot(Z[t], L, axes=1)   # (n_sims, n_assets)
        ret_ind = mu_vec + shocks                # (n_sims, n_assets)
        retornos_ativos[t] = ret_ind
        ret_port = np.sum(pesos_vec * ret_ind, axis=1)
        retornos_port[t] = ret_port

    valores = np.zeros((n_periodos + 1, n_sims))
    valores[0] = capital_inicial
    for t in range(1, n_periodos + 1):
        valores[t] = (valores[t - 1] + aporte_mensal) * (1 + retornos_port[t - 1])

    return valores, retornos_port, retornos_ativos, cov_anual


def plot_traj_hist(valores, titulo):
    """
    Gráfico básico: trajetórias + histograma de valor final.
    """
    final_values = valores[-1]

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.08,
        subplot_titles=("Trajetórias simuladas", "Distribuição do valor final"),
    )

    # Trajetórias
    for i in range(min(50, valores.shape[1])):
        fig.add_trace(
            go.Scatter(
                x=list(range(valores.shape[0])),
                y=valores[:, i],
                mode="lines",
                line=dict(width=1, color="rgba(0,0,150,0.2)"),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    if show_mean:
        fig.add_trace(
            go.Scatter(
                x=list(range(valores.shape[0])),
                y=np.mean(valores, axis=1),
                name="Média",
                line=dict(color="blue", width=3),
            ),
            row=1,
            col=1,
        )
    if show_p5:
        fig.add_trace(
            go.Scatter(
                x=list(range(valores.shape[0])),
                y=np.percentile(valores, 5, axis=1),
                name="P5",
                line=dict(color="red", width=2, dash="dash"),
            ),
            row=1,
            col=1,
        )
    if show_p95:
        fig.add_trace(
            go.Scatter(
                x=list(range(valores.shape[0])),
                y=np.percentile(valores, 95, axis=1),
                name="P95",
                line=dict(color="green", width=2, dash="dash"),
            ),
            row=1,
            col=1,
        )

    # Histograma valor final
    fig.add_trace(
        go.Histogram(
            x=final_values,
            nbinsx=40,
            marker_color="steelblue",
            name="Valor final",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(title=titulo, height=700, showlegend=True)
    fig.update_xaxes(title_text="Período (meses)", row=1, col=1)
    fig.update_yaxes(title_text="Valor", row=1, col=1)
    fig.update_xaxes(title_text="Valor final", row=2, col=1)

    return fig

# =========================================================
# LAYOUT – ABAS
# =========================================================
tab1, tab2 = st.tabs(["Ativo único", "Carteira"])

# ======================== TAB 1 ==========================
with tab1:
    st.subheader("Monte Carlo – ativo único")

    c1, c2 = st.columns(2)
    with c1:
        mu_anual = st.number_input(
            "Retorno médio anual (%)",
            min_value=-100.0,
            max_value=200.0,
            value=12.0,
        ) / 100
    with c2:
        sigma_anual = st.number_input(
            "Volatilidade anual (%)",
            min_value=0.0,
            max_value=500.0,
            value=20.0,
        ) / 100

    if st.button("Simular ativo único"):
        valores, retornos = sim_ativo_unico(
            mu_anual,
            sigma_anual,
            n_periodos,
            n_sims,
            capital_inicial,
            aporte_mensal,
        )

        fig = plot_traj_hist(valores, "Simulação Monte Carlo – Ativo único")
        st.plotly_chart(fig, use_container_width=True)

        final_values = valores[-1]
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Média", f"R$ {np.mean(final_values):,.0f}")
        with m2:
            st.metric("Mediana", f"R$ {np.median(final_values):,.0f}")
        with m3:
            st.metric("P5", f"R$ {np.percentile(final_values, 5):,.0f}")
        with m4:
            st.metric("P95", f"R$ {np.percentile(final_values, 95):,.0f}")

# ======================== TAB 2 ==========================
with tab2:
    st.subheader("Monte Carlo – carteira (motor robusto)")

    n_assets = st.number_input("Nº de ativos na carteira", 2, 15, 4)
    n_assets = int(n_assets)
    ativos = [f"Ativo {i+1}" for i in range(n_assets)]

    st.markdown("### μ e σ por ativo")
    mu_list, sigma_list, peso_list = [], [], []
    for i, nome in enumerate(ativos):
        c1, c2, c3 = st.columns(3)
        with c1:
            mu_i = st.number_input(
                f"μ anual {nome} (%)",
                min_value=-100.0,
                max_value=200.0,
                value=10.0 + i * 2,
                key=f"mu_{i}",
            ) / 100
            mu_list.append(mu_i)
        with c2:
            sigma_i = st.number_input(
                f"σ anual {nome} (%)",
                min_value=0.0,
                max_value=500.0,
                value=15.0 + i * 3,
                key=f"sigma_{i}",
            ) / 100
            sigma_list.append(sigma_i)
        with c3:
            peso_i = st.number_input(
                f"Peso {nome}",
                min_value=0.0,
                max_value=1.0,
                value=1.0 / n_assets,
                step=0.05,
                key=f"peso_{i}",
            )
            peso_list.append(peso_i)

    pesos_vec = np.array(peso_list)
    if pesos_vec.sum() > 0:
        pesos_vec = pesos_vec / pesos_vec.sum()
    else:
        pesos_vec = np.ones(n_assets) / n_assets

    st.markdown("### Matriz de correlação")
    st.write("Por simplicidade, começamos com ρ=0.3 fora da diagonal (editável abaixo).")

    corr_matrix = np.full((n_assets, n_assets), 0.3)
    np.fill_diagonal(corr_matrix, 1.0)

    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            corr_ij = st.number_input(
                f"ρ({ativos[i]},{ativos[j]})",
                min_value=-1.0,
                max_value=1.0,
                value=float(corr_matrix[i, j]),
                step=0.05,
                key=f"corr_{i}_{j}",
            )
            corr_matrix[i, j] = corr_ij
            corr_matrix[j, i] = corr_ij

    if st.button("Simular carteira"):
        mu_vec = np.array(mu_list)
        sigma_vec = np.array(sigma_list)

        valores, retornos_port, retornos_ativos, cov_anual = sim_carteira(
            mu_vec,
            sigma_vec,
            corr_matrix,
            pesos_vec,
            n_periodos,
            n_sims,
            capital_inicial,
            aporte_mensal,
        )

        fig2 = plot_traj_hist(valores, "Simulação Monte Carlo – Carteira")
        st.plotly_chart(fig2, use_container_width=True)

        final_values = valores[-1]
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Média", f"R$ {np.mean(final_values):,.0f}")
        with m2:
            st.metric("Mediana", f"R$ {np.median(final_values):,.0f}")
        with m3:
            st.metric("P5", f"R$ {np.percentile(final_values, 5):,.0f}")
        with m4:
            st.metric("P95", f"R$ {np.percentile(final_values, 95):,.0f}")

        st.markdown("#### Covariância anual usada (Σ = D · Corr · D)")
        df_cov = pd.DataFrame(cov_anual, index=ativos, columns=ativos)
        st.dataframe(df_cov)

        st.markdown("#### Matriz de correlação usada")
        df_corr = pd.DataFrame(corr_matrix, index=ativos, columns=ativos)
        st.dataframe(df_corr)
