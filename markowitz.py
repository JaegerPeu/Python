# app.py
import numpy as np
import pandas as pd
import streamlit as st
import scipy.optimize as solver
import plotly.graph_objects as go

st.set_page_config(page_title="Markowitz - Fronteira Eficiente", layout="wide")

# =========================================================
# Sidebar
# =========================================================
st.sidebar.title("Parâmetros")

periodos_ano = st.sidebar.number_input(
    "Períodos por ano (anualização + RF)",
    min_value=1, max_value=10_000, value=252, step=1,
    help="Ex.: 252 (diário), 12 (mensal)."
)

seed = st.sidebar.number_input("Seed (Monte Carlo)", min_value=0, max_value=10_000, value=7, step=1)
n_mc = st.sidebar.slider("Monte Carlo (pontos)", 100, 50_000, 3000, 100)
n_front = st.sidebar.slider("Pontos na curva", 30, 400, 140, 10)

st.sidebar.subheader("Otimização")
opt_mode = st.sidebar.selectbox(
    "Escolha o objetivo",
    [
        "Minimizar volatilidade (GMV)",
        "Maximizar Sharpe",
        "Risco-meta (vol <= alvo) e maximiza retorno",
    ],
)

rf_aa_pct = 0.0
if opt_mode == "Maximizar Sharpe":
    rf_aa_pct = st.sidebar.number_input("RF (% a.a.)", value=12.0, step=0.25, format="%.2f")

# =========================================================
# 1) Dados (hard-coded por enquanto)
# =========================================================
ativo = ["a", "b", "c"]
n = len(ativo)

a = np.array([100.00,106.07,108.43,109.51,112.58,115.41,111.53,113.57,115.23,116.22,109.41,108.06,110.88,107.98,108.00,110.92,106.03,107.94,109.79,110.45,113.22,113.10,120.00,116.90,113.93,115.44,115.86,116.35,120.63,121.13,127.05,130.42,131.42,141.65,141.06,144.13,143.11,151.91,154.80,150.62,150.18,149.93,152.01,155.76,158.65,157.75,150.66,143.34,146.91,142.90,140.80,142.27,146.66,141.28,146.85,156.52,163.30,159.69,162.21,159.25,163.53])
b = np.array([80.00,81.90,80.44,73.97,73.60,74.64,76.55,71.77,71.82,67.23,69.84,74.60,67.99,73.57,78.14,77.14,77.87,77.46,79.79,79.79,81.67,73.85,75.97,71.29,77.31,85.37,88.37,80.84,85.68,84.20,95.22,101.75,99.72,110.73,106.53,100.61,109.81,112.52,122.18,123.19,120.13,115.87,117.96,123.80,122.54,125.31,124.32,125.09,124.91,120.93,119.61,109.75,120.19,124.44,122.51,118.56,118.58,113.15,115.22,114.71,109.46])
c = np.array([120.00,116.95,114.79,116.81,113.81,110.95,112.07,116.70,124.23,119.88,119.79,112.53,106.49,114.75,113.76,104.03,113.78,120.40,120.22,112.24,115.76,113.02,106.18,104.15,104.01,109.92,112.37,112.92,107.27,104.70,106.35,105.25,103.56,99.88,100.13,99.72,100.74,96.92,98.52,98.71,104.30,105.55,107.52,110.29,106.93,102.01,97.28,95.40,89.59,88.33,82.63,80.18,84.82,89.41,88.68,87.22,87.12,83.78,85.84,88.74,88.95])

prec = pd.DataFrame({"a": a, "b": b, "c": c})

# =========================================================
# 2) Retornos / média / cov (por período)
# =========================================================
ri = prec / prec.shift(1) - 1
ri = ri.dropna()

mu = ri.mean().values       # por período
S = ri.cov().values         # por período

def port_ret_per(w): return float(np.dot(w, mu))              # por período (decimal)
def port_vol_per(w): return float(np.sqrt(w.T @ S @ w))       # por período (decimal)
def obj_var_per(w):  return float(w.T @ S @ w)                # variância por período

m = float(periodos_ano)

# anualização para EXIBIÇÃO/PLOT (aceita escalar ou array)
def to_ret_aa(x):
    return np.asarray(x, dtype=float) * m

def to_vol_aa(x):
    return np.asarray(x, dtype=float) * np.sqrt(m)

# =========================================================
# 3) Suitability: bounds por ativo em %
# =========================================================
st.sidebar.subheader("Suitability: limites por ativo (%)")

if "bounds_df_pct" not in st.session_state:
    st.session_state["bounds_df_pct"] = pd.DataFrame(
        {"min_%": np.zeros(n), "max_%": 100 * np.ones(n)},
        index=ativo
    )

bounds_df_pct = st.sidebar.data_editor(
    st.session_state["bounds_df_pct"],
    use_container_width=True
).astype(float)

if (bounds_df_pct["min_%"] > bounds_df_pct["max_%"]).any():
    st.error("Há ativos com min_% > max_%. Corrija os limites.")
    st.stop()

sum_lb = (bounds_df_pct["min_%"] / 100).sum()
sum_ub = (bounds_df_pct["max_%"] / 100).sum()
if not (sum_lb <= 1 <= sum_ub):
    st.error(f"Bounds inviáveis: soma(min)={sum_lb:.3f} e soma(max)={sum_ub:.3f}. Precisa permitir soma=1.")
    st.stop()

bounds = [(bounds_df_pct.loc[t, "min_%"] / 100, bounds_df_pct.loc[t, "max_%"] / 100) for t in ativo]
cons_sum1 = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},)

lb = np.array([b[0] for b in bounds], dtype=float)
ub = np.array([b[1] for b in bounds], dtype=float)

def feasible_start():
    w = lb.copy()
    rem = 1.0 - w.sum()
    slack = (ub - lb)
    s = slack.sum()
    if rem < -1e-10 or s <= 1e-12:
        return np.full(n, 1 / n)

    w = w + rem * (slack / s)
    w = np.minimum(np.maximum(w, lb), ub)

    # ajuste final de soma
    diff = 1.0 - w.sum()
    j = int(np.argmax(ub - w))
    w[j] = np.minimum(ub[j], w[j] + diff)
    return w

x0 = feasible_start()

# =========================================================
# 4) RF a.a. -> RF por período (Sharpe)
# =========================================================
rf_aa = rf_aa_pct / 100
rf_per = (1.0 + rf_aa) ** (1.0 / m) - 1.0

# =========================================================
# 5) Min/max retorno viável (por período) + GMV robusto
# =========================================================
def solve_minmax_return(sign=+1):
    obj = lambda w: sign * port_ret_per(w)
    return solver.minimize(
        obj, x0, method="SLSQP",
        bounds=bounds, constraints=cons_sum1,
        options={"maxiter": 6000, "ftol": 1e-12}
    )

res_minR = solve_minmax_return(+1)
res_maxR = solve_minmax_return(-1)
if not (res_minR.success and res_maxR.success):
    st.error("Falha ao achar retorno min/max viável. Bounds apertados ou numérico.")
    st.stop()

ret_min_per = port_ret_per(res_minR.x)
ret_max_per = port_ret_per(res_maxR.x)

np.random.seed(int(seed))

def sample_feasible_weights(k, max_tries=600_000):
    out = []
    rem = 1.0 - lb.sum()
    if rem < 0:
        return np.array(out)

    for _ in range(max_tries):
        y = np.random.dirichlet(np.ones(n))
        w = lb + rem * y
        if np.all(w <= ub + 1e-12) and np.all(w >= lb - 1e-12):
            out.append(w)
            if len(out) >= k:
                break
    return np.array(out)

def solve_gmv_multistart(n_tries=40):
    starts = [x0]
    extra = sample_feasible_weights(max(0, n_tries - 1))
    if len(extra) > 0:
        starts += list(extra[: n_tries - 1])

    best_vol = None
    best_res = None
    for w0 in starts:
        res = solver.minimize(
            obj_var_per, w0, method="SLSQP",
            bounds=bounds, constraints=cons_sum1,
            options={"maxiter": 10000, "ftol": 1e-14}
        )
        if res.success:
            v = port_vol_per(res.x)
            if (best_vol is None) or (v < best_vol):
                best_vol = v
                best_res = res
    return best_res

res_gmv = solve_gmv_multistart(n_tries=40)
if (res_gmv is None) or (not res_gmv.success):
    st.error("Falha ao achar GMV (mínima variância).")
    st.stop()

w_gmv = res_gmv.x
ret_gmv_per = port_ret_per(w_gmv)
vol_gmv_per = port_vol_per(w_gmv)

# =========================================================
# 6) Fronteira (por período) -> exibe anualizada
# =========================================================
faixa_ret_per = np.linspace(ret_min_per, ret_max_per, n_front)
faixa_ret_per = np.unique(np.sort(np.append(faixa_ret_per, ret_gmv_per)))

risk_curve_per, ret_curve_per, w_curve = [], [], []
x_front = x0.copy()

for alvo in faixa_ret_per:
    cons = (
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'eq', 'fun': lambda w, r=alvo: port_ret_per(w) - r},
    )
    res = solver.minimize(
        port_vol_per, x_front, method="SLSQP",
        bounds=bounds, constraints=cons,
        options={"maxiter": 6000, "ftol": 1e-12}
    )
    if res.success:
        risk_curve_per.append(res.fun)
        ret_curve_per.append(alvo)
        w_curve.append(res.x)
        x_front = res.x

risk_curve_per = np.array(risk_curve_per, dtype=float)
ret_curve_per = np.array(ret_curve_per, dtype=float)
w_curve = np.array(w_curve, dtype=float)

mask_ef = ret_curve_per >= ret_gmv_per
risk_ef_per, ret_ef_per = risk_curve_per[mask_ef], ret_curve_per[mask_ef]

risk_curve_aa = to_vol_aa(risk_curve_per)
ret_curve_aa = to_ret_aa(ret_curve_per)
risk_ef_aa = to_vol_aa(risk_ef_per)
ret_ef_aa = to_ret_aa(ret_ef_per)

# =========================================================
# 7) Risco-meta (input em % a.a.) -> converte p/ por período
# =========================================================
target_vol_aa = None
target_vol_per = None

if opt_mode == "Risco-meta (vol <= alvo) e maximiza retorno":
    vmin_aa = float(np.min(risk_curve_aa)) if len(risk_curve_aa) else float(to_vol_aa(vol_gmv_per))
    vmax_aa = float(np.max(risk_curve_aa)) if len(risk_curve_aa) else max(vmin_aa * 3, vmin_aa + 1e-6)

    target_vol_aa_pct = st.sidebar.slider(
        "Risco-meta (% a.a.)",
        min_value=float(vmin_aa * 100),
        max_value=float(vmax_aa * 100),
        value=float(float(to_vol_aa(vol_gmv_per)) * 100),
        step=float(max((vmax_aa - vmin_aa) * 100 / 200, 0.01))
    )
    target_vol_aa = target_vol_aa_pct / 100
    target_vol_per = target_vol_aa / np.sqrt(m)

# =========================================================
# 8) Carteira otimizada (por período) -> exibe anualizada
# =========================================================
def solve_user_portfolio():
    if opt_mode == "Minimizar volatilidade (GMV)":
        return True, w_gmv, "GMV"

    if opt_mode == "Maximizar Sharpe":
        def neg_sharpe(w):
            v = port_vol_per(w)
            if v <= 1e-12:
                return 1e9
            return -((port_ret_per(w) - rf_per) / v)

        res = solver.minimize(
            neg_sharpe, x0, method="SLSQP",
            bounds=bounds, constraints=cons_sum1,
            options={"maxiter": 10000, "ftol": 1e-12}
        )
        return res.success, res.x, "MaxSharpe"

    if opt_mode == "Risco-meta (vol <= alvo) e maximiza retorno":
        cons = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': lambda w, tv=target_vol_per: tv - port_vol_per(w)},
        )
        res = solver.minimize(
            lambda w: -port_ret_per(w), x0, method="SLSQP",
            bounds=bounds, constraints=cons,
            options={"maxiter": 10000, "ftol": 1e-12}
        )
        return res.success, res.x, "TargetVol"

    return False, None, "?"

ok, w_opt, tag = solve_user_portfolio()
if not ok:
    st.error("A otimização não convergiu com os bounds atuais. Tente relaxar limites ou mudar objetivo.")
    st.stop()

ret_opt_per = port_ret_per(w_opt)
vol_opt_per = port_vol_per(w_opt)

ret_min_aa = float(to_ret_aa(ret_min_per))
ret_max_aa = float(to_ret_aa(ret_max_per))

ret_gmv_aa = float(to_ret_aa(ret_gmv_per))
vol_gmv_aa = float(to_vol_aa(vol_gmv_per))

ret_opt_aa = float(to_ret_aa(ret_opt_per))
vol_opt_aa = float(to_vol_aa(vol_opt_per))

# Sharpe por período e anualizado
shp_per = ((ret_opt_per - rf_per) / vol_opt_per) if (vol_opt_per > 1e-12) else np.nan
shp_aa = shp_per * np.sqrt(m) if np.isfinite(shp_per) else np.nan

# =========================================================
# 9) Métricas (2 linhas) em % a.a.
# =========================================================
st.title("Markowitz (retorno e risco anualizados)")

r1a, r1b = st.columns(2)
r1a.metric("Retorno mínimo viável (% a.a.)", f"{ret_min_aa:.2%}")
r1b.metric("Retorno máximo viável (% a.a.)", f"{ret_max_aa:.2%}")

r2a, r2b = st.columns(2)
r2a.metric("GMV (ret, vol) % a.a.", f"{ret_gmv_aa:.2%} | {vol_gmv_aa:.2%}")
r2b.metric("Otimizada (ret, vol) % a.a.", f"{ret_opt_aa:.2%} | {vol_opt_aa:.2%}")

st.subheader(f"Pesos da carteira otimizada ({opt_mode})")
st.dataframe((pd.Series(w_opt, index=ativo) * 100).round(2).to_frame("w (%)"), use_container_width=True)

if opt_mode == "Maximizar Sharpe":
    st.caption(
        f"RF: {rf_aa_pct:.2f}% a.a. ⇒ {rf_per:.4%} por período | "
        f"Sharpe: {shp_per:.4f} (por período), {shp_aa:.4f} (anualizado)"
    )

# =========================================================
# 10) Monte Carlo viável (por período) -> plota anualizado
# =========================================================
np.random.seed(int(seed))
W = sample_feasible_weights(n_mc)

if W.size == 0:
    st.warning("Não consegui gerar pontos Monte Carlo viáveis com esses bounds (muito apertados).")

mc_ret_per = np.array([port_ret_per(w) for w in W], dtype=float) if W.size else np.array([])
mc_vol_per = np.array([port_vol_per(w) for w in W], dtype=float) if W.size else np.array([])

mc_ret_aa = to_ret_aa(mc_ret_per)
mc_vol_aa = to_vol_aa(mc_vol_per)

# =========================================================
# 11) Plotly com hover (eixos em % a.a.)
# =========================================================
fig = go.Figure()

if len(mc_vol_aa) > 0:
    fig.add_trace(go.Scatter(
        x=mc_vol_aa, y=mc_ret_aa,
        mode="markers",
        name=f"Monte Carlo viável ({len(mc_vol_aa)})",
        marker=dict(size=6, color="black", opacity=0.30),
        hovertemplate="Vol a.a.=%{x:.2%}<br>Ret a.a.=%{y:.2%}<extra></extra>"
    ))

if len(risk_curve_aa) > 0:
    weights_hover = np.array([
        "<br>".join([f"{ativo[i]}={w[i]*100:.2f}%" for i in range(n)])
        for w in w_curve
    ], dtype=object)

    fig.add_trace(go.Scatter(
        x=risk_curve_aa, y=ret_curve_aa,
        mode="lines+markers",
        name="Curva completa (min risco p/ retorno)",
        marker=dict(size=5, color="red"),
        line=dict(width=3, color="red"),
        customdata=weights_hover,
        hovertemplate="Vol a.a.=%{x:.2%}<br>Ret a.a.=%{y:.2%}<br>%{customdata}<extra></extra>"
    ))

if len(risk_ef_aa) > 0:
    fig.add_trace(go.Scatter(
        x=risk_ef_aa, y=ret_ef_aa,
        mode="lines",
        name="Fronteira eficiente (>= GMV)",
        line=dict(width=4, color="dodgerblue"),
        hovertemplate="Vol a.a.=%{x:.2%}<br>Ret a.a.=%{y:.2%}<extra></extra>"
    ))

fig.add_trace(go.Scatter(
    x=[vol_gmv_aa], y=[ret_gmv_aa],
    mode="markers",
    name="GMV (mínima volatilidade)",
    marker=dict(size=14, color="green"),
    hovertemplate="GMV<br>Vol a.a.=%{x:.2%}<br>Ret a.a.=%{y:.2%}<extra></extra>"
))

fig.add_trace(go.Scatter(
    x=[vol_opt_aa], y=[ret_opt_aa],
    mode="markers",
    name=f"Otimizada: {tag}",
    marker=dict(size=16, color="gold", line=dict(width=1, color="black"), symbol="star"),
    hovertemplate="Otimizada<br>Vol a.a.=%{x:.2%}<br>Ret a.a.=%{y:.2%}<extra></extra>"
))

fig.update_layout(
    height=700,
    xaxis=dict(title="Risco (Volatilidade) % a.a.", tickformat=".2%"),
    yaxis=dict(title="Retorno Esperado % a.a.", tickformat=".2%"),
    hovermode="closest",
    legend=dict(orientation="h"),
)

st.plotly_chart(fig, use_container_width=True)
