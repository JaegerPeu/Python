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
    "Períodos por ano (só p/ converter RF a.a.)",
    min_value=1, max_value=10_000, value=252, step=1,
    help="Ex.: 252 (diário), 12 (mensal). Usado apenas para converter RF a.a. para RF por período."
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

target_vol_pct = None  # vol por período (%)
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
# 2) Retornos / média / cov (POR PERÍODO)
# =========================================================
ri = prec / prec.shift(1) - 1
ri = ri.dropna()

mu = ri.mean().values       # por período
S = ri.cov().values         # por período

def port_ret(w): return float(np.dot(w, mu))              # por período (decimal)
def port_vol(w): return float(np.sqrt(w.T @ S @ w))       # por período (decimal)
def obj_var(w):  return float(w.T @ S @ w)                # variância por período

# =========================================================
# 3) Suitability: bounds por ativo em %
# =========================================================
st.sidebar.subheader("Suitability: limites por ativo (%)")

if "bounds_df_pct" not in st.session_state:
    st.session_state["bounds_df_pct"] = pd.DataFrame(
        {"min_%": np.zeros(n), "max_%": 100 * np.ones(n)},
        index=ativo
    )

bounds_df_pct = st.sidebar.data_editor(st.session_state["bounds_df_pct"], use_container_width=True).astype(float)

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
    diff = 1.0 - w.sum()
    j = int(np.argmax(ub - w))
    w[j] = np.minimum(ub[j], w[j] + diff)
    return w

x0 = feasible_start()

# =========================================================
# 4) RF a.a. -> RF por período (para Sharpe)
# =========================================================
rf_aa = rf_aa_pct / 100
rf_per = (1.0 + rf_aa) ** (1.0 / float(periodos_ano)) - 1.0  # por período

# =========================================================
# 5) Min/max retorno viável + GMV robusto
# =========================================================
def solve_minmax_return(sign=+1):
    obj = lambda w: sign * port_ret(w)
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

ret_min = port_ret(res_minR.x)
ret_max = port_ret(res_maxR.x)

np.random.seed(int(seed))

def sample_feasible_weights(k, max_tries=500_000):
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
            obj_var, w0, method="SLSQP",
            bounds=bounds, constraints=cons_sum1,
            options={"maxiter": 10000, "ftol": 1e-14}
        )
        if res.success:
            v = port_vol(res.x)
            if (best_vol is None) or (v < best_vol):
                best_vol = v
                best_res = res
    return best_res

res_gmv = solve_gmv_multistart(n_tries=40)
if (res_gmv is None) or (not res_gmv.success):
    st.error("Falha ao achar GMV (mínima variância).")
    st.stop()

w_gmv = res_gmv.x
ret_gmv = port_ret(w_gmv)
vol_gmv = port_vol(w_gmv)

# =========================================================
# 6) Fronteira (por período)
# =========================================================
faixa_ret = np.linspace(ret_min, ret_max, n_front)
faixa_ret = np.unique(np.sort(np.append(faixa_ret, ret_gmv)))

risk_curve, ret_curve, w_curve = [], [], []
x_front = x0.copy()

for alvo in faixa_ret:
    cons = (
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'eq', 'fun': lambda w, r=alvo: port_ret(w) - r},
    )
    res = solver.minimize(
        port_vol, x_front, method="SLSQP",
        bounds=bounds, constraints=cons,
        options={"maxiter": 6000, "ftol": 1e-12}
    )
    if res.success:
        risk_curve.append(res.fun)
        ret_curve.append(alvo)
        w_curve.append(res.x)
        x_front = res.x

risk_curve = np.array(risk_curve, dtype=float)
ret_curve = np.array(ret_curve, dtype=float)
w_curve = np.array(w_curve, dtype=float)

mask_ef = ret_curve >= ret_gmv
risk_ef, ret_ef = risk_curve[mask_ef], ret_curve[mask_ef]

# =========================================================
# 7) Risco-meta (% por período)
# =========================================================
if opt_mode == "Risco-meta (vol <= alvo) e maximiza retorno":
    vmin = float(np.min(risk_curve)) if len(risk_curve) else float(vol_gmv)
    vmax = float(np.max(risk_curve)) if len(risk_curve) else float(max(vmin * 3, vmin + 1e-6))
    target_vol_pct = st.sidebar.slider(
        "Risco-meta (% por período)",
        min_value=float(vmin * 100),
        max_value=float(vmax * 100),
        value=float(vol_gmv * 100),
        step=float(max((vmax - vmin) * 100 / 200, 0.01))
    )

target_vol = (target_vol_pct / 100) if (target_vol_pct is not None) else None

# =========================================================
# 8) Carteira otimizada
# =========================================================
def solve_user_portfolio():
    if opt_mode == "Minimizar volatilidade (GMV)":
        return True, w_gmv, "GMV"

    if opt_mode == "Maximizar Sharpe (RF a.a. convertida)":
        def neg_sharpe(w):
            v = port_vol(w)
            if v <= 1e-12:
                return 1e9
            return -((port_ret(w) - rf_per) / v)

        res = solver.minimize(
            neg_sharpe, x0, method="SLSQP",
            bounds=bounds, constraints=cons_sum1,
            options={"maxiter": 10000, "ftol": 1e-12}
        )
        return res.success, res.x, "MaxSharpe"

    if opt_mode == "Risco-meta (vol <= alvo) e maximiza retorno":
        cons = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': lambda w, tv=target_vol: tv - port_vol(w)},
        )
        res = solver.minimize(
            lambda w: -port_ret(w), x0, method="SLSQP",
            bounds=bounds, constraints=cons,
            options={"maxiter": 10000, "ftol": 1e-12}
        )
        return res.success, res.x, "TargetVol"

    return False, None, "?"

ok, w_opt, tag = solve_user_portfolio()
if not ok:
    st.error("A otimização não convergiu com os bounds atuais. Tente relaxar limites ou mudar objetivo.")
    st.stop()

ret_opt = port_ret(w_opt)
vol_opt = port_vol(w_opt)

# Sharpe por período e (opcional) anualizado
shp_per = ((ret_opt - rf_per) / vol_opt) if (vol_opt > 1e-12) else np.nan
shp_aa = shp_per * np.sqrt(float(periodos_ano)) if np.isfinite(shp_per) else np.nan

# =========================================================
# 9) Métricas (2 linhas)
# =========================================================
st.title("Markowitz")

r1a, r1b = st.columns(2)
r1a.metric("Retorno mínimo viável (% por período)", f"{ret_min:.2%}")
r1b.metric("Retorno máximo viável (% por período)", f"{ret_max:.2%}")

r2a, r2b = st.columns(2)
r2a.metric("GMV (ret, vol) por período", f"{ret_gmv:.2%} | {vol_gmv:.2%}")
r2b.metric("Otimizada (ret, vol) por período", f"{ret_opt:.2%} | {vol_opt:.2%}")

st.subheader(f"Pesos da carteira otimizada ({opt_mode})")
st.dataframe(
    (pd.Series(w_opt, index=ativo) * 100).round(2).to_frame("w (%)"),
    use_container_width=True
)

if opt_mode == "Maximizar Sharpe (RF a.a. convertida)":
    st.caption(
        f"RF: {rf_aa_pct:.2f}% a.a.  ⇒  {rf_per:.4%} por período | "
        f"Sharpe: {shp_per:.4f} (por período), {shp_aa:.4f} (anualizado)"
    )

# =========================================================
# 10) Monte Carlo viável (para plot)
# =========================================================
np.random.seed(int(seed))
W = sample_feasible_weights(n_mc)
mc_ret = np.array([port_ret(w) for w in W], dtype=float)
mc_vol = np.array([port_vol(w) for w in W], dtype=float)

# =========================================================
# 11) Plotly com hover (tudo por período)
# =========================================================
fig = go.Figure()

if len(mc_vol) > 0:
    fig.add_trace(go.Scatter(
        x=mc_vol, y=mc_ret,
        mode="markers",
        name=f"Monte Carlo viável ({len(mc_vol)})",
        marker=dict(size=6, color="black", opacity=0.30),
        hovertemplate="Vol=%{x:.2%}<br>Ret=%{y:.2%}<extra></extra>"
    ))

if len(risk_curve) > 0:
    weights_hover = np.array([
        "<br>".join([f"{ativo[i]}={w[i]*100:.2f}%" for i in range(n)])
        for w in w_curve
    ], dtype=object)

    fig.add_trace(go.Scatter(
        x=risk_curve, y=ret_curve,
        mode="lines+markers",
        name="Curva completa (min risco p/ retorno)",
        marker=dict(size=5, color="red"),
        line=dict(width=3, color="red"),
        customdata=weights_hover,
        hovertemplate="Vol=%{x:.2%}<br>Ret=%{y:.2%}<br>%{customdata}<extra></extra>"
    ))

if len(risk_ef) > 0:
    fig.add_trace(go.Scatter(
        x=risk_ef, y=ret_ef,
        mode="lines",
        name="Fronteira eficiente (>= GMV)",
        line=dict(width=4, color="dodgerblue"),
        hovertemplate="Vol=%{x:.2%}<br>Ret=%{y:.2%}<extra></extra>"
    ))

fig.add_trace(go.Scatter(
    x=[vol_gmv], y=[ret_gmv],
    mode="markers",
    name="GMV (mínima volatilidade)",
    marker=dict(size=14, color="green"),
    hovertemplate="GMV<br>Vol=%{x:.2%}<br>Ret=%{y:.2%}<extra></extra>"
))

fig.add_trace(go.Scatter(
    x=[vol_opt], y=[ret_opt],
    mode="markers",
    name=f"Otimizada: {tag}",
    marker=dict(size=16, color="gold", line=dict(width=1, color="black"), symbol="star"),
    hovertemplate="Otimizada<br>Vol=%{x:.2%}<br>Ret=%{y:.2%}<extra></extra>"
))

fig.update_layout(
    height=700,
    xaxis=dict(title="Risco (Volatilidade) % por período", tickformat=".2%"),
    yaxis=dict(title="Retorno Esperado % por período", tickformat=".2%"),
    hovermode="closest",
    legend=dict(orientation="h"),
)

st.plotly_chart(fig, use_container_width=True)
