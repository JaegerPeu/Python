import os
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import date

# ----------------------------------------------------
# CONFIG INICIAL E LOGO (fundo preto fixo + fallback)
# ----------------------------------------------------
st.set_page_config(layout="wide", page_title="Dashboard Institucional – Fundos")

PLOT_TEMPLATE = "plotly_white"

# --- CSS limpo e responsivo ---
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .logo-container {
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 25px;
    }
    h1 {
        font-size: 30px !important;
        font-weight: 700;
        margin: 0 0 5px 0;
    }
    .logo-box {
        background-color: #000000;
        padding: 10px 15px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def load_svg(*filenames: str) -> str:
    for fn in filenames:
        try:
            with open(fn, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            continue
    return "<div style='color:#fff;font:12px sans-serif'>LOGO</div>"

svg_logo = load_svg("logo_dark.svg", "logo_light.svg", "logo.svg")

st.markdown(
    f"""
    <div class="logo-container">
        <div class="logo-box" style="width: 180px;">{svg_logo}</div>
        <h1>🏦 Dashboard – Fundos</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------
# CREDENCIAIS LOCAIS (sem st.secrets)
# ----------------------------------------------------
CD_USER = "solutionswm"
CD_PASS = "Soluti%40ns2023"
CD_URL  = "https://api.comdinheiro.com.br/v1/ep1/import-data"

URL_PARAM = (
    "HistoricoIndicadoresFundos001.php%3F%26cnpjs%3D03890892000131%2B38090006000170"
    "%2B49692303000101%2B55597133000189"
    "%2B55753904000180%2B57499088000155%2B57565778000165%2B57682222000159%2B60334542000122%2B60800845000193_unica"
    "%26data_ini%3D13112023%26data_fim%3D31129999%26indicadores%3Dnome_fundo%2Bpatrimonio%2Bcaptacao%2Bresgate%2Bforma%2Bexclusivo"
    "%26op01%3Dtabela_v%26num_casas%3D2%26enviar_email%3D0%26periodicidade%3Ddiaria%26cabecalho_excel%3Dmodo3"
    "%26transpor%3D0%26asc_desc%3Ddesc%26tipo_grafico%3Dlinha%26relat_alias_automatico%3Dcmd_alias_01"
)

# ----------------------------------------------------
# HELPERS
# ----------------------------------------------------
def _fmt_brl(v):
    if pd.isna(v): return "-"
    return f"R$ {v:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")

def _fmt_pct(v):
    if pd.isna(v): return "-"
    return f"{v:+.1f}%"

# ----------------------------------------------------
# FETCH + CACHE
# ----------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_raw():
    payload = f"username={CD_USER}&password={CD_PASS}&URL={URL_PARAM}&format=json3"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    r = requests.post(CD_URL, data=payload, headers=headers, timeout=60)
    r.raise_for_status()
    data = r.json()

    table_key = list(data["tables"].keys())[0]
    rows = data["tables"][table_key]

    recs = []
    for v in rows.values():
        if v.get("col0"):
            recs.append(
                {
                    "Data": v["col0"],
                    "Fundo": v["col1"],
                    "Patrimonio": v.get("col2", "").replace(",", ".") if v.get("col2") else None,
                    "Captacao": v.get("col3", "").replace(",", ".") if v.get("col3") else None,
                    "Resgate": v.get("col4", "").replace(",", ".") if v.get("col4") else None,
                    "Forma": v.get("col5", "").strip().capitalize() if v.get("col5") else None,
                    "Exclusivo": v.get("col6", "").strip().capitalize() if v.get("col6") else None,
                }
            )
    df = pd.DataFrame(recs)
    df["Data"] = pd.to_datetime(df["Data"], format="%d/%m/%Y", errors="coerce")
    for c in ["Patrimonio", "Captacao", "Resgate"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Forma" not in df.columns: df["Forma"] = None
    if "Exclusivo" not in df.columns: df["Exclusivo"] = None
    df = df.dropna(subset=["Data", "Fundo"]).sort_values(["Fundo", "Data"])
    return df

# ----------------------------------------------------
# AGREGAÇÃO MENSAL
# ----------------------------------------------------
@st.cache_data
def slice_period_monthly(df, start, end):
    dfd = df[(df["Data"] >= start) & (df["Data"] <= end)].copy()
    if dfd.empty:
        return dfd.assign(AnoMes=pd.PeriodIndex([], freq="M"))
    dfd["AnoMes"] = dfd["Data"].dt.to_period("M")
    dfd = dfd.sort_values(["Fundo", "Data"])
    pl_last = (
        dfd.groupby(["Fundo", "AnoMes"], as_index=False)
        .tail(1)[["Fundo", "AnoMes", "Data", "Patrimonio"]]
        .rename(columns={"Data": "DataRef"})
    )
    flows = dfd.groupby(["Fundo", "AnoMes"], as_index=False)[["Captacao", "Resgate"]].sum()
    m = pl_last.merge(flows, on=["Fundo", "AnoMes"], how="left")
    m["Captacao"] = m["Captacao"].fillna(0)
    m["Resgate"] = m["Resgate"].fillna(0)
    m["Captação Líquida"] = m["Captacao"] - m["Resgate"]
    m["Data"] = m["AnoMes"].dt.to_timestamp("M") - pd.offsets.MonthBegin(1)
    m = m.sort_values(["Fundo", "Data"])
    m["ΔPL"] = m.groupby("Fundo")["Patrimonio"].diff()
    m["Efeito Mercado"] = m["ΔPL"] - m["Captação Líquida"]
    return m[m["Patrimonio"].notna() & (m["Patrimonio"] > 0)]

def build_overview(dfp):
    if dfp.empty or "Patrimonio" not in dfp.columns:
        return pd.DataFrame(), np.nan, np.nan, np.nan
    agg = (
        dfp.groupby("Data", as_index=False)[["Patrimonio", "Captação Líquida"]]
        .sum()
        .dropna(subset=["Patrimonio"])
    )
    if agg.empty: return agg, np.nan, np.nan, np.nan
    pl_ini, pl_fim = agg.iloc[0]["Patrimonio"], agg.iloc[-1]["Patrimonio"]
    var_pct = (pl_fim / pl_ini - 1) * 100 if pl_ini else np.nan
    capt_liq_tot = agg["Captação Líquida"].sum()
    return agg, pl_fim, var_pct, capt_liq_tot

@st.cache_data
def ranking_fundos(dfp):
    if dfp.empty: return pd.DataFrame()
    g = (
        dfp.groupby("Fundo")
        .agg(
            PL_Inicial=("Patrimonio", "first"),
            PL_Final=("Patrimonio", "last"),
            Captação_Líquida=("Captação Líquida", "sum"),
            Efeito_Mercado=("Efeito Mercado", "sum"),
        )
        .reset_index()
    )
    g["Variação_%"] = (g["PL_Final"] / g["PL_Inicial"] - 1) * 100
    return g.sort_values("Variação_%", ascending=False)

# ----------------------------------------------------
# SIDEBAR ÚNICA: FILTROS + PERÍODO
# ----------------------------------------------------
df_raw = fetch_raw()

with st.sidebar:
    st.header("🎚️ Filtros Gerais")

    if "Data" in df_raw.columns:
        df_latest = df_raw.sort_values("Data").groupby("Fundo", as_index=False).tail(1)
    else:
        df_latest = df_raw.copy()

    formas_opts = sorted(df_latest["Forma"].dropna().unique()) if "Forma" in df_latest.columns else []
    exclusivos_opts = sorted(df_latest["Exclusivo"].dropna().unique()) if "Exclusivo" in df_latest.columns else []
    fundos_opts = sorted(df_raw["Fundo"].dropna().unique()) if "Fundo" in df_raw.columns else []

    forma_sel = st.multiselect("Forma:", options=formas_opts, default=formas_opts)
    exclusivo_sel = st.multiselect("Exclusivo:", options=exclusivos_opts, default=exclusivos_opts)
    fundo_sel = st.multiselect("Fundos:", options=fundos_opts, default=fundos_opts)

    st.markdown("---")
    st.header("🗓️ Período")

    min_d, max_d = df_raw["Data"].min().date(), df_raw["Data"].max().date()
    this_year = max_d.year
    preset = st.radio("Selecione um preset:",
        ["YTD (ano corrente)", "Últimos 12 meses", "Ano passado", "Custom"], index=0)

    if preset == "YTD (ano corrente)":
        start, end = date(this_year, 1, 1), max_d
    elif preset == "Últimos 12 meses":
        end, start = max_d, date(end.year - 1, end.month, 1)
    elif preset == "Ano passado":
        start, end = date(this_year - 1, 1, 1), date(this_year - 1, 12, 31)
    else:
        start, end = st.date_input("Custom (início/fim)",
                                   value=(min_d, max_d),
                                   min_value=min_d, max_value=max_d)
    st.caption(f"Período ativo: **{start.strftime('%d/%m/%Y')}** a **{end.strftime('%d/%m/%Y')}**")

# aplica filtros
if forma_sel and "Forma" in df_raw.columns:
    fundos_forma = df_latest[df_latest["Forma"].isin(forma_sel)]["Fundo"]
    df_raw = df_raw[df_raw["Fundo"].isin(fundos_forma)]
if exclusivo_sel and "Exclusivo" in df_raw.columns:
    fundos_exclus = df_latest[df_latest["Exclusivo"].isin(exclusivo_sel)]["Fundo"]
    df_raw = df_raw[df_raw["Fundo"].isin(fundos_exclus)]
if fundo_sel:
    df_raw = df_raw[df_raw["Fundo"].isin(fundo_sel)]

start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
dfp = slice_period_monthly(df_raw, start_ts, end_ts)

# ----------------------------------------------------
# KPIs
# ----------------------------------------------------
agg, pl_fim, var_pct, capt_liq_tot = build_overview(dfp)
k1, k2, k3, k4 = st.columns(4)
k1.metric("💰 PL Total (fim do período)", _fmt_brl(pl_fim), _fmt_pct(var_pct))
k2.metric("📈 Captação líquida (acum.)", _fmt_brl(capt_liq_tot))
k3.metric("🧾 Nº de fundos", f"{dfp['Fundo'].nunique():,}")
k4.metric("🗓️ Meses no período", f"{agg['Data'].nunique() if not agg.empty else 0}")
st.divider()

# ----------------------------------------------------
# GRÁFICOS
# ----------------------------------------------------
st.subheader("Evolução Consolidada (Mensal)")
if not agg.empty:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=agg["Data"], y=agg["Captação Líquida"], name="Captação Líquida (R$)", yaxis="y2", opacity=0.5))
    fig.add_trace(go.Scatter(x=agg["Data"], y=agg["Patrimonio"], name="Patrimônio Total (R$)", line=dict(width=3)))
    fig.update_layout(template=PLOT_TEMPLATE, title="PL Total vs Captação Líquida (mensal)",
                      xaxis_title="Data", yaxis_title="Patrimônio (R$)",
                      yaxis2=dict(title="Captação Líquida (R$)", overlaying="y", side="right"),
                      legend=dict(orientation="h", y=-0.2), margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Sem dados para o período selecionado.")

st.subheader("🏆 Ranking por Fundo (período selecionado)")
rk = ranking_fundos(dfp)
if not rk.empty:
    c_dl = st.columns([1, 5])[0]
    c_dl.download_button("⬇️ Exportar ranking (CSV)",
                         rk.to_csv(index=False).encode("utf-8"),
                         "ranking_fundos.csv", "text/csv")
    st.dataframe(
        rk.style.format({
            "PL_Inicial": _fmt_brl,
            "PL_Final": _fmt_brl,
            "Captação_Líquida": _fmt_brl,
            "Efeito_Mercado": _fmt_brl,
            "Variação_%": _fmt_pct,
        }), use_container_width=True, hide_index=True)
else:
    st.info("Ranking indisponível para o período.")
st.divider()

# ----------------------------------------------------
# HEATMAP
# ----------------------------------------------------
st.subheader("🌡️ Heatmap de Captação Líquida (mês × fundo)")
if not dfp.empty:
    heat = (
        dfp.assign(AnoMes=dfp["Data"].dt.to_period("M").astype(str))
        .pivot_table(index="Fundo", columns="AnoMes", values="Captação Líquida", aggfunc="sum")
        .fillna(0)
    )
    heat = heat.loc[heat.sum(axis=1).sort_values(ascending=False).index]
    fig_hm = px.imshow(
        heat, aspect="auto", title="Captação Líquida por Fundo e Mês",
        labels=dict(color="Captação Líquida (R$)"),
        color_continuous_scale="RdYlGn",
        zmin=-abs(heat.values).max(), zmax=abs(heat.values).max(),
    )
    fig_hm.update_layout(template=PLOT_TEMPLATE, margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig_hm, use_container_width=True)
else:
    st.info("Sem dados para montar o heatmap.")
st.divider()

# ----------------------------------------------------
# DRILL DOWN
# ----------------------------------------------------
st.subheader("🔎 Drill-down por Fundo")
if not dfp.empty:
    fundos = sorted(dfp["Fundo"].unique())
    f_sel = st.selectbox("Selecione o fundo:", fundos)
    dff = dfp[dfp["Fundo"] == f_sel].sort_values("Data")
    if not dff.empty:
        fig_f = go.Figure()
        fig_f.add_trace(go.Bar(x=dff["Data"], y=dff["Captação Líquida"], name="Captação Líquida (R$)", yaxis="y2", opacity=0.5))
        fig_f.add_trace(go.Scatter(x=dff["Data"], y=dff["Patrimonio"], name="Patrimônio (R$)", line=dict(width=3)))
        fig_f.update_layout(template=PLOT_TEMPLATE, title=f"Série Mensal – {f_sel}",
                            xaxis_title="Data", yaxis_title="Patrimônio (R$)",
                            yaxis2=dict(title="Captação Líquida (R$)", overlaying="y", side="right"),
                            legend=dict(orientation="h", y=-0.2), margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_f, use_container_width=True)
