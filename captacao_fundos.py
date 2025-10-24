import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta

# ====================================================
# CONFIG INICIAL
# ====================================================
st.set_page_config(
    page_title="Dashboard Institucional – Fundos",
    layout="wide",
    initial_sidebar_state="expanded"  # abre a sidebar por padrão
)
PLOT_TEMPLATE = "plotly_white"

# ====================================================
# ESTILO VISUAL (CSS)
# ====================================================
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.logo-container {
    display: flex; align-items: center; gap: 15px; margin-bottom: 25px;
}
h1 {
    font-size: 30px !important; font-weight: 700; margin: 0;
}
.logo-box {
    background-color: #000; padding: 10px 15px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
}
</style>
""", unsafe_allow_html=True)

# ====================================================
# LOGO E TÍTULO
# ====================================================
def load_svg(*names):
    for n in names:
        try:
            with open(n, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            continue
    return "<div style='color:#fff;font-size:12px;'>LOGO</div>"

svg_logo = load_svg("logo_dark.svg", "logo_light.svg", "logo.svg")

st.markdown(f"""
<div class="logo-container">
    <div class="logo-box" style="width:180px;">{svg_logo}</div>
    <h1>🏦 Dashboard – Fundos</h1>
</div>
""", unsafe_allow_html=True)

# ====================================================
# CONFIG COMDINHEIRO
# ====================================================
CD_USER = "solutionswm"
CD_PASS = "Soluti%40ns2023"
CD_URL = "https://api.comdinheiro.com.br/v1/ep1/import-data"

URL_PARAM = (
    "HistoricoIndicadoresFundos001.php%3F%26cnpjs%3D"
    "03890892000131%2B38090006000170%2B44196960000144%2B57565778000165%2B41545780000132%2B43216301000160"
    "%2B49692303000101%2B54973692000183%2B55597133000189%2B55753904000180%2B57499088000155%2B57682222000159"
    "%2B60334542000122%2B60800845000193_unica"
    "%26data_ini%3D13112023%26data_fim%3D31129999"
    "%26indicadores%3Dnome_fundo%2Bpatrimonio%2Bcaptacao%2Bresgate%2Bforma%2Bexclusivo"
    "%26op01%3Dtabela_v%26num_casas%3D2%26periodicidade%3Ddiaria"
    "%26cabecalho_excel%3Dmodo3%26transpor%3D0%26asc_desc%3Ddesc"
)

# ====================================================
# FUNÇÕES DE UTILIDADE
# ====================================================
def fmt_brl(v):
    return "-" if pd.isna(v) else f"R$ {v:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")

def fmt_pct(v):
    return "-" if pd.isna(v) else f"{v:+.1f}%"

# ====================================================
# 1. BUSCA E NORMALIZAÇÃO DE DADOS
# ====================================================
@st.cache_data(ttl=3600)
def fetch_data():
    payload = f"username={CD_USER}&password={CD_PASS}&URL={URL_PARAM}&format=json3"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    r = requests.post(CD_URL, data=payload, headers=headers, timeout=60)
    r.raise_for_status()
    data = r.json()
    rows = list(data["tables"].values())[0]

    records = []
    for v in rows.values():
        if v.get("col0"):
            records.append({
                "Data": v["col0"],
                "Fundo": v["col1"],
                "Patrimonio": v.get("col2", "").replace(",", ".") if v.get("col2") else None,
                "Captacao": v.get("col3", "").replace(",", ".") if v.get("col3") else None,
                "Resgate": v.get("col4", "").replace(",", ".") if v.get("col4") else None,
                "Forma": v.get("col5", "").strip().capitalize() if v.get("col5") else None,
                "Exclusivo": v.get("col6", "").strip().capitalize() if v.get("col6") else None,
            })

    df = pd.DataFrame(records)
    df["Data"] = pd.to_datetime(df["Data"], format="%d/%m/%Y", errors="coerce")
    for c in ["Patrimonio", "Captacao", "Resgate"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Data", "Fundo"]).sort_values(["Fundo", "Data"])
    return df

# ====================================================
# 2. AGREGAÇÃO MENSAL
# ====================================================
def aggregate_monthly(df, start, end):
    df = df[(df["Data"] >= start) & (df["Data"] <= end)].copy()
    if df.empty:
        return pd.DataFrame()

    df["AnoMes"] = df["Data"].dt.to_period("M")
    df.sort_values(["Fundo", "Data"], inplace=True)

    pl = (
        df.groupby(["Fundo", "AnoMes"], as_index=False)
        .tail(1)[["Fundo", "AnoMes", "Patrimonio"]]
    )

    flows = df.groupby(["Fundo", "AnoMes"], as_index=False)[["Captacao", "Resgate"]].sum()
    m = pl.merge(flows, on=["Fundo", "AnoMes"], how="left")
    m["Captação Líquida"] = m["Captacao"].fillna(0) - m["Resgate"].fillna(0)
    m["Data"] = m["AnoMes"].dt.to_timestamp("M")
    m["ΔPL"] = m.groupby("Fundo")["Patrimonio"].diff()
    m["Efeito Mercado"] = m["ΔPL"] - m["Captação Líquida"]
    return m[m["Patrimonio"] > 0]

# ====================================================
# 3. SIDEBAR: FILTROS
# ====================================================
df_raw = fetch_data()
with st.sidebar:
    st.header("🎚️ Filtros Gerais")

    df_latest = df_raw.sort_values("Data").groupby("Fundo", as_index=False).tail(1)
    formas = sorted(df_latest["Forma"].dropna().unique()) if "Forma" in df_latest.columns else []
    exclusivos = sorted(df_latest["Exclusivo"].dropna().unique()) if "Exclusivo" in df_latest.columns else []
    fundos = sorted(df_raw["Fundo"].dropna().unique())

    forma_sel = st.multiselect("Forma:", formas, default=formas)
    exc_sel = st.multiselect("Exclusivo:", exclusivos, default=exclusivos)
    fundo_sel = st.multiselect("Fundos:", fundos, default=fundos)

    st.markdown("---")
    st.header("🗓️ Período")

    min_d, max_d = df_raw["Data"].min().date(), df_raw["Data"].max().date()
    # Ajuste para limitar final d-2 (hoje menos dois dias)
    max_d_adjusted = max_d - timedelta(days=2)
    this_year = max_d_adjusted.year

    preset = st.radio("Selecione um preset:", ["YTD", "Últimos 12 meses", "Ano passado", "Custom"], index=0)

    if preset == "YTD":
        start, end = date(this_year, 1, 1), max_d_adjusted
    elif preset == "Últimos 12 meses":
        # Corrigido para evitar erro de variável usada antes da definição
        start = date(max_d_adjusted.year - 1, max_d_adjusted.month, 1)
        end = max_d_adjusted
    elif preset == "Ano passado":
        start = date(this_year - 1, 1, 1)
        end = date(this_year - 1, 12, 31)
    else:
        start, end = st.date_input("Custom (início/fim)", (min_d, max_d_adjusted), min_value=min_d, max_value=max_d_adjusted)

# aplica filtros
if forma_sel and "Forma" in df_raw.columns:
    fundos_forma = df_latest[df_latest["Forma"].isin(forma_sel)]["Fundo"]
    df_raw = df_raw[df_raw["Fundo"].isin(fundos_forma)]
if exc_sel and "Exclusivo" in df_raw.columns:
    fundos_exc = df_latest[df_latest["Exclusivo"].isin(exc_sel)]["Fundo"]
    df_raw = df_raw[df_raw["Fundo"].isin(fundos_exc)]
if fundo_sel:
    df_raw = df_raw[df_raw["Fundo"].isin(fundo_sel)]

start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
df_monthly = aggregate_monthly(df_raw, start_ts, end_ts)

# ====================================================
# 4. MÉTRICAS GERAIS
# ====================================================
if df_monthly.empty:
    st.warning("Sem dados no período selecionado.")
    st.stop()

agg = df_monthly.groupby("Data", as_index=False)[["Patrimonio", "Captação Líquida"]].sum()
pl_ini, pl_fim = agg.iloc[0]["Patrimonio"], agg.iloc[-1]["Patrimonio"]
var_pct = (pl_fim / pl_ini - 1) * 100 if pl_ini else np.nan
capt_liq_tot = agg["Captação Líquida"].sum()

k1, k2, k3, k4 = st.columns(4)
k1.metric("💰 PL Total", fmt_brl(pl_fim), fmt_pct(var_pct))
k2.metric("📈 Captação Líquida (acum.)", fmt_brl(capt_liq_tot))
k3.metric("🧾 Nº de Fundos", df_monthly["Fundo"].nunique())
k4.metric("🗓️ Meses", agg["Data"].nunique())
st.divider()

# ====================================================
# 5. EVOLUÇÃO CONSOLIDADA
# ====================================================
st.subheader("📊 Evolução Consolidada (Mensal)")
fig = go.Figure()
fig.add_trace(go.Bar(x=agg["Data"], y=agg["Captação Líquida"], name="Captação Líquida (R$)", yaxis="y2", opacity=0.5))
fig.add_trace(go.Scatter(x=agg["Data"], y=agg["Patrimonio"], name="Patrimônio (R$)", line=dict(width=3)))
fig.update_layout(template=PLOT_TEMPLATE, yaxis2=dict(overlaying="y", side="right"),
                  legend=dict(orientation="h", y=-0.2), margin=dict(l=10, r=10, t=60, b=10))
st.plotly_chart(fig, use_container_width=True)
st.divider()

# ====================================================
# 6. RANKING DE FUNDOS
# ====================================================
st.subheader("🏆 Ranking por Fundo")
rank = (
    df_monthly.groupby("Fundo")
    .agg(
        PL_Inicial=("Patrimonio", "first"),
        PL_Final=("Patrimonio", "last"),
        Captação_Líquida=("Captação Líquida", "sum"),
        Efeito_Mercado=("Efeito Mercado", "sum"),
    )
    .reset_index()
)
rank["Variação_%"] = (rank["PL_Final"] / rank["PL_Inicial"] - 1) * 100
st.dataframe(
    rank.style.format({
        "PL_Inicial": fmt_brl, "PL_Final": fmt_brl,
        "Captação_Líquida": fmt_brl, "Efeito_Mercado": fmt_brl,
        "Variação_%": fmt_pct,
    }),
    use_container_width=True, hide_index=True
)
st.download_button("⬇️ Exportar Ranking (CSV)", rank.to_csv(index=False).encode("utf-8"), "ranking.csv", "text/csv")
st.divider()

# ====================================================
# 7. HEATMAP
# ====================================================
st.subheader("🌡️ Heatmap – Captação Líquida (Mês x Fundo)")
heat = (
    df_monthly.assign(AnoMes=df_monthly["Data"].dt.to_period("M").astype(str))
    .pivot_table(index="Fundo", columns="AnoMes", values="Captação Líquida", aggfunc="sum")
    .fillna(0)
)
heat = heat.loc[heat.sum(axis=1).sort_values(ascending=False).index]
fig_hm = px.imshow(heat, aspect="auto", color_continuous_scale="RdYlGn", title="Captação Líquida por Fundo e Mês")
fig_hm.update_layout(template=PLOT_TEMPLATE, margin=dict(l=10, r=10, t=60, b=10))
st.plotly_chart(fig_hm, use_container_width=True)
st.divider()

# ====================================================
# 8. DRILL DOWN POR FUNDO
# ====================================================
st.subheader("🔎 Drill-down por Fundo")
f_sel = st.selectbox("Selecione o fundo:", sorted(df_monthly["Fundo"].unique()))
dff = df_monthly[df_monthly["Fundo"] == f_sel]
if not dff.empty:
    figf = go.Figure()
    figf.add_trace(go.Bar(x=dff["Data"], y=dff["Captação Líquida"], name="Captação Líquida", yaxis="y2", opacity=0.5))
    figf.add_trace(go.Scatter(x=dff["Data"], y=dff["Patrimonio"], name="Patrimônio", line=dict(width=3)))
    figf.update_layout(
        template=PLOT_TEMPLATE,
        title=f"Série Mensal – {f_sel}",
        yaxis2=dict(overlaying="y", side="right"),
        legend=dict(orientation="h", y=-0.2)
    )
    st.plotly_chart(figf, use_container_width=True)
