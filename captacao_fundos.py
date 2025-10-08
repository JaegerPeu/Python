import os
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import date
import streamlit as st
import streamlit as st
import os

# ----------------------------------------------------
# CONFIG INICIAL E LOGO (fundo preto fixo + fallback)
# ----------------------------------------------------
st.set_page_config(layout="wide", page_title="Dashboard Institucional – Fundos")

# Tema padrão dos gráficos Plotly
PLOT_TEMPLATE = "plotly_white"

# --- CSS limpo e responsivo ---
st.markdown(
    """
    <style>
    /* Oculta menu e footer padrão */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Layout do logo e título */
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

    /* Fundo fixo preto para o logo (independe do tema) */
    .logo-box {
        background-color: #000000;
        padding: 10px 15px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    /* Tema do app segue o navegador */
    @media (prefers-color-scheme: dark) {
        body { background-color: #0E1117 !important; color: #EAEAEA !important; }
        h1 { color: #EAEAEA !important; }
    }
    @media (prefers-color-scheme: light) {
        body { background-color: #FFFFFF !important; color: #1E1E1E !important; }
        h1 { color: #1E1E1E !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- utilitário para ler SVG com fallback ---
def load_svg(*filenames: str) -> str:
    for fn in filenames:
        try:
            with open(fn, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            continue
    # placeholder simples caso nenhum arquivo exista
    return "<div style='color:#fff;font:12px sans-serif'>LOGO</div>"

# escolha a ordem de preferência dos arquivos de logo
svg_logo = load_svg("logo_light.svg", "logo.svg", "logo_dark.svg")

# --- Renderiza cabeçalho (logo SEMPRE sobre fundo preto) ---
st.markdown(
    f"""
    <div class="logo-container">
        <div class="logo-box" style="width: 180px;">{svg_logo}</div>
        <h1>🏦 Dashboard Institucional – Fundos</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------
# CREDENCIAIS LOCAIS (sem st.secrets)
# ----------------------------------------------------
CD_USER = "solutionswm"
CD_PASS = "Soluti%40ns2023"  # o %40 é o @ codificado na URL
CD_URL  = "https://api.comdinheiro.com.br/v1/ep1/import-data"

# Nota: periodicidade diária → vamos agregar para MÊS no código
URL_PARAM = (
    "HistoricoIndicadoresFundos001.php%3F%26cnpjs%3D03890892000131%2B38090006000170%2B44196960000144"
    "%2B41545780000132%2B43216301000160%2B49692303000101%2B54973692000183%2B55597133000189"
    "%2B55753904000180%2B57499088000155%2B57565778000165%2B57682222000159%2B60334542000122"
    "%26data_ini%3D13112023%26data_fim%3D31129999%26indicadores%3Dnome_fundo%2Bpatrimonio%2Bcaptacao%2Bresgate"
    "%26op01%3Dtabela_v%26num_casas%3D2%26enviar_email%3D0%26periodicidade%3Ddiaria%26cabecalho_excel%3Dmodo3"
    "%26transpor%3D0%26asc_desc%3Ddesc%26tipo_grafico%3Dlinha%26relat_alias_automatico%3Dcmd_alias_01"
)

# ----------------------------------------------------
# HELPERS
# ----------------------------------------------------
def _fmt_brl(v):
    if pd.isna(v):
        return "-"
    return f"R$ {v:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")

def _fmt_pct(v):
    if pd.isna(v):
        return "-"
    return f"{v:+.1f}%"

# ----------------------------------------------------
# 1) FETCH + CACHE
# ----------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_raw():
    payload = f"username={CD_USER}&password={CD_PASS}&URL={URL_PARAM}&format=json3"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    r = requests.post(CD_URL, data=payload, headers=headers, timeout=60)
    r.raise_for_status()
    data = r.json()

    # Normalização
    table_key = list(data["tables"].keys())[0]
    rows = data["tables"][table_key]

    recs = []
    for v in rows.values():
        if v.get("col0"):
            recs.append(
                {
                    "Data": v["col0"],
                    "Fundo": v["col1"],
                    "Patrimonio": v["col2"].replace(",", ".") if v["col2"] else None,
                    "Captacao": v["col3"].replace(",", ".") if v["col3"] else None,
                    "Resgate": v["col4"].replace(",", ".") if v["col4"] else None,
                }
            )
    df = pd.DataFrame(recs)
    df["Data"] = pd.to_datetime(df["Data"], format="%d/%m/%Y", errors="coerce")
    for c in ["Patrimonio", "Captacao", "Resgate"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Data", "Fundo"]).sort_values(["Fundo", "Data"])
    return df

# ----------------------------------------------------
# 2) PRESET DE PERÍODO + FILTRO
# ----------------------------------------------------
def period_presets_ui(df):
    min_d = df["Data"].min().date()
    max_d = df["Data"].max().date()
    this_year = max_d.year

    with st.sidebar:
        st.header("🗓️ Período")
        preset = st.radio(
            "Selecione um preset:",
            ["YTD (ano corrente)", "Últimos 12 meses", "Ano passado", "Custom"],
            index=0,
        )

        if preset == "YTD (ano corrente)":
            start = date(this_year, 1, 1)
            end = max_d
        elif preset == "Últimos 12 meses":
            end = max_d
            start = date(end.year - 1, end.month, 1)
        elif preset == "Ano passado":
            start = date(this_year - 1, 1, 1)
            end = date(this_year - 1, 12, 31)
        else:
            start, end = st.date_input(
                "Custom (início/fim)", value=(min_d, max_d), min_value=min_d, max_value=max_d
            )

        st.caption(f"Período ativo: **{start.strftime('%d/%m/%Y')}** a **{end.strftime('%d/%m/%Y')}**")
    return pd.Timestamp(start), pd.Timestamp(end)

# ----------------------------------------------------
# 3) AGREGAÇÃO MENSAL (corrige captação e ΔPL)
# ----------------------------------------------------
@st.cache_data
def slice_period_monthly(df, start, end):
    """
    - Soma CAPTACAO/RESGATE por mês e por fundo
    - PL mensal = ÚLTIMO valor do mês por fundo (EoM)
    - ΔPL = diferença mensal do PL
    - Efeito Mercado = ΔPL - Captação Líquida
    """
    dfd = df[(df["Data"] >= start) & (df["Data"] <= end)].copy()
    if dfd.empty:
        return dfd.assign(AnoMes=pd.PeriodIndex([], freq="M"))

    dfd["AnoMes"] = dfd["Data"].dt.to_period("M")
    dfd = dfd.sort_values(["Fundo", "Data"])

    # PL do mês = último registro dentro do mês
    pl_last = (
        dfd.groupby(["Fundo", "AnoMes"], as_index=False)
        .tail(1)[["Fundo", "AnoMes", "Data", "Patrimonio"]]
        .rename(columns={"Data": "DataRef"})
    )

    # Fluxos do mês (somatório diário)
    flows = (
        dfd.groupby(["Fundo", "AnoMes"], as_index=False)[["Captacao", "Resgate"]]
        .sum()
    )

    m = pl_last.merge(flows, on=["Fundo", "AnoMes"], how="left")
    m["Captacao"] = m["Captacao"].fillna(0)
    m["Resgate"] = m["Resgate"].fillna(0)
    m["Captação Líquida"] = m["Captacao"] - m["Resgate"]

    # Data = fim do mês (para eixo X coerente)
    m["Data"] = m["AnoMes"].dt.to_timestamp("M")
    m = m.sort_values(["Fundo", "Data"])

        # ΔPL e Efeito Mercado
    m["ΔPL"] = m.groupby("Fundo")["Patrimonio"].diff()
    m["Efeito Mercado"] = m["ΔPL"] - m["Captação Líquida"]

    # 🔹 Remove fundos/meses sem PL (mantém apenas dados válidos)
    m = m[m["Patrimonio"].notna() & (m["Patrimonio"] > 0)]

    return m


def build_overview(dfp):
    # Garante que o DataFrame tenha dados válidos
    if dfp.empty or "Patrimonio" not in dfp.columns:
        return pd.DataFrame(), np.nan, np.nan, np.nan

    # Agrega por mês (somando captação e pegando PL total)
    agg = (
        dfp.groupby("Data", as_index=False)[["Patrimonio", "Captação Líquida"]]
        .sum()
        .dropna(subset=["Patrimonio"])
    )

    # Se ainda assim não houver dados, retorna nulos
    if agg.empty:
        return agg, np.nan, np.nan, np.nan

    # Filtra apenas meses com PL válido (exclui futuros vazios)
    agg = agg[agg["Patrimonio"].notna() & (agg["Patrimonio"] > 0)]
    if agg.empty:
        return pd.DataFrame(), np.nan, np.nan, np.nan

    # Calcula os indicadores com segurança
    pl_ini = agg.iloc[0]["Patrimonio"]
    pl_fim = agg.iloc[-1]["Patrimonio"]
    var_pct = (pl_fim / pl_ini - 1) * 100 if pl_ini else np.nan
    capt_liq_tot = agg["Captação Líquida"].sum()

    return agg, pl_fim, var_pct, capt_liq_tot


@st.cache_data
def ranking_fundos(dfp):
    if dfp.empty:
        return pd.DataFrame()
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
# 4) UI – CONTROLES & DADOS
# ----------------------------------------------------
df_raw = fetch_raw()
start_ts, end_ts = period_presets_ui(df_raw)
dfp = slice_period_monthly(df_raw, start_ts, end_ts)  # <<< agora MENSAL

# KPIs
agg, pl_fim, var_pct, capt_liq_tot = build_overview(dfp)

k1, k2, k3, k4 = st.columns(4)
k1.metric("💰 PL Total (fim do período)", _fmt_brl(pl_fim), _fmt_pct(var_pct))
k2.metric("📈 Captação líquida (acum.)", _fmt_brl(capt_liq_tot))
k3.metric("🧾 Nº de fundos", f"{dfp['Fundo'].nunique():,}")
k4.metric("🗓️ Meses no período", f"{agg['Data'].nunique() if not agg.empty else 0}")

st.divider()

# ----------------------------------------------------
# 5) OVERVIEW – Evolução consolidada
# ----------------------------------------------------
st.subheader("Evolução Consolidada (Mensal)")
if not agg.empty:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=agg["Data"],
            y=agg["Captação Líquida"],
            name="Captação Líquida (R$)",
            yaxis="y2",
            opacity=0.5,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=agg["Data"],
            y=agg["Patrimonio"],
            name="Patrimônio Total (R$)",
            line=dict(width=3),
        )
    )
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title="PL Total vs Captação Líquida (mensal)",
        xaxis_title="Data",
        yaxis_title="Patrimônio (R$)",
        yaxis2=dict(title="Captação Líquida (R$)", overlaying="y", side="right"),
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Sem dados para o período selecionado.")

# ----------------------------------------------------
# 6) RANKING – Fundos
# ----------------------------------------------------
st.subheader("🏆 Ranking por Fundo (período selecionado)")
rk = ranking_fundos(dfp)
if not rk.empty:
    c_dl = st.columns([1, 5])[0]
    c_dl.download_button("⬇️ Exportar ranking (CSV)", rk.to_csv(index=False).encode("utf-8"), "ranking_fundos.csv", "text/csv")

    st.dataframe(
        rk.style.format(
            {
                "PL_Inicial": _fmt_brl,
                "PL_Final": _fmt_brl,
                "Captação_Líquida": _fmt_brl,
                "Efeito_Mercado": _fmt_brl,
                "Variação_%": _fmt_pct,
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("Ranking indisponível para o período.")

st.divider()

# ----------------------------------------------------
# 7) HEATMAP – Captação líquida (mês × fundo)
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
        heat,
        aspect="auto",
        title="Captação Líquida por Fundo e Mês",
        labels=dict(color="R$"),
        text_auto=False,
    )
    fig_hm.update_layout(template=PLOT_TEMPLATE, margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig_hm, use_container_width=True)

    c2 = st.columns([1, 5])[0]
    c2.download_button("⬇️ Exportar heatmap (CSV)", heat.reset_index().to_csv(index=False).encode("utf-8"),
                       "heatmap_captacao.csv", "text/csv")
else:
    st.info("Sem dados para montar o heatmap.")

st.divider()

# ----------------------------------------------------
# 8) DRILL-DOWN POR FUNDO (mantido)
# ----------------------------------------------------
st.subheader("🔎 Drill-down por Fundo")
if not dfp.empty:
    fundos = sorted(dfp["Fundo"].unique())
    f_sel = st.selectbox("Selecione o fundo:", fundos)

    dff = dfp[dfp["Fundo"] == f_sel].sort_values("Data").copy()
    if not dff.empty:
        fig_f = go.Figure()
        fig_f.add_trace(go.Bar(x=dff["Data"], y=dff["Captação Líquida"], name="Captação Líquida (R$)", yaxis="y2", opacity=0.5))
        fig_f.add_trace(go.Scatter(x=dff["Data"], y=dff["Patrimonio"], name="Patrimônio (R$)", line=dict(width=3)))
        fig_f.update_layout(
            template=PLOT_TEMPLATE,
            title=f"Série Mensal – {f_sel}",
            xaxis_title="Data",
            yaxis_title="Patrimônio (R$)",
            yaxis2=dict(title="Captação Líquida (R$)", overlaying="y", side="right"),
            legend=dict(orientation="h", y=-0.2),
            margin=dict(l=10, r=10, t=60, b=10),
        )
        st.plotly_chart(fig_f, use_container_width=True)

        wf = dff.copy()
        wf["Mês"] = wf["Data"].dt.to_period("M").astype(str)
        agg_wf = wf.groupby("Mês", as_index=False)[["Captação Líquida", "Efeito Mercado"]].sum()
        agg_wf["ΔPL (Total)"] = agg_wf["Captação Líquida"] + agg_wf["Efeito Mercado"]

        fig_w = go.Figure()
        fig_w.add_trace(go.Bar(x=agg_wf["Mês"], y=agg_wf["Captação Líquida"], name="Captação Líquida"))
        fig_w.add_trace(go.Bar(x=agg_wf["Mês"], y=agg_wf["Efeito Mercado"], name="Efeito Mercado"))
        fig_w.update_layout(
            barmode="relative",
            template=PLOT_TEMPLATE,
            title=f"Decomposição Mensal do ΔPL – {f_sel}",
            xaxis_title="Mês",
            yaxis_title="Variação de PL (R$)",
            margin=dict(l=10, r=10, t=60, b=10),
        )
        st.plotly_chart(fig_w, use_container_width=True)

        dff_out = dff[["Data", "Fundo", "Patrimonio", "Captacao", "Resgate", "Captação Líquida", "ΔPL", "Efeito Mercado"]]
        cc1 = st.columns([1, 5])[0]
        cc1.download_button("⬇️ Exportar dados do fundo (CSV)",
                            dff_out.to_csv(index=False).encode("utf-8"),
                            f"{f_sel[:25].replace(' ', '_')}_mensal.csv",
                            "text/csv")
        st.dataframe(
            dff_out.style.format({
                "Patrimonio": _fmt_brl,
                "Captacao": _fmt_brl,
                "Resgate": _fmt_brl,
                "Captação Líquida": _fmt_brl,
                "ΔPL": _fmt_brl,
                "Efeito Mercado": _fmt_brl,
            }),
            use_container_width=True, hide_index=True
        )

# ----------------------------------------------------
# 9) CONTROLES EXTRAS
# ----------------------------------------------------
with st.sidebar:
    st.header("⚙️ Opções")
    if st.button("🔄 Atualizar dados (limpar cache)"):
        st.cache_data.clear()
        st.success("Cache limpo! Recarregue o app.")

    st.caption("""Nota: dados de fluxo são somados no mês; PL é o último do mês.
               
               Variação_% = (PLFinal/PLInicial) -1)""")











