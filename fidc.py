import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests

st.set_page_config(
    page_title="Monitoramento FIDC",
    layout="wide"
)

st.title("Monitoramento – Solutions FIDC")

# =========================
# 1) Entrada do arquivo
# =========================
st.sidebar.header("Configurações")

uploaded_file = st.sidebar.file_uploader(
    "Carregue a base do fundo (Excel com abas macro e micro)",
    type=["xlsx", "xls"]
)

if uploaded_file is None:
    st.info("Carregue o arquivo para iniciar a análise.")
    st.stop()

# =========================
# 2) Leitura das abas
# =========================
sheets = pd.read_excel(uploaded_file, sheet_name=None)

macro_key = None
micro_key = None
for k in sheets.keys():
    kl = str(k).strip().lower()
    if "macro" in kl:
        macro_key = k
    if "micro" in kl:
        micro_key = k

if macro_key is None or micro_key is None:
    st.error("Não encontrei abas 'macro' e 'micro' no arquivo. Verifique os nomes.")
    st.write("Abas encontradas:", list(sheets.keys()))
    st.stop()

df_macro = sheets[macro_key].copy()
df_micro = sheets[micro_key].copy()

# =========================
# 3) Normalização de colunas
# =========================
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("%", "pct", regex=False)
        .str.replace("ç", "c")
        .str.replace("ã", "a")
        .str.replace("á", "a")
        .str.replace("é", "e")
        .str.replace("ó", "o")
        .str.replace("í", "i")
    )
    df.columns = cols
    return df

df_macro = normalize_cols(df_macro)
df_micro = normalize_cols(df_micro)

# ---------- MICRO ----------
df_micro = normalize_cols(df_micro)

df_micro = df_micro.rename(columns={
    "Data_posição": "Data_posicao",
    "Forma_de_condominio": "Forma_condominio",   # <-- no seu arquivo vira isso
    "PL_FUNDO": "PL_FUNDO",
    "Sub_Ponderada": "Sub_Ponderada",
    "PDD_Ponderada": "PDD_Ponderada",
    "pctPL": "pct_PL",    # %PL -> pctPL no normalize
})


if "Data_posicao" in df_micro.columns:
    df_micro["Data_posicao"] = pd.to_datetime(df_micro["Data_posicao"], errors="coerce")

percent_cols_micro = [
    "PDD",
    "Subordinação",
    "pct_PL",
    "Sub_Ponderada",
    "PDD_Ponderada",
]
for c in percent_cols_micro:
    if c in df_micro.columns:
        if df_micro[c].dtype == "object":
            df_micro[c] = (
                df_micro[c]
                .str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False)
            )
        df_micro[c] = pd.to_numeric(df_micro[c], errors="coerce")

if "PL_FUNDO" in df_micro.columns and df_micro["PL_FUNDO"].dtype == "object":
    df_micro["PL_FUNDO"] = (
        df_micro["PL_FUNDO"]
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    df_micro["PL_FUNDO"] = pd.to_numeric(df_micro["PL_FUNDO"], errors="coerce")

# ---------- MACRO ----------
# ajuste o nome da coluna de PL se necessário (veja no debug abaixo)
df_macro = df_macro.rename(columns={
    "Valor_PL": "PL_macro",   # troque "Valor_PL" se o nome normalizado for outro
    "Ativo": "Ativo",
    "pct": "pct",
    "%": "pct",
    "Carrego": "Carrego",
    "Subordinação": "Subordinacao",
    "PDD": "PDD",
})

num_macro = ["pct", "Carrego", "Subordinacao", "PDD", "PL_macro"]
for c in num_macro:
    if c in df_macro.columns:
        if df_macro[c].dtype == "object":
            df_macro[c] = (
                df_macro[c]
                .str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False)
            )
        df_macro[c] = pd.to_numeric(df_macro[c], errors="coerce")

# =========================
# 4) Filtro de data (micro)
# =========================
if "Data_posicao" in df_micro.columns:
    max_date = df_micro["Data_posicao"].max()
    min_date = df_micro["Data_posicao"].min()

    sel_date = st.sidebar.date_input(
        "Data de posição",
        value=max_date.date() if pd.notnull(max_date) else None,
        min_value=min_date.date() if pd.notnull(min_date) else None,
        max_value=max_date.date() if pd.notnull(max_date) else None,
    )

    df_micro = df_micro[df_micro["Data_posicao"] == pd.to_datetime(sel_date)]

# =========================
# 5) KPIs – micro / macro
# =========================
pl_total = df_macro["PL_macro"].sum() if "PL_macro" in df_macro.columns else np.nan
n_fidcs = df_micro[df_micro["PRODUTO"] == "FIDC"]["Ativo"].nunique() if "PRODUTO" in df_micro.columns else np.nan
n_ativos_macro = df_macro["Ativo"].nunique() if "Ativo" in df_macro.columns else np.nan

pdd_pl_micro = df_micro["PDD_Ponderada"].sum() if "PDD_Ponderada" in df_micro.columns else np.nan
sub_pl_micro = df_micro["Sub_Ponderada"].sum() if "Sub_Ponderada" in df_micro.columns else np.nan

TX_ADM = 0.008  # 0,8%

if all(c in df_macro.columns for c in ["pct", "Carrego"]):
    cdi_plus = (df_macro["pct"] * df_macro["Carrego"]).sum()
else:
    cdi_plus = np.nan

if not np.isnan(cdi_plus):
    cdi_liq = ((1 + cdi_plus) / (1 + TX_ADM)) - 1
else:
    cdi_liq = np.nan

if all(c in df_macro.columns for c in ["pct", "Subordinacao"]):
    sub_macro = (df_macro["pct"] * df_macro["Subordinacao"]).sum()
else:
    sub_macro = np.nan

if all(c in df_macro.columns for c in ["pct", "PDD"]):
    pdd_macro = (df_macro["pct"] * df_macro["PDD"]).sum()
else:
    pdd_macro = np.nan

# =========================
# 6) Cards de métricas
# =========================
col1, col2, col3, col4 = st.columns(4)
col5, col6, col7, col8 = st.columns(4)

if not np.isnan(pl_total):
    pl_str = f"{pl_total:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
else:
    pl_str = "-"

col1.metric("PL", f"R$ {pl_str}")
col2.metric("Nº de FIDCs", int(n_fidcs) if not np.isnan(n_fidcs) else "-")
col3.metric("Nº de Ativos (macro)", int(n_ativos_macro) if not np.isnan(n_ativos_macro) else "-")
col4.metric("PDD (micro)", f"{pdd_pl_micro*100:,.2f} %" if not np.isnan(pdd_pl_micro) else "-")

col5.metric("CDI+ (carrego)", f"{cdi_plus*100:,.2f} %" if not np.isnan(cdi_plus) else "-")
col6.metric("Taxa Adm.", f"{TX_ADM*100:,.2f} %")
col7.metric("CDI Líquido", f"{cdi_liq*100:,.2f} %" if not np.isnan(cdi_liq) else "-")
col8.metric("Subordinação (macro)", f"{sub_macro*100:,.2f} %" if not np.isnan(sub_macro) else "-")

st.markdown("---")

# =========================
# 7) Linha 1 de gráficos – barras + treemap
# =========================
# g1 = st.columns(1)

# with g1:
#     st.subheader("%PL por Ativo (Top 15) – Micro")
#     if "pct_PL" in df_micro.columns:
#         df_pl = (
#             df_micro.groupby("Ativo", as_index=False)["pct_PL"]
#             .sum()
#             .sort_values("pct_PL", ascending=False)
#             .head(15)
#         )
#         st.bar_chart(df_pl.set_index("Ativo")["pct_PL"], use_container_width=True)
#     else:
#         st.warning("Coluna '%PL' não encontrada na aba micro (esperado 'pct_PL' após normalização).")

# with g1:
st.subheader("Concentração de %PL por Gestora")
if all(c in df_micro.columns for c in ["Gestora", "Ativo", "pct_PL"]):
    df_treemap = (
        df_micro.groupby(["Gestora", "Ativo"], as_index=False)["pct_PL"]
        .sum()
    )
    fig_t = px.treemap(
        df_treemap,
        path=["Gestora", "Ativo"],
        values="pct_PL",
        color="Gestora",
    )
    fig_t.update_traces(texttemplate="%{label}<br>%{value:.2%}")
    st.plotly_chart(fig_t, use_container_width=True)
else:
    st.warning("Para o treemap são necessárias as colunas 'Gestora', 'Ativo' e 'pct_PL' na aba micro.")

# =========================
# 8) Linha 2 – Cotas, Retorno‑alvo, Condomínio
# =========================
c1, c2 = st.columns(2)

# Cotas dos ativos na carteira (micro)
with c1:
    st.subheader("Cotas dos ativos na carteira")
    if all(c in df_micro.columns for c in ["Cota", "pct_PL"]):
        df_cota = (
            df_micro.groupby("Cota", as_index=False)["pct_PL"]
            .sum()
            .sort_values("pct_PL", ascending=False)
        )
        fig_cota = px.pie(
            df_cota,
            names="Cota",
            values="pct_PL",
            title="Distribuição de %PL por tipo de cota",
        )
        fig_cota.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_cota, use_container_width=True)
    else:
        st.warning("Para o gráfico de cotas, preciso de 'Cota' e 'pct_PL' na aba micro.")

# Retorno‑alvo (macro) – carrego ponderado pelo %
with c2:
    st.subheader("Retorno‑alvo")
    if all(c in df_macro.columns for c in ["Ativo", "pct", "Carrego"]):
        df_ret = df_macro.copy()
        df_ret["peso_carrego"] = df_ret["pct"] * df_ret["Carrego"]

        # cria label no formato CDI+X%
        df_ret["Carrego_label"] = df_ret["Carrego"].apply(
            lambda x: f"CDI+{x*100:.2f}%"
        )

        fig_ret = px.pie(
            df_ret,
            names="Carrego_label",   # usa o label formatado
            values="peso_carrego",
            title="Distribuição do carrego por ativo principal",
        )
        fig_ret.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_ret, use_container_width=True)
    else:
        st.warning("Para o retorno‑alvo, preciso de 'Ativo', 'pct' e 'Carrego' na aba macro.")

c3, c4 = st.columns(2)
# Condomínio (micro) – %PL por Forma de condomínio
with c3:
    st.subheader("Condomínio")
    if all(c in df_micro.columns for c in ["Forma_condominio", "pct_PL"]):
        df_cond = (
            df_micro.groupby("Forma_condominio", as_index=False)["pct_PL"]
            .sum()
            .sort_values("pct_PL", ascending=False)
        )
        fig_cond = px.pie(
            df_cond,
            names="Forma_condominio",
            values="pct_PL",
            title="Distribuição de %PL por condomínio",
        )
        fig_cond.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_cond, use_container_width=True)
    else:
        st.warning("Para o gráfico de condomínio, preciso de 'Forma_condominio' e 'pct_PL' na aba micro.")
        
with c4:
    st.subheader("Alocação por setor")
    if all(c in df_micro.columns for c in ["Industry", "pct_PL"]):
        df_setor = (
            df_micro.groupby("Industry", as_index=False)["pct_PL"]
            .sum()
            .sort_values("pct_PL", ascending=False)
        )
        fig_setor = px.pie(
            df_setor,
            names="Industry",
            values="pct_PL",
            title="Distribuição de %PL por setor",
        )
        fig_setor.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_setor, use_container_width=True)
    else:
        st.warning("Para o gráfico de setores, preciso de 'Industry' e 'pct_PL' na aba micro.")



    

# =========================
# 10) Evolução do PL via API Comdinheiro
# =========================
st.markdown("---")
st.subheader("Evolução do PL")

@st.cache_data(ttl=60*30)
def carregar_pl_comdinheiro():
    url = "https://api.comdinheiro.com.br/v1/ep1/import-data"

    payload = (
        "username=solutionswm"
        "&password=Soluti%40ns2025"
        "&URL=HistoricoIndicadoresFundos001.php%3F%26cnpjs%3D60800845000193_unica"
        "%26data_ini%3D15072025%26data_fim%3Ddmenos2%26indicadores%3Dpatrimonio"
        "%26op01%3Dtabela_h%26num_casas%3D2%26enviar_email%3D0"
        "%26periodicidade%3Ddiaria%26cabecalho_excel%3Dmodo2"
        "%26transpor%3D0%26asc_desc%3Ddesc%26tipo_grafico%3Dlinha"
        "%26relat_alias_automatico%3Dcmd_alias_01"
        "&format=json3"
    )

    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    r = requests.post(url, data=payload, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()  # dict com meta / tables

    # tables: { "tab0": { "lin0": {...}, "lin1": {...}, ... } }
    tables = data.get("tables", {})
    tab0 = tables.get("tab0", {})

    # converte linX -> lista de linhas
    rows = []
    for key in sorted(tab0.keys(), key=lambda x: int(x.replace("lin", ""))):
        row = tab0[key]
        rows.append([row.get("col0"), row.get("col1")])

    # primeira linha (lin0) é cabeçalho
    header = rows[0]
    data_rows = rows[1:]

    df_pl = pd.DataFrame(data_rows, columns=header)
    return df_pl

try:
    df_pl = carregar_pl_comdinheiro()
    #st.write("Colunas evolução PL (debug):", list(df_pl.columns))

    # renomeia para Data / PL
    # no seu JSON: col0 = "Data", col1 = "Patrimônio\nR$\n\n..."
    rename_map = {}
    for c in df_pl.columns:
        cl = c.lower()
        if "data" in cl:
            rename_map[c] = "Data"
        if "patrim" in cl:
            rename_map[c] = "PL"

    df_pl = df_pl.rename(columns=rename_map)

    if all(c in df_pl.columns for c in ["Data", "PL"]):
        # converter tipos
        df_pl["Data"] = pd.to_datetime(df_pl["Data"], format="%d/%m/%Y", errors="coerce")
        df_pl["PL"] = (
            df_pl["PL"]
            .astype(str)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        df_pl["PL"] = pd.to_numeric(df_pl["PL"], errors="coerce")

        df_pl = df_pl.sort_values("Data")

        st.line_chart(
            df_pl.set_index("Data")["PL"],
            use_container_width=True
        )
    else:
        st.warning("Não encontrei colunas de Data e PL na resposta da API. Veja o debug acima.")

except Exception as e:
    st.error(f"Erro ao consultar a API da Comdinheiro: {e}")



