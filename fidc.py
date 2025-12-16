import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Dashboard FIDC", layout="wide")

# === Leitura dos dados ===
@st.cache_data
def load_data():
    df = pd.read_excel("fidc.xlsx")

    # Padroniza nomes de colunas: tira espaços e acentos básicos
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("ç", "c")
        .str.replace("ã", "a")
        .str.replace("á", "a")
        .str.replace("é", "e")
        .str.replace("í", "i")
        .str.replace("ó", "o")
        .str.replace("ú", "u")
    )

    # Renomeia algumas chaves importantes para ficar consistente
    rename_map = {
        "Data_posicao": "Data_Posicao",
        "%_PL": "%_PL",          # já vem como %_PL depois do replace
        "PL_FUNDO": "PL_FUNDO",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Garante tipos
    if "Data_Posicao" in df.columns:
        df["Data_Posicao"] = pd.to_datetime(df["Data_Posicao"], errors="coerce")

    # Converte %_PL e PL_FUNDO para número (tirando % se vier como texto)
    if "%_PL" in df.columns:
        df["%_PL"] = (
            df["%_PL"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        df["%_PL"] = pd.to_numeric(df["%_PL"], errors="coerce")

    if "PL_FUNDO" in df.columns:
        df["PL_FUNDO"] = (
            df["PL_FUNDO"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        df["PL_FUNDO"] = pd.to_numeric(df["PL_FUNDO"], errors="coerce")

    return df

df = load_data()

# === Sidebar: filtros ===
st.sidebar.header("Filtros")

# Filtro de Data de Posição
if "Data_Posicao" in df.columns:
    datas = sorted(df["Data_Posicao"].dropna().unique())
    data_sel = st.sidebar.selectbox(
        "Data de posição",
        options=datas,
        format_func=lambda d: d.strftime("%d/%m/%Y"),
    )
else:
    data_sel = None

origens = ["Todos"] + sorted([o for o in df["Origem"].dropna().unique()]) if "Origem" in df.columns else ["Todos"]
origem_sel = st.sidebar.selectbox("Origem", origens)

cotas = ["Todas"] + sorted([c for c in df["Cota"].dropna().unique()]) if "Cota" in df.columns else ["Todas"]
cota_sel = st.sidebar.selectbox("Tipo de Cota", cotas)

industrys = ["Todas"] + sorted([i for i in df["Industry"].dropna().unique()]) if "Industry" in df.columns else ["Todas"]
industry_sel = st.sidebar.selectbox("Industry", industrys)

# Filtro por faixa de % PL
if "%_PL" in df.columns:
    min_pl, max_pl = float(df["%_PL"].min()), float(df["%_PL"].max())
    faixa_pl = st.sidebar.slider(
        "Faixa de % PL",
        min_value=float(round(min_pl, 2)),
        max_value=float(round(max_pl, 2)),
        value=(float(0), float(round(max_pl, 2))),
    )
else:
    faixa_pl = (0.0, 100.0)

# Aplica filtros
df_filt = df.copy()

if data_sel is not None:
    df_filt = df_filt[df_filt["Data_Posicao"] == data_sel]

if origem_sel != "Todos" and "Origem" in df_filt.columns:
    df_filt = df_filt[df_filt["Origem"] == origem_sel]

if cota_sel != "Todas" and "Cota" in df_filt.columns:
    df_filt = df_filt[df_filt["Cota"] == cota_sel]

if industry_sel != "Todas" and "Industry" in df_filt.columns:
    df_filt = df_filt[df_filt["Industry"] == industry_sel]

if "%_PL" in df_filt.columns:
    df_filt = df_filt[(df_filt["%_PL"] >= faixa_pl[0]) & (df_filt["%_PL"] <= faixa_pl[1])]

# === KPIs no topo ===
st.title("Dashboard FIDC")

col1, col2, col3 = st.columns(3)
with col1:
    if "%_PL" in df_filt.columns:
        st.metric("Soma % PL (filtro)", f"{df_filt['%_PL'].sum():.2f}%")
    else:
        st.metric("Soma % PL (filtro)", "N/D")

with col2:
    if "PL_FUNDO" in df_filt.columns:
        # Aqui assumindo que PL_FUNDO é % do PL total (se for R$, mude o formato)
        st.metric("Soma PL FUNDO (filtro)", f"{df_filt['PL_FUNDO'].sum():.2f}")
    else:
        st.metric("Soma PL FUNDO (filtro)", "N/D")

with col3:
    if "Ativo" in df_filt.columns:
        st.metric("Qtd. Ativos (filtro)", df_filt["Ativo"].nunique())
    else:
        st.metric("Qtd. Ativos (filtro)", "N/D")

if df_filt.empty:
    st.warning("Nenhum dado após os filtros selecionados.")
    st.stop()

# === Gráficos principais ===
col_g1, col_g2 = st.columns(2)

# Tree map: hierarquia Origem -> Industry -> Ativo
with col_g1:
    st.subheader("Tree map por Origem / Industry / Ativo")
    if {"Origem", "Industry", "Ativo", "%_PL"}.issubset(df_filt.columns):
        fig_treemap = px.treemap(
            df_filt,
            path=[
                df_filt["Origem"].fillna("Sem origem"),
                df_filt["Industry"].fillna("Sem industry"),
                "Ativo",
            ],
            values="%_PL",
            color="Industry",
            color_discrete_sequence=px.colors.qualitative.Set3,
            hover_data={
                "%_PL": ":.2f",
                "PL_FUNDO": ":.2f",
            } if "PL_FUNDO" in df_filt.columns else {"%_PL": ":.2f"},
        )
        fig_treemap.update_layout(margin=dict(t=30, l=0, r=0, b=0))
        st.plotly_chart(fig_treemap, use_container_width=True)
    else:
        st.info("Colunas necessárias para o treemap não estão disponíveis.")

# Pizza por Industry
with col_g2:
    st.subheader("Distribuição por Industry (% PL)")
    if "Industry" in df_filt.columns and "%_PL" in df_filt.columns:
        df_ind = (
            df_filt.groupby("Industry", dropna=False)["%_PL"]
            .sum()
            .reset_index()
            .rename(columns={"%_PL": "PL_Industry"})
        )
        df_ind["Industry"] = df_ind["Industry"].fillna("Sem industry")
        df_ind = df_ind.sort_values("PL_Industry", ascending=False)
        fig_pizza = px.pie(
            df_ind,
            names="Industry",
            values="PL_Industry",
            hole=0.3,
        )
        st.plotly_chart(fig_pizza, use_container_width=True)
    else:
        st.info("Colunas necessárias para o gráfico de pizza não estão disponíveis.")

# === Top N ativos por % PL ===
st.subheader("Top ativos por % PL")
top_n = st.slider("Quantidade de ativos no ranking", 5, 30, 10)

if "%_PL" in df_filt.columns and "Ativo" in df_filt.columns:
    df_top = df_filt.sort_values("%_PL", ascending=False).head(top_n)
    fig_bar = px.bar(
        df_top.sort_values("%_PL"),
        x="%_PL",
        y="Ativo",
        color="Cota" if "Cota" in df_top.columns else None,
        orientation="h",
        text="%_PL",
    )
    fig_bar.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig_bar.update_layout(margin=dict(l=0, r=20, t=30, b=0))
    st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.info("Colunas necessárias para o ranking de ativos não estão disponíveis.")

# # Tabela detalhada (opcional)
# st.subheader("Tabela detalhada (dados filtrados)")
# st.dataframe(
#     df_filt.sort_values("%_PL", ascending=False),
#     use_container_width=True,
#     hide_index=True,
# )
