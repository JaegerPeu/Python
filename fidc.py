import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Dashboard FIDC", layout="wide")

# === Leitura dos dados ===
@st.cache_data
def load_data():
    df = pd.read_excel(r"C:\Users\PedroAugustoBernarde\Downloads\fidc.xlsx")
    # Padroniza nomes de colunas, se necessário
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    return df

df = load_data()

# Garante colunas esperadas (ajuste se os nomes forem diferentes)
# Esperado: Data_posição, Ativo, PRODUTO, Cota, Industry, %_PL, Origem, W%, PL_FUNDO
# Renomeia % PL e PL FUNDO para facilitar
rename_map = {
    "% PL": "%_PL",
    "PL FUNDO": "PL_FUNDO",
    "Data_posição": "Data_Posicao"
}
df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

# === Sidebar: filtros ===
st.sidebar.header("Filtros")

origens = ["Todos"] + sorted([o for o in df["Origem"].dropna().unique()])
origem_sel = st.sidebar.selectbox("Origem", origens)

cotas = ["Todas"] + sorted([c for c in df["Cota"].dropna().unique()])
cota_sel = st.sidebar.selectbox("Tipo de Cota", cotas)

industrys = ["Todas"] + sorted([i for i in df["Industry"].dropna().unique()])
industry_sel = st.sidebar.selectbox("Industry", industrys)

# Filtro por faixa de % PL
min_pl, max_pl = float(df["%_PL"].min()), float(df["%_PL"].max())
faixa_pl = st.sidebar.slider(
    "Faixa de % PL",
    min_value=float(0),
    max_value=float(round(max_pl, 2)),
    value=(float(0), float(round(max_pl, 2)))
)

# Aplica filtros
df_filt = df.copy()
if origem_sel != "Todos":
    df_filt = df_filt[df_filt["Origem"] == origem_sel]
if cota_sel != "Todas":
    df_filt = df_filt[df_filt["Cota"] == cota_sel]
if industry_sel != "Todas":
    df_filt = df_filt[df_filt["Industry"] == industry_sel]
df_filt = df_filt[(df_filt["%_PL"] >= faixa_pl[0]) & (df_filt["%_PL"] <= faixa_pl[1])]

# === KPIs no topo ===
st.title("Dashboard FIDC")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Soma % PL (filtro)", f"{df_filt['%_PL'].sum():.2f}%")
with col2:
    if "PL_FUNDO" in df_filt.columns:
        st.metric("Soma PL FUNDO (filtro)", f"{df_filt['PL_FUNDO'].sum():.2f}%")
with col3:
    st.metric("Qtd. Ativos (filtro)", df_filt["Ativo"].nunique())

# === Gráficos principais ===
col_g1, col_g2 = st.columns(2)

# Tree map: hierarquia Origem -> Industry -> Ativo
with col_g1:
    st.subheader("Tree map por Origem / Industry / Ativo")
    if not df_filt.empty:
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
            hover_data={"%_PL": ":.2f", "PL_FUNDO": ":.2f"} if "PL_FUNDO" in df_filt.columns else {"%_PL": ":.2f"},
        )
        fig_treemap.update_layout(margin=dict(t=30, l=0, r=0, b=0))
        st.plotly_chart(fig_treemap, use_container_width=True)
    else:
        st.info("Nenhum dado após os filtros.")

# Pizza por Industry
with col_g2:
    st.subheader("Distribuição por Industry (% PL)")
    if not df_filt.empty:
        df_ind = (
            df_filt.groupby("Industry", dropna=False)["%_PL"]
            .sum()
            .reset_index()
            .rename(columns={"%_PL": "PL_Industry"})
        )
        df_ind["Industry"] = df_ind["Industry"].fillna("Sem industry")
        fig_pizza = px.pie(
            df_ind,
            names="Industry",
            values="PL_Industry",
            hole=0.3,
        )
        st.plotly_chart(fig_pizza, use_container_width=True)
    else:
        st.info("Nenhum dado após os filtros.")

# === Top N ativos por % PL ===
st.subheader("Top ativos por % PL")
top_n = st.slider("Quantidade de ativos no ranking", 5, 30, 10)
if not df_filt.empty:
    df_top = df_filt.sort_values("%_PL", ascending=False).head(top_n)
    fig_bar = px.bar(
        df_top.sort_values("%_PL"),
        x="%_PL",
        y="Ativo",
        color="Cota",
        orientation="h",
        text="%_PL",
    )
    fig_bar.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig_bar.update_layout(margin=dict(l=0, r=20, t=30, b=0))
    st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.info("Nenhum dado após os filtros.")

# # === Tabela detalhada ===
# st.subheader("Tabela detalhada (dados filtrados)")
# st.dataframe(
#     df_filt.sort_values("%_PL", ascending=False),
#     use_container_width=True,
#     hide_index=True,
# )
