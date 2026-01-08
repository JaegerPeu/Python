import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests

st.set_page_config(page_title="Monitoramento FIDC", layout="wide")
st.title("Monitoramento – Solutions FIDC")

# =========================
# Sidebar: entrada e opções
# =========================
st.sidebar.header("Configurações")

uploaded_file = st.sidebar.file_uploader(
    "Carregue a base do fundo (Excel com abas macro e micro)",
    type=["xlsx", "xls"]
)

# =========================
# Utils
# =========================
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    # normalização simples (sem depender de unidecode)
    def _norm(s: str) -> str:
        s = str(s).strip()
        repl = {
            " ": "_",
            "%": "pct",
            "ç": "c", "Ç": "C",
            "ã": "a", "Ã": "A",
            "á": "a", "Á": "A",
            "à": "a", "À": "A",
            "â": "a", "Â": "A",
            "é": "e", "É": "E",
            "ê": "e", "Ê": "E",
            "í": "i", "Í": "I",
            "ó": "o", "Ó": "O",
            "ô": "o", "Ô": "O",
            "õ": "o", "Õ": "O",
            "ú": "u", "Ú": "U",
            "–": "_", "—": "_",
        }
        for a, b in repl.items():
            s = s.replace(a, b)
        return s
    df = df.copy()
    df.columns = [_norm(c) for c in df.columns]
    return df


def to_numeric_ptbr(series: pd.Series) -> pd.Series:
    # aceita número já numérico e também strings "1.234,56"
    if series.dtype != "object":
        return pd.to_numeric(series, errors="coerce")
    return pd.to_numeric(
        series.astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False),
        errors="coerce"
    )


def parse_data_posicao(series: pd.Series) -> pd.Series:
    # se já for datetime, ok; se vier string, tenta converter
    s = pd.to_datetime(series, errors="coerce", dayfirst=True)
    # fallback (caso apareça mmddyy/inteiro sem separador)
    mask = s.isna()
    if mask.any():
        raw = series.astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)
        s2 = pd.to_datetime(raw, format="%m%d%y", errors="coerce")
        s.loc[mask] = s2.loc[mask]
    return s


@st.cache_data
def load_excel(file) -> tuple[pd.DataFrame, pd.DataFrame]:
    sheets = pd.read_excel(file, sheet_name=None)
    macro_key = None
    micro_key = None
    for k in sheets.keys():
        kl = str(k).strip().lower()
        if "macro" in kl:
            macro_key = k
        if "micro" in kl:
            micro_key = k
    if macro_key is None or micro_key is None:
        raise ValueError(f"Não encontrei abas 'macro' e 'micro'. Abas: {list(sheets.keys())}")

    df_macro = sheets[macro_key].copy()
    df_micro = sheets[micro_key].copy()
    return df_macro, df_micro


def standardize_micro(df_micro_raw: pd.DataFrame) -> pd.DataFrame:
    df = normalize_cols(df_micro_raw)

    # mapeia nomes do seu arquivo para nomes internos
    rename_map = {
        "Data_posição": "Data_posicao",
        "Data_posicao": "Data_posicao",
        "Forma_de_condominio": "Forma_condominio",
        "Forma_de_condomínio": "Forma_condominio",
        "Forma_de_condominio_": "Forma_condominio",
        "Forma_de_condominio__": "Forma_condominio",

        # pesos e colunas chave
        "pctPL": "pct_PL",

        # do seu arquivo: "PL FUNDO" -> "PL_FUNDO" após normalize
        "PL_FUNDO": "PL_FUNDO",

        "Sub_Ponderada": "Sub_Ponderada",
        "PDD_Ponderada": "PDD_Ponderada",
    }

    rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    if "Data_posicao" in df.columns:
        df["Data_posicao"] = parse_data_posicao(df["Data_posicao"])

    for c in ["PDD", "Subordinacao", "Subordinação", "pct_PL", "Sub_Ponderada", "PDD_Ponderada"]:
        if c in df.columns:
            df[c] = to_numeric_ptbr(df[c])

    if "PL_FUNDO" in df.columns:
        df["PL_FUNDO"] = to_numeric_ptbr(df["PL_FUNDO"])

    return df


def standardize_macro(df_macro_raw: pd.DataFrame) -> pd.DataFrame:
    df = normalize_cols(df_macro_raw)
    df = df.loc[:, ~df.columns.astype(str).str.lower().str.startswith("unnamed")]

    rename_map = {
        "Valor_PL": "PL_macro",
        "pct": "pct",
        "Carrego": "Carrego",
        "Subordinacao": "Subordinacao",
        "Subordinação": "Subordinacao",
        "PDD": "PDD",
        "Ativo": "Ativo",
    }
    rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    for c in ["PL_macro", "pct", "Carrego", "Subordinacao", "PDD"]:
        if c in df.columns:
            df[c] = to_numeric_ptbr(df[c])

    return df


# =========================
# Carregamento
# =========================
if uploaded_file is None:
    st.info("Carregue o arquivo para iniciar a análise.")
    st.stop()

try:
    df_macro_raw, df_micro_raw = load_excel(uploaded_file)
except Exception as e:
    st.error(f"Erro ao ler o Excel: {e}")
    st.stop()

df_micro_all = standardize_micro(df_micro_raw)
df_macro = standardize_macro(df_macro_raw)

# =========================
# Validações / Datas
# =========================
if "Data_posicao" not in df_micro_all.columns:
    st.error("Não encontrei a coluna de data na aba micro (esperado 'Data_posição' / 'Data_posicao').")
    st.stop()

dates = sorted([d for d in df_micro_all["Data_posicao"].dropna().unique()])
if len(dates) == 0:
    st.error("Não consegui interpretar nenhuma data em Data_posicao.")
    st.stop()

min_date = pd.to_datetime(min(dates)).date()
max_date = pd.to_datetime(max(dates)).date()

sel_date = st.sidebar.date_input(
    "Data (snapshot)",
    value=max_date,
    min_value=min_date,
    max_value=max_date
)
sel_date = pd.to_datetime(sel_date)

date_range = st.sidebar.date_input(
    "Período (evolução)",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    dt_ini = pd.to_datetime(date_range[0])
    dt_fim = pd.to_datetime(date_range[1])
else:
    dt_ini, dt_fim = pd.to_datetime(min_date), pd.to_datetime(max_date)

df_micro_current = df_micro_all[df_micro_all["Data_posicao"] == sel_date].copy()
df_micro_hist = df_micro_all[(df_micro_all["Data_posicao"] >= dt_ini) & (df_micro_all["Data_posicao"] <= dt_fim)].copy()

# =========================
# KPIs Snapshot (dia)
# =========================
def kpis_snapshot(df_micro_day: pd.DataFrame, df_macro_: pd.DataFrame) -> dict:
    # PL do fundo no dia (somando PL FUNDO por linha/ativo)
    pl_micro = df_micro_day["PL_FUNDO"].sum() if "PL_FUNDO" in df_micro_day.columns else np.nan

    # contagens
    if all(c in df_micro_day.columns for c in ["PRODUTO", "Ativo"]):
        n_fidcs = df_micro_day.loc[df_micro_day["PRODUTO"].astype(str).str.upper().eq("FIDC"), "Ativo"].nunique()
    else:
        n_fidcs = np.nan

    n_ativos_micro = df_micro_day["Ativo"].nunique() if "Ativo" in df_micro_day.columns else np.nan
    n_ativos_macro = df_macro_["Ativo"].nunique() if "Ativo" in df_macro_.columns else np.nan

    # riscos ponderados (já vêm ponderados na planilha)
    pdd_micro = df_micro_day["PDD_Ponderada"].sum() if "PDD_Ponderada" in df_micro_day.columns else np.nan
    sub_micro = df_micro_day["Sub_Ponderada"].sum() if "Sub_Ponderada" in df_micro_day.columns else np.nan

    # Macro (não histórico, pois macro no seu arquivo não tem data)
    TX_ADM = 0.008  # 0,8%

    if all(c in df_macro_.columns for c in ["pct", "Carrego"]):
        cdi_plus = (df_macro_["pct"] * df_macro_["Carrego"]).sum()
        cdi_liq = ((1 + cdi_plus) / (1 + TX_ADM)) - 1
    else:
        cdi_plus = np.nan
        cdi_liq = np.nan

    if all(c in df_macro_.columns for c in ["pct", "Subordinacao"]):
        sub_macro = (df_macro_["pct"] * df_macro_["Subordinacao"]).sum()
    else:
        sub_macro = np.nan

    if all(c in df_macro_.columns for c in ["pct", "PDD"]):
        pdd_macro = (df_macro_["pct"] * df_macro_["PDD"]).sum()
    else:
        pdd_macro = np.nan

    return dict(
        pl_micro=pl_micro,
        n_fidcs=n_fidcs,
        n_ativos_micro=n_ativos_micro,
        n_ativos_macro=n_ativos_macro,
        pdd_micro=pdd_micro,
        sub_micro=sub_micro,
        cdi_plus=cdi_plus,
        cdi_liq=cdi_liq,
        sub_macro=sub_macro,
        pdd_macro=pdd_macro,
        tx_adm=TX_ADM
    )

kpi = kpis_snapshot(df_micro_current, df_macro)

def fmt_brl(x):
    if pd.isna(x):
        return "-"
    s = f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {s}"

def fmt_pct(x):
    if pd.isna(x):
        return "-"
    return f"{x*100:,.2f} %".replace(",", "X").replace(".", ",").replace("X", ".")

# ============================================================
# >>> AJUSTE: PL do card via Comdinheiro (na data do snapshot)
# ============================================================
@st.cache_data(ttl=60*30)
def carregar_pl_comdinheiro(username: str, password: str) -> pd.DataFrame:
    # doc: Comdinheiro usa POST com username/password, endpoint import_data e retorno em json/xml [page:2]
    url = "https://api.comdinheiro.com.br/v1/ep1/import-data"

    payload = (
        f"username={username}"
        f"&password={password}"
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
    data = r.json()

    tables = data.get("tables", {})
    tab0 = tables.get("tab0", {})
    rows = []
    for key in sorted(tab0.keys(), key=lambda x: int(x.replace("lin", ""))):
        row = tab0[key]
        rows.append([row.get("col0"), row.get("col1")])

    header = rows[0]
    data_rows = rows[1:]
    df_pl = pd.DataFrame(data_rows, columns=header)

    rename_map = {}
    for c in df_pl.columns:
        cl = str(c).lower()
        if "data" in cl:
            rename_map[c] = "Data"
        if "patrim" in cl:
            rename_map[c] = "PL"
    df_pl = df_pl.rename(columns=rename_map)

    if not all(c in df_pl.columns for c in ["Data", "PL"]):
        raise ValueError("Resposta da API sem colunas Data/PL.")

    df_pl["Data"] = pd.to_datetime(df_pl["Data"], format="%d/%m/%Y", errors="coerce").dt.normalize()
    df_pl["PL"] = to_numeric_ptbr(df_pl["PL"])
    df_pl = df_pl.dropna(subset=["Data"]).sort_values("Data")
    return df_pl


def pl_cmd_no_dia(df_pl_cmd: pd.DataFrame, sel_date_: pd.Timestamp):
    if df_pl_cmd is None or df_pl_cmd.empty:
        return np.nan, None

    d = pd.to_datetime(sel_date_).normalize()
    tmp = df_pl_cmd.copy()
    tmp = tmp.dropna(subset=["Data", "PL"]).sort_values("Data")

    exact = tmp[tmp["Data"] == d]
    if not exact.empty:
        return float(exact["PL"].iloc[-1]), "exata"

    prev = tmp[tmp["Data"] <= d]
    if not prev.empty:
        return float(prev["PL"].iloc[-1]), "anterior"

    return np.nan, None


df_pl_cmd = None
pl_fundo_cmd = np.nan
pl_fundo_status = None

# st.secrets é o local correto para credenciais no Streamlit [page:1]
user = st.secrets.get("COMDINHEIRO_USER", "")
pwd = st.secrets.get("COMDINHEIRO_PASS", "")

if user and pwd:
    try:
        df_pl_cmd = carregar_pl_comdinheiro(user, pwd)
        pl_fundo_cmd, pl_fundo_status = pl_cmd_no_dia(df_pl_cmd, sel_date)
    except Exception:
        df_pl_cmd = None
        pl_fundo_cmd = np.nan
        pl_fundo_status = None

# =========================
# Cards
# =========================
c1, c2, c3, c4 = st.columns(4)
c5, c6, c7, c8 = st.columns(4)

# >>> AJUSTE AQUI: card de PL agora usa Comdinheiro (se disponível)
if not pd.isna(pl_fundo_cmd):
    titulo_pl = "PL do fundo"
    if pl_fundo_status == "anterior":
        titulo_pl += " (último disp.)"
    c1.metric(titulo_pl, fmt_brl(pl_fundo_cmd))
else:
    # fallback: mantém o comportamento antigo
    c1.metric("PL (micro, soma PL FUNDO)", fmt_brl(kpi["pl_micro"]))

c3.metric("Nº de FIDCs", int(kpi["n_fidcs"]) if not pd.isna(kpi["n_fidcs"]) else "-")
c2.metric("Taxa Adm", "0,8%")
c4.metric("Nº de Ativos (macro)", int(kpi["n_ativos_macro"]) if not pd.isna(kpi["n_ativos_macro"]) else "-")

c5.metric("PDD (ponderado)", fmt_pct(kpi["pdd_micro"]))
c6.metric("Subordinação (ponderado)", fmt_pct(kpi["sub_micro"]))
c7.metric("Carrego: CDI+", fmt_pct(kpi["cdi_plus"]))
c8.metric("Carrego Líq.: CDI+", fmt_pct(kpi["cdi_liq"]))

st.divider()

# =========================
# Evolução dos KPIs (micro)
# =========================
with st.expander("Evolução dos KPIs"):

    agg_map = {}
    if "PL_FUNDO" in df_micro_hist.columns:
        agg_map["PL_FUNDO"] = "sum"
    if "PDD_Ponderada" in df_micro_hist.columns:
        agg_map["PDD_Ponderada"] = "sum"
    if "Sub_Ponderada" in df_micro_hist.columns:
        agg_map["Sub_Ponderada"] = "sum"
    if "Ativo" in df_micro_hist.columns:
        agg_map["Ativo"] = pd.Series.nunique
    if "pct_PL" in df_micro_hist.columns:
        agg_map["pct_PL"] = "sum"

    df_kpi_time = (df_micro_hist
        .groupby("Data_posicao", as_index=False)
        .agg(agg_map)
        .rename(columns={
            "PL_FUNDO": "PL_total",
            "PDD_Ponderada": "PDD_micro",
            "Sub_Ponderada": "Sub_micro",
            "Ativo": "n_ativos",
            "pct_PL": "pct_pl_sum"
        })
        .sort_values("Data_posicao")
    )

    g1, g2, g3 = st.columns(3)

    with g1:
        if "PL_total" in df_kpi_time.columns:
            fig = px.line(df_kpi_time, x="Data_posicao", y="PL_total", markers=True, title="PL total (micro)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Sem coluna PL_FUNDO para calcular PL total.")

    with g2:
        if "PDD_micro" in df_kpi_time.columns:
            fig = px.line(df_kpi_time, x="Data_posicao", y="PDD_micro", markers=True, title="PDD ponderada (micro)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Sem PDD_Ponderada para evolução.")

    with g3:
        if "Sub_micro" in df_kpi_time.columns:
            fig = px.line(df_kpi_time, x="Data_posicao", y="Sub_micro", markers=True, title="Subordinação ponderada (micro)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Sem Sub_Ponderada para evolução.")

    with st.expander("Checagens rápidas da base (micro)"):
        if "pct_pl_sum" in df_kpi_time.columns:
            st.write("Soma de %PL por data (ideal ~ 1.0):")
            st.dataframe(df_kpi_time[["Data_posicao", "pct_pl_sum"]], use_container_width=True)
        st.write("Snapshot:")
        st.dataframe(df_micro_current, use_container_width=True)

st.divider()

# =========================
# Snapshot: gráficos do dia
# =========================
st.subheader("Snapshot (data selecionada)")

st.caption("Concentração de %PL por Gestora")
if all(c in df_micro_current.columns for c in ["Gestora", "Ativo", "pct_PL"]):
    df_treemap = df_micro_current.groupby(["Gestora", "Ativo"], as_index=False)["pct_PL"].sum()
    fig_t = px.treemap(df_treemap, path=["Gestora", "Ativo"], values="pct_PL", color="Gestora")
    fig_t.update_traces(texttemplate="%{label}<br>%{value:.2%}")
    st.plotly_chart(fig_t, use_container_width=True)
else:
    st.warning("Para o treemap preciso de 'Gestora', 'Ativo' e 'pct_PL'.")

colB, colC = st.columns(2)

with colB:
    st.caption("Alocação por setor (Industry)")
    if all(c in df_micro_current.columns for c in ["Industry", "pct_PL"]):
        df_setor = (df_micro_current.groupby("Industry", as_index=False)["pct_PL"]
                    .sum().sort_values("pct_PL", ascending=False))
        fig = px.pie(df_setor, names="Industry", values="pct_PL", title="Distribuição de %PL por setor")
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Para setores preciso de 'Industry' e 'pct_PL'.")

with colC:
    st.caption("Cotas")
    if all(c in df_micro_current.columns for c in ["Cota", "pct_PL"]):
        df_cota = (df_micro_current.groupby("Cota", as_index=False)["pct_PL"]
                   .sum().sort_values("pct_PL", ascending=False))
        fig = px.pie(df_cota, names="Cota", values="pct_PL", title="Distribuição de %PL por tipo de cota")
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Para cotas preciso de 'Cota' e 'pct_PL'.")

colD, colE = st.columns(2)
with colD:
    st.caption("Condomínio")
    if all(c in df_micro_current.columns for c in ["Forma_condominio", "pct_PL"]):
        df_cond = (df_micro_current.groupby("Forma_condominio", as_index=False)["pct_PL"]
                   .sum().sort_values("pct_PL", ascending=False))
        fig = px.pie(df_cond, names="Forma_condominio", values="pct_PL", title="Distribuição de %PL por condomínio")
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Para condomínio preciso de 'Forma_condominio' e 'pct_PL'.")

with colE:
    st.caption("Retorno-alvo (macro)")
    if all(c in df_macro.columns for c in ["Ativo", "pct", "Carrego"]):
        df_ret = df_macro.copy()
        df_ret["peso_carrego"] = df_ret["pct"] * df_ret["Carrego"]
        df_ret["Carrego_label"] = df_ret["Carrego"].apply(lambda x: f"CDI+{x*100:.2f}%")
        fig = px.pie(df_ret, names="Carrego_label", values="peso_carrego", title="Distribuição do carrego (macro)")
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Para retorno-alvo preciso de 'Ativo', 'pct' e 'Carrego' na aba macro.")

st.divider()

# =========================
# (Opcional) Comdinheiro: PL / Cota / Retornos
# =========================
st.subheader("Evolução (Comdinheiro) – opcional")

use_cmd = st.toggle("Consultar Comdinheiro", value=False)

@st.cache_data(ttl=60*30)
def carregar_cotas_cmd(username: str, password: str) -> pd.DataFrame:
    """
    HistoricoCotacao002: retorna numero_indice com base_num_indice=1 para:
    - Fundo (col1)
    - CDI   (col2)
    lin0 é header e vem com col0 vazio, então pulamos lin0 e lemos col0/1/2. [page:2]
    """
    url = "https://api.comdinheiro.com.br/v1/ep1/import-data"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    payload = (
        f"username={username}"
        f"&password={password}"
        "&URL=HistoricoCotacao002.php%3F%26x%3D60800845000193_unica%2BCDI"
        "%26data_ini%3D15072025%26data_fim%3Ddmenos2%26pagina%3D1"
        "%26d%3DMOEDA_ORIGINAL%26g%3D1%26m%3D0"
        "%26info_desejada%3Dnumero_indice"
        "%26retorno%3Ddiscreto%26tipo_data%3Ddu_br"
        "%26tipo_ajuste%3Dtodosajustes%26num_casas%3D8%26enviar_email%3D0"
        "%26ordem_legenda%3D1%26cabecalho_excel%3Dmodo1"
        "%26classes_ativos%3Dfklk448oj5v5r"
        "%26ordem_data%3D0%26rent_acum%3Drent_acum"
        "%26preco_nd_ant%3D0%26base_num_indice%3D1%26flag_num_indice%3D0"
        "%26eixo_x%3DData%26startX%3D0%26max_list_size%3D20"
        "%26line_width%3D2%26tipo_grafico%3Dline%26tooltip%3Dunica"
        "&format=json3"
    )

    r = requests.post(url, data=payload, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()

    tab0 = data.get("tables", {}).get("tab0", {})
    if not tab0:
        raise ValueError("Resposta sem tables.tab0 (sem dados).")

    # lê as linhas (pula header lin0)
    rows = []
    for key in sorted(tab0.keys(), key=lambda x: int(x.replace("lin", ""))):
        if key == "lin0":
            continue
        row = tab0[key]
        rows.append([row.get("col0"), row.get("col1"), row.get("col2")])

    df = pd.DataFrame(rows, columns=["Data", "Fundo", "CDI"])

    df["Data"] = pd.to_datetime(df["Data"], format="%d/%m/%Y", errors="coerce").dt.normalize()
    df["Fundo"] = to_numeric_ptbr(df["Fundo"])
    df["CDI"] = to_numeric_ptbr(df["CDI"])

    df = df.dropna(subset=["Data"]).sort_values("Data")
    return df

def montar_retornos(df_nivel: pd.DataFrame) -> pd.DataFrame:
    df = df_nivel.copy().sort_values("Data")

    out = df[["Data"]].copy()

    # níveis base 1 (já acumulados)
    out["Fundo_nivel"] = df["Fundo"]
    out["CDI_nivel"] = df["CDI"]

    # retorno diário (discreto)
    out["Fundo_ret_diario"] = df["Fundo"].pct_change()
    out["CDI_ret_diario"] = df["CDI"].pct_change()

    # retorno acumulado desde o início da série (base 1)
    out["Fundo_ret_acum"] = df["Fundo"] - 1
    out["CDI_ret_acum"] = df["CDI"] - 1

    return out

if use_cmd:
    user = st.secrets.get("COMDINHEIRO_USER", "")
    pwd = st.secrets.get("COMDINHEIRO_PASS", "")

    if not user or not pwd:
        st.warning("Secrets não configurados. Desmarque o toggle ou configure o .streamlit/secrets.toml.")
    else:
        modo = st.radio(
            "O que plotar?",
            ["PL", "Cota (base 1)", "Retorno diário", "Retorno acumulado"],
            horizontal=True
        )

        try:
            if modo == "PL":
                if df_pl_cmd is None or df_pl_cmd.empty:
                    df_pl_cmd = carregar_pl_comdinheiro(user, pwd)

                fig = px.line(df_pl_cmd, x="Data", y="PL", title="PL via Comdinheiro", markers=True)
                st.plotly_chart(fig, use_container_width=True)

            else:
                df_nivel = carregar_cotas_cmd(user, pwd)
                df_ret = montar_retornos(df_nivel)

                if modo == "Cota (base 1)":
                    fig = px.line(
                        df_ret, x="Data",
                        y=["Fundo_nivel", "CDI_nivel"],
                        title="Cota/Índice (base 1): Fundo vs CDI",
                        markers=True
                    )
                    st.plotly_chart(fig, use_container_width=True)

                elif modo == "Retorno diário":
                    fig = px.line(
                        df_ret, x="Data",
                        y=["Fundo_ret_diario", "CDI_ret_diario"],
                        title="Retorno diário: Fundo vs CDI",
                        markers=False
                    )
                    st.plotly_chart(fig, use_container_width=True)

                else:  # Retorno acumulado
                    fig = px.line(
                        df_ret, x="Data",
                        y=["Fundo_ret_acum", "CDI_ret_acum"],
                        title="Retorno acumulado: Fundo vs CDI",
                        markers=True
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with st.expander("Dados (Comdinheiro)"):
                    st.dataframe(df_ret, use_container_width=True)

        except Exception as e:
            st.error(f"Erro ao consultar Comdinheiro: {e}")
else:
    st.caption("Desligado.")

