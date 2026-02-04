import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date

API_URL = "https://api.comdinheiro.com.br/v1/ep1/import-data"
HEADERS = {"Content-Type": "application/x-www-form-urlencoded"}

CLASSIFICACAO_ATIVOS = {
    "105%CDI": {"Grande Classe": "Renda Fixa", "Classe": "Pós Fixado"},
    "ANBIMA_IRFM": {"Grande Classe": "Renda Fixa", "Classe": "Pré Fixado"},
    "RF_INFLACAO_SWM": {"Grande Classe": "Renda Fixa", "Classe": "Inflação"},
    "ANBIMA_IHFA": {"Grande Classe": "Fundos", "Classe": "Fundos de Gestão Ativa"},
    "IBOV": {"Grande Classe": "Renda Variável", "Classe": "Ações Brasil"},
    "US:SP500": {"Grande Classe": "Renda Variável", "Classe": "Internacional"},
    "IFIX": {"Grande Classe": "Renda Variável", "Classe": "Fundos Imobiliários"},
    "ALTERNATIVO": {"Grande Classe": "Alternativos", "Classe": "Alternativos"},

}

CARACTERISTICAS_PORT = {
    "PORTFOLIO_SWM_I": {
        "Perfil (Suitability)": "Conservador",
        "Objetivo da carteira": "Preservação patrimonial e liquidez",
        "Tipo de investidor": "Investidor Geral",
        "Retorno 06 meses": "-",
        "Retorno 12 meses": "-",
        "Retorno 24 meses": "-",
        "Meses acima do Benchmark": "-",
        "Meses abaixo do Benchmark": "-",
        "Maior rent. mensal": "-",
        "Menor rent. mensal": "-",
    },
    "PORTFOLIO_SWM_II": {
        "Perfil (Suitability)": "Conservador",
        "Objetivo da carteira": "reservação patrimonial com rendimento",
        "Tipo de investidor": "Investidor Geral",
        "Retorno 06 meses": "-",
        "Retorno 12 meses": "-",
        "Retorno 24 meses": "-",
        "Meses acima do Benchmark": "-",
        "Meses abaixo do Benchmark": "-",
        "Maior rent. mensal": "-",
        "Menor rent. mensal": "-",
    },
    "PORTFOLIO_SWM_III": {
        "Perfil (Suitability)": "Balanceado",
        "Objetivo da carteira": "Crescimento conservador com preservação patrimonial",
        "Tipo de investidor": "Investidor Geral",
        "Retorno 06 meses": "-",
        "Retorno 12 meses": "-",
        "Retorno 24 meses": "-",
        "Meses acima do Benchmark": "-",
        "Meses abaixo do Benchmark": "-",
        "Maior rent. mensal": "-",
        "Menor rent. mensal": "-",
    },
    "PORTFOLIO_SWM_IV": {
        "Perfil (Suitability)": "Balanceado",
        "Objetivo da carteira": "Crescimento equilibrado com controle de risco",
        "Tipo de investidor": "Investidor Geral",
        "Retorno 06 meses": "-",
        "Retorno 12 meses": "-",
        "Retorno 24 meses": "-",
        "Meses acima do Benchmark": "-",
        "Meses abaixo do Benchmark": "-",
        "Maior rent. mensal": "-",
        "Menor rent. mensal": "-",
    },
    "PORTFOLIO_SWM_V": {
        "Perfil (Suitability)": "Moderado/Agressivo",
        "Objetivo da carteira": "Crescimento de patrimônio no longo prazo",
        "Tipo de investidor": "Investidor Geral",
        "Retorno 06 meses": "-",
        "Retorno 12 meses": "-",
        "Retorno 24 meses": "-",
        "Meses acima do Benchmark": "-",
        "Meses abaixo do Benchmark": "-",
        "Maior rent. mensal": "-",
        "Menor rent. mensal": "-",
    },
    "PORTFOLIO_SWM_VI": {
        "Perfil (Suitability)": "Agressivo",
        "Objetivo da carteira": "Crescimento agressivo com alta tolerância à volatilidade",
        "Tipo de investidor": "Investidor Geral",
        "Retorno 06 meses": "-",
        "Retorno 12 meses": "-",
        "Retorno 24 meses": "-",
        "Meses acima do Benchmark": "-",
        "Meses abaixo do Benchmark": "-",
        "Maior rent. mensal": "-",
        "Menor rent. mensal": "-",
    },
}

PAYLOAD_RET_DIA = (
    "username=solutionswm&password=Soluti%40ns2025&URL=HistoricoCotacao002.php%3F"
    "%26x%3DPORTFOLIO_SWM_I%2BPORTFOLIO_SWM_II%2BPORTFOLIO_SWM_III%2BPORTFOLIO_SWM_IV"
    "%2BPORTFOLIO_SWM_V%2BPORTFOLIO_SWM_VI%2BEXPLODE%28PORTFOLIO_SWM_V%29"
    "%26data_ini%3D30122022%26data_fim%3D31129999%26pagina%3D1"
    "%26d%3DMOEDA_ORIGINAL%26g%3D1%26m%3D1%26info_desejada%3Dretorno"
    "%26retorno%3Ddiscreto%26tipo_data%3Ddu_br%26tipo_ajuste%3Dtodosajustes"
    "%26num_casas%3D2%26enviar_email%3D0%26ordem_legenda%3D1"
    "%26cabecalho_excel%3Dmodo1%26classes_ativos%3Dfklk448oj5v5r"
    "%26ordem_data%3D0%26rent_acum%3Dnada%26minY%3D%26maxY%3D%26deltaY%3D"
    "%26preco_nd_ant%3D0%26base_num_indice%3D100%26flag_num_indice%3D0"
    "%26eixo_x%3DData%26startX%3D0%26max_list_size%3D20"
    "%26line_width%3D2%26titulo_grafico%3D%26legenda_eixoy%3D"
    "%26tipo_grafico%3Dline%26script%3D%26tooltip%3Dunica&format=json3"
)

PAYLOAD_PROP = (
    "username=solutionswm&password=Soluti%40ns2025&URL=ComparaFundos001.php%3F%26datas%3D31%2F12%2F9999%26cnpjs%3DEXPLODE%28PORTFOLIO_SWM_I%29%26indicadores%3Dexplode%28composicao_portfolio_SWM%29%2Bpapel%26num_casas%3D2%26pc%3Dnome_fundo%26flag_transpor%3D0%26enviar_email%3D0%26mostrar_da%3D0%26op01%3Dtabela%26oculta_cabecalho_sup%3D0%26relat_alias_automatico%3Dcmd_alias_01&format=json3"
)

@st.cache_data(show_spinner=True)
def fetch_data(payload: str, tab_name: str = "tab0") -> pd.DataFrame:
    response = requests.post(API_URL, data=payload, headers=HEADERS)
    response.raise_for_status()
    json_data = response.json()
    tab = json_data["tables"][tab_name]

    headers = [tab["lin0"][col] for col in sorted(tab["lin0"].keys())]
    rows = []
    for i in range(1, len(tab)):
        linha = tab.get(f"lin{i}", None)
        if linha:
            row = [linha[col] for col in sorted(linha.keys())]
            rows.append(row)

    df = pd.DataFrame(rows, columns=headers)
    return df

def tratar_df_numerico(df: pd.DataFrame, colunas_ignoradas: list) -> pd.DataFrame:
    """Converte colunas numéricas de forma segura, lidando com strings e números misturados"""
    colunas_numericas = [c for c in df.columns if c not in colunas_ignoradas]
    df_copy = df.copy()
    
    for col in colunas_numericas:
        # Sempre converte para string primeiro para aplicar .str methods
        df_copy[col] = (
            df_copy[col]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .str.replace("%", "", regex=False)
            .str.strip()
        )
        df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")
    
    # Preenche NaN com 0 apenas nas colunas numéricas
    df_copy[colunas_numericas] = df_copy[colunas_numericas].fillna(0)
    return df_copy

def criar_sunburst(df_prop_class: pd.DataFrame, portfolio: str):
    df_sunburst = df_prop_class[df_prop_class[portfolio] > 0].copy()
    df_sunburst = df_sunburst[["Grande Classe", "Classe", "nome_fundo", portfolio]].copy()
    df_sunburst.columns = ["Grande Classe", "Classe", "Ativo", "Proporcao"]
    df_sunburst = df_sunburst.dropna(subset=["Proporcao"])
    df_sunburst = df_sunburst[df_sunburst["Proporcao"] > 0]

    total_prop = df_sunburst["Proporcao"].sum()
    if total_prop > 0:
        df_sunburst["Proporcao"] = df_sunburst["Proporcao"] / total_prop

    fig = px.sunburst(
        df_sunburst,
        path=["Grande Classe", "Classe", "Ativo"],
        values="Proporcao",
        color="Grande Classe",
        color_discrete_map={
            "Fundos": "#404751",
            "Renda Fixa": "#102134",
            "Renda Variável": "#896f3d",
            "Alternativos": "#FFFFFF",
            "Outros": "#A8A8A8"
        },
        hover_data={"Proporcao": ":.2%"},
        title=f"Composição do Portfolio: {portfolio}"
    )
    return df_sunburst, fig



    
    
def gerar_alocacao_por_categoria(df_prop_class: pd.DataFrame, portfolio: str):
    df_alocacao = df_prop_class[["Identificador", "Grande Classe", portfolio]].copy()
    df_alocacao = df_alocacao[df_alocacao[portfolio] > 0]
    alocacao_cat = df_alocacao.groupby("Grande Classe")[portfolio].sum().reset_index()
    alocacao_cat.columns = ["Categoria", "Alocação"]
    alocacao_cat = alocacao_cat.sort_values("Alocação", ascending=False)
    alocacao_cat["Alocação %"] = alocacao_cat["Alocação"] / alocacao_cat["Alocação"].sum()
    return alocacao_cat[["Categoria", "Alocação %"]].reset_index(drop=True)

def gerar_distribuicao_risco(df_prop_class: pd.DataFrame, portfolio: str):
    df_dist = df_prop_class[["Identificador", "Classe", portfolio]].copy()
    df_dist = df_dist[df_dist[portfolio] > 0]
    dist_classe = df_dist.groupby("Classe")[portfolio].sum().reset_index()
    dist_classe.columns = ["Classe de Ativo", "Distribuição"]
    dist_classe = dist_classe.sort_values("Distribuição", ascending=False)
    dist_classe["Distribuição %"] = dist_classe["Distribuição"] / dist_classe["Distribuição"].sum()
    return dist_classe[["Classe de Ativo", "Distribuição %"]].reset_index(drop=True)

def gerar_retorno_mensal(df_cota: pd.DataFrame) -> pd.DataFrame:
    if df_cota.empty:
        return pd.DataFrame()

    df_mensal = df_cota.copy()
    df_mensal["Ano"] = df_mensal["Data"].dt.year
    df_mensal["Mes"] = df_mensal["Data"].dt.month
    df_mensal["AnoMes"] = df_mensal["Data"].dt.to_period("M")

    ret_mensal = df_mensal.groupby(["Ano", "Mes", "AnoMes"])["Retorno Portfolio"].apply(
        lambda x: (1 + x).prod() - 1
    ).reset_index()

    ret_mensal["Mes_Nome"] = ret_mensal["AnoMes"].dt.strftime("%b/%Y")
    ret_mensal = ret_mensal[["Mes_Nome", "Retorno Portfolio"]].copy()
    ret_mensal.columns = ["Período", "Retorno Mensal"]
    ret_mensal = ret_mensal.sort_values("Período", ascending=False).reset_index(drop=True)

    ret_mensal["Retorno Mensal"] = ret_mensal["Retorno Mensal"].apply(lambda x: f"{x:.2%}")

    return ret_mensal

def main():
    st.set_page_config(page_title="SWM - Backtest", layout="wide")
    st.title("SWM | Portfolio")
    st.subheader("Uso interno")

    MOSTRAR_CONSULTAS = False

    st.markdown(
        """
        <style>
        [data-testid="stMetricValue"] {
            white-space: normal !important;
            overflow-wrap: break-word !important;
            font-size: 1.05rem !important;
        }
        [data-testid="metric-container"] {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 8px;
            border: 2px solid #e1e5e9;
            margin: 0.5rem 0;
        }
        [data-testid="metric-container"] .stMetric > label {
            font-weight: bold !important;
            color: #1f2937 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if MOSTRAR_CONSULTAS:
        tab_consultas, tab_laminas = st.tabs(["Consultas", "Lâminas"])
    else:
        tab_laminas, = st.tabs(["Lâminas"])

    # CARREGAMENTO DE DADOS - VERSÃO CORRIGIDA
    with st.spinner("Carregando dados base..."):
        df_ret_raw = fetch_data(PAYLOAD_RET_DIA, tab_name="tab1")
        df_prop_raw = fetch_data(PAYLOAD_PROP, tab_name="tab0")

        # Processamento seguro dos retornos
        df_ret = df_ret_raw.copy()



        df_ret = tratar_df_numerico(df_ret, colunas_ignoradas=["Data"])
        
        # Conversão segura da coluna de data
        if "Data" in df_ret_raw.columns:
            df_ret["Data"] = pd.to_datetime(df_ret_raw["Data"], format="%d/%m/%Y", errors="coerce")

        # Processamento das proporções
        df_prop = df_prop_raw.copy()
        df_prop.columns = df_prop.columns.str.strip()
        
        # Tratamento seguro da coluna Identificador
        if "Identificador" in df_prop.columns:
            df_prop["Identificador"] = (
                df_prop["Identificador"].astype(str).str.strip()
            )
            df_prop["Identificador"] = df_prop["Identificador"].replace("nan", "").replace("", "")
            mask_vazio = df_prop["Identificador"] == ""
            df_prop.loc[mask_vazio, "Identificador"] = (
                df_prop.loc[mask_vazio, "nome_fundo"].astype(str).str.strip()
            )
        else:
            df_prop["Identificador"] = df_prop["nome_fundo"].astype(str).str.strip()

        if "nome_fundo" in df_prop.columns:
            df_prop["nome_fundo"] = df_prop["nome_fundo"].astype(str).str.strip()

        # Processamento numérico seguro das proporções
        df_prop = tratar_df_numerico(df_prop, colunas_ignoradas=["nome_fundo", "Identificador"])
        
        # Converte proporções para formato decimal (0-1)
        for col in df_prop.columns:
            if col not in ["nome_fundo", "Identificador"]:
                df_prop[col] = df_prop[col] / 100.0

        # Classificação dos ativos
        df_prop_class = df_prop.copy()
        df_prop_class["Grande Classe"] = df_prop_class["Identificador"].map(
            lambda x: CLASSIFICACAO_ATIVOS.get(x.strip(), {}).get("Grande Classe", "Outros")
        )
        df_prop_class["Classe"] = df_prop_class["Identificador"].map(
            lambda x: CLASSIFICACAO_ATIVOS.get(x.strip(), {}).get("Classe", "Outros")
        )
        mask_na_grande = df_prop_class["Grande Classe"] == "Outros"
        df_prop_class.loc[mask_na_grande, "Grande Classe"] = df_prop_class.loc[mask_na_grande, "nome_fundo"].map(
            lambda x: CLASSIFICACAO_ATIVOS.get(x.strip(), {}).get("Grande Classe", "Outros")
        )
        df_prop_class.loc[mask_na_grande, "Classe"] = df_prop_class.loc[mask_na_grande, "nome_fundo"].map(
            lambda x: CLASSIFICACAO_ATIVOS.get(x.strip(), {}).get("Classe", "Outros")
        )

        cols_display = ["Identificador", "nome_fundo", "Grande Classe", "Classe"] + [
            c for c in df_prop_class.columns if c not in ["Identificador", "nome_fundo", "Grande Classe", "Classe"]
        ]
        df_prop_class = df_prop_class[cols_display]

    st.markdown("---")

    # CONSULTAS (opcional)
    if MOSTRAR_CONSULTAS:
        with tab_consultas:
            st.subheader("📈 Payload 1 - Retorno diário (bruto)")
            st.dataframe(df_ret_raw, use_container_width=True, height=300)
            st.subheader("📊 Payload 2 - Proporção (bruto)")
            st.dataframe(df_prop_raw, use_container_width=True, height=300)

    # LÂMINAS - MAIN CONTENT
    with tab_laminas:
        portfolios = [
            col for col in df_prop_class.columns
            if col not in ["Identificador", "nome_fundo", "Grande Classe", "Classe"]
        ]

        df_bench = (
            df_prop_class[["nome_fundo", "Identificador"]]
            .dropna()
            .drop_duplicates()
        )

        df_bench = df_bench[df_bench["Identificador"].isin(df_ret.columns)]
        benchmarks_nomes = sorted(df_bench["nome_fundo"].tolist())

        st.subheader("Seletores")

        col_port, col_bench = st.columns(2)

        with col_port:
            if "portfolio_sel" not in st.session_state:
                st.session_state.portfolio_sel = portfolios[0]

            portfolio_sel = st.selectbox(
                "Portfolio:",
                options=portfolios,
                index=portfolios.index(st.session_state.portfolio_sel),
                key="portfolio_sel_global",
            )
            st.session_state.portfolio_sel = portfolio_sel

        with col_bench:
            if "benchmark_nome_sel" not in st.session_state:
                st.session_state.benchmark_nome_sel = "Nenhum"

            benchmark_options = ["Nenhum"] + benchmarks_nomes
            try:
                benchmark_idx = benchmark_options.index(st.session_state.benchmark_nome_sel)
            except ValueError:
                benchmark_idx = 0

            benchmark_nome_sel = st.selectbox(
                "Benchmark:",
                options=benchmark_options,
                index=benchmark_idx,
                key="benchmark_sel_global",
            )
            st.session_state.benchmark_nome_sel = benchmark_nome_sel

            if benchmark_nome_sel != "Nenhum":
                linha_bench = df_bench[df_bench["nome_fundo"] == benchmark_nome_sel]
                benchmark_ident = (
                    linha_bench["Identificador"].iloc[0]
                    if not linha_bench.empty
                    else None
                )
            else:
                benchmark_ident = None

        st.markdown("---")

        # PROPORÇÃO (BACKTEST) - 4 colunas por linha
        with st.expander("Proporção (Backtest)", expanded=False):
            proporcoes_orig = df_prop_class[["Identificador", "nome_fundo", portfolio_sel]].copy()
            proporcoes_orig.columns = ["Identificador", "nome_fundo", "Proporcao_original"]

            st.markdown(
                "Edite as proporções (%) por ativo para backtest "
                "(a soma não precisa ser 100%, normalizamos internamente nesta sessão)."
            )

            proporcoes_editadas = []
            N_COLS = 4
            cols_line = st.columns(N_COLS)

            for idx, (i, row) in enumerate(proporcoes_orig.iterrows()):
                if idx % N_COLS == 0 and idx != 0:
                    cols_line = st.columns(N_COLS)
                col = cols_line[idx % N_COLS]

                valor_default = float(row["Proporcao_original"] * 100)

                with col:
                    novo_valor = st.number_input(
                        f"{row['nome_fundo']}",
                        min_value=0.0,
                        max_value=100.0,
                        value=valor_default,
                        step=0.5,
                        key=f"prop_{portfolio_sel}_{i}",
                    )
                    proporcoes_editadas.append(novo_valor / 100.0)

            soma_pct = sum(proporcoes_editadas) * 100
            st.write(
                f"Soma das proporções informadas: *{soma_pct:.2f}%* "
                "(será normalizada para 100% nos cálculos desta tela)"
            )

        # Atualiza df_prop_class com proporções editadas
        df_prop_class_atual = df_prop_class.copy()
        if len(proporcoes_editadas) == len(proporcoes_orig):
            df_prop_class_atual[portfolio_sel] = proporcoes_editadas
        else:
            df_prop_class_atual[portfolio_sel] = df_prop_class[portfolio_sel]

        # RETORNOS DIÁRIOS PORTFÓLIO + BENCHMARK
        pesos = df_prop_class_atual[["Identificador", portfolio_sel]].copy()
        ativos_portfolio = pesos[pesos[portfolio_sel] > 0]["Identificador"].tolist()

        cols_ativos_portfolio = [ident for ident in ativos_portfolio if ident in df_ret.columns]

        cols_ret = ["Data"] + cols_ativos_portfolio

        if benchmark_ident and benchmark_ident in df_ret.columns:
            if benchmark_ident not in cols_ret:
                cols_ret.append(benchmark_ident)

        df_ret_port = df_ret[cols_ret].copy()

        if cols_ativos_portfolio:
            mat_ret = df_ret_port[cols_ativos_portfolio]
            w = (
                pesos.set_index("Identificador")[portfolio_sel]
                .reindex(cols_ativos_portfolio)
                .fillna(0.0)
            )
            df_ret_port["Retorno Portfolio"] = mat_ret.mul(w, axis=1).sum(axis=1)
        else:
            df_ret_port["Retorno Portfolio"] = 0.0

        if benchmark_ident and benchmark_ident in df_ret.columns:
            df_ret_port["Retorno Benchmark"] = df_ret[benchmark_ident]
        else:
            if "Retorno Benchmark" in df_ret_port.columns:
                df_ret_port.drop(columns=["Retorno Benchmark"], inplace=True, errors="ignore")

        # LÂMINA PRINCIPAL
        with st.expander("Lâmina", expanded=True):
            df_ret_port_valid = df_ret_port.dropna(subset=["Retorno Portfolio"])
            datas_validas = df_ret_port_valid["Data"].dropna().unique()
            if len(datas_validas) > 0:
                min_data_port = pd.to_datetime(datas_validas).min()
                max_data_port = pd.to_datetime(datas_validas).max()
            else:
                min_data_port = df_ret["Data"].min()
                max_data_port = df_ret["Data"].max()

            st.subheader("Período para cálculo da cota")

            col_d1, col_d2 = st.columns(2)

            # limites globais do picker
            min_data_picker = df_ret["Data"].min().date()
            max_dado = max_data_port.date()
            hoje = date.today()
            max_data_picker = min(max_dado, hoje)

            key_ini = f"comp_data_ini_{portfolio_sel}"
            key_fim = f"comp_data_fim_{portfolio_sel}"

            # clamp em session_state se já existir
            if key_ini in st.session_state:
                if st.session_state[key_ini] < min_data_picker:
                    st.session_state[key_ini] = min_data_picker
                if st.session_state[key_ini] > max_data_picker:
                    st.session_state[key_ini] = max_data_picker

            if key_fim in st.session_state:
                if st.session_state[key_fim] < min_data_picker:
                    st.session_state[key_fim] = min_data_picker
                if st.session_state[key_fim] > max_data_picker:
                    st.session_state[key_fim] = max_data_picker

            default_ini = min_data_port.date()
            if default_ini < min_data_picker:
                default_ini = min_data_picker
            if default_ini > max_data_picker:
                default_ini = max_data_picker

            default_fim = max_data_picker

            data_ini = col_d1.date_input(
                "Data inicial",
                value=default_ini,
                min_value=min_data_picker,
                max_value=max_data_picker,
                key=key_ini,
            )

            data_fim = col_d2.date_input(
                "Data final",
                value=default_fim,
                min_value=min_data_picker,
                max_value=max_data_picker,
                key=key_fim,
            )

            if data_fim < data_ini:
                st.error("Data final não pode ser menor que a data inicial.")
                df_cota = pd.DataFrame()
            else:
                mask = (df_ret_port["Data"] >= pd.to_datetime(data_ini)) & (
                    df_ret_port["Data"] <= pd.to_datetime(data_fim)
                )
                df_cota = df_ret_port[mask].sort_values("Data").copy()

            # Características do portfolio
            caract = CARACTERISTICAS_PORT.get(
                portfolio_sel,
                {
                    "Perfil (Suitability)": "-",
                    "Objetivo da carteira": "-",
                    "Tipo de investidor": "-",
                    "Retorno 06 meses": "-",
                    "Retorno 12 meses": "-",
                    "Retorno 24 meses": "-",
                    "Meses acima do Benchmark": "-",
                    "Meses abaixo do Benchmark": "-",
                    "Maior rent. mensal": "-",
                    "Menor rent. mensal": "-",
                },
            ).copy()

            if not df_cota.empty:
                df_cota["Cota"] = (1 + df_cota["Retorno Portfolio"]).cumprod()
                df_cota["Retorno Acum"] = df_cota["Cota"] - 1
                df_cota["Retorno Acum %"] = df_cota["Retorno Acum"] * 100

                if "Retorno Benchmark" in df_cota.columns:
                    df_cota["Cota Benchmark"] = (1 + df_cota["Retorno Benchmark"]).cumprod()
                    df_cota["Retorno Acum Benchmark"] = df_cota["Cota Benchmark"] - 1
                    df_cota["Retorno Acum Benchmark %"] = df_cota["Retorno Acum Benchmark"] * 100

                ultima_data = df_cota["Data"].max()

                def retorno_janela(meses: int) -> str:
                    inicio = ultima_data - pd.DateOffset(months=meses)
                    janela = df_cota[df_cota["Data"] >= inicio].copy()
                    if janela.empty:
                        return "-"
                    cota_ini = janela["Cota"].iloc[0]
                    cota_fim = janela["Cota"].iloc[-1]
                    ret = cota_fim / cota_ini - 1
                    return f"{ret:.2%}"

                caract["Retorno 06 meses"] = retorno_janela(6)
                caract["Retorno 12 meses"] = retorno_janela(12)
                caract["Retorno 24 meses"] = retorno_janela(24)

                df_mensal_card = df_cota.copy()
                df_mensal_card["Ano"] = df_mensal_card["Data"].dt.year
                df_mensal_card["Mes"] = df_mensal_card["Data"].dt.month
                ret_mensal_card = df_mensal_card.groupby(["Ano", "Mes"])["Retorno Portfolio"].apply(
                    lambda x: (1 + x).prod() - 1
                )
                if len(ret_mensal_card) > 0:
                    caract["Maior rent. mensal"] = f"{ret_mensal_card.max():.2%}"
                    caract["Menor rent. mensal"] = f"{ret_mensal_card.min():.2%}"

                if "Retorno Benchmark" in df_cota.columns:
                    df_mensal_bench = df_cota.copy()
                    df_mensal_bench["Ano"] = df_mensal_bench["Data"].dt.year
                    df_mensal_bench["Mes"] = df_mensal_bench["Data"].dt.month

                    ret_mensal_bench = df_mensal_bench.groupby(["Ano", "Mes"])["Retorno Benchmark"].apply(
                        lambda x: (1 + x).prod() - 1
                    )

                    comp = pd.DataFrame({
                        "ret_carteira": ret_mensal_card,
                        "ret_bench": ret_mensal_bench,
                    }).dropna()

                    if not comp.empty:
                        meses_acima = (comp["ret_carteira"] > comp["ret_bench"]).sum()
                        meses_abaixo = (comp["ret_carteira"] < comp["ret_bench"]).sum()
                        caract["Meses acima do Benchmark"] = int(meses_acima)
                        caract["Meses abaixo do Benchmark"] = int(meses_abaixo)

            # MÉTRICAS
            col1, col2, col3 = st.columns(3)
            col1.metric("Perfil (Suitability)", caract.get("Perfil (Suitability)", "-"), border=True)
            col2.metric("Objetivo da carteira", caract.get("Objetivo da carteira", "-"), border=True)
            col3.metric("Tipo de investidor", caract.get("Tipo de investidor", "-"), border=True)

            col4, col5, col6 = st.columns(3)
            col4.metric("Retorno 06 meses", caract.get("Retorno 06 meses", "-"), border=True)
            col5.metric("Retorno 12 meses", caract.get("Retorno 12 meses", "-"), border=True)
            col6.metric("Retorno 24 meses", caract.get("Retorno 24 meses", "-"), border=True)

            col7, col8 = st.columns(2)
            col7.metric("Maior rent. mensal", caract.get("Maior rent. mensal", "-"), border=True)
            col8.metric("Menor rent. mensal", caract.get("Menor rent. mensal", "-"), border=True)

            col9, col10 = st.columns(2)
            col9.metric("Meses acima do Benchmark", caract.get("Meses acima do Benchmark", "-"), border=True)
            col10.metric("Meses abaixo do Benchmark", caract.get("Meses abaixo do Benchmark", "-"), border=True)

            col1c, col2c = st.columns([3, 3])

            with col1c:
                st.subheader("Composição Gráfica")
                df_sunburst, fig_sunburst = criar_sunburst(df_prop_class_atual, portfolio_sel)
                fig_sunburst.update_layout(
                    margin=dict(t=50, l=0, r=0, b=0),
                    height=500,
                    font_size=12,
                )
                fig_sunburst.update_traces(
                    textinfo="label+percent entry",
                    textfont_size=11,
                )
                st.plotly_chart(fig_sunburst, use_container_width=True)

            with col2c:
                st.subheader("Alocação por Categoria")
                df_alocacao = gerar_alocacao_por_categoria(df_prop_class_atual, portfolio_sel)
                df_aloc_display = df_alocacao.copy()
                df_aloc_display["Alocação %"] = df_aloc_display["Alocação %"].apply(lambda x: f"{x:.2%}")
                st.dataframe(df_aloc_display, use_container_width=True, hide_index=True)

                st.subheader("Distribuição de Risco por Classe")
                df_distribuicao = gerar_distribuicao_risco(df_prop_class_atual, portfolio_sel)
                
                fig_bar = px.bar(
                    df_distribuicao,
                    x="Distribuição %",
                    y="Classe de Ativo",
                    orientation="h",
                    text="Distribuição %",
                )
                fig_bar.update_traces(
                    texttemplate="%{x:.1%}",
                    textposition="outside",
                    marker_color="#102134"
                )
                fig_bar.update_xaxes(tickformat=".0%")
                fig_bar.update_layout(
                    xaxis_title="Peso (%)",
                    yaxis_title="Classe de Ativo",
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=300,
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            # GRÁFICO DE RETORNO ACUMULADO (%)
            if not df_cota.empty:
                fig_cota = px.line(
                    df_cota,
                    x="Data",
                    y="Retorno Acum %",
                    title="Evolução do retorno acumulado do Portfolio (Backtest)",
                )

                if "Retorno Acum Benchmark %" in df_cota.columns:
                    fig_cota.add_scatter(
                        x=df_cota["Data"],
                        y=df_cota["Retorno Acum Benchmark %"],
                        mode="lines",
                        name=f"Benchmark - {benchmark_nome_sel}",
                    )

                fig_cota.update_layout(
                    xaxis_title="Data",
                    yaxis_title="Retorno acumulado (%)",
                    height=350,
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig_cota, use_container_width=True)
            else:
                st.info("Não há dados de retorno no intervalo selecionado.")

            # TABELA DE RETORNO MENSAL
            st.markdown("---")
            st.subheader("Retornos Mensais")

            if not df_cota.empty:
                df_m = df_cota.copy()
                df_m["Ano"] = df_m["Data"].dt.year
                df_m["Mes"] = df_m["Data"].dt.month

                ret_m_carteira = (
                    df_m.groupby(["Ano", "Mes"])["Retorno Portfolio"]
                    .apply(lambda x: (1 + x).prod() - 1)
                    .reset_index()
                )

                anos = ret_m_carteira["Ano"].unique()
                meses = np.arange(1, 13)
                idx_completo = (
                    pd.MultiIndex.from_product([anos, meses], names=["Ano", "Mes"])
                    .to_frame(index=False)
                )
                ret_m_carteira = (
                    idx_completo
                    .merge(ret_m_carteira, on=["Ano", "Mes"], how="left")
                )

                tabela_cart = ret_m_carteira.pivot(index="Ano", columns="Mes", values="Retorno Portfolio")
                tabela_cart = tabela_cart.reindex(columns=meses)

                ret_ano = (1 + tabela_cart.fillna(0)).prod(axis=1) - 1
                tabela_cart["Ano Acumulado"] = ret_ano

                mapa_meses = {
                    1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr",
                    5: "Mai", 6: "Jun", 7: "Jul", 8: "Ago",
                    9: "Set", 10: "Out", 11: "Nov", 12: "Dez",
                }
                tabela_cart = tabela_cart.rename(columns=mapa_meses)

                tabela_cart_fmt = tabela_cart.applymap(
                    lambda x: "" if pd.isna(x) else f"{x*100:.2f}"
                )
                tabela_cart_fmt.insert(0, "Ativo", "Carteira")

                resultado = tabela_cart_fmt.reset_index()
                cols_ordem = [
                    "Ano", "Ativo",
                    "Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
                    "Jul", "Ago", "Set", "Out", "Nov", "Dez",
                    "Ano Acumulado",
                ]
                resultado = resultado[cols_ordem]

                st.dataframe(resultado, use_container_width=True, hide_index=True)
            else:
                st.info("Selecione um período válido para visualizar a tabela mensal.")

if __name__ == "__main__":
    main()
