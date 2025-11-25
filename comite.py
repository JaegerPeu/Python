import numpy as np
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import calendar
# from st_aggrid import AgGrid, GridOptionsBuilder

API_URL = "https://api.comdinheiro.com.br/v1/ep1/import-data"

GRANDES_CLASSES = {
    "Renda Fixa": ["Pós Fixado","Pré Fixado","Inflação"],
    "Fundos": ["Multimercado", "Crédito Privado", "Ações","Ações Internacionais","Renda Fixa Global"],
    "ETF": ["Ações","Ações Internacionais","Alternativos","Renda Fixa Global"],
    "Índice":["Alternativos","Ações"]
}

ATIVOS_PARA_CLASSE = {
    'DIVO11' : 'Ações',
    'IVVB11' : 'Ações Internacionais',
    'PEVC11' : 'Alternativos',
    'BNDX11' : 'Renda Fixa Global',
    'Solutions Optimum Cdi FICFI MM' : 'Multimercado',
    'BTG Pactual Crédito Corporativo I FIF em Cotas de FI RF' : 'Credito Privado',
    'Spx Seahawk FIF em Cotas de FI RF Access' : 'Credito Privado',
    'BTG Pactual Cred Corp Incentivado Ipca FIF em Cotas de FFI em Infra RF' : 'Credito Privado',
    'ANBIMA_IRFM1+' : 'Credito Privado',
    'V8 Speedway Long Short FICFIf MM' : 'Multimercado',
    'Genoa Capital Arpa CIC de CI MM - Resp Limitada' : 'Ações',
    'Absolute Pace Long Biased Advisory FICFIFA' : 'Ações',
    'Pimco Income Dólar FICFIf MM IE LP - Resp Limitada' : 'Renda Fixa Global',
    'Real Investor FICFIFA - Resp Limitada' : 'Ações',
    'BTGp Aqr Long-Biased Equities FIF em Cotas de FIA' : 'Ações Internacionais',
    'Schroder Gaia Contour Tech Equity Long & Short Brl Fifcic MM IE - Resp Limitada' : 'Ações Internacionais',
    'IFIX' : 'Alternativos',
    'IBOV' : 'Ações',
    'ANBIMA_IDADI' : 'Pós Fixado',
    'cdi+2%aa' : 'Pós Fixado',
    'ANBIMA_IDAIPCA' : 'Pós Fixado'
}

@st.cache_data
def fetch_data(payload, tab_name="tab0"):
    HEADERS = {'Content-Type': 'application/x-www-form-urlencoded'}
    response = requests.post(API_URL, data=payload, headers=HEADERS)
    response.raise_for_status()
    json_data = response.json()
    tab = json_data["tables"][tab_name]
    headers = [tab["lin0"][col] for col in sorted(tab["lin0"].keys())]
    rows = []
    for i in range(1, len(tab)):
        linha = tab.get(f"lin{i}", None)
        if linha:
            row = [linha[col].replace(",", ".") if isinstance(linha[col], str) else linha[col] for col in sorted(linha.keys())]
            rows.append(row)
    df = pd.DataFrame(rows, columns=headers)

    # Convert columns to numeric where possible
    for col in df.columns[1:]:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass

    # Division by 100 removed here to preserve original API percent integer scale
    return df

def main():
    st.set_page_config(layout="wide", page_title="Lâmina Comitê SWM")
    st.title("Lâmina Comitê SWM")

    payload1 = (
        "username=solutionswm&password=Soluti%40ns2023&URL=ComparaFundos001.php%3F%26datas%3D20%2F10%2F2025"
        "%26cnpjs%3DPortfolio_AvinI%2BPortfolio_AvinII%2BPortfolio_AvinIII%2BPortfolio_AvinIV%2BPortfolio_AvinV%2BPortfolio_AvinVI%2BEXPLODE%28Portfolio_AvinV%29"
        "%26indicadores%3Dexplode%28Relatorio_Comite_Offshore%29%26num_casas%3D2%26pc%3Dnome_fundo%26flag_transpor%3D0%26enviar_email%3D0"
        "%26mostrar_da%3D0%26op01%3Dtabela%26oculta_cabecalho_sup%3D0%26relat_alias_automatico%3Dcmd_alias_01&format=json3"
    )

    payload2 = (
        "username=solutionswm&password=Soluti%40ns2023&URL=ComparaFundos001.php%3F%26datas%3D20%2F10%2F2025"
        "%26cnpjs%3DEXPLODE%28Portfolio_AvinV%29%26indicadores%3Dexplode%28composicao_portfolio_comite_on%29"
        "%26num_casas%3D2%26pc%3Dnome_fundo%26flag_transpor%3D0%26enviar_email%3D0%26mostrar_da%3D0"
        "%26op01%3Dtabela%26oculta_cabecalho_sup%3D0%26relat_alias_automatico%3Dcmd_alias_01&format=json3"
    )

    payload3 = (
        "username=solutionswm&password=Soluti%40ns2023&URL=HistoricoCotacao002.php%3F%26x%3DPortfolio_AvinI%2BPortfolio_AvinII%2BPortfolio_AvinIII%2BPortfolio_AvinIV%2BPortfolio_AvinV%2BPortfolio_AvinVI%2BEXPLODE%28Portfolio_AvinV%29"
        "%26data_ini%3D25062025%26data_fim%3D31129999%26pagina%3D1%26d%3DMOEDA_ORIGINAL%26g%3D1%26m%3D1%26info_desejada%3Dretorno"
        "%26retorno%3Ddiscreto%26tipo_data%3Ddu_br%26tipo_ajuste%3Dtodosajustes%26num_casas%3D2%26enviar_email%3D0"
        "%26ordem_legenda%3D1%26cabecalho_excel%3Dmodo1%26classes_ativos%3Dfklk448oj5v5r%26ordem_data%3D0"
        "%26rent_acum%3Dnada%26minY%3D%26maxY%3D%26deltaY%3D%26preco_nd_ant%3D0%26base_num_indice%3D100"
        "%26flag_num_indice%3D0%26eixo_x%3DData%26startX%3D0%26max_list_size%3D20%26line_width%3D2"
        "%26titulo_grafico%3D%26legenda_eixoy%3D%26tipo_grafico%3Dline%26script%3D%26tooltip%3Dunica&format=json3"
    )

    payload4 = (
        "username=solutionswm&password=Soluti%40ns2023&URL=HistoricoCotacao002.php%3F%26x%3DPortfolio_AvinI%2BPortfolio_AvinII%2BPortfolio_AvinIII%2BPortfolio_AvinIV%2BPortfolio_AvinV%2BPortfolio_AvinVI%2BEXPLODE%28Portfolio_AvinV%29"
        "%26data_ini%3D25062025%26data_fim%3D31129999%26pagina%3D1%26d%3DMOEDA_ORIGINAL%26g%3D1%26m%3D0%26info_desejada%3Dretorno_acum"
        "%26retorno%3Ddiscreto%26tipo_data%3Ddu_br%26tipo_ajuste%3Dtodosajustes%26num_casas%3D2%26enviar_email%3D0"
        "%26ordem_legenda%3D1%26cabecalho_excel%3Dmodo1%26classes_ativos%3Dfklk448oj5v5r%26ordem_data%3D0"
        "%26rent_acum%3Dnada%26minY%3D%26maxY%3D%26deltaY%3D%26preco_nd_ant%3D0%26base_num_indice%3D100"
        "%26flag_num_indice%3D0%26eixo_x%3DData%26startX%3D0%26max_list_size%3D20%26line_width%3D2"
        "%26titulo_grafico%3D%26legenda_eixoy%3D%26tipo_grafico%3Dline%26script%3D%26tooltip%3Dunica&format=json3"
    )

    tab_consultas, tab_laminas = st.tabs(["Consultas", "Lâminas"])

    with tab_consultas:
        df1 = fetch_data(payload1, tab_name="tab0")
        df2 = fetch_data(payload2, tab_name="tab0")
        df3 = fetch_data(payload3, tab_name="tab1")
        df3.iloc[:, 1:] = df3.iloc[:, 1:] *100

        df4 = fetch_data(payload4, tab_name="tab1")

        with st.expander("Consulta 1: Retorno Períodos", expanded=False):
            st.dataframe(df1)

        with st.expander("Consulta 2: Composição Portfólio", expanded=False):
            st.dataframe(df2)

            portfolios = df2.columns[1:]
            n_cols = 3
            for i in range(0, len(portfolios), n_cols):
                cols = st.columns(n_cols)
                for j, portfolio in enumerate(portfolios[i:i + n_cols]):
                    pie_data = df2[[df2.columns[0], portfolio]].copy()
                    pie_data = pie_data[pie_data[portfolio] > 0]
                    fig = px.pie(pie_data, names=df2.columns[0], values=portfolio, title=portfolio)
                    with cols[j]:
                        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Consulta 3: Retorno Mensal", expanded=False):
            st.dataframe(df3)
            if 'Data' in df3.columns:
                df3['Data'] = pd.to_datetime(df3['Data'], dayfirst=True, errors='coerce')
                cols_para_plotar = [col for col in df3.columns if col != 'Data' and pd.api.types.is_numeric_dtype(df3[col])]
                fig = px.line(
                    df3,
                    x='Data',
                    y=cols_para_plotar,
                    markers=True,
                    title='Retorno Mensal',
                    labels={'value': 'Retorno (%)', 'variable': 'Portfolio'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("Coluna 'Data' não encontrada para plotar série temporal.")

        with st.expander("Consulta 4: Rentabilidade Acumulada", expanded=False):
            if 'Data' in df4.columns:
                df4['Data'] = pd.to_datetime(df4['Data'], dayfirst=True, errors='coerce')
                data_min = df4['Data'].min()
                data_max = df4['Data'].max()

                data_inicio = st.date_input("Data início", value=data_min, min_value=data_min, max_value=data_max, key="consulta4_data_inicio")
                data_fim = st.date_input("Data fim", value=data_max, min_value=data_min, max_value=data_max, key="consulta4_data_fim")

                if data_fim > data_inicio:
                    temp_df = df4[(df4['Data'] >= pd.to_datetime(data_inicio)) & (df4['Data'] <= pd.to_datetime(data_fim))].copy()

                    if not temp_df.empty:
                        df_plot = temp_df.copy()

                        fig = px.line(
                            df_plot,
                            x='Data',
                            y=df_plot.columns[1:],
                            markers=False,
                            title="Rentabilidade Acumulada no Período",
                            labels={'value': 'Retorno acumulado (%)', 'variable': 'Portfolio'}
                        )
                        fig.update_yaxes(ticksuffix="%")
                        st.plotly_chart(fig, use_container_width=True)

                        retorno_acum_formatado = df_plot.iloc[[-1]].set_index('Data').T
                        retorno_acum_formatado.columns = ['% Último Dia']
                        retorno_acum_formatado = retorno_acum_formatado.applymap(lambda x: f"{x:.2f}%")

                        st.write("Rentabilidade acumulada no período selecionado (final):")
                        st.dataframe(retorno_acum_formatado, use_container_width=False)
                    else:
                        st.write("Nenhum dado disponível para o período selecionado.")
                else:
                    st.write("A data de fim deve ser maior que a data de início.")
            else:
                st.write("Coluna 'Data' não encontrada para plotar série temporal.")

    with tab_laminas:
        carteira_options = list(df2.columns[1:])
        carteira_selecionada = st.selectbox("Selecione o Portfolio", carteira_options)
        st.header(f"Portfolio: {carteira_selecionada}")

        if ("carteira_selecionada_atual" not in st.session_state or
            st.session_state.carteira_selecionada_atual != carteira_selecionada):
            st.session_state.carteira_selecionada_atual = carteira_selecionada
            st.session_state.alocacoes_atualizadas = list(df2[carteira_selecionada].astype(float))

        df_sb = pd.DataFrame()
        df_sb["Raiz"] = [carteira_selecionada] * len(df2)
        df_sb["Ativo"] = df2["nome_fundo"] if "nome_fundo" in df2.columns else df2.index
        df_sb["Proporcao"] = st.session_state.alocacoes_atualizadas

        df_sb["Classe"] = df_sb["Ativo"].map(ATIVOS_PARA_CLASSE).fillna(df_sb["Ativo"])

        df_sb["Grande Classe"] = df_sb["Classe"].apply(
            lambda x: next((g for g, classes in GRANDES_CLASSES.items() if x in classes), "Outros")
        )
        df_sb = df_sb.dropna(subset=["Raiz", "Grande Classe", "Classe", "Proporcao"])

        COLOR_MAP = {
            carteira_selecionada: "#FFFFFF",
            "Equities": "#896F3D",
            "Fixed Income": "#102134",
            "Alternatives": "#C8BEAA",
            "Outros": "#C8BEAA",
            "Desconhecido": "#C8BEAA"
        }

        with st.expander("Asset Allocation %", expanded=False):
            col_a1, col_a2, col_a3 = st.columns([3, 1, 1])
            with col_a1:
                st.subheader("Alocação de Ativos")
                n_inputs_por_linha = 4
                n_total = len(df_sb)
                nova_alocacao = []

                for start_idx in range(0, n_total, n_inputs_por_linha):
                    cols = st.columns(n_inputs_por_linha)
                    for i, idx in enumerate(range(start_idx, min(start_idx + n_inputs_por_linha, n_total))):
                        row = df_sb.iloc[idx]
                        valor_atual_pct = st.session_state.alocacoes_atualizadas[idx] * 100
                        novo_valor = cols[i].number_input(
                            f'{row["Ativo"]} (%)',
                            min_value=0.0,
                            max_value=100.0,
                            value=valor_atual_pct,
                            step=0.5,
                            key=f"input_{idx}_{st.session_state.carteira_selecionada_atual}"
                        )
                        nova_alocacao.append(novo_valor / 100)

                st.session_state.alocacoes_atualizadas = nova_alocacao

                soma_pct = sum(nova_alocacao) * 100
                st.write(f"**Soma das alocações:** {soma_pct:.2f}%")

                df_sb["Proporcao"] = nova_alocacao
                soma = sum(nova_alocacao)
                if abs(soma - 1) > 0.001:
                    st.warning("A soma das alocações não é 100%. Valores serão proporcionados.")
                    df_sb["Proporcao"] = df_sb["Proporcao"] / soma

        with st.expander("Composição", expanded=False):
            ativos = df1[df1.columns[0]].tolist()
            aloc_dict = dict(zip(df_sb["Ativo"], df_sb["Proporcao"]))
            periodos = df1.columns[1:]

            ret_agg = {}
            for periodo in periodos:
                retorno_periodo = 0.0
                for i, ativo in enumerate(ativos):
                    aloc = aloc_dict.get(ativo, 0)
                    ret_ativo = pd.to_numeric(df1.at[i, periodo], errors='coerce')
                    if pd.notnull(ret_ativo):
                        retorno_periodo += aloc * ret_ativo
                ret_agg[periodo] = retorno_periodo

            col1, col2, col3, col4 = st.columns([1, 2.5, 0.5, 0.5])

            with col1:
                aloc_classe = (
                    df_sb.groupby("Classe")["Proporcao"].sum()
                    .loc[lambda x: x > 0]
                    .reset_index()
                    .set_index("Classe")
                )
                st.dataframe(
                    aloc_classe.style.format({"Proporcao": "{:.2%}"}).set_table_styles(
                        [{'selector': 'td, th', 'props': [('text-align', 'center')]}]
                    ),
                    use_container_width=True
                )
            with col2:
                fig = px.sunburst(
                    df_sb,
                    path=["Grande Classe", "Classe", "Ativo"],
                    values="Proporcao",
                    color="Grande Classe",
                    color_discrete_map=COLOR_MAP,
                    hover_data={"Proporcao": ":.2%"}
                )
                fig.update_traces(textinfo="label+percent entry")
                fig.update_layout(margin=dict(t=35, l=0, r=0, b=0))
                st.plotly_chart(fig, use_container_width=True)

            with col3:
                st.markdown("#### Ret & Vol")
                for periodo in periodos:
                    col3.metric(label=periodo, value=f"{ret_agg[periodo]:.2f}%")

            with col4:
                st.markdown("#### Stats")
                ativos_b3 = [c for c in df3.columns if c != 'Data']
                prop_dict = dict(zip(df_sb["Ativo"], df_sb["Proporcao"]))
                proporcoes = [prop_dict.get(ativo, 0) for ativo in ativos_b3]

                retornos = df3[ativos_b3].astype(float).fillna(0).to_numpy() / 100.00
                proporcoes_np = np.array(proporcoes).reshape(1, -1)

                retornos_ptf = (retornos * proporcoes_np).sum(axis=1)

                mean = np.mean(retornos_ptf) * 100
                best = np.max(retornos_ptf) * 100
                worst = np.min(retornos_ptf) * 100
                std = np.std(retornos_ptf) * 100

                stats = {
                    "Retorno Médio Mensal": mean,
                    "Maior Retorno Mensal": best,
                    "Menor Retorno Mensal": worst,
                    "Desvio Padrão": std
                }

                for stat_name, stat_val in stats.items():
                    col4.metric(label=stat_name, value=f"{stat_val:.2f}%")

if __name__ == "__main__":
    main()
