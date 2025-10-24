import numpy as np
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import calendar

API_URL = "https://api.comdinheiro.com.br/v1/ep1/import-data"
HEADERS = {'Content-Type': 'application/x-www-form-urlencoded'}

# =======================
# MÓDULO DE CLASSIFICAÇÃO DE ATIVOS
GRANDES_CLASSES = {
    "Equities": ["Equities US", "Equities Global", "Equities EM"],
    "Fixed Income": ["Money Markets", "Investment Grade (3-10)", "High Yield", "Emerging Markets", "Investment Grade (1-3)"],
    "Alternatives": ["Crypto", "Gold"]
}

ATIVOS_PARA_CLASSE = {
    "US:SPY": "Equities US",
    "US:VT": "Equities Global",
    "US:USFR": "Money Markets",
    "US:SPIB": "Investment Grade (3-10)",
    "US:NYSEARCA:SPTL": "Investment Grade (3-10)",
    "US:NASDAQ:SHY": "Investment Grade (1-3)",
    "US:NYSEARCA:EEM": "Emerging Markets",
    "US:SPHY": "High Yield",
    "US:NASDAQ:EMB": "High Yield",
    "US:GDX": "Gold",
    "US:BITO": "Crypto"
}

@st.cache_data
def fetch_data(payload, tab_name="tab0"):
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
    for col in df.columns[1:]:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass
    numeric_cols = df.select_dtypes(include=['number']).columns
    df.loc[:, numeric_cols] = df.loc[:, numeric_cols] / 100
    return df

def main():
    st.set_page_config(layout="wide")
    st.title("Lâmina Comitê SWM")

    payload1 = (
        "username=solutionswm&password=Soluti%40ns2023&"
        "URL=ComparaFundos001.php%3F%26datas%3D31%2F12%2F9999%26cnpjs%3DEXPLODE%28GrowthandIncome_I%29%2BIncome_I%2BIncome_II%2BIncome_III%2BGrowthandIncome_I%2BGrowthandIncome_II%2BGrowthandIncome_III%2BGrowth_II%2BGrowth_III%26indicadores%3Dexplode%28Relatorio_Comite_Offshore%29%26num_casas%3D2%26pc%3Dnome_fundo%26flag_transpor%3D0%26enviar_email%3D0%26mostrar_da%3D0%26op01%3Dtabela%26oculta_cabecalho_sup%3D0%26relat_alias_automatico%3Dcmd_alias_01"
        "&format=json3"
    )

    payload2 = (
        "username=solutionswm&password=Soluti%40ns2023&"
        "URL=ComparaFundos001.php%3F%26datas%3D31%2F12%2F9999%26cnpjs%3DEXPLODE%28GrowthandIncome_I%29"
        "%26indicadores%3Dexplode%28composicao_portfolio_comite%29"
        "%26num_casas%3D2%26pc%3Dnome_fundo%26flag_transpor%3D0%26enviar_email%3D0"
        "%26mostrar_da%3D0%26op01%3Dtabela%26oculta_cabecalho_sup%3D0%26relat_alias_automatico%3Dcmd_alias_01"
        "&format=json3"
    )
    
    payload3 = (
        "username=solutionswm&password=Soluti%40ns2023&"
        "URL=HistoricoCotacao002.php%3F"
        "%26x%3DEXPLODE%28GrowthandIncome_I%29%2BIncome_I%2BIncome_II%2BIncome_III%"
        "2BGrowthandIncome_I%2BGrowthandIncome_II%2BGrowthandIncome_III%2BGrowth_II%"
        "2BGrowth_III%26data_ini%3D01012023%26data_fim%3D21102025%26pagina%3D1%26d%"
        "3DMOEDA_ORIGINAL%26g%3D0%26m%3D1%26info_desejada%3Dretorno%26retorno%3Ddiscreto%26tipo_data%3Ddu_br%26tipo_ajuste%3Dtodosajustes%26num_casas%3D2%26enviar_email%3D0%26ordem_legenda%3D1%26cabecalho_excel%3Dmodo1%26classes_ativos%3Dfklk448oj5v5r%26ordem_data%3D0%26rent_acum%3Dnada%26minY%3D%26maxY%3D%26deltaY%3D%26preco_nd_ant%3D0%26base_num_indice%3D100%26flag_num_indice%3D0%26eixo_x%3DData%26startX%3D0%26max_list_size%3D20%26line_width%3D2%26titulo_grafico%3D%26legenda_eixoy%3D%26tipo_grafico%3Dline%26script%3D%26tooltip%3Dunica"
        "&format=json3"
    )
    
    payload4 = (
        "username=solutionswm&password=Soluti%40ns2023&"
        "URL=HistoricoCotacao002.php%3F%26x%3DEXPLODE%28GrowthandIncome_I%29%2BIncome_I%2BIncome_II%2BIncome_III%2BGrowthandIncome_I%2BGrowthandIncome_II%2BGrowthandIncome_III%2BGrowth_II%2BGrowth_III%26data_ini%3D01012023%26"
        "data_fim%3D31129999%26pagina%3D1%26d%3DMOEDA_ORIGINAL%26g%3D1%26m%3D0%26info_desejada%3Dnumero_indice%26retorno%3Ddiscreto%26tipo_data%3Ddu_br%26tipo_ajuste%3Dtodosajustes%26num_casas%3D2%26enviar_email%3D0%26ordem_legenda%3D1%26cabecalho_excel"
        "%3Dmodo1%26classes_ativos%3Dfklk448oj5v5r%26ordem_data%3D0%26rent_acum%3Dnada%26minY%3D%26maxY%3D%26deltaY%3D%26preco_nd_ant%3D0%26base_num_indice%3D100%26flag_num_indice%3D0%26eixo_x%3DData%26startX%3D0%26max_list_size%3D20%26line_width%3D2%26titulo_grafico%3D%26legenda_eixoy%3D%26tipo_grafico%3Dline%26script%3D%26tooltip%3Dunica&format=json3"
    )
    
    tab_consultas, tab_laminas = st.tabs(["Consultas", "Lâminas"])

    with tab_consultas:
        df1 = fetch_data(payload1, tab_name="tab0")
        df2 = fetch_data(payload2, tab_name="tab0")
        df3 = fetch_data(payload3, tab_name="tab1")
        df3.iloc[:, 1:] = df3.iloc[:, 1:] * 10000
    
        df4 = fetch_data(payload4, tab_name="tab1")
        if df4.shape[1] > 1:
            df4.iloc[:, 1:] = df4.iloc[:, 1:].apply(pd.to_numeric, errors='coerce') * 100
    
        with st.expander("Consulta 1: Retorno Períodos", expanded=False):
            st.dataframe(df1)
    
        with st.expander("Consulta 2: Composição Portfólio", expanded=False):
            st.dataframe(df2)
            portfolios = df2.columns[1:]
            n_cols = 3
            for i in range(0, len(portfolios), n_cols):
                cols = st.columns(n_cols)
                for j, portfolio in enumerate(portfolios[i:i+n_cols]):
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
        
                data_inicio = st.date_input("Data início", value=data_min, min_value=data_min, max_value=data_max)
                data_fim = st.date_input("Data fim", value=data_max, min_value=data_min, max_value=data_max)
        
                if data_fim > data_inicio:
                    temp_df = df4[(df4['Data'] >= pd.to_datetime(data_inicio)) & (df4['Data'] <= pd.to_datetime(data_fim))].copy()
                    if temp_df.shape[1] > 1:
                        temp_df.iloc[:, 1:] = temp_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce') * 100
        
                    if not temp_df.empty:
                        base = temp_df.iloc[0, 1:]
                        rent_acum = temp_df.iloc[:, 1:].div(base).sub(1)
                        df_plot = pd.concat([temp_df['Data'], rent_acum * 100], axis=1)  # monta gráfico em %
                        fig = px.line(
                            df_plot,
                            x='Data',
                            y=df_plot.columns[1:],
                            markers=False,
                            title="Rentabilidade Acumulada Ajustada no Intervalo",
                            labels={'value': 'Retorno acumulado (%)', 'variable': 'Portfolio'}
                        )
                        fig.update_yaxes(ticksuffix="%")
                        st.plotly_chart(fig, use_container_width=True)
        
                        retorno_acum_formatado = rent_acum.iloc[[-1]].applymap(lambda x: f"{x*100:.2f}%")
                        st.write("Rentabilidade acumulada no período selecionado (final):")
                        st.dataframe(retorno_acum_formatado)
        
                    else:
                        st.write("Nenhum dado disponível para o período selecionado.")
                else:
                    st.write("A data de fim deve ser maior que a data de início.")
            else:
                st.write("Coluna 'Data' não encontrada para plotar série temporal.")

    # Aba Laminas
    with tab_laminas:
        df2 = fetch_data(payload2, tab_name="tab0")
        carteira_options = list(df2.columns[1:])
        carteira_selecionada = st.selectbox("Selecione o Portfolio", carteira_options)
        st.header(f"Portfolio: {carteira_selecionada}")

        df_sb = pd.DataFrame()
        df_sb["Raiz"] = [carteira_selecionada] * len(df2)
        df_sb["Ativo"] = df2["nome_fundo"] if "nome_fundo" in df2.columns else df2.index
        df_sb["Proporcao"] = df2[carteira_selecionada].astype(float)

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

        if "alocacoes_atualizadas" not in st.session_state:
            st.session_state.alocacoes_atualizadas = list(df_sb["Proporcao"])

        with st.expander("Asset Allocation %", expanded=False):
            if st.button("Resetar alocações para valores originais"):
                st.session_state.alocacoes_atualizadas = list(df_sb["Proporcao"])

            n_cols = 4
            cols = st.columns(n_cols)
            nova_alocacao = []

            for i, (idx, row) in enumerate(df_sb.iterrows()):
                col = cols[i % n_cols]
                key_name = f"aloc_{idx}_{st.session_state.get('reset_flag', False)}"
                valor_atual_pct = st.session_state.alocacoes_atualizadas[i] * 100
                novo_valor_pct = col.number_input(
                    f'{row["Ativo"]} (%)', min_value=0.0, max_value=100.0,
                    value=valor_atual_pct, step=0.5, key=key_name
                )
                nova_alocacao.append(novo_valor_pct / 100)

            st.session_state.alocacoes_atualizadas = nova_alocacao

            soma_pct = sum(nova_alocacao) * 100
            st.write(f"**Soma das alocações:** {soma_pct:.2f}%")

            df_sb["Proporcao"] = nova_alocacao
            soma = sum(nova_alocacao)
            if abs(soma - 1) > 0.001:
                st.warning("A soma das alocações não é 100%. O gráfico será ajustado proporcionalmente.")
                df_sb["Proporcao"] = df_sb["Proporcao"] / soma

            df_classe = df_sb.groupby("Classe", as_index=False).agg({"Proporcao": "sum"})
            df_classe = df_classe[df_classe["Proporcao"] > 0]

        with st.expander("Composição", expanded=False):
            col1, col2, col3 = st.columns([1,2,1])

            with col1:
                st.subheader("Alocação Classe")
                st.dataframe(
                    df_classe.reset_index(drop=True)
                    .style.format({"Proporcao": "{:.2%}"})
                    .set_table_styles([{'selector': 'td, th', 'props': [('text-align', 'center')]}]),
                    use_container_width=True
                )

            with col2:
                st.subheader("Asset Allocation")
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
                st.subheader("Retornos por Período")
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
                    ret_agg[periodo] = retorno_periodo * 100

                df_retorno_ponderado = pd.DataFrame.from_dict(ret_agg, orient="index", columns=[carteira_selecionada])

                st.dataframe(
                    df_retorno_ponderado.style.format("{:.2f}%")
                    .set_table_styles([{'selector': 'td, th', 'props': [('text-align', 'center')]}]),
                    use_container_width=True
                )

                ativos_b3 = [c for c in df3.columns if c != 'Data']
                prop_dict = dict(zip(df_sb["Ativo"], df_sb["Proporcao"]))
                proporcoes = [prop_dict.get(ativo, 0) for ativo in ativos_b3]

                retornos = df3[ativos_b3].astype(float).fillna(0).to_numpy() / 100.0
                proporcoes_np = np.array(proporcoes).reshape(1, -1)

                retornos_ptf = (retornos * proporcoes_np).sum(axis=1)

                st.subheader("Estatísticas de Retorno")
                mean = np.mean(retornos_ptf) * 100
                best = np.max(retornos_ptf) * 100
                worst = np.min(retornos_ptf) * 100
                std = np.std(retornos_ptf) * 100
                df_stats = pd.DataFrame({
                    "": ["Retorno Médio Mensal", "Maior Retorno Mensal", "Menor Retorno Mensal", "Desvio Padrão"],
                    "Valor": [f"{mean:.2f}%", f"{best:.2f}%", f"{worst:.2f}%", f"{std:.2f}%"]
                })

                st.dataframe(
                    df_stats.set_index("")
                    .style.set_table_styles([
                        {'selector': 'th.col0, td.col0', 'props': 'text-align: center;'},
                        {'selector': 'th.col1, td.col1', 'props': 'text-align: center;'}
                    ]),
                    use_container_width=True
                )

        with st.expander("Portfolio Backtest (Rentabilidade Acumulada)", expanded=False):
            df3['Data'] = pd.to_datetime(df3['Data'], dayfirst=True, errors='coerce')

            ativos = [c for c in df3.columns if c != 'Data']
            prop_dict = dict(zip(df_sb["Ativo"], df_sb["Proporcao"]))
            proporcoes = [prop_dict.get(ativo, 0) for ativo in ativos]

            retornos = df3[ativos].astype(float).fillna(0).to_numpy() / 100.0
            proporcoes_np = np.array(proporcoes).reshape(1, -1)

            retornos_ptf = (retornos * proporcoes_np).sum(axis=1)

            df_backtest = pd.DataFrame({
                "Data": df3['Data'],
                "Retorno (%)": retornos_ptf * 100
            })
            df_backtest["Ano"] = df_backtest["Data"].dt.year
            df_backtest["Mes"] = df_backtest["Data"].dt.month

            tabela_retorno = df_backtest.pivot(index="Ano", columns="Mes", values="Retorno (%)")
            tabela_retorno.columns = [calendar.month_abbr[m] for m in tabela_retorno.columns]
            MES_ORDER = [calendar.month_abbr[m] for m in range(1, 13)]
            tabela_retorno = tabela_retorno.reindex(columns=MES_ORDER)

            fy = df_backtest.groupby('Ano')["Retorno (%)"].apply(lambda x: ((1 + x/100).prod() - 1) * 100)
            tabela_retorno['FY'] = fy

            tabela_fmt = tabela_retorno.applymap(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")

            st.subheader("Portfolio Backtest")
            st.dataframe(tabela_fmt,use_container_width=True)

if __name__ == "__main__":
    main()
